"""Frozen paired qualification for the format-4 browser policy. Never promotes an artifact."""

import argparse
import hashlib
import itertools
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from .artifacts import digest
from .assessment import (
    ASSESSMENT_VERSION,
    SEAT_SCHEDULE,
    assess_parallel,
    load_checkpoint,
)
from .bundle import runtime_sources, training_commit
from .controlled_train import atomic_json
from .evaluation_styles import MIXTURE_STYLES, POPULATION_SCHEDULE
from .holdout import reserve, validate_reservations
from .release_recurrent import BrowserActor
from .release_recurrent_bluff import diagnose
from .statistics import pooled


def load(path):
    return json.loads(Path(path).read_text())


def fingerprint(data):
    return hashlib.sha256(
        json.dumps(
            data, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def freeze(path, data):
    if path.exists():
        if load(path) != data:
            raise ValueError(f"Frozen configuration changed: {path}")
    else:
        atomic_json(path, data)


def copy_checked(source, target, expected):
    if digest(source) != expected:
        raise ValueError(f"Input changed: {source}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(source, temp)
        os.replace(temp, target)
    if digest(target) != expected:
        raise ValueError(f"Frozen copy changed: {target}")
    return expected


def specifications():
    jobs = {}

    def add(name, **changes):
        jobs[name] = {
            "kind": "matches",
            "precision": "int8",
            "players": 0,
            "opponent": "baseline",
            "partner": "learned",
            "ablation": "none",
            **changes,
        }

    add("quality-reference", precision="fp32")
    add("quality-quantized")
    for style in MIXTURE_STYLES:
        add("opponent-" + style, opponent=style)
    for i in range(2):
        add(f"historical-{i}", opponent_file=f"inputs/historical-{i}.pt")
        add(f"crossplay-{i}", players=4, partner_file=f"inputs/partner-{i}.pt")
    for style in ("literal", "cautious", "call-heavy", "silent", "ignore"):
        add("partner-" + style, players=4, partner=style)
    add("population-reference", opponent="population")
    add("population-no-adaptation", opponent="population", ablation="adaptation")
    for partner in ("learned", "literal"):
        add(
            "communication-" + partner,
            players=4,
            opponent="population",
            partner=partner,
        )
        add(
            "no-communication-" + partner,
            players=4,
            opponent="population",
            partner=partner,
            ablation="signals",
        )
    for style in ("baseline", "cautious"):
        add("bluff-" + style, kind="bluff", players=2, opponent=style)
    for n, seed in ((2, 901), (2, 902), (2, 903), (4, 904), (4, 905), (4, 906)):
        add(f"response-{n}p-{seed}", players=n, response_seed=seed)
    return jobs


def prepare(root, project, revision):
    root = Path(root)
    project = Path(project)
    run = project / "ml/runs/reference-envido-v1"
    candidate = run / "browser-candidate"
    m = load(candidate / "manifest.json")
    if (
        type(m.get("approved")) is not bool
        or m.get("format") != 4
        or m.get("architecture")
        != "candidate-768-96-64-48-32-32-1-signal6-lstm64-residual-team-play-v2"
        or m.get("rules") != "2"
        or m.get("training_kind") != "rmappo-v1"
        or m.get("quantization") != "symmetric-int8-per-output-f32-execution"
        or m.get("bytes") != (candidate / "policy.bin").stat().st_size
        or m.get("wasm_bytes") != (candidate / "policy.wasm").stat().st_size
    ):
        raise ValueError("Unsupported or inconsistent release manifest")
    cp = Path(load(run / "frozen-candidate.json")["checkpoint"])
    if digest(cp) != m["checkpoint_sha256"]:
        raise ValueError("Incorrect trained checkpoint")
    code = Path(__file__).resolve().parents[2]
    sources = {
        f.relative_to(code).as_posix(): digest(f)
        for f in sorted((code / "ml/retruco_ml").glob("*.py"))
    }
    legacy = project / "ml/runs/evaluation-validation-v2"
    historic = load(legacy / "inputs/manifest.json")
    files = {}

    def copy(source, name, expected):
        files[name] = copy_checked(source, root / name, expected)

    copy(cp, "inputs/candidate.pt", m["checkpoint_sha256"])
    for name in [
        "manifest.json",
        "policy.bin",
        "policy.wasm",
        "reference.f32",
        "training-source-manifest.json",
    ]:
        copy(candidate / name, "inputs/bundle/" + name, digest(candidate / name))
    for role in ("partners", "historical"):
        for i, label in enumerate(historic[role][:2]):
            name = f"inputs/{'partner' if role == 'partners' else role}-{i}.pt"
            copy(legacy / label, name, historic["files"][label])
    for name in (
        "bin/selfplay",
        "bin/libretruco_policy.dylib",
        "bin/retruco_policy.wasm",
    ):
        files[name] = digest(root / name)
    external = load(root / "training-plan.json")
    if (
        files["bin/retruco_policy.wasm"] != m["wasm_sha256"]
        or files["bin/selfplay"] != external["simulator"]
        or files["bin/libretruco_policy.dylib"] != external["native_runtime"]
        or runtime_sources(code / "ml/runtime") != m["runtime_source_sha256"]
    ):
        raise ValueError("Game engine or runtime mismatch")
    record = project / "ml/runs/rmappo-retraining-v1/source/manifest.json"
    if (
        digest(record) != m["training_snapshot_sha256"]
        or training_commit(record, project) != m["training_source_commit"]
    ):
        raise ValueError("Training source provenance mismatch")
    if (
        files["inputs/bundle/training-source-manifest.json"]
        != m["training_snapshot_sha256"]
    ):
        raise ValueError("Bundle contains the wrong training snapshot")
    for sub in ("reports", "receipts", "logs", "partial"):
        (root / sub).mkdir(exist_ok=True)
    config = {
        "protocol": "format4-release-qualification-v1",
        "source_commit": revision,
        "source_files": sources,
        "simulator_commit": load(root.parent / "paired-configuration.json")["backend"],
        "files": files,
        "model_sha256": m["sha256"],
        "wasm_sha256": m["wasm_sha256"],
        "checkpoint_sha256": m["checkpoint_sha256"],
        "choice_policy": m["choice_policy"],
        "team_play": m["team_play"],
        "pairs": 2000,
        "bluff_pairs": 1000,
        "training_plan_sha256": digest(root / "training-plan.json"),
        "human_primary_plan_sha256": digest(root.parent / "paired-configuration.json"),
        "seat_schedule": SEAT_SCHEDULE,
        "opponent_schedule": POPULATION_SCHEDULE,
        "opponent_styles": list(MIXTURE_STYLES),
        "specifications": specifications(),
        "behavior_thresholds": {
            "bluff": "positive policy probability and lower paired win-gain CI above zero on baseline weak-opening-v1",
            "adaptation": "lower paired win-gain CI above zero on the population",
            "signals": "lower paired win-gain CI above zero with the literal human-role partner",
            "quantization": "upper paired FP32-minus-INT8 loss CI at most .01",
            "response_opponents": "three distinct fresh learner seeds per mode, at least 3600 active training seconds per seed",
        },
        "deceptive_role": "scripted deceptive opponents are human-role seats, so the candidate and other CPU actors continue to obey truthful signal masks",
    }
    freeze(root / "full-configuration.json", config)
    ident = fingerprint(config)
    row = reserve(project / "ml/runs/release-seeds.json", ident, 2000, 1000)
    freeze(root / "full-reservation.json", row)
    return ident


def check_inputs(root, project):
    conf = load(root / "full-configuration.json")
    ident = fingerprint(conf)
    row = load(root / "full-reservation.json")
    ledger = load(project / "ml/runs/release-seeds.json")
    if ledger["format"] != 1:
        raise ValueError("Unsupported reservation ledger")
    validate_reservations(ledger["reservations"])
    if row["fingerprint"] != ident or ledger["reservations"].get(ident) != row:
        raise ValueError("Unverified holdout reservation")
    if conf["specifications"] != specifications():
        raise ValueError("Protocol plan mismatch")
    code = Path(__file__).resolve().parents[2]
    if any(digest(code / name) != h for name, h in conf["source_files"].items()):
        raise ValueError("Changed evaluation code")
    if any(digest(root / name) != h for name, h in conf["files"].items()):
        raise ValueError("Changed evaluation inputs")
    if digest(root / "training-plan.json") != conf["training_plan_sha256"]:
        raise ValueError("Changed opponent plan")
    return conf, ident, row


def attest(root, name, ident):
    result = root / "reports" / f"{name}.json"
    receipt = root / "receipts" / f"{name}.json"
    log = root / "logs" / f"{name}.log"
    proof = load(receipt)
    if (
        proof.get("fingerprint") != ident
        or proof.get("report_sha256") != digest(result)
        or proof.get("log_sha256") != digest(log)
        or proof.get("exit_code") != 0
        or not log.stat().st_size
    ):
        raise ValueError(f"Invalid report receipt: {name}")
    d = load(result)
    if d.get("evaluation_fingerprint") != ident:
        raise ValueError("Report identity mismatch")
    return d


def response_checkpoint(root, players, seed, conf):
    folder = root / "opponents" / f"{players}p-{seed}"
    final = load(folder / "latest.json")
    p = folder / final["file"]
    if digest(p) != final["sha256"] or final["training_seconds"] < 3600:
        raise ValueError("Incomplete or changed best-response checkpoint")
    d = torch.load(p, weights_only=True, map_location="cpu")
    if (
        d["initialization"] != {"kind": "random", "seed": seed}
        or d["initial_seed"] != seed
        or d["config"]["players"] != players
        or d["total_seconds"] < 3600
        or d["decisions"] <= 0
        or d["update"] <= 0
        or d["matches"] <= 0
        or d["process_cpu_seconds"] <= 0
        or d["opponent_checkpoint_sha256"] != conf["checkpoint_sha256"]
        or d["opponent_model_sha256"] != conf["model_sha256"]
        or d["wasm_sha256"] != conf["wasm_sha256"]
        or d["config"]["profile"] != conf["choice_policy"]
        or bytes(d["fixed_opponent_blob"].tolist())
        != (root / "inputs/bundle/policy.bin").read_bytes()
    ):
        raise ValueError("Best-response training target or provenance mismatch")
    torch.manual_seed(seed)
    from .model import Policy

    init = hashlib.sha256(
        b"".join(t.numpy().tobytes() for t in Policy().state_dict().values())
    ).hexdigest()
    if init != d["initial_model_sha256"]:
        raise ValueError("Opponent initialization mismatch")
    return p


def run_report(args):
    root, project, name, workers, environments = args
    torch.set_num_threads(1)
    conf, ident, row = check_inputs(root, project)
    spec = conf["specifications"][name]
    receipt = root / "receipts" / f"{name}.json"
    if receipt.exists():
        attest(root, name, ident)
        return name
    model = BrowserActor(
        root / "inputs/candidate.pt",
        root / "inputs/bundle",
        root / "bin/libretruco_policy.dylib",
        fp32=spec["precision"] == "fp32",
    )
    opponent_file = root / spec["opponent_file"] if "opponent_file" in spec else None
    if "response_seed" in spec:
        opponent_file = response_checkpoint(
            root, spec["players"], spec["response_seed"], conf
        )
    partner_file = root / spec["partner_file"] if "partner_file" in spec else None
    opp = load_checkpoint(opponent_file) if opponent_file else None
    part = load_checkpoint(partner_file) if partner_file else None
    simulator = root / "bin/selfplay"
    log = root / "logs" / f"{name}.log"
    if spec["kind"] == "bluff":
        report = diagnose(
            model, simulator, row["bluff_pairs"], row["bluff_seed"], spec["opponent"]
        )
        log.write_text(
            f"Checked {report['pairs']} matched hidden deals. Deployed opening offers: {report['deployed_offers']}.\n"
        )
    else:
        configs = []
        folder = root / "partial" / name
        folder.mkdir(exist_ok=True)
        for n, score, flor in itertools.product(
            [spec["players"]] if spec["players"] else [2, 4], [12, 24], [False, True]
        ):
            p = folder / f"{n}-{score}-{int(flor)}.json"
            if p.exists():
                part_result = load(p)
                if (
                    part_result["fingerprint"] != ident
                    or part_result["specification"] != spec
                    or fingerprint(part_result["row"]) != part_result["row_sha256"]
                ):
                    raise ValueError("Invalid partial configuration")
                result = part_result["row"]
            else:
                result = assess_parallel(
                    model,
                    row["pairs"],
                    n,
                    score,
                    flor,
                    row["seed"],
                    spec["opponent"],
                    opp,
                    spec["partner"],
                    part,
                    spec["ablation"],
                    workers,
                    environments,
                    str(simulator),
                )
                atomic_json(
                    p,
                    {
                        "fingerprint": ident,
                        "specification": spec,
                        "row_sha256": fingerprint(result),
                        "row": result,
                    },
                )
            configs.append(result)
            with log.open("a") as stream:
                stream.write(
                    f"{n} seats, target {score}, flor {flor}: {result['pairs']} paired clusters, {result['win_rate']:.8f}\n"
                )
        report = {
            "seed": row["seed"],
            "pairs": row["pairs"],
            "configurations": configs,
            "pooled": pooled(configs),
            "opponent": digest(opponent_file) if opponent_file else spec["opponent"],
            "partner": digest(partner_file) if partner_file else spec["partner"],
            "ablation": spec["ablation"],
            "opponent_schedule": POPULATION_SCHEDULE
            if spec["opponent"] == "population" and opp is None
            else "fixed-opponent-v1",
            "opponent_styles": list(MIXTURE_STYLES)
            if spec["opponent"] == "population" and opp is None
            else None,
            "assessment_version": ASSESSMENT_VERSION,
            "seat_schedule": SEAT_SCHEDULE,
        }
    report.update(
        evaluation_fingerprint=ident,
        evaluation_source_commit=conf["source_commit"],
        checkpoint_sha256=conf["checkpoint_sha256"],
        quantized_sha256=None if spec["precision"] == "fp32" else conf["model_sha256"],
        wasm_sha256=conf["wasm_sha256"],
        specification=spec,
        approved=False,
    )
    path = root / "reports" / f"{name}.json"
    atomic_json(path, report)
    atomic_json(
        receipt,
        {
            "fingerprint": ident,
            "report_sha256": digest(path),
            "log_sha256": digest(log),
            "exit_code": 0,
        },
    )
    attest(root, name, ident)
    return name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "base", "responses"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--environments", type=int, default=32)
    a = p.parse_args()
    root, project = a.root.resolve(), a.project.resolve()
    if a.command == "prepare":
        code = Path(__file__).resolve().parents[2]
        if subprocess.check_output(
            [
                "git",
                "-C",
                str(code),
                "status",
                "--porcelain",
                "--",
                "ml/retruco_ml",
                "ml/runtime",
            ]
        ):
            raise ValueError("Freeze the qualification source before reserving deals")
        rev = subprocess.check_output(
            ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
        ).strip()
        print(
            "Frozen full format-4 evaluation", prepare(root, project, rev), flush=True
        )
        return
    conf, ident, row = check_inputs(root, project)
    names = [
        n
        for n, s in conf["specifications"].items()
        if ("response_seed" in s) == (a.command == "responses")
    ]
    names.sort(
        key=lambda n: (
            0
            if n.startswith("bluff-")
            else 1
            if n
            in (
                "population-reference",
                "population-no-adaptation",
                "communication-literal",
                "no-communication-literal",
            )
            else 2,
            n,
        )
    )
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futures = [
            pool.submit(run_report, (root, project, n, 1, a.environments))
            for n in names
        ]
        errors = []
        for f in as_completed(futures):
            name = names[futures.index(f)]
            try:
                print("Complete", f.result(), flush=True)
            except Exception as error:
                errors.append((name, str(error)))
                print("Error", name, repr(error), flush=True)
    if errors:
        raise SystemExit(str(errors))


if __name__ == "__main__":
    main()
