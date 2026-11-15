"""Bounded native-exposure/credit discovery and fresh-seed target confirmation."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

from .controlled_train import atomic_json
from .cooperation_refusal_campaign import (
    preservation_snapshot,
    validate_inputs,
    verify_preservation,
)
from .cooperation_refusal_compare import clusters, read
from .historical_league import _fingerprint
from .improvement_budget import KEY
from .partner_pool import load_partner_pool
from .recurrent import file_hash
from .snapshot import freeze_source, verify_snapshot
from .statistics import interval
from .target_compare import (
    assessment,
    discovery_comparison,
    final_comparison,
    select_checkpoint,
    verified_report,
    verify_frozen_choices,
)
from .target_protocol import (
    ablation_modes,
    confirmation_budgets,
    estimate_confirmation_round,
    selection_rank,
    selection_specs,
    training_command,
    training_settings,
    validate_controls,
)

KIND = "target-improvement-v1"


class BudgetExhausted(RuntimeError):
    pass


class CampaignInterrupted(RuntimeError):
    pass


def interrupt_campaign(signum, frame):
    # Let the normal exception path reap every owned child process group.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise CampaignInterrupted("campaign interrupted by signal " + str(signum))


def verify_partners(root, config, metadata):
    heldout = root / "inputs" / metadata["parents"]["103"]["file"]
    weights = torch.load(heldout, map_location="cpu", weights_only=True)["model"]
    return load_partner_pool(
        root / "partners/manifest.json",
        forbidden_hashes=[
            metadata["parents"]["103"]["sha256"],
            *config["reserved_partner_sha256"],
        ],
        forbidden_fingerprints=[
            _fingerprint(weights),
            *config["reserved_partner_fingerprints"],
        ],
    )


def immutable_inventory(repository, root, version):
    result = preservation_snapshot(repository, root)
    if version == 1:
        return result
    runs = repository / "ml/runs"
    paths = set((repository / "ml/reports").glob("*.json"))
    paths.update(
        p
        for p in runs.rglob("*.json")
        if p.name not in ("status.json", "chunk.json") and "__pycache__" not in p.parts
    )
    paths.update(runs.rglob("*.jsonl"))
    for manifest in runs.rglob("source/manifest.json"):
        paths.update(
            p
            for p in manifest.parent.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
    for p in sorted(paths):
        if not p.resolve().is_relative_to(root.resolve()):
            result["files"][str(p.resolve())] = {
                "sha256": file_hash(p),
                "bytes": p.stat().st_size,
            }
    result["checkpoint_count"] = sum(name.endswith(".pt") for name in result["files"])
    result["scope"] = (
        "pre-existing checkpoints, frozen source inventories/files and JSON reports/configurations; status/chunk excluded"
    )
    return result


def validate_config(config):
    expected = {
        "L99": {"native_fraction": 0.0, "gae_lambda": 0.99},
        "N99": {"native_fraction": 0.5, "gae_lambda": 0.99},
        "L100": {"native_fraction": 0.0, "gae_lambda": 1.0},
        "N100": {"native_fraction": 0.5, "gae_lambda": 1.0},
    }
    version = config.get("version")
    if (
        version not in (1, 2, 3)
        or config.get("actor_memory") != 64
        or config.get("critic_hidden") != [256, 128]
        or config.get("headsup_fraction") != (0.0 if version == 3 else 0.4)
        or config.get("heldout_parent") != 103
        or config.get("mode_win_rate_targets")
        != {"1v1": 0.70, "2v2": 0.65 if version == 1 else 0.60}
    ):
        raise ValueError("unsupported native exposure x GAE target study")
    if version == 1 and config.get("recipes") != expected:
        raise ValueError("unsupported native exposure x GAE target study")
    if version >= 2:
        validate_controls(config)
        reserve = config.get("validation_reserve_hours")
        if (
            type(reserve) not in (int, float)
            or not 0 < reserve < config["max_wall_hours"]
        ):
            raise ValueError("final validation reserve must fit the campaign budget")
        for key in ("reserved_partner_sha256", "reserved_partner_fingerprints"):
            if not isinstance(config.get(key), list) or any(
                not isinstance(x, str)
                or len(x) != 64
                or any(c not in "0123456789abcdef" for c in x)
                for x in config[key]
            ):
                raise ValueError("invalid partner exclusions")
    for key in (
        "discovery_choices",
        "steps",
        "workers",
        "environments",
        "batch_size",
        "calibration_choices",
        "league_interval",
        "evaluation_workers",
    ):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError("invalid positive capacity: " + key)
    if type(config.get("warmup_max")) is not int or config["warmup_max"] < 2:
        raise ValueError("at least two critic warmups required")
    for key in (
        "max_wall_hours",
        "memory_gib",
        "parent_anchor",
        "response_anchor",
        "signal_anchor",
    ):
        if (
            not isinstance(config.get(key), (int, float))
            or not math.isfinite(config[key])
            or config[key] <= 0
        ):
            raise ValueError("invalid finite positive setting: " + key)
    for key in ("discovery_seeds", "confirmation_seeds"):
        seeds = config[key]
        if (
            len(set(seeds)) != len(seeds)
            or any(type(seed) is not int for seed in seeds)
            or not seeds
        ):
            raise ValueError("distinct integer training seeds required")
        if not config.get("smoke") and len(seeds) != 3:
            raise ValueError("three training seeds required per stage")
    if set(config["discovery_seeds"]) & set(config["confirmation_seeds"]):
        raise ValueError("fresh confirmation training seeds required")
    checkpoints = config["confirmation_checkpoints"]
    if (
        not checkpoints
        or any(type(x) is not int or x < 1 for x in checkpoints)
        or checkpoints != sorted(set(checkpoints))
    ):
        raise ValueError("strictly increasing checkpoint budgets required")
    primary = {"native-headsup"} | {
        "native-" + p
        for p in ("literal-v2", "trusting-v2", "cautious-v2", "reference-human")
    }
    if {x["id"] for x in config["primary_specs"]} != primary or len(
        config["primary_specs"]
    ) != 5:
        raise ValueError("complete native target coverage required")
    for spec in config["primary_specs"]:
        if (
            spec["opponent"] != "native"
            or spec["players"] != (2 if spec["id"] == "native-headsup" else 4)
            or (spec["players"] == 4 and spec["id"] != "native-" + spec["partner"])
        ):
            raise ValueError(
                "native target specs cannot substitute mixed-opponent evaluations"
            )
    specs = config["primary_specs"] + config["diagnostic_specs"]
    if len({x["id"] for x in specs}) != len(specs):
        raise ValueError("duplicate evaluation specs")
    ranges = []
    for family in ("discovery", "diagnostic", "selection", "final", "ablation"):
        seed, pairs = config[family + "_seed"], config[family + "_pairs"]
        if (
            type(seed) is not int
            or type(pairs) is not int
            or pairs < 2
            or not 100000 <= seed < seed + pairs < 200000
        ):
            raise ValueError("invalid development evaluation stream")
        span = set(range(seed, seed + pairs))
        if any(span & old for old in ranges):
            raise ValueError(
                "discovery/selection/final/ablation deals must be disjoint"
            )
        ranges.append(span)


def validate_endpoint(
    saved, config, recipe, seed, budget, starter_sha, inputs_sha, root=None
):
    settings = saved.get("configuration", {})
    expected = {
        "variant": recipe,
        "seed": seed,
        "choices": budget,
        "warm_start": "legacy-champion",
        "actor_memory": 64,
        "critic_hidden": [256, 128],
        "self_play_league": True,
        "league_interval": config["league_interval"],
        "heldout_parent": 103,
        "cooperation_practice": True,
        "headsup_fraction": config["headsup_fraction"],
        **(
            config["recipes"][recipe]
            if config["version"] == 1
            else training_settings(config, recipe, root)
        ),
        **{
            key: config[key]
            for key in ("parent_anchor", "response_anchor", "signal_anchor")
        },
    }
    if (
        saved.get("training_kind") != "improvement-mappo-v1"
        or saved.get("choices", 0) < budget
        or any(settings.get(key) != value for key, value in expected.items())
        or saved.get("contracts", {}).get("starter_sha256") != starter_sha
        or saved.get("contracts", {}).get("inputs_sha256") != inputs_sha
    ):
        raise ValueError("target trial endpoint provenance or intervention mismatch")
    if config["version"] >= 2:
        source = root / "source/ml/retruco_ml"
        if (
            saved["contracts"].get("source")
            != {p.name: file_hash(p) for p in source.glob("*.py")}
            or saved["contracts"].get("simulator_sha256")
            != file_hash(root / "source/bin/selfplay")
            or saved["contracts"].get("rng_protocol") != "separate-optimizer-v1"
            or "optimizer_random" not in saved
        ):
            raise ValueError("endpoint source or RNG protocol mismatch")
        if config["version"] >= 3:
            from .recurrent import TEAM_ADAPTER_VERSION

            if settings.get("team_adapter_version") != TEAM_ADAPTER_VERSION:
                raise ValueError("endpoint team adapter version differs")
            base = torch.load(
                root / "inputs/legacy-champion-lstm16.pt",
                map_location="cpu",
                weights_only=True,
            )["model"]
            for name, value in base.items():
                if name.startswith(
                    ("state.", "action.", "score.", "signal.")
                ) and not torch.equal(saved["model"][name], value):
                    raise ValueError(
                        "team specialist modified protected base parameter: " + name
                    )
        if config["recipes"][recipe]["partner_pool"]:
            metadata = validate_inputs(root / "inputs")
            pool = verify_partners(root, config, metadata)
            recipe_contract = json.loads(saved["contracts"]["recipe"])
            if recipe_contract.get("version") == "native-opponent-mixture-v1":
                recipe_contract = recipe_contract["remainder"]
            if recipe_contract.get("partner_pool") != pool.configuration:
                raise ValueError("endpoint partner pool provenance mismatch")


def freeze_json(path, value):
    receipt = path.with_suffix(".sha256")
    if path.exists():
        if (
            not receipt.is_file()
            or receipt.read_text().strip() != file_hash(path)
            or read(path) != value
        ):
            raise ValueError("frozen choice or assessment changed: " + str(path))
    else:
        atomic_json(path, value)
        receipt.write_text(file_hash(path) + "\n")


def check_deadline(started, previous, limit, now=None):
    if previous + (time.monotonic() if now is None else now) - started >= limit:
        raise BudgetExhausted("bounded twelve-hour target campaign budget reached")


def coverage(root, config):
    expected = (1 + len(config["recipes"]) * len(config["discovery_seeds"])) * (
        len(config["primary_specs"]) + len(config["diagnostic_specs"])
    )
    expected += (
        1 + len(config["confirmation_seeds"]) * len(config["confirmation_checkpoints"])
    ) * len(selection_specs(config))
    expected += (1 + len(config["confirmation_seeds"])) * (
        len(config["primary_specs"]) + len(config["diagnostic_specs"])
    )
    expected += 2 * 2 * len(ablation_modes(config))
    if config["version"] >= 2:
        return exact_coverage(root, config)
    completed = []
    for path in root.rglob("*.json"):
        receipt = path.with_suffix(".sha256")
        if not receipt.is_file():
            continue
        report = read(path)
        if "configurations" in report and receipt.read_text().strip() == file_hash(
            path
        ):
            completed.append(str(path.relative_to(root)))
    return {
        "completed_evaluation_reports": len(completed),
        "expected_evaluation_reports": expected,
        "complete": len(completed) == expected,
        "completed_reports": completed,
    }


def exact_coverage(root, config):
    """Enumerate required paths; unrelated or duplicate reports never count."""
    expected = []
    starter = root / "inputs/legacy-champion-lstm16.pt"
    for folder, checkpoint in [
        (root / "discovery-baseline", starter),
        *[
            (
                root / "discovery" / f"{recipe}-seed-{seed}",
                root / "discovery" / f"{recipe}-seed-{seed}" / "candidate.pt",
            )
            for recipe in config["recipes"]
            for seed in config["discovery_seeds"]
        ],
    ]:
        for family in ("primary", "diagnostic"):
            stream = "discovery" if family == "primary" else "diagnostic"
            expected += [
                (
                    folder / family / (s["id"] + ".json"),
                    s,
                    config[stream + "_pairs"],
                    config[stream + "_seed"],
                    checkpoint,
                    "normal",
                )
                for s in config[family + "_specs"]
            ]
    for folder, checkpoint in [
        (root / "selection-baseline", starter),
        *[
            (
                root
                / "confirmation"
                / f"seed-{seed}"
                / f"validation-budget-{budget:09d}",
                root
                / "confirmation"
                / f"seed-{seed}"
                / f"candidate-budget-{budget:09d}.pt",
            )
            for seed in config["confirmation_seeds"]
            for budget in confirmation_budgets(root, config)
        ],
    ]:
        expected += [
            (
                folder / (s["id"] + ".json"),
                s,
                config["selection_pairs"],
                config["selection_seed"],
                checkpoint,
                "normal",
            )
            for s in selection_specs(config)
        ]
    for label in [
        "starter",
        *[f"seed-{seed}" for seed in config["confirmation_seeds"]],
    ]:
        choice = root / f"selection-{label.split('-')[-1]}.json"
        checkpoint = (
            starter
            if label == "starter"
            else Path(read(choice)["checkpoint"])
            if choice.exists()
            else None
        )
        expected += [
            (
                root / "final" / label / (s["id"] + ".json"),
                s,
                config["final_pairs"],
                config["final_seed"],
                checkpoint,
                "normal",
            )
            for s in config["primary_specs"] + config["diagnostic_specs"]
        ]
    nominee = root / "nominee.json"
    for label in ("starter", "nominee"):
        checkpoint = (
            starter
            if label == "starter"
            else Path(read(nominee)["checkpoint"])
            if nominee.exists()
            else None
        )
        for partner in ("literal-v2", "trusting-v2"):
            spec = {"opponent": "native", "partner": partner, "players": 4}
            expected += [
                (
                    root / "ablations" / label / f"{partner}-{mode}.json",
                    spec,
                    config["ablation_pairs"],
                    config["ablation_seed"],
                    checkpoint,
                    mode,
                )
                for mode in ablation_modes(config)
            ]
    completed = []
    invalid = []
    inputs = root / "inputs/manifest.json"
    inputs_sha = file_hash(inputs) if inputs.exists() else None
    for path, spec, pairs, seed, checkpoint, mode in expected:
        if not path.exists():
            continue
        try:
            if checkpoint is None or not checkpoint.resolve().is_relative_to(
                root.resolve()
            ):
                raise ValueError("unknown or outside checkpoint")
            verified_report(path, spec, pairs, seed, checkpoint, inputs_sha, mode)
            completed.append(str(path.relative_to(root)))
        except (ValueError, OSError, KeyError, TypeError) as error:
            invalid.append({"path": str(path.relative_to(root)), "error": str(error)})
    return {
        "completed_evaluation_reports": len(completed),
        "expected_evaluation_reports": len(expected),
        "complete": len(completed) == len(expected) and not invalid,
        "completed_reports": completed,
        "invalid_reports": invalid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.resume:
        if read(root / "campaign-contract.json").get("kind") not in (
            KIND,
            "target-improvement-v2",
            "target-improvement-v3",
        ):
            raise ValueError("wrong target campaign kind")
    elif root.exists():
        raise ValueError("new target output must not exist")
    root.mkdir(parents=True, exist_ok=args.resume)
    with (root / "campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("target campaign already live")
        old = read(root / "status.json") if (root / "status.json").exists() else {}
        previous = old.get("campaign_seconds", 0.0)
        started = time.monotonic()
        stage = {"stage": "prepare"}

        def status(state, **extra):
            atomic_json(
                root / "status.json",
                {
                    "status": state,
                    "pid": os.getpid(),
                    "campaign_seconds": previous + time.monotonic() - started,
                    "supervisor_source": str(Path(__file__).resolve()),
                    **stage,
                    **extra,
                },
            )

        try:
            signal.signal(signal.SIGTERM, interrupt_campaign)
            signal.signal(signal.SIGINT, interrupt_campaign)
            if not args.resume:
                if not args.config or not args.inputs:
                    raise ValueError("config and frozen input manifest required")
                config = read(args.config)
                config["smoke"] = args.smoke
                if args.smoke:
                    config.update(
                        discovery_seeds=[501],
                        confirmation_seeds=[601],
                        discovery_choices=32,
                        confirmation_checkpoints=[32, 64],
                        steps=32,
                        workers=1,
                        environments=1,
                        batch_size=8,
                        calibration_choices=16,
                        warmup_max=2,
                        discovery_pairs=2,
                        diagnostic_pairs=2,
                        selection_pairs=2,
                        final_pairs=2,
                        ablation_pairs=2,
                    )
                    # Smoke results must never consume future production selection
                    # or final-confirmation deal streams.
                    if config["version"] >= 2:
                        config.update(
                            discovery_seed=100010,
                            diagnostic_seed=100020,
                            selection_seed=100030,
                            final_seed=100040,
                            ablation_seed=100050,
                        )
                validate_config(config)
                repository = Path(__file__).resolve().parents[2]
                atomic_json(
                    root / "preservation.json",
                    immutable_inventory(repository, root, config["version"]),
                )
                validate_inputs(args.inputs.resolve().parent)
                shutil.copytree(args.inputs.resolve().parent, root / "inputs")
                if config["version"] >= 2:
                    partner_manifest = (
                        args.config.resolve().parent / config["partner_pool_source"]
                    ).resolve()
                    load_partner_pool(partner_manifest)
                    shutil.copytree(partner_manifest.parent, root / "partners")
                    verify_partners(root, config, validate_inputs(root / "inputs"))
                snapshot = freeze_source(
                    root / "source", repository, allow_dirty=args.smoke
                )
                atomic_json(root / "configuration.json", config)
                atomic_json(
                    root / "campaign-contract.json",
                    {
                        "kind": f"target-improvement-v{config['version']}",
                        "configuration_sha256": file_hash(root / "configuration.json"),
                        "inputs_sha256": file_hash(root / "inputs/manifest.json"),
                        "source_manifest_sha256": file_hash(
                            root / "source/manifest.json"
                        ),
                        "preservation_sha256": file_hash(root / "preservation.json"),
                        **(
                            {
                                "partner_manifest_sha256": file_hash(
                                    root / "partners/manifest.json"
                                )
                            }
                            if config["version"] >= 2
                            else {}
                        ),
                    },
                )
            config = read(root / "configuration.json")
            validate_config(config)
            contract = read(root / "campaign-contract.json")
            for name, key in [
                ("configuration.json", "configuration_sha256"),
                ("inputs/manifest.json", "inputs_sha256"),
                ("source/manifest.json", "source_manifest_sha256"),
                ("preservation.json", "preservation_sha256"),
            ]:
                if file_hash(root / name) != contract[key]:
                    raise ValueError("immutable target contract changed: " + name)
            snapshot = verify_snapshot(root / "source")
            metadata = validate_inputs(root / "inputs")
            if config["version"] >= 2:
                if (
                    file_hash(root / "partners/manifest.json")
                    != contract["partner_manifest_sha256"]
                ):
                    raise ValueError("frozen partner manifest changed")
                verify_partners(root, config, metadata)
            if args.prepare:
                status("prepared", source_commit=snapshot["source_commit"])
                return
            env = dict(os.environ, PYTHONPATH=str(root / "source/ml"))
            env[KEY] = str(started + config["max_wall_hours"] * 3600 - previous)
            frozen = root / "source/ml/retruco_ml/target_campaign.py"
            if Path(__file__).resolve() != frozen.resolve():
                lock.close()
                os.chdir(root / "source/ml")
                os.execve(
                    sys.executable,
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "retruco_ml.target_campaign",
                        "--output",
                        str(root),
                        "--resume",
                    ],
                    env,
                )

            def deadline():
                check_deadline(started, previous, config["max_wall_hours"] * 3600)

            def run_many(items):
                for start in range(0, len(items), config["evaluation_workers"]):
                    deadline()
                    children = []
                    launch_times = {}
                    try:
                        for command, log in items[
                            start : start + config["evaluation_workers"]
                        ]:
                            log.parent.mkdir(parents=True, exist_ok=True)
                            with log.open("a") as stream:
                                child = subprocess.Popen(
                                    [sys.executable, "-u", "-m", *command],
                                    cwd=root / "source/ml",
                                    env=env,
                                    stdout=stream,
                                    stderr=subprocess.STDOUT,
                                    start_new_session=True,
                                )
                            children.append((child, command, log))
                            launch_times[child.pid] = time.monotonic()
                        while True:
                            alive = [
                                item for item in children if item[0].poll() is None
                            ]
                            if not alive:
                                break
                            deadline()
                            status(
                                "running",
                                child_pids=[x[0].pid for x in alive],
                                work_items=[
                                    {"pid": c.pid, "command": command, "log": str(log)}
                                    for c, command, log in alive
                                ],
                            )
                            remaining = max(
                                0.001,
                                config["max_wall_hours"] * 3600
                                - previous
                                - (time.monotonic() - started),
                            )
                            try:
                                alive[0][0].wait(timeout=min(30, remaining))
                            except subprocess.TimeoutExpired:
                                pass
                        for child, command, log in children:
                            if child.returncode == 75:
                                raise BudgetExhausted(
                                    "target child stopped at cooperative deadline"
                                )
                            if child.returncode:
                                raise RuntimeError(
                                    f"child {child.pid} exited {child.returncode}; inspect {log}"
                                )
                            with (root / "work-completions.jsonl").open("a") as stream:
                                stream.write(
                                    json.dumps(
                                        {
                                            "command": command,
                                            "seconds": time.monotonic()
                                            - launch_times[child.pid],
                                        }
                                    )
                                    + "\n"
                                )
                    except BaseException:
                        for child, _, _ in children:
                            if child.poll() is None:
                                try:
                                    os.killpg(child.pid, signal.SIGTERM)
                                except ProcessLookupError:
                                    pass
                        for child, _, _ in children:
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                try:
                                    os.killpg(child.pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                                child.wait()
                        raise

            simulator = root / "source/bin/selfplay"
            starter = (
                root / "inputs" / metadata["candidates"]["legacy-champion"]["file"]
            )

            def evaluate(checkpoint, folder, specs, pairs, seed, signals="normal"):
                folder.mkdir(parents=True, exist_ok=True)
                pending = []
                for spec in specs:
                    path = folder / f"{spec['id']}.json"
                    receipt = path.with_suffix(".sha256")
                    if path.exists():
                        if not receipt.exists():
                            archive = root / "interrupted-evaluations"
                            archive.mkdir(exist_ok=True)
                            shutil.move(
                                str(path),
                                str(
                                    archive
                                    / (
                                        hashlib.sha256(str(path).encode()).hexdigest()
                                        + f"-{time.time_ns()}.json"
                                    )
                                ),
                            )
                        else:
                            verified_report(
                                path,
                                spec,
                                pairs,
                                seed,
                                checkpoint,
                                contract["inputs_sha256"],
                                signals,
                            )
                            continue
                    command = [
                        "retruco_ml.improvement_evaluate",
                        str(checkpoint),
                        "--inputs",
                        str(root / "inputs/manifest.json"),
                        "--simulator",
                        str(simulator),
                        "--output",
                        str(path),
                        "--opponent",
                        spec["opponent"],
                        "--partner",
                        spec["partner"],
                        "--players",
                        str(spec["players"]),
                        "--pairs",
                        str(pairs),
                        "--seed",
                        str(seed),
                        "--signals",
                        signals,
                    ]
                    log = (
                        root
                        / "evaluation-logs"
                        / (hashlib.sha256(str(path).encode()).hexdigest()[:20] + ".log")
                    )
                    pending.append((command, log))
                run_many(pending)
                for spec in specs:
                    verified_report(
                        folder / f"{spec['id']}.json",
                        spec,
                        pairs,
                        seed,
                        checkpoint,
                        contract["inputs_sha256"],
                        signals,
                    )

            def train(folder, recipe, seed, budget):
                candidate = folder / f"candidate-budget-{budget:09d}.pt"
                receipt = candidate.with_suffix(".sha256")
                if candidate.exists():
                    if not receipt.exists() or receipt.read_text().strip() != file_hash(
                        candidate
                    ):
                        raise ValueError("immutable target endpoint changed")
                    validate_endpoint(
                        torch.load(candidate, map_location="cpu", weights_only=True),
                        config,
                        recipe,
                        seed,
                        budget,
                        file_hash(starter),
                        contract["inputs_sha256"],
                        root,
                    )
                    return candidate
                checkpoint = folder / "checkpoint.pt"
                current = (
                    torch.load(checkpoint, map_location="cpu", weights_only=True)
                    if checkpoint.exists()
                    else {}
                )
                if (
                    current.get("choices", 0) < budget
                    or current.get("configuration", {}).get("choices", 0) < budget
                ):
                    command = training_command(
                        config, recipe, root, folder, seed, budget, simulator
                    )
                    if checkpoint.exists():
                        command += ["--resume"]
                    run_many([(command, folder.parent / (folder.name + ".log"))])
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                validate_endpoint(
                    saved,
                    config,
                    recipe,
                    seed,
                    budget,
                    file_hash(starter),
                    contract["inputs_sha256"],
                    root,
                )
                shutil.copy2(checkpoint, candidate)
                receipt.write_text(file_hash(candidate) + "\n")
                return candidate

            for family in ("primary", "diagnostic"):
                stage.update(stage="discovery-baseline", family=family)
                evaluate(
                    starter,
                    root / "discovery-baseline" / family,
                    config[family + "_specs"],
                    config[
                        "discovery_pairs" if family == "primary" else "diagnostic_pairs"
                    ],
                    config[
                        "discovery_seed" if family == "primary" else "diagnostic_seed"
                    ],
                )
            for offset, seed in enumerate(config["discovery_seeds"]):
                names = list(config["recipes"])
                order = names[offset:] + names[:offset]
                for recipe in order:
                    stage.update(
                        stage="discovery-training",
                        recipe=recipe,
                        seed=seed,
                        target_choices=config["discovery_choices"],
                    )
                    folder = root / "discovery" / f"{recipe}-seed-{seed}"
                    checkpoint = train(
                        folder, recipe, seed, config["discovery_choices"]
                    )
                    compact = folder / "candidate.pt"
                    receipt = compact.with_suffix(".sha256")
                    if not compact.exists():
                        shutil.copy2(checkpoint, compact)
                        receipt.write_text(file_hash(compact) + "\n")
                    elif (
                        not receipt.exists()
                        or receipt.read_text().strip() != file_hash(compact)
                        or file_hash(compact) != file_hash(checkpoint)
                    ):
                        raise ValueError("discovery endpoint alias changed")
                    for family in ("primary", "diagnostic"):
                        stage.update(stage="discovery-validation", family=family)
                        evaluate(
                            checkpoint,
                            folder / family,
                            config[family + "_specs"],
                            config[
                                "discovery_pairs"
                                if family == "primary"
                                else "diagnostic_pairs"
                            ],
                            config[
                                "discovery_seed"
                                if family == "primary"
                                else "diagnostic_seed"
                            ],
                        )
            recipe_choice = discovery_comparison(root, config)
            freeze_json(root / "recipe-choice.json", recipe_choice)
            if (
                config["version"] >= 2
                and not config.get("smoke")
                and recipe_choice["exploratory_unguarded_continuation"]
            ):
                result = {
                    "status": "stopped-no-safe-recipe",
                    "discovery": recipe_choice,
                    "retained_checkpoint": str(starter),
                    "approved": False,
                    "coverage": coverage(root, config),
                    "preservation": verify_preservation(
                        read(root / "preservation.json")
                    ),
                    "scope": "No discovery recipe passed regression guards. Fresh confirmation was not run; target attainment is unconfirmed.",
                }
                atomic_json(root / "partial-results.json", result)
                status(
                    "stopped-no-safe-recipe",
                    result=str(root / "partial-results.json"),
                    child_pids=[],
                    work_items=[],
                )
                return
            recipe = recipe_choice["selected_recipe"]
            stage.update(stage="selection-baseline", recipe=recipe, seed=None)
            selected_specs = selection_specs(config)
            evaluate(
                starter,
                root / "selection-baseline",
                selected_specs,
                config["selection_pairs"],
                config["selection_seed"],
            )
            initial, _ = assessment(
                root / "selection-baseline",
                root / "selection-baseline",
                selected_specs,
                config["selection_pairs"],
                config["selection_seed"],
                starter,
                starter,
                contract["inputs_sha256"],
                config,
            )
            selections = {
                str(seed): {
                    "checkpoint": str(starter),
                    "sha256": file_hash(starter),
                    "assessment": initial,
                    "seed": seed,
                    "origin": "starter",
                    "trained": False,
                    "selection_budget": 0,
                }
                for seed in config["confirmation_seeds"]
            }
            # Advance each fresh seed equally before extending any seed further.
            completed_budgets = []
            previous_budget = 0
            frozen_plan = (root / "confirmation-plan.json").exists()
            for budget in confirmation_budgets(root, config):
                if (
                    config["version"] >= 2
                    and not frozen_plan
                    and not config.get("smoke")
                ):
                    elapsed = previous + time.monotonic() - started
                    cached = all(
                        (
                            root
                            / "confirmation"
                            / f"seed-{seed}"
                            / f"assessment-{budget}.json"
                        ).exists()
                        for seed in config["confirmation_seeds"]
                    )
                    predicted = estimate_confirmation_round(
                        root, config, recipe, budget, previous_budget
                    )
                    if (
                        not cached
                        and elapsed + predicted
                        >= (
                            config["max_wall_hours"]
                            - config["validation_reserve_hours"]
                        )
                        * 3600
                    ):
                        if completed_budgets:
                            break
                        raise BudgetExhausted(
                            "insufficient time for fresh-seed confirmation with final-evaluation reserve"
                        )
                for seed in config["confirmation_seeds"]:
                    folder = root / "confirmation" / f"seed-{seed}"
                    stage.update(
                        stage="confirmation-training", seed=seed, target_choices=budget
                    )
                    checkpoint = train(folder, recipe, seed, budget)
                    evaluation = folder / f"validation-budget-{budget:09d}"
                    stage["stage"] = "confirmation-selection"
                    evaluate(
                        checkpoint,
                        evaluation,
                        selected_specs,
                        config["selection_pairs"],
                        config["selection_seed"],
                    )
                    current, _ = assessment(
                        evaluation,
                        root / "selection-baseline",
                        selected_specs,
                        config["selection_pairs"],
                        config["selection_seed"],
                        checkpoint,
                        starter,
                        contract["inputs_sha256"],
                        config,
                    )
                    freeze_json(folder / f"assessment-{budget}.json", current)
                    selections[str(seed)] = select_checkpoint(
                        selections[str(seed)],
                        current,
                        checkpoint,
                        seed,
                        budget,
                        starter,
                        config,
                    )
                completed_budgets.append(budget)
                previous_budget = budget
            if config["version"] >= 2:
                freeze_json(
                    root / "confirmation-plan.json",
                    {
                        "completed_checkpoints": completed_budgets,
                        "planned_checkpoints": config["confirmation_checkpoints"],
                        "scope": "same completed budget rounds for every fresh seed; reserve final evaluation time",
                    },
                )
            for seed in config["confirmation_seeds"]:
                freeze_json(root / f"selection-{seed}.json", selections[str(seed)])
            candidates = [
                {
                    "checkpoint": str(starter),
                    "sha256": file_hash(starter),
                    "assessment": initial,
                    "origin": "starter",
                    "trained": False,
                    "selection_budget": 0,
                    "evaluation_label": "starter",
                }
            ]
            candidates += [
                {**row, "evaluation_label": f"seed-{seed}"}
                for seed, row in selections.items()
            ]
            nominee = max(
                candidates, key=lambda row: selection_rank(row["assessment"], config)
            )
            freeze_json(root / "nominee.json", nominee)
            verify_frozen_choices(root, config, selections, nominee)
            # No final confirmation observation is opened before every selection
            # and the single nominee have immutable receipts.
            stage.update(stage="independent-confirmation", seed=None)
            models = {
                "starter": starter,
                **{
                    f"seed-{seed}": Path(row["checkpoint"])
                    for seed, row in selections.items()
                },
            }
            for label, checkpoint in models.items():
                stage["candidate"] = label
                evaluate(
                    checkpoint,
                    root / "final" / label,
                    config["primary_specs"] + config["diagnostic_specs"],
                    config["final_pairs"],
                    config["final_seed"],
                )
            for label, checkpoint in [
                ("starter", starter),
                ("nominee", Path(nominee["checkpoint"])),
            ]:
                stage.update(stage="signal-ablation", candidate=label)
                for partner in ("literal-v2", "trusting-v2"):
                    for mode in ablation_modes(config):
                        evaluate(
                            checkpoint,
                            root / "ablations" / label,
                            [
                                {
                                    "id": partner + "-" + mode,
                                    "opponent": "native",
                                    "partner": partner,
                                    "players": 4,
                                }
                            ],
                            config["ablation_pairs"],
                            config["ablation_seed"],
                            mode,
                        )
            report = final_comparison(root, config, recipe_choice, selections, nominee)
            report["signal_ablations"] = {}
            report["discovery"] = recipe_choice
            for label in ("starter", "nominee"):
                for partner in ("literal-v2", "trusting-v2"):
                    normal = read(
                        root / "ablations" / label / (partner + "-normal.json")
                    )
                    report["signal_ablations"][label + "/" + partner] = {}
                    for mode in ablation_modes(config)[1:]:
                        delta = clusters(normal) - clusters(
                            read(
                                root
                                / "ablations"
                                / label
                                / (partner + "-" + mode + ".json")
                            )
                        )
                        report["signal_ablations"][label + "/" + partner][mode] = {
                            "benefit": float(delta.mean()),
                            "benefit_ci95": interval(delta),
                        }
            report["coverage"] = coverage(root, config)
            report["preservation"] = verify_preservation(
                read(root / "preservation.json")
            )
            report["source_commit"] = snapshot["source_commit"]
            atomic_json(root / "comparison.json", report)
            if (
                not report["coverage"]["complete"]
                or not report["preservation"]["passed"]
            ):
                raise ValueError(
                    "completion coverage or checkpoint preservation failed"
                )
            stage.update(stage="complete", candidate=None)
            status(
                "complete",
                approved=False,
                result=str(root / "comparison.json"),
                child_pids=[],
                work_items=[],
            )
        except BudgetExhausted as error:
            atomic_json(
                root / "partial-results.json",
                {
                    "status": "budget-exhausted",
                    "coverage": coverage(root, config),
                    "preservation": verify_preservation(
                        read(root / "preservation.json")
                    ),
                    "approved": False,
                    "scope": "incomplete development coverage; compare completed receipts without implying full protocol success",
                },
            )
            status(
                "budget-exhausted",
                error=str(error),
                approved=False,
                child_pids=[],
                work_items=[],
            )
        except CampaignInterrupted as error:
            status("interrupted", error=str(error), child_pids=[], work_items=[])
        except BaseException as error:
            status("failed", error=str(error), child_pids=[], work_items=[])
            raise


if __name__ == "__main__":
    main()
