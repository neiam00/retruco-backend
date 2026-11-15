"""Bounded serial Model D campaign; parents and historical evidence are read-only."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from .recurrent import file_hash
from .snapshot import freeze_source, verify_snapshot
from .statistics import paired_loss


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def subset(report, players):
    return {
        **report,
        "configurations": [
            c for c in report["configurations"] if c["players"] == players
        ],
    }


def assess(candidate, baseline, config):
    required = config["selection_partners"]
    out = {}
    safe = True
    scores = []
    for partner in required:
        before, after = subset(baseline[partner], 4), subset(candidate[partner], 4)
        loss = paired_loss(before, after)
        gain = -loss["loss"]
        ci = [-loss["cluster_ci95"][1], -loss["cluster_ci95"][0]]
        mean = sum(c["win_rate"] for c in after["configurations"]) / 4
        scores.append(mean)
        safe &= ci[0] >= -config["nonregression_tolerance"]
        out[partner] = {"win_rate": mean, "gain": gain, "gain_ci95": ci}
    loss = paired_loss(
        subset(baseline["reference-cpu"], 2), subset(candidate["reference-cpu"], 2)
    )
    ci = [-loss["cluster_ci95"][1], -loss["cluster_ci95"][0]]
    safe &= ci[0] >= -config["nonregression_tolerance"]
    return {
        "macro": sum(scores) / len(scores),
        "nonregression_pass": bool(safe),
        "headsup_gain": -loss["loss"],
        "headsup_gain_ci95": ci,
        "partners": out,
        "approved": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    inputs = root / "inputs/manifest.json"
    if (
        not inputs.exists()
        or json.loads(inputs.read_text()).get("kind") != "cooperation-inputs-v1"
    ):
        raise ValueError("prepare immutable Model D input copies first")
    if args.resume and not (root / "configuration.json").exists():
        raise ValueError("campaign not prepared")
    if not args.resume and (root / "configuration.json").exists():
        raise ValueError("campaign already exists; use resume")
    with (root / "campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("cooperation campaign already live")

        def status(state, **extra):
            atomic_json(
                root / "status.json", {"status": state, "pid": os.getpid(), **extra}
            )

        try:
            if not args.resume:
                if not args.config:
                    raise ValueError("new campaign requires config")
                config = json.loads(args.config.read_text())
                config["smoke"] = args.smoke
                if (
                    config.get("version") != 1
                    or not config["seeds"]
                    or len(set(config["seeds"])) != len(config["seeds"])
                    or not set(config["seeds"]) <= {101, 102, 103}
                    or config["signal_policy"] != "truthful-cpu-v1"
                ):
                    raise ValueError("unsupported campaign contract")
                if args.smoke:
                    config.update(
                        seeds=[101],
                        hours_per_seed=0.0001,
                        chunk_hours=0.0001,
                        steps=32,
                        workers=1,
                        environments=1,
                        validation_pairs=2,
                        sanity_pairs=2,
                    )
                freeze_source(
                    root / "source",
                    Path(__file__).resolve().parents[2],
                    allow_dirty=args.smoke,
                )
                atomic_json(root / "configuration.json", config)
            else:
                config = json.loads((root / "configuration.json").read_text())
            if not (root / "source").exists():
                freeze_source(
                    root / "source",
                    Path(__file__).resolve().parents[2],
                    allow_dirty=config.get("smoke", False),
                )
            manifest = verify_snapshot(root / "source")
            if args.prepare:
                status("prepared", source_commit=manifest["source_commit"])
                return
            environment = dict(os.environ, PYTHONPATH=str(root / "source/ml"))
            metadata = json.loads(inputs.read_text())
            simulator = root / "source/bin/selfplay"
            reference = root / "inputs" / metadata["reference"]["file"]
            context = {"stage": "baseline", "seed": None}

            def run(command, log):
                with log.open("a") as out:
                    child = subprocess.Popen(
                        [sys.executable, "-u", "-m", *command],
                        cwd=root / "source/ml",
                        env=environment,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                    )
                    status("running", command=command, child_pid=child.pid, **context)
                    try:
                        code = child.wait()
                    except BaseException:
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        raise
                    if code:
                        raise RuntimeError(f"child exited {code}; inspect {log}")

            def evaluate(
                checkpoint, folder, pairs, seed, partners, signals="normal", players=0
            ):
                folder.mkdir(parents=True, exist_ok=True)
                reports = {}
                digest = file_hash(checkpoint)
                for partner in partners:
                    mode = players or (0 if partner == "reference-cpu" else 4)
                    path = folder / f"{partner}-{signals}.json"
                    if path.exists():
                        report = json.loads(path.read_text())
                        if (
                            report["checkpoint_sha256"] != digest
                            or report["seed"] != seed
                            or report["partner"] != partner
                            or report["signals"] != signals
                            or report["reference_sha256"] != file_hash(reference)
                        ):
                            raise ValueError("stale evaluation receipt")
                    else:
                        run(
                            [
                                "retruco_ml.cooperation_evaluate",
                                str(checkpoint),
                                "--reference",
                                str(reference),
                                "--output",
                                str(path),
                                "--simulator",
                                str(simulator),
                                "--pairs",
                                str(pairs),
                                "--seed",
                                str(seed),
                                "--partner",
                                partner,
                                "--signals",
                                signals,
                                "--players",
                                str(mode),
                            ],
                            root / "evaluation.log",
                        )
                        report = json.loads(path.read_text())
                    reports[partner] = report
                return reports

            baselines = {}
            selection = {}
            for seed in config["seeds"]:
                context.update(stage="baseline", seed=seed)
                parent = root / "inputs" / metadata["parents"][str(seed)]["file"]
                baselines[seed] = evaluate(
                    parent,
                    root / f"baseline-{seed}",
                    config["validation_pairs"],
                    config["validation_seed"],
                    config["validation_partners"],
                )
                baseline_score = assess(baselines[seed], baselines[seed], config)[
                    "macro"
                ]
                path = root / f"selection-{seed}.json"
                if path.exists():
                    selection[seed] = json.loads(path.read_text())
                else:
                    selection[seed] = {
                        "checkpoint": str(parent),
                        "sha256": file_hash(parent),
                        "macro": baseline_score,
                        "origin": "parent",
                        "approved": False,
                        "candidates": [],
                    }
                    atomic_json(path, selection[seed])
            while True:
                pending = False
                for seed in config["seeds"]:
                    context.update(stage="validation", seed=seed)
                    folder = root / f"D-seed-{seed}"
                    checkpoint = folder / "checkpoint.pt"
                    seconds = 0.0
                    if checkpoint.exists():
                        seconds = torch.load(
                            checkpoint, map_location="cpu", weights_only=True
                        )["total_seconds"]
                    receipt = folder / "chunk.json"
                    if receipt.exists() and checkpoint.exists():
                        timing = json.loads(receipt.read_text())
                        if timing["checkpoint_sha256"] == file_hash(checkpoint):
                            seconds = timing["total_seconds"]
                    # Every completed chunk is evaluated before more training,
                    # including recovery after a supervisor interruption.
                    if checkpoint.exists():
                        digest = file_hash(checkpoint)
                        if not any(
                            c["sha256"] == digest for c in selection[seed]["candidates"]
                        ):
                            number = len(selection[seed]["candidates"]) + 1
                            candidate = folder / f"candidate-{number:03d}.pt"
                            shutil.copy2(checkpoint, candidate)
                            reports = evaluate(
                                candidate,
                                folder / f"validation-{number:03d}",
                                config["validation_pairs"],
                                config["validation_seed"],
                                config["validation_partners"],
                            )
                            assessment = assess(reports, baselines[seed], config)
                            entry = {
                                "checkpoint": str(candidate),
                                "sha256": digest,
                                "assessment": assessment,
                                "training_seconds": seconds,
                            }
                            selection[seed]["candidates"].append(entry)
                            if (
                                assessment["nonregression_pass"]
                                and assessment["macro"]
                                > selection[seed]["macro"]
                                + config["minimum_macro_gain"]
                            ):
                                selection[seed].update(
                                    checkpoint=str(candidate),
                                    sha256=digest,
                                    macro=assessment["macro"],
                                    origin=f"candidate-{number:03d}",
                                )
                            atomic_json(
                                root / f"selection-{seed}.json", selection[seed]
                            )
                    remaining = config["hours_per_seed"] - seconds / 3600
                    if remaining <= 0:
                        continue
                    pending = True
                    command = [
                        "retruco_ml.cooperation_train",
                        "--output",
                        str(folder),
                        "--inputs",
                        str(inputs),
                        "--seed",
                        str(seed),
                        "--hours",
                        str(min(remaining, config["chunk_hours"])),
                        "--steps",
                        str(config["steps"]),
                        "--workers",
                        str(config["workers"]),
                        "--environments",
                        str(config["environments"]),
                        "--memory-gib",
                        str(config["memory_gib"]),
                        "--warmup-updates",
                        str(config["critic_only_warmup_updates"]),
                        "--simulator",
                        str(simulator),
                    ]
                    if checkpoint.exists():
                        command += ["--resume"]
                    context.update(stage="training", seed=seed)
                    run(command, root / f"train-{seed}.log")
                if not pending:
                    break
            if not config.get("smoke"):
                for seed in config["seeds"]:
                    context.update(stage="sanity", seed=seed)
                    checkpoint = Path(selection[seed]["checkpoint"])
                    parent = root / "inputs" / metadata["parents"][str(seed)]["file"]
                    evaluate(
                        parent,
                        root / f"sanity-parent-{seed}",
                        config["sanity_pairs"],
                        config["sanity_seed"],
                        config["partners"],
                    )
                    evaluate(
                        checkpoint,
                        root / f"sanity-{seed}",
                        config["sanity_pairs"],
                        config["sanity_seed"],
                        config["partners"],
                    )
                    for signals in ("no-send", "no-receive"):
                        context.update(stage="signal-ablation", seed=seed)
                        evaluate(
                            checkpoint,
                            root / f"sanity-{seed}",
                            config["sanity_pairs"],
                            config["sanity_seed"],
                            ["literal-v2", "trusting-v2"],
                            signals,
                            4,
                        )
            atomic_json(
                root / "result.json",
                {
                    "selections": selection,
                    "source_commit": manifest["source_commit"],
                    "approved": False,
                    "scope": "synthetic cooperation development campaign; human review pending",
                },
            )
            status("complete", approved=False, result=str(root / "result.json"))
        except BaseException as error:
            status("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
