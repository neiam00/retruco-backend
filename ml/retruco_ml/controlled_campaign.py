"""Serial, frozen-source factorial trials; never write a parent campaign."""

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .controlled_compare import (
    cluster_scores,
    compare,
    crossed_interval,
    read,
    summarize,
)
from .controlled_train import atomic_json
from .recurrent import file_hash
from .snapshot import freeze_source, verify_snapshot


def finish_comparison(root, report):
    config = report["configuration"]
    diagnostic = {}
    for arm in config["arms"]:
        diagnostic[arm] = {}
        for partner in config["diagnostic_partners"]:
            parent = read(root / "baseline-diagnostic" / f"{partner}-normal.json")
            values = np.stack(
                [
                    cluster_scores(
                        read(
                            root
                            / f"{arm}-seed-{seed}"
                            / "diagnostic"
                            / f"{partner}-normal.json"
                        )
                    )
                    for seed in config["seeds"]
                ]
            )
            diagnostic[arm][partner] = summarize(values, cluster_scores(parent))
    report["diagnostic_partners"] = diagnostic
    best = report["exploratory_best_arm"]
    ablations = {}
    for label in ("parent", best):
        folders = (
            [root / "ablations/parent"]
            if label == "parent"
            else [
                root / "ablations" / f"{best}-seed-{seed}" for seed in config["seeds"]
            ]
        )
        ablations[label] = {}
        for partner in config["ablation_partners"]:
            normal = np.stack(
                [
                    cluster_scores(read(folder / f"{partner}-normal.json"))
                    for folder in folders
                ]
            )
            ablations[label][partner] = {}
            for mode in ("no-send", "no-receive"):
                missing = np.stack(
                    [
                        cluster_scores(read(folder / f"{partner}-{mode}.json"))
                        for folder in folders
                    ]
                )
                effect = normal - missing
                ablations[label][partner][mode] = {
                    "benefit": float(effect.mean()),
                    "benefit_ci95": crossed_interval(effect),
                }
    report["signal_ablations"] = ablations
    report["ablation_scope"] = (
        "parent and exploratory best observed arm across all trial seeds, on separate development deals; architecture chosen using primary validation"
    )
    report["release_holdouts_used"] = False
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.resume:
        contract_path = root / "campaign-contract.json"
        if (
            not contract_path.exists()
            or read(contract_path).get("kind") != "controlled-cooperation-screen-v1"
        ):
            raise ValueError(
                "resume requires this controlled campaign; parent runs are read-only"
            )
        if not (root / "configuration.json").exists():
            raise ValueError("campaign must be prepared before resume")
    if not args.resume and root.exists():
        raise ValueError("new controlled campaign output must not exist")
    root.mkdir(parents=True, exist_ok=args.resume)
    with (root / "campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("controlled campaign already running")
        context = {"stage": "prepare"}

        def status(state, **extra):
            atomic_json(
                root / "status.json",
                {"status": state, "pid": os.getpid(), **context, **extra},
            )

        try:
            if not args.resume:
                if not args.config or not args.inputs:
                    raise ValueError(
                        "new campaign requires configuration and immutable input manifest"
                    )
                config = read(args.config)
                if (
                    config["version"] != 1
                    or config["arms"] != list("ABCD")
                    or config["seeds"] != [101, 102, 103]
                    or config["parent_seed"] != 101
                ):
                    raise ValueError("unsupported factorial screen")
                config["smoke"] = args.smoke
                if args.smoke:
                    config.update(
                        seeds=[101],
                        choices_per_run=32,
                        steps=32,
                        workers=1,
                        environments=1,
                        calibration_choices=16,
                        warmup_max=2,
                        validation_pairs=2,
                        diagnostic_pairs=2,
                        ablation_pairs=2,
                        diagnostic_partners=["sparse-truthful"],
                    )
                shutil.copytree(args.inputs.resolve().parent, root / "inputs")
                # Freeze before saving the ready configuration. A failed prepare
                # remains an isolated failed output, never a runnable partial job.
                manifest = freeze_source(
                    root / "source",
                    Path(__file__).resolve().parents[2],
                    allow_dirty=args.smoke,
                )
                atomic_json(root / "configuration.json", config)
                atomic_json(
                    root / "campaign-contract.json",
                    {
                        "kind": "controlled-cooperation-screen-v1",
                        "configuration_sha256": file_hash(root / "configuration.json"),
                        "inputs_sha256": file_hash(root / "inputs/manifest.json"),
                        "source_manifest_sha256": file_hash(
                            root / "source/manifest.json"
                        ),
                    },
                )
            config = read(root / "configuration.json")
            contract = read(root / "campaign-contract.json")
            for filename, key in [
                ("configuration.json", "configuration_sha256"),
                ("inputs/manifest.json", "inputs_sha256"),
                ("source/manifest.json", "source_manifest_sha256"),
            ]:
                if file_hash(root / filename) != contract[key]:
                    raise ValueError(
                        "controlled campaign contract changed: " + filename
                    )
            manifest = verify_snapshot(root / "source")
            if args.prepare:
                status("prepared", source_commit=manifest["source_commit"])
                return
            environment = dict(os.environ, PYTHONPATH=str(root / "source/ml"))
            metadata = read(root / "inputs/manifest.json")
            parent = root / "inputs" / metadata["parents"]["101"]["file"]
            reference = root / "inputs" / metadata["reference"]["file"]
            simulator = root / "source/bin/selfplay"

            def run(command, log):
                with log.open("a") as stream:
                    child = subprocess.Popen(
                        [sys.executable, "-u", "-m", *command],
                        cwd=root / "source/ml",
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                    status("running", child_pid=child.pid, command=command)
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

            def evaluate(checkpoint, folder, partners, pairs, seed, modes=("normal",)):
                folder.mkdir(parents=True, exist_ok=True)
                for partner in partners:
                    for mode in modes:
                        path = folder / f"{partner}-{mode}.json"
                        receipt = path.with_suffix(".sha256")
                        if path.exists():
                            saved = read(path)
                            if (
                                not receipt.exists()
                                or receipt.read_text().strip() != file_hash(path)
                                or saved["checkpoint_sha256"] != file_hash(checkpoint)
                                or saved["seed"] != seed
                                or saved["partner"] != partner
                                or saved["signals"] != mode
                                or saved["reference_sha256"] != file_hash(reference)
                                or any(
                                    c["pairs"] != pairs for c in saved["configurations"]
                                )
                            ):
                                raise ValueError(
                                    "evaluation receipt mismatch: " + str(path)
                                )
                            continue
                        run(
                            [
                                "retruco_ml.controlled_evaluate",
                                str(checkpoint),
                                "--reference",
                                str(reference),
                                "--simulator",
                                str(simulator),
                                "--output",
                                str(path),
                                "--partner",
                                partner,
                                "--pairs",
                                str(pairs),
                                "--seed",
                                str(seed),
                                "--signals",
                                mode,
                                "--players",
                                "0" if partner == "reference-cpu" else "4",
                            ],
                            root / "evaluation.log",
                        )
                        receipt.write_text(file_hash(path) + "\n")

            context.update(stage="baseline")
            evaluate(
                parent,
                root / "baseline",
                config["primary_partners"],
                config["validation_pairs"],
                config["validation_seed"],
            )
            evaluate(
                parent,
                root / "baseline-diagnostic",
                config["diagnostic_partners"],
                config["diagnostic_pairs"],
                config["diagnostic_seed"],
            )
            for offset, seed in enumerate(config["seeds"]):
                order = config["arms"][offset:] + config["arms"][:offset]
                for arm in order:
                    folder = root / f"{arm}-seed-{seed}"
                    context.update(stage="training", arm=arm, seed=seed)
                    command = [
                        "retruco_ml.controlled_train",
                        "--output",
                        str(folder),
                        "--inputs",
                        str(root / "inputs/manifest.json"),
                        "--simulator",
                        str(simulator),
                        "--arm",
                        arm,
                        "--seed",
                        str(seed),
                        "--choices",
                        str(config["choices_per_run"]),
                        "--steps",
                        str(config["steps"]),
                        "--workers",
                        str(config["workers"]),
                        "--environments",
                        str(config["environments"]),
                        "--calibration-choices",
                        str(config["calibration_choices"]),
                        "--warmup-max",
                        str(config["warmup_max"]),
                        "--memory-gib",
                        str(config["memory_gib"]),
                    ]
                    if (folder / "checkpoint.pt").exists():
                        command += ["--resume"]
                    state = (
                        read(folder / "status.json")
                        if (folder / "status.json").exists()
                        else {}
                    )
                    if state.get("status") != "complete":
                        run(command, root / "training.log")
                    checkpoint = folder / "checkpoint.pt"
                    context.update(stage="validation")
                    evaluate(
                        checkpoint,
                        folder / "validation",
                        config["primary_partners"],
                        config["validation_pairs"],
                        config["validation_seed"],
                    )
                    context.update(stage="partner-diagnostics")
                    evaluate(
                        checkpoint,
                        folder / "diagnostic",
                        config["diagnostic_partners"],
                        config["diagnostic_pairs"],
                        config["diagnostic_seed"],
                    )
            context.update(stage="comparison", arm=None, seed=None)
            report = compare(root)
            atomic_json(root / "comparison-primary.json", report)
            best = report["exploratory_best_arm"]
            context.update(stage="signal-ablation", arm="parent", seed=None)
            evaluate(
                parent,
                root / "ablations/parent",
                config["ablation_partners"],
                config["ablation_pairs"],
                config["ablation_seed"],
                ("normal", "no-send", "no-receive"),
            )
            for seed in config["seeds"]:
                context.update(stage="signal-ablation", arm=best, seed=seed)
                evaluate(
                    root / f"{best}-seed-{seed}/checkpoint.pt",
                    root / "ablations" / f"{best}-seed-{seed}",
                    config["ablation_partners"],
                    config["ablation_pairs"],
                    config["ablation_seed"],
                    ("normal", "no-send", "no-receive"),
                )
            report = finish_comparison(root, report)
            report["source_commit"] = manifest["source_commit"]
            atomic_json(root / "comparison.json", report)
            context.update(stage="complete", arm=None, seed=None)
            status("complete", approved=False, result=str(root / "comparison.json"))
        except BaseException as error:
            status("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
