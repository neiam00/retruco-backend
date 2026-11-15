"""Checkpoint-preserving, bounded cooperation/population x anchoring trials."""

import argparse
import fcntl
import hashlib
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from .controlled_train import atomic_json
from .cooperation_refusal_compare import (
    ablation_comparison,
    compare,
    read,
    validate_report,
)
from .improvement_budget import KEY
from .recurrent import file_hash
from .snapshot import freeze_source, verify_snapshot

KIND = "cooperation-refusal-v1"


class BudgetExhausted(RuntimeError):
    pass


def preservation_snapshot(repository, output):
    files = {}
    for path in sorted((repository / "ml/runs").rglob("*.pt")):
        if not path.resolve().is_relative_to(output.resolve()):
            files[str(path.resolve())] = {
                "sha256": file_hash(path),
                "bytes": path.stat().st_size,
            }
    return {
        "files": files,
        "scope": "all pre-existing backend ml/runs .pt files; read-only verification",
    }


def verify_preservation(manifest):
    mismatches = []
    for name, entry in manifest["files"].items():
        path = Path(name)
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or file_hash(path) != entry["sha256"]
        ):
            mismatches.append(name)
    return {
        "checked": len(manifest["files"]),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def validate_config(config):
    expected_arms = {
        arm: {"cooperation_practice": practice, "anchor": anchor}
        for arm, practice, anchor in [
            ("A", False, False),
            ("B", True, False),
            ("C", False, True),
            ("D", True, True),
        ]
    }
    if (
        config.get("version") != 1
        or set(config.get("arms", {})) != set("ABCD")
        or config["arms"] != expected_arms
        or config.get("actor_memory") != 64
        or config.get("critic_hidden") != [256, 128]
        or config.get("heldout_parent") != 103
        or config.get("mode_win_rate_targets") != {"1v1": 0.70, "2v2": 0.65}
        or not config.get("seeds")
        or len(set(config["seeds"])) != len(config["seeds"])
        or config.get("choices", 0) < 1
        or config.get("max_wall_hours", 0) <= 0
    ):
        raise ValueError("invalid controlled cooperation/refusal configuration")
    if (not config.get("smoke") and len(config["seeds"]) != 3) or any(
        type(seed) is not int for seed in config["seeds"]
    ):
        raise ValueError(
            "production trials require three distinct integer training seeds"
        )
    for key in (
        "choices",
        "steps",
        "workers",
        "environments",
        "batch_size",
        "calibration_choices",
        "evaluation_workers",
        "league_interval",
    ):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError("invalid positive training capacity: " + key)
    if type(config.get("warmup_max")) is not int or config["warmup_max"] < 2:
        raise ValueError("critic warmup needs at least two updates")
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
            raise ValueError("invalid finite positive configuration: " + key)
    expected = {"native-headsup"} | {
        "native-" + p
        for p in ("literal-v2", "trusting-v2", "cautious-v2", "reference-human")
    }
    if {s["id"] for s in config["primary_specs"]} != expected or len(
        config["primary_specs"]
    ) != 5:
        raise ValueError("complete native target coverage is required")
    if any(
        spec["opponent"] != "native"
        or spec["players"] != (2 if spec["id"] == "native-headsup" else 4)
        or spec["id"]
        != ("native-headsup" if spec["players"] == 2 else "native-" + spec["partner"])
        for spec in config["primary_specs"]
    ):
        raise ValueError("native targets cannot use mixed-opponent evaluations")
    ranges = []
    for name in ("primary_seed", "diagnostic_seed", "ablation_seed"):
        pairs = config[name.replace("_seed", "_pairs")]
        if (
            type(pairs) is not int
            or type(config[name]) is not int
            or pairs < 2
            or not 100000 <= config[name] < config[name] + pairs < 200000
        ):
            raise ValueError("development seed range or pair count invalid")
        ranges.append(set(range(config[name], config[name] + pairs)))
    if any(ranges[i] & ranges[j] for i in range(3) for j in range(i)):
        raise ValueError("development evaluation deal ranges must be disjoint")
    ids = [spec["id"] for spec in config["primary_specs"] + config["diagnostic_specs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate evaluation identity")


def coverage(root, config):
    paths = []
    for family in ("primary", "diagnostic"):
        paths += [
            root / "baseline" / family / f"{spec['id']}.json"
            for spec in config[family + "_specs"]
        ]
        paths += [
            root / "trials" / f"{arm}-seed-{seed}" / family / f"{spec['id']}.json"
            for arm in config["arms"]
            for seed in config["seeds"]
            for spec in config[family + "_specs"]
        ]
    completed = [
        str(path.relative_to(root))
        for path in paths
        if path.exists()
        and path.with_suffix(".sha256").exists()
        and path.with_suffix(".sha256").read_text().strip() == file_hash(path)
    ]
    return {
        "completed_primary_and_diagnostic": len(completed),
        "expected_primary_and_diagnostic": len(paths),
        "completed_paths": completed,
        "incomplete": len(completed) != len(paths),
    }


def validate_inputs(root):
    manifest = read(root / "manifest.json")
    for group in ("parents", "critics", "candidates"):
        for entry in manifest.get(group, {}).values():
            path = (root / entry["file"]).resolve()
            if (
                not path.is_relative_to(root.resolve())
                or file_hash(path) != entry["sha256"]
            ):
                raise ValueError("frozen input changed or outside root")
    reference = manifest["reference"]
    path = (root / reference["file"]).resolve()
    if (
        not path.is_relative_to(root.resolve())
        or file_hash(path) != reference["sha256"]
    ):
        raise ValueError("frozen reference changed")
    starter = manifest["candidates"]["legacy-champion"]
    if starter["file"] != "legacy-champion-lstm16.pt":
        raise ValueError("expected exact stronger starter migration")
    return manifest


def validate_endpoint(saved, config, arm, seed, starter_sha, inputs_sha):
    settings = saved.get("configuration", {})
    expected = {
        "variant": arm,
        "seed": seed,
        "choices": config["choices"],
        "warm_start": "legacy-champion",
        "actor_memory": 64,
        "critic_hidden": [256, 128],
        "gae_lambda": 0.99,
        "self_play_league": True,
        "league_interval": config["league_interval"],
        "heldout_parent": config["heldout_parent"],
        "cooperation_practice": config["arms"][arm]["cooperation_practice"],
    }
    expected.update(
        {
            key: config[key] if config["arms"][arm]["anchor"] else 0.0
            for key in ("parent_anchor", "response_anchor", "signal_anchor")
        }
    )
    if (
        saved.get("training_kind") != "improvement-mappo-v1"
        or saved.get("choices", 0) < config["choices"]
        or any(settings.get(key) != value for key, value in expected.items())
        or saved.get("contracts", {}).get("starter_sha256") != starter_sha
        or saved.get("contracts", {}).get("inputs_sha256") != inputs_sha
    ):
        raise ValueError("fixed-endpoint provenance or training intervention mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--inputs",
        type=Path,
        help="existing frozen improvement input manifest; copied without migration",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.resume:
        if read(root / "campaign-contract.json").get("kind") != KIND:
            raise ValueError("wrong campaign kind")
    elif root.exists():
        raise ValueError("new campaign output must not exist")
    root.mkdir(parents=True, exist_ok=args.resume)
    with (root / "campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("controlled campaign already running")
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
            if not args.resume:
                if not args.config or not args.inputs:
                    raise ValueError("new campaign needs config and frozen inputs")
                config = read(args.config)
                config["smoke"] = args.smoke
                if args.smoke:
                    config.update(
                        seeds=[401],
                        choices=32,
                        steps=32,
                        workers=1,
                        environments=1,
                        batch_size=8,
                        calibration_choices=16,
                        warmup_max=2,
                        primary_pairs=2,
                        diagnostic_pairs=2,
                        ablation_pairs=2,
                    )
                validate_config(config)
                repository = Path(__file__).resolve().parents[2]
                atomic_json(
                    root / "preservation.json", preservation_snapshot(repository, root)
                )
                validate_inputs(args.inputs.resolve().parent)
                shutil.copytree(args.inputs.resolve().parent, root / "inputs")
                snapshot = freeze_source(
                    root / "source", repository, allow_dirty=args.smoke
                )
                atomic_json(root / "configuration.json", config)
                atomic_json(
                    root / "campaign-contract.json",
                    {
                        "kind": KIND,
                        "configuration_sha256": file_hash(root / "configuration.json"),
                        "inputs_sha256": file_hash(root / "inputs/manifest.json"),
                        "source_manifest_sha256": file_hash(
                            root / "source/manifest.json"
                        ),
                        "preservation_sha256": file_hash(root / "preservation.json"),
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
                    raise ValueError("immutable campaign contract changed: " + name)
            snapshot = verify_snapshot(root / "source")
            metadata = validate_inputs(root / "inputs")
            if args.prepare:
                status("prepared", source_commit=snapshot["source_commit"])
                return
            env = dict(os.environ, PYTHONPATH=str(root / "source/ml"))
            env[KEY] = str(started + config["max_wall_hours"] * 3600 - previous)
            frozen = (
                root / "source/ml/retruco_ml/cooperation_refusal_campaign.py"
            ).resolve()
            if Path(__file__).resolve() != frozen:
                lock.close()
                os.chdir(root / "source/ml")
                os.execve(
                    sys.executable,
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "retruco_ml.cooperation_refusal_campaign",
                        "--output",
                        str(root),
                        "--resume",
                    ],
                    env,
                )

            def deadline():
                if (
                    previous + time.monotonic() - started
                    >= config["max_wall_hours"] * 3600
                ):
                    raise BudgetExhausted("bounded campaign wall-clock budget reached")

            def run_many(items):
                for start in range(0, len(items), config["evaluation_workers"]):
                    deadline()
                    children = []
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
                                )
                            children.append((child, command, log))
                        while any(child.poll() is None for child, _, _ in children):
                            deadline()
                            alive = [
                                (child, command, log)
                                for child, command, log in children
                                if child.poll() is None
                            ]
                            if not alive:
                                break
                            status(
                                "running",
                                child_pids=[x[0].pid for x in alive],
                                work_items=[
                                    {"pid": c.pid, "command": cmd, "log": str(log)}
                                    for c, cmd, log in alive
                                ],
                            )
                            try:
                                alive[0][0].wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                pass
                        for child, command, log in children:
                            if child.returncode == 75:
                                raise BudgetExhausted(
                                    "child stopped at cooperative campaign deadline"
                                )
                            if child.returncode:
                                raise RuntimeError(
                                    f"child {child.pid} exited {child.returncode}; inspect {log}"
                                )
                    except BaseException:
                        for child, _, _ in children:
                            if child.poll() is None:
                                child.terminate()
                        for child, _, _ in children:
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait()
                        raise

            simulator = root / "source/bin/selfplay"
            starter = (
                root / "inputs" / metadata["candidates"]["legacy-champion"]["file"]
            )

            def evaluate(checkpoint, folder, specs, pairs, seed, signals="normal"):
                folder.mkdir(parents=True, exist_ok=True)
                digest = file_hash(checkpoint)
                pending = []
                for spec in specs:
                    path = folder / f"{spec['id']}.json"
                    receipt = path.with_suffix(".sha256")
                    if path.exists():
                        if (
                            not receipt.exists()
                            or receipt.read_text().strip() != file_hash(path)
                        ):
                            raise ValueError(
                                "incomplete or changed evaluation receipt: " + str(path)
                            )
                        validate_report(
                            read(path),
                            spec,
                            pairs,
                            seed,
                            digest,
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
                    path = folder / f"{spec['id']}.json"
                    receipt = path.with_suffix(".sha256")
                    if not receipt.exists() or receipt.read_text().strip() != file_hash(
                        path
                    ):
                        raise ValueError("evaluation receipt missing")
                    validate_report(
                        read(path),
                        spec,
                        pairs,
                        seed,
                        digest,
                        contract["inputs_sha256"],
                        signals,
                    )

            def endpoint(folder, arm, seed):
                candidate = folder / "candidate.pt"
                receipt = candidate.with_suffix(".sha256")
                if candidate.exists():
                    if not receipt.exists() or receipt.read_text().strip() != file_hash(
                        candidate
                    ):
                        raise ValueError("immutable endpoint changed")
                    validate_endpoint(
                        torch.load(candidate, map_location="cpu", weights_only=True),
                        config,
                        arm,
                        seed,
                        file_hash(starter),
                        contract["inputs_sha256"],
                    )
                    return candidate
                checkpoint = folder / "checkpoint.pt"
                current = (
                    torch.load(checkpoint, map_location="cpu", weights_only=True)
                    if checkpoint.exists()
                    else {}
                )
                if (
                    current.get("choices", 0) < config["choices"]
                    or read(folder / "status.json").get("status") != "complete"
                ):
                    command = [
                        "retruco_ml.improvement_train",
                        "--output",
                        str(folder),
                        "--inputs",
                        str(root / "inputs/manifest.json"),
                        "--simulator",
                        str(simulator),
                        "--variant",
                        arm,
                        "--seed",
                        str(seed),
                        "--warm-start",
                        "legacy-champion",
                        "--choices",
                        str(config["choices"]),
                        "--gae-lambda",
                        ".99",
                        "--actor-memory",
                        "64",
                        "--critic-hidden",
                        "256",
                        "128",
                        "--self-play-league",
                        "--league-interval",
                        str(config["league_interval"]),
                        "--heldout-parent",
                        str(config["heldout_parent"]),
                    ]
                    for key in (
                        "steps",
                        "workers",
                        "environments",
                        "batch_size",
                        "warmup_max",
                        "calibration_choices",
                        "memory_gib",
                    ):
                        command += ["--" + key.replace("_", "-"), str(config[key])]
                    settings = config["arms"][arm]
                    if settings["cooperation_practice"]:
                        command += ["--cooperation-practice"]
                    if settings["anchor"]:
                        for key in (
                            "parent_anchor",
                            "response_anchor",
                            "signal_anchor",
                        ):
                            command += ["--" + key.replace("_", "-"), str(config[key])]
                    if checkpoint.exists():
                        command += ["--resume"]
                    run_many([(command, folder.parent / (folder.name + ".log"))])
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                validate_endpoint(
                    saved,
                    config,
                    arm,
                    seed,
                    file_hash(starter),
                    contract["inputs_sha256"],
                )
                shutil.copy2(checkpoint, candidate)
                receipt.write_text(file_hash(candidate) + "\n")
                return candidate

            for family in ("primary", "diagnostic"):
                stage.update(stage="baseline", family=family)
                evaluate(
                    starter,
                    root / "baseline" / family,
                    config[family + "_specs"],
                    config[family + "_pairs"],
                    config[family + "_seed"],
                )
            for offset, seed in enumerate(config["seeds"]):
                arms = list(config["arms"])
                order = arms[offset:] + arms[:offset]
                for arm in order:
                    stage.update(
                        stage="training",
                        arm=arm,
                        seed=seed,
                        target_choices=config["choices"],
                    )
                    folder = root / "trials" / f"{arm}-seed-{seed}"
                    checkpoint = endpoint(folder, arm, seed)
                    for family in ("primary", "diagnostic"):
                        stage.update(stage="validation", family=family)
                        evaluate(
                            checkpoint,
                            folder / family,
                            config[family + "_specs"],
                            config[family + "_pairs"],
                            config[family + "_seed"],
                        )
            stage.update(stage="comparison", arm=None, seed=None)
            report = compare(root, config)
            atomic_json(root / "comparison-primary.json", report)
            arm = report["exploratory_best_arm"]
            labels = {"starter": starter}
            labels.update(
                {
                    f"{arm}-seed-{seed}": root
                    / "trials"
                    / f"{arm}-seed-{seed}"
                    / "candidate.pt"
                    for seed in config["seeds"]
                }
            )
            atomic_json(
                root / "ablation-choice.json",
                {
                    "arm": arm,
                    "labels": {k: str(v) for k, v in labels.items()},
                    "scope": "exploratory best native2v2 mean; all trained seeds retained",
                },
            )
            for label, checkpoint in labels.items():
                stage.update(stage="signal-ablation", candidate=label)
                for mode in ("normal", "no-send", "no-receive"):
                    spec = {
                        "id": mode,
                        "opponent": "native",
                        "partner": "literal-v2",
                        "players": 4,
                    }
                    evaluate(
                        checkpoint,
                        root / "ablations" / label,
                        [spec],
                        config["ablation_pairs"],
                        config["ablation_seed"],
                        mode,
                    )
            report["signal_ablations"] = ablation_comparison(root, labels)
            report["preservation"] = verify_preservation(
                read(root / "preservation.json")
            )
            report["source_commit"] = snapshot["source_commit"]
            report["expected_evaluation_reports"] = (
                1 + len(config["arms"]) * len(config["seeds"])
            ) * (len(config["primary_specs"]) + len(config["diagnostic_specs"])) + (
                1 + len(config["seeds"])
            ) * 3
            report["completed_evaluation_reports"] = len(
                list(root.rglob("*.sha256"))
            ) - len(config["arms"]) * len(config["seeds"])
            atomic_json(root / "comparison.json", report)
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
                    "scope": "incomplete development coverage; no completed factorial comparison or acceptance claim",
                },
            )
            status(
                "budget-exhausted",
                error=str(error),
                approved=False,
                child_pids=[],
                work_items=[],
            )
        except BaseException as error:
            status("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
