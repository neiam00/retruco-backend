"""Bounded source screening, factorial discovery and fresh-seed confirmation."""

import argparse
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from .controlled_train import atomic_json
from .improvement_budget import KEY
from .improvement_compare import assess, discovery_comparison, final_comparison, read
from .improvement_inputs import prepare_inputs
from .recurrent import file_hash
from .snapshot import freeze_source, verify_snapshot


class BudgetExhausted(RuntimeError):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.resume:
        path = root / "campaign-contract.json"
        if not path.exists() or read(path).get("kind") != "model-improvement-v1":
            raise ValueError(
                "only this improvement campaign can be resumed; prior runs remain read-only"
            )
    elif root.exists():
        raise ValueError("new improvement output must not exist")
    root.mkdir(parents=True, exist_ok=args.resume)
    with (root / "campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("improvement campaign already live")
        stage = {"stage": "prepare"}
        old_status = (
            read(root / "status.json") if (root / "status.json").exists() else {}
        )
        previous = old_status.get("campaign_seconds", 0.0)
        started = time.monotonic()

        def status(state, **fields):
            atomic_json(
                root / "status.json",
                {
                    "status": state,
                    "pid": os.getpid(),
                    "campaign_seconds": previous + time.monotonic() - started,
                    "supervisor_source": str(Path(__file__).resolve()),
                    **stage,
                    **fields,
                },
            )

        try:
            if not args.resume:
                if not all((args.config, args.inputs, args.legacy, args.controls)):
                    raise ValueError(
                        "new campaign requires configuration and frozen source candidates"
                    )
                config = read(args.config)
                if config["version"] != 1 or set(config["recipes"]) != {
                    "U95",
                    "U99",
                    "R95",
                    "R99",
                }:
                    raise ValueError("unsupported improvement protocol")
                config["smoke"] = args.smoke
                if args.smoke:
                    original_final_specs = config["final_specs"]
                    config.update(
                        discovery_seeds=[201],
                        confirmation_seeds=[301],
                        discovery_choices=32,
                        confirmation_checkpoints=[32, 64],
                        steps=32,
                        workers=1,
                        environments=1,
                        batch_size=8,
                        warmup_max=2,
                        calibration_choices=16,
                        source_pairs=2,
                        discovery_pairs=2,
                        selection_pairs=2,
                        confirmation_pairs=2,
                        ablation_pairs=2,
                    )
                    config["screen_specs"] = [
                        x
                        for x in config["screen_specs"]
                        if x["id"]
                        in (
                            "native-literal-v2",
                            "parent-literal-v2",
                            "reactive-literal-v2",
                            "native-headsup",
                        )
                    ]
                    config["final_specs"] = (
                        [
                            spec
                            for spec in original_final_specs
                            if spec["id"]
                            in (
                                "native-headsup",
                                "native-literal-v2",
                                "native-reference-human",
                                "native-trusting-v2",
                                "native-cautious-v2",
                            )
                        ]
                        if config.get("mode_win_rate_targets")
                        else config["screen_specs"]
                    )
                prepare_inputs(
                    root / "inputs",
                    args.inputs.resolve(),
                    args.legacy.resolve(),
                    args.controls.resolve(),
                )
                snapshot = freeze_source(
                    root / "source",
                    Path(__file__).resolve().parents[2],
                    allow_dirty=args.smoke,
                )
                atomic_json(root / "configuration.json", config)
                atomic_json(
                    root / "campaign-contract.json",
                    {
                        "kind": "model-improvement-v1",
                        "configuration_sha256": file_hash(root / "configuration.json"),
                        "inputs_sha256": file_hash(root / "inputs/manifest.json"),
                        "source_manifest_sha256": file_hash(
                            root / "source/manifest.json"
                        ),
                    },
                )
            config = read(root / "configuration.json")
            contract = read(root / "campaign-contract.json")
            for name, key in [
                ("configuration.json", "configuration_sha256"),
                ("inputs/manifest.json", "inputs_sha256"),
                ("source/manifest.json", "source_manifest_sha256"),
            ]:
                if file_hash(root / name) != contract[key]:
                    raise ValueError("campaign contract changed: " + name)
            snapshot = verify_snapshot(root / "source")
            if args.prepare:
                status("prepared", source_commit=snapshot["source_commit"])
                return
            metadata = read(root / "inputs/manifest.json")
            env = dict(os.environ, PYTHONPATH=str(root / "source/ml"))
            env[KEY] = str(started + config["max_wall_hours"] * 3600 - previous)
            frozen_file = (
                root / "source/ml/retruco_ml/improvement_campaign.py"
            ).resolve()
            if Path(__file__).resolve() != frozen_file:
                # Supervisor selection code must be frozen too, not only children.
                lock.close()
                os.chdir(root / "source/ml")
                os.execve(
                    sys.executable,
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "retruco_ml.improvement_campaign",
                        "--output",
                        str(root),
                        "--resume",
                    ],
                    env,
                )

            def freeze_json(path, value):
                receipt = path.with_suffix(".sha256")
                if path.exists():
                    if (
                        read(path) != value
                        or not receipt.exists()
                        or receipt.read_text().strip() != file_hash(path)
                    ):
                        raise ValueError(
                            "immutable selection receipt changed: " + str(path)
                        )
                else:
                    atomic_json(path, value)
                    receipt.write_text(file_hash(path) + "\n")

            simulator = root / "source/bin/selfplay"
            cache = {}

            def run(command, log):
                if (
                    previous + time.monotonic() - started
                    >= config["max_wall_hours"] * 3600
                ):
                    raise BudgetExhausted(
                        "bounded campaign wall-clock budget reached at a stage boundary"
                    )
                with (root / log).open("a") as out:
                    child = subprocess.Popen(
                        [sys.executable, "-u", "-m", *command],
                        cwd=root / "source/ml",
                        env=env,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                    )
                    status("running", child_pid=child.pid, command=command)
                    try:
                        while True:
                            try:
                                code = child.wait(timeout=30)
                                break
                            except subprocess.TimeoutExpired:
                                status("running", child_pid=child.pid, command=command)
                    except BaseException:
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        raise
                    if code == 75:
                        raise BudgetExhausted(
                            "child stopped cooperatively at the campaign deadline"
                        )
                    if code:
                        raise RuntimeError(f"child exited {code}; inspect {root / log}")

            def evaluate(
                checkpoint,
                folder,
                specs,
                pairs,
                seed,
                mode="normal",
                card_policy="policy",
            ):
                folder.mkdir(parents=True, exist_ok=True)
                digest = file_hash(checkpoint)
                pending = []
                for spec in specs:
                    path = folder / f"{spec['id']}.json"
                    receipt = path.with_suffix(".sha256")
                    key = (
                        digest,
                        spec["opponent"],
                        spec["partner"],
                        spec["players"],
                        pairs,
                        seed,
                        mode,
                        card_policy,
                    )
                    if path.exists() and not receipt.exists():
                        # A killed evaluator can publish JSON just before its receipt.
                        # Preserve that observation and rerun instead of trusting it.
                        archive = root / "interrupted-evaluations"
                        archive.mkdir(exist_ok=True)
                        target = archive / (
                            hashlib.sha256(str(path).encode()).hexdigest() + ".json"
                        )
                        if target.exists():
                            target = archive / (
                                target.stem + "-" + str(time.time_ns()) + ".json"
                            )
                        shutil.move(str(path), str(target))
                    if path.exists():
                        report = read(path)
                        if (
                            not receipt.exists()
                            or receipt.read_text().strip() != file_hash(path)
                            or report["checkpoint_sha256"] != digest
                            or report["inputs_sha256"] != contract["inputs_sha256"]
                            or report["seed"] != seed
                            or report["pairs"] != pairs
                            or report["signals"] != mode
                            or report.get("card_policy") != card_policy
                            or report["opponent"] != spec["opponent"]
                            or report["partner"] != spec["partner"]
                        ):
                            raise ValueError("changed evaluation receipt: " + str(path))
                        cache[key] = path
                    elif key in cache:
                        shutil.copy2(cache[key], path)
                        receipt.write_text(file_hash(path) + "\n")
                    else:
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
                            mode,
                            "--card-policy",
                            card_policy,
                        ]
                        pending.append((command, path, receipt, key))
                workers = config.get("evaluation_workers", 4)
                logs = root / "evaluation-logs"
                logs.mkdir(exist_ok=True)
                for start in range(0, len(pending), workers):
                    if (
                        previous + time.monotonic() - started
                        >= config["max_wall_hours"] * 3600
                    ):
                        raise BudgetExhausted(
                            "bounded campaign wall-clock budget reached before evaluation batch"
                        )
                    children = []
                    try:
                        for command, path, receipt, key in pending[
                            start : start + workers
                        ]:
                            log = logs / (
                                hashlib.sha256(str(path).encode()).hexdigest()[:20]
                                + ".log"
                            )
                            with log.open("a") as stream:
                                child = subprocess.Popen(
                                    [sys.executable, "-u", "-m", *command],
                                    cwd=root / "source/ml",
                                    env=env,
                                    stdout=stream,
                                    stderr=subprocess.STDOUT,
                                )
                            children.append((child, command, path, receipt, key, log))
                        status(
                            "running",
                            child_pid=children[0][0].pid,
                            command=children[0][1],
                            child_pids=[item[0].pid for item in children],
                            work_items=[
                                {
                                    "pid": item[0].pid,
                                    "command": item[1],
                                    "log": str(item[5]),
                                }
                                for item in children
                            ],
                        )
                        for child, command, path, receipt, key, log in children:
                            while True:
                                try:
                                    code = child.wait(timeout=30)
                                    break
                                except subprocess.TimeoutExpired:
                                    status(
                                        "running",
                                        child_pid=child.pid,
                                        command=command,
                                        child_pids=[
                                            item[0].pid
                                            for item in children
                                            if item[0].poll() is None
                                        ],
                                        work_items=[
                                            {
                                                "pid": item[0].pid,
                                                "command": item[1],
                                                "log": str(item[5]),
                                            }
                                            for item in children
                                        ],
                                    )
                            if code == 75:
                                raise BudgetExhausted(
                                    "evaluation stopped cooperatively at the campaign deadline"
                                )
                            if code:
                                raise RuntimeError(
                                    f"evaluation child {child.pid} exited {code}; inspect {log}"
                                )
                            receipt.write_text(file_hash(path) + "\n")
                            cache[key] = path
                    except BaseException as error:
                        for child, *_ in children:
                            if (
                                isinstance(error, BudgetExhausted)
                                and child.poll() is None
                            ):
                                try:
                                    child.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    pass
                            if child.poll() is None:
                                child.terminate()
                        for child, *_ in children:
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait()
                        raise

            def train(folder, recipe, seed, budget, starter):
                if folder.exists() and not (folder / "checkpoint.pt").exists():
                    archive = root / "interrupted-initialization"
                    archive.mkdir(exist_ok=True)
                    target = archive / folder.name
                    suffix = 0
                    while target.exists():
                        suffix += 1
                        target = archive / f"{folder.name}-{suffix}"
                    shutil.move(str(folder), str(target))
                current = (
                    torch.load(
                        folder / "checkpoint.pt", map_location="cpu", weights_only=True
                    )
                    if (folder / "checkpoint.pt").exists()
                    else {}
                )
                if (
                    current.get("choices", 0) < budget
                    or current.get("configuration", {}).get("choices", 0) < budget
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
                        recipe,
                        "--seed",
                        str(seed),
                        "--warm-start",
                        starter,
                        "--choices",
                        str(budget),
                        "--gae-lambda",
                        str(config["recipes"][recipe]["gae_lambda"]),
                        "--steps",
                        str(config["steps"]),
                        "--workers",
                        str(config["workers"]),
                        "--environments",
                        str(config["environments"]),
                        "--batch-size",
                        str(config["batch_size"]),
                        "--warmup-max",
                        str(config["warmup_max"]),
                        "--calibration-choices",
                        str(config["calibration_choices"]),
                        "--memory-gib",
                        str(config["memory_gib"]),
                    ]
                    command += [
                        "--actor-memory",
                        str(config.get("actor_memory", 16)),
                        "--critic-hidden",
                        *map(str, config.get("critic_hidden", [96, 64])),
                    ]
                    if config["recipes"][recipe]["reactive"]:
                        command += ["--reactive"]
                    if config.get("self_play_league"):
                        command += [
                            "--self-play-league",
                            "--league-interval",
                            str(config.get("league_interval", 50000)),
                        ]
                    if (folder / "checkpoint.pt").exists():
                        command += ["--resume"]
                    run(command, "training.log")
                candidate = folder / f"candidate-budget-{budget:09d}.pt"
                if candidate.exists():
                    receipt = candidate.with_suffix(".sha256")
                    if not receipt.exists() or receipt.read_text().strip() != file_hash(
                        candidate
                    ):
                        raise ValueError("changed immutable candidate")
                else:
                    current = torch.load(
                        folder / "checkpoint.pt", map_location="cpu", weights_only=True
                    )
                    if (
                        current["choices"] < budget
                        or current["configuration"]["choices"] != budget
                    ):
                        raise ValueError(
                            "cannot label a later checkpoint as an earlier candidate"
                        )
                    shutil.copy2(folder / "checkpoint.pt", candidate)
                    candidate.with_suffix(".sha256").write_text(
                        file_hash(candidate) + "\n"
                    )
                saved = torch.load(candidate, map_location="cpu", weights_only=True)
                settings = saved["configuration"]
                if (
                    saved.get("training_kind") != "improvement-mappo-v1"
                    or settings["seed"] != seed
                    or settings["variant"] != recipe
                    or settings["warm_start"] != starter
                    or settings["choices"] != budget
                    or settings["gae_lambda"] != config["recipes"][recipe]["gae_lambda"]
                    or settings["reactive"] != config["recipes"][recipe]["reactive"]
                    or saved["contracts"]["inputs_sha256"] != contract["inputs_sha256"]
                    or saved["contracts"]["starter_sha256"]
                    != metadata["candidates"][starter]["sha256"]
                ):
                    raise ValueError("candidate metadata does not match this trial")
                if settings.get("actor_memory", 16) != config.get(
                    "actor_memory", 16
                ) or settings.get("critic_hidden", [96, 64]) != config.get(
                    "critic_hidden", [96, 64]
                ):
                    raise ValueError("candidate capacity differs from campaign")
                if settings.get("self_play_league", False) != config.get(
                    "self_play_league", False
                ) or settings.get("league_interval", 50000) != config.get(
                    "league_interval", 50000
                ):
                    raise ValueError(
                        "candidate league configuration differs from campaign"
                    )
                return candidate

            candidates = {
                name: root / "inputs" / entry["file"]
                for name, entry in metadata["candidates"].items()
            }
            if config.get("smoke"):
                candidates = {
                    name: candidates[name] for name in ("d-parent", "legacy-champion")
                }
            for name, path in candidates.items():
                if file_hash(path) != metadata["candidates"][name]["sha256"]:
                    raise ValueError("frozen candidate changed: " + name)
            stage.update(stage="source-screening")
            for name, checkpoint in candidates.items():
                stage["candidate"] = name
                evaluate(
                    checkpoint,
                    root / "sources" / name,
                    config["screen_specs"],
                    config["source_pairs"],
                    config["source_seed"],
                )
            assessments = {
                name: assess(
                    root / "sources" / name,
                    root / "sources/d-parent",
                    config["screen_specs"],
                )
                for name in candidates
            }
            eligible = [
                name
                for name, value in assessments.items()
                if value["observed_guard_pass"]
            ]
            starter = max(eligible, key=lambda name: assessments[name]["win_rate"])
            source_choice = {
                "starter": starter,
                "checkpoint": str(candidates[starter]),
                "sha256": file_hash(candidates[starter]),
                "assessments": assessments,
                "approved": False,
            }
            freeze_json(root / "source-choice.json", source_choice)
            stage.update(stage="discovery-baseline", candidate=starter)
            evaluate(
                candidates[starter],
                root / "discovery-baseline",
                config["screen_specs"],
                config["discovery_pairs"],
                config["discovery_seed"],
            )
            for offset, seed in enumerate(config["discovery_seeds"]):
                names = list(config["recipes"])
                order = names[offset:] + names[:offset]
                for recipe in order:
                    folder = root / "discovery" / f"{recipe}-seed-{seed}"
                    stage.update(
                        stage="discovery-training",
                        recipe=recipe,
                        seed=seed,
                        target_choices=config["discovery_choices"],
                    )
                    checkpoint = train(
                        folder, recipe, seed, config["discovery_choices"], starter
                    )
                    stage["stage"] = "discovery-validation"
                    evaluate(
                        checkpoint,
                        folder / "validation",
                        config["screen_specs"],
                        config["discovery_pairs"],
                        config["discovery_seed"],
                    )
            recipe_choice = discovery_comparison(root, config, starter)
            freeze_json(root / "recipe-choice.json", recipe_choice)
            selected_recipe = recipe_choice["selected_recipe"]
            stage.update(stage="selection-baseline", recipe=selected_recipe, seed=None)
            for name in ("d-parent", "starter"):
                checkpoint = (
                    candidates["d-parent"]
                    if name == "d-parent"
                    else candidates[starter]
                )
                evaluate(
                    checkpoint,
                    root / "selection" / name,
                    config["screen_specs"],
                    config["selection_pairs"],
                    config["selection_seed"],
                )
            baseline = root / "selection/d-parent"
            starter_assessment = assess(
                root / "selection/starter", baseline, config["screen_specs"]
            )
            selections = {}
            for seed in config["confirmation_seeds"]:
                folder = root / "confirmation" / f"seed-{seed}"
                best = {
                    "checkpoint": str(candidates[starter]),
                    "sha256": file_hash(candidates[starter]),
                    "assessment": starter_assessment,
                    "origin": "starter",
                    "evaluation_label": "starter",
                    "seed": seed,
                }
                for budget in config["confirmation_checkpoints"]:
                    stage.update(
                        stage="confirmation-training",
                        recipe=selected_recipe,
                        seed=seed,
                        target_choices=budget,
                    )
                    checkpoint = train(folder, selected_recipe, seed, budget, starter)
                    stage["stage"] = "confirmation-selection"
                    evaluation = folder / f"validation-budget-{budget:09d}"
                    evaluate(
                        checkpoint,
                        evaluation,
                        config["screen_specs"],
                        config["selection_pairs"],
                        config["selection_seed"],
                    )
                    result = assess(evaluation, baseline, config["screen_specs"])
                    if (
                        result["observed_guard_pass"]
                        and result["win_rate"] > best["assessment"]["win_rate"] + 0.002
                    ):
                        best = {
                            "checkpoint": str(checkpoint),
                            "sha256": file_hash(checkpoint),
                            "assessment": result,
                            "origin": f"budget-{budget}",
                            "evaluation_label": f"seed-{seed}",
                            "seed": seed,
                        }
                selections[str(seed)] = best
                freeze_json(root / f"selection-{seed}.json", best)
            nominee = max(
                [
                    {
                        "checkpoint": str(candidates[starter]),
                        "sha256": file_hash(candidates[starter]),
                        "assessment": starter_assessment,
                        "origin": "starter",
                        "evaluation_label": "starter",
                    },
                    *selections.values(),
                ],
                key=lambda value: value["assessment"]["win_rate"],
            )
            # Freeze selection before any independent confirmation deal is opened.
            freeze_json(root / "nominee.json", nominee)
            stage.update(stage="independent-confirmation", seed=None)
            models = {
                "d-parent": candidates["d-parent"],
                "starter": candidates[starter],
                **{
                    f"seed-{seed}": Path(selections[str(seed)]["checkpoint"])
                    for seed in config["confirmation_seeds"]
                },
            }
            for name, checkpoint in models.items():
                stage["candidate"] = name
                evaluate(
                    checkpoint,
                    root / "confirmation-evaluation" / name,
                    config["final_specs"],
                    config["confirmation_pairs"],
                    config["confirmation_seed"],
                )
            ablation_specs = [
                {
                    "id": f"{family}-{partner}-{mode}",
                    "opponent": family,
                    "partner": partner,
                    "players": 4,
                }
                for family in ("native", "reactive")
                for partner in ("literal-v2", "trusting-v2")
                for mode in ("normal", "no-send", "no-receive")
            ]
            for name, checkpoint in [
                ("d-parent", candidates["d-parent"]),
                ("nominee", Path(nominee["checkpoint"])),
            ]:
                stage.update(stage="signal-ablation", candidate=name)
                for spec in ablation_specs:
                    actual_mode = (
                        "no-send"
                        if spec["id"].endswith("no-send")
                        else "no-receive"
                        if spec["id"].endswith("no-receive")
                        else "normal"
                    )
                    evaluate(
                        checkpoint,
                        root / "ablations" / name,
                        [spec],
                        config["ablation_pairs"],
                        config["ablation_seed"],
                        actual_mode,
                    )
            for name, checkpoint in [
                ("d-parent", candidates["d-parent"]),
                ("nominee", Path(nominee["checkpoint"])),
            ]:
                stage.update(stage="card-selection-ablation", candidate=name)
                for family in ("native", "reactive"):
                    for partner in ("literal-v2", "trusting-v2"):
                        for card_policy in ("policy", "highest"):
                            spec = {
                                "id": f"{family}-{partner}-{card_policy}",
                                "opponent": family,
                                "partner": partner,
                                "players": 4,
                            }
                            evaluate(
                                checkpoint,
                                root / "card-ablations" / name,
                                [spec],
                                config["ablation_pairs"],
                                config["card_ablation_seed"],
                                card_policy=card_policy,
                            )
            report = final_comparison(
                root, config, source_choice, recipe_choice, selections, nominee
            )
            report["source_commit"] = snapshot["source_commit"]
            atomic_json(root / "comparison.json", report)
            stage.update(stage="complete", candidate=None, seed=None)
            status("complete", approved=False, result=str(root / "comparison.json"))
        except BudgetExhausted as error:
            status("budget-exhausted", error=str(error), approved=False)
        except BaseException as error:
            status("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
