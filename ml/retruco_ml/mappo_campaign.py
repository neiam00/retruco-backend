"""Serial controlled experiments, optionally queued behind an existing campaign.

No process signals, checkpoint writes, or source mutations in the watched run.
"""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from .recurrent import file_hash
from .snapshot import freeze_source, verify_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--wait-for",
        type=Path,
        help="wait read-only for curriculum.lock and require successful completion",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="freeze inputs/source and exit before starting any experiment",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="bounded pipeline check: one seed, one epoch, 32 decisions, no evaluation",
    )
    args = parser.parse_args()
    root = args.output.resolve()
    if not args.resume and root.exists():
        parser.error("campaign output must be new")
    if args.resume and not root.is_dir():
        parser.error("campaign output missing")
    if not args.resume and (args.warm_start is None or args.config is None):
        parser.error("new campaign requires parent and config")
    if args.wait_for and (
        root == args.wait_for.resolve() or args.wait_for.resolve() in root.parents
    ):
        parser.error("output cannot be inside watched campaign")
    root.mkdir(parents=True, exist_ok=args.resume)
    with (root / "campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("experiment campaign is already live")

        def status(state, **extra):
            p = root / "status.tmp"
            p.write_text(
                json.dumps({"status": state, "pid": os.getpid(), **extra}) + "\n"
            )
            p.replace(root / "status.json")

        try:
            if not args.resume:
                config = json.loads(args.config.read_text())
                if (
                    config["version"] != 1
                    or config["arms"] != ["A", "B", "C", "D"]
                    or len(set(config["seeds"])) < 3
                ):
                    raise ValueError("four arms and three independent seeds required")
                config["smoke"] = args.smoke
                config["wait_for"] = (
                    str(args.wait_for.resolve()) if args.wait_for else None
                )
                if args.smoke:
                    config.update(
                        seeds=[config["seeds"][0]],
                        decisions_per_arm_seed=32,
                        steps_per_update=32,
                        workers=1,
                        environments=1,
                        epochs=1,
                    )
                (root / "configuration.json").write_text(
                    json.dumps(config, indent=2) + "\n"
                )
                # Single open file handles atomic replacement of a live source path.
                with (
                    args.warm_start.open("rb") as source,
                    (root / "parent.pt").open("xb") as out,
                ):
                    shutil.copyfileobj(source, out)
                (root / "parent.json").write_text(
                    json.dumps(
                        {
                            "path": str(args.warm_start.resolve()),
                            "sha256": file_hash(root / "parent.pt"),
                        }
                    )
                    + "\n"
                )
                freeze_source(
                    root / "source",
                    Path(__file__).resolve().parents[2],
                    allow_dirty=args.smoke,
                )
            else:
                config = json.loads((root / "configuration.json").read_text())
            if (
                args.resume
                and args.wait_for
                and config.get("wait_for") != str(args.wait_for.resolve())
            ):
                raise ValueError("resume cannot change the watched campaign")
            manifest = verify_snapshot(root / "source")
            if (
                file_hash(root / "parent.pt")
                != json.loads((root / "parent.json").read_text())["sha256"]
            ):
                raise ValueError("frozen parent changed")
            # The runner itself must also use the frozen source on later resumes.
            environment = dict(os.environ, PYTHONPATH=str(root / "source/ml"))
            if args.prepare:
                status(
                    "prepared",
                    source_commit=manifest["source_commit"],
                    parent_sha256=file_hash(root / "parent.pt"),
                )
                return
            if config.get("wait_for"):
                watched_root = Path(config["wait_for"])
                if root == watched_root or watched_root in root.parents:
                    raise ValueError("experiment output overlaps watched campaign")
                status("waiting-for-curriculum", watched=str(watched_root))
                with (watched_root / "curriculum.lock").open("r") as watched:
                    while True:
                        try:
                            fcntl.flock(watched, fcntl.LOCK_SH | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            time.sleep(60)
                    previous = json.loads((watched_root / "status.json").read_text())
                    if previous["status"] != "complete":
                        raise RuntimeError(
                            "watched curriculum stopped without completing; experiments not started"
                        )

            def run(command, log):
                with log.open("a") as out:
                    child = subprocess.Popen(
                        [sys.executable, "-u", "-m", *command],
                        cwd=root / "source/ml",
                        env=environment,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                    )
                    status("running", command=command, child_pid=child.pid)
                    try:
                        code = child.wait()
                    except BaseException:
                        child.terminate()
                        child.wait()
                        raise
                    if code:
                        raise RuntimeError(f"experiment exited {code}; inspect {log}")

            for seed in config["seeds"]:
                # Rotate order across seeds to reduce systematic machine-load bias.
                arms = config["arms"]
                offset = config["seeds"].index(seed) % 4
                for arm in arms[offset:] + arms[:offset]:
                    folder = root / f"{arm}-seed-{seed}"
                    final = folder / "status.json"
                    state = json.loads(final.read_text()) if final.exists() else {}
                    if state.get("status") != "complete":
                        command = [
                            "retruco_ml.mappo_train",
                            "--output",
                            str(folder),
                            "--arm",
                            arm,
                            "--seed",
                            str(seed),
                            "--decisions",
                            str(config["decisions_per_arm_seed"]),
                            "--steps",
                            str(config["steps_per_update"]),
                            "--workers",
                            str(config["workers"]),
                            "--environments",
                            str(config["environments"]),
                            "--sequence",
                            str(config["sequence_events"]),
                            "--epochs",
                            str(config["epochs"]),
                            "--memory-gib",
                            str(config["memory_gib"]),
                            "--max-hours",
                            str(config["max_hours_per_arm_seed"]),
                            "--simulator",
                            str(root / "source/bin/selfplay"),
                        ]
                        command += (
                            ["--resume"]
                            if (folder / "checkpoint.pt").exists()
                            else ["--warm-start", str(root / "parent.pt")]
                        )
                        run(command, root / f"{arm}-seed-{seed}.log")
                        if json.loads(final.read_text())["status"] != "complete":
                            raise RuntimeError(
                                "arm reached time budget before equal-decision target; resume explicitly"
                            )
                    if not config.get("smoke"):
                        for partner in config["validation_partners"]:
                            report = folder / f"validation-{partner}.json"
                            if report.exists():
                                continue
                            run(
                                [
                                    "retruco_ml.mappo_evaluate",
                                    str(folder / "checkpoint.pt"),
                                    "--output",
                                    str(report),
                                    "--pairs",
                                    str(config["validation_pairs"]),
                                    "--seed",
                                    str(config["validation_seed"]),
                                    "--partner",
                                    partner,
                                    "--simulator",
                                    str(root / "source/bin/selfplay"),
                                ],
                                root / f"{arm}-seed-{seed}.validation.log",
                            )
            if not config.get("smoke"):
                folders = [
                    root / f"{arm}-seed-{seed}"
                    for seed in config["seeds"]
                    for arm in config["arms"]
                ]
                common = min(
                    torch.load(
                        f / "checkpoint.pt", map_location="cpu", weights_only=True
                    )["total_seconds"]
                    for f in folders
                )
                chosen = {}
                for folder in folders:
                    options = [
                        *sorted((folder / "snapshots").glob("*.pt")),
                        folder / "checkpoint.pt",
                    ]
                    candidates = [
                        (
                            torch.load(p, map_location="cpu", weights_only=True)[
                                "total_seconds"
                            ],
                            p,
                        )
                        for p in options
                    ]
                    candidates = [
                        (seconds, p)
                        for seconds, p in candidates
                        if seconds <= common + 1e-6
                    ]
                    if not candidates:
                        raise RuntimeError(
                            "no common-wall checkpoint; reduce snapshot interval for a fresh comparison"
                        )
                    seconds, checkpoint = max(candidates, key=lambda row: row[0])
                    chosen[folder.name] = {
                        "checkpoint": str(checkpoint),
                        "seconds": seconds,
                        "target_seconds": common,
                        "gap_seconds": common - seconds,
                    }
                    for partner in config["validation_partners"]:
                        report = folder / f"validation-{partner}-wall.json"
                        if report.exists():
                            continue
                        run(
                            [
                                "retruco_ml.mappo_evaluate",
                                str(checkpoint),
                                "--output",
                                str(report),
                                "--pairs",
                                str(config["validation_pairs"]),
                                "--seed",
                                str(config["validation_seed"]),
                                "--partner",
                                partner,
                                "--simulator",
                                str(root / "source/bin/selfplay"),
                            ],
                            root / f"{folder.name}.validation.log",
                        )
                (root / "wall-checkpoints.json").write_text(
                    json.dumps(chosen, indent=2) + "\n"
                )
                for budget in ("decisions", "wall"):
                    run(
                        ["retruco_ml.mappo_compare", str(root), "--budget", budget],
                        root / f"comparison-{budget}.log",
                    )
            status("complete", approved=False, source_commit=manifest["source_commit"])
        except BaseException as error:
            status("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
