"""Run three isolated pilot seeds and select by validation, never final test data.

Subprocesses use a frozen source/binary snapshot so further development cannot
change the algorithm between pilots. This runner never publishes a model.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-hours", type=float, default=1)
    parser.add_argument("--pairs", type=int, default=128)
    args = parser.parse_args()
    if args.pilot_hours <= 0 or args.pairs < 2:
        parser.error("positive pilot-hours and at least two validation pairs required")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "campaign.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError("this campaign already has a live supervisor") from error
    source = root / "source"
    backend = Path(__file__).resolve().parents[2]
    if not source.exists():
        staging = root / "source.tmp"
        if staging.exists():
            shutil.rmtree(staging)
        (staging / "ml").mkdir(parents=True)
        shutil.copytree(
            Path(__file__).parent,
            staging / "ml/retruco_ml",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        (staging / "bin").mkdir()
        shutil.copy2(backend / "bin/selfplay", staging / "bin/selfplay")
        shutil.copy2(backend / "ml/uv.lock", staging / "ml/uv.lock")
        hashes = {
            str(p.relative_to(staging)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in staging.rglob("*")
            if p.is_file()
        }
        (staging / "manifest.json").write_text(
            json.dumps({"files": hashes, "torch": torch.__version__}, indent=2) + "\n"
        )
        staging.rename(source)
    manifest = json.loads((source / "manifest.json").read_text())
    for filename, digest in manifest["files"].items():
        if hashlib.sha256((source / filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError("training source snapshot was modified")
    environment = dict(os.environ, PYTHONPATH=str(source / "ml"))
    status = {
        "pid": os.getpid(),
        "started": time.time(),
        "status": "running",
        "pilot_hours": args.pilot_hours,
        "pairs": args.pairs,
    }

    def record(**updates):
        status.update(updates)
        temporary = root / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(root / "status.json")

    def run(command, log):
        with log.open("a") as output:
            child = subprocess.Popen(
                [sys.executable, "-u", "-m", *command],
                cwd=source / "ml",
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            record(child_pid=child.pid, command=command)
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
                raise RuntimeError(f"pilot subprocess exited {code}; see {log}")

    def interrupted(*_):
        raise KeyboardInterrupt("campaign interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    rankings = []
    record()
    try:
        for seed in (42, 43, 44):
            folder = root / f"seed-{seed}"
            folder.mkdir(exist_ok=True)
            checkpoint = folder / "checkpoint.pt"
            elapsed = 0
            if checkpoint.exists():
                elapsed = float(
                    torch.load(checkpoint, map_location="cpu", weights_only=True).get(
                        "total_seconds", 0
                    )
                )
            record(seed=seed, stage="training")
            remaining = args.pilot_hours - elapsed / 3600
            if remaining > 0:
                command = [
                    "retruco_ml.train",
                    "--output",
                    str(folder),
                    "--seed",
                    str(seed),
                    "--hours",
                    str(remaining),
                    "--players",
                    "2",
                    "--workers",
                    "8",
                    "--environments",
                    "64",
                    "--device",
                    "cpu",
                ]
                if checkpoint.exists():
                    command.append("--resume")
                run(command, folder / "train.log")
            record(stage="validation")
            evaluation = folder / "validation.json"
            previous = json.loads(evaluation.read_text()) if evaluation.exists() else {}
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if previous.get("checkpoint_sha256") != digest or any(
                c["pairs"] != args.pairs for c in previous.get("configurations", [])
            ):
                run(
                    [
                        "retruco_ml.evaluate",
                        str(checkpoint),
                        "--pairs",
                        str(args.pairs),
                        "--players",
                        "2",
                        "--seed",
                        "100000",
                        "--output",
                        str(evaluation),
                    ],
                    folder / "validation.log",
                )
            report = json.loads(evaluation.read_text())
            score = sum(c["win_rate"] for c in report["configurations"]) / len(
                report["configurations"]
            )
            rankings.append(
                {
                    "seed": seed,
                    "validation_win_rate": score,
                    "checkpoint": str(checkpoint),
                }
            )
            record(rankings=rankings)
        selected = max(rankings, key=lambda x: x["validation_win_rate"])
        (root / "selection.json").write_text(
            json.dumps(
                {"approved": False, "selected": selected, "rankings": rankings},
                indent=2,
            )
            + "\n"
        )
        record(status="complete", stage="pilot-selection", child_pid=None)
    except BaseException as error:
        record(status="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
