"""Resume the selected pilot through heads-up and mixed-mode training phases.

The pilot lock, not a stale status file, determines whether selection is ready.
Training and validation use the pilot's immutable source snapshot throughout.
No phase completion approves a model or touches release branches.
"""

import argparse
import fcntl
import hashlib
import io
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


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checkpoint(path):
    data = path.read_bytes()
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True), sha(
        data
    )


def phase_remaining(saved_seconds, base_seconds, target_hours):
    if not all(math.isfinite(x) for x in (saved_seconds, base_seconds, target_hours)):
        raise ValueError("phase timing must be finite")
    if saved_seconds < base_seconds:
        raise ValueError("phase checkpoint predates its starting state")
    return max(0.0, target_hours - (saved_seconds - base_seconds) / 3600)


def validation_score(report, expected_hash, players, pairs):
    expected = {
        (p, t, f)
        for p in ((players,) if players else (2, 4))
        for t in (12, 24)
        for f in (False, True)
    }
    rows = report["configurations"]
    if report["checkpoint_sha256"] != expected_hash or report["seed"] != 100000:
        raise ValueError("validation checkpoint or seed mismatch")
    if (
        len(rows) != len(expected)
        or {(c["players"], c["target"], c["flor"]) for c in rows} != expected
    ):
        raise ValueError("validation rules coverage mismatch")
    scores = []
    for row in rows:
        values = row["pair_scores"]
        if (
            row["pairs"] != pairs
            or len(values) != pairs
            or any(type(x) not in (int, float) or x not in (0, 0.5, 1) for x in values)
        ):
            raise ValueError("invalid validation paired scores")
        scores.append(sum(values) / pairs)
    return sum(scores) / len(scores)


def promote(folder, metadata, candidate, report, players, pairs):
    digest = sha(candidate.read_bytes())
    score = validation_score(report, digest, players, pairs)
    if score > metadata["best_score"]:
        temporary = folder / "best.tmp"
        shutil.copy2(candidate, temporary)
        temporary.replace(folder / "best.pt")
        atomic_json(folder / "best-validation.json", report)
        metadata.update(best_score=score, best_checkpoint_sha256=digest)
    return score


def selection_if_ready(pilots, smoke=False):
    with (pilots / "campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        status = json.loads((pilots / "status.json").read_text())
        if status["status"] != "complete":
            raise RuntimeError(
                "pilot supervisor is not live and has not completed; inspect/resume it first"
            )
        if not smoke and (
            status.get("pilot_hours", 0) < 1 or status.get("pairs", 0) < 128
        ):
            raise ValueError(
                "pilot training/validation is below the full campaign minimum"
            )
        selection = json.loads((pilots / "selection.json").read_text())
        if {r["seed"] for r in selection["rankings"]} != {42, 43, 44}:
            raise ValueError("three independent pilot seeds required")
        for row in selection["rankings"]:
            path = Path(row["checkpoint"])
            _, digest = checkpoint(path)
            validation = json.loads((path.parent / "validation.json").read_text())
            if validation["checkpoint_sha256"] != digest:
                raise ValueError("pilot selection references a changed checkpoint")
            expected = {(2, t, f) for t in (12, 24) for f in (False, True)}
            configurations = validation["configurations"]
            if (
                validation["seed"] != 100000
                or len(configurations) != 4
                or {(c["players"], c["target"], c["flor"]) for c in configurations}
                != expected
            ):
                raise ValueError("pilot validation coverage or seed mismatch")
            scores = []
            for config in configurations:
                values = config["pair_scores"]
                if (
                    config["pairs"] != status["pairs"]
                    or len(values) != config["pairs"]
                    or any(
                        type(x) not in (int, float) or x not in (0, 0.5, 1)
                        for x in values
                    )
                ):
                    raise ValueError("invalid pilot paired scores")
                scores.append(sum(values) / len(values))
            if row["validation_win_rate"] != sum(scores) / len(scores):
                raise ValueError("pilot selection score differs from observed results")
        if selection["selected"] not in selection["rankings"]:
            raise ValueError("selected checkpoint is not a validated pilot")
        if selection["selected"] != max(
            selection["rankings"], key=lambda row: row["validation_win_rate"]
        ):
            raise ValueError("pilot selection is not the validation winner")
        result = dict(selection["selected"])
        result["checkpoint_sha256"] = checkpoint(Path(result["checkpoint"]))[1]
        return result


def verify_snapshot(source):
    manifest = json.loads((source / "manifest.json").read_text())
    for filename, digest in manifest["files"].items():
        if sha((source / filename).read_bytes()) != digest:
            raise ValueError("training source snapshot changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="allow short pilot fixtures only for a bounded pipeline smoke test",
    )
    parser.add_argument("--heads-up-hours", type=float, default=12)
    parser.add_argument("--mixed-hours", type=float, default=24)
    parser.add_argument("--chunk-hours", type=float, default=1)
    parser.add_argument("--validation-pairs", type=int, default=128)
    args = parser.parse_args()
    hours = (args.heads_up_hours, args.mixed_hours, args.chunk_hours)
    if not all(math.isfinite(h) and h > 0 for h in hours) or args.validation_pairs < 2:
        parser.error("positive hours and at least two validation pairs required")
    if args.heads_up_hours + args.mixed_hours > 48:
        parser.error(
            "training phases must leave room for evaluation within the campaign budget"
        )
    if args.smoke and (
        args.heads_up_hours + args.mixed_hours > 0.01 or args.validation_pairs > 8
    ):
        parser.error(
            "smoke mode is limited to short phases and at most eight validation pairs"
        )
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "curriculum.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "this curriculum already has a live supervisor"
            ) from error
        run_curriculum(args, root)


def run_curriculum(args, root):
    pilots = args.pilots.resolve()
    status = {
        "pid": os.getpid(),
        "status": "waiting-for-pilots",
        "started": time.time(),
        "smoke": args.smoke,
    }

    def record(**updates):
        status.update(updates)
        atomic_json(root / "status.json", status)

    def interrupted(*_):
        raise KeyboardInterrupt("curriculum interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    record()
    try:
        selected = selection_if_ready(pilots, args.smoke)
        while selected is None:
            if not args.wait:
                raise RuntimeError("pilots are still running; use --wait")
            time.sleep(30)
            selected = selection_if_ready(pilots, args.smoke)
        source = root / "source"
        if not source.exists():
            staging = root / "source.tmp"
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(
                pilots / "source", staging, ignore=shutil.ignore_patterns("__pycache__")
            )
            verify_snapshot(staging)
            staging.rename(source)
        verify_snapshot(source)
        from .upgrade_run import verify_curriculum_source

        verify_curriculum_source(root, pilots)
        environment = dict(os.environ, PYTHONPATH=str(source / "ml"))

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
                    raise RuntimeError(f"subprocess exited {code}; inspect {log}")

        starting = Path(selected["checkpoint"])
        if checkpoint(starting)[1] != selected["checkpoint_sha256"]:
            raise ValueError("selected pilot changed after validation")
        configuration = {
            "pilots": str(pilots),
            "selected_sha256": selected["checkpoint_sha256"],
            "heads_up_hours": args.heads_up_hours,
            "mixed_hours": args.mixed_hours,
            "chunk_hours": args.chunk_hours,
            "validation_pairs": args.validation_pairs,
            "smoke": args.smoke,
        }
        configuration_path = root / "configuration.json"
        if (
            configuration_path.exists()
            and json.loads(configuration_path.read_text()) != configuration
        ):
            raise ValueError("resume settings differ from the recorded curriculum")
        atomic_json(configuration_path, configuration)
        for name, players, hours in [
            ("heads-up", 2, args.heads_up_hours),
            ("mixed", 0, args.mixed_hours),
        ]:
            folder = root / name
            folder.mkdir(exist_ok=True)
            metadata_path = folder / "phase.json"
            active = folder / "checkpoint.pt"
            if not metadata_path.exists():
                saved, digest = checkpoint(starting)
                shutil.copy2(starting, active)
                atomic_json(
                    metadata_path,
                    {
                        "base_seconds": saved["total_seconds"],
                        "target_hours": hours,
                        "players": players,
                        "origin_sha256": digest,
                        "chunks": 0,
                        "best_score": -1,
                    },
                )
            metadata = json.loads(metadata_path.read_text())
            if metadata["players"] != players or metadata["target_hours"] != hours:
                raise ValueError(
                    "resume phase settings differ from the recorded curriculum"
                )
            origin = folder / "origin.pt"
            if not origin.exists():
                if checkpoint(starting)[1] != metadata["origin_sha256"]:
                    raise ValueError(
                        "recorded phase origin is unavailable; restore it before resuming"
                    )
                shutil.copy2(starting, origin)
            if checkpoint(origin)[1] != metadata["origin_sha256"]:
                raise ValueError("phase origin changed")
            if metadata.get("origin_validated_sha256") != metadata["origin_sha256"]:
                evaluation = folder / "origin-validation.json"
                report = None
                # A pilot or previous phase may already have the exact baseline
                # assessment. Reuse only matching hashes, seeds and rules coverage.
                for path in (
                    evaluation,
                    starting.parent / "validation.json",
                    starting.parent / "best-validation.json",
                ):
                    try:
                        candidate_report = json.loads(path.read_text())
                        validation_score(
                            candidate_report,
                            metadata["origin_sha256"],
                            players,
                            args.validation_pairs,
                        )
                        report = candidate_report
                        break
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                record(status="running", phase=name, stage="baseline-validation")
                if report is None:
                    run(
                        [
                            "retruco_ml.evaluate",
                            str(origin),
                            "--players",
                            str(players),
                            "--pairs",
                            str(args.validation_pairs),
                            "--seed",
                            "100000",
                            "--output",
                            str(evaluation),
                        ],
                        folder / "validation.log",
                    )
                    report = json.loads(evaluation.read_text())
                atomic_json(evaluation, report)
                promote(
                    folder, metadata, origin, report, players, args.validation_pairs
                )
                metadata["origin_validated_sha256"] = metadata["origin_sha256"]
                atomic_json(metadata_path, metadata)
                record(best_validation_score=metadata["best_score"])
            while True:
                saved, digest = checkpoint(active)
                # A completed or interrupted training chunk is validated before
                # another chunk begins, including after the phase time expires.
                if digest != metadata.get(
                    "last_validated_sha256", metadata["origin_sha256"]
                ):
                    number = metadata["chunks"] + 1
                    candidate = folder / f"candidate-{number:03d}.pt"
                    shutil.copy2(active, candidate)
                    evaluation = folder / f"validation-{number:03d}.json"
                    record(status="running", phase=name, stage="validation")
                    try:
                        report = json.loads(evaluation.read_text())
                    except (FileNotFoundError, ValueError):
                        report = {}
                    if report.get("checkpoint_sha256") != digest:
                        run(
                            [
                                "retruco_ml.evaluate",
                                str(candidate),
                                "--players",
                                str(players),
                                "--pairs",
                                str(args.validation_pairs),
                                "--seed",
                                "100000",
                                "--output",
                                str(evaluation),
                            ],
                            folder / "validation.log",
                        )
                        report = json.loads(evaluation.read_text())
                    promote(
                        folder,
                        metadata,
                        candidate,
                        report,
                        players,
                        args.validation_pairs,
                    )
                    metadata.update(chunks=number, last_validated_sha256=digest)
                    atomic_json(metadata_path, metadata)
                    record(best_validation_score=metadata["best_score"])
                remaining = phase_remaining(
                    saved["total_seconds"], metadata["base_seconds"], hours
                )
                if remaining == 0:
                    break
                record(
                    status="running",
                    phase=name,
                    stage="training",
                    remaining_hours=remaining,
                )
                run(
                    [
                        "retruco_ml.train",
                        "--resume",
                        "--output",
                        str(folder),
                        "--hours",
                        str(min(remaining, args.chunk_hours)),
                        "--players",
                        str(players),
                        "--seed",
                        str(selected["seed"]),
                        "--workers",
                        "8",
                        "--environments",
                        "64",
                        "--device",
                        "cpu",
                    ],
                    folder / "train.log",
                )
            starting = folder / "best.pt"
            if not starting.exists():
                raise RuntimeError("phase has no validated checkpoint")
        atomic_json(
            root / "candidate.json",
            {
                "approved": False,
                "checkpoint": str(starting),
                "checkpoint_sha256": sha(starting.read_bytes()),
                "training_snapshot_sha256": sha(
                    (source / "manifest.json").read_bytes()
                ),
                "selection_basis": "validation only; final release holdout unconsumed",
            },
        )
        record(
            status="complete",
            stage="candidate-ready-for-release-assessment",
            child_pid=None,
        )
    except BaseException as error:
        record(status="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
