"""Interleaved end-to-end A/B runs on immutable sources and checkpoint copies."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from .artifacts import digest
from .snapshot import freeze_source, verify_snapshot


def continuation_hash(saved):
    """Exclude timings and paths; include every state affecting continuation."""
    result = hashlib.sha256()

    def add(value):
        if isinstance(value, torch.Tensor):
            result.update(str((value.dtype, tuple(value.shape))).encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                add(key)
                add(value[key])
        elif isinstance(value, (tuple, list)):
            result.update(type(value).__name__.encode())
            for item in value:
                add(item)
        else:
            result.update(json.dumps(value, allow_nan=False).encode() + b"\0")

    for key in (
        "model",
        "optimizer",
        "population",
        "random",
        "torch_random",
        "mps_random",
        "update",
        "initial_seed",
        "initialization",
        "fixed_opponent",
        "opponent_checkpoint_sha256",
        "opponent_model_sha256",
    ):
        add(key)
        add(saved.get(key))
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.updates < 2:
        parser.error("need at least one repeat and a warmup plus measured update")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    baseline = args.baseline_source.resolve()
    baseline_manifest = verify_snapshot(baseline)
    optimized = root / "source"
    optimized_manifest = freeze_source(
        optimized, Path(__file__).resolve().parents[2], allow_dirty=args.allow_dirty
    )
    checkpoint = root / "input.pt"
    checkpoint.write_bytes(args.checkpoint.read_bytes())
    report = {
        "approved": False,
        "checkpoint_sha256": digest(checkpoint),
        "baseline_source": baseline_manifest,
        "optimized_source": optimized_manifest,
        "configuration": vars(args)
        | {
            "baseline_source": str(baseline),
            "checkpoint": str(args.checkpoint),
            "output": str(root),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "note": "Serial interleaved trials; first update is warmup. Other local jobs are not stopped. Includes rollout, PPO, memory monitoring and checkpoint saves.",
        "trials": [],
        "summaries": [],
    }

    def save():
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    variants = [
        ("baseline-json", baseline, None),
        ("packed-json", optimized, "json"),
        ("packed-binary", optimized, "binary"),
    ]
    for players in (2, 4):
        expected = None
        for repeat in range(args.repeats):
            order = variants[repeat % 3 :] + variants[: repeat % 3]
            if repeat % 2:
                order = list(reversed(order))
            for name, source, ipc in order:
                folder = root / f"{players}-{repeat}-{name}"
                folder.mkdir()
                (folder / "checkpoint.pt").write_bytes(checkpoint.read_bytes())
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "retruco_ml.train",
                    "--resume",
                    "--output",
                    str(folder),
                    "--updates",
                    str(args.updates),
                    "--steps",
                    str(args.steps),
                    "--players",
                    str(players),
                    "--workers",
                    str(args.workers),
                    "--environments",
                    str(args.environments),
                    "--device",
                    "cpu",
                    "--hours",
                    "1",
                    "--memory-gib",
                    "4",
                ]
                if ipc:
                    command += ["--ipc", ipc]
                print(
                    json.dumps({"players": players, "repeat": repeat, "variant": name}),
                    flush=True,
                )
                with (folder / "train.log").open("w") as log:
                    subprocess.run(
                        command,
                        cwd=source / "ml",
                        env=dict(os.environ, PYTHONPATH=str(source / "ml")),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                metrics = [
                    json.loads(line)
                    for line in (folder / "metrics.jsonl").read_text().splitlines()
                ]
                state = torch.load(
                    folder / "checkpoint.pt", map_location="cpu", weights_only=True
                )
                fingerprint = continuation_hash(state)
                if expected is None:
                    expected = fingerprint
                decisions = sum(m["decisions"] for m in metrics[1:])
                seconds = metrics[-1]["elapsed_seconds"] - metrics[0]["elapsed_seconds"]
                report["trials"].append(
                    {
                        "players": players,
                        "repeat": repeat,
                        "variant": name,
                        "decisions": decisions,
                        "seconds": seconds,
                        "decisions_per_second": decisions / seconds,
                        "peak_group_rss_bytes": max(
                            m["peak_group_rss_bytes"] for m in metrics
                        ),
                        "continuation_hash": fingerprint,
                        "identical_continuation": fingerprint == expected,
                        "log_sha256": digest(folder / "train.log"),
                        "metrics_sha256": digest(folder / "metrics.jsonl"),
                    }
                )
                save()
                if fingerprint != expected:
                    raise RuntimeError("optimization changed continuation state")
        rows = [r for r in report["trials"] if r["players"] == players]
        rates = {
            name: statistics.median(
                r["decisions_per_second"] for r in rows if r["variant"] == name
            )
            for name, _, _ in variants
        }
        report["summaries"].append(
            {
                "players": players,
                "median_decisions_per_second": rates,
                "packing_speedup": rates["packed-json"] / rates["baseline-json"],
                "combined_speedup": rates["packed-binary"] / rates["baseline-json"],
            }
        )
        save()
    report["complete"] = True
    save()
    print(json.dumps(report["summaries"]))


if __name__ == "__main__":
    main()
