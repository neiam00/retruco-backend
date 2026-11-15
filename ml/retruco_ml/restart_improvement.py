"""Explicit source upgrade for a failed, never-updated improvement campaign.

Old artifacts stay immutable. Ordinary --resume continues to reject source edits.
"""

import argparse
import copy
import fcntl
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from .controlled_train import atomic_json
from .recurrent import file_hash
from .snapshot import verify_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old, new = args.source.resolve(), args.output.resolve()
    if new.exists() or new.is_relative_to(old) or old.is_relative_to(new):
        raise ValueError("restart requires a new, separate campaign directory")
    with (old / "campaign.lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old_status = json.loads((old / "status.json").read_text())
        if old_status["status"] != "failed":
            raise ValueError("only a failed campaign can be upgraded here")
        old_source = verify_snapshot(old / "source")
        trials = list((old / "discovery").glob("*/checkpoint.pt"))
        if len(trials) != 1:
            raise ValueError("restart only supports the first failed discovery trial")
        origin = trials[0]
        saved = torch.load(origin, map_location="cpu", weights_only=True)
        if (
            saved.get("training_kind") != "improvement-mappo-v1"
            or saved["phase"] != "warmup"
            or any(saved[k] != 0 for k in ("update", "warmups", "choices", "decisions"))
        ):
            raise ValueError(
                "this upgrade must not discard any completed training or critic update"
            )
        if saved["actor_optimizer"]["state"] or saved["critic_optimizer"]["state"]:
            raise ValueError("expected untouched optimizer state")
        calibration = origin.parent / "calibration.pt"
        if file_hash(calibration) != saved["calibration_sha256"]:
            raise ValueError("origin calibration changed")
        metadata = json.loads((old / "inputs/manifest.json").read_text())
        legacy = Path(metadata["legacy_origin"]["origin"])
        if file_hash(legacy) != metadata["legacy_origin"]["sha256"]:
            raise ValueError("original legacy input changed")
        controls = Path(metadata["candidates"]["C-101"]["origin"]).parent.parent
        command = [
            sys.executable,
            "-m",
            "retruco_ml.improvement_campaign",
            "--output",
            str(new),
            "--config",
            str(old / "configuration.json"),
            "--inputs",
            str(old / "inputs/manifest.json"),
            "--legacy",
            str(legacy),
            "--controls",
            str(controls),
            "--prepare",
        ]
        subprocess.run(command, check=True)
        new_source = verify_snapshot(new / "source")
        target = new / origin.relative_to(old)
        target.parent.mkdir(parents=True)
        shutil.copy2(calibration, target.parent / "calibration.pt")
        migrated = copy.deepcopy(saved)
        for key, value in {
            "output": str(target.parent),
            "inputs": str(new / "inputs/manifest.json"),
            "simulator": str(new / "source/bin/selfplay"),
        }.items():
            migrated["configuration"][key] = value
        source = new / "source/ml/retruco_ml"
        migrated["contracts"]["source"] = {
            p.name: file_hash(p) for p in source.glob("*.py")
        }
        migrated["contracts"]["simulator_sha256"] = file_hash(
            new / "source/bin/selfplay"
        )
        migrated["contracts"]["inputs_sha256"] = file_hash(new / "inputs/manifest.json")
        new_metadata = json.loads((new / "inputs/manifest.json").read_text())
        starter = migrated["configuration"]["warm_start"]
        if (
            migrated["contracts"]["starter_sha256"]
            != new_metadata["candidates"][starter]["sha256"]
        ):
            raise ValueError("restart changed the selected actor/critic input")
        temporary = target.with_suffix(".tmp")
        torch.save(migrated, temporary)
        temporary.replace(target)
        # Independent round-trip audit: only these explicitly permitted metadata
        # fields may change. Tensor bytes, optimizer state, RNG and all counters
        # must match exactly, including critic calibration and best states.
        actual = torch.load(target, map_location="cpu", weights_only=True)

        def equal(left, right):
            if isinstance(left, torch.Tensor):
                return (
                    isinstance(right, torch.Tensor)
                    and left.dtype == right.dtype
                    and torch.equal(left, right)
                )
            if isinstance(left, dict):
                return left.keys() == right.keys() and all(
                    equal(v, right[k]) for k, v in left.items()
                )
            if isinstance(left, (tuple, list)):
                return (
                    type(left) is type(right)
                    and len(left) == len(right)
                    and all(equal(a, b) for a, b in zip(left, right, strict=True))
                )
            return left == right

        for key in saved:
            if key not in ("configuration", "contracts") and not equal(
                saved[key], actual[key]
            ):
                raise ValueError("restart changed state: " + key)
        if file_hash(target.parent / "calibration.pt") != saved["calibration_sha256"]:
            raise ValueError("restart changed calibration bytes")
        provenance = {
            "kind": "never-updated-improvement-source-upgrade-v1",
            "approved": False,
            "origin_root": str(old),
            "origin_checkpoint": str(origin),
            "origin_checkpoint_sha256": file_hash(origin),
            "checkpoint": str(target),
            "checkpoint_sha256": file_hash(target),
            "calibration_sha256": saved["calibration_sha256"],
            "old_source_commit": old_source["source_commit"],
            "new_source_commit": new_source["source_commit"],
            "old_source_manifest_sha256": file_hash(old / "source/manifest.json"),
            "new_source_manifest_sha256": file_hash(new / "source/manifest.json"),
            "configuration_changes": {
                k: [saved["configuration"][k], actual["configuration"][k]]
                for k in ("output", "inputs", "simulator")
            },
            "contract_changes": ["source", "simulator_sha256", "inputs_sha256"],
            "state_preserved_exactly": True,
            "prior_campaign_seconds": old_status["campaign_seconds"],
        }
        atomic_json(new / "restart-provenance.json", provenance)
        status = json.loads((new / "status.json").read_text())
        status["campaign_seconds"] += old_status["campaign_seconds"]
        atomic_json(new / "status.json", status)
        print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
