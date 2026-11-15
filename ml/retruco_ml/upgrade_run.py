"""Fork a stopped run onto committed trainer sources without changing its state."""

import argparse
import fcntl
import json
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

import torch

from .artifacts import digest
from .bundle import training_commit
from .model import ENCODER_VERSION, Policy
from .snapshot import freeze_source, training_snapshot, verify_snapshot


def verify_curriculum_source(root, pilots):
    """An explicit upgrade may replace code, never silently edit a snapshot."""
    source = root / "source"
    verify_snapshot(source)
    marker = root / "source-upgrade.json"
    if not marker.exists():
        if (source / "manifest.json").read_bytes() != (
            pilots / "source/manifest.json"
        ).read_bytes():
            raise ValueError(
                "curriculum snapshot differs from the selected pilot snapshot"
            )
        return
    record = json.loads(marker.read_text())
    if (
        record["format"] != 1
        or digest(source / "manifest.json") != record["source_manifest_sha256"]
    ):
        raise ValueError("upgraded trainer source changed")
    previous = record["previous_sources"]
    if not previous:
        raise ValueError("missing prior trainer provenance")
    for item in previous:
        path = (root / item["directory"]).resolve()
        if root.resolve() not in path.parents:
            raise ValueError("prior source is outside the run")
        verify_snapshot(path)
        if digest(path / "manifest.json") != item["manifest_sha256"]:
            raise ValueError("prior source manifest changed")
    if (root / previous[0]["directory"] / "manifest.json").read_bytes() != (
        pilots / "source/manifest.json"
    ).read_bytes():
        raise ValueError("upgraded curriculum does not descend from its pilot source")


def upgrade(original, destination, repository):
    original = original.resolve()
    destination = destination.resolve()
    repository = repository.resolve()
    if (
        destination.exists()
        or original == destination
        or original in destination.parents
        or destination in original.parents
    ):
        raise ValueError("destination must be a new directory outside the original run")
    configuration = original / "configuration.json"
    is_curriculum = configuration.exists() and "heads_up_hours" in json.loads(
        configuration.read_text()
    )
    checkpoints = sorted(
        p
        for p in original.rglob("checkpoint.pt")
        if "upgrade-inputs" not in p.parts and "source" not in p.parts
    )
    if not checkpoints:
        raise ValueError("source run has no resumable checkpoints")
    with ExitStack() as locks:
        lock_paths = ([original / "curriculum.lock"] if is_curriculum else []) + [
            p.parent / "train.lock" for p in checkpoints
        ]
        for path in lock_paths:
            handle = locks.enter_context(path.open("a"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    "source run is still running; stop its supervisor/trainer before upgrading"
                )
        manifest = training_snapshot(checkpoints[0])
        old_source = manifest.parent
        verify_snapshot(old_source)
        training_commit(manifest, repository)
        previous = []
        marker = original / "source-upgrade.json"
        if marker.exists():
            previous = json.loads(marker.read_text())["previous_sources"]
        if is_curriculum:
            config = json.loads((original / "configuration.json").read_text())
            verify_curriculum_source(original, Path(config["pilots"]))
        for path in checkpoints:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if (
                saved.get("encoder") != ENCODER_VERSION
                or saved.get("training_kind", "ppo") != "ppo"
            ):
                raise ValueError("only compatible PPO checkpoints can use this upgrade")
            Policy().load_state_dict(saved["model"])
            if not {
                "optimizer",
                "population",
                "random",
                "torch_random",
                "update",
                "total_seconds",
            }.issubset(saved):
                raise ValueError("checkpoint lacks optimizer, RNG or progress state")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="upgrade-", dir=destination.parent))
        try:
            # Dependency lockfiles are part of a frozen source's hash inventory.
            # Exclude only process locks, never uv.lock (or future package locks).
            shutil.copytree(
                original,
                staging,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.tmp",
                    "train.lock",
                    "curriculum.lock",
                    "evaluation.lock",
                    "verification.lock",
                    "qat.lock",
                ),
            )
            old_hash = digest(manifest)
            archive = "source-before-" + old_hash[:16]
            if (staging / "source").exists():
                (staging / "source").rename(staging / archive)
            else:
                shutil.copytree(
                    old_source,
                    staging / archive,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
            previous = [*previous, {"directory": archive, "manifest_sha256": old_hash}]
            inputs = {}
            (staging / "upgrade-inputs").mkdir(exist_ok=True)
            for path in checkpoints:
                relative = path.relative_to(original)
                expected = digest(path)
                if digest(staging / relative) != expected:
                    raise ValueError("checkpoint changed during upgrade")
                archived = "upgrade-inputs/" + expected + ".pt"
                shutil.copy2(path, staging / archived)
                inputs[str(relative)] = {"sha256": expected, "archived": archived}
            freeze_source(staging / "source", repository)
            record = {
                "format": 1,
                "original_run": str(original),
                "kind": "curriculum" if is_curriculum else "trainer",
                "source_manifest_sha256": digest(staging / "source/manifest.json"),
                "previous_sources": previous,
                "checkpoint_inputs": inputs,
                "approved": False,
                "note": "Weights, optimizer, RNG, population and phase budgets are copied unchanged.",
            }
            (staging / "source-upgrade.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
            # Old status files describe another process; the new runner writes
            # its own PID when explicitly started by the user.
            (staging / "status.json").write_text(
                json.dumps(
                    {"status": "ready-to-resume", "pid": None, "approved": False}
                )
                + "\n"
            )
            if is_curriculum:
                verify_curriculum_source(staging, Path(config["pilots"]))
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = upgrade(args.source, args.output, Path(__file__).resolve().parents[2])
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "kind": record["kind"],
                "checkpoints": len(record["checkpoint_inputs"]),
                "approved": False,
            }
        )
    )


if __name__ == "__main__":
    main()
