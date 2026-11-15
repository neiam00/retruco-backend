"""Freeze Python sources, lockfile and native simulator before a training job."""

import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from .artifacts import digest


def verify_snapshot(source):
    manifest = json.loads((source / "manifest.json").read_text())
    for key, value in (
        ("torch", str(torch.__version__)),
        ("python", platform.python_version()),
        ("numpy", np.__version__),
    ):
        if key in manifest and manifest[key] != value:
            raise ValueError("frozen training environment changed: " + key)
    actual = {
        str(p.relative_to(source))
        for p in source.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p != source / "manifest.json"
    }
    if actual != set(manifest["files"]):
        raise ValueError("frozen training source inventory changed")
    for name, expected in manifest["files"].items():
        if digest(source / name) != expected:
            raise ValueError("frozen training source changed: " + name)
    return manifest


def freeze_source(destination, repository, allow_dirty=False):
    if destination.exists():
        return verify_snapshot(destination)

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=repository, text=True
        ).strip()

    revision = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise ValueError("commit training sources before a non-smoke job")
    subprocess.run(
        ["go", "build", "-o", "bin/selfplay", "./cmd/selfplay"],
        cwd=repository,
        check=True,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="snapshot-", dir=destination.parent))
    try:
        (staging / "ml").mkdir()
        (staging / "bin").mkdir()
        shutil.copytree(
            repository / "ml/retruco_ml",
            staging / "ml/retruco_ml",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy2(repository / "ml/uv.lock", staging / "ml/uv.lock")
        shutil.copy2(repository / "bin/selfplay", staging / "bin/selfplay")
        if git("rev-parse", "HEAD") != revision or (
            not dirty and git("status", "--porcelain")
        ):
            raise ValueError("training sources changed while creating snapshot")
        files = {
            str(p.relative_to(staging)): digest(p)
            for p in staging.rglob("*")
            if p.is_file()
        }
        manifest = {
            "files": files,
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "source_commit": revision,
            "source_dirty": dirty,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def training_snapshot(checkpoint):
    for parent in checkpoint.parents:
        candidate = parent / "source/manifest.json"
        if candidate.is_file():
            return candidate
    raise ValueError("teacher checkpoint has no frozen training source manifest")
