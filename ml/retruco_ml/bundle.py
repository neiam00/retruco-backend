"""Build a provenance-bound candidate bundle. This command cannot approve it."""

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
from pathlib import Path

import torch

from .artifacts import digest
from .export import export
from .model import ENCODER_VERSION, Policy
from .snapshot import training_snapshot
from .team_play_probe import CHOICE_PROFILES


def runtime_sources(runtime):
    paths = [
        runtime / "Cargo.toml",
        runtime / "Cargo.lock",
        runtime / ".cargo/config.toml",
        *sorted((runtime / "src").glob("*.rs")),
    ]
    return hashlib.sha256(
        json.dumps(
            {str(p.relative_to(runtime)): digest(p) for p in paths}, sort_keys=True
        ).encode()
    ).hexdigest()


def training_commit(snapshot, backend):
    metadata = json.loads(snapshot.read_text())
    source = snapshot.parent
    binary = source / "bin/selfplay"
    if digest(binary) != metadata["files"]["bin/selfplay"]:
        raise ValueError("training binary does not match the frozen manifest")
    info = subprocess.check_output(["go", "version", "-m", str(binary)], text=True)
    match = re.search(r"vcs.revision=([0-9a-f]{40})", info)
    if not match or "vcs.modified=false" not in info:
        raise ValueError("training binary has no clean source revision")
    revision = match.group(1)
    for name, expected in metadata["files"].items():
        if name.endswith(".py") or name == "ml/uv.lock":
            if digest(source / name) != expected:
                raise ValueError("training source snapshot changed")
    verify_committed_sources(metadata, revision, backend)
    return revision


def verify_committed_sources(metadata, revision, backend):
    if not re.fullmatch("[0-9a-f]{40}", revision or ""):
        raise ValueError("invalid training source revision")
    for name, expected in metadata["files"].items():
        if name.endswith(".py") or name == "ml/uv.lock":
            committed = subprocess.check_output(
                ["git", "show", f"{revision}:{name}"], cwd=backend
            )
            if hashlib.sha256(committed).hexdigest() != expected:
                raise ValueError(
                    "training source does not match its recorded Git revision"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--training-snapshot", type=Path)
    parser.add_argument(
        "--team-play",
        choices=("closing-card-and-flor-v1", "closing-card-and-flor-v2"),
        help="enable the evaluated public team decision rule after sampling",
    )
    parser.add_argument(
        "--choice-policy",
        choices=CHOICE_PROFILES,
        help="CPU candidate selection provided by the game host and simulator",
    )
    args = parser.parse_args()
    if args.choice_policy and args.team_play != "closing-card-and-flor-v2":
        parser.error("the CPU choice policy requires closing-card-and-flor-v2")
    backend = Path(__file__).resolve().parents[2]
    runtime = backend / "ml/runtime"
    source_hash = runtime_sources(runtime)
    subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "--offline",
            "--release",
            "--target",
            "wasm32v1-none",
        ],
        cwd=runtime,
        check=True,
    )
    if runtime_sources(runtime) != source_hash:
        raise RuntimeError("runtime sources changed during build")
    # Read once: a live trainer can atomically replace its checkpoint meanwhile.
    checkpoint_data = args.checkpoint.read_bytes()
    saved = torch.load(
        io.BytesIO(checkpoint_data), map_location="cpu", weights_only=True
    )
    if saved.get("encoder") != ENCODER_VERSION:
        raise ValueError("incompatible checkpoint encoder")
    if args.team_play and saved.get("kind") != "rmappo-v1":
        raise ValueError("team decision profile requires the format-4 RMAPPO actor")
    if saved.get("kind") == "rmappo-v1":
        from .mappo_export import actor_from_checkpoint
        from .mappo_export import export as recurrent_export

        model = actor_from_checkpoint(saved)
        if args.team_play and (
            model.candidate_features or model.memory.hidden_size != 64
        ):
            raise ValueError("team decision profile requires the format-4 LSTM64 actor")
        manifest = recurrent_export(model, args.destination)
    else:
        model = Policy()
        model.load_state_dict(saved["model"])
        manifest = export(model, args.destination)
    wasm = runtime / "target/wasm32v1-none/release/retruco_policy.wasm"
    shutil.copy2(wasm, args.destination / "policy.wasm")
    snapshot = args.training_snapshot
    if snapshot is None:
        try:
            snapshot = training_snapshot(args.checkpoint)
        except ValueError:
            pass
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=backend, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--", "ml/retruco_ml", "ml/runtime"],
            cwd=backend,
            text=True,
        ).strip()
    )
    manifest.update(
        {
            "model_id": "cpu-v2-" + manifest["sha256"][:16],
            "checkpoint_sha256": hashlib.sha256(checkpoint_data).hexdigest(),
            "training_snapshot_sha256": digest(snapshot) if snapshot else None,
            "training_source_commit": training_commit(snapshot, backend)
            if snapshot
            else None,
            "exporter_commit": commit,
            "runtime_source_commit": None if dirty else commit,
            "runtime_source_sha256": source_hash,
            "source_dirty": dirty,
            "wasm_sha256": digest(wasm),
            "wasm_bytes": wasm.stat().st_size,
            "actor_parameters": manifest.get("parameters", 85127),
            "training_update": saved.get("update", saved.get("updates")),
            "training_seconds": saved.get(
                "total_seconds", saved.get("training_seconds")
            ),
            "training_choices": saved.get("choices"),
            "training_kind": saved.get("kind", saved.get("training_kind", "ppo")),
            "torch_version": str(torch.__version__),
        }
    )
    if args.team_play:
        manifest["team_play"] = args.team_play
        # Older workers validate this existing field and must decline a profile
        # whose post-sampling decisions they cannot execute.
        manifest["architecture"] += "-team-play-" + args.team_play.rsplit("-", 1)[1]
        manifest["model_id"] += "-" + args.team_play
    if args.choice_policy:
        manifest["choice_policy"] = args.choice_policy
        manifest["model_id"] += "-" + args.choice_policy
    if snapshot:
        shutil.copy2(snapshot, args.destination / "training-source-manifest.json")
    if saved.get("training_kind") == "qat":
        teacher_manifest = args.checkpoint.parent / "teacher-source-manifest.json"
        metadata = saved["qat"]
        if digest(teacher_manifest) != metadata["teacher_training_snapshot_sha256"]:
            raise ValueError("QAT teacher training provenance changed")
        verify_committed_sources(
            json.loads(teacher_manifest.read_text()),
            metadata["teacher_training_source_commit"],
            backend,
        )
        for key in (
            "teacher_checkpoint_sha256",
            "teacher_training_source_commit",
            "teacher_training_snapshot_sha256",
        ):
            manifest[key] = metadata[key]
        shutil.copy2(
            teacher_manifest, args.destination / "teacher-training-source-manifest.json"
        )
    model_size = len(
        gzip.compress((args.destination / "policy.bin").read_bytes(), mtime=0)
    )
    wasm_size = len(gzip.compress(wasm.read_bytes(), mtime=0))
    if wasm_size > 64 * 1024 or model_size + wasm_size > 192 * 1024:
        raise ValueError("candidate assets exceed budget before worker overhead")
    manifest["gzip_bytes"] = {"model": model_size, "wasm": wasm_size}
    (args.destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
