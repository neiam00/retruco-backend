"""Immutable candidate copies and distribution-preserving legacy migration."""

import copy
import json
import shutil

import torch

from .recurrent import Actor, Critic, file_hash


def migrate_legacy(saved):
    with torch.random.fork_rng():
        torch.manual_seed(609)
        actor, critic = Actor(True), Critic(True)
        actor.warm_start(saved["model"])
        with torch.no_grad():
            for parameter in critic.parameters():
                parameter.zero_()
            critic.layers[0].weight[64:].normal_(std=0.001)
            critic.layers[0].bias[64:].fill_(0.01)
            critic.layers[0].weight[:64, -64:] = torch.eye(64)
            critic.layers[2].weight[:64, :64] = torch.eye(64)
            critic.layers[4].weight.copy_(saved["model"]["value.weight"])
            critic.layers[4].bias.copy_(saved["model"]["value.bias"])
    if sum(parameter.numel() for parameter in actor.parameters()) != 91463:
        raise ValueError("actor footprint changed")
    return {
        "model": actor.state_dict(),
        "critic": critic.state_dict(),
        "encoder": 2,
        "memory_version": 1,
        "configuration": {"arm": "D"},
        "approved": False,
        "migration": "zero-residual-lstm16; exact legacy actor and initial value preservation",
    }


def prepare_inputs(destination, original_manifest, legacy_checkpoint, controls_root):
    if destination.exists():
        raise ValueError("immutable improvement inputs already exist")
    shutil.copytree(original_manifest.parent, destination)
    metadata = json.loads((destination / "manifest.json").read_text())
    metadata["kind"] = "improvement-inputs-v1"
    metadata["candidates"] = {"d-parent": copy.deepcopy(metadata["parents"]["101"])}
    original = destination / "legacy-champion-original.pt"
    shutil.copy2(legacy_checkpoint, original)
    migrated = destination / "legacy-champion-lstm16.pt"
    torch.save(
        migrate_legacy(torch.load(original, map_location="cpu", weights_only=True)),
        migrated,
    )
    metadata["legacy_origin"] = {
        "origin": str(legacy_checkpoint.resolve()),
        "file": original.name,
        "sha256": file_hash(original),
    }
    metadata["candidates"]["legacy-champion"] = {
        "file": migrated.name,
        "sha256": file_hash(migrated),
        "origin_sha256": file_hash(original),
    }
    for seed in (101, 102, 103):
        source = controls_root / f"C-seed-{seed}/checkpoint.pt"
        target = destination / f"credit-control-{seed}.pt"
        shutil.copy2(source, target)
        metadata["candidates"][f"C-{seed}"] = {
            "file": target.name,
            "sha256": file_hash(target),
            "origin": str(source.resolve()),
        }
    (destination / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def selected_start(actor, critic, metadata, root, name):
    entry = metadata["candidates"][name]
    path = (root / entry["file"]).resolve()
    if not path.is_relative_to(root.resolve()) or file_hash(path) != entry["sha256"]:
        raise ValueError("immutable selected starter changed")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    from .model_capacity import actor_from_checkpoint

    source_actor = actor_from_checkpoint(saved)
    actor.load_state_dict(source_actor.state_dict())
    if "critic" in saved:
        critic.load_state_dict(saved["critic"])
    return entry["sha256"]
