"""Versioned experimental actor export; critic and parent data are never exported."""

import argparse
import copy
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np
import torch

from .model import DEPLOY_LAYERS
from .model_capacity import (
    MAX_ACTOR_PARAMETERS,
    MAX_MODEL_BYTES,
)
from .model_capacity import (
    actor_from_checkpoint as capacity_actor,
)
from .recurrent import ARMS, CHECKPOINT_KIND


def layers(actor):
    if getattr(actor, "team_adapter", None) is not None:
        raise ValueError(
            "team-adapter export requires a new validated browser runtime; legacy export would omit learned layers"
        )
    result = [
        (actor.get_submodule(n).weight, actor.get_submodule(n).bias)
        for n in DEPLOY_LAYERS
    ]
    if actor.recurrent:
        m = actor.memory
        if hasattr(m, "core"):
            if getattr(m, "cell", None) != "lstm":
                raise ValueError("browser export supports LSTM, not GRU")
            m = m.core
            weights = torch.cat((m.weight_ih_l0, m.weight_hh_l0), 1)
            bias = m.bias_ih_l0 + m.bias_hh_l0
        else:
            weights = torch.cat((m.weight_ih, m.weight_hh), 1)
            bias = m.bias_ih + m.bias_hh
        result.extend(
            [(weights, bias), (actor.projection.weight, actor.projection.bias)]
        )
    return result


def export(actor, destination):
    actor_layers = layers(actor)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    size = actor.memory.hidden_size if actor.recurrent else 0
    rmappo = hasattr(actor.memory, "core") if actor.recurrent else False
    candidates = getattr(actor, "candidate_features", 0)
    if candidates not in (0, 1) or (candidates and not rmappo):
        raise ValueError("unsupported public candidate architecture")
    version = (
        5
        if candidates
        else 4
        if rmappo
        else 3
        if size in (32, 64)
        else 2
        if actor.recurrent
        else 1
    )
    parameters = sum(p.numel() for p in actor.parameters())
    if parameters > MAX_ACTOR_PARAMETERS:
        raise ValueError("actor exceeds parameter budget")
    data = bytearray(b"RTML" + struct.pack("<II", version, 2))
    reference = bytearray()
    if version in (3, 4, 5):
        data += struct.pack("<I", size)
    for weight, bias in actor_layers:
        w = weight.detach().cpu().numpy().astype("<f4")
        b = bias.detach().cpu().numpy().astype("<f4")
        if not np.isfinite(w).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite actor")
        scale = np.maximum(abs(w).max(1) / 127, 1e-12).astype("<f4")
        q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
        data += (
            struct.pack("<II", *w.shape) + scale.tobytes() + b.tobytes() + q.tobytes()
        )
        reference += w.tobytes() + b.tobytes()
    if len(data) > MAX_MODEL_BYTES:
        raise ValueError("actor exceeds size budget")
    manifest = {
        "format": version,
        "encoder": 2,
        "action_encoder": 1,
        "rules": "2",
        "approved": False,
        "architecture": "candidate-768-96-64-48-32-32-1-signal6"
        + (f"-lstm{size}-residual" if actor.recurrent else ""),
        "memory_size": size,
        "memory_version": 1 if actor.recurrent else 0,
        "memory_reset": "hand" if actor.recurrent else "none",
        "quantization": "symmetric-int8-per-output-f32-execution",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "parameters": sum(p.numel() for p in actor.parameters()),
    }
    if rmappo:
        manifest.update(
            cell="lstm",
            inference_math="centered-f64-heads-f32-state",
            training_kind="rmappo-v1",
        )
    if candidates:
        manifest.update(
            candidate_features=1,
            architecture=manifest["architecture"] + "-public-candidates-v1",
        )
    (destination / "policy.bin").write_bytes(data)
    (destination / "reference.f32").write_bytes(reference)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def quantized(actor):
    result = copy.deepcopy(actor)
    with torch.no_grad():
        for w, _ in layers(result)[:6]:
            s = w.abs().amax(1).clamp(min=1.27e-10) / 127
            w.copy_((w / s[:, None]).round().clamp(-127, 127) * s[:, None])
        if result.recurrent:
            m = result.memory
            if hasattr(m, "core"):
                if m.cell != "lstm":
                    raise ValueError("browser quantization supports LSTM, not GRU")
                ih, hh = m.core.weight_ih_l0, m.core.weight_hh_l0
            else:
                ih, hh = m.weight_ih, m.weight_hh
            w = torch.cat((ih, hh), 1)
            s = w.abs().amax(1).clamp(min=1.27e-10) / 127
            w = (w / s[:, None]).round().clamp(-127, 127) * s[:, None]
            ih.copy_(w[:, :64])
            hh.copy_(w[:, 64:])
            w = result.projection.weight
            s = w.abs().amax(1).clamp(min=1.27e-10) / 127
            w.copy_((w / s[:, None]).round().clamp(-127, 127) * s[:, None])
    return result


def load_actor(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    return actor_from_checkpoint(saved)


def actor_from_checkpoint(saved):
    if saved.get("kind") == "rmappo-v1":
        if (
            saved.get("memory_version") != 1
            or saved.get("encoder") != 2
            or saved.get("configuration", {}).get("cell") != "lstm"
        ):
            raise ValueError("unsupported RMAPPO browser checkpoint")
        from .rmappo_model import actor_from_checkpoint as rmappo_actor

        return rmappo_actor(saved).eval()
    if (
        saved.get("training_kind")
        not in (CHECKPOINT_KIND, "cooperation-mappo-v1", "improvement-mappo-v1")
        or saved.get("memory_version") != 1
        or saved.get("encoder") != 2
    ):
        raise ValueError("unsupported controlled actor checkpoint")
    actor = capacity_actor(saved)
    if actor.recurrent != ARMS[saved["configuration"]["arm"]][0]:
        raise ValueError("actor arm differs from weights")
    actor.eval()
    return actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--wasm", type=Path, help="bind the exact evaluator for browser evaluation"
    )
    args = parser.parse_args()
    payload = args.checkpoint.read_bytes()
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    manifest = export(actor_from_checkpoint(saved), args.destination)
    manifest.update(
        checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
        training_contracts=saved.get("contracts", {}),
        parent_sha256=saved.get(
            "parent_sha256",
            saved.get("configuration", {}).get(
                "parent_sha256", saved.get("contracts", {}).get("parent_sha256")
            ),
        ),
    )
    if saved.get("signal_policy"):
        manifest["signal_policy"] = saved["signal_policy"]
    if args.wasm:
        import shutil

        shutil.copy2(args.wasm, args.destination / "policy.wasm")
        manifest["wasm_sha256"] = hashlib.sha256(args.wasm.read_bytes()).hexdigest()
    (args.destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
