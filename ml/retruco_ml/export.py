"""Export a candidate, explicitly unapproved until release evaluation passes."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch

from .model import ACTION_ENCODER_VERSION, DEPLOY_LAYERS, ENCODER_VERSION, Policy


def serialize(model):
    data = bytearray(b"RTML" + struct.pack("<II", 1, ENCODER_VERSION))
    reference = bytearray()
    for name in DEPLOY_LAYERS:
        layer = model.get_submodule(name)
        weights = layer.weight.detach().cpu().numpy().astype("<f4")
        bias = layer.bias.detach().cpu().numpy().astype("<f4")
        if not np.isfinite(weights).all() or not np.isfinite(bias).all():
            raise ValueError("cannot export nonfinite parameters")
        scales = np.maximum(np.max(np.abs(weights), axis=1) / 127, 1e-12).astype("<f4")
        quantized = np.clip(np.rint(weights / scales[:, None]), -127, 127).astype(
            np.int8
        )
        data += (
            struct.pack("<II", *weights.shape)
            + scales.tobytes()
            + bias.tobytes()
            + quantized.tobytes()
        )
        reference += weights.tobytes() + bias.tobytes()
    if len(data) > 128 * 1024:
        raise ValueError("model exceeds release size budget")
    return bytes(data), bytes(reference)


def export(model, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    data, reference = serialize(model)
    (destination / "policy.bin").write_bytes(data)
    (destination / "reference.f32").write_bytes(reference)
    manifest = {
        "format": 1,
        "encoder": ENCODER_VERSION,
        "action_encoder": ACTION_ENCODER_VERSION,
        "rules": "2",
        "approved": False,
        "architecture": "candidate-768-96-64-48-32-32-1-signal6",
        "quantization": "symmetric-int8-per-output-f32-execution",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("encoder") != ENCODER_VERSION:
        raise ValueError("checkpoint encoder is incompatible")
    model = Policy()
    model.load_state_dict(checkpoint["model"])
    print(json.dumps(export(model, args.destination)))


if __name__ == "__main__":
    main()
