"""Strict model binary decoding for independent quantization evaluation."""

import copy
import hashlib
import struct

import numpy as np
import torch

from .model import ACTION_ENCODER_VERSION, DEPLOY_LAYERS, ENCODER_VERSION, Policy


def read_quantized(data, reference=None):
    if len(data) > 128 * 1024 or len(data) < 12 or data[:4] != b"RTML":
        raise ValueError("invalid model header or size")
    version, encoder = struct.unpack_from("<II", data, 4)
    if version != 1 or encoder != ENCODER_VERSION:
        raise ValueError("unsupported model encoding")
    model = copy.deepcopy(reference) if reference is not None else Policy()
    offset = 12
    with torch.no_grad():
        for name in DEPLOY_LAYERS:
            layer = model.get_submodule(name)
            if offset + 8 > len(data):
                raise ValueError("truncated model")
            rows, cols = struct.unpack_from("<II", data, offset)
            offset += 8
            if (rows, cols) != tuple(layer.weight.shape):
                raise ValueError("unexpected layer dimensions")
            end = offset + rows * 8 + rows * cols
            if end > len(data):
                raise ValueError("truncated model")
            scales = np.frombuffer(data, dtype="<f4", count=rows, offset=offset).copy()
            bias = np.frombuffer(
                data, dtype="<f4", count=rows, offset=offset + rows * 4
            ).copy()
            quantized = np.frombuffer(
                data, dtype=np.int8, count=rows * cols, offset=offset + rows * 8
            ).reshape(rows, cols)
            if (
                not np.isfinite(scales).all()
                or not np.isfinite(bias).all()
                or (scales <= 0).any()
            ):
                raise ValueError("nonfinite model")
            weights = quantized.astype(np.float32) * scales[:, None]
            if not np.isfinite(weights).all():
                raise ValueError("overflowing model")
            layer.weight.copy_(torch.from_numpy(weights))
            layer.bias.copy_(torch.from_numpy(bias))
            offset = end
    if offset != len(data):
        raise ValueError("trailing model bytes")
    model.eval()
    return model


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(manifest, model_bytes, wasm_bytes):
    """Validate the release metadata contract as well as the payload identity."""
    expected = {
        "format": 1,
        "encoder": ENCODER_VERSION,
        "action_encoder": ACTION_ENCODER_VERSION,
        "rules": "2",
        "architecture": "candidate-768-96-64-48-32-32-1-signal6",
        "quantization": "symmetric-int8-per-output-f32-execution",
        "bytes": len(model_bytes),
        "wasm_bytes": len(wasm_bytes),
        "actor_parameters": 85127,
        "sha256": hashlib.sha256(model_bytes).hexdigest(),
        "wasm_sha256": hashlib.sha256(wasm_bytes).hexdigest(),
    }
    for key, value in expected.items():
        if type(manifest.get(key)) is not type(value) or manifest[key] != value:
            raise ValueError("incompatible or mismatched manifest field: " + key)
    if type(manifest.get("approved")) is not bool:
        raise ValueError("manifest approval must be a boolean")
    if len(model_bytes) > 128 * 1024:
        raise ValueError("model exceeds footprint budget")
