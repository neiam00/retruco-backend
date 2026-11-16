"""Calibrate public opponent bidding-history inputs on an initialization-only actor."""

import argparse
import hashlib
import math
from pathlib import Path

import torch


def calibrate(source, destination, gain):
    if not math.isfinite(gain) or gain < 1:
        raise ValueError("Expected a finite history gain of at least one")
    data = torch.load(source, weights_only=True, map_location="cpu")
    if data.get("kind") != "rmappo-v1" or data.get("initialization_only") is not True:
        raise ValueError("Expected an initialization-only recurrent actor checkpoint")
    w = data["actor"]["state.0.weight"]
    if w.shape[1] != 768:
        raise ValueError("Unexpected actor input layout")
    # Relative seats 1 and 3 are opposing seats. In heads-up play seat 3 is padding.
    cols = [
        i
        for seat in (1, 3)
        for i in (
            *range(352 + 8 * seat, 360 + 8 * seat),
            *range(512 + 40 * seat, 552 + 40 * seat),
        )
    ]
    maximum = w.abs().amax(1, keepdim=True)
    w[:, cols] = torch.clamp(w[:, cols] * gain, min=-maximum, max=maximum)
    if not torch.equal(maximum, w.abs().amax(1, keepdim=True)):
        raise ValueError("History calibration changed the INT8 scale")
    data["calibration"] = {
        **data.get("calibration", {}),
        "opponent_history_gain": gain,
        "history_parent_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    data["approved"] = False
    # Existing resumable and release checkpoints must never be overwritten.
    with destination.open("xb") as stream:
        torch.save(data, stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--gain", type=float, required=True)
    args = parser.parse_args()
    calibrate(args.source, args.destination, args.gain)
