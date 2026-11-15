"""Offline score-head calibration; original checkpoints remain immutable.

The scale is baked into stored weights, so ordinary inference/evaluation uses
the calibrated policy without an alternate decoder or actor-input feature.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from .model_capacity import (
    actor_from_checkpoint,
    critic_from_checkpoint,
    widen_actor,
    widen_critic,
)
from .recurrent import file_hash


def calibrated_weights(model, scale):
    if (
        type(scale) not in (int, float)
        or not math.isfinite(scale)
        or not 0.125 <= scale <= 8
    ):
        raise ValueError("game score scale must be finite and within [.125,8]")
    result = copy.deepcopy(model)
    with torch.no_grad():
        result.score[2].weight.mul_(scale)
        result.score[2].bias.mul_(scale)
    return result.eval()


def prepare(source, output, scales=(0.5, 1.0, 2.0, 4.0)):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("calibration output must be new")
    if len(scales) != len(set(scales)) or 1 not in scales:
        raise ValueError("unique scales including the unmodified control required")
    digest = file_hash(source)
    saved = torch.load(source, map_location="cpu", weights_only=True)
    if saved.get("encoder") != 2 or saved.get("memory_version") != 1:
        raise ValueError("calibration requires compatible recurrent checkpoint")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1701)
        actor = widen_actor(actor_from_checkpoint(saved), 64)
        critic = widen_critic(critic_from_checkpoint(saved), (256, 128))
    candidates = [(scale, calibrated_weights(actor, scale)) for scale in scales]
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for scale, model in candidates:
        label = "game-scale-" + str(scale).replace(".", "p")
        path = output / (label + ".pt")
        payload = {
            "model": model.state_dict(),
            "critic": critic.state_dict(),
            "encoder": 2,
            "memory_version": 1,
            "signal_policy": "truthful-cpu-v1",
            "approved": False,
            "configuration": {"actor_memory": 64, "critic_hidden": [256, 128]},
            "policy_calibration": {
                "version": 1,
                "game_score_scale": scale,
                "signal_score_scale": 1.0,
                "source_sha256": digest,
                "source": str(source),
                "optimizer_updates": 0,
                "scope": "new offline candidate, not evidence of strength or release approval",
            },
        }
        torch.save(payload, path)
        rows.append(
            {
                "label": label,
                "scale": scale,
                "checkpoint": path.name,
                "sha256": file_hash(path),
            }
        )
    if file_hash(source) != digest:
        raise ValueError("source checkpoint changed during preparation")
    manifest = {
        "version": 1,
        "source": str(source),
        "source_sha256": digest,
        "candidates": rows,
        "approved": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.scales)))


if __name__ == "__main__":
    main()
