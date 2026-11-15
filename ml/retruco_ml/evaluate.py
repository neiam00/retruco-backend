"""Held-out paired-match baseline evaluation; does not approve a release.

One cluster is two matches using the same deal seed and opposite learned teams.
The confidence interval resamples clusters, not correlated individual games.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .bridge import Simulator
from .model import ENCODER_VERSION, Policy
from .splits import RELEASE_START


def evaluate(model, pairs, players, target, flor, seed):
    results = []
    with Simulator() as simulator:
        for pair in range(pairs):
            wins = 0
            for team in range(2):
                torch.manual_seed(seed + pair)
                observation = simulator.call(
                    [
                        {
                            "op": "reset",
                            "id": 0,
                            "seed": seed + pair,
                            "config": {
                                "players": players,
                                "target": target,
                                "florRequired": flor,
                            },
                            "roles": [True] * 4,
                        }
                    ]
                )[0]
                for _ in range(10000):
                    if observation["done"]:
                        break
                    if observation["seat"] % 2 == team:
                        with torch.no_grad():
                            distribution, _ = model.distribution(observation)
                            action = int(distribution.sample())
                    else:
                        action = observation["baseline"]
                    observation = simulator.call(
                        [{"op": "step", "id": 0, "action": action}]
                    )[0]
                if not observation["done"]:
                    raise RuntimeError("evaluation match stalled")
                wins += int(observation["winner"] == team)
            results.append(wins / 2)
    values = np.array(results)
    rng = np.random.default_rng(seed)
    bootstrap = [
        rng.choice(values, size=len(values), replace=True).mean() for _ in range(3000)
    ]
    return {
        "players": players,
        "target": target,
        "flor": flor,
        "pairs": pairs,
        "win_rate": float(values.mean()),
        "cluster_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "pair_scores": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--players", type=int, choices=(0, 2, 4), default=0)
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=RELEASE_START)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 2:
        parser.error("at least two independent deal pairs are required")
    torch.set_num_threads(2)
    model = Policy()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("encoder") != ENCODER_VERSION:
        raise ValueError("checkpoint encoder is incompatible")
    model.load_state_dict(saved["model"])
    model.eval()
    configurations = []
    for players in (args.players,) if args.players else (2, 4):
        for target in (12, 24):
            for flor in (False, True):
                result = evaluate(model, args.pairs, players, target, flor, args.seed)
                configurations.append(result)
                print(
                    json.dumps({k: v for k, v in result.items() if k != "pair_scores"}),
                    flush=True,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "approved": False,
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": hashlib.sha256(
                    args.checkpoint.read_bytes()
                ).hexdigest(),
                "seed": args.seed,
                "configurations": configurations,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
