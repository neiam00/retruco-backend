"""Conditional bluff-value diagnostic over a single well-defined information set.

Uniform hidden deals are valid here because the opening has no prior actions or
signals. This deliberately does not claim a posterior for arbitrary mid-hand play.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from .artifacts import digest, read_quantized
from .assessment import choose, load_checkpoint
from .bridge import Simulator
from .export import serialize
from .population import STYLES, scripted
from .splits import BLUFF_START
from .statistics import interval


def diagnose(model, pairs, seed, target=12, opponent="baseline"):
    returns = []
    points = []
    initial = None
    bluff_probability = None
    with Simulator() as simulator:
        for pair in range(pairs):
            wins = []
            hand_points = []
            for branch in ("raise", "quiet"):
                observation = simulator.call(
                    [
                        {
                            "op": "reset",
                            "id": 0,
                            "seed": seed + pair,
                            "fixture": "weak-opening-v1",
                            "config": {
                                "players": 2,
                                "target": target,
                                "florRequired": False,
                            },
                        }
                    ]
                )[0]
                information = (observation["features"], observation["actions"])
                if initial is None:
                    initial = information
                if information != initial:
                    raise RuntimeError("hidden deal altered the acting information set")
                with torch.no_grad():
                    distribution, _ = model.distribution(observation)
                legal = observation["legal"]
                calls = [
                    i
                    for i, c in enumerate(legal)
                    if c["type"] == "call"
                    and c.get("kind") == "truco"
                    and c.get("amount") == 3
                ]
                raised = next(
                    i
                    for i in calls
                    if not legal[i].get("declaration") and not legal[i].get("mode")
                )
                bluff_probability = float(distribution.probs[raised])
                if branch == "raise":
                    action = raised
                else:
                    plays = [
                        i
                        for i, c in enumerate(legal)
                        if c["type"] == "play" and not c.get("declaration")
                    ]
                    action = max(plays, key=lambda i: float(distribution.probs[i]))
                streams = [random.Random((seed + pair) * 11 + seat) for seat in (0, 1)]
                first_points = None
                observation = simulator.call(
                    [{"op": "step", "id": 0, "action": action}]
                )[0]
                for _ in range(10000):
                    if first_points is None and (
                        observation["done"] or observation["view"]["handNumber"] > 1
                    ):
                        first_points = (
                            observation["scores"][0] - observation["scores"][1]
                        )
                    if observation["done"]:
                        break
                    seat = observation["seat"]
                    action = (
                        choose(model, observation, streams[seat])
                        if seat == 0
                        else scripted(observation, opponent, streams[seat])
                    )
                    observation = simulator.call(
                        [{"op": "step", "id": 0, "action": action}]
                    )[0]
                if not observation["done"]:
                    raise RuntimeError("diagnostic match stalled")
                wins.append(int(observation["winner"] == 0))
                hand_points.append(first_points)
            returns.append(wins[0] - wins[1])
            points.append(hand_points[0] - hand_points[1])
    return {
        "approved": False,
        "fixture": "weak-opening-v1",
        "pairs": pairs,
        "seed": seed,
        "target": target,
        "opponent": opponent,
        "checkpoint_sha256": model.checkpoint_sha256,
        "policy_raise_probability": bluff_probability,
        "win_probability_gain": sum(returns) / pairs,
        "win_gain_ci95": interval(returns),
        "first_hand_point_gain": sum(points) / pairs,
        "point_gain_ci95": interval(points),
        "paired_win_gains": returns,
        "paired_point_gains": points,
        "interpretation": "Conditional on the fixed weak opening, named opponent and this continuation policy; not universal bluff accuracy.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--quantized", type=Path)
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=BLUFF_START)
    parser.add_argument("--target", type=int, choices=(12, 24), default=12)
    parser.add_argument("--opponent", choices=STYLES, default="baseline")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 2:
        parser.error("at least two hidden-deal pairs required")
    torch.set_num_threads(2)
    model = load_checkpoint(args.checkpoint)
    if args.quantized:
        data = args.quantized.read_bytes()
        if data != serialize(model)[0]:
            raise ValueError("quantized model is not the exact checkpoint export")
        model = read_quantized(data, model)
    report = diagnose(model, args.pairs, args.seed, args.target, args.opponent)
    report["quantized_sha256"] = digest(args.quantized) if args.quantized else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("paired_")}))


if __name__ == "__main__":
    main()
