"""Paired validation of experimental actors; never consumes final holdout seeds."""

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
import torch

from .bridge import Simulator
from .mappo_export import actor_from_checkpoint
from .model import Policy
from .population import scripted
from .recurrent import Replay
from .statistics import interval, pooled


def evaluate(actor, reference, simulator, pairs=128, seed=180000, partner="reference"):
    if pairs < 2 or not 100000 <= seed or seed + pairs >= 200000:
        raise ValueError("controlled validation must stay within [100000,200000)")
    configurations = []
    for players in (2, 4):
        for target in (12, 24):
            for flor in (False, True):
                scores = []
                for deal in range(pairs):
                    wins = []
                    # Same seeded deal and dealer; evaluate every learner seat.
                    for learner in range(players):
                        torch.manual_seed(seed + deal * 17 + learner)
                        rng = random.Random(seed + deal * 17 + learner)
                        roles = [True] * 4
                        if players == 4 and partner != "reference":
                            roles[(learner + 2) % 4] = False
                        obs = simulator.call(
                            [
                                {
                                    "op": "reset",
                                    "training": True,
                                    "id": 0,
                                    "seed": seed + deal,
                                    "config": {
                                        "players": players,
                                        "target": target,
                                        "florRequired": flor,
                                    },
                                    "roles": roles,
                                }
                            ]
                        )[0]
                        replays = {s: Replay(actor) for s in range(players)}
                        cursors = [0] * 4
                        hand = 0
                        for _ in range(10000):
                            if obs["done"]:
                                wins.append(float(obs["winner"] == learner % 2))
                                break
                            page = obs["training"]["memory"]
                            seat = obs["seat"]
                            if page["hand"] != hand:
                                hand = page["hand"]
                                cursors = [0] * 4
                            while True:
                                cursors[seat] = page["end"]
                                if seat == learner:
                                    replays[seat].apply(page)
                                if page["end"] == page["total"]:
                                    break
                                obs = simulator.call(
                                    [
                                        {
                                            "op": "observe",
                                            "id": 0,
                                            "memoryHand": hand,
                                            "memoryCursors": cursors,
                                        }
                                    ]
                                )[0]
                                page = obs["training"]["memory"]
                            if (
                                obs["kind"] == "game"
                                and obs["legal"][0]["type"] == "cut"
                            ):
                                action = 0
                            elif seat == learner:
                                with torch.no_grad():
                                    context = actor.context(
                                        torch.as_tensor(
                                            np.asarray(obs["features"], np.float32)
                                        ),
                                        replays[seat].memory,
                                    )
                                    action = int(
                                        actor.distribution(context, obs).sample()
                                    )
                            elif players == 4 and seat == (learner + 2) % 4:
                                if partner == "reference":
                                    with torch.no_grad():
                                        action = int(
                                            reference.distribution(obs)[0].sample()
                                        )
                                else:
                                    action = scripted(obs, partner, rng)
                            else:
                                action = scripted(obs, "baseline", rng)
                            obs = simulator.call(
                                [
                                    {
                                        "op": "step",
                                        "id": 0,
                                        "action": action,
                                        "memoryHand": hand,
                                        "memoryCursors": cursors,
                                    }
                                ]
                            )[0]
                        else:
                            raise RuntimeError("evaluation match exceeded budget")
                    scores.append(sum(wins) / len(wins))
                configurations.append(
                    {
                        "players": players,
                        "target": target,
                        "flor": flor,
                        "pairs": pairs,
                        "pair_scores": scores,
                        "win_rate": sum(scores) / pairs,
                        "cluster_ci95": interval(scores),
                    }
                )
    return {
        "seed": seed,
        "configurations": configurations,
        "pooled": pooled(configurations),
        "partner": partner,
        "approved": False,
        "scope": "paired validation against legacy heuristic; not release evidence",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=180000)
    parser.add_argument(
        "--partner",
        choices=(
            "reference",
            "literal",
            "silent",
            "deceptive",
            "aggressive",
            "cautious",
            "ignore",
        ),
        default="reference",
    )
    parser.add_argument("--simulator", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    payload = args.checkpoint.read_bytes()
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    reference = Policy()
    reference.load_state_dict(saved["parent"]["model"])
    reference.eval()
    with Simulator(args.simulator, transport="binary") as simulator:
        report = evaluate(
            actor_from_checkpoint(saved),
            reference,
            simulator,
            args.pairs,
            args.seed,
            args.partner,
        )
    report["checkpoint_sha256"] = hashlib.sha256(payload).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["pooled"]))


if __name__ == "__main__":
    main()
