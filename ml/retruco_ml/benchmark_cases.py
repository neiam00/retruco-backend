"""Authorized numeric inputs for browser timing, plus labeled capacity cases."""

import argparse
import copy
import json
from pathlib import Path

from .bridge import Simulator
from .model import ACTION_ENCODER_VERSION, ENCODER_VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    selected = {}
    with Simulator() as simulator:
        for players in (2, 4):
            observation = simulator.call(
                [
                    {
                        "op": "reset",
                        "id": 0,
                        "seed": 81813,
                        "roles": [True] * 4,
                        "config": {
                            "players": players,
                            "target": 24,
                            "florRequired": True,
                        },
                    }
                ]
            )[0]
            for _ in range(10000):
                if observation["done"]:
                    break
                if observation["version"] != ENCODER_VERSION:
                    raise ValueError("rebuild simulator")
                signal = observation["kind"] == "signal"
                if signal or observation["legal"][0]["type"] != "cut":
                    key = f"{players}-player-" + ("signal" if signal else "game")
                    item = {
                        "name": key,
                        "features": observation["features"],
                        "actions": observation["actions"],
                        "signal": signal,
                        "synthetic": False,
                        "view": observation["view"],
                        "seat": observation["seat"],
                        "history": {"signals": len(observation.get("signals", []))},
                    }
                    if key not in selected or len(item["actions"]) >= len(
                        selected[key]["actions"]
                    ):
                        selected[key] = item
                action = 1 if signal else observation["baseline"]
                observation = simulator.call(
                    [{"op": "step", "id": 0, "action": action}]
                )[0]
        base = max(
            (item for item in selected.values() if not item["signal"]),
            key=lambda item: len(item["actions"]),
        )
        # Synthetic capacity probes are labelled explicitly. Numeric features
        # still come from Go; this script does not duplicate the encoder layout.
        context = {
            "cpu": [True] * 4,
            "complete": True,
            "signalsComplete": True,
            "cantos": [
                {
                    "hand": base["view"]["handNumber"],
                    "event": {
                        "type": "call",
                        "seat": i % base["view"]["config"]["players"],
                        "kind": ("truco", "envido", "flor", "privo")[i % 4],
                        "amount": i % 9 + 1,
                    },
                }
                for i in range(64)
            ],
            "signals": [
                {
                    "seat": i % base["view"]["config"]["players"],
                    "kind": (
                        "perico",
                        "perica",
                        "seven_spades",
                        "seven_coins",
                        "receta",
                    )[i % 5],
                }
                for i in range(20)
            ],
        }
        context["signalCounts"] = [[0] * 5 for _ in range(4)]
        for i in range(20):
            context["signalCounts"][i % base["view"]["config"]["players"]][i % 5] += 1
        for key, count in (("history", len(base["actions"])), ("capacity", 1024)):
            view = copy.deepcopy(base["view"])
            view["legal"] = [
                view["legal"][i % len(view["legal"])] for i in range(count)
            ]
            encoded = simulator.call(
                [
                    {
                        "op": "encode",
                        "id": 0,
                        "view": view,
                        "seat": base["seat"],
                        "kind": "game",
                        "context": context,
                    }
                ]
            )[0]
            selected[key] = {
                **base,
                "name": "synthetic-full-history-" + str(count) + "-candidates",
                "synthetic": True,
                "features": encoded["features"],
                "actions": encoded["actions"],
                "view": view,
                "history": {"signals": 20, "cantos": 64, "encoder": "shared-go"},
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "format": 2,
                "encoder": ENCODER_VERSION,
                "action_encoder": ACTION_ENCODER_VERSION,
                "cases": list(selected.values()),
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
