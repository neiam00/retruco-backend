"""Identical-record comparison of scalar and packed recurrent PPO updates."""

import argparse
import copy
import json
import random
import statistics
import time
from pathlib import Path

import torch

from .cooperation_rollout import collect
from .cooperation_train import load_inputs
from .mappo_train import update as scalar
from .recurrent_batch import update as batched
from .rollout import SimulatorPool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(700)
    actor, critic, reference, population, frozen, _ = load_inputs(args.inputs, 101)
    start = time.monotonic()
    with SimulatorPool(4, "binary", args.simulator) as pool:
        records, matches, events, behavior = collect(
            actor,
            critic,
            reference,
            pool,
            population,
            random.Random(700),
            args.steps,
            environments=16,
            frozen=frozen,
        )
    collection = time.monotonic() - start
    trials = []
    for trial in range(args.trials):
        row = {}
        for name, update in (
            [("scalar", scalar), ("batched", batched)]
            if trial % 2 == 0
            else [("batched", batched), ("scalar", scalar)]
        ):
            a, c = copy.deepcopy(actor), copy.deepcopy(critic)
            optimizer = torch.optim.Adam([*a.parameters(), *c.parameters()], lr=1e-4)
            begin = time.monotonic()
            loss = update(a, c, optimizer, records, random.Random(701), epochs=4)
            row[name] = {"seconds": time.monotonic() - begin, "loss": loss}
        trials.append(row)
    old = statistics.median(x["scalar"]["seconds"] for x in trials)
    new = statistics.median(x["batched"]["seconds"] for x in trials)
    report = {
        "records": len(records),
        "matches": matches,
        "events": events,
        "collection_seconds": collection,
        "trials": trials,
        "update_speedup": old / new,
        "end_to_end_estimated_speedup_same_collection": (collection + old)
        / (collection + new),
        "behavior": behavior,
        "scope": "same real cooperation rollout, identical PPO objective and minibatch order; floating-point reductions differ",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
