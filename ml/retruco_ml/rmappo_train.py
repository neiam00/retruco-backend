"""Fresh joint RMAPPO training. Prior checkpoints are opponents, never initialization."""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

from .controlled_train import atomic_json
from .improvement_budget import DeadlineMonitor, DeadlineReached
from .improvement_evaluate import frozen_models
from .improvement_population import improvement_scripted
from .learning_choices import monte_carlo_credit
from .recurrent import file_hash
from .resources import MemoryBudget
from .rmappo_model import (
    RecurrentActor as Actor,
)
from .rmappo_model import (
    RecurrentCritic as Critic,
)
from .rmappo_model import (
    actor_from_checkpoint,
    critic_from_checkpoint,
)
from .rmappo_population import Population
from .rmappo_ppo import update
from .rmappo_rollout import collect
from .rollout import SimulatorPool
from .team_play_probe import CHOICE_PROFILES, LIVE_CHOICES


def save(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--simulator", type=Path, required=True)
    p.add_argument("--cell", choices=("gru", "lstm"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--choices", type=int, required=True)
    p.add_argument("--steps", type=int, default=8192)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--environments", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--warmup-choices", type=int, default=2048)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-updates", type=int, default=1000000)
    p.add_argument(
        "--choice-policy",
        choices=(*CHOICE_PROFILES, "unfiltered"),
        default=None,
        help="defaults to live CPU choices on new runs and the saved contract on resume",
    )
    a = p.parse_args()
    root = a.output.resolve()
    if root.exists() != a.resume:
        p.error("fresh output must not exist; resume requires an existing run")
    if (
        min(
            a.choices,
            a.steps,
            a.environments,
            a.workers,
            a.batch_size,
            a.warmup_choices,
        )
        < 1
    ):
        p.error("invalid workload")
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    optimizer_rng = random.Random(a.seed + 70000)
    manifest = json.loads(a.inputs.read_text())
    models = frozen_models(manifest, a.inputs.resolve().parent)
    reference = models.pop("reference")
    models.pop("frozen:103")
    actor = Actor(a.cell)
    critic = Critic(a.cell)
    ao = torch.optim.Adam(actor.parameters(), lr=3e-4, eps=1e-5)
    co = torch.optim.Adam(critic.parameters(), lr=3e-4, eps=1e-5)
    pop = Population(models)
    pop.add(actor)
    config = {
        "cell": a.cell,
        "seed": a.seed,
        "initialization": "random; no warm start",
        "actor_memory": 64,
        "critic_memory": 128,
        "critic_encoder": [256, 128],
        "steps": a.steps,
        "environments": a.environments,
        "workers": a.workers,
        "batch_size": a.batch_size,
        "warmups": a.warmups,
        "warmup_choices": a.warmup_choices,
        "inputs_sha256": file_hash(a.inputs),
        "simulator_sha256": file_hash(a.simulator),
        "gae_lambda": 0.99,
        "headsup_fraction": 0.4,
        "reward": "terminal team match +/-1",
        "actor_parameters": sum(x.numel() for x in actor.parameters()),
        "critic_parameters": sum(x.numel() for x in critic.parameters()),
    }
    choices = updates = warmups = 0
    seconds = 0.0
    last_snapshot = 0
    if a.resume:
        saved = torch.load(
            root / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        stored_policy = saved["configuration"].get("choice_policy")
        requested_policy = None if a.choice_policy == "unfiltered" else a.choice_policy
        if a.choice_policy is not None and requested_policy != stored_policy:
            raise ValueError("resume choice policy changed")
        if stored_policy:
            config["choice_policy"] = stored_policy
        if saved["configuration"] != config:
            raise ValueError("resume configuration changed")
        actor.load_state_dict(actor_from_checkpoint(saved).state_dict())
        critic.load_state_dict(critic_from_checkpoint(saved).state_dict())
        ao.load_state_dict(saved["actor_optimizer"])
        co.load_state_dict(saved["critic_optimizer"])
        choices, updates, warmups, seconds = (
            saved["choices"],
            saved["updates"],
            saved["warmups"],
            saved["training_seconds"],
        )
        last_snapshot = saved["last_snapshot"]
        pop.league = [
            actor_from_checkpoint(s).eval().requires_grad_(False)
            for s in saved["league"]
        ]
        rng.setstate(saved["python_rng"])
        optimizer_rng.setstate(saved["optimizer_rng"])
        torch.set_rng_state(saved["torch_rng"])
    else:
        if a.choice_policy != "unfiltered":
            config["choice_policy"] = a.choice_policy or LIVE_CHOICES
        root.mkdir(parents=True)
        (root / "snapshots").mkdir()
        atomic_json(root / "configuration.json", config)

    def payload():
        return {
            "kind": "rmappo-v1",
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "configuration": config,
            "actor_optimizer": ao.state_dict(),
            "critic_optimizer": co.state_dict(),
            "choices": choices,
            "updates": updates,
            "warmups": warmups,
            "training_seconds": seconds,
            "last_snapshot": last_snapshot,
            "python_rng": rng.getstate(),
            "optimizer_rng": optimizer_rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "league": [
                {
                    "kind": "rmappo-v1",
                    "configuration": {"cell": a.cell},
                    "actor": x.state_dict(),
                }
                for x in pop.league
            ],
            "encoder": 2,
            "memory_version": 1,
            "approved": False,
        }

    def status(state, **extra):
        atomic_json(
            root / "status.json",
            {
                "status": state,
                "pid": os.getpid(),
                "cell": a.cell,
                "seed": a.seed,
                "update": updates,
                "total_choices": choices,
                "target_choices": a.choices,
                "warmups": warmups,
                "training_seconds": seconds,
                "checkpoint": str(root / "checkpoint.pt"),
                **extra,
            },
        )

    if not a.resume:
        save(root / "checkpoint.pt", payload())
        save(root / "initial.pt", payload())
    try:
        with SimulatorPool(a.workers, "binary", a.simulator) as pool:
            monitor = DeadlineMonitor(
                MemoryBudget([s.process for s in pool.simulators], 8)
            )
            while (
                warmups < a.warmups or choices < a.choices
            ) and updates < a.max_updates:
                warm = warmups < a.warmups
                status("critic-warmup" if warm else "training")
                start = time.monotonic()
                records, matches, events, counts = collect(
                    actor,
                    critic,
                    reference,
                    pool,
                    pop,
                    rng,
                    a.warmup_choices if warm else min(a.steps, a.choices - choices),
                    environments=a.environments,
                    monitor=monitor,
                    script_fn=improvement_scripted,
                    headsup_fraction=0.4,
                    choice_policy=config.get("choice_policy"),
                )
                if warm:
                    # GAE(lambda=1) telescopes to each seat's terminal team reward.
                    groups = {}
                    for r in records:
                        groups.setdefault((r.match, r.learner_seat), []).append(r)
                    for rows in groups.values():
                        monte_carlo_credit(rows, rows[-1].target)
                metrics = update(
                    actor,
                    critic,
                    ao,
                    co,
                    records,
                    optimizer_rng,
                    batch_size=a.batch_size,
                    monitor=monitor,
                    critic_only=warm,
                )
                elapsed = time.monotonic() - start
                seconds += elapsed
                if warm:
                    warmups += 1
                else:
                    choices += len(records)
                    updates += 1
                if not warm and choices // 50000 > last_snapshot // 50000:
                    pop.add(actor)
                    last_snapshot = choices
                    save(
                        root / "snapshots" / f"choices-{choices:09d}.pt",
                        {
                            "kind": "rmappo-v1",
                            "configuration": config,
                            "actor": actor.state_dict(),
                            "encoder": 2,
                            "memory_version": 1,
                            "choices": choices,
                            "approved": False,
                        },
                    )
                save(root / "checkpoint.pt", payload())
                metrics.update(
                    stage="critic-warmup" if warm else "training",
                    update=updates,
                    warmups=warmups,
                    total_choices=choices,
                    choices_this_update=len(records),
                    choices_per_second=len(records) / elapsed,
                    elapsed_seconds=elapsed,
                    total_seconds=seconds,
                    matches=matches,
                    events=events,
                    league_snapshots=len(pop.league),
                    peak_rss=monitor.peak,
                    **counts,
                )
                with (root / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics), flush=True)
            status("chunk-complete", choices_per_second=choices / max(seconds, 1e-9))
    except DeadlineReached:
        status("budget-exhausted")
        raise SystemExit(75)
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
