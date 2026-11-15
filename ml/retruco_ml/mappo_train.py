"""Isolated four-arm training. Never upgrades or writes a legacy checkpoint."""

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .mappo_rollout import collect
from .model import ENCODER_VERSION, Policy
from .population import Population
from .recurrent import ARMS, CHECKPOINT_KIND, MEMORY_VERSION, Actor, Critic, file_hash
from .resources import MemoryBudget
from .rollout import SimulatorPool


def update(
    actor,
    critic,
    optimizer,
    records,
    rng,
    epochs=4,
    batch_size=512,
    sequence=32,
    monitor=None,
):
    advantages = torch.tensor([r.advantage for r in records])
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    chunks = {}
    for i, r in enumerate(records):
        chunks.setdefault((id(r.events), (r.end - 1) // sequence), []).append(i)
    chunks = list(chunks.values())
    last = 0.0
    for _ in range(epochs):
        rng.shuffle(chunks)
        for start in range(0, len(chunks), max(1, batch_size // sequence)):
            if monitor:
                monitor.check()
            losses = []
            for indices in chunks[start : start + max(1, batch_size // sequence)]:
                first = records[indices[0]]
                lower = ((first.end - 1) // sequence) * sequence
                upper = min(len(first.events), lower + sequence)
                memory = actor.initial()
                contexts = {}
                if actor.recurrent:
                    # Reconstruct with current weights, not stale rollout states.
                    with torch.no_grad():
                        for event in first.events[:lower]:
                            memory = actor.advance(torch.as_tensor(event), memory)
                    wanted = {records[i].end for i in indices}
                    for cursor in range(lower, upper):
                        features = torch.as_tensor(first.events[cursor])
                        memory = actor.advance(features, memory)
                        if cursor + 1 in wanted:
                            contexts[cursor + 1] = actor.context(features, memory)
                for i in indices:
                    r = records[i]
                    features = torch.as_tensor(
                        np.asarray(r.observation["features"], np.float32)
                    )
                    context = (
                        contexts[r.end]
                        if actor.recurrent
                        else actor.context(features, memory)
                    )
                    distribution = actor.distribution(context, r.observation)
                    ratio = (
                        distribution.log_prob(torch.tensor(r.action)) - r.logp
                    ).exp()
                    policy = -torch.minimum(
                        ratio * advantages[i], ratio.clamp(0.8, 1.2) * advantages[i]
                    )
                    value = critic(features, torch.as_tensor(r.privileged), context)
                    losses.append(
                        policy
                        + 0.5 * (value - r.target).square()
                        - 0.01 * distribution.entropy()
                    )
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite controlled PPO loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            optimizer.step()
            last = float(loss.detach())
    return last


def build(arm, parent, seed):
    recurrent, centralized = ARMS[arm]
    with torch.random.fork_rng():
        torch.manual_seed(seed + 200)
        actor = Actor(recurrent)
        actor.warm_start(parent["model"])
        torch.manual_seed(seed + 100)
        critic = Critic(centralized)
    return actor, critic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--decisions", type=int, default=1000000)
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--environments", type=int, default=16)
    parser.add_argument("--sequence", type=int, choices=(32, 64), default=32)
    parser.add_argument("--players", type=int, choices=(0, 2, 4), default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--memory-gib", type=float, default=8)
    parser.add_argument("--simulator", type=Path)
    parser.add_argument("--max-hours", type=float, default=24)
    args = parser.parse_args()
    if args.resume == bool(args.warm_start):
        parser.error("choose exactly one of --warm-start or --resume")
    if (
        min(args.decisions, args.steps, args.epochs) <= 0
        or not 1 <= args.workers <= 64
        or not 1 <= args.environments <= 64
    ):
        parser.error("invalid training capacity")
    if (
        not math.isfinite(args.max_hours)
        or args.max_hours <= 0
        or not math.isfinite(args.memory_gib)
        or args.memory_gib <= 0
    ):
        parser.error("positive finite resource limits required")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    source = Path(__file__).resolve().parent
    contracts = {
        "encoder": ENCODER_VERSION,
        "memory": MEMORY_VERSION,
        "reset": "hand",
        "simulator": file_hash(args.simulator) if args.simulator else None,
        "python": {
            name: file_hash(source / name)
            for name in (
                "recurrent.py",
                "mappo_train.py",
                "mappo_rollout.py",
                "model.py",
                "population.py",
                "bridge.py",
                "resources.py",
            )
        },
    }
    root = args.output.resolve()
    # No generic --output can accidentally target an existing legacy run.
    if root.exists() and not args.resume:
        parser.error("warm-start output must be a new directory")
    if args.resume and not root.is_dir():
        parser.error("resume directory missing")
    if args.warm_start and (
        root == args.warm_start.resolve().parent
        or root in args.warm_start.resolve().parents
    ):
        parser.error("output must not contain its parent checkpoint")
    root.mkdir(parents=True, exist_ok=args.resume)
    with (root / "train.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("controlled experiment already running")
        path = root / "checkpoint.pt"
        if args.resume:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved.get("training_kind") != CHECKPOINT_KIND:
                raise ValueError("not a controlled MAPPO checkpoint")
            if saved.get("evaluation_only"):
                raise ValueError("evaluation snapshot cannot resume training")
            if saved.get("contracts") != contracts:
                raise ValueError(
                    "resume source/simulator contract changed; use the frozen source"
                )
            configuration = saved["configuration"]
            for key in (
                "arm",
                "seed",
                "sequence",
                "players",
                "steps",
                "epochs",
                "workers",
                "environments",
            ):
                if configuration[key] != getattr(args, key):
                    raise ValueError("resume configuration differs: " + key)
            parent = saved["parent"]
            parent_hash = saved["parent_sha256"]
        else:
            parent_bytes = args.warm_start.read_bytes()
            parent_hash = hashlib.sha256(parent_bytes).hexdigest()
            parent = torch.load(
                io.BytesIO(parent_bytes), map_location="cpu", weights_only=True
            )
            if (
                parent.get("training_kind", "ppo") not in ("ppo", "qat")
                or parent.get("encoder") != ENCODER_VERSION
            ):
                raise ValueError("warm-start requires a legacy compatible actor")
            saved = None
        actor, critic = build(args.arm, parent, args.seed)
        if any(not torch.isfinite(p).all() for p in actor.parameters()):
            raise ValueError("nonfinite parent actor")
        reference = Policy()
        reference.load_state_dict(parent["model"])
        reference.eval()
        population = Population(parent.get("population"))
        optimizer = torch.optim.Adam(
            [*actor.parameters(), *critic.parameters()], lr=3e-4
        )
        configuration = {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        }
        total = completed = 0
        previous = 0.0
        if saved:
            actor.load_state_dict(saved["model"])
            critic.load_state_dict(saved["critic"])
            optimizer.load_state_dict(saved["optimizer"])
            rng.setstate(saved["random"])
            torch.set_rng_state(saved["torch_random"])
            total, completed, previous = (
                saved["decisions"],
                saved["update"],
                saved["total_seconds"],
            )
        started = time.monotonic()
        next_snapshot = (int(previous // 60) + 1) * 60

        def status(state, **extra):
            temporary = root / "status.tmp"
            temporary.write_text(
                json.dumps(
                    {
                        "status": state,
                        "arm": args.arm,
                        "pid": os.getpid(),
                        "update": completed,
                        "decisions": total,
                        **extra,
                    }
                )
                + "\n"
            )
            temporary.replace(root / "status.json")

        status("running")
        try:
            with SimulatorPool(
                args.workers, transport="binary", executable=args.simulator
            ) as pool:
                monitor = MemoryBudget(
                    [s.process for s in pool.simulators], args.memory_gib
                )
                monitor.check(force=True)
                while total < args.decisions:
                    begin = time.monotonic()
                    actor.eval()
                    critic.eval()
                    records, matches, events = collect(
                        actor,
                        critic,
                        reference,
                        pool,
                        population,
                        rng,
                        args.steps,
                        args.players,
                        args.environments,
                        monitor,
                    )
                    actor.train()
                    critic.train()
                    loss = update(
                        actor,
                        critic,
                        optimizer,
                        records,
                        rng,
                        epochs=args.epochs,
                        sequence=args.sequence,
                        monitor=monitor,
                    )
                    total += len(records)
                    completed += 1
                    data = {
                        "training_kind": CHECKPOINT_KIND,
                        "encoder": ENCODER_VERSION,
                        "memory_version": MEMORY_VERSION,
                        "approved": False,
                        "model": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "random": rng.getstate(),
                        "torch_random": torch.get_rng_state(),
                        "update": completed,
                        "decisions": total,
                        "total_seconds": previous + time.monotonic() - started,
                        "configuration": configuration,
                        "parent": parent,
                        "parent_sha256": parent_hash,
                        "contracts": contracts,
                        "population_policy": "frozen parent and parent history; only designated learner records",
                    }
                    temp = path.with_suffix(".tmp")
                    torch.save(data, temp)
                    temp.replace(path)
                    if completed == 1 or data["total_seconds"] >= next_snapshot:
                        snapshots = root / "snapshots"
                        snapshots.mkdir(exist_ok=True)
                        compact = {
                            k: data[k]
                            for k in (
                                "training_kind",
                                "encoder",
                                "memory_version",
                                "approved",
                                "model",
                                "configuration",
                                "total_seconds",
                                "decisions",
                                "update",
                                "parent_sha256",
                                "contracts",
                            )
                        }
                        compact.update(
                            parent={"model": parent["model"]}, evaluation_only=True
                        )
                        torch.save(
                            compact,
                            snapshots / f"seconds-{int(data['total_seconds']):07d}.pt",
                        )
                        next_snapshot = (int(data["total_seconds"] // 60) + 1) * 60
                    metric = {
                        "update": completed,
                        "decisions": len(records),
                        "total_decisions": total,
                        "matches": matches,
                        "events": events,
                        "loss": loss,
                        "decisions_per_second": len(records)
                        / (time.monotonic() - begin),
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_group_rss_bytes": monitor.peak,
                    }
                    with (root / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(metric) + "\n")
                    print(json.dumps(metric), flush=True)
                    status("running")
                    if time.monotonic() - started >= args.max_hours * 3600:
                        break
            status(
                "complete" if total >= args.decisions else "time-budget",
                checkpoint=str(path),
            )
        except BaseException as error:
            status("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
