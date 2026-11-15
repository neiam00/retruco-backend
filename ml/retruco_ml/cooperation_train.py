"""Model D fine-tuning with honest CPU claims and a versioned partner population."""

import argparse
import fcntl
import json
import os
import random
import time
from pathlib import Path

import torch

from .cooperation_population import CooperationPopulation
from .cooperation_rollout import collect
from .model import Policy
from .recurrent import Actor, Critic, file_hash
from .recurrent_batch import update
from .resources import MemoryBudget
from .rollout import SimulatorPool

KIND = "cooperation-mappo-v1"


def load_inputs(manifest_path, seed):
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent

    def load(entry):
        path = (root / entry["file"]).resolve()
        if root.resolve() not in path.parents or file_hash(path) != entry["sha256"]:
            raise ValueError("frozen cooperation input changed")
        return torch.load(path, map_location="cpu", weights_only=True)

    reference = load(manifest["reference"])
    parents = {int(k): load(v) for k, v in manifest["parents"].items()}
    critics = {int(k): load(v) for k, v in manifest["critics"].items()}
    actor = Actor(True)
    actor.load_state_dict(parents[seed]["model"])
    critic = Critic(True)
    critic.load_state_dict(critics[seed]["critic"])
    for parameter in [*actor.parameters(), *critic.parameters()]:
        if not torch.isfinite(parameter).all():
            raise ValueError("nonfinite parent weights")
    legacy = Policy()
    legacy.load_state_dict(reference["model"])
    legacy.eval()
    frozen = {}
    for identity, saved in parents.items():
        policy = Actor(True)
        policy.load_state_dict(saved["model"])
        policy.eval()
        frozen[f"frozen:D{identity}"] = policy
    population = CooperationPopulation(
        reference.get("population"), learned_controllers=tuple(frozen)
    )
    return actor, critic, legacy, population, frozen, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hours", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--environments", type=int, default=32)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--memory-gib", type=float, default=8)
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--updates", type=int, default=100000)
    args = parser.parse_args()
    import math

    if (
        not math.isfinite(args.hours)
        or args.hours <= 0
        or not 1 <= args.workers <= 64
        or not 1 <= args.environments <= 64
        or min(args.steps, args.updates) <= 0
        or args.warmup_updates < 0
    ):
        parser.error("invalid campaign capacity")
    root = args.output.resolve()
    if root.exists() and not args.resume:
        parser.error("new cooperation output must not exist")
    if args.resume and not root.exists():
        parser.error("resume output missing")
    if root == args.inputs.resolve().parent or root in args.inputs.resolve().parents:
        parser.error("output cannot contain frozen inputs")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed + 9000)
    rng = random.Random(args.seed + 9000)
    actor, critic, reference, population, frozen, inputs = load_inputs(
        args.inputs, args.seed
    )
    source = Path(__file__).resolve().parent
    contracts = {
        "inputs_sha256": file_hash(args.inputs),
        "simulator_sha256": file_hash(args.simulator),
        "recipe": population.recipe,
        "signal_policy": "truthful-cpu-v1",
        "encoder": 2,
        "memory": 1,
        "reset": "hand",
        "arm": "D",
        "source": {p.name: file_hash(p) for p in source.glob("*.py")},
    }
    configuration = {
        k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=1e-4)
    root.mkdir(parents=True, exist_ok=args.resume)
    with (root / "train.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("cooperation trainer already live")
        checkpoint = root / "checkpoint.pt"
        completed = decisions = 0
        previous = 0.0
        if args.resume:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (
                saved.get("training_kind") != KIND
                or saved.get("evaluation_only")
                or saved.get("contracts") != contracts
            ):
                raise ValueError(
                    "resume requires exact frozen cooperation source and inputs"
                )
            for key in ("seed", "steps", "workers", "environments", "warmup_updates"):
                if saved["configuration"][key] != configuration[key]:
                    raise ValueError("resume settings changed: " + key)
            actor.load_state_dict(saved["model"])
            critic.load_state_dict(saved["critic"])
            optimizer.load_state_dict(saved["optimizer"])
            rng.setstate(saved["random"])
            torch.set_rng_state(saved["torch_random"])
            completed = saved["update"]
            decisions = saved["decisions"]
            previous = saved["total_seconds"]
            if (root / "chunk.json").exists():
                timing = json.loads((root / "chunk.json").read_text())
                if timing["checkpoint_sha256"] == file_hash(checkpoint):
                    previous = timing["total_seconds"]
        started = time.monotonic()
        next_snapshot = (int(previous // 300) + 1) * 300

        def status(state, **extra):
            temp = root / "status.tmp"
            temp.write_text(
                json.dumps(
                    {
                        "status": state,
                        "pid": os.getpid(),
                        "arm": "D",
                        "seed": args.seed,
                        "update": completed,
                        "decisions": decisions,
                        **extra,
                    }
                )
                + "\n"
            )
            temp.replace(root / "status.json")

        status("running")
        try:
            with SimulatorPool(args.workers, "binary", args.simulator) as pool:
                monitor = MemoryBudget(
                    [s.process for s in pool.simulators], args.memory_gib
                )
                monitor.check(force=True)
                for _ in range(args.updates):
                    begin = time.monotonic()
                    actor.eval()
                    critic.eval()
                    records, matches, events, behavior = collect(
                        actor,
                        critic,
                        reference,
                        pool,
                        population,
                        rng,
                        args.steps,
                        environments=args.environments,
                        monitor=monitor,
                        frozen=frozen,
                    )
                    collected = time.monotonic()
                    actor.train()
                    critic.train()
                    warmup = completed < args.warmup_updates
                    loss = update(
                        actor,
                        critic,
                        optimizer,
                        records,
                        rng,
                        monitor=monitor,
                        policy_weight=0.0 if warmup else 1.0,
                    )
                    optimized = time.monotonic()
                    decisions += len(records)
                    completed += 1
                    data = {
                        "training_kind": KIND,
                        "encoder": 2,
                        "memory_version": 1,
                        "approved": False,
                        "model": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "random": rng.getstate(),
                        "torch_random": torch.get_rng_state(),
                        "update": completed,
                        "decisions": decisions,
                        "total_seconds": previous + time.monotonic() - started,
                        "configuration": {**configuration, "arm": "D"},
                        "contracts": contracts,
                        "parent_sha256": inputs["parents"][str(args.seed)]["sha256"],
                        "critic_parent_sha256": inputs["critics"][str(args.seed)][
                            "sha256"
                        ],
                        "parent": {"model": reference.state_dict()},
                        "signal_policy": "truthful-cpu-v1",
                    }
                    temp = checkpoint.with_suffix(".tmp")
                    torch.save(data, temp)
                    temp.replace(checkpoint)
                    if completed == 1 or data["total_seconds"] >= next_snapshot:
                        folder = root / "snapshots"
                        folder.mkdir(exist_ok=True)
                        compact = {
                            k: v
                            for k, v in data.items()
                            if k
                            not in ("optimizer", "random", "torch_random", "critic")
                        }
                        compact["evaluation_only"] = True
                        torch.save(compact, folder / f"update-{completed:06d}.pt")
                        next_snapshot = (int(data["total_seconds"] // 300) + 1) * 300
                    metric = {
                        "update": completed,
                        "decisions": len(records),
                        "total_decisions": decisions,
                        "matches": matches,
                        "events": events,
                        "loss": loss,
                        "critic_warmup": warmup,
                        "collect_seconds": collected - begin,
                        "optimize_seconds": optimized - collected,
                        "checkpoint_seconds": time.monotonic() - optimized,
                        "decisions_per_second": len(records)
                        / (time.monotonic() - begin),
                        "elapsed_seconds": time.monotonic() - started,
                        "total_seconds": data["total_seconds"],
                        "peak_group_rss_bytes": monitor.peak,
                        "behavior": behavior,
                    }
                    with (root / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(metric) + "\n")
                    print(json.dumps(metric), flush=True)
                    status("running")
                    if time.monotonic() - started >= args.hours * 3600:
                        break
            elapsed = previous + time.monotonic() - started
            receipt = root / "chunk.tmp"
            receipt.write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": file_hash(checkpoint),
                        "update": completed,
                        "total_seconds": elapsed,
                    }
                )
                + "\n"
            )
            receipt.replace(root / "chunk.json")
            status("chunk-complete", checkpoint=str(checkpoint), total_seconds=elapsed)
        except BaseException as error:
            status("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
