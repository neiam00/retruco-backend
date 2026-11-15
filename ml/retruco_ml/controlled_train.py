"""Fresh, bounded three-seed controls from one immutable Model D parent."""

import argparse
import copy
import fcntl
import json
import os
import random
import time
from pathlib import Path

import torch

from .cooperation_rollout import collect
from .cooperation_train import load_inputs
from .learning_choices import choice_credit, meaningful, monte_carlo_credit
from .mappo_rollout import credit
from .recurrent import file_hash
from .resources import MemoryBudget
from .rollout import SimulatorPool
from .stability import ARMS, update

KIND = "controlled-cooperation-v1"


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def credit_for(corrected):
    def assign(trajectory, reward):
        for record in trajectory:
            record.terminal_reward = float(reward)
        if corrected:
            return choice_credit(trajectory, reward)
        credit(trajectory, reward)
        return trajectory

    return assign


def calibration_set(
    actor,
    critic,
    reference,
    pool,
    population,
    frozen,
    seed,
    size,
    environments,
    monitor,
):
    # This separate stream cannot alter the training deal or action RNG streams.
    with torch.random.fork_rng():
        torch.manual_seed(seed + 29000)
        records, _, _, _ = collect(
            actor,
            critic,
            reference,
            pool,
            population,
            random.Random(seed + 29000),
            size,
            environments=environments,
            frozen=frozen,
            monitor=monitor,
            credit_fn=monte_carlo_credit,
            choice_budget=True,
        )
    records = [r for r in records if meaningful(r)]
    from .recurrent_batch import PackedRecords

    packed = PackedRecords(records)
    contexts = [None] * len(records)
    with torch.no_grad():
        for start in range(0, len(packed.chunks), 16):
            values, indices = packed.contexts(actor, packed.chunks[start : start + 16])
            for row, index in enumerate(indices):
                contexts[index] = values[row].clone()
    return {
        "features": packed.features,
        "privileged": packed.privileged,
        "contexts": torch.stack(contexts),
        "targets": packed.targets,
    }


def calibration_score(critic, dataset):
    with torch.no_grad():
        values = critic(dataset["features"], dataset["privileged"], dataset["contexts"])
        target = dataset["targets"]
        mse = float((target - values).square().mean())
        variance = float(target.var(unbiased=False))
        return {
            "mse": mse,
            "explained_variance": None
            if variance < 1e-10
            else 1 - float((target - values).var(unbiased=False)) / variance,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--choices", type=int, default=200000)
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--environments", type=int, default=32)
    parser.add_argument("--calibration-choices", type=int, default=2048)
    parser.add_argument("--warmup-max", type=int, default=8)
    parser.add_argument("--memory-gib", type=float, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int, default=100000)
    args = parser.parse_args()
    if (
        min(
            args.choices,
            args.steps,
            args.workers,
            args.environments,
            args.calibration_choices,
            args.max_updates,
        )
        < 1
        or args.warmup_max < 2
    ):
        parser.error("invalid controlled capacity")
    root = args.output.resolve()
    if root.exists() != args.resume:
        parser.error("new output must not exist; resume requires an existing output")
    if root in args.inputs.resolve().parents:
        parser.error("output cannot contain immutable inputs")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed + 9000)
    rng = random.Random(args.seed + 9000)
    actor, critic, reference, population, frozen, inputs = load_inputs(args.inputs, 101)
    parent = copy.deepcopy(actor).eval()
    options = ARMS[args.arm]
    stable = options["stability"]
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-5 if stable else 1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
    configuration = {
        k: str(v.resolve()) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ("resume", "max_updates")
    }
    source = Path(__file__).resolve().parent
    contracts = {
        "source": {p.name: file_hash(p) for p in source.glob("*.py")},
        "simulator_sha256": file_hash(args.simulator),
        "inputs_sha256": file_hash(args.inputs),
        "recipe": population.recipe,
        "parent_seed": 101,
        "parent_sha256": inputs["parents"]["101"]["sha256"],
        "arm_options": options,
        "signal_policy": "truthful-cpu-v1",
    }
    root.mkdir(parents=True, exist_ok=args.resume)
    with (root / "train.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("controlled trainer already live")
        checkpoint = root / "checkpoint.pt"
        warmups = updates = choices = decisions = 0
        phase, previous, misses = "warmup", 0.0, 0
        best_score, best_critic, best_optimizer = None, None, None
        if args.resume:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if file_hash(root / "calibration.pt") != saved["calibration_sha256"]:
                raise ValueError("frozen critic calibration set changed")
            if (
                saved.get("training_kind") != KIND
                or saved["contracts"] != contracts
                or saved["configuration"] != configuration
            ):
                raise ValueError(
                    "resume requires exact controlled source, inputs and configuration"
                )
            actor.load_state_dict(saved["model"])
            critic.load_state_dict(saved["critic"])
            actor_optimizer.load_state_dict(saved["actor_optimizer"])
            critic_optimizer.load_state_dict(saved["critic_optimizer"])
            rng.setstate(saved["random"])
            torch.set_rng_state(saved["torch_random"])
            warmups, updates, choices, decisions = (
                saved[k] for k in ("warmups", "update", "choices", "decisions")
            )
            phase, previous, misses = (
                saved["phase"],
                saved["total_seconds"],
                saved["calibration_misses"],
            )
            best_score, best_critic, best_optimizer = (
                saved[k]
                for k in ("best_calibration", "best_critic", "best_critic_optimizer")
            )
        started = time.monotonic()

        def status(state, **extra):
            atomic_json(
                root / "status.json",
                {
                    "status": state,
                    "pid": os.getpid(),
                    "arm": args.arm,
                    "seed": args.seed,
                    "phase": phase,
                    "update": updates,
                    "warmups": warmups,
                    "choices": choices,
                    "choice_budget": args.choices,
                    "decisions": decisions,
                    **extra,
                },
            )

        def save():
            data = {
                "training_kind": KIND,
                "model": actor.state_dict(),
                "critic": critic.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "random": rng.getstate(),
                "torch_random": torch.get_rng_state(),
                "warmups": warmups,
                "update": updates,
                "choices": choices,
                "decisions": decisions,
                "phase": phase,
                "total_seconds": previous + time.monotonic() - started,
                "configuration": configuration,
                "contracts": contracts,
                "approved": False,
                "calibration_sha256": file_hash(root / "calibration.pt"),
                "calibration_misses": misses,
                "best_calibration": best_score,
                "best_critic": best_critic,
                "best_critic_optimizer": best_optimizer,
                "signal_policy": "truthful-cpu-v1",
                "memory_version": 1,
                "encoder": 2,
            }
            temp = checkpoint.with_suffix(".tmp")
            torch.save(data, temp)
            temp.replace(checkpoint)

        def metric(data):
            with (root / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(data, allow_nan=False) + "\n")
            print(json.dumps(data, allow_nan=False), flush=True)

        try:
            with SimulatorPool(args.workers, "binary", args.simulator) as pool:
                monitor = MemoryBudget(
                    [s.process for s in pool.simulators], args.memory_gib
                )
                monitor.check(force=True)
                calibration_path = root / "calibration.pt"
                if calibration_path.exists():
                    dataset = torch.load(
                        calibration_path, map_location="cpu", weights_only=True
                    )
                else:
                    dataset = calibration_set(
                        actor,
                        critic,
                        reference,
                        pool,
                        population,
                        frozen,
                        args.seed,
                        args.calibration_choices,
                        args.environments,
                        monitor,
                    )
                    torch.save(dataset, calibration_path)
                if best_score is None:
                    best_score = calibration_score(critic, dataset)
                    best_critic, best_optimizer = (
                        copy.deepcopy(critic.state_dict()),
                        copy.deepcopy(critic_optimizer.state_dict()),
                    )
                while phase == "warmup":
                    begin = time.monotonic()
                    assign = (
                        monte_carlo_credit
                        if stable
                        else credit_for(options["choice_credit"])
                    )
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
                        credit_fn=assign,
                        choice_budget=True,
                    )
                    stats = update(
                        actor,
                        critic,
                        actor_optimizer,
                        critic_optimizer,
                        records,
                        rng,
                        parent,
                        stable=stable,
                        policy_weight=0.0,
                        monitor=monitor,
                    )
                    warmups += 1
                    score = calibration_score(critic, dataset)
                    improved = score["mse"] < best_score["mse"] - 0.005
                    if score["mse"] < best_score["mse"]:
                        best_score = score
                        best_critic, best_optimizer = (
                            copy.deepcopy(critic.state_dict()),
                            copy.deepcopy(critic_optimizer.state_dict()),
                        )
                    misses = 0 if improved else misses + 1
                    stop = warmups >= 2 and (
                        not stable or misses >= 2 or warmups >= args.warmup_max
                    )
                    if stop:
                        if stable:
                            critic.load_state_dict(best_critic)
                            critic_optimizer.load_state_dict(best_optimizer)
                        phase = "training"
                        # Common main-training streams across all arms, independent
                        # of calibration and variable warmup duration.
                        rng = random.Random(args.seed + 19000)
                        torch.manual_seed(args.seed + 19000)
                    metric(
                        {
                            "phase": "warmup",
                            "warmup": warmups,
                            "arm": args.arm,
                            "seed": args.seed,
                            "calibration": score,
                            "best_calibration": best_score,
                            "diagnostics": stats,
                            "seconds": time.monotonic() - begin,
                            "choices": sum(meaningful(r) for r in records),
                        }
                    )
                    save()
                    status("running")
                completed_now = 0
                while choices < args.choices and completed_now < args.max_updates:
                    begin = time.monotonic()
                    records, matches, events, behavior = collect(
                        actor,
                        critic,
                        reference,
                        pool,
                        population,
                        rng,
                        min(args.steps, args.choices - choices),
                        environments=args.environments,
                        monitor=monitor,
                        frozen=frozen,
                        credit_fn=credit_for(options["choice_credit"]),
                        choice_budget=True,
                    )
                    collected = time.monotonic()
                    stats = update(
                        actor,
                        critic,
                        actor_optimizer,
                        critic_optimizer,
                        records,
                        rng,
                        parent,
                        stable=stable,
                        monitor=monitor,
                    )
                    actual = sum(meaningful(r) for r in records)
                    choices += actual
                    decisions += len(records)
                    updates += 1
                    completed_now += 1
                    save()
                    metric(
                        {
                            "phase": "training",
                            "update": updates,
                            "arm": args.arm,
                            "seed": args.seed,
                            "choices": actual,
                            "total_choices": choices,
                            "decisions": len(records),
                            "total_decisions": decisions,
                            "matches": matches,
                            "events": events,
                            "choices_per_second": actual / (time.monotonic() - begin),
                            "collect_seconds": collected - begin,
                            "optimize_seconds": time.monotonic() - collected,
                            "total_seconds": previous + time.monotonic() - started,
                            "peak_group_rss_bytes": monitor.peak,
                            "behavior": behavior,
                            "diagnostics": stats,
                        }
                    )
                    status("running")
                    if (
                        choices >= args.choices
                        or choices // 50000 > (choices - actual) // 50000
                    ):
                        snapshots = root / "snapshots"
                        snapshots.mkdir(exist_ok=True)
                        compact = {
                            "model": actor.state_dict(),
                            "approved": False,
                            "evaluation_only": True,
                            "training_kind": KIND,
                            "choices": choices,
                            "update": updates,
                            "contracts": contracts,
                            "signal_policy": "truthful-cpu-v1",
                        }
                        torch.save(compact, snapshots / f"choices-{choices:09d}.pt")
                status(
                    "complete" if choices >= args.choices else "chunk-complete",
                    checkpoint=str(checkpoint),
                    total_seconds=previous + time.monotonic() - started,
                )
        except BaseException as error:
            status("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
