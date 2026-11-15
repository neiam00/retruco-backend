"""Train population PPO on complete matches using per-seat advantages."""

import argparse
import fcntl
import json
import math
import os
import random
import time
from pathlib import Path

import torch

from .model import ENCODER_VERSION, PackedObservations, Policy
from .population import Population
from .resources import MemoryBudget
from .rollout import SimulatorPool, collect, seat_returns

advantages = seat_returns


def update(model, optimizer, records, rng, epochs=4, batch_size=512, monitor=None):
    # Ragged candidate lists are padded only within each minibatch.
    adv = torch.tensor([r[4] for r in records])
    normalized = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    packed = PackedObservations([r[0] for r in records])
    device = next(model.parameters()).device
    actions = torch.tensor([r[1] for r in records], device=device)
    old_logp = torch.tensor([r[2] for r in records], device=device)
    returns = torch.tensor([r[5] for r in records], device=device)
    normalized = normalized.to(device)
    indices = list(range(len(records)))
    last_loss = 0.0
    for _ in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            if monitor:
                monitor.check()
            selected = indices[start : start + batch_size]
            selected_tensor = torch.tensor(selected, device=device)
            distribution, value = packed.distribution(model, selected, device)
            logp = distribution.log_prob(actions[selected_tensor])
            ratio = (logp - old_logp[selected_tensor]).exp()
            advantage = normalized[selected_tensor]
            policy_loss = -torch.minimum(
                ratio * advantage, ratio.clamp(0.8, 1.2) * advantage
            ).mean()
            value_loss = (value - returns[selected_tensor]).square().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * distribution.entropy().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            last_loss = float(loss.detach())
    return last_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hours", type=float, default=1)
    parser.add_argument("--updates", type=int, default=100000)
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--players", type=int, choices=(0, 2, 4), default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument(
        "--ipc",
        choices=("binary", "json"),
        default="binary",
        help="numeric simulator transport; JSON remains available for older binaries",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--memory-gib", type=float, default=16)
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        help="train a fresh empirical best response against one frozen opponent",
    )
    parser.add_argument(
        "--opponent-quantized",
        type=Path,
        help="use this checkpoint's exact exported int8 target policy",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        not math.isfinite(args.hours)
        or args.hours <= 0
        or args.steps <= 0
        or args.updates <= 0
    ):
        parser.error("hours, updates and steps must be positive")
    if args.opponent_quantized and not args.opponent_checkpoint:
        parser.error("--opponent-quantized requires --opponent-checkpoint")
    torch.set_num_threads(2)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if not 1 <= args.workers <= 64 or not 1 <= args.environments <= 64:
        parser.error("workers and environments must be in [1,64]")
    if args.device == "auto":
        from .benchmark import benchmark

        timing = benchmark()
        print(json.dumps({"device_benchmark_seconds": timing}), flush=True)
        args.device = timing["selected"]
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable on this machine")
    model = Policy().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    run_lock = (args.output / "train.lock").open("a")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("this training directory already has a live trainer")
    checkpoint = args.output / "checkpoint.pt"
    if checkpoint.exists() and not args.resume:
        parser.error(
            "output already has a checkpoint; use --resume or a new run directory"
        )
    completed = 0
    previous_seconds = 0.0
    population = Population()
    opponent_digest = None
    opponent_model_digest = None
    initial_seed = args.seed
    initialization = {"kind": "random", "seed": initial_seed}
    if args.opponent_checkpoint:
        from .assessment import load_checkpoint

        opponent = load_checkpoint(args.opponent_checkpoint)
        if args.opponent_quantized:
            from .artifacts import digest, read_quantized
            from .export import serialize

            data = args.opponent_quantized.read_bytes()
            if data != serialize(opponent)[0]:
                raise ValueError(
                    "quantized opponent is not the exact checkpoint export"
                )
            opponent = read_quantized(data, opponent)
            opponent_model_digest = digest(args.opponent_quantized)
        population.fixed_opponent = opponent.state_dict()
        opponent_digest = opponent.checkpoint_sha256
    if args.resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("encoder") != ENCODER_VERSION:
            raise ValueError("checkpoint encoder is incompatible; start a new run")
        if saved.get("training_kind", "ppo") != "ppo":
            raise ValueError("resume this checkpoint with its QAT trainer, not PPO")
        if opponent_digest is not None and opponent_digest != saved.get(
            "opponent_checkpoint_sha256"
        ):
            raise ValueError("resume target opponent differs from checkpoint")
        if args.opponent_checkpoint and opponent_model_digest != saved.get(
            "opponent_model_sha256"
        ):
            raise ValueError("resume target opponent precision differs from checkpoint")
        population = Population(saved["population"], saved.get("fixed_opponent"))
        opponent_digest = saved.get("opponent_checkpoint_sha256")
        opponent_model_digest = saved.get("opponent_model_sha256")
        initial_seed = saved.get("initial_seed", saved["config"]["seed"])
        initialization = saved.get(
            "initialization", {"kind": "legacy-unknown", "seed": initial_seed}
        )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.setstate(saved["random"])
        torch.set_rng_state(saved["torch_random"])
        completed = saved["update"]
        previous_seconds = saved.get("total_seconds", 0.0)
        if args.device == "mps" and saved.get("mps_random") is not None:
            torch.mps.set_rng_state(saved["mps_random"])
    started = time.monotonic()
    with SimulatorPool(args.workers, transport=args.ipc) as simulator:
        monitor = MemoryBudget(
            [s.process for s in simulator.simulators], args.memory_gib
        )
        monitor.check(force=True)
        for iteration in range(completed, completed + args.updates):
            update_started = time.monotonic()
            # Rollout sampling always runs on CPU to avoid tiny GPU transfers.
            frozen = Policy()
            frozen.load_state_dict(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
            frozen.eval()
            records, matches, controllers = collect(
                frozen,
                simulator,
                population,
                rng,
                args.steps,
                args.players,
                args.environments,
                monitor,
            )
            loss = update(model, optimizer, records, rng, monitor=monitor)
            monitor.check(force=True)
            if (iteration + 1) % 10 == 0 or not population.snapshots:
                population.add(model)
            # Atomic saves only at complete-match/update boundaries.
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "random": rng.getstate(),
                    "torch_random": torch.get_rng_state(),
                    "update": iteration + 1,
                    "approved": False,
                    "encoder": ENCODER_VERSION,
                    "population": population.snapshots,
                    "total_seconds": previous_seconds + time.monotonic() - started,
                    "fixed_opponent": population.fixed_opponent,
                    "opponent_checkpoint_sha256": opponent_digest,
                    "opponent_model_sha256": opponent_model_digest,
                    "initial_seed": initial_seed,
                    "initialization": initialization,
                    "mps_random": torch.mps.get_rng_state()
                    if args.device == "mps"
                    else None,
                    "config": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                },
                temporary,
            )
            os.replace(temporary, checkpoint)
            metrics = {
                "update": iteration + 1,
                "matches": matches,
                "decisions": len(records),
                "loss": loss,
                "controllers": controllers,
                "device": args.device,
                "elapsed_seconds": time.monotonic() - started,
                "peak_group_rss_bytes": monitor.peak,
                "decisions_per_second": len(records)
                / (time.monotonic() - update_started),
                "matches_per_hour": matches
                * 3600
                / (time.monotonic() - update_started),
            }
            with (args.output / "metrics.jsonl").open("a") as log:
                log.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            if time.monotonic() - started >= args.hours * 3600:
                break


if __name__ == "__main__":
    main()
