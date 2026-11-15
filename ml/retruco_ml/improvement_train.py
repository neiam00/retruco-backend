"""Decision-uniform, reactive-opponent Model D improvement training."""

import argparse
import copy
import fcntl
import json
import math
import os
import random
import time
from pathlib import Path

import torch

from .benchmark_population import NativeLeaguePopulation, NativePracticePopulation
from .cooperation_practice import CooperationPracticePopulation
from .cooperation_rollout import collect
from .cooperation_train import load_inputs
from .decision_ppo import anchor_coefficients, update, validate_update_options
from .historical_league import HistoricalLeaguePopulation, _fingerprint
from .improvement_budget import DeadlineMonitor, DeadlineReached
from .improvement_inputs import selected_start
from .improvement_population import ImprovementPopulation, improvement_scripted
from .learning_choices import meaningful, monte_carlo_credit
from .mappo_rollout import credit
from .model_capacity import (
    MAX_ACTOR_PARAMETERS,
    MEMORY_SIZES,
    actor_from_checkpoint,
    critic_from_checkpoint,
    widen_actor,
    widen_critic,
)
from .partner_pool import load_partner_pool
from .recurrent import file_hash
from .resources import MemoryBudget
from .rollout import SimulatorPool


def compact_actor_snapshot(actor, configuration, contracts, choices, updates):
    """Self-describing actor-only artifact; never a resumable optimizer state."""
    return {
        "model": actor.state_dict(),
        "approved": False,
        "evaluation_only": True,
        "training_kind": KIND,
        "choices": choices,
        "update": updates,
        "contracts": contracts,
        "signal_policy": "truthful-cpu-v1",
        "encoder": 2,
        "memory_version": 1,
        "configuration": {**configuration, "arm": "D"},
    }


KIND = "improvement-mappo-v1"


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def credit_for(gae_lambda):
    def assign(trajectory, reward):
        for record in trajectory:
            record.terminal_reward = float(reward)
        selected = [record for record in trajectory if meaningful(record)]
        credit(selected, reward, lam=gae_lambda)
        return selected

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
    headsup_fraction=0.2,
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
            script_fn=improvement_scripted,
            headsup_fraction=headsup_fraction,
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
            "brier": float(
                (((values + 1) / 2).clamp(0, 1) - (target + 1) / 2).square().mean()
            ),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--actor-memory", type=int, choices=MEMORY_SIZES, default=16)
    parser.add_argument("--critic-hidden", type=int, nargs=2, default=[96, 64])
    parser.add_argument("--warm-start", default="d-parent")
    parser.add_argument(
        "--warm-start-checkpoint",
        type=Path,
        help="read-only actor/critic source for a fresh capacity run; optimizer state starts fresh",
    )
    parser.add_argument(
        "--gae-lambda", type=float, choices=(0.95, 0.99, 1.0), default=0.95
    )
    parser.add_argument("--reactive", action="store_true")
    parser.add_argument("--self-play-league", action="store_true")
    parser.add_argument("--cooperation-practice", action="store_true")
    parser.add_argument("--partner-manifest", type=Path)
    parser.add_argument("--partner-policy-version", type=int, choices=(2, 3), default=2)
    parser.add_argument("--team-adapter", choices=("residual", "contextual"))
    parser.add_argument("--parent-anchor", type=float, default=0.0)
    parser.add_argument("--response-anchor", type=float, default=0.0)
    parser.add_argument("--signal-anchor", type=float, default=0.0)
    parser.add_argument("--heldout-parent", type=int, choices=(102, 103))
    parser.add_argument("--native-fraction", type=float, default=0.0)
    parser.add_argument("--headsup-fraction", type=float, default=0.2)
    parser.add_argument("--league-interval", type=int, default=50000)
    parser.add_argument("--league-anchor-fraction", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-4)
    parser.add_argument("--actor-max-grad-norm", type=float, default=0.5)
    parser.add_argument("--critic-max-grad-norm", type=float, default=0.5)
    parser.add_argument("--extra-critic-epochs", type=int, default=0)
    parser.add_argument("--response-kl-target", type=float, default=0.02)
    parser.add_argument("--response-kl-stop", type=float, default=0.03)
    parser.add_argument("--response-kl-rollback", type=float, default=0.05)
    parser.add_argument("--response-min-choices", type=int, default=32)
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
    update_options = {
        name: getattr(args, name)
        for name in (
            "entropy_coef",
            "clip_range",
            "target_kl",
            "actor_max_grad_norm",
            "critic_max_grad_norm",
            "extra_critic_epochs",
            "response_kl_target",
            "response_kl_stop",
            "response_kl_rollback",
            "response_min_choices",
        )
    }
    update_options["epochs"] = args.ppo_epochs
    update_options["sequence"] = args.sequence_length
    validate_update_options(**update_options)
    if (
        not math.isfinite(args.actor_learning_rate)
        or not 1e-5 <= args.actor_learning_rate <= 3e-4
        or not math.isfinite(args.critic_learning_rate)
        or not 0 < args.critic_learning_rate < 1
        or args.batch_size < 1
    ):
        parser.error("invalid optimizer learning rate or batch size")
    if (
        min(
            args.choices,
            args.steps,
            args.workers,
            args.environments,
            args.calibration_choices,
            args.max_updates,
            args.league_interval,
        )
        < 1
        or args.warmup_max < 2
    ):
        parser.error("invalid controlled capacity")
    anchor_coefficients(
        [], args.parent_anchor, args.response_anchor, args.signal_anchor
    )
    if args.cooperation_practice and not args.self_play_league:
        parser.error("cooperation practice requires the historical league")
    if (
        args.partner_manifest or args.partner_policy_version != 2
    ) and not args.cooperation_practice:
        parser.error("partner controls require cooperation practice")
    if not 0 <= args.headsup_fraction <= 1 or not 0 <= args.native_fraction <= 1:
        parser.error("population fractions must be finite and in [0,1]")
    if args.native_fraction and not args.self_play_league:
        parser.error("native mixture requires the historical league")
    if (
        not math.isfinite(args.league_anchor_fraction)
        or not 0 <= args.league_anchor_fraction <= 1
    ):
        parser.error("invalid fixed league anchor fraction")
    if args.league_anchor_fraction and not args.self_play_league:
        parser.error("a fixed league anchor requires the historical league")
    root = args.output.resolve()
    if root.exists() != args.resume:
        parser.error("new output must not exist; resume requires an existing output")
    if root in args.inputs.resolve().parents:
        parser.error("output cannot contain immutable inputs")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed + 9000)
    rng = random.Random(args.seed + 9000)
    # Minibatch/critic epoch changes must not consume the episode-sampling stream.
    optimizer_rng = random.Random(args.seed + 59000)
    actor, critic, reference, population, frozen, inputs = load_inputs(args.inputs, 101)
    forbidden_hashes, forbidden_fingerprints = [], []
    if args.heldout_parent is not None:
        heldout = frozen.pop(f"frozen:D{args.heldout_parent}")
        forbidden_hashes.append(inputs["parents"][str(args.heldout_parent)]["sha256"])
        forbidden_fingerprints.append(_fingerprint(heldout.state_dict()))
    partner_pool = (
        load_partner_pool(
            args.partner_manifest,
            forbidden_hashes=forbidden_hashes,
            forbidden_fingerprints=forbidden_fingerprints,
        )
        if args.partner_manifest
        else None
    )
    starter_hash = selected_start(
        actor, critic, inputs, args.inputs.resolve().parent, args.warm_start
    )
    if args.warm_start_checkpoint is not None:
        origin = torch.load(
            args.warm_start_checkpoint, map_location="cpu", weights_only=True
        )
        if origin.get("encoder") != 2 or origin.get("memory_version") != 1:
            raise ValueError("incompatible warm-start observation contract")
        actor, critic = actor_from_checkpoint(origin), critic_from_checkpoint(origin)
        starter_hash = file_hash(args.warm_start_checkpoint)
    actor = widen_actor(actor, args.actor_memory)
    critic = widen_critic(critic, args.critic_hidden)
    if args.team_adapter:
        if actor.team_adapter is None:
            actor.enable_team_adapter(args.team_adapter)
        elif actor.team_adapter != args.team_adapter:
            raise ValueError("team adapter warm-start mode differs")
    elif getattr(actor, "team_adapter", None) is not None:
        raise ValueError("team adapter warm start requires an explicit matching mode")
    if sum(p.numel() for p in actor.parameters()) > MAX_ACTOR_PARAMETERS:
        raise ValueError("actor exceeds parameter budget")
    parent = copy.deepcopy(actor).eval()
    population_type = (
        CooperationPracticePopulation
        if args.cooperation_practice
        else HistoricalLeaguePopulation
        if args.self_play_league
        else ImprovementPopulation
    )
    if args.native_fraction:
        population_type = (
            NativePracticePopulation
            if args.cooperation_practice
            else NativeLeaguePopulation
        )
    population = population_type(
        population.snapshots,
        learned_controllers=tuple(frozen),
        reactive=args.reactive,
        **(
            {
                "partner_actor": parent,
                "partner_pool": partner_pool,
                "partner_policy_version": args.partner_policy_version,
            }
            if args.cooperation_practice
            else {}
        ),
        **(
            {"anchor_actor": parent, "anchor_fraction": args.league_anchor_fraction}
            if args.league_anchor_fraction
            else {}
        ),
        **({"native_fraction": args.native_fraction} if args.native_fraction else {}),
    )
    if args.self_play_league:
        legacy = inputs.get("candidates", {}).get("legacy-champion")
        if legacy is not None:
            path = (args.inputs.resolve().parent / legacy["file"]).resolve()
            if (
                not path.is_relative_to(args.inputs.resolve().parent)
                or file_hash(path) != legacy["sha256"]
            ):
                raise ValueError("frozen league seed changed")
            population.add_actor(
                actor_from_checkpoint(
                    torch.load(path, map_location="cpu", weights_only=True)
                ),
                0,
            )
        population.add_actor(actor, 0)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.critic_learning_rate
    )
    configuration = {
        k: str(v.resolve()) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ("resume", "max_updates")
    }
    if args.team_adapter:
        from .recurrent import TEAM_ADAPTER_VERSION

        configuration["team_adapter_version"] = TEAM_ADAPTER_VERSION
    source = Path(__file__).resolve().parent
    contracts = {
        "source": {p.name: file_hash(p) for p in source.glob("*.py")},
        "simulator_sha256": file_hash(args.simulator),
        "inputs_sha256": file_hash(args.inputs),
        "recipe": population.recipe,
        "parent_seed": 101
        if args.warm_start == "d-parent" and args.warm_start_checkpoint is None
        else None,
        "parent_sha256": starter_hash,
        "starter_sha256": starter_hash,
        "reference_parent_seed": 101,
        "reference_parent_sha256": inputs["parents"]["101"]["sha256"],
        "anchor_source_sha256": starter_hash,
        "rng_protocol": "separate-optimizer-v1",
        "uniform_decision_batch": args.batch_size,
        "lambda": args.gae_lambda,
        "reactive": args.reactive,
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
                or {k: v for k, v in saved["configuration"].items() if k != "choices"}
                != {
                    k: v
                    for k, v in {**configuration, "arm": "D"}.items()
                    if k != "choices"
                }
            ):
                raise ValueError(
                    "resume requires exact controlled source, inputs and configuration"
                )
            if args.choices < saved["configuration"]["choices"]:
                raise ValueError("resume budget cannot rewind a requested budget")
            actor.load_state_dict(saved["model"])
            critic.load_state_dict(saved["critic"])
            actor_optimizer.load_state_dict(saved["actor_optimizer"])
            critic_optimizer.load_state_dict(saved["critic_optimizer"])
            if args.self_play_league:
                population.load_state_dict(saved["league"])
            rng.setstate(saved["random"])
            optimizer_rng.setstate(saved["optimizer_random"])
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
                    "arm": "D",
                    "variant": args.variant,
                    "seed": args.seed,
                    "phase": phase,
                    "update": updates,
                    "warmups": warmups,
                    "choices": choices,
                    "choice_budget": args.choices,
                    "decisions": decisions,
                    "actor_memory": args.actor_memory,
                    "critic_hidden": args.critic_hidden,
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
                "optimizer_random": optimizer_rng.getstate(),
                "torch_random": torch.get_rng_state(),
                "warmups": warmups,
                "update": updates,
                "choices": choices,
                "decisions": decisions,
                "phase": phase,
                "total_seconds": previous + time.monotonic() - started,
                "configuration": {**configuration, "arm": "D"},
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
                "league": population.state_dict() if args.self_play_league else None,
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
                monitor = DeadlineMonitor(
                    MemoryBudget([s.process for s in pool.simulators], args.memory_gib)
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
                        args.headsup_fraction,
                    )
                    temporary = calibration_path.with_suffix(".tmp")
                    torch.save(dataset, temporary)
                    temporary.replace(calibration_path)
                if best_score is None:
                    best_score = calibration_score(critic, dataset)
                    best_critic, best_optimizer = (
                        copy.deepcopy(critic.state_dict()),
                        copy.deepcopy(critic_optimizer.state_dict()),
                    )
                if not checkpoint.exists():
                    save()
                status("running")
                while phase == "warmup":
                    begin = time.monotonic()

                    def assign(trajectory, reward):
                        selected = [
                            record for record in trajectory if meaningful(record)
                        ]
                        monte_carlo_credit(selected, reward)
                        return selected

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
                        script_fn=improvement_scripted,
                        headsup_fraction=args.headsup_fraction,
                    )
                    stats = update(
                        actor,
                        critic,
                        actor_optimizer,
                        critic_optimizer,
                        records,
                        optimizer_rng,
                        parent,
                        policy_weight=0.0,
                        monitor=monitor,
                        batch_size=args.batch_size,
                        **update_options,
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
                    stop = warmups >= 2 and (misses >= 2 or warmups >= args.warmup_max)
                    if stop:
                        critic.load_state_dict(best_critic)
                        critic_optimizer.load_state_dict(best_optimizer)
                        phase = "training"
                        # Common main-training streams across all arms, independent
                        # of calibration and variable warmup duration.
                        rng = random.Random(args.seed + 19000)
                        torch.manual_seed(args.seed + 19000)
                        optimizer_rng = random.Random(args.seed + 69000)
                    metric(
                        {
                            "phase": "warmup",
                            "warmup": warmups,
                            "arm": "D",
                            "variant": args.variant,
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
                        credit_fn=credit_for(args.gae_lambda),
                        choice_budget=True,
                        script_fn=improvement_scripted,
                        headsup_fraction=args.headsup_fraction,
                    )
                    collected = time.monotonic()
                    stats = update(
                        actor,
                        critic,
                        actor_optimizer,
                        critic_optimizer,
                        records,
                        optimizer_rng,
                        parent,
                        monitor=monitor,
                        batch_size=args.batch_size,
                        parent_anchor=args.parent_anchor,
                        response_anchor=args.response_anchor,
                        signal_anchor=args.signal_anchor,
                        **update_options,
                    )
                    actual = sum(meaningful(r) for r in records)
                    choices += actual
                    decisions += len(records)
                    updates += 1
                    completed_now += 1
                    if (
                        args.self_play_league
                        and choices // args.league_interval
                        > (choices - actual) // args.league_interval
                    ):
                        population.add_actor(actor, choices)
                    save()
                    metric(
                        {
                            "phase": "training",
                            "update": updates,
                            "arm": "D",
                            "variant": args.variant,
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
                            "league_snapshots": population.snapshot_count
                            if args.self_play_league
                            else 0,
                            "choices_by_mode": {
                                str(n): int(
                                    sum(
                                        r.observation["features"][0] == n / 4
                                        for r in records
                                    )
                                )
                                for n in (2, 4)
                            },
                        }
                    )
                    status("running")
                    if (
                        choices >= args.choices
                        or choices // 50000 > (choices - actual) // 50000
                    ):
                        snapshots = root / "snapshots"
                        snapshots.mkdir(exist_ok=True)
                        compact = compact_actor_snapshot(
                            actor, configuration, contracts, choices, updates
                        )
                        torch.save(compact, snapshots / f"choices-{choices:09d}.pt")
                # Record an increased requested budget even when a prior complete
                # match already overshot it and no extra update is necessary.
                save()
                probe = calibration_set(
                    actor,
                    critic,
                    reference,
                    pool,
                    population,
                    frozen,
                    args.seed + 700000 + choices,
                    args.calibration_choices,
                    args.environments,
                    monitor,
                    args.headsup_fraction,
                )
                score = calibration_score(critic, probe)
                strata = {}
                for players, encoded in ((2, 0.5), (4, 1.0)):
                    mask = probe["features"][:, 0] == encoded
                    if mask.any():
                        subset = {key: value[mask] for key, value in probe.items()}
                        strata[str(players)] = calibration_score(critic, subset)
                for trick in range(3):
                    mask = probe["features"][:, 4] == trick / 2
                    if mask.any():
                        strata[f"round_{trick}"] = calibration_score(
                            critic, {k: v[mask] for k, v in probe.items()}
                        )
                atomic_json(
                    root / "critic-probe.json",
                    {
                        "choices": choices,
                        "update": updates,
                        "scope": "independent current-policy complete-match Monte Carlo probe; no actor/critic updates",
                        "checkpoint_sha256": file_hash(checkpoint),
                        "overall": score,
                        "players": strata,
                    },
                )
                probe_report = json.loads((root / "critic-probe.json").read_text())
                atomic_json(
                    root / f"critic-probe-choices-{choices:09d}.json", probe_report
                )
                status(
                    "complete" if choices >= args.choices else "chunk-complete",
                    checkpoint=str(checkpoint),
                    total_seconds=previous + time.monotonic() - started,
                )
        except DeadlineReached as error:
            status(
                "budget-exhausted",
                error=str(error),
                checkpoint=str(checkpoint) if checkpoint.exists() else None,
            )
            raise SystemExit(75)
        except BaseException as error:
            status("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
