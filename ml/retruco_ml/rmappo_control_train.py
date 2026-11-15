"""Matched-parent controls for partner balance and response calibration."""

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import torch

from .controlled_train import atomic_json
from .improvement_budget import DeadlineMonitor, DeadlineReached
from .improvement_evaluate import frozen_models
from .improvement_population import improvement_scripted
from .learning_choices import monte_carlo_credit
from .model_capacity import actor_from_checkpoint as legacy_actor
from .recurrent import file_hash
from .resources import MemoryBudget
from .rmappo_balance import (
    BalanceResult,
    Stratum,
    balance_choices,
    native_partner_balance,
)
from .rmappo_betting import ResponseAnchor
from .rmappo_headsup_protection import (
    CALIBRATED_STARTER_SHA256,
    HeadsUpProtection,
    verify_protected_weights,
)
from .rmappo_model import (
    actor_from_checkpoint,
    critic_from_checkpoint,
)
from .rmappo_population import Population, population_config
from .rmappo_ppo import update
from .rmappo_rollout import collect
from .rollout import SimulatorPool
from .team_play_probe import CHOICE_PROFILES, LIVE_CHOICES


def save(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def covered_collection(
    actor, critic, reference, pool, pop, rng, minimum, *, team_only=False, **kwargs
):
    records = []
    matches = events = 0
    counts = {}
    config = native_partner_balance()
    strata_required = set(config.targets)
    target_profile = getattr(pop, "profile", "original") in (
        "target",
        "reference_focus",
    )
    if target_profile:
        strata_required.update(
            Stratum("2v2", "human", s + "-v2")
            for s in ("literal", "trusting", "cautious")
        )
        if team_only:
            if kwargs.get("players", 4) != 4:
                raise ValueError(
                    "protected actor collection requires four-player matches"
                )
            kwargs["players"] = 4
            strata_required = {s for s in strata_required if s.mode == "2v2"}
        records, matches, events, counts = collect(
            actor, critic, reference, pool, pop, rng, minimum, **kwargs
        )
        strata = Counter(record.balance_stratum for record in records)
        if strata.keys() - strata_required:
            raise ValueError("unconfigured target population stratum")
        n = len(records)
        if not n:
            raise ValueError("target collection returned no genuine choices")
        undercovered = sorted(
            s for s in strata_required if strata[s] < config.min_choices_per_stratum
        )
        observed = {
            s.key: {
                "sampled_choices": strata[s],
                "sampled_fraction": strata[s] / n,
                "effective_mass": strata[s] / n,
                "loss_weight": 1.0,
                "coverage_ok": strata[s] >= config.min_choices_per_stratum,
            }
            for s in sorted(strata_required)
        }
        # Uniform PPO needs no per-batch stratum coverage. Report rare or
        # absent strata without conditioning the declared match population
        # on repeatedly collecting until those strata happen to appear.
        metrics = {
            "sampled_choices": n,
            "weight_sum": float(n),
            "mean_loss_weight": 1.0,
            "max_loss_weight": 1.0,
            "effective_sample_size": float(n),
            "effective_sample_fraction": 1.0,
            "coverage_ok": not undercovered,
            "target_met": None,
            "maximum_target_mass_error": None,
            "missing_strata": [s.key for s in undercovered if not strata[s]],
            "undercovered_strata": [s.key for s in undercovered],
            "capped_strata": [],
            "strata": observed,
            "coverage_policy": "diagnostic only; no stratum rejection or top-up",
            "objective_kind": "unweighted genuine choices; population probabilities apply to matches",
        }
        return records, matches, events, counts, BalanceResult((1.0,) * n, metrics)
    for attempt in range(16):
        more, m, n, stats = collect(
            actor,
            critic,
            reference,
            pool,
            pop,
            rng,
            max(512, minimum - len(records)),
            **kwargs,
        )
        for record in more:
            record.match += matches
        records.extend(more)
        matches += m
        events += n
        for key, value in stats.items():
            counts[key] = counts.get(key, 0) + value
        strata = Counter(record.balance_stratum for record in records)
        if len(records) >= minimum and all(
            strata[s] >= config.min_choices_per_stratum for s in strata_required
        ):
            return records, matches, events, counts, balance_choices(records, config)
    raise RuntimeError("insufficient partner coverage after bounded collection")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--simulator", type=Path, required=True)
    p.add_argument("--cell", choices=("gru", "lstm"), required=True)
    p.add_argument("--parent-checkpoint", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--partner-balance", action="store_true")
    p.add_argument("--response-anchor", type=float, default=0.0)
    p.add_argument("--arm", choices=("A", "B", "C", "D"), required=True)
    p.add_argument(
        "--population-profile",
        choices=("original", "target", "reference_focus"),
        default="original",
    )
    p.add_argument(
        "--protect-headsup",
        action="store_true",
        help="retain the calibrated heads-up policy and learn from four-player matches",
    )
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--gae-lambda", type=float, default=0.99)
    p.add_argument("--response-kl-stop", type=float, default=0.0)
    p.add_argument("--response-kl-rollback", type=float, default=0.0)
    p.add_argument("--candidate-features", type=int, choices=(0, 1), default=0)
    p.add_argument("--semantic-exploration", type=float, default=0.0)
    p.add_argument("--semantic-anneal-choices", type=int, default=1000000)
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
    expected = {
        "A": (False, 0.0),
        "B": (True, 0.0),
        "C": (False, 0.1),
        "D": (True, 0.1),
    }
    if (a.partner_balance, a.response_anchor) != expected[a.arm]:
        p.error("arm label and treatment flags disagree")
    if a.population_profile in ("target", "reference_focus") and (
        a.partner_balance or a.response_anchor
    ):
        p.error("target population uses uniform PPO with no response teacher penalty")
    if a.population_profile == "reference_focus" and not a.protect_headsup:
        p.error("reference-focus continuation requires heads-up protection")
    if a.protect_headsup and (
        a.population_profile == "original"
        or a.candidate_features
        or a.semantic_exploration
    ):
        p.error(
            "protected continuation requires the calibrated architecture and ordinary PPO"
        )
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
    if a.save_every < 0:
        p.error("save-every must be nonnegative")
    if a.gae_lambda not in (0.99, 1.0):
        p.error("supported return estimators are .99 and 1.0")
    if not (
        (a.response_kl_stop == 0 and a.response_kl_rollback == 0)
        or 0 < a.response_kl_stop < a.response_kl_rollback < 1
    ):
        p.error("invalid response KL guard")
    if not 0 <= a.semantic_exploration <= 0.1 or a.semantic_anneal_choices < 1:
        p.error("invalid semantic exploration schedule")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    optimizer_rng = random.Random(a.seed + 70000)
    manifest = json.loads(a.inputs.read_text())
    models = frozen_models(manifest, a.inputs.resolve().parent)
    reference = models.pop("reference")
    models.pop("frozen:103")
    origin = torch.load(a.parent_checkpoint, map_location="cpu", weights_only=True)
    if origin.get("kind") != "rmappo-v1" or origin["configuration"]["cell"] != a.cell:
        raise ValueError("incompatible RMAPPO parent")
    if a.response_anchor not in (0.0, 0.1):
        raise ValueError("unsupported response anchor")
    actor = actor_from_checkpoint(origin)
    critic = critic_from_checkpoint(origin)
    if a.candidate_features and not actor.candidate_features:
        actor.enable_candidate_features(a.candidate_features)
    if actor.candidate_features != a.candidate_features:
        raise ValueError("cannot discard trained candidate features")
    teacher = (
        legacy_actor(torch.load(a.teacher, map_location="cpu", weights_only=True))
        .eval()
        .requires_grad_(False)
    )
    anchor = (
        ResponseAnchor(teacher, weight=a.response_anchor) if a.response_anchor else None
    )
    protection = None
    if a.protect_headsup:
        if file_hash(a.parent_checkpoint) != CALIBRATED_STARTER_SHA256:
            raise ValueError(
                "protected continuation requires the exact calibrated starter"
            )
        protection = HeadsUpProtection(actor, origin["actor"])
    ao = (
        protection.optimizer()
        if protection
        else torch.optim.Adam(actor.parameters(), lr=3e-4, eps=1e-5)
    )
    co = torch.optim.Adam(critic.parameters(), lr=3e-4, eps=1e-5)
    pop = Population(models, profile=a.population_profile)
    pop.league = [
        actor_from_checkpoint(x).eval().requires_grad_(False) for x in origin["league"]
    ]
    if not pop.league:
        pop.add(actor)
    config = {
        "cell": a.cell,
        "seed": a.seed,
        "torch_threads": 1,
        "deterministic_algorithms": True,
        "initialization": "frozen RMAPPO parent; fresh optimizers",
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
        "gae_lambda": a.gae_lambda,
        "headsup_fraction": 0.4,
        "reward": "terminal team match +/-1",
        "response_kl_stop": a.response_kl_stop,
        "response_kl_rollback": a.response_kl_rollback,
        "candidate_features": a.candidate_features,
        "semantic_exploration": a.semantic_exploration,
        "semantic_anneal_choices": a.semantic_anneal_choices,
        "actor_parameters": sum(x.numel() for x in actor.parameters()),
        "critic_parameters": sum(x.numel() for x in critic.parameters()),
        "arm": a.arm,
        "partner_balance": a.partner_balance,
        "response_anchor": a.response_anchor,
        "balance_targets": native_partner_balance().to_dict()
        if a.population_profile == "original"
        else None,
        "population_profile": a.population_profile,
        "population": population_config(a.population_profile),
        "save_every": a.save_every,
        "parent_sha256": file_hash(a.parent_checkpoint),
        "teacher_sha256": file_hash(a.teacher),
        "parent_choices": origin["choices"],
    }
    if protection:
        config.update(
            headsup_protection=1,
            headsup_fraction=0.0,
            players=4,
            editable_actor_weights=int(protection.allowed.sum()),
        )
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
        compatible_config = dict(config)
        if (
            "candidate_features" not in saved["configuration"]
            and a.candidate_features == 0
        ):
            compatible_config.pop("candidate_features")
        if (
            "semantic_exploration" not in saved["configuration"]
            and a.semantic_exploration == 0
        ):
            compatible_config.pop("semantic_exploration")
            compatible_config.pop("semantic_anneal_choices")
        if (
            "response_kl_stop" not in saved["configuration"]
            and a.response_kl_stop == a.response_kl_rollback == 0
        ):
            compatible_config.pop("response_kl_stop")
            compatible_config.pop("response_kl_rollback")
        if (
            a.population_profile == "original"
            and "population_profile" not in saved["configuration"]
            and "population" not in saved["configuration"]
        ):
            # Pre-profile checkpoints already used the unchanged original RNG path.
            compatible_config.pop("population_profile")
            compatible_config.pop("population")
        if a.save_every == 0 and "save_every" not in saved["configuration"]:
            compatible_config.pop("save_every")
        if saved["configuration"] != compatible_config:
            raise ValueError("resume configuration changed")
        if protection:
            verify_protected_weights(origin["actor"], saved["actor"])
        config = saved["configuration"]
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
    if a.save_every:
        (root / "endpoints").mkdir(exist_ok=True)

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
                    "configuration": {
                        "cell": a.cell,
                        "candidate_features": x.candidate_features,
                    },
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
                "arm": a.arm,
                "update": updates,
                "total_choices": choices,
                "target_choices": a.choices,
                "warmups": warmups,
                "training_seconds": seconds,
                "checkpoint": str(root / "checkpoint.pt"),
                **extra,
            },
        )

    def save_endpoint(nominal):
        endpoint = root / "endpoints" / f"choices-{nominal}.pt"
        receipt = endpoint.with_suffix(".sha256")
        if endpoint.exists() or receipt.exists():
            raise FileExistsError("immutable endpoint already exists: " + str(endpoint))
        if protection:
            verify_protected_weights(origin["actor"], actor.state_dict())
        save(endpoint, {**payload(), "endpoint_nominal_choices": nominal})
        with receipt.open("x") as stream:
            stream.write(file_hash(endpoint) + "\n")

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
                records, matches, events, counts, balanced = covered_collection(
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
                    headsup_fraction=config["headsup_fraction"],
                    team_only=bool(protection),
                    gae_lambda=a.gae_lambda,
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
                    choice_weights=balanced.weights if a.partner_balance else None,
                    response_anchor=anchor,
                    response_kl_stop=a.response_kl_stop,
                    response_kl_rollback=a.response_kl_rollback,
                    semantic_exploration=0.0
                    if warm
                    else a.semantic_exploration
                    * max(0.0, 1 - choices / a.semantic_anneal_choices),
                )
                metrics["partner_balance"] = {
                    **balanced.metrics,
                    "applied": a.partner_balance,
                    "actual_objective_masses": {
                        k: (
                            v["effective_mass"]
                            if a.partner_balance
                            else v["sampled_fraction"]
                        )
                        for k, v in balanced.metrics["strata"].items()
                    },
                }
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
                if not warm and a.save_every:
                    first = (choices - len(records)) // a.save_every + 1
                    for boundary in range(first, choices // a.save_every + 1):
                        save_endpoint(boundary * a.save_every)
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
            if a.save_every and choices >= a.choices and a.choices % a.save_every:
                final = root / "endpoints" / f"choices-{a.choices}.pt"
                if not a.resume or not final.exists():
                    save_endpoint(a.choices)
            status("chunk-complete", choices_per_second=choices / max(seconds, 1e-9))
    except DeadlineReached:
        status("budget-exhausted")
        raise SystemExit(75)
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
