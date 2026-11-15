"""Uniform genuine-decision PPO with recurrent replay and measured update size.

Event windows reconstruct BPTT; their density never sets a decision's loss weight.
"""

import copy
import math
from collections import Counter

import torch

from .learning_choices import meaningful
from .recurrent_batch import PackedRecords
from .stability import cache_logits, categorical_kl, diagnostics, reference_distribution


def decision_batches(packed, rng, batch_size=64):
    if batch_size < 1:
        raise ValueError("positive decision batch size required")
    indices = list(range(len(packed.records)))
    if not indices or any(not meaningful(record) for record in packed.records):
        raise ValueError("uniform PPO accepts genuine choices only")
    rng.shuffle(indices)
    count = math.ceil(len(indices) / batch_size)
    base, remainder = divmod(len(indices), count)
    denominator = len(indices) / count
    result, start = [], 0
    for batch in range(count):
        size = base + int(batch < remainder)
        groups = {}
        for index in indices[start : start + size]:
            record = packed.records[index]
            key = (id(record.events), (record.end - 1) // packed.sequence)
            groups.setdefault(key, []).append(index)
        result.append((list(groups.values()), denominator))
        start += size
    return result


def replay_identity_error(packed, logits):
    pairs = [(float(logits[i][r.action]), r.logp) for i, r in enumerate(packed.records)]
    if any(not math.isfinite(value) for pair in pairs for value in pair):
        raise ValueError(
            "rollout/recurrent replay policy mismatch: nonfinite log probability"
        )
    error = max((abs(current - previous) for current, previous in pairs), default=0.0)
    if error > 2e-4:
        raise ValueError(f"rollout/recurrent replay policy mismatch: {error}")
    return error


def policy_kl(actor, packed, old):
    totals = {
        "all": [0.0, 0],
        "game": [0.0, 0],
        "signal": [0.0, 0],
        "response": [0.0, 0],
    }
    for record in packed.records:
        observation = record.observation
        family = observation.get("responseFamily")
        if (
            observation.get("responseChoice", False)
            and isinstance(family, str)
            and family
        ):
            totals.setdefault("response:" + family, [0.0, 0])
    with torch.no_grad():
        for start in range(0, len(packed.chunks), 16):
            contexts, indices = packed.contexts(
                actor, packed.chunks[start : start + 16]
            )
            current = packed.distribution(actor, contexts, indices)
            previous = reference_distribution(old, indices, current.logits.shape)
            divergence = categorical_kl(previous, current)
            for kind in totals:
                rows = [
                    row
                    for row, index in enumerate(indices)
                    if kind == "all"
                    or packed.records[index].observation["kind"] == kind
                    or (
                        kind == "response"
                        and packed.records[index].observation.get(
                            "responseChoice", False
                        )
                    )
                    or (
                        kind.startswith("response:")
                        and packed.records[index].observation.get(
                            "responseChoice", False
                        )
                        and packed.records[index].observation.get("responseFamily")
                        == kind.partition(":")[2]
                    )
                ]
                if rows:
                    totals[kind][0] += float(divergence[rows].sum())
                    totals[kind][1] += len(rows)

    return {
        key: value / count if count else 0.0 for key, (value, count) in totals.items()
    }


def anchor_coefficients(
    records, parent_anchor=0.0, response_anchor=0.0, signal_anchor=0.0
):
    """Per-decision soft preservation, with no preference for accept or send."""
    if any(
        not math.isfinite(v) or not 0 <= v <= 1
        for v in (parent_anchor, response_anchor, signal_anchor)
    ):
        raise ValueError("anchor coefficients must be finite and between zero and one")
    return torch.tensor(
        [
            parent_anchor
            + response_anchor * bool(r.observation.get("responseChoice", False))
            + signal_anchor * (r.observation["kind"] == "signal")
            for r in records
        ]
    )


def validate_update_options(
    *,
    epochs=4,
    target_kl=0.015,
    entropy_coef=0.01,
    clip_range=0.2,
    actor_max_grad_norm=0.5,
    critic_max_grad_norm=0.5,
    extra_critic_epochs=0,
    response_kl_target=0.02,
    response_kl_stop=0.03,
    response_kl_rollback=0.05,
    response_min_choices=32,
    sequence=32,
):
    if (
        not 0 < target_kl < 0.1
        or type(epochs) is not int
        or epochs < 1
        or type(extra_critic_epochs) is not int
        or extra_critic_epochs < 0
        or type(response_min_choices) is not int
        or response_min_choices < 1
        or type(sequence) is not int
        or not 1 <= sequence <= 256
        or not math.isfinite(entropy_coef)
        or entropy_coef < 0
        or not 0 < clip_range < 1
        or any(
            not math.isfinite(v) or v <= 0
            for v in (actor_max_grad_norm, critic_max_grad_norm)
        )
        or not 0 < response_kl_target <= response_kl_stop < response_kl_rollback < 1
    ):
        raise ValueError("invalid update capacity")


def update(
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    records,
    rng,
    parent_actor,
    policy_weight=1.0,
    monitor=None,
    batch_size=64,
    epochs=4,
    target_kl=0.015,
    parent_anchor=0.0,
    response_anchor=0.0,
    signal_anchor=0.0,
    entropy_coef=0.01,
    clip_range=0.2,
    actor_max_grad_norm=0.5,
    critic_max_grad_norm=0.5,
    extra_critic_epochs=0,
    response_kl_target=0.02,
    response_kl_stop=0.03,
    response_kl_rollback=0.05,
    response_min_choices=32,
    sequence=32,
):
    validate_update_options(
        epochs=epochs,
        target_kl=target_kl,
        entropy_coef=entropy_coef,
        clip_range=clip_range,
        actor_max_grad_norm=actor_max_grad_norm,
        critic_max_grad_norm=critic_max_grad_norm,
        extra_critic_epochs=extra_critic_epochs,
        response_kl_target=response_kl_target,
        response_kl_stop=response_kl_stop,
        response_kl_rollback=response_kl_rollback,
        response_min_choices=response_min_choices,
        sequence=sequence,
    )
    packed = PackedRecords(records, sequence=sequence)
    anchors = anchor_coefficients(
        records, parent_anchor, response_anchor, signal_anchor
    )
    old = cache_logits(actor, packed)
    parent = cache_logits(parent_actor, packed)
    identity = replay_identity_error(packed, old)
    actor_stopped = rolled_back = False
    response_choices = sum(
        bool(r.observation.get("responseChoice", False)) for r in records
    )
    response_families = Counter(
        r.observation["responseFamily"]
        for r in records
        if r.observation.get("responseChoice", False)
        and isinstance(r.observation.get("responseFamily"), str)
        and r.observation["responseFamily"]
    )
    family_max_kl = {family: 0.0 for family in response_families}
    final_divergence = {}
    response_guard_enabled = (
        bool(policy_weight) and response_choices >= response_min_choices
    )
    response_max_kl = 0.0
    response_stopped = response_rolled_back = False
    initial_lr = actor_optimizer.param_groups[0]["lr"]
    batches, steps, critic_steps = [], 0, 0
    actor_norms, critic_norms = [], []
    last_policy = last_value = 0.0
    for _ in range(epochs):
        saved_actor = (
            copy.deepcopy(actor.state_dict())
            if policy_weight and not actor_stopped
            else None
        )
        saved_optimizer = (
            copy.deepcopy(actor_optimizer.state_dict())
            if saved_actor is not None
            else None
        )
        for chunk_group, denominator in decision_batches(packed, rng, batch_size):
            if monitor:
                monitor.check()
            contexts, indices = packed.contexts(actor, chunk_group)
            distribution = packed.distribution(actor, contexts, indices)
            ratio = (
                distribution.log_prob(packed.actions[indices]) - packed.old[indices]
            ).exp()
            advantage = packed.advantages[indices]
            objective = torch.minimum(
                ratio * advantage,
                ratio.clamp(1 - clip_range, 1 + clip_range) * advantage,
            )
            policy_loss = (
                -(objective + entropy_coef * distribution.entropy()).sum() / denominator
            )
            if anchors.any():
                teacher = reference_distribution(
                    parent, indices, distribution.logits.shape
                )
                policy_loss = (
                    policy_loss
                    + (anchors[indices] * categorical_kl(teacher, distribution)).sum()
                    / denominator
                )
            predictions = critic(
                packed.features[indices], packed.privileged[indices], contexts
            )
            value_loss = (
                0.5
                * (predictions - packed.targets[indices]).square().sum()
                / denominator
            )
            loss = (
                value_loss + policy_weight * (0 if actor_stopped else 1) * policy_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite decision PPO objective")
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss.backward()
            if not policy_weight or actor_stopped:
                for parameter in actor.parameters():
                    parameter.grad = None
            actor_norm = float(
                torch.nn.utils.clip_grad_norm_(actor.parameters(), actor_max_grad_norm)
            )
            critic_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), critic_max_grad_norm
                )
            )
            if not math.isfinite(actor_norm) or not math.isfinite(critic_norm):
                raise RuntimeError("nonfinite decision PPO gradient")
            if policy_weight and not actor_stopped:
                actor_norms.append(actor_norm)
            critic_norms.append(critic_norm)
            actor_optimizer.step()
            critic_optimizer.step()
            critic_steps += 1
            steps += int(policy_weight and not actor_stopped)
            batches.append((len(indices), 1 / denominator))
            last_policy, last_value = (
                float(policy_loss.detach()),
                float(value_loss.detach()),
            )
        if policy_weight and not actor_stopped:
            divergence = policy_kl(actor, packed, old)
            final_divergence = divergence
            response_kl = divergence.get("response", 0.0)
            for family, count in response_families.items():
                family_kl = divergence.get("response:" + family, 0.0)
                family_max_kl[family] = max(family_max_kl[family], family_kl)
                if count >= response_min_choices:
                    response_kl = max(response_kl, family_kl)
            response_max_kl = max(response_max_kl, response_kl)
            response_rollback = (
                response_guard_enabled and response_kl > response_kl_rollback
            )
            response_stop = response_guard_enabled and response_kl > response_kl_stop
            if (
                divergence["all"] > 0.03
                or max(divergence["game"], divergence["signal"]) > 0.05
                or response_rollback
            ):
                actor.load_state_dict(saved_actor)
                actor_optimizer.load_state_dict(saved_optimizer)
                for group in actor_optimizer.param_groups:
                    group["lr"] = max(1e-5, group["lr"] * 0.5)
                actor_stopped = rolled_back = True
                response_stopped = response_rolled_back = response_rollback
            elif (
                divergence["all"] > 0.02
                or max(divergence["game"], divergence["signal"]) > 0.03
                or response_stop
            ):
                actor_stopped = True
                response_stopped = response_stop
    # Extra value fitting uses the final actor as a frozen feature provider.
    # No actor optimizer call or actor gradient is made in these passes.
    for _ in range(extra_critic_epochs):
        for chunk_group, denominator in decision_batches(packed, rng, batch_size):
            if monitor:
                monitor.check()
            with torch.no_grad():
                contexts, indices = packed.contexts(actor, chunk_group)
            predictions = critic(
                packed.features[indices], packed.privileged[indices], contexts
            )
            value_loss = (
                0.5
                * (predictions - packed.targets[indices]).square().sum()
                / denominator
            )
            if not torch.isfinite(value_loss):
                raise RuntimeError("nonfinite decision PPO critic objective")
            critic_optimizer.zero_grad()
            value_loss.backward()
            critic_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), critic_max_grad_norm
                )
            )
            if not math.isfinite(critic_norm):
                raise RuntimeError("nonfinite decision PPO critic gradient")
            critic_norms.append(critic_norm)
            critic_optimizer.step()
            critic_steps += 1
            last_value = float(value_loss.detach())
    result = diagnostics(actor, critic, packed, old, parent, clip_range)
    if rolled_back and response_families:
        final_divergence = policy_kl(actor, packed, old)
    final_response_kl = result["response"].get("exact_kl", 0.0)
    for family, count in response_families.items():
        if count >= response_min_choices:
            final_response_kl = max(
                final_response_kl, final_divergence.get("response:" + family, 0.0)
            )
    actual_kl = result["all_choices"]["exact_kl"]
    if policy_weight and not rolled_back:
        normalized = max(
            actual_kl / target_kl,
            result["game"].get("exact_kl", 0.0) / 0.02,
            result["signal"].get("exact_kl", 0.0) / 0.02,
        )
        if response_guard_enabled:
            normalized = max(normalized, final_response_kl / response_kl_target)
        factor = (
            1 / 1.5
            if actor_stopped or normalized > 1.5
            else (1.5 if normalized < 1 / 1.5 else 1.0)
        )
        for group in actor_optimizer.param_groups:
            group["lr"] = min(3e-4, max(1e-5, group["lr"] * factor))
    result.update(
        policy_loss=last_policy,
        value_loss=last_value,
        replay_identity_max_logp_error=identity,
        actor_optimizer_steps_attempted=steps,
        epochs=epochs,
        actor_early_stopped=actor_stopped,
        critic_optimizer_steps=critic_steps,
        extra_critic_epochs=extra_critic_epochs,
        actor_epoch_rolled_back=rolled_back,
        actor_lr_used=initial_lr,
        actor_lr_next=actor_optimizer.param_groups[0]["lr"],
        batch_choices_min=min(size for size, _ in batches),
        batch_choices_max=max(size for size, _ in batches),
        per_decision_loss_weight_min=min(weight for _, weight in batches),
        per_decision_loss_weight_max=max(weight for _, weight in batches),
        genuine_choices=len(records),
    )
    result.update(
        response_choices=response_choices,
        signal_choices=sum(r.observation["kind"] == "signal" for r in records),
        anchor_coefficients={
            "all": parent_anchor,
            "response": response_anchor,
            "signal": signal_anchor,
        },
    )
    result["anchor_loss_mean"] = (
        parent_anchor * result["all_choices"]["parent_kl"]
        + response_anchor
        * result["response"].get("parent_kl", 0.0)
        * result["response_choices"]
        / len(records)
        + signal_anchor
        * result["signal"].get("parent_kl", 0.0)
        * result["signal_choices"]
        / len(records)
    )
    result["gradient_norms"] = {
        name: {
            "mean_preclip": sum(values) / len(values) if values else 0.0,
            "max_preclip": max(values, default=0.0),
            "limit": limit,
            "clipped_fraction": sum(v > limit for v in values) / len(values)
            if values
            else 0.0,
        }
        for name, values, limit in [
            ("actor", actor_norms, actor_max_grad_norm),
            ("critic", critic_norms, critic_max_grad_norm),
        ]
    }
    response_max_kl = max(response_max_kl, final_response_kl)
    result["response_kl_guard"] = {
        "enabled": response_guard_enabled,
        "choices": response_choices,
        "minimum_choices": response_min_choices,
        "target": response_kl_target,
        "stop_threshold": response_kl_stop,
        "rollback_threshold": response_kl_rollback,
        "max_observed_kl": response_max_kl,
        "exact_kl": final_response_kl,
        "aggregate_kl": result["response"].get("exact_kl", 0.0),
        "warning": response_max_kl > response_kl_stop,
        "stopped": response_stopped,
        "rolled_back": response_rolled_back,
        "families": {
            family: {
                "choices": count,
                "enabled": bool(policy_weight) and count >= response_min_choices,
                "exact_kl": final_divergence.get("response:" + family, 0.0),
                "max_observed_kl": family_max_kl[family],
            }
            for family, count in response_families.items()
        },
    }
    result["update_controls"] = {
        "entropy_coef": entropy_coef,
        "clip_range": clip_range,
        "sequence": sequence,
    }
    return result
