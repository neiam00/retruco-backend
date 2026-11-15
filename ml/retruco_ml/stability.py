"""Instrumented recurrent PPO; legacy control and conservative update bundle.

No recurrent state is cached across optimizer steps. Reference logits use the
same authorized journals and legal masks as the learner, including off-turn events.
"""

import copy
import math

import torch

from .learning_choices import choice_count, meaningful
from .recurrent_batch import PackedRecords

ARMS = {
    "A": {"stability": False, "choice_credit": False},
    "B": {"stability": True, "choice_credit": False},
    "C": {"stability": False, "choice_credit": True},
    "D": {"stability": True, "choice_credit": True},
}


def categorical_kl(reference, candidate):
    # Categorical.logits retains finite log probabilities even when exp(logp)
    # underflows. Avoid 0 * (-inf - -inf) at masked/padded positions.
    support = reference.probs > 0
    left = reference.logits.masked_fill(~support, 0)
    right = candidate.logits.masked_fill(~support, 0)
    return (reference.probs * (left - right)).sum(-1)


def cache_logits(actor, packed):
    result = [None] * len(packed.records)
    with torch.no_grad():
        for start in range(0, len(packed.chunks), 16):
            contexts, indices = packed.contexts(
                actor, packed.chunks[start : start + 16]
            )
            distribution = packed.distribution(actor, contexts, indices)
            for row, index in enumerate(indices):
                count = (
                    6
                    if packed.records[index].observation["kind"] == "signal"
                    else len(packed.candidates[index])
                )
                result[index] = distribution.logits[row, :count].clone()
    return result


def reference_distribution(logits, indices, shape):
    values = logits[indices[0]].new_full(shape, -torch.inf)
    for row, index in enumerate(indices):
        values[row, : len(logits[index])] = logits[index]
    return torch.distributions.Categorical(logits=values)


def groups(packed, rng, stable):
    chunks = packed.chunks
    rng.shuffle(chunks)
    if not stable:
        return [chunks[i : i + 16] for i in range(0, len(chunks), 16)]
    result, group, count = [], [], 0
    for chunk in chunks:
        if group and count + len(chunk) > 512:
            result.append(group)
            group, count = [], 0
        group.append(chunk)
        count += len(chunk)
    if group:
        result.append(group)
    return result


def diagnostics(actor, critic, packed, old, parent, clip):
    values, divergences, anchors, ratios, entropies = {}, {}, {}, {}, {}
    with torch.no_grad():
        for start in range(0, len(packed.chunks), 16):
            contexts, indices = packed.contexts(
                actor, packed.chunks[start : start + 16]
            )
            distribution = packed.distribution(actor, contexts, indices)
            before = reference_distribution(old, indices, distribution.logits.shape)
            teacher = reference_distribution(parent, indices, distribution.logits.shape)
            kl = categorical_kl(before, distribution)
            anchor = categorical_kl(teacher, distribution)
            ratio = (
                distribution.log_prob(packed.actions[indices]) - packed.old[indices]
            ).exp()
            prediction = critic(
                packed.features[indices], packed.privileged[indices], contexts
            )
            entropy = distribution.entropy()
            for row, index in enumerate(indices):
                values[index] = float(prediction[row])
                divergences[index], anchors[index] = float(kl[row]), float(anchor[row])
                ratios[index], entropies[index] = float(ratio[row]), float(entropy[row])

    def summary(indices):
        if not indices:
            return {"count": 0}
        predicted = torch.tensor([values[i] for i in indices])
        target = packed.targets[indices]
        variance = float(target.var(unbiased=False))
        return {
            "count": len(indices),
            "exact_kl": sum(divergences[i] for i in indices) / len(indices),
            "parent_kl": sum(anchors[i] for i in indices) / len(indices),
            "clip_fraction": sum(abs(ratios[i] - 1) > clip for i in indices)
            / len(indices),
            "entropy": sum(entropies[i] for i in indices) / len(indices),
            "normalized_entropy": sum(
                entropies[i] / math.log(choice_count(packed.records[i].observation))
                for i in indices
            )
            / len(indices),
            "value_mse": float((predicted - target).square().mean()),
            "terminal_value_mse": float(
                (
                    predicted
                    - torch.tensor(
                        [
                            getattr(
                                packed.records[i],
                                "terminal_reward",
                                packed.records[i].target,
                            )
                            for i in indices
                        ]
                    )
                )
                .square()
                .mean()
            ),
            "explained_variance": None
            if variance < 1e-10
            else 1 - float((target - predicted).var(unbiased=False)) / variance,
        }

    real = [i for i, record in enumerate(packed.records) if meaningful(record)]
    return {
        "all_choices": summary(real),
        "game": summary(
            [i for i in real if packed.records[i].observation["kind"] == "game"]
        ),
        "signal": summary(
            [i for i in real if packed.records[i].observation["kind"] == "signal"]
        ),
        "response": summary(
            [
                i
                for i in real
                if packed.records[i].observation.get("responseChoice", False)
            ]
        ),
        "claim_signal": summary(
            [
                i
                for i in real
                if packed.records[i].observation["kind"] == "signal"
                and any(packed.records[i].observation.get("signalMask", [True] * 6)[1:])
            ]
        ),
        "forced_records": len(packed.records) - len(real),
    }


def update(
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    records,
    rng,
    parent_actor,
    stable=False,
    policy_weight=1.0,
    monitor=None,
    target_kl=0.01,
    epochs=None,
):
    packed = PackedRecords(records)
    old = cache_logits(actor, packed)
    parent = cache_logits(parent_actor, packed)
    clip = 0.1 if stable else 0.2
    epochs = (2 if stable else 4) if epochs is None else epochs
    actor_stopped = False
    rolled_back = False
    batch_sizes, choice_sizes = [], []
    last_policy = last_value = 0.0
    completed_epochs = 0
    for _ in range(epochs):
        actor_state = (
            copy.deepcopy(actor.state_dict())
            if stable and policy_weight and not actor_stopped
            else None
        )
        optimizer_state = (
            copy.deepcopy(actor_optimizer.state_dict())
            if actor_state is not None
            else None
        )
        for chunk_group in groups(packed, rng, stable):
            if monitor:
                monitor.check()
            contexts, indices = packed.contexts(actor, chunk_group)
            distribution = packed.distribution(actor, contexts, indices)
            ratio = (
                distribution.log_prob(packed.actions[indices]) - packed.old[indices]
            ).exp()
            advantage = packed.advantages[indices]
            policy = -torch.minimum(
                ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage
            ).mean()
            actor_loss = policy - 0.01 * distribution.entropy().mean()
            if stable:
                teacher = reference_distribution(
                    parent, indices, distribution.logits.shape
                )
                actor_loss = (
                    actor_loss + 0.01 * categorical_kl(teacher, distribution).mean()
                )
            prediction = critic(
                packed.features[indices], packed.privileged[indices], contexts
            )
            errors = (prediction - packed.targets[indices]).square()
            if stable:
                old_values = torch.tensor([records[i].value for i in indices])
                clipped = old_values + (prediction - old_values).clamp(-0.2, 0.2)
                errors = torch.maximum(
                    errors, (clipped - packed.targets[indices]).square()
                )
            value_loss = 0.5 * errors.mean()
            loss = value_loss + (
                policy_weight * actor_loss if not actor_stopped else actor_loss * 0
            )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite controlled PPO objective")
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss.backward()
            if not policy_weight or actor_stopped:
                for parameter in actor.parameters():
                    parameter.grad = None
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            actor_optimizer.step()
            critic_optimizer.step()
            last_policy, last_value = float(policy.detach()), float(value_loss.detach())
            batch_sizes.append(len(indices))
            choice_sizes.append(sum(meaningful(records[i]) for i in indices))
        completed_epochs += 1
        if stable and policy_weight and not actor_stopped:
            current = diagnostics(actor, critic, packed, old, parent, clip)
            kl = current["all_choices"]["exact_kl"]
            if kl > 1.5 * target_kl:
                actor.load_state_dict(actor_state)
                actor_optimizer.load_state_dict(optimizer_state)
                for group in actor_optimizer.param_groups:
                    group["lr"] = max(1e-6, group["lr"] * 0.5)
                actor_stopped = rolled_back = True
            elif kl > target_kl:
                actor_stopped = True
    result = diagnostics(actor, critic, packed, old, parent, clip)
    result.update(
        policy_loss=last_policy,
        value_loss=last_value,
        epochs=completed_epochs,
        actor_early_stopped=actor_stopped,
        actor_epoch_rolled_back=rolled_back,
        actor_lr=actor_optimizer.param_groups[0]["lr"],
        batch_records_mean=sum(batch_sizes) / len(batch_sizes),
        batch_choices_mean=sum(choice_sizes) / len(choice_sizes),
    )
    return result
