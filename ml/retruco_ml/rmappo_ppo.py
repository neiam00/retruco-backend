"""RMAPPO on independent seat trajectories with separately replayed critic RNNs."""

import copy
import math
from collections import Counter

import numpy as np
import torch

from .decision_ppo import decision_batches, policy_kl, replay_identity_error
from .recurrent_batch import PackedRecords
from .stability import cache_logits


class CriticRecords:
    def __init__(self, records, sequence=32):
        self.records, self.sequence = records, sequence
        self.events = {}
        for r in records:
            if id(r.critic_events) not in self.events:
                self.events[id(r.critic_events)] = torch.from_numpy(
                    np.asarray(r.critic_events, np.float32)
                )

    def values(self, critic, indices):
        groups = {}
        for i in indices:
            r = self.records[i]
            groups.setdefault(
                (id(r.critic_events), (r.critic_end - 1) // self.sequence), []
            ).append(i)
        keys = list(groups)
        starts = [window * self.sequence for _, window in keys]
        lengths = [
            min(self.sequence, len(self.events[key]) - start)
            for (key, _), start in zip(keys, starts)
        ]
        prefix = torch.zeros(len(keys), max(starts), 896)
        window = torch.zeros(len(keys), self.sequence, 896)
        for row, ((key, _), start, n) in enumerate(zip(keys, starts, lengths)):
            prefix[row, :start] = self.events[key][:start]
            window[row, :n] = self.events[key][start : start + n]
        prefix_valid = (
            torch.arange(prefix.shape[1])[None, :] < torch.tensor(starts)[:, None]
        )
        state = critic.recompute_prefix(
            prefix[:, :, :768], prefix[:, :, 768:], valid=prefix_valid
        )
        window = window[:, : max(lengths)]
        window_valid = (
            torch.arange(window.shape[1])[None, :] < torch.tensor(lengths)[:, None]
        )
        values, _ = critic.replay(
            window[:, :, :768], window[:, :, 768:], state, valid=window_valid
        )
        result = {}
        for row, (key, start) in enumerate(zip(keys, starts)):
            for i in groups[key]:
                result[i] = values[row, self.records[i].critic_end - start - 1]
        return torch.stack([result[i] for i in indices])


def update(
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    records,
    rng,
    epochs=4,
    batch_size=512,
    sequence=32,
    monitor=None,
    critic_only=False,
    choice_weights=None,
    response_anchor=None,
    response_kl_stop=0.0,
    response_kl_rollback=0.0,
    response_min_choices=32,
    semantic_exploration=0.0,
):
    if (
        type(response_min_choices) is not int
        or response_min_choices < 1
        or not (
            (response_kl_stop == 0 and response_kl_rollback == 0)
            or 0 < response_kl_stop < response_kl_rollback < 1
        )
    ):
        raise ValueError("invalid response KL guard")
    packed = PackedRecords(records, sequence)
    from .semantic_exploration import SemanticExploration

    exploration = (
        SemanticExploration(semantic_exploration).prepare(packed)
        if semantic_exploration
        else None
    )
    exploration_losses = []
    weights = torch.ones(len(records))
    if choice_weights is not None:
        from .rmappo_balance import normalize_advantages

        if len(choice_weights) != len(records) or any(
            not math.isfinite(w) or w < 0 for w in choice_weights
        ):
            raise ValueError("invalid aligned objective weights")
        weights = torch.tensor(choice_weights, dtype=torch.float32)
        if abs(float(weights.mean()) - 1) > 1e-5:
            raise ValueError("objective weights must average one")
        packed.advantages = torch.tensor(
            normalize_advantages([r.advantage for r in records], choice_weights)
        )
    if response_anchor is not None:
        response_anchor.prepare(packed, weights=weights)
        response_anchor.capture_before(actor)
    anchor_losses = []
    central = CriticRecords(records, sequence)
    old_logits = cache_logits(actor, packed)
    identity = replay_identity_error(packed, old_logits)
    families = Counter(
        r.observation.get("responseFamily", "unknown")
        for r in records
        if r.observation.get("responseChoice", False)
    )
    guarded = ["response"] if sum(families.values()) >= response_min_choices else []
    guarded += [
        "response:" + name for name, n in families.items() if n >= response_min_choices
    ]
    guard_enabled = bool(response_kl_stop and guarded and not critic_only)
    response_max_kl = 0.0
    response_stopped = response_rolled_back = False
    attempted_steps = rolled_back_steps = 0
    with torch.no_grad():
        error = 0.0
        for start in range(0, len(records), batch_size):
            ids = list(range(start, min(start + batch_size, len(records))))
            predicted = central.values(critic, ids)
            error = max(
                error,
                float(
                    (predicted - torch.tensor([records[i].value for i in ids]))
                    .abs()
                    .max()
                ),
            )
    if error > 2e-4:
        raise ValueError(f"critic replay mismatch: {error}")
    actor_steps = critic_steps = 0
    stopped = critic_only
    max_kl = max_actor_grad = max_critic_grad = 0.0
    old_values = torch.tensor([r.value for r in records])
    for _ in range(epochs):
        saved_actor = (
            copy.deepcopy(actor.state_dict()) if guard_enabled and not stopped else None
        )
        saved_optimizer = (
            copy.deepcopy(actor_optimizer.state_dict())
            if saved_actor is not None
            else None
        )
        epoch_start_steps = actor_steps
        for chunks, denominator in decision_batches(packed, rng, batch_size):
            if monitor:
                monitor.check()
            contexts, ids = packed.contexts(actor, chunks)
            distribution = packed.distribution(actor, contexts, ids)
            logratio = (
                distribution.log_prob(packed.actions[ids]) - packed.old[ids].double()
            )
            ratio = logratio.exp()
            approximate_kl = float(
                (weights[ids] * ((ratio - 1) - logratio)).sum().detach()
                / weights[ids].sum().clamp_min(1e-8)
            )
            max_kl = max(max_kl, approximate_kl)
            if approximate_kl > 0.02:
                stopped = True
            advantage = packed.advantages[ids]
            objective = torch.minimum(
                ratio * advantage, ratio.clamp(0.8, 1.2) * advantage
            )
            policy_loss = (
                -(weights[ids] * (objective + 0.01 * distribution.entropy())).sum()
                / denominator
            )
            if response_anchor is not None and not stopped:
                anchor_loss = response_anchor.loss(distribution, ids, denominator)
                policy_loss = policy_loss + anchor_loss
                anchor_losses.append(float(anchor_loss.detach()))
            if semantic_exploration and not stopped:
                extra = exploration.loss(distribution, ids, denominator)
                policy_loss = policy_loss + extra
                exploration_losses.append(float(extra.detach()))
            values = central.values(critic, ids)
            clipped = old_values[ids] + (values - old_values[ids]).clamp(-0.2, 0.2)
            value_loss = (
                0.5
                * (
                    weights[ids]
                    * torch.maximum(
                        (values - packed.targets[ids]).square(),
                        (clipped - packed.targets[ids]).square(),
                    )
                ).sum()
                / denominator
            )
            if not torch.isfinite(value_loss + policy_loss):
                raise RuntimeError("nonfinite RMAPPO loss")
            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            if not stopped:
                policy_loss.backward()
            cg = float(torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5))
            ag = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5))
            if not math.isfinite(cg + ag):
                raise RuntimeError("nonfinite RMAPPO gradient")
            max_actor_grad = max(max_actor_grad, ag)
            max_critic_grad = max(max_critic_grad, cg)
            critic_optimizer.step()
            critic_steps += 1
            if not stopped:
                actor_optimizer.step()
                actor_steps += 1
                attempted_steps += 1
        if saved_actor is not None:
            divergence = policy_kl(actor, packed, old_logits)
            current = max(divergence.get(key, 0.0) for key in guarded)
            response_max_kl = max(response_max_kl, current)
            if current > response_kl_stop:
                stopped = response_stopped = True
            if current > response_kl_rollback:
                actor.load_state_dict(saved_actor)
                actor_optimizer.load_state_dict(saved_optimizer)
                rolled_back_steps += actor_steps - epoch_start_steps
                actor_steps = epoch_start_steps
                stopped = response_stopped = response_rolled_back = True
    diagnostics = (
        {"coefficient": 0.0}
        if response_anchor is None
        else response_anchor.diagnostics(actor)
    )
    diagnostics["weighted_response_anchor_loss_mean"] = (
        sum(anchor_losses) / len(anchor_losses) if anchor_losses else 0.0
    )
    final_kl = policy_kl(actor, packed, old_logits) if guard_enabled else {}
    sampled_responses = Counter()
    for r in records:
        if r.observation.get("responseChoice", False):
            a = r.observation["actions"][r.action]
            sampled_responses[
                "accept"
                if a[5] == 1 and a[11] == 1
                else "decline"
                if a[5] == 1 and a[12] == 1
                else "other"
            ] += 1
    return {
        "response_anchor": diagnostics,
        "replay_identity_max_logp_error": identity,
        "critic_replay_error": error,
        "sampled_legal_responses": dict(sampled_responses),
        "semantic_exploration_coefficient": semantic_exploration,
        "semantic_exploration_loss_mean": sum(exploration_losses)
        / len(exploration_losses)
        if exploration_losses
        else 0.0,
        "semantic_response_rows": len(exploration.groups) if exploration else 0,
        "response_guard_enabled": guard_enabled,
        "response_guard_min_choices": response_min_choices,
        "response_max_exact_kl": response_max_kl,
        "response_final_exact_kl": final_kl,
        "response_guard_stopped": response_stopped,
        "response_guard_rolled_back": response_rolled_back,
        "actor_steps_attempted": attempted_steps,
        "actor_steps_rolled_back": rolled_back_steps,
        "critic_target_bias_before": float((old_values - packed.targets).mean()),
        "critic_target_mse_before": float(
            (old_values - packed.targets).square().mean()
        ),
        "actor_steps": actor_steps,
        "critic_steps": critic_steps,
        "max_approx_kl": max_kl,
        "actor_stopped": stopped,
        "actor_grad_norm": max_actor_grad,
        "critic_grad_norm": max_critic_grad,
        "response_choices": sum(
            r.observation.get("responseChoice", False) for r in records
        ),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
    }
