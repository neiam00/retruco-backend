"""Experimental public-information, response-only semantic policy anchor.

This auxiliary loss transfers a frozen policy's state-dependent wager decisions.
It does not change rewards, legal candidates, sampling, or the deployed actor.
Only (command type, command kind) probability masses are anchored: declaration,
mode, card and raise-amount choices remain conditional policy decisions. A
teacher is a prior to test, not an oracle or a desired acceptance/refusal quota.
"""

import copy
import math

import numpy as np
import torch

from .recurrent_batch import PackedRecords
from .stability import cache_logits

# Shared Go EncodeAction ABI v1: six type bits + empty, ten kind bits + empty.
_TYPES = ("cut", "play", "stack", "burn", "call", "respond", "")
_KINDS = (
    "truco",
    "envido",
    "ley_envido",
    "flor",
    "accept",
    "decline",
    "raise",
    "flowers_no",
    "falta",
    "present_flor",
    "",
)


def _groups(observation):
    if not observation.get("responseChoice", False):
        return None
    if observation["kind"] != "game":
        raise ValueError("response anchor requires an actual game response choice")
    actions = np.asarray(observation["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 48 or len(actions) < 2:
        raise ValueError("response anchor requires legal 48-feature candidates")
    header = actions[:, :18]
    if (
        not np.isfinite(actions).all()
        or not np.isin(header, (0, 1)).all()
        or not (header[:, :7].sum(1) == 1).all()
        or not (header[:, 7:18].sum(1) == 1).all()
    ):
        raise ValueError("response anchor action encoder mismatch")
    keys = [(int(a[:7].argmax()), int(a[7:18].argmax())) for a in actions]
    if (5, 4) not in keys or (5, 5) not in keys:
        raise ValueError("response anchor requires both legal accept and decline")
    unique = sorted(set(keys))
    return tuple(
        (
            f"{_TYPES[k[0]]}:{_KINDS[k[1]]}",
            torch.tensor([i for i, key in enumerate(keys) if key == k]),
        )
        for k in unique
    )


def _marginal(logits, groups):
    # Log-space aggregation keeps very unlikely but legal responses finite.
    values = torch.stack(
        [torch.logsumexp(logits[ids.to(logits.device)], 0) for _, ids in groups]
    )
    if not torch.isfinite(values).all():
        raise ValueError(
            "response anchor encountered a masked or nonfinite legal group"
        )
    return values - torch.logsumexp(values, 0)


def _kl(teacher, learner):
    teacher = teacher.to(device=learner.device, dtype=learner.dtype)
    probabilities = teacher.exp()
    return (probabilities * (teacher - learner)).sum()


class ResponseAnchor:
    """Prepare once per rollout, add loss to PPO, then inspect final diagnostics.

    ``prepare(packed, weights=None)`` defaults to uniform record weights.
    ``loss(distribution, indices, denominator, weights=None)`` can override with
    a full-rollout weight vector, not minibatch weights. The normalization is
    N_all / sum(w_response), retaining the uniform decision minibatch clock.
    ``diagnostics(actor, weights=None)`` replays only response records and reports
    weighted means before/after the update. Teacher parameters never train.
    """

    def __init__(self, teacher, weight=0.1):
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError(
                "response anchor weight must be finite and between zero and one"
            )
        self.weight = float(weight)
        self.teacher = copy.deepcopy(teacher).eval().requires_grad_(False)
        self.packed = None

    def _weights(self, weights):
        if self.packed is None:
            raise RuntimeError("prepare response anchor before use")
        if weights is None:
            return self.weights
        values = torch.as_tensor(weights, dtype=torch.float64).detach().cpu()
        if (
            values.shape != (len(self.packed.records),)
            or not torch.isfinite(values).all()
            or (values < 0).any()
        ):
            raise ValueError(
                "response anchor needs finite nonnegative full-rollout weights"
            )
        return values

    def prepare(self, packed, weights=None):
        self.packed = packed
        self.weights = torch.ones(len(packed.records), dtype=torch.float64)
        self.weights = self._weights(weights)
        self.groups = {}
        for index, record in enumerate(packed.records):
            groups = _groups(record.observation)
            if groups is not None:
                self.groups[index] = groups
        self.indices = list(self.groups)
        self.reference = {}
        self.before = None
        self.response_packed = None
        if self.indices:
            self.response_packed = PackedRecords(
                [packed.records[i] for i in self.indices], packed.sequence
            )
            # PackedRecords.contexts/candidate_distribution use only authorized
            # features, the seat's journal and its exact legal candidate list.
            logits = cache_logits(self.teacher, self.response_packed)
            for index, values in zip(self.indices, logits, strict=True):
                self.reference[index] = _marginal(values, self.groups[index]).detach()
        return self

    def capture_before(self, actor):
        """Call before optimizer steps; useful for measuring this update's change."""
        self._weights(None)
        self.before = self._actor_marginals(actor)
        return self

    def _actor_marginals(self, actor):
        if not self.indices:
            return {}
        logits = cache_logits(actor, self.response_packed)
        return {
            index: _marginal(values, self.groups[index]).detach()
            for index, values in zip(self.indices, logits, strict=True)
        }

    def loss(self, distribution, indices, denominator, weights=None):
        values = self._weights(weights)
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("response anchor needs a positive minibatch denominator")
        if distribution.logits.ndim != 2 or len(indices) != len(distribution.logits):
            raise ValueError(
                "response anchor batch indices must align with distribution"
            )
        if len(set(indices)) != len(indices) or any(
            i < 0 or i >= len(values) for i in indices
        ):
            raise ValueError("response anchor requires unique valid record indices")
        # Avoid -inf padded logits even for a batch with no response records.
        result = distribution.probs.sum() * 0.0
        mass = float(values[self.indices].sum())
        if not mass or not self.weight:
            return result
        for row, index in enumerate(indices):
            if index in self.groups and values[index] > 0:
                learner = _marginal(distribution.logits[row], self.groups[index])
                result = result + float(values[index]) * _kl(
                    self.reference[index], learner
                )
        return result * (self.weight * len(values) / mass / denominator)

    def diagnostics(self, actor, weights=None):
        values = self._weights(weights)
        after = self._actor_marginals(actor)

        def summary(indices):
            mass = float(values[indices].sum())
            result = {
                "response_count": len(indices),
                "response_weight": mass,
                "teacher_group_mass": {},
                "learner_group_mass": {},
            }
            if not mass:
                return {
                    **result,
                    "semantic_teacher_kl_before": None,
                    "semantic_teacher_kl_after": None,
                    "weighted_anchor_loss": 0.0,
                }
            total = previous = 0.0
            for index in indices:
                fraction = float(values[index]) / mass
                total += fraction * float(_kl(self.reference[index], after[index]))
                if self.before is not None:
                    previous += fraction * float(
                        _kl(self.reference[index], self.before[index])
                    )
                for group, (name, _) in enumerate(self.groups[index]):
                    for key, logits in (
                        ("teacher_group_mass", self.reference),
                        ("learner_group_mass", after),
                    ):
                        result[key][name] = result[key].get(
                            name, 0.0
                        ) + fraction * float(logits[index][group].exp())
            return {
                **result,
                "semantic_teacher_kl_before": previous
                if self.before is not None
                else None,
                "semantic_teacher_kl_after": total,
                "weighted_anchor_loss": self.weight * total,
            }

        families = {}
        for index in self.indices:
            family = self.packed.records[index].observation.get(
                "responseFamily", "unknown"
            )
            families.setdefault(family, []).append(index)
        return {
            "kind": "semantic-response-anchor-v1",
            "coefficient": self.weight,
            **summary(self.indices),
            "families": {name: summary(indices) for name, indices in families.items()},
        }
