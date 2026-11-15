"""Training-only, state-conditional exploration of legal response categories.

Uniform-to-policy KL has a finite log-space gradient even when a response has
almost disappeared. It is applied to semantic groups, not their declaration or
amount multiplicity. It changes no legal actions and imposes no response quota.
"""

import math

import torch

from .rmappo_betting import _groups, _marginal


class SemanticExploration:
    def __init__(self, coefficient):
        if not math.isfinite(coefficient) or not 0 <= coefficient <= 0.1:
            raise ValueError("invalid semantic exploration coefficient")
        self.coefficient = coefficient

    def prepare(self, packed):
        self.groups = {
            i: g
            for i, r in enumerate(packed.records)
            if (g := _groups(r.observation)) is not None
        }
        self.scale = len(packed.records) / max(1, len(self.groups))
        return self

    def loss(self, distribution, indices, denominator):
        terms = []
        for row, index in enumerate(indices):
            if index in self.groups:
                logp = _marginal(distribution.logits[row], self.groups[index])
                terms.append(-logp.mean() - math.log(len(logp)))
        if not terms:
            return distribution.logits.new_zeros(())
        return self.coefficient * self.scale * torch.stack(terms).sum() / denominator
