"""Training-only weight fake quantization; deployment still uses plain Policy."""

import copy

import numpy as np
import torch
from torch import nn

from .model import DEPLOY_LAYERS, batch_distribution


class QuantizedLinear(nn.Linear):
    def forward(self, values):
        scales = self.weight.detach().abs().amax(dim=1).div(127).clamp(min=1e-12)
        weights = torch.fake_quantize_per_channel_affine(
            self.weight,
            scales,
            torch.zeros_like(scales, dtype=torch.int32),
            0,
            -127,
            127,
        )
        return torch.nn.functional.linear(values, weights, self.bias)


def quantization_aware(model):
    """Keep state_dict keys unchanged and exclude the training-only value head."""
    result = copy.deepcopy(model)
    for name in DEPLOY_LAYERS:
        layer = result.get_submodule(name)
        replacement = QuantizedLinear(layer.in_features, layer.out_features)
        replacement.load_state_dict(layer.state_dict())
        parent, key = name.rsplit(".", 1) if "." in name else ("", name)
        setattr(result.get_submodule(parent), key, replacement)
    for parameter in result.value.parameters():
        parameter.requires_grad_(False)
    return result


def policy_kl(reference, student):
    # Padded candidates have zero probability and -inf log probability. Replace
    # both padded logits before subtracting, so no 0 * NaN reaches backprop.
    mask = reference.probs > 0
    delta = reference.logits.masked_fill(~mask, 0) - student.logits.masked_fill(
        ~mask, 0
    )
    return (reference.probs * delta).sum(dim=1).mean()


def distribution_metrics(reference, student, observations, batch_size=128):
    kl = []
    variation = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            batch = observations[start : start + batch_size]
            a, _ = batch_distribution(reference, batch)
            b, _ = batch_distribution(student, batch)
            mask = a.probs > 0
            delta = a.logits.masked_fill(~mask, 0) - b.logits.masked_fill(~mask, 0)
            kl.extend((a.probs * delta).sum(1).tolist())
            variation.extend(((a.probs - b.probs).abs().sum(1) / 2).tolist())
    if not kl or not np.isfinite(kl).all() or not np.isfinite(variation).all():
        raise ValueError("nonfinite or empty quantization assessment")
    return {
        "observations": len(kl),
        "mean_kl": float(np.mean(kl)),
        "mean_total_variation": float(np.mean(variation)),
        "p95_total_variation": float(np.quantile(variation, 0.95)),
        "max_total_variation": max(variation),
    }
