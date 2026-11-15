"""Allow team learning only through encoder-v2 inputs that cannot exist in 1v1.

cpu/learning/encode.go uses seats relative to the observer. Seats 2 and 3
are padded in 1v1, including in every event passed to the recurrent journal.
This leaves both the runtime architecture and the old checkpoint format intact.
"""

import torch

# Seat views, match and hand canto counts, CPU roles, canto family histories,
# and delivered-signal counts for the two absent seats, respectively.
TEAM_INPUT_RANGES = (
    (83, 127),
    (368, 384),
    (400, 416),
    (478, 480),
    (592, 672),
    (694, 716),
)
INPUT_WEIGHT = "state.0.weight"
CALIBRATED_STARTER_SHA256 = (
    "e9fef3ff4598520ea066edacde748e14c9893ff7bc2939af4a07b3e6ebe5798a"
)


def team_weight_mask(starter_weight):
    if starter_weight.shape != (96, 768):
        raise ValueError("heads-up protection requires the encoder-v2 input layer")
    mask = torch.zeros_like(starter_weight, dtype=torch.bool)
    for start, end in TEAM_INPUT_RANGES:
        mask[:, start:end] = True
    # The int8 exporter derives a scale from the largest magnitude in each
    # row. Keep one original maximum and bound all editable weights to it.
    mask.scatter_(1, starter_weight.abs().argmax(dim=1, keepdim=True), False)
    return mask


def verify_protected_weights(starter, candidate):
    """Verify native and per-output int8 heads-up invariance from stored tensors."""
    if candidate.keys() != starter.keys():
        raise ValueError("protected actor architecture changed")
    for key, original in starter.items():
        value = candidate[key]
        if value.shape != original.shape or value.dtype != original.dtype:
            raise ValueError(f"protected actor tensor changed shape or dtype: {key}")
        if key != INPUT_WEIGHT:
            if not torch.equal(value, original):
                raise ValueError(f"protected actor tensor changed: {key}")
            continue
        allowed = team_weight_mask(original)
        if not torch.equal(value[~allowed], original[~allowed]):
            raise ValueError("an input weight used by the protected policy changed")
        if not torch.equal(value.abs().amax(dim=1), original.abs().amax(dim=1)):
            raise ValueError("a protected int8 row scale changed")


class HeadsUpProtection:
    """Constrain optimizer gradients and actual stored weights, including on resume."""

    def __init__(self, actor, starter_state):
        verify_protected_weights(starter_state, actor.state_dict())
        actor.requires_grad_(False)
        self.weight = actor.state[0].weight.requires_grad_(True)
        self.original = starter_state[INPUT_WEIGHT].detach().to(self.weight).clone()
        self.allowed = team_weight_mask(self.original)
        self.limit = self.original.abs().amax(dim=1, keepdim=True)
        self.weight.register_hook(lambda grad: grad.masked_fill(~self.allowed, 0))

    @torch.no_grad()
    def enforce_bounds(self):
        bounded = self.weight.clamp(min=-self.limit, max=self.limit)
        self.weight.copy_(torch.where(self.allowed, bounded, self.original))

    def optimizer(self):
        optimizer = torch.optim.Adam([self.weight], lr=3e-4, eps=1e-5)
        optimizer.register_step_post_hook(lambda *_: self.enforce_bounds())
        return optimizer
