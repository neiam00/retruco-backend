"""Version-1 candidate features from authorized encoder-v2 inputs only.

The public lead is the unique highest visible, unburned, unconcealed card in
the current trick, not a prediction of the eventual winner. No hidden cards,
controller identities or private views are consumed.
"""

import torch

FEATURE_VERSION = 1
FEATURE_NAMES = (
    "card_action",
    "card_strength",
    "own_min_strength",
    "own_max_strength",
    "above_own_min",
    "below_own_max",
    "stack_top_strength",
    "round",
    "partner_remaining",
    "opponents_remaining_mean",
    "visible_max_strength",
    "partner_visible_strength",
    "partner_unique_visible_lead",
    "beats_visible_max",
    "ties_visible_max",
    "own_remaining_fraction",
)


def contextual_card_features(features, actions):
    if features.shape != (*actions.shape[:-1], 768) or actions.shape[-1] != 48:
        raise ValueError("unaligned team candidate inputs")
    shape = actions.shape[:-1]
    f, a = features.reshape(-1, 768), actions.reshape(-1, 48)
    own = f[:, 21:36].reshape(-1, 3, 5)
    available = (own[:, :, 0] == 1) & (own[:, :, 4] == 0)
    strengths = own[:, :, 3]
    index = (a[:, 32] * 2).round().long().clamp(0, 2)
    selected = strengths.gather(1, index[:, None]).squeeze(1)
    top = strengths.gather(
        1, (a[:, 33] * 2).round().long().clamp(0, 2)[:, None]
    ).squeeze(1)
    low = strengths.masked_fill(~available, torch.inf).amin(1)
    high = strengths.masked_fill(~available, -torch.inf).amax(1)
    low = torch.where(available.any(1), low, torch.zeros_like(low))
    high = torch.where(available.any(1), high, torch.zeros_like(high))
    seats = f[:, 39:127].reshape(-1, 4, 22)
    plays = seats[:, :, 1:].reshape(-1, 4, 3, 7)
    trick = (f[:, 4] * 2).round().long().clamp(0, 2)
    current = plays.gather(2, trick[:, None, None, None].expand(-1, 4, 1, 7)).squeeze(2)
    visible = (
        (current[:, :, 0] == 1)
        & (current[:, :, 4] == 1)
        & (current[:, :, 5] == 0)
        & (current[:, :, 6] == 0)
    )
    ranks = current[:, :, 3].masked_fill(~visible, -1)
    maximum = ranks.amax(1)
    any_visible = visible.any(1)
    partner = torch.where(visible[:, 2], ranks[:, 2], torch.zeros_like(maximum))
    lead = (
        visible[:, 2]
        & (ranks[:, 2] == maximum)
        & ((ranks == maximum[:, None]).sum(1) == 1)
    )
    visible_max = torch.where(any_visible, maximum, torch.zeros_like(maximum))
    card_action = (a[:, 1:4] == 1).any(1)
    result = torch.stack(
        (
            card_action.to(f.dtype),
            selected,
            low,
            high,
            selected - low,
            high - selected,
            torch.where(a[:, 2] == 1, top, torch.zeros_like(top)),
            f[:, 4],
            seats[:, 2, 0],
            (seats[:, 1, 0] + seats[:, 3, 0]) / 2,
            visible_max,
            partner,
            lead.to(f.dtype),
            (any_visible & (selected > maximum)).to(f.dtype),
            (any_visible & (selected == maximum)).to(f.dtype),
            available.sum(1).to(f.dtype) / 3,
        ),
        -1,
    )
    active = (
        (f[:, 0] == 1) & card_action & available.gather(1, index[:, None]).squeeze(1)
    )
    return torch.where(active[:, None], result, torch.zeros_like(result)).reshape(
        *shape, 16
    )
