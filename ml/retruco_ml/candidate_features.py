"""Public candidate relationships in reserved action-v1 columns 36..47.

The raw Go action encoder stays unchanged. Versioned actors derive these
features solely from their authorized encoder-v2 view. Ordinary visible leads
are not guaranteed wins; certainty requires the public deciding perico proof.
"""

import torch

VERSION = 1
NAMES = (
    "card_strength",
    "above_own_min",
    "below_own_max",
    "card_minus_visible_max",
    "partner_unique_lead",
    "stack_top_strength",
    "own_remaining",
    "round",
    "public_truco_winner",
    "bet_hand_value",
    "action_stake_fraction",
    "match_margin_for_bet",
)


def augment(features, actions):
    if (
        features is None
        or features.shape != (*actions.shape[:-1], 768)
        or actions.shape[-1] != 48
    ):
        raise ValueError(
            "candidate features require aligned authorized local observations"
        )
    shape = actions.shape
    f = features.reshape(-1, 768)
    a = actions.reshape(-1, 48)
    own = f[:, 21:36].reshape(-1, 3, 5)
    available = (own[:, :, 0] == 1) & (own[:, :, 4] == 0)
    ranks = own[:, :, 3]
    index = (a[:, 32] * 2).round().long().clamp(0, 2)
    selected = ranks.gather(1, index[:, None]).squeeze(1)
    low = ranks.masked_fill(~available, torch.inf).amin(1)
    high = ranks.masked_fill(~available, -torch.inf).amax(1)
    low = torch.where(available.any(1), low, torch.zeros_like(low))
    high = torch.where(available.any(1), high, torch.zeros_like(high))
    seats = f[:, 39:127].reshape(-1, 4, 22)
    plays = seats[:, :, 1:].reshape(-1, 4, 3, 7)
    trick = (f[:, 4] * 2).round().long().clamp(0, 2)
    current = plays.gather(2, trick[:, None, None, None].expand(-1, 4, 1, 7)).squeeze(2)
    seat_mask = (
        torch.arange(4, device=f.device)[None, :]
        < (f[:, 0] * 4).round().long()[:, None]
    )
    visible = (
        (current[:, :, 0] == 1)
        & (current[:, :, 4] == 1)
        & (current[:, :, 5] == 0)
        & (current[:, :, 6] == 0)
        & seat_mask
    )
    public_ranks = current[:, :, 3].masked_fill(~visible, -1)
    maximum = public_ranks.amax(1)
    any_visible = visible.any(1)
    partner_lead = (
        (f[:, 0] == 1)
        & visible[:, 2]
        & (public_ranks[:, 2] == maximum)
        & ((public_ranks == maximum[:, None]).sum(1) == 1)
    )
    teams = f.new_tensor([1.0, -1.0, 1.0, -1.0])

    def completed(round_number):
        p = plays[:, :, round_number]
        complete = (
            (
                (p[:, :, 4] == 1)
                & (p[:, :, 6] == 0)
                & ((p[:, :, 5] == 1) | (p[:, :, 0] == 1))
            )
            | ~seat_mask
        ).all(1)
        rank = p[:, :, 3].masked_fill((p[:, :, 5] == 1) | ~seat_mask, -1)
        best = rank.amax(1)
        unique = (rank == best[:, None]).sum(1) == 1
        winner = teams[rank.argmax(1)]
        return torch.where(
            complete & unique & (best >= 0), winner, torch.zeros_like(winner)
        )

    first, second = completed(0), completed(1)
    perico = visible & (public_ranks == 1)
    perico_team = (perico.to(f.dtype) * teams[None, :]).sum(1)
    deciding = ((trick == 1) & (first == perico_team) & (first != 0)) | (
        (trick == 2) & (first != 0) & (second == -first)
    )
    proof = torch.where(
        (f[:, 3] == 0.5) & (f[:, 13] == 1) & (perico.sum(1) == 1) & deciding,
        perico_team,
        torch.zeros_like(first),
    )
    card_action = (a[:, 1:4] == 1).any(1) & available.gather(1, index[:, None]).squeeze(
        1
    )
    call = a[:, 4] == 1
    respond = a[:, 5] == 1
    bet = call | respond
    pending = f[:, 127] == 1
    pending_kind = f[:, 128]
    truco = (call & (a[:, 7] == 1)) | (
        respond & pending & torch.isclose(pending_kind, f.new_tensor(0.2))
    )
    envido = (call & ((a[:, 8] == 1) | (a[:, 9] == 1))) | (
        respond
        & pending
        & (
            torch.isclose(pending_kind, f.new_tensor(0.4))
            | torch.isclose(pending_kind, f.new_tensor(0.6))
        )
    )
    flor = (call & (a[:, 10] == 1)) | (
        respond & pending & torch.isclose(pending_kind, f.new_tensor(0.8))
    )
    decline = respond & (a[:, 12] == 1)
    # Engine units are /100, target is /24. A refusal risks the previous stake;
    # an acceptance risks the pending stake. This is descriptive, not a rule.
    stake = torch.where(respond, torch.where(decline, f[:, 131], f[:, 130]), a[:, 35])
    target = (f[:, 1] * 24).clamp_min(1)
    extra = torch.stack(
        (
            selected,
            selected - low,
            high - selected,
            torch.where(any_visible, selected - maximum, torch.zeros_like(selected)),
            partner_lead.to(f.dtype),
            torch.where(
                a[:, 2] == 1,
                ranks.gather(
                    1, (a[:, 33] * 2).round().long().clamp(0, 2)[:, None]
                ).squeeze(1),
                torch.zeros_like(selected),
            ),
            available.sum(1).to(f.dtype) / 3,
            f[:, 4],
            torch.where(truco, proof, torch.zeros_like(proof)),
            torch.where(
                envido,
                f[:, 36],
                torch.where(flor, f[:, 37], torch.zeros_like(selected)),
            ),
            torch.where(bet, stake * 100 / target, torch.zeros_like(stake)),
            torch.where(bet, f[:, 8] - f[:, 9], torch.zeros_like(selected)),
        ),
        -1,
    )
    extra = torch.cat(
        (
            torch.where(
                card_action[:, None], extra[:, :8], torch.zeros_like(extra[:, :8])
            ),
            extra[:, 8:],
        ),
        -1,
    )
    return torch.cat((a[:, :36], extra.to(a.dtype)), -1).reshape(shape)
