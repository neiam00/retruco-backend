"""Version-three synthetic partners: a card claim is not a future action.

These are training/diagnostic alternatives, not replacements for the frozen v2
benchmark. They use only the authorized observation and preserve declarations.
"""

from .cooperation_population import (
    PROFILES,
    _partner_claims,
    _position,
    cooperation_scripted,
)

STYLES = ("literal-v3", "trusting-v3", "cautious-v3")
CLAIM_RANKS = {"perico": 17, "perica": 16, "seven_spades": 13, "seven_coins": 12}


def contextual_scripted(observation, style, rng):
    if style not in STYLES:
        raise ValueError("unknown contextual partner")
    previous = style.replace("-v3", "-v2")
    profile = PROFILES[previous]
    if observation["kind"] == "signal":
        return cooperation_scripted(observation, previous, rng)
    view, legal, baseline = (
        observation["view"],
        observation["legal"],
        observation["baseline"],
    )
    if legal[0]["type"] == "cut" or view.get("phase") != "playing":
        return baseline
    own, visible_best, teammate_winning, partner_pending = _position(observation)
    claims = _partner_claims(observation)
    claim_strengths = [CLAIM_RANKS[x] / 17 for x in claims]
    own_strong = sum(x >= 12 / 17 - 1e-6 for x in own.values())
    team_strong = own_strong + len(claim_strengths)
    pending = view.get("pending")

    # Keep independently meaningful declarations/modes intact when selecting a
    # different card or response. None of these profiles changes response roles.
    def compatible(indices):
        original = legal[baseline]
        return [
            i
            for i in indices
            if all(legal[i].get(k) == original.get(k) for k in ("declaration", "mode"))
        ]

    if pending:
        accepts = compatible(
            [
                i
                for i, a in enumerate(legal)
                if a["type"] == "respond" and a.get("kind") == "accept"
            ]
        )
        declines = compatible(
            [
                i
                for i, a in enumerate(legal)
                if a["type"] == "respond" and a.get("kind") == "decline"
            ]
        )
        if not accepts or not declines:
            return baseline
        family = pending.get("kind")
        if family == "truco":
            stake = pending.get("value", 3)
            needed = 2 if stake >= 9 else 1
            willing = team_strong >= needed or teammate_winning
            # On the deciding trick, an unplayed claim already beaten by a
            # visible opposing card cannot justify accepting on its own.
            if view["round"] == 2 and not teammate_winning:
                future = list(own.values()) + (
                    claim_strengths if partner_pending else []
                )
                willing = willing and max(future, default=0.0) >= visible_best - 1e-6
            # A raised wager needs corroboration; a single claimed card does
            # not guarantee two tricks or prescribe what the partner will play.
            if (
                stake >= 6
                and own_strong == 0
                and len(claim_strengths) < 2
                and not teammate_winning
            ):
                willing = False
        elif family in ("envido", "ley_envido", "flor"):
            holding = view.get("holding") or {}
            threshold = (
                profile.envido_terminal
                if pending.get("terminal")
                else profile.envido_accept
            )
            value = (
                holding.get("flor", 0) if family == "flor" else holding.get("envido", 0)
            )
            willing = value >= threshold
        else:
            return baseline
        return accepts[0] if willing else declines[0]
    # New bids require our own corroborating card. A claim alone does not
    # commit the teammate to winning either the current or subsequent trick.
    if own_strong and team_strong >= 2:
        bids = compatible(
            [
                i
                for i, a in enumerate(legal)
                if a["type"] == "call"
                and a.get("kind") == "truco"
                and a.get("amount") == 3
            ]
        )
        if bids and rng.random() < profile.initiate_rate:
            return bids[0]
    if legal[baseline]["type"] != "play":
        return baseline
    plays = [
        i
        for i, a in enumerate(legal)
        if a["type"] == "play"
        and a.get("card", 0) in own
        and {k: v for k, v in a.items() if k != "card"}
        == {k: v for k, v in legal[baseline].items() if k != "card"}
    ]
    if teammate_winning and plays:
        return min(plays, key=lambda i: own[legal[i].get("card", 0)])
    # In particular, do not throw away our winning card merely because a
    # teammate who has not acted has announced a strong card.
    return baseline
