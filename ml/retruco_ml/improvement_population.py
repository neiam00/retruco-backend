"""Frozen-mixture controls and bounded, public-information reactive opponents.

Controller strings carry match-stable parameters; no profile identity is added
to actor observations. These are synthetic human proxies, not human evidence.
"""

import json
import math

from .cooperation_population import (
    RECIPE,
    CooperationPopulation,
    _partner_claims,
    _position,
    cooperation_scripted,
)

REACTIVE_EVALUATION_STYLES = ("reactive-heldout-v1", "reactive-resistant-v1")


def reactive_style(rng, heldout=False):
    """Encode one profile once per match, including a resistant subpopulation."""
    if heldout:
        rule, sensitivity = "corroborated", rng.uniform(1.05, 1.35)
    else:
        draw = rng.random()
        rule = (
            "resistant" if draw < 0.2 else "skeptical" if draw < 0.4 else "responsive"
        )
        sensitivity = 0.0 if rule == "resistant" else rng.uniform(0.4, 0.95)
    return f"reactive-v1:{rule}:{sensitivity:.6f}:{rng.uniform(0.54, 0.70):.6f}:{rng.uniform(0.35, 0.75):.6f}"


class ImprovementPopulation(CooperationPopulation):
    def __init__(
        self,
        snapshots=None,
        fixed_opponent=None,
        learned_controllers=(),
        reactive=False,
    ):
        if reactive and fixed_opponent is not None:
            raise ValueError("a fixed opponent cannot also use the reactive mixture")
        super().__init__(snapshots, fixed_opponent, learned_controllers)
        self.reactive = bool(reactive)
        self.recipe = json.dumps(
            {
                "version": "improvement-population-v1",
                "base": RECIPE,
                "reactive_fraction": 0.2 if reactive else 0.0,
                "retained_base_fraction": 0.8 if reactive else 1.0,
                "reactive_role": "human",
                "profile": "public-confidence-v1",
                "training_sensitivity": [0.4, 0.95],
                "resistant_fraction": 0.2,
                "skeptical_fraction": 0.2,
                "maximum_raise": 6,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def episode(self, players, rng):
        episode = super().episode(players, rng)
        # The control takes precisely the original RNG path. Partner assignment
        # and its authority are inherited before any opponent substitution.
        if self.reactive and rng.random() < 0.2:
            controller = "script:" + reactive_style(rng)
            for seat in range(players):
                if seat % 2 != episode.learner % 2:
                    episode.controllers[seat] = controller
                    episode.roles[seat] = False
        return episode


def public_opponent_plays(observation):
    """Visible (round, strength) pairs, using the shared Go encoder's ranks."""
    view, features, seat = (
        observation["view"],
        observation["features"],
        observation["seat"],
    )
    players = view["config"]["players"]
    result = []
    for other in view.get("seats", []):
        if other["seat"] % 2 == seat % 2:
            continue
        offset = (other["seat"] - seat) % players
        for round_index, play in enumerate(other.get("plays", [])[:3]):
            if (
                play.get("present")
                and play.get("card")
                and not play.get("burned")
                and not play.get("concealed")
            ):
                result.append(
                    (round_index, float(features[43 + 22 * offset + 7 * round_index]))
                )
    return result


def _profile(style):
    if style == "reactive-heldout-v1":
        return "corroborated", 1.2, 0.62, 0.6
    if style == "reactive-resistant-v1":
        return "resistant", 0.0, 0.62, 0.6
    parts = style.split(":")
    if len(parts) != 5 or parts[0] != "reactive-v1":
        raise ValueError("invalid reactive controller")
    rule = parts[1]
    sensitivity, threshold, initiative = map(float, parts[2:])
    if (
        rule not in ("responsive", "skeptical", "resistant", "corroborated")
        or not all(math.isfinite(x) for x in (sensitivity, threshold, initiative))
        or not 0 <= sensitivity <= 1.35
        or not 0.4 <= threshold <= 0.85
        or not 0 <= initiative <= 1
        or (rule == "resistant" and sensitivity != 0)
    ):
        raise ValueError("invalid reactive profile parameters")
    return rule, sensitivity, threshold, initiative


def improvement_scripted(observation, style, rng):
    if not style.startswith("reactive-"):
        return cooperation_scripted(observation, style, rng)
    rule, sensitivity, threshold, initiative = _profile(style)
    # Signals retain their named truthful semantics for every reactive profile.
    fallback = cooperation_scripted(observation, "literal-v2", rng)
    if observation["kind"] == "signal":
        mask = observation.get("signalMask") or observation.get("truthMask")
        return fallback if mask is None or mask[fallback] else 0
    view, legal = observation["view"], observation["legal"]
    if view.get("phase") != "playing" or legal[0]["type"] == "cut":
        return fallback
    own, _, teammate_winning, _ = _position(observation)
    strengths = sorted(own.values(), reverse=True)
    if not strengths:
        return fallback
    visible = public_opponent_plays(observation)
    weak = [max(0.0, (0.5 - strength) / 0.5) for _, strength in visible]
    evidence = 0.35 * sum(weak) / len(weak) if weak else 0.0
    if any(strength >= 12 / 17 for _, strength in visible):
        evidence = min(evidence, 0.05)
    if rule == "skeptical":
        evidence = min(evidence, 0.12 if view.get("round", 0) == 0 else 0.05)
    elif rule == "corroborated" and sum(strength < 0.5 for _, strength in visible) < 2:
        evidence = 0.0
    second = strengths[1] if len(strengths) > 1 else 0.0
    confidence = 0.2 + 0.45 * strengths[0] + 0.25 * second
    confidence += (
        0.06 * min(2, len(_partner_claims(observation))) + 0.06 * teammate_winning
    )
    confidence += sensitivity * evidence
    scores = view.get("scores", [0, 0])
    team = observation["seat"] % 2
    target = max(1, view["config"].get("target", 24))
    pressure = max(-1.0, min(1.0, (scores[1 - team] - scores[team]) / target))
    threshold -= 0.06 * pressure
    pending = view.get("pending")
    if pending:
        if pending.get("kind") != "truco":
            return fallback
        stake = pending.get("value", 3)
        willing = confidence >= threshold + 0.09 * max(0, (stake - 3) / 3)
        # Weak visible play alone never makes a very weak own holding callable.
        willing = willing and strengths[0] >= 0.4
        desired = "accept" if willing else "decline"
        options = [
            i
            for i, action in enumerate(legal)
            if action["type"] == "respond" and action.get("kind") == desired
        ]
        if options:
            return options[0]
        return fallback
    bids = [
        i
        for i, action in enumerate(legal)
        if action["type"] == "call"
        and action.get("kind") == "truco"
        and 0 < action.get("amount", 0) <= 6
        and not action.get("declaration")
        and not action.get("mode")
    ]
    if bids and strengths[0] >= 0.4 and confidence >= threshold:
        bid = min(bids, key=lambda i: legal[i]["amount"])
        if legal[bid]["amount"] <= 3 or (
            sum(x >= 12 / 17 for x in strengths) >= 2 and confidence >= threshold + 0.18
        ):
            probability = initiative * min(1.0, max(0.0, (confidence - 0.4) / 0.6))
            if rng.random() < probability:
                return bid
    # Do not let the inherited policy bypass this profile's wager rule/cap.
    if legal[fallback]["type"] == "call" and legal[fallback].get("kind") == "truco":
        alternatives = [
            i
            for i, action in enumerate(legal)
            if action["type"] == "play" and not action.get("declaration")
        ]
        if alternatives:
            return max(
                alternatives, key=lambda i: own.get(legal[i].get("card", 0), -1.0)
            )
    return fallback
