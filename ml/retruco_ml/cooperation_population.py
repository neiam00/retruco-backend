"""Versioned synthetic cooperative conventions; legacy evidence stays unchanged."""

import json
from dataclasses import dataclass

from .model import ENCODER_VERSION
from .population import Episode, Population, scripted, truthful_signals

RECIPE = {
    "version": "cooperation-v2",
    "partners": {
        "human-v2": 0.6,
        "reference-human-role": 0.2,
        "reference-cpu-role": 0.1,
        "silent-or-erratic": 0.1,
    },
    "human_styles": {
        "literal-v2": 2,
        "trusting-v2": 2,
        "cautious-v2": 1,
        "assertive-v2": 1,
    },
    "opponents": {
        "frozen-learned": 0.4,
        "reference": 0.25,
        "history": 0.2,
        "scripted-v2": 0.15,
    },
    "missing_learned_fallback": "reference",
    "missing_history_fallback": "reference",
    "signals": "truthful remaining-card claims except erratic human proxy; silent allowed",
    "erratic_false_claim_probability": 0.2,
    "erratic_opponent_role": "human",
}
RECIPE_JSON = json.dumps(RECIPE, sort_keys=True, separators=(",", ":"))
COOPERATION_STYLES = (
    "literal-v2",
    "trusting-v2",
    "cautious-v2",
    "assertive-v2",
    "erratic-v2",
    "silent-v2",
)


@dataclass(frozen=True)
class Profile:
    signal_rate: float
    trust: bool
    envido_accept: int
    envido_terminal: int
    initiate_rate: float
    lapse_rate: float = 0


PROFILES = {
    "literal-v2": Profile(0.95, True, 28, 32, 0.35),
    "trusting-v2": Profile(0.8, True, 27, 31, 0.3),
    "cautious-v2": Profile(0.65, True, 30, 33, 0.15),
    "assertive-v2": Profile(0.85, True, 25, 29, 0.65),
    "erratic-v2": Profile(0.45, True, 27, 31, 0.3, 0.2),
    "silent-v2": Profile(0, False, 28, 32, 0.25),
}


class CooperationPopulation(Population):
    """Match-stable profiles; controller identity and response role are separate.

    The collector supplies frozen recurrent models for learned_controllers;
    inherited models(reference) supplies current/history:N/fixed legacy models.
    """

    recipe = RECIPE_JSON

    def __init__(self, snapshots=None, fixed_opponent=None, learned_controllers=()):
        super().__init__(snapshots, fixed_opponent)
        self.learned_controllers = tuple(learned_controllers)
        if any(
            not isinstance(name, str)
            or not name
            or name in ("learner", "current", "fixed")
            or name.startswith(("script:", "history:"))
            for name in self.learned_controllers
        ):
            raise ValueError(
                "learned controllers require distinct frozen registry names"
            )
        if len(set(self.learned_controllers)) != len(self.learned_controllers):
            raise ValueError("duplicate learned controller")

    def episode(self, players, rng):
        if players not in (2, 4):
            raise ValueError("cooperation population supports two or four players")
        learner = rng.randrange(players)
        draw = rng.random()
        if self.fixed_opponent is not None:
            opponent = "fixed"
        elif self.learned_controllers and draw < 0.4:
            opponent = rng.choice(self.learned_controllers)
        elif draw < 0.65:
            opponent = "current"
        elif draw < 0.85:
            opponent = self.historical(rng)
        else:
            opponent = "script:" + rng.choice(COOPERATION_STYLES)
        controllers = [opponent] * players
        controllers[learner] = "learner"
        roles = [controller != "script:erratic-v2" for controller in controllers]
        roles += [True] * (4 - players)
        if players == 4:
            partner = (learner + 2) % 4
            draw = rng.random()
            if draw < 0.6:
                style = rng.choice(
                    (
                        "literal-v2",
                        "literal-v2",
                        "trusting-v2",
                        "trusting-v2",
                        "cautious-v2",
                        "assertive-v2",
                    )
                )
                controllers[partner], roles[partner] = "script:" + style, False
            elif draw < 0.8:
                controllers[partner], roles[partner] = "current", False
            elif draw < 0.9:
                controllers[partner] = "current"
            else:
                controllers[partner] = "script:" + rng.choice(
                    ("silent-v2", "erratic-v2")
                )
                roles[partner] = False
        return Episode(controllers, roles, learner)


def _card_key(card):
    return card["suit"], card["number"]


def _partner_claims(observation):
    """Still-available partner claims, refuted only by authorized information."""
    view, seat = observation["view"], observation["seat"]
    if view["config"]["players"] != 4 or not view.get("vira"):
        return set()
    vira = view["vira"]
    cards = {
        "perico": (vira["suit"], 12 if vira["number"] == 11 else 11),
        "perica": (vira["suit"], 12 if vira["number"] == 10 else 10),
        "seven_spades": ("spades", 7),
        "seven_coins": ("coins", 7),
    }
    unavailable = {_card_key(c["card"]) for c in view.get("hand", [])}
    for other in view.get("seats", []):
        for play in other.get("plays", []):
            if (
                play.get("card")
                and not play.get("burned")
                and not play.get("concealed")
            ):
                unavailable.add(_card_key(play["card"]))
    claims = set()
    for signal in observation.get("signals", []):
        if signal["seat"] != (seat + 2) % 4:
            continue
        kinds = (
            ("perico", "perica") if signal["kind"] == "receta" else (signal["kind"],)
        )
        claims.update(
            kind for kind in kinds if kind in cards and cards[kind] not in unavailable
        )
    return claims


def _position(observation):
    # Authoritative encoder-v2 strengths: own slot 21+5*i, seats 39+22*offset.
    # Strength is the fourth card component, normalized by 17. No rank encoder.
    if (
        observation.get("version") != ENCODER_VERSION
        or len(observation.get("features", [])) != 768
    ):
        raise ValueError("cooperative proxy requires authorized encoder-v2 features")
    view, features = observation["view"], observation["features"]
    own = {
        c["index"]: features[24 + 5 * c["index"]] for c in view["hand"] if not c["used"]
    }
    best, winning_teams = 0.0, set()
    partner_pending = False
    for other in view["seats"]:
        play = other["plays"][view["round"]]
        if other["seat"] == (observation["seat"] + 2) % 4:
            partner_pending = not play.get("present", False)
        if not play.get("card") or play.get("burned") or play.get("concealed"):
            continue
        offset = (other["seat"] - observation["seat"]) % view["config"]["players"]
        strength = features[43 + 22 * offset + 7 * view["round"]]
        if strength > best:
            best, winning_teams = strength, {other["team"]}
        elif strength == best:
            winning_teams.add(other["team"])
    return own, best, winning_teams == {observation["seat"] % 2}, partner_pending


def cooperation_scripted(observation, style, rng, profile=None):
    if (
        observation["kind"] == "signal"
        and observation["view"].get("config", {}).get("players") != 4
    ):
        return 0
    if style in ("literal-v3", "trusting-v3", "cautious-v3"):
        if profile is not None:
            raise ValueError("contextual profiles have a versioned fixed definition")
        from .contextual_partners import contextual_scripted

        return contextual_scripted(observation, style, rng)
    if style not in PROFILES:
        return scripted(observation, style, rng)
    profile, view = profile or PROFILES[style], observation["view"]
    if observation["kind"] == "signal":
        truth = truthful_signals(view)
        if rng.random() >= profile.signal_rate:
            return 0
        # This profile represents an imperfect human, never the deployed CPU.
        # CPU-role seats remain protected by the simulator's truthful mask.
        if style == "erratic-v2" and rng.random() < 0.2:
            false_claims = [kind for kind in range(1, 6) if kind not in truth]
            return rng.choice(false_claims) if false_claims else 0
        if not truth:
            return 0
        return 5 if 5 in truth else rng.choice(truth)
    legal, baseline = observation["legal"], observation["baseline"]
    if legal[0]["type"] == "cut":
        return 0
    if view.get("phase") != "playing":
        return baseline
    own, best, teammate_winning, partner_pending = _position(observation)
    if profile.lapse_rate and rng.random() < profile.lapse_rate:
        return baseline
    claims = _partner_claims(observation) if profile.trust else set()
    strong = sum(value >= 12 / 17 - 1e-6 for value in own.values())
    team_strong = strong + len(claims)
    pending = view.get("pending")
    accepts = [
        i
        for i, action in enumerate(legal)
        if action["type"] == "respond" and action.get("kind") == "accept"
    ]
    if pending and accepts:
        if pending["kind"] == "truco":
            needed = 2 if pending.get("value", 3) >= 9 else 1
            willing = team_strong >= needed or teammate_winning
        elif pending["kind"] in ("envido", "ley_envido", "flor"):
            # Strength claims do not establish a teammate's point total.
            holding = view.get("holding") or {}
            threshold = (
                profile.envido_terminal
                if pending.get("terminal")
                else profile.envido_accept
            )
            value = (
                holding.get("flor", 0)
                if pending["kind"] == "flor"
                else holding.get("envido", 0)
            )
            willing = value >= threshold
        else:
            return baseline  # Preserve baseline privo handling.
        return accepts[0] if willing else baseline
    if not pending and team_strong >= 2:
        bids = [
            i
            for i, action in enumerate(legal)
            if action["type"] == "call"
            and action.get("kind") == "truco"
            and action.get("amount") == 3
            and not action.get("declaration")
            and not action.get("mode")
        ]
        if bids and rng.random() < profile.initiate_rate:
            return bids[0]
    # Preserve baseline declarations, stacking, and non-play decisions.
    if legal[baseline]["type"] != "play" or legal[baseline].get("declaration"):
        return baseline
    plays = [
        i
        for i, action in enumerate(legal)
        if action["type"] == "play"
        and not action.get("declaration")
        and action.get("card", 0) in own
    ]
    if not plays:
        return baseline
    # Interpret the fixed signal vocabulary, not arbitrary hidden card ranks.
    claim_ranks = {"perico": 17, "perica": 16, "seven_spades": 13, "seven_coins": 12}
    claimed_best = max((claim_ranks[c] / 17 for c in claims), default=0.0)
    if teammate_winning or (partner_pending and claimed_best > best + 1e-6):
        return min(plays, key=lambda i: own[legal[i].get("card", 0)])
    return baseline
