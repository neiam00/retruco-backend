"""Behavior counts use only the acting seat's authorized, unrefuted claims."""

from .cooperation_population import _partner_claims
from .population import SIGNALS


def redundant_signal(observation, action):
    """Whether a truthful outgoing claim repeats information visible to sender."""
    if observation["kind"] != "signal" or action == 0:
        return False

    def expand(kind):
        return {"perico", "perica"} if kind == "receta" else {kind}

    previous = set()
    for signal in observation.get("signals", []):
        if signal["seat"] == observation["seat"]:
            previous.update(expand(signal["kind"]))
    return expand(SIGNALS[action]) <= previous


class DecisionContexts:
    def __init__(self):
        self.counts = {
            name: {
                "game_choices": 0,
                "pending": 0,
                "accepts": 0,
                "declines": 0,
                "response_opportunities": 0,
                "response_accepts": 0,
                "response_declines": 0,
                "response_other": 0,
            }
            for name in ("partner_claim", "no_partner_claim")
        }
        self.by_family = {}

    def observe(self, observation, action):
        if observation["kind"] != "game":
            return
        claims = _partner_claims(observation)
        bucket = self.counts["partner_claim" if claims else "no_partner_claim"]
        bucket["game_choices"] += int(len(observation["legal"]) > 1)
        kinds = {
            a.get("kind") for a in observation["legal"] if a.get("type") == "respond"
        }
        if {"accept", "decline"} <= kinds:
            kind = observation["legal"][action].get("kind")
            bucket["response_opportunities"] += 1
            bucket["response_accepts"] += int(kind == "accept")
            bucket["response_declines"] += int(kind == "decline")
            bucket["response_other"] += int(kind not in ("accept", "decline"))
            pending = observation["view"].get("pending") or {}
            key = "/".join(
                map(
                    str,
                    (
                        observation["view"]["config"]["players"],
                        pending.get("kind", "unknown"),
                        pending.get("value", 0),
                        bool(pending.get("terminal")),
                        bool(claims),
                    ),
                )
            )
            family = self.by_family.setdefault(
                key, dict.fromkeys(("opportunities", "accepts", "declines", "other"), 0)
            )
            family["opportunities"] += 1
            family["accepts"] += int(kind == "accept")
            family["declines"] += int(kind == "decline")
            family["other"] += int(kind not in ("accept", "decline"))
        if observation["view"].get("pending"):
            bucket["pending"] += 1
            kind = observation["legal"][action].get("kind")
            bucket["accepts"] += int(kind == "accept")
            bucket["declines"] += int(kind == "decline")
