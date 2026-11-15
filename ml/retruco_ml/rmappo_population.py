"""Match-stable historical opposition and genuine joint teammate learning."""

import copy
import math
from dataclasses import dataclass


def population_config(profile):
    """Match probabilities, not learner-choice or loss-weight percentages."""
    if profile == "original":
        result = {
            "full_selfplay_probability": 0.25,
            "non_selfplay_opponents": {"native": 0.5, "historical": 0.4, "fixed": 0.1},
            "teammates": {
                "learner": 0.5,
                "frozen": 0.05,
                "reference": 0.1,
                **{s + "-v3": 0.35 / 3 for s in ("literal", "trusting", "cautious")},
            },
        }
    elif profile in ("target", "reference_focus"):
        result = {
            "full_selfplay_probability": 0.0,
            "non_selfplay_opponents": {
                "native": 0.75,
                "historical": 0.20,
                "fixed": 0.05,
            },
            "teammates": {
                "learner": 0.20,
                "frozen": 0.02,
                "reference": 0.30,
                **{
                    s + "-" + v: 0.08
                    for v in ("v2", "v3")
                    for s in ("literal", "trusting", "cautious")
                },
            },
        }
        if profile == "reference_focus":
            partners = result["teammates"]
            scale = (1 - 0.60) / (1 - partners["reference"])
            result["teammates"] = {
                name: 0.60 if name == "reference" else mass * scale
                for name, mass in partners.items()
            }
    else:
        raise ValueError("unknown RMAPPO population profile")
    if not 0 <= result["full_selfplay_probability"] <= 1:
        raise ValueError("invalid full-selfplay probability")
    for key in ("non_selfplay_opponents", "teammates"):
        weights = result[key].values()
        if any(
            not math.isfinite(x) or not 0 <= x <= 1 for x in weights
        ) or not math.isclose(math.fsum(weights), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("invalid population probabilities: " + key)
    return result


def _draw(weights, rng):
    threshold = rng.random()
    cumulative = 0.0
    for name, mass in weights.items():
        cumulative += mass
        if threshold < cumulative:
            return name
    # Accommodate the final floating-point addition only; random() is < 1.
    return next(reversed(weights))


@dataclass
class Episode:
    learner: int
    controllers: list
    roles: list


class Population:
    def __init__(self, fixed, league=None, profile="original"):
        if not fixed or "frozen:103" in fixed:
            raise ValueError("require training opponents; D103 is evaluation-only")
        self.fixed = fixed
        self.league = list(league or [])
        self.profile = profile
        self.distribution = population_config(profile)
        if profile in ("target", "reference_focus") and set(fixed) != {
            "frozen:101",
            "frozen:102",
        }:
            raise ValueError("target population requires exactly frozen D101/D102")

    def add(self, actor):
        self.league.append(copy.deepcopy(actor).eval().requires_grad_(False))
        self.league = self.league[-16:]

    def models(self, reference):
        return {
            **self.fixed,
            "reference": reference,
            **{f"league:{i}": m for i, m in enumerate(self.league)},
        }

    def episode(self, players, rng):
        if players not in (2, 4):
            raise ValueError("RMAPPO population requires two or four players")
        learner = rng.randrange(players)
        roles = [True] * 4
        if self.profile in ("target", "reference_focus"):
            if not self.league:
                raise ValueError("target population requires its historical league")
            family = _draw(self.distribution["non_selfplay_opponents"], rng)
            if family == "native":
                opponent = "script:baseline"
            elif family == "historical":
                opponent = f"league:{len(self.league) - 1 if rng.random() < 0.5 else rng.randrange(len(self.league))}"
            else:
                opponent = rng.choice(tuple(self.fixed))
            controllers = [opponent] * players
            controllers[learner] = "learner"
            if players == 4:
                friend = (learner + 2) % 4
                partner = _draw(self.distribution["teammates"], rng)
                if partner == "learner":
                    controllers[friend] = "learner"
                elif partner == "frozen":
                    controllers[friend] = rng.choice(tuple(self.fixed))
                else:
                    controllers[friend] = (
                        "reference" if partner == "reference" else "script:" + partner
                    )
                    roles[friend] = False
            return Episode(learner, controllers, roles)
        if rng.random() < 0.25:
            return Episode(learner, ["learner"] * players, roles)
        draw = rng.random()
        if draw < 0.5:
            opponent = "script:baseline"
        elif draw < 0.9 and self.league:
            opponent = f"league:{len(self.league) - 1 if rng.random() < 0.5 else rng.randrange(len(self.league))}"
        else:
            opponent = rng.choice(tuple(self.fixed))
        controllers = [opponent] * players
        controllers[learner] = "learner"
        if players == 4:
            friend = (learner + 2) % 4
            draw = rng.random()
            if draw < 0.5:
                controllers[friend] = "learner"
            elif draw < 0.85:
                controllers[friend] = "script:" + rng.choice(
                    ("literal-v3", "trusting-v3", "cautious-v3")
                )
                roles[friend] = False
            elif draw < 0.95:
                controllers[friend] = "reference"
                roles[friend] = False
            else:
                controllers[friend] = rng.choice(tuple(self.fixed))
        return Episode(learner, controllers, roles)
