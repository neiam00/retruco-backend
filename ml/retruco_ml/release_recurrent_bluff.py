"""Weak-opening counterfactual using the deployed action support and hand journal."""

import hashlib
import random

import numpy as np

from .evaluation_styles import evaluation_scripted
from .release_recurrent import SeatJournals
from .rollout import SimulatorPool
from .statistics import interval


def opening_raise(obs):
    return next(
        (
            i
            for i, c in enumerate(obs["legal"])
            if c["type"] == "call"
            and c.get("kind") == "truco"
            and c.get("amount") == 3
            and not c.get("mode")
            and not c.get("declaration")
        ),
        None,
    )


def diagnose(actor, simulator, pairs, seed, opponent):
    wins = []
    points = []
    information = None
    probability = None
    raw_offers = offered = 0
    with SimulatorPool(1, "binary", str(simulator)) as pool:
        for pair in range(pairs):
            outcomes = []
            first_points = []
            base = {
                "op": "reset",
                "id": 0,
                "seed": seed + pair,
                "fixture": "weak-opening-v1",
                "config": {"players": 2, "target": 12, "florRequired": False},
            }
            raw = pool.call([{**base, "id": 1}])[0]
            raw_offers += opening_raise(raw) is not None
            for branch in ("raise", "quiet"):
                journals = SeatJournals(actor)
                req = dict(base)
                journals.reset(req, [0])
                obs = pool.call([req])[0]
                if obs["seat"] != 0 or obs["choicePolicy"] != actor.profile:
                    raise ValueError("Wrong fixed opening seat or profile")
                journals.consume(pool, {0: obs})
                memory = journals.memory(0, 0)
                signature = hashlib.sha256(
                    np.asarray(obs["features"], dtype="<f4").tobytes()
                    + np.asarray(obs["actions"], dtype="<f4").tobytes()
                    + memory.tobytes()
                ).hexdigest()
                if information is None:
                    information = signature
                if signature != information:
                    raise ValueError(
                        "The sampled hidden deal altered the acting information set"
                    )
                action = opening_raise(obs)
                if action is None:
                    if offered:
                        raise ValueError("Inconsistent deployed opening support")
                    probability = 0.0
                    break
                probs = actor.probabilities(obs, memory)
                if probability is not None and not np.isclose(
                    probability, probs[action], atol=0, rtol=1e-12
                ):
                    raise ValueError("Inconsistent deployed opening probability")
                probability = float(probs[action])
                if branch == "raise":
                    offered += 1
                else:
                    quiet = [
                        i
                        for i, c in enumerate(obs["legal"])
                        if c["type"] == "play" and not c.get("declaration")
                    ]
                    action = max(quiet, key=lambda i: probs[i])
                rng = [random.Random((seed + pair) * 11 + s) for s in (0, 1)]
                first = None
                for _ in range(10000):
                    obs = pool.call(
                        [
                            {
                                "op": "step",
                                "id": 0,
                                "action": action,
                                **journals.parameters(0),
                            }
                        ]
                    )[0]
                    if first is None and (obs["done"] or obs["view"]["handNumber"] > 1):
                        first = obs["scores"][0] - obs["scores"][1]
                    if obs["done"]:
                        outcomes.append(int(obs["winner"] == 0))
                        first_points.append(first)
                        break
                    journals.consume(pool, {0: obs})
                    action = (
                        journals.choose(0, obs, rng[0])
                        if obs["seat"] == 0
                        else evaluation_scripted(obs, opponent, rng[1])
                    )
                else:
                    raise RuntimeError("Incomplete conditional-bluff match")
            if outcomes:
                if len(outcomes) != 2:
                    raise ValueError("Incomplete counterfactual pair")
                wins.append(outcomes[0] - outcomes[1])
                points.append(first_points[0] - first_points[1])
    return {
        "fixture": "weak-opening-v1",
        "opponent": opponent,
        "target": 12,
        "pairs": pairs,
        "seed": seed,
        "information_sha256": information,
        "raw_rule_offers": raw_offers,
        "deployed_offers": offered,
        "policy_raise_probability": probability,
        "paired_win_gains": wins,
        "paired_point_gains": points,
        "win_probability_gain": float(np.mean(wins)) if wins else None,
        "win_gain_ci95": interval(wins) if len(wins) > 1 else None,
        "first_hand_point_gain": float(np.mean(points)) if points else None,
        "point_gain_ci95": interval(points) if len(points) > 1 else None,
        "scope": "No forced off-policy opening when the deployed host excludes the raise.",
    }
