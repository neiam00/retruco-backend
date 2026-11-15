"""Held-out scripted policies. None is sampled by the training Population."""

import random

from .population import STYLES, scripted, truthful_signals

UNSEEN_STYLES = ("patient", "counter-raiser")
MIXTURE_STYLES = (*STYLES[1:], *UNSEEN_STYLES)
EVALUATION_STYLES = (*STYLES, *UNSEEN_STYLES, "population")
POPULATION_SCHEDULE = "seeded-evaluation-population-v1"


def opponent_for_pair(style, seed, pair):
    if style != "population":
        return style
    # Separate from action/deal/visibility RNGs. Both seat-swapped matches use
    # the same style and that style remains fixed for the complete match.
    return random.Random((seed + pair) ^ 0x51EC7A).choice(MIXTURE_STYLES)


def evaluation_scripted(observation, style, rng):
    if style in STYLES:
        return scripted(observation, style, rng)
    if style not in UNSEEN_STYLES:
        raise ValueError("unknown evaluation opponent")
    if observation["kind"] == "signal":
        truth = truthful_signals(observation["view"])
        return (
            rng.choice(truth)
            if style == "patient" and truth and rng.random() < 0.5
            else 0
        )
    legal = observation["legal"]
    baseline = observation["baseline"]
    if legal[0]["type"] == "cut":
        return 0
    raises = [i for i, a in enumerate(legal) if a.get("kind") == "raise"]
    accepts = [i for i, a in enumerate(legal) if a.get("kind") == "accept"]
    plays = [i for i, a in enumerate(legal) if a["type"] == "play"]
    if style == "counter-raiser":
        if raises:
            return max(raises, key=lambda i: legal[i].get("amount", 0))
        if accepts:
            return accepts[0]
        # Challenge initiated bids, but do not initiate voluntary bids.
        if legal[baseline]["type"] == "call" and plays:
            return plays[0]
    if style == "patient" and observation["view"]["round"] < 2:
        # Preserve the native response/card choices, withholding early calls.
        if legal[baseline]["type"] == "call" and plays:
            return plays[0]
    return baseline
