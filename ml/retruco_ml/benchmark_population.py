"""Explicit benchmark exposure while retaining frozen historical opponents."""

import json
import math

from .cooperation_practice import CooperationPracticePopulation
from .historical_league import HistoricalLeaguePopulation


class NativeOpponentMix:
    def __init__(self, *args, native_fraction=0.0, **kwargs):
        if not math.isfinite(native_fraction) or not 0 <= native_fraction <= 1:
            raise ValueError("native fraction must be finite and within [0,1]")
        super().__init__(*args, **kwargs)
        self.native_fraction = native_fraction
        self.recipe = json.dumps(
            {
                "version": "native-opponent-mixture-v1",
                "native_fraction": native_fraction,
                "remainder": json.loads(self.recipe),
                "native_controller": "script:baseline",
                "native_role": "cpu",
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def episode(self, players, rng):
        episode = super().episode(players, rng)
        if self.native_fraction and rng.random() < self.native_fraction:
            for seat in range(players):
                if seat % 2 != episode.learner % 2:
                    episode.controllers[seat] = "script:baseline"
                    episode.roles[seat] = True
        return episode


class NativeLeaguePopulation(NativeOpponentMix, HistoricalLeaguePopulation):
    pass


class NativePracticePopulation(NativeOpponentMix, CooperationPracticePopulation):
    pass
