"""Partner-role coverage without changing the historical opponent league.

Profiles are training-only controllers, never actor features. The independent
D103 evaluation partner is deliberately absent from this population.
"""

import copy
import json

from .historical_league import HistoricalLeaguePopulation, _fingerprint


class CooperationPracticePopulation(HistoricalLeaguePopulation):
    def __init__(
        self,
        *args,
        partner_actor,
        partner_pool=None,
        partner_policy_version=2,
        **kwargs,
    ):
        if type(partner_policy_version) is not int or partner_policy_version not in (
            2,
            3,
        ):
            raise ValueError("unsupported partner policy version")
        super().__init__(*args, **kwargs)
        self.partner_pool = partner_pool
        self.partner_policy_version = partner_policy_version
        if partner_pool is not None:
            partner_pool.verify()
        self.partner_actor = copy.deepcopy(partner_actor).eval().requires_grad_(False)
        self.partner_fingerprint = _fingerprint(self.partner_actor.state_dict())
        self.recipe = json.dumps(
            {
                "version": "cooperation-practice-v1",
                "base": json.loads(self.recipe),
                "human_target_profiles": 0.45,
                "reference_human": 0.10,
                "silent_or_erratic_human": 0.10,
                "frozen_starter_cpu": 0.15,
                "literal_or_trusting_cpu": 0.15,
                "reference_cpu": 0.05,
                "partner_sha256": self.partner_fingerprint,
                "opponents": "unchanged historical league",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if partner_pool is not None or partner_policy_version != 2:
            recipe = json.loads(self.recipe)
            recipe["partner_policy_version"] = partner_policy_version
            recipe["partner_pool"] = (
                None if partner_pool is None else partner_pool.configuration
            )
            self.recipe = json.dumps(recipe, sort_keys=True, separators=(",", ":"))

    def models(self, reference):
        result = super().models(reference)
        result["practice:starter"] = copy.deepcopy(self.partner_actor)
        if self.partner_pool is not None:
            result.update(self.partner_pool.models())
        return result

    def state_dict(self):
        if self.partner_pool is not None:
            self.partner_pool.verify()
        return super().state_dict()

    def load_state_dict(self, state):
        if self.partner_pool is not None:
            self.partner_pool.verify()
        return super().load_state_dict(state)

    def episode(self, players, rng):
        episode = super().episode(players, rng)
        if players == 2:
            return episode
        partner = (episode.learner + 2) % 4
        draw = rng.random()
        if draw < 0.45:
            controller, cpu = (
                "script:" + rng.choice(("literal-v2", "trusting-v2", "cautious-v2")),
                False,
            )
        elif draw < 0.55:
            controller, cpu = "current", False
        elif draw < 0.65:
            controller, cpu = "script:" + rng.choice(("silent-v2", "erratic-v2")), False
        elif draw < 0.80:
            controller, cpu = "practice:starter", True
        elif draw < 0.95:
            controller, cpu = (
                "script:" + rng.choice(("literal-v2", "trusting-v2")),
                True,
            )
        else:
            controller, cpu = "current", True
        if controller == "practice:starter" and self.partner_pool is not None:
            controller = self.partner_pool.sample_match(rng)
        if self.partner_policy_version == 3 and controller in (
            "script:literal-v2",
            "script:trusting-v2",
            "script:cautious-v2",
        ):
            controller = controller[:-1] + "3"
        episode.controllers[partner], episode.roles[partner] = controller, cpu
        return episode
