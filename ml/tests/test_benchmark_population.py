"""Training exposure and unbiased terminal-credit contracts."""

import copy
import random
import unittest
from collections import Counter
from types import SimpleNamespace

import torch

from retruco_ml.benchmark_population import NativePracticePopulation
from retruco_ml.cooperation_practice import CooperationPracticePopulation
from retruco_ml.improvement_train import credit_for
from retruco_ml.recurrent import Actor


class BenchmarkPopulationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(2)

    def test_native_sampling_changes_only_opponents_and_preserves_frozen_partner(self):
        actor = Actor(True, memory_size=64)
        base = CooperationPracticePopulation(
            partner_actor=actor, learned_controllers=("frozen:D101", "frozen:D102")
        )
        mixed = NativePracticePopulation(
            partner_actor=actor,
            learned_controllers=("frozen:D101", "frozen:D102"),
            native_fraction=0.5,
        )
        zero = NativePracticePopulation(
            partner_actor=actor,
            learned_controllers=("frozen:D101", "frozen:D102"),
            native_fraction=0.0,
        )
        for pop in (base, mixed, zero):
            pop.add_actor(actor, 0)
        counts = Counter()
        for seed in range(4000):
            for players in (2, 4):
                a, b = random.Random(seed), random.Random(seed)
                original = base.episode(players, a)
                self.assertEqual(original, zero.episode(players, b))
                self.assertEqual(a.getstate(), b.getstate())
                changed = mixed.episode(players, random.Random(seed))
                self.assertEqual(changed.learner, original.learner)
                for seat in range(players):
                    if seat % 2 == original.learner % 2:
                        self.assertEqual(
                            (original.controllers[seat], original.roles[seat]),
                            (changed.controllers[seat], changed.roles[seat]),
                        )
                opponents = [
                    changed.controllers[s]
                    for s in range(players)
                    if s % 2 != original.learner % 2
                ]
                counts["native"] += all(x == "script:baseline" for x in opponents)
                counts["league"] += all(x.startswith("league:") for x in opponents)
        self.assertTrue(0.47 < counts["native"] / 8000 < 0.53)
        self.assertTrue(0.39 < counts["league"] / 8000 < 0.46)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            zero.load_state_dict(mixed.state_dict())

    def test_lambda_one_uses_terminal_outcome_even_with_bad_critic(self):
        for reward in (-1, 1):
            rng = random.Random(77)
            records = [
                SimpleNamespace(
                    value=rng.uniform(-4, 4),
                    observation={"kind": "game", "actions": [[], []]},
                    target=None,
                    advantage=None,
                )
                for _ in range(500)
            ]
            # A forced event is excluded from the learning clock.
            records.insert(
                10,
                SimpleNamespace(
                    value=100.0,
                    observation={
                        "kind": "signal",
                        "signalMask": [True, False, False, False, False, False],
                    },
                    target=None,
                    advantage=None,
                ),
            )
            chosen = credit_for(1.0)(records, reward)
            self.assertEqual(len(chosen), 500)
            for r in chosen:
                self.assertAlmostEqual(r.target, reward, places=10)
                self.assertAlmostEqual(r.advantage, reward - r.value, places=10)
                self.assertEqual(r.terminal_reward, reward)
            legacy = copy.deepcopy(records)
            credit_for(0.99)(legacy, reward)
            self.assertNotAlmostEqual(legacy[0].target, reward, places=3)


if __name__ == "__main__":
    unittest.main()
