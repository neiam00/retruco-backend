import random
import unittest

import torch

from retruco_ml.model import ENCODER_VERSION, Policy, batch_distribution
from retruco_ml.population import Population, scripted, truthful_signals
from retruco_ml.rollout import SimulatorPool, collect


class PopulationTests(unittest.TestCase):
    def test_exploiter_training_faces_only_the_frozen_target(self):
        target = Policy()
        population = Population(fixed_opponent=target.state_dict())
        rng = random.Random(9)
        for _ in range(100):
            episode = population.episode(4, rng)
            for seat, controller in enumerate(episode.controllers):
                if seat % 2 != episode.learner % 2:
                    self.assertEqual(controller, "fixed")
        learner = Policy()
        fixed = population.models(learner)["fixed"]
        with torch.no_grad():
            learner.state[0].weight.add_(1)
        torch.testing.assert_close(fixed.state[0].weight, target.state[0].weight)

    def test_human_partners_own_cantos_but_learner_role_stays_cpu(self):
        rng = random.Random(19)
        population = Population()
        episodes = [population.episode(4, rng) for _ in range(1000)]
        self.assertTrue(any(not e.roles[(e.learner + 2) % 4] for e in episodes))
        self.assertTrue(
            any(
                e.controllers[(e.learner + 1) % 4].startswith("script:")
                for e in episodes
            )
        )
        for e in episodes:
            self.assertEqual(e.controllers.count("learner"), 1)
            self.assertTrue(e.roles[e.learner])
            if not e.roles[(e.learner + 2) % 4]:
                self.assertTrue(
                    e.controllers[(e.learner + 2) % 4].startswith("script:")
                )

    def test_literal_signals_use_only_remaining_authorized_cards(self):
        view = {
            "config": {"players": 4},
            "vira": {"suit": "coins", "number": 11},
            "hand": [
                {"card": {"suit": "coins", "number": 12}, "used": False},
                {"card": {"suit": "coins", "number": 10}, "used": True},
                {"card": {"suit": "spades", "number": 7}, "used": False},
            ],
        }
        self.assertEqual(truthful_signals(view), [1, 3])
        observation = {"view": view, "kind": "signal"}
        rng = random.Random(1)
        self.assertTrue(
            {scripted(observation, "literal", rng) for _ in range(100)} <= {0, 1, 3}
        )

    def test_ragged_inference_never_assigns_probability_to_padding(self):
        torch.manual_seed(3)
        observations = [
            {
                "version": ENCODER_VERSION,
                "features": [0.0] * 768,
                "kind": "game",
                "actions": [[0.0] * 48] * n,
            }
            for n in [1, 3, 12]
        ]
        with torch.no_grad():
            distribution, _ = batch_distribution(Policy(), observations)
        for row, n in enumerate([1, 3, 12]):
            self.assertEqual(float(distribution.probs[row, n:].sum()), 0)
            self.assertAlmostEqual(
                float(distribution.probs[row, :n].sum()), 1, places=6
            )

    def test_parallel_collection_completes_and_is_seed_reproducible(self):
        torch.set_num_threads(2)
        torch.manual_seed(29)
        policy = Policy()
        population = Population()
        with SimulatorPool(2) as pool:
            torch.manual_seed(39)
            first = collect(policy, pool, population, random.Random(51), 32, 4, 4)
            torch.manual_seed(39)
            second = collect(policy, pool, population, random.Random(51), 32, 4, 4)
        self.assertEqual(first, second)
        records, matches, counts = first
        self.assertGreaterEqual(len(records), 32)
        self.assertGreater(matches, 0)
        self.assertGreater(counts["learner"], 0)
        self.assertTrue(
            all(
                "view" not in record[0] and "controllers" not in record[0]
                for record in records
            )
        )


if __name__ == "__main__":
    unittest.main()
