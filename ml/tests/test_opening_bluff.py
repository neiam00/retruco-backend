import math
import unittest

import numpy as np
import torch

from retruco_ml.recurrent import Actor, limit_opening_bluff
from retruco_ml.recurrent_batch import candidate_distribution


class OpeningBluffTests(unittest.TestCase):
    def test_crowded_opening_cannot_turn_a_low_stakes_bluff_into_a_frequent_call(self):
        raw = torch.tensor([[90.0, 91.0, 0.0, -2.0], [-7.0, 1.0, 0.0, -2.0]])
        marked = torch.tensor([[True, True, False, False], [True, False, False, False]])
        scored = limit_opening_bluff(raw, marked)
        self.assertLess(scored[0].softmax(0)[:2].sum().item(), 1 / (1 + math.exp(3)))
        torch.testing.assert_close(scored[0, 2:], raw[0, 2:])
        torch.testing.assert_close(scored[1], raw[1])

    def test_scalar_and_packed_policies_sample_the_same_weak_opening(self):
        torch.manual_seed(17)
        actor = Actor(True, 64).eval()
        actions = np.zeros((3, 48), dtype=np.float32)
        actions[0, [4, 7]] = 1
        actions[0, 35] = 0.03
        actions[0, 47] = -1
        actions[1:, 1] = 1
        actions[2, 32] = 1
        obs = {"version": 2, "kind": "game", "actions": actions}
        quiet = {**obs, "actions": actions[1:]}
        context = torch.zeros(64)
        scalar = actor.distribution(context, obs)
        packed = candidate_distribution(actor, context.expand(2, -1), [obs, quiet])
        self.assertGreater(scalar.probs[0].item(), 0)
        self.assertLessEqual(scalar.probs[0].item(), 1 / (1 + math.exp(3)))
        torch.testing.assert_close(scalar.probs, packed.probs[0])
        torch.testing.assert_close(
            actor.distribution(context, quiet).probs, packed.probs[1, :2]
        )
