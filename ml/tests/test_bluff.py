import unittest

import torch

from retruco_ml.bluff import diagnose
from retruco_ml.model import Policy


class BluffTests(unittest.TestCase):
    def test_counterfactuals_hold_information_fixed_and_complete_both_branches(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        model = Policy()
        model.checkpoint_sha256 = "test-only"
        report = diagnose(model, 4, 370000, opponent="cautious")
        self.assertEqual(len(report["paired_win_gains"]), 4)
        self.assertEqual(len(report["paired_point_gains"]), 4)
        self.assertTrue(0 <= report["policy_raise_probability"] <= 1)
        self.assertFalse(report["approved"])
        self.assertGreater(
            report["first_hand_point_gain"],
            0,
            "a cautious opponent folds to the forced raise",
        )


if __name__ == "__main__":
    unittest.main()
