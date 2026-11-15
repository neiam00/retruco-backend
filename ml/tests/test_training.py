import unittest

import torch

from retruco_ml.model import Policy
from retruco_ml.train import advantages


class TrainingTests(unittest.TestCase):
    def test_terminal_credit_stays_with_seat(self):
        records = [(None, 0, 0.0, 0.0), (None, 0, 0.0, 0.0)]
        winning = advantages(records, 1, lam=1)
        losing = advantages(records, -1, lam=1)
        self.assertEqual([r[4] for r in winning], [1, 1])
        self.assertEqual([r[4] for r in losing], [-1, -1])

    def test_candidate_permutation_equivariance(self):
        torch.manual_seed(7)
        policy = Policy()
        state = torch.randn(1, 768)
        candidates = torch.randn(1, 4, 48)
        original, _, _ = policy(state, candidates)
        permutation = [2, 0, 3, 1]
        permuted, _, _ = policy(state, candidates[:, permutation])
        torch.testing.assert_close(original[:, permutation], permuted)


if __name__ == "__main__":
    unittest.main()
