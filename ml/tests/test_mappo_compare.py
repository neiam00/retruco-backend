import unittest

import numpy as np

from retruco_ml.mappo_compare import contrast


class MappoComparisonTests(unittest.TestCase):
    def test_more_deals_do_not_hide_variation_between_training_seeds(self):
        values = np.repeat(np.array([[-0.2], [0], [0.2]]), 128, axis=1)
        report = contrast(values)
        self.assertAlmostEqual(report["difference"], 0)
        self.assertLessEqual(report["ci95"][0], -0.19)
        self.assertGreaterEqual(report["ci95"][1], 0.19)
        np.testing.assert_allclose(report["per_training_seed"], [-0.2, 0, 0.2])
        with self.assertRaises(ValueError):
            contrast(values[:2])
        values[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            contrast(values)
