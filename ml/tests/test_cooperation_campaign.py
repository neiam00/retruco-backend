import unittest

from retruco_ml.cooperation_campaign import assess


def report(value, headsup=0.5):
    return {
        "seed": 188000,
        "configurations": [
            {
                "players": players,
                "target": target,
                "flor": flor,
                "pairs": 32,
                "pair_scores": [headsup if players == 2 else value] * 32,
                "win_rate": headsup if players == 2 else value,
            }
            for players in (2, 4)
            for target in (12, 24)
            for flor in (False, True)
        ],
    }


class CooperationSelectionTests(unittest.TestCase):
    def test_aggregate_gain_cannot_hide_literal_or_headsup_regression(self):
        config = {
            "selection_partners": [
                "reference-human",
                "literal-v2",
                "trusting-v2",
                "cautious-v2",
            ],
            "nonregression_tolerance": 0.03,
        }
        baseline = {
            partner: report(0.5)
            for partner in [*config["selection_partners"], "reference-cpu"]
        }
        candidate = {partner: report(1.0) for partner in baseline}
        candidate["literal-v2"] = report(0.0)
        score = assess(candidate, baseline, config)
        self.assertGreater(score["macro"], 0.5)
        self.assertFalse(score["nonregression_pass"])
        candidate["literal-v2"] = report(1.0)
        candidate["reference-cpu"] = report(1.0, headsup=0.0)
        self.assertFalse(assess(candidate, baseline, config)["nonregression_pass"])
        candidate["reference-cpu"] = report(1.0)
        self.assertTrue(assess(candidate, baseline, config)["nonregression_pass"])
        self.assertEqual(assess(baseline, baseline, config)["macro"], 0.5)
