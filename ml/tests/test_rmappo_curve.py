import copy
import json
import tempfile
import unittest
from pathlib import Path

from retruco_ml.rmappo_curve import (
    acceptance,
    metric,
    paired_delta,
    receipt,
    stream_inventory,
    verify_receipts,
)


def report(scores, *, families=None):
    return {
        "seed": 149000,
        "pairs": len(scores),
        "partner": "literal-v2",
        "opponent": "native",
        "signals": "normal",
        "card_policy": "policy",
        "configurations": [
            {
                "players": 4,
                "target": 12,
                "flor": False,
                "pairs": len(scores),
                "pair_scores": scores,
            }
        ],
        "response_families": families or {},
    }


class CheckpointCurveTests(unittest.TestCase):
    def test_shared_deals_remain_two_clusters_across_three_training_seeds(self):
        candidates = [report([1.0, 0.0]), report([0.0, 1.0]), report([0.5, 0.5])]
        references = [report([0.25, 0.25]) for _ in candidates]
        result = metric(candidates, references)
        self.assertEqual(result["deal_clusters"], 2)
        self.assertEqual(result["win_rate"], 0.5)
        self.assertEqual(result["cluster_ci95"], [0.5, 0.5])
        delta = result["paired_champion_delta"]
        self.assertEqual(delta["delta"], 0.25)
        self.assertEqual(delta["cluster_ci95"], [0.25, 0.25])
        self.assertEqual(delta["deal_clusters"], 2)

    def test_pairing_rejects_different_stream_opponent_or_configuration(self):
        reference = report([0.0, 1.0])
        for field, value in [
            ("seed", 147000),
            ("opponent", "parent"),
            ("partner", "independent-partner"),
            ("pairs", 3),
        ]:
            candidate = copy.deepcopy(reference)
            candidate[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                paired_delta([candidate], [reference])
        candidate = copy.deepcopy(reference)
        candidate["configurations"][0]["flor"] = True
        with self.assertRaises(ValueError):
            paired_delta([candidate], [reference])

    def test_acceptance_weights_opportunities_and_separates_canto_and_claim(self):
        first = report(
            [0.5, 0.5],
            families={
                "4/truco/3/False/True": {
                    "opportunities": 2,
                    "accepts": 2,
                    "declines": 0,
                    "other": 0,
                },
                "4/truco/3/False/False": {
                    "opportunities": 4,
                    "accepts": 3,
                    "declines": 1,
                    "other": 0,
                },
                "4/envido/2/False/True": {
                    "opportunities": 5,
                    "accepts": 1,
                    "declines": 3,
                    "other": 1,
                },
            },
        )
        second = report(
            [0.5, 0.5],
            families={
                "4/truco/6/False/True": {
                    "opportunities": 8,
                    "accepts": 0,
                    "declines": 7,
                    "other": 1,
                },
            },
        )
        families = acceptance([first, second])["families"]
        self.assertEqual(families["4/truco/True"]["opportunities"], 10)
        self.assertEqual(families["4/truco/True"]["acceptance_rate"], 0.2)
        self.assertEqual(families["4/truco/False"]["acceptance_rate"], 0.75)
        self.assertEqual(families["4/envido/True"]["decline_rate"], 0.6)

    def test_inventory_rejects_partial_overlap_with_previous_development_deals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prior.json").write_text(json.dumps({"seed": 148990, "pairs": 20}))
            with self.assertRaisesRegex(ValueError, "already used"):
                stream_inventory(root, 149000, 32, root / "new")
            unused = stream_inventory(root, 149010, 32, root / "new")
            self.assertEqual(unused["deal_range_inclusive"], [149010, 149041])
            self.assertEqual(unused["overlapping_recorded_ranges"], [])

    def test_preservation_verification_detects_changed_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"old checkpoint")
            original = receipt(checkpoint)
            verify_receipts([original])
            checkpoint.write_bytes(b"new checkpoint")
            with self.assertRaisesRegex(ValueError, "protected artifact changed"):
                verify_receipts([original])


if __name__ == "__main__":
    unittest.main()
