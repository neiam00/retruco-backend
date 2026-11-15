import json
import tempfile
import unittest
from pathlib import Path

from retruco_ml.recurrent import file_hash
from retruco_ml.rmappo_coverage import coverage


class RMAPPOCoverageTests(unittest.TestCase):
    def test_only_exact_protocol_bound_authenticated_paths_count(self):
        c = json.loads(
            (
                Path(__file__).resolve().parents[2] / "ml/experiments/rmappo-v1.json"
            ).read_text()
        )
        c["seeds"] = [11001]
        c["selection_pairs"] = 2
        with tempfile.TemporaryDirectory() as tmp:
            r = Path(tmp)
            (r / "inputs").mkdir()
            cp = r / "inputs/legacy-champion-lstm16.pt"
            cp.write_bytes(b"frozen reference")
            inputs = r / "inputs/manifest.json"
            inputs.write_text("{}")
            p = r / "selection/reference/native-headsup.json"
            p.parent.mkdir(parents=True)
            report = {
                "protocol": "improvement-evaluation-v1",
                "seed": c["selection_seed"],
                "pairs": 2,
                "partner": "reference-cpu",
                "opponent": "native",
                "signals": "normal",
                "card_policy": "policy",
                "checkpoint_sha256": file_hash(cp),
                "inputs_sha256": file_hash(inputs),
                "configurations": [
                    {
                        "players": 2,
                        "target": t,
                        "flor": f,
                        "pairs": 2,
                        "matches": 4,
                        "pair_scores": [0.5, 0.5],
                        "pair_net_points_per_hand": [0.0, 0.0],
                        "win_rate": 0.5,
                    }
                    for t in (12, 24)
                    for f in (False, True)
                ],
            }
            p.write_text(json.dumps(report))
            p.with_suffix(".sha256").write_text(file_hash(p))
            extra = r / "unrelated.json"
            extra.write_text(p.read_text())
            extra.with_suffix(".sha256").write_text(file_hash(extra))
            result = coverage(r, c)
            self.assertEqual(result["authenticated"], 1)
            self.assertEqual(result["expected"], 92)
            self.assertFalse(result["complete"])
            report["configurations"][0]["win_rate"] = 0.9
            p.write_text(json.dumps(report))
            p.with_suffix(".sha256").write_text(file_hash(p))
            result = coverage(r, c)
            self.assertEqual(result["authenticated"], 0)
            self.assertEqual(len(result["invalid"]), 1)


if __name__ == "__main__":
    unittest.main()
