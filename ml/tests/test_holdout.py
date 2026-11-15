import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from retruco_ml.holdout import reserve
from retruco_ml.splits import BLUFF_START, RELEASE_START


class HoldoutTests(unittest.TestCase):
    def test_resume_reuses_only_the_same_evaluation_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            first = reserve(path, "a" * 64)
            self.assertEqual(reserve(path, "a" * 64), first)
            second = reserve(path, "b" * 64)
            self.assertEqual(
                (first["seed"], second["seed"]), (RELEASE_START, RELEASE_START + 2000)
            )
            self.assertEqual(
                (first["bluff_seed"], second["bluff_seed"]),
                (BLUFF_START, BLUFF_START + 1000),
            )
            with self.assertRaisesRegex(ValueError, "counts changed"):
                reserve(path, "a" * 64, 4000)

    def test_concurrent_reservations_do_not_overlap_and_corruption_is_not_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            with ThreadPoolExecutor(max_workers=4) as executor:
                rows = list(
                    executor.map(lambda i: reserve(path, f"{i:064x}"), range(4))
                )
            self.assertEqual(
                sorted(r["seed"] for r in rows),
                [RELEASE_START + i * 2000 for i in range(4)],
            )
            ledger = json.loads(path.read_text())
            values = list(ledger["reservations"].values())
            values[1]["seed"] = values[0]["seed"]
            path.write_text(json.dumps(ledger))
            with self.assertRaisesRegex(ValueError, "overlapping"):
                reserve(path, "f" * 64)


if __name__ == "__main__":
    unittest.main()
