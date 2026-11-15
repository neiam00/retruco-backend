import copy
import tempfile
import unittest
from pathlib import Path

import torch

from retruco_ml.recurrent import file_hash
from retruco_ml.rmappo_game_calibration import initializer, promote
from retruco_ml.rmappo_model import (
    RecurrentActor,
    RecurrentCritic,
    actor_from_checkpoint,
)


class CalibrationTests(unittest.TestCase):
    def test_saved_initializer_changes_only_deployable_game_log_odds_and_never_overwrites(
        self,
    ):
        torch.manual_seed(533)
        actor = RecurrentActor("lstm")
        critic = RecurrentCritic("lstm")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "parent.pt"
            q = root / "warm.pt"
            saved = {
                "kind": "rmappo-v1",
                "configuration": {"cell": "lstm"},
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "choices": 123,
                "updates": 2,
                "league": [],
                "encoder": 2,
                "memory_version": 1,
                "actor_optimizer": {"fake": True},
            }
            torch.save(saved, p)
            before = file_hash(p)
            initializer(p, q, 2.0)
            scaled = torch.load(q, weights_only=True)
            other = actor_from_checkpoint(scaled)
            self.assertEqual(file_hash(p), before)
            self.assertTrue(scaled["initialization_only"])
            self.assertNotIn("actor_optimizer", scaled)
            h = torch.randn(1, 64)
            a = torch.randn(8, 48)
            x = actor.game_logits(h.expand(8, -1), a)
            y = other.game_logits(h.expand(8, -1), a)
            torch.testing.assert_close(
                y - y.max(), 2 * (x - x.max()), atol=1e-12, rtol=1e-10
            )
            torch.testing.assert_close(
                actor.signal_logits(h), other.signal_logits(h), atol=0, rtol=0
            )
            with self.assertRaises(FileExistsError):
                initializer(p, q, 2.0)

    def test_no_promotion_from_unconfirmed_gain_or_regressing_configuration(self):
        def row(h, t, lo):
            return {
                "acceptable_strength": h >= 0.7 and t >= 0.6,
                "modes": {
                    "2": {
                        "win_rate": h,
                        "gain_ci95": [lo, 0.1],
                        "target_met": h >= 0.7,
                    },
                    "4": {
                        "win_rate": t,
                        "gain_ci95": [lo, 0.1],
                        "target_met": t >= 0.6,
                    },
                },
                "strata": {"primary": {"gain": 0.02, "gain_ci95": [0, 0.04]}},
                "configuration_regressions": [],
            }

        base = row(0.67, 0.56, 0)
        candidate = row(0.71, 0.61, 0.01)
        self.assertTrue(promote(candidate, base, {"primary"}))
        self.assertFalse(promote(row(0.71, 0.61, -0.01), base, {"primary"}))
        bad = copy.deepcopy(candidate)
        bad["configuration_regressions"] = [{"stratum": "primary", "gain": -0.04}]
        self.assertFalse(promote(bad, base, {"primary"}))


if __name__ == "__main__":
    unittest.main()
