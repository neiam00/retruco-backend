import copy
import tempfile
import unittest
from pathlib import Path

import torch

from retruco_ml.model_capacity import actor_from_checkpoint
from retruco_ml.policy_calibration import calibrated_weights, prepare
from retruco_ml.recurrent import Actor, Critic, centered_categorical, file_hash


class PolicyCalibrationTests(unittest.TestCase):
    def test_scale_changes_only_game_distribution_and_preserves_signal_support(self):
        torch.manual_seed(42)
        actor = Actor(True, 64)
        before = copy.deepcopy(actor.state_dict())
        context = torch.randn(9, 64)
        actions = torch.randn(9, 48)
        raw = actor.game_logits(context, actions)
        signal = {
            "version": 2,
            "kind": "signal",
            "signalMask": [True, False, True, False, False, True],
        }
        for scale in (0.5, 1.0, 2.0, 4.0):
            changed = calibrated_weights(actor, scale)
            torch.testing.assert_close(
                changed.game_logits(context, actions), raw * scale, atol=0, rtol=0
            )
            expected = centered_categorical(raw * scale).probs
            torch.testing.assert_close(
                centered_categorical(changed.game_logits(context, actions)).probs,
                expected,
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                changed.distribution(context[0], signal).probs,
                actor.distribution(context[0], signal).probs,
                atol=0,
                rtol=0,
            )
            self.assertEqual(sum(p.numel() for p in changed.parameters()), 122567)
        for key, value in before.items():
            torch.testing.assert_close(actor.state_dict()[key], value, atol=0, rtol=0)
        for invalid in (0, -1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                calibrated_weights(actor, invalid)

    def test_new_candidates_reload_with_normal_actor_and_preserve_source(self):
        torch.manual_seed(43)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "parent.pt"
            torch.save(
                {
                    "model": Actor(True).state_dict(),
                    "critic": Critic(True).state_dict(),
                    "encoder": 2,
                    "memory_version": 1,
                },
                source,
            )
            before = file_hash(source)
            manifest = prepare(source, root / "copies")
            for entry in manifest["candidates"]:
                path = root / "copies" / entry["checkpoint"]
                saved = torch.load(path, weights_only=True)
                actor = actor_from_checkpoint(saved)
                self.assertEqual(actor.memory.hidden_size, 64)
                self.assertEqual(saved["policy_calibration"]["source_sha256"], before)
                self.assertEqual(entry["sha256"], file_hash(path))
            self.assertEqual(before, file_hash(source))
            with self.assertRaises(ValueError):
                prepare(source, root / "copies")


if __name__ == "__main__":
    unittest.main()
