import fcntl
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from retruco_ml.model import ENCODER_VERSION, Policy


class ResumeTests(unittest.TestCase):
    def run_trainer(self, root, *arguments):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "retruco_ml.train",
                "--device",
                "cpu",
                "--output",
                str(root),
                *arguments,
            ],
            capture_output=True,
            text=True,
        )

    def test_live_run_and_existing_checkpoint_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (root / "train.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = self.run_trainer(root)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("already has a live trainer", result.stderr)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"preserve this checkpoint")
            result = self.run_trainer(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("use --resume", result.stderr)
            self.assertEqual(checkpoint.read_bytes(), b"preserve this checkpoint")

    def test_resuming_quantized_target_as_float32_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.pt"
            torch.save(
                {"encoder": ENCODER_VERSION, "model": Policy().state_dict()}, target
            )
            torch.save(
                {
                    "encoder": ENCODER_VERSION,
                    "opponent_checkpoint_sha256": hashlib.sha256(
                        target.read_bytes()
                    ).hexdigest(),
                    "opponent_model_sha256": "saved-quantized-model",
                },
                root / "checkpoint.pt",
            )
            result = self.run_trainer(
                root, "--resume", "--opponent-checkpoint", str(target)
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("precision differs", result.stderr)


if __name__ == "__main__":
    unittest.main()
