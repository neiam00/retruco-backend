import fcntl
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from retruco_ml.artifacts import digest
from retruco_ml.model import Policy
from retruco_ml.upgrade_run import upgrade, verify_curriculum_source


def source(path, content):
    path.mkdir(parents=True)
    (path / "trainer.py").write_text(content)
    (path / "ml").mkdir()
    (path / "ml/uv.lock").write_text("locked dependencies")
    manifest = {
        "files": {name: digest(path / name) for name in ("trainer.py", "ml/uv.lock")}
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


class UpgradeTests(unittest.TestCase):
    def test_stopped_curriculum_fork_preserves_checkpoint_and_phase_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old"
            new = root / "new"
            pilots = root / "pilots"
            source(old / "source", "old trainer")
            source(pilots / "source", "old trainer")
            phase = old / "heads-up"
            phase.mkdir()
            model = Policy()
            torch.save(
                {
                    "encoder": 2,
                    "model": model.state_dict(),
                    "optimizer": torch.optim.Adam(model.parameters()).state_dict(),
                    "population": [model.state_dict()],
                    "random": ("preserved",),
                    "torch_random": torch.get_rng_state(),
                    "update": 123,
                    "total_seconds": 7200,
                },
                phase / "checkpoint.pt",
            )
            (old / "configuration.json").write_text(
                json.dumps(
                    {"pilots": str(pilots), "heads_up_hours": 12, "mixed_hours": 24}
                )
            )
            (phase / "phase.json").write_text(
                json.dumps(
                    {"base_seconds": 3600, "target_hours": 12, "best_score": 0.7}
                )
            )
            before = digest(phase / "checkpoint.pt")
            # Building and proving a Git revision is exercised by the CLI smoke
            # test; this test owns file migration and immutable-source lineage.
            with (
                patch("retruco_ml.upgrade_run.training_commit", return_value="a" * 40),
                patch(
                    "retruco_ml.upgrade_run.freeze_source",
                    side_effect=lambda path, repo: source(path, "new trainer"),
                ),
            ):
                report = upgrade(old, new, root)
            self.assertEqual(digest(phase / "checkpoint.pt"), before)
            self.assertEqual(digest(new / "heads-up/checkpoint.pt"), before)
            self.assertEqual(
                (phase / "phase.json").read_bytes(),
                (new / "heads-up/phase.json").read_bytes(),
            )
            self.assertEqual(
                (old / "configuration.json").read_bytes(),
                (new / "configuration.json").read_bytes(),
            )
            self.assertEqual(
                digest(
                    new
                    / report["checkpoint_inputs"]["heads-up/checkpoint.pt"]["archived"]
                ),
                before,
            )
            verify_curriculum_source(new, pilots)
            (
                new / report["previous_sources"][0]["directory"] / "trainer.py"
            ).write_text("changed")
            with self.assertRaises(ValueError):
                verify_curriculum_source(new, pilots)
            with self.assertRaises(ValueError):
                upgrade(old, new, root)

    def test_live_trainer_cannot_be_forked_or_interrupted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old"
            old.mkdir()
            target = root / "new"
            (old / "checkpoint.pt").write_bytes(b"unchanged")
            with (old / "train.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "still running"):
                    upgrade(old, target, root)
            self.assertFalse(target.exists())
            self.assertEqual((old / "checkpoint.pt").read_bytes(), b"unchanged")


if __name__ == "__main__":
    unittest.main()
