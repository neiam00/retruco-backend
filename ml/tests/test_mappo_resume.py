"""Real simulator/checkpoint continuation, using only disposable experiment data."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from retruco_ml.model import Policy
from retruco_ml.recurrent import file_hash


class MappoResumeTests(unittest.TestCase):
    def test_queued_campaign_remembers_wait_dependency_and_never_starts_after_failure(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent.pt"
            torch.save(
                {"model": Policy().state_dict(), "encoder": 2, "population": []}, parent
            )
            watched = root / "watched"
            watched.mkdir()
            (watched / "curriculum.lock").touch()
            (watched / "status.json").write_text('{"status":"failed"}')
            output = root / "experiment"
            config = (
                Path(__file__).resolve().parents[1] / "experiments/mappo-lstm16-v1.json"
            )
            command = [
                sys.executable,
                "-m",
                "retruco_ml.mappo_campaign",
                "--output",
                str(output),
            ]
            prepare = subprocess.run(
                [
                    *command,
                    "--prepare",
                    "--smoke",
                    "--warm-start",
                    str(parent),
                    "--config",
                    str(config),
                    "--wait-for",
                    str(watched),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(prepare.returncode, 0, prepare.stderr)
            original = file_hash(parent)
            resumed = subprocess.run(
                [*command, "--resume"], capture_output=True, text=True
            )
            self.assertNotEqual(resumed.returncode, 0)
            self.assertIn("without completing", resumed.stderr)
            self.assertFalse(any(output.glob("A-seed-*")))
            self.assertEqual(file_hash(parent), original)
            self.assertEqual(
                (watched / "status.json").read_text(), '{"status":"failed"}'
            )

    def test_split_run_restores_actor_critic_adam_and_rng_exactly(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent.pt"
            torch.manual_seed(4)
            torch.save(
                {"model": Policy().state_dict(), "encoder": 2, "population": []}, parent
            )
            digest = file_hash(parent)
            simulator = Path(__file__).resolve().parents[2] / "bin/selfplay"

            def run(folder, target, resume=False, arm="D"):
                command = [
                    sys.executable,
                    "-m",
                    "retruco_ml.mappo_train",
                    "--output",
                    str(folder),
                    "--arm",
                    arm,
                    "--seed",
                    "101",
                    "--steps",
                    "32",
                    "--decisions",
                    str(target),
                    "--workers",
                    "1",
                    "--environments",
                    "1",
                    "--epochs",
                    "1",
                    "--simulator",
                    str(simulator),
                ]
                command += ["--resume"] if resume else ["--warm-start", str(parent)]
                return subprocess.run(command, capture_output=True, text=True)

            split = root / "split"
            result = run(split, 32)
            self.assertEqual(result.returncode, 0, result.stderr)
            first = torch.load(split / "checkpoint.pt", weights_only=True)
            target = first["decisions"] + 1
            result = run(split, target, True)
            self.assertEqual(result.returncode, 0, result.stderr)
            whole = root / "whole"
            result = run(whole, target)
            self.assertEqual(result.returncode, 0, result.stderr)
            a = torch.load(split / "checkpoint.pt", weights_only=True)
            b = torch.load(whole / "checkpoint.pt", weights_only=True)

            def equal(x, y):
                if torch.is_tensor(x):
                    self.assertTrue(torch.equal(x, y))
                elif isinstance(x, dict):
                    self.assertEqual(x.keys(), y.keys())
                    for k in x:
                        equal(x[k], y[k])
                elif isinstance(x, (tuple, list)):
                    self.assertEqual(len(x), len(y))
                    for i, j in zip(x, y, strict=True):
                        equal(i, j)
                else:
                    self.assertEqual(x, y)

            for field in (
                "model",
                "critic",
                "optimizer",
                "random",
                "torch_random",
                "decisions",
                "update",
                "contracts",
            ):
                equal(a[field], b[field])
            self.assertEqual(file_hash(parent), digest)
            before = file_hash(split / "checkpoint.pt")
            bad = run(split, target, True, arm="B")
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("resume configuration differs", bad.stderr)
            self.assertEqual(file_hash(split / "checkpoint.pt"), before)
            bad = run(split, target)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("new directory", bad.stderr)
            self.assertEqual(file_hash(split / "checkpoint.pt"), before)


if __name__ == "__main__":
    unittest.main()
