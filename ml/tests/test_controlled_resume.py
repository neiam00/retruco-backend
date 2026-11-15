import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from retruco_ml.model import Policy
from retruco_ml.recurrent import Actor, Critic


class ControlledResumeTests(unittest.TestCase):
    def test_split_run_restores_optimizer_rng_choices_and_calibration(self):
        simulator = Path(__file__).resolve().parents[2] / "bin/selfplay"
        self.assertTrue(
            simulator.exists(), "build bin/selfplay before the native resume test"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            torch.manual_seed(4)
            actor, critic = Actor(True), Critic(True)
            manifest = {"kind": "cooperation-inputs-v1", "parents": {}, "critics": {}}

            def save(name, value):
                p = inputs / name
                torch.save(value, p)
                return {
                    "file": name,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }

            for seed in (101, 102, 103):
                manifest["parents"][str(seed)] = save(
                    f"parent-{seed}.pt", {"model": actor.state_dict()}
                )
                manifest["critics"][str(seed)] = save(
                    f"critic-{seed}.pt", {"critic": critic.state_dict()}
                )
            manifest["reference"] = save(
                "reference.pt", {"model": Policy().state_dict()}
            )
            path = inputs / "manifest.json"
            path.write_text(json.dumps(manifest))

            def run(folder, updates, resume=False):
                command = [
                    sys.executable,
                    "-m",
                    "retruco_ml.controlled_train",
                    "--output",
                    str(folder),
                    "--inputs",
                    str(path),
                    "--simulator",
                    str(simulator),
                    "--arm",
                    "D",
                    "--seed",
                    "101",
                    "--choices",
                    "100000",
                    "--steps",
                    "32",
                    "--workers",
                    "1",
                    "--environments",
                    "1",
                    "--calibration-choices",
                    "16",
                    "--warmup-max",
                    "2",
                    "--max-updates",
                    str(updates),
                ]
                if resume:
                    command += ["--resume"]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr[-2500:])
                return torch.load(folder / "checkpoint.pt", weights_only=True)

            whole = run(root / "whole", 2)
            run(root / "split", 1)
            resumed = run(root / "split", 1, True)
            for name in ("model", "critic"):
                for key in whole[name]:
                    torch.testing.assert_close(
                        whole[name][key], resumed[name][key], rtol=0, atol=0
                    )
            for name in ("actor_optimizer", "critic_optimizer"):
                for identity, values in whole[name]["state"].items():
                    for key, value in values.items():
                        torch.testing.assert_close(
                            value, resumed[name]["state"][identity][key], rtol=0, atol=0
                        )
            self.assertEqual(whole["choices"], resumed["choices"])
            self.assertEqual(whole["random"], resumed["random"])
            torch.testing.assert_close(
                whole["torch_random"], resumed["torch_random"], rtol=0, atol=0
            )
            calibration = root / "split/calibration.pt"
            calibration.write_bytes(calibration.read_bytes() + b"changed")
            command = [
                sys.executable,
                "-m",
                "retruco_ml.controlled_train",
                "--output",
                str(root / "split"),
                "--inputs",
                str(path),
                "--simulator",
                str(simulator),
                "--arm",
                "D",
                "--seed",
                "101",
                "--choices",
                "100000",
                "--steps",
                "32",
                "--workers",
                "1",
                "--environments",
                "1",
                "--calibration-choices",
                "16",
                "--warmup-max",
                "2",
                "--resume",
            ]
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("calibration set changed", rejected.stderr)
            for entry in [
                *manifest["parents"].values(),
                *manifest["critics"].values(),
                manifest["reference"],
            ]:
                self.assertEqual(
                    hashlib.sha256((inputs / entry["file"]).read_bytes()).hexdigest(),
                    entry["sha256"],
                )


if __name__ == "__main__":
    unittest.main()
