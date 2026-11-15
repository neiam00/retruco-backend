import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from retruco_ml.model import Policy
from retruco_ml.recurrent import Actor, Critic, file_hash


class CooperationResumeTests(unittest.TestCase):
    def test_new_campaign_restores_optimizer_rng_and_never_edits_parent_files(self):
        torch.set_num_threads(2)
        torch.manual_seed(50)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            inputs.mkdir()
            reference = {"model": Policy().state_dict(), "population": []}
            actor = Actor(True)
            critic = Critic(True)
            parent = {
                "model": actor.state_dict(),
                "critic": critic.state_dict(),
                "encoder": 2,
                "memory_version": 1,
                "configuration": {"arm": "D"},
            }

            def save(name, data):
                path = inputs / name
                torch.save(data, path)
                return {"file": name, "sha256": file_hash(path)}

            manifest = {
                "kind": "cooperation-inputs-v1",
                "reference": save("reference.pt", reference),
                "parents": {},
                "critics": {},
            }
            for seed in (101, 102, 103):
                manifest["parents"][str(seed)] = save(f"actor-{seed}.pt", parent)
                manifest["critics"][str(seed)] = save(f"critic-{seed}.pt", parent)
            path = inputs / "manifest.json"
            path.write_text(json.dumps(manifest))
            original = {p: file_hash(p) for p in inputs.iterdir()}

            def run(folder, updates, resume=False):
                command = [
                    sys.executable,
                    "-m",
                    "retruco_ml.cooperation_train",
                    "--output",
                    str(folder),
                    "--inputs",
                    str(path),
                    "--seed",
                    "101",
                    "--steps",
                    "32",
                    "--workers",
                    "1",
                    "--environments",
                    "2",
                    "--updates",
                    str(updates),
                    "--hours",
                    ".1",
                    "--warmup-updates",
                    "0",
                    "--simulator",
                    str(Path(__file__).resolve().parents[2] / "bin/selfplay"),
                ]
                if resume:
                    command += ["--resume"]
                return subprocess.run(command, capture_output=True, text=True)

            split = root / "split"
            whole = root / "whole"
            for folder, updates, resume in [
                (split, 1, False),
                (split, 1, True),
                (whole, 2, False),
            ]:
                result = run(folder, updates, resume)
                self.assertEqual(result.returncode, 0, result.stderr)
            a = torch.load(split / "checkpoint.pt", weights_only=True)
            b = torch.load(whole / "checkpoint.pt", weights_only=True)
            for field in ("model", "critic"):
                for key, value in a[field].items():
                    torch.testing.assert_close(
                        value, b[field][key], atol=2e-6, rtol=2e-4
                    )
            self.assertEqual(a["decisions"], b["decisions"])
            self.assertEqual(a["random"], b["random"])
            self.assertTrue(torch.equal(a["torch_random"], b["torch_random"]))
            for key, state in a["optimizer"]["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(
                        value, b["optimizer"]["state"][key][name], atol=2e-6, rtol=2e-4
                    )
            receipt = json.loads((split / "chunk.json").read_text())
            self.assertEqual(
                receipt["checkpoint_sha256"], file_hash(split / "checkpoint.pt")
            )
            self.assertGreaterEqual(receipt["total_seconds"], a["total_seconds"])
            self.assertEqual({p: file_hash(p) for p in inputs.iterdir()}, original)
            bad = run(split, 1, False)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("must not exist", bad.stderr)


if __name__ == "__main__":
    unittest.main()
