import copy
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from retruco_ml.improvement_population import improvement_scripted
from retruco_ml.model import Policy
from retruco_ml.recurrent import Actor as LegacyActor
from retruco_ml.recurrent import file_hash
from retruco_ml.rmappo_betting import ResponseAnchor
from retruco_ml.rmappo_control_train import covered_collection
from retruco_ml.rmappo_model import RecurrentActor, RecurrentCritic
from retruco_ml.rmappo_population import Population
from retruco_ml.rmappo_ppo import update
from retruco_ml.rollout import SimulatorPool

ROOT = Path(__file__).resolve().parents[2]


class RMAPPOControlTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(53)

    def fixtures(self, root):
        inputs = root / "inputs"
        inputs.mkdir()
        manifest = {"parents": {}}
        for seed in (101, 102, 103):
            p = inputs / f"{seed}.pt"
            torch.save({"model": LegacyActor(True).state_dict()}, p)
            manifest["parents"][str(seed)] = {"file": p.name, "sha256": file_hash(p)}
        p = inputs / "reference.pt"
        torch.save({"model": Policy().state_dict()}, p)
        manifest["reference"] = {"file": p.name, "sha256": file_hash(p)}
        (inputs / "manifest.json").write_text(json.dumps(manifest))
        teacher = inputs / "teacher.pt"
        torch.save(
            {
                "model": LegacyActor(True).state_dict(),
                "encoder": 2,
                "memory_version": 1,
                "configuration": {"arm": "D"},
            },
            teacher,
        )
        actor, critic = RecurrentActor("gru"), RecurrentCritic("gru")
        parent = root / "parent.pt"
        torch.save(
            {
                "kind": "rmappo-v1",
                "configuration": {"cell": "gru"},
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "choices": 1000,
                "league": [
                    {
                        "kind": "rmappo-v1",
                        "configuration": {"cell": "gru"},
                        "actor": actor.state_dict(),
                    }
                ],
            },
            parent,
        )
        return inputs / "manifest.json", teacher, parent

    def command(self, root, inputs, teacher, parent, arm, name, updates):
        cmd = [
            sys.executable,
            "-m",
            "retruco_ml.rmappo_control_train",
            "--output",
            str(root / name),
            "--inputs",
            str(inputs),
            "--teacher",
            str(teacher),
            "--parent-checkpoint",
            str(parent),
            "--simulator",
            str(ROOT / "bin/selfplay"),
            "--cell",
            "gru",
            "--seed",
            "21901",
            "--arm",
            arm,
            "--choices",
            "100000",
            "--steps",
            "256",
            "--environments",
            "2",
            "--workers",
            "1",
            "--batch-size",
            "128",
            "--warmups",
            "1",
            "--warmup-choices",
            "128",
            "--max-updates",
            str(updates),
        ]
        if arm in ("B", "D"):
            cmd.append("--partner-balance")
        cmd += ["--response-anchor", str(0.1 if arm in ("C", "D") else 0.0)]
        return cmd

    def run_command(self, cmd):
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout[-1500:] + p.stderr[-1500:])

    def test_all_treatments_start_with_identical_parent_and_fresh_optimizer_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, teacher, parent = self.fixtures(root)
            original = torch.load(parent, weights_only=True)
            states = []
            for arm in "ABCD":
                self.run_command(
                    self.command(root, inputs, teacher, parent, arm, arm, 0)
                )
                state = torch.load(root / arm / "initial.pt", weights_only=True)
                states.append(state)
                for part in ("actor", "critic"):
                    for key in original[part]:
                        torch.testing.assert_close(
                            state[part][key], original[part][key], atol=0, rtol=0
                        )
                self.assertEqual(state["choices"], 0)
                self.assertEqual(state["actor_optimizer"]["state"], {})
                self.assertEqual(state["critic_optimizer"]["state"], {})
            for state in states[1:]:
                self.assertEqual(state["python_rng"], states[0]["python_rng"])
                self.assertEqual(state["optimizer_rng"], states[0]["optimizer_rng"])
                self.assertTrue(torch.equal(state["torch_rng"], states[0]["torch_rng"]))
            bad = self.command(root, inputs, teacher, parent, "A", "mislabeled", 0) + [
                "--partner-balance"
            ]
            result = subprocess.run(bad, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("arm label and treatment flags disagree", result.stderr)

    def test_real_combined_update_uses_actual_choice_weights_and_frozen_public_teacher(
        self,
    ):
        actor, critic = RecurrentActor("gru"), RecurrentCritic("gru")
        teacher = LegacyActor(True)
        population = Population(
            {"frozen:101": LegacyActor(True), "frozen:102": LegacyActor(True)}
        )
        population.add(actor)
        with SimulatorPool(1, "binary", ROOT / "bin/selfplay") as pool:
            records, _, _, _, balance = covered_collection(
                actor,
                critic,
                Policy(),
                pool,
                population,
                random.Random(711),
                512,
                environments=4,
                script_fn=improvement_scripted,
                headsup_fraction=0.4,
            )
        self.assertTrue(balance.metrics["coverage_ok"])
        self.assertAlmostEqual(sum(balance.weights), len(records), places=6)
        anchor = ResponseAnchor(teacher)
        before = copy.deepcopy(anchor.teacher.state_dict())
        result = update(
            actor,
            critic,
            torch.optim.Adam(actor.parameters(), lr=3e-4),
            torch.optim.Adam(critic.parameters(), lr=3e-4),
            records,
            random.Random(33),
            epochs=1,
            batch_size=128,
            choice_weights=balance.weights,
            response_anchor=anchor,
        )
        self.assertGreater(result["response_anchor"]["response_count"], 0)
        self.assertGreater(
            result["response_anchor"]["weighted_response_anchor_loss_mean"], 0
        )
        self.assertLess(result["replay_identity_max_logp_error"], 2e-4)
        for key, value in anchor.teacher.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in anchor.teacher.parameters()))

    def test_combined_continuation_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, teacher, parent = self.fixtures(root)
            self.run_command(
                self.command(root, inputs, teacher, parent, "D", "whole", 2)
            )
            self.run_command(
                self.command(root, inputs, teacher, parent, "D", "split", 1)
            )
            self.run_command(
                self.command(root, inputs, teacher, parent, "D", "split", 2)
                + ["--resume"]
            )
            a = torch.load(root / "whole/checkpoint.pt", weights_only=True)
            b = torch.load(root / "split/checkpoint.pt", weights_only=True)
            for part in ("actor", "critic"):
                for key in a[part]:
                    torch.testing.assert_close(
                        a[part][key], b[part][key], atol=0, rtol=0
                    )
            self.assertEqual(a["choices"], b["choices"])
            self.assertTrue(torch.equal(a["torch_rng"], b["torch_rng"]))
            self.assertEqual(a["python_rng"], b["python_rng"])


if __name__ == "__main__":
    unittest.main()
