import hashlib
import json
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

import test_rmappo_control_training as control_helpers
import torch

from retruco_ml.improvement_population import improvement_scripted
from retruco_ml.model import Policy
from retruco_ml.recurrent import Actor as LegacyActor
from retruco_ml.recurrent import file_hash
from retruco_ml.rmappo_control_train import covered_collection
from retruco_ml.rmappo_model import RecurrentActor, RecurrentCritic
from retruco_ml.rmappo_population import Population
from retruco_ml.rollout import SimulatorPool

ROOT = Path(__file__).resolve().parents[2]


class Draws:
    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values)

    def randrange(self, stop):
        return 0

    def choice(self, values):
        return values[0]


class TargetPopulationTests(unittest.TestCase):
    def test_focused_practice_changes_partner_frequency_and_preserves_response_authority(
        self,
    ):
        fixed = {"frozen:101": None, "frozen:102": None}
        counts = {}
        for profile, expected in (("target", 0.30), ("reference_focus", 0.60)):
            population = Population(fixed, [None], profile=profile)
            rng = random.Random(612)
            references = native = 0
            for _ in range(10000):
                ep = population.episode(4, rng)
                friend = (ep.learner + 2) % 4
                if ep.controllers[friend] == "reference":
                    references += 1
                    self.assertFalse(ep.roles[friend])
                native += ep.controllers[(ep.learner + 1) % 4] == "script:baseline"
                self.assertNotIn("frozen:103", ep.controllers)
            self.assertAlmostEqual(references / 10000, expected, delta=0.015)
            self.assertAlmostEqual(native / 10000, 0.75, delta=0.015)
            counts[profile] = population.distribution["teammates"]
        for partner, mass in counts["target"].items():
            if partner != "reference":
                self.assertAlmostEqual(
                    counts["reference_focus"][partner] / 0.4, mass / 0.7
                )

    def test_original_population_preserves_checkpoint_rng_stream(self):
        # Captured from the pre-profile implementation: this stream is part of
        # the continuation contract for existing saved RNG states.
        population = Population({"frozen:101": None, "frozen:102": None}, [None] * 3)
        rng = random.Random(8771)
        episodes = []
        for i in range(100):
            ep = population.episode(2 if i % 3 == 0 else 4, rng)
            episodes.append([ep.learner, ep.controllers, ep.roles])
        self.assertEqual(
            hashlib.sha256(json.dumps(episodes, sort_keys=True).encode()).hexdigest(),
            "a8a8b5281b1b54db23edcbe034abd2f155db1d87e6e2cd3291bf6290e17aa30c",
        )

    def test_target_boundaries_roles_and_reserved_partner(self):
        population = Population(
            {"frozen:101": None, "frozen:102": None}, [None] * 3, profile="target"
        )
        for draw, expected in [
            (0.0, "script:baseline"),
            (0.749999, "script:baseline"),
            (0.75, "league:2"),
            (0.949999, "league:2"),
            (0.95, "frozen:101"),
        ]:
            with self.subTest(opponent=draw):
                ep = population.episode(2, Draws([draw, 0.1]))
                self.assertEqual(ep.controllers, ["learner", expected])
                self.assertTrue(all(ep.roles))
        partners = [
            (0.1, "learner"),
            (0.21, "frozen:101"),
            (0.3, "reference"),
            (0.56, "script:literal-v2"),
            (0.64, "script:trusting-v2"),
            (0.72, "script:cautious-v2"),
            (0.8, "script:literal-v3"),
            (0.88, "script:trusting-v3"),
            (0.96, "script:cautious-v3"),
        ]
        for draw, expected in partners:
            with self.subTest(partner=expected):
                ep = population.episode(4, Draws([0.0, draw]))
                self.assertEqual(
                    ep.controllers,
                    ["learner", "script:baseline", expected, "script:baseline"],
                )
                self.assertEqual(ep.roles[2], expected in ("learner", "frozen:101"))
                self.assertTrue(ep.roles[0] and ep.roles[1] and ep.roles[3])
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            Population(
                {"frozen:101": None, "frozen:103": None}, [None], profile="target"
            )
        with self.assertRaisesRegex(ValueError, "D101/D102"):
            Population({"frozen:101": None}, [None], profile="target")
        with self.assertRaisesRegex(ValueError, "historical league"):
            Population(
                {"frozen:101": None, "frozen:102": None}, profile="target"
            ).episode(4, random.Random(1))

    def test_sparse_target_batch_is_not_rejected_or_topped_up(self):
        torch.set_num_threads(1)
        torch.manual_seed(492)
        actor, critic = RecurrentActor("gru"), RecurrentCritic("gru")
        population = Population(
            {"frozen:101": LegacyActor(True), "frozen:102": LegacyActor(True)},
            profile="target",
        )
        population.add(actor)
        with SimulatorPool(1, "binary", ROOT / "bin/selfplay") as pool:
            records, matches, _, counts, balance = covered_collection(
                actor,
                critic,
                Policy(),
                pool,
                population,
                random.Random(4),
                1,
                players=4,
                environments=1,
                script_fn=improvement_scripted,
            )
        self.assertGreaterEqual(len(records), 1)
        self.assertEqual(matches, 1)
        self.assertEqual(counts["opponent_matches_native"], 1)
        self.assertTrue(
            all(r.balance_stratum.key == "2v2/human/cautious-v2" for r in records)
        )
        self.assertTrue(all(w == 1.0 for w in balance.weights))
        self.assertEqual(balance.metrics["effective_sample_fraction"], 1.0)
        self.assertIsNone(balance.metrics["target_met"])
        self.assertFalse(balance.metrics["coverage_ok"])
        self.assertEqual(len(balance.metrics["strata"]), 10)
        self.assertEqual(len(balance.metrics["missing_strata"]), 9)
        self.assertIn("2v2/cpu/frozen", balance.metrics["undercovered_strata"])
        self.assertTrue(
            all(
                s["effective_mass"] == s["sampled_fraction"]
                for s in balance.metrics["strata"].values()
            )
        )

    def test_profile_persistence_and_legacy_resume(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        fixture = control_helpers.RMAPPOControlTrainingTests()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, teacher, parent = fixture.fixtures(root)
            cmd = fixture.command(root, inputs, teacher, parent, "A", "target", 0) + [
                "--population-profile",
                "target",
            ]
            fixture.run_command(cmd)
            saved = torch.load(root / "target/checkpoint.pt", weights_only=True)
            self.assertEqual(saved["configuration"]["population_profile"], "target")
            self.assertEqual(
                saved["configuration"]["population"]["non_selfplay_opponents"],
                {"native": 0.75, "historical": 0.20, "fixed": 0.05},
            )
            self.assertEqual(saved["actor_optimizer"]["state"], {})
            fixture.run_command(cmd + ["--resume"])
            changed = fixture.command(
                root, inputs, teacher, parent, "A", "target", 0
            ) + ["--resume"]
            failed = subprocess.run(changed, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("resume configuration changed", failed.stderr)
            original = fixture.command(
                root, inputs, teacher, parent, "A", "original", 0
            )
            fixture.run_command(original)
            path = root / "original/checkpoint.pt"
            saved = torch.load(path, weights_only=True)
            for key in ("population_profile", "population", "save_every"):
                saved["configuration"].pop(key)
            torch.save(saved, path)
            fixture.run_command(original + ["--resume"])
            self.assertEqual(
                torch.load(path, weights_only=True)["configuration"],
                saved["configuration"],
            )
            invalid = fixture.command(root, inputs, teacher, parent, "C", "bad", 0) + [
                "--population-profile",
                "target",
            ]
            failed = subprocess.run(invalid, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("no response teacher penalty", failed.stderr)
            self.assertFalse((root / "bad").exists())

    def test_saved_endpoint_is_full_resumable_and_immutable(self):
        torch.set_num_threads(1)
        torch.manual_seed(18)
        fixture = control_helpers.RMAPPOControlTrainingTests()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, teacher, parent = fixture.fixtures(root)
            cmd = fixture.command(root, inputs, teacher, parent, "A", "trial", 1) + [
                "--save-every",
                "1000",
                "--steps",
                "1024",
            ]
            fixture.run_command(cmd)
            endpoint = root / "trial/endpoints/choices-1000.pt"
            saved = torch.load(endpoint, weights_only=True)
            self.assertGreaterEqual(saved["choices"], 1000)
            self.assertEqual(saved["endpoint_nominal_choices"], 1000)
            self.assertTrue(saved["actor_optimizer"]["state"])
            self.assertTrue(saved["critic_optimizer"]["state"])
            self.assertTrue(saved["league"])
            self.assertIn("python_rng", saved)
            self.assertIn("optimizer_rng", saved)
            self.assertIn("torch_rng", saved)
            self.assertEqual(
                endpoint.with_suffix(".sha256").read_text().strip(), file_hash(endpoint)
            )
            before = file_hash(endpoint)
            resumed = fixture.command(
                root, inputs, teacher, parent, "A", "trial", 2
            ) + ["--save-every", "1000", "--steps", "1024", "--resume"]
            fixture.run_command(resumed)
            self.assertEqual(file_hash(endpoint), before)
            changed = fixture.command(
                root, inputs, teacher, parent, "A", "trial", 2
            ) + ["--save-every", "2000", "--steps", "1024", "--resume"]
            failed = subprocess.run(changed, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("resume configuration changed", failed.stderr)


if __name__ == "__main__":
    unittest.main()
