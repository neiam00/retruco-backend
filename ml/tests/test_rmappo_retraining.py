"""Response safeguards and exact resumption of the corrected-rule retraining."""

import copy
import random
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

import test_rmappo_control_training as control_tests
import torch

from retruco_ml.improvement_population import improvement_scripted
from retruco_ml.model import Policy
from retruco_ml.recurrent import Actor as LegacyActor
from retruco_ml.recurrent import file_hash
from retruco_ml.rmappo_controls_campaign import campaign_input
from retruco_ml.rmappo_model import RecurrentActor, RecurrentCritic
from retruco_ml.rmappo_population import Population
from retruco_ml.rmappo_ppo import update
from retruco_ml.rmappo_rollout import collect
from retruco_ml.rmappo_target_campaign import (
    distinct_evaluations,
    training_command,
    validate,
)
from retruco_ml.rollout import SimulatorPool

ROOT = Path(__file__).resolve().parents[2]


class ResponseGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(220)
        cls.actor = RecurrentActor("lstm")
        cls.critic = RecurrentCritic("lstm")
        pop = Population(
            {"frozen:101": LegacyActor(True), "frozen:102": LegacyActor(True)},
            profile="target",
        )
        pop.add(cls.actor)
        with SimulatorPool(1, "binary", ROOT / "bin/selfplay") as pool:
            cls.records, _, _, _ = collect(
                cls.actor,
                cls.critic,
                Policy(),
                pool,
                pop,
                random.Random(220),
                256,
                environments=2,
                script_fn=improvement_scripted,
                headsup_fraction=0.4,
            )
        assert any(r.observation.get("responseChoice") for r in cls.records)

    def attempt(self, divergence):
        actor, critic = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        ao = torch.optim.Adam(actor.parameters(), lr=3e-4, eps=1e-5)
        co = torch.optim.Adam(critic.parameters(), lr=3e-4, eps=1e-5)
        # Initialize Adam's moments without moving the actor, so rollback must
        # restore existing optimizer state, not merely clear an empty mapping.
        for p in actor.parameters():
            p.grad = torch.zeros_like(p)
        ao.step()
        ao.zero_grad(set_to_none=True)
        before = copy.deepcopy(ao.state_dict())
        family = next(
            r.observation["responseFamily"]
            for r in self.records
            if r.observation.get("responseChoice")
        )
        with patch(
            "retruco_ml.rmappo_ppo.policy_kl",
            side_effect=[
                {"all": 0.001, "response": 0.001, "response:" + family: divergence},
                {"all": 0.0, "response": 0.0, "response:" + family: 0.0},
            ],
        ) as measured:
            result = update(
                actor,
                critic,
                ao,
                co,
                self.records,
                random.Random(80),
                epochs=2,
                batch_size=512,
                response_kl_stop=0.03,
                response_kl_rollback=0.05,
                response_min_choices=1,
            )
        self.assertEqual(measured.call_count, 2)
        return actor, critic, ao, before, result

    def test_rare_response_family_rollback_restores_actor_and_existing_adam(self):
        actor, critic, ao, before, result = self.attempt(0.06)
        self.assertTrue(result["response_guard_rolled_back"])
        self.assertEqual(result["actor_steps"], 0)
        self.assertGreater(result["actor_steps_rolled_back"], 0)
        self.assertGreater(result["critic_steps"], 1)
        for key, value in self.actor.state_dict().items():
            torch.testing.assert_close(actor.state_dict()[key], value, atol=0, rtol=0)
        self.assertTrue(
            any(
                not torch.equal(v, critic.state_dict()[k])
                for k, v in self.critic.state_dict().items()
            )
        )
        after = ao.state_dict()
        self.assertEqual(after["param_groups"], before["param_groups"])
        for key, state in before["state"].items():
            for field, value in state.items():
                torch.testing.assert_close(
                    after["state"][key][field], value, atol=0, rtol=0
                )

    def test_stop_retains_small_response_change_and_continues_critic_learning(self):
        actor, _, _, _, result = self.attempt(0.04)
        self.assertTrue(result["response_guard_stopped"])
        self.assertFalse(result["response_guard_rolled_back"])
        self.assertGreater(result["actor_steps"], 0)
        self.assertGreater(result["critic_steps"], result["actor_steps"])
        self.assertTrue(
            any(
                not torch.equal(v, actor.state_dict()[k])
                for k, v in self.actor.state_dict().items()
            )
        )


class FullReturnResumeTests(unittest.TestCase):
    def test_contextual_candidate_and_semantic_schedule_resume_preserve_training_state(
        self,
    ):
        helper = control_tests.RMAPPOControlTrainingTests()
        helper.setUp()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, teacher, parent = helper.fixtures(root)
            flags = [
                "--population-profile",
                "target",
                "--candidate-features",
                "1",
                "--semantic-exploration",
                ".02",
                "--semantic-anneal-choices",
                "1000000",
                "--response-kl-stop",
                ".03",
                "--response-kl-rollback",
                ".05",
            ]

            def cmd(name, n):
                return (
                    helper.command(root, inputs, teacher, parent, "A", name, n) + flags
                )

            helper.run_command(cmd("whole", 2))
            helper.run_command(cmd("split", 1))
            helper.run_command(cmd("split", 2) + ["--resume"])
            a = torch.load(root / "whole/checkpoint.pt", weights_only=True)
            b = torch.load(root / "split/checkpoint.pt", weights_only=True)
            self.assertEqual(a["configuration"]["candidate_features"], 1)
            for part in ("actor", "critic"):
                for key in a[part]:
                    torch.testing.assert_close(
                        a[part][key], b[part][key], atol=0, rtol=0
                    )
            self.assertEqual(a["choices"], b["choices"])
            self.assertEqual(a["python_rng"], b["python_rng"])
            self.assertEqual(a["optimizer_rng"], b["optimizer_rng"])
            self.assertTrue(torch.equal(a["torch_rng"], b["torch_rng"]))
            from retruco_ml.rmappo_target_campaign import verify_initial

            verify_initial(root / "whole", parent)

    def test_full_match_return_with_response_guard_resumes_exactly_and_rejects_estimator_change(
        self,
    ):
        helper = control_tests.RMAPPOControlTrainingTests()
        helper.setUp()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, teacher, parent = helper.fixtures(root)
            flags = [
                "--population-profile",
                "target",
                "--gae-lambda",
                "1.0",
                "--response-kl-stop",
                ".03",
                "--response-kl-rollback",
                ".05",
            ]

            def cmd(name, n):
                return (
                    helper.command(root, inputs, teacher, parent, "A", name, n) + flags
                )

            helper.run_command(cmd("whole", 2))
            helper.run_command(cmd("split", 1))
            helper.run_command(cmd("split", 2) + ["--resume"])
            a = torch.load(root / "whole/checkpoint.pt", weights_only=True)
            b = torch.load(root / "split/checkpoint.pt", weights_only=True)
            for part in ("actor", "critic"):
                for key in a[part]:
                    torch.testing.assert_close(
                        a[part][key], b[part][key], atol=0, rtol=0
                    )
            self.assertEqual(a["choices"], b["choices"])
            self.assertEqual(a["python_rng"], b["python_rng"])
            self.assertEqual(a["optimizer_rng"], b["optimizer_rng"])
            self.assertTrue(torch.equal(a["torch_rng"], b["torch_rng"]))
            import subprocess

            before = file_hash(root / "split/checkpoint.pt")
            changed = subprocess.run(
                cmd("split", 3) + ["--resume", "--gae-lambda", ".99"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("resume configuration changed", changed.stderr)
            self.assertEqual(before, file_hash(root / "split/checkpoint.pt"))


class CampaignContractTests(unittest.TestCase):
    def test_campaign_inputs_resolve_outside_the_repository_root(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder).resolve()
            inputs = repo / "ml/runs/inputs"
            inputs.mkdir(parents=True)
            parent = inputs / "parent.pt"
            parent.write_bytes(b"frozen parent")
            with chdir(inputs):
                for name in ("ml/runs/inputs/parent.pt", "parent.pt", str(parent)):
                    with self.subTest(name=name):
                        self.assertEqual(campaign_input(repo, name), parent)

    def test_protected_study_registers_only_calibrated_team_continuations(self):
        import json

        config = json.loads(
            (ROOT / "ml/experiments/rmappo-reference-focus-v1.json").read_text()
        )
        validate(config)
        for name, expected in [("protected", "target"), ("focused", "reference_focus")]:
            command = training_command(Path("/campaign"), config, name, 6101, 1000000)
            self.assertIn("--protect-headsup", command)
            self.assertEqual(
                command[command.index("--population-profile") + 1], expected
            )
            self.assertEqual(
                command[command.index("--parent-checkpoint") + 1],
                "/campaign/parents/6101.pt",
            )
        for key, value in [
            ("selection_pairs", 64),
            ("players", 2),
            ("budgets", [500000, 1000000]),
            ("headsup_protection", 0),
            ("populations", {}),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate({**config, key: value})
        damaged = copy.deepcopy(config)
        for parent in damaged["parent_sources"].values():
            parent["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "calibrated starter"):
            validate(damaged)

    def test_context_study_requires_original_nominee_and_full_common_rounds(self):
        import json

        config = json.loads(
            (ROOT / "ml/experiments/rmappo-context-exploration-v1.json").read_text()
        )
        validate(config)
        for key, value, message in [
            ("prior_nominee", None, "prior nominee"),
            ("budgets", [500000, 1000000, 2000000], "common continuation milestones"),
            ("semantic_anneal_choices", 500000, "contextual exploration bundle"),
        ]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                validate({**config, key: value})

    def test_only_identical_weights_and_identical_deals_can_reuse_an_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.pt"
            b = root / "b.pt"
            a.write_bytes(b"same weights")
            b.write_bytes(a.read_bytes())
            spec = {
                "id": "native-headsup",
                "players": 2,
                "opponent": "native",
                "partner": "reference-cpu",
            }
            jobs = [
                {
                    "checkpoint": p,
                    "pairs": 64,
                    "seed": 132000,
                    "spec": spec,
                    "path": root / f"{i}.json",
                }
                for i, p in enumerate((a, b))
            ]
            unique, aliases = distinct_evaluations(jobs)
            self.assertEqual(len(unique), 1)
            self.assertEqual(aliases, [(jobs[1], jobs[0])])
            jobs[1]["seed"] += 1
            self.assertEqual(len(distinct_evaluations(jobs)[0]), 2)
            jobs[1]["seed"] -= 1
            b.write_bytes(b"other weights")
            self.assertEqual(len(distinct_evaluations(jobs)[0]), 2)

    def test_new_study_requires_one_parent_and_frozen_response_limits(self):
        import json

        config = json.loads((ROOT / "ml/experiments/rmappo-v1.json").read_text())
        config.update(
            version=2,
            cell="lstm",
            profiles=["gae99", "mc"],
            response_kl_stop=0.03,
            response_kl_rollback=0.05,
            seeds=[4101, 4102, 4103],
            budgets=[500000, 1000000, 2000000],
            save_every=250000,
            warmup_choices=8192,
            max_wall_hours=12,
            evaluation_reserve_hours=4,
            selection_seed=132000,
            selection_pairs=64,
            selection_diagnostic_seed=133000,
            selection_diagnostic_pairs=32,
            final_seed=155000,
            final_pairs=2000,
            final_diagnostic_seed=158000,
            final_diagnostic_pairs=128,
        )
        config["parent_sources"] = {
            str(s): {"file": f"{s}.pt", "sha256": "same"} for s in config["seeds"]
        }
        validate(config)
        config["parent_sources"][str(config["seeds"][1])]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "share the screened parent"):
            validate(config)
        config["response_kl_stop"] = 0.1
        with self.assertRaisesRegex(ValueError, "frozen response guard"):
            validate(config)


if __name__ == "__main__":
    unittest.main()
