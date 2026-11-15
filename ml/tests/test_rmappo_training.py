import copy
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from retruco_ml.improvement_population import improvement_scripted
from retruco_ml.model import Policy
from retruco_ml.recurrent import Actor as LegacyActor
from retruco_ml.recurrent import file_hash
from retruco_ml.rmappo_campaign import validate
from retruco_ml.rmappo_model import RecurrentActor, RecurrentCritic
from retruco_ml.rmappo_population import Episode, Population
from retruco_ml.rmappo_ppo import update
from retruco_ml.rmappo_rollout import JointDecision, collect, team_credit
from retruco_ml.rollout import SimulatorPool

ROOT = Path(__file__).resolve().parents[2]


class JointPopulation(Population):
    def episode(self, players, rng):
        return Episode(0, ["learner"] * players, [True] * 4)


class RMAPPOTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)

    def test_heads_up_collection_has_no_signal_credit_and_rejects_legacy_signal_turns(
        self,
    ):
        actor, critic, reference = (
            RecurrentActor("lstm"),
            RecurrentCritic("lstm"),
            Policy(),
        )
        pop = JointPopulation({"frozen:101": reference})
        pop.add(actor)
        before = copy.deepcopy(actor.state_dict())
        with SimulatorPool(1, "binary", ROOT / "bin/selfplay") as pool:
            records, matches, events, counts = collect(
                actor,
                critic,
                reference,
                pool,
                pop,
                random.Random(291),
                32,
                players=2,
                environments=1,
                script_fn=improvement_scripted,
            )
        self.assertGreaterEqual(len(records), 32)
        self.assertTrue(all(r.observation["kind"] == "game" for r in records))
        self.assertEqual(counts["signal_decisions"], 0)
        self.assertGreaterEqual(counts["choice_policy_decisions"], len(records))
        self.assertTrue(
            all(
                r.observation.get("choicePolicy") == "value-and-team-v10"
                for r in records
            )
        )
        for key, value in actor.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

        class LegacySignalTransport:
            def call(self, requests):
                return [
                    {"kind": "signal", "view": {"config": {"players": 2}}}
                    for _ in requests
                ]

        with self.assertRaisesRegex(ValueError, "corrected simulator"):
            collect(
                actor,
                critic,
                reference,
                LegacySignalTransport(),
                pop,
                random.Random(291),
                1,
                players=2,
                environments=1,
                script_fn=improvement_scripted,
            )

    def test_team_credit_never_interleaves_seats_and_forced_actions_do_not_discount(
        self,
    ):
        def record(seat, forced=False):
            return JointDecision(
                [],
                0,
                {"kind": "game", "actions": [[]] if forced else [[], []]},
                np.zeros(128),
                0,
                0,
                0.2,
                learner_seat=seat,
            )

        trajectories = {s: [record(s), record(s, True), record(s)] for s in range(4)}
        selected = team_credit(trajectories, 0, lam=1.0)
        self.assertEqual(len(selected), 8)
        for r in selected:
            self.assertAlmostEqual(r.target, 1.0 if r.learner_seat % 2 == 0 else -1.0)
        self.assertTrue(all(r.observation["actions"] == [[], []] for r in selected))

    def test_joint_simulator_trajectories_replay_update_and_keep_actor_private(self):
        for cell in ("gru", "lstm"):
            actor, critic = RecurrentActor(cell), RecurrentCritic(cell)
            reference = Policy()
            pop = JointPopulation({"frozen:101": reference})
            pop.add(actor)
            with SimulatorPool(1, "binary", ROOT / "bin/selfplay") as pool:
                records, matches, events, counts = collect(
                    actor,
                    critic,
                    reference,
                    pool,
                    pop,
                    random.Random(211),
                    64,
                    players=4,
                    environments=1,
                    script_fn=improvement_scripted,
                )
            self.assertGreaterEqual(len(records), 64)
            self.assertGreater(events, 0)
            self.assertEqual(counts["learner_trajectories"], 4 * matches)
            by_seat = {s: [r for r in records if r.learner_seat == s] for s in range(4)}
            self.assertTrue(all(by_seat.values()))
            self.assertEqual(by_seat[0][-1].target, by_seat[2][-1].target)
            self.assertEqual(by_seat[1][-1].target, by_seat[3][-1].target)
            self.assertEqual(by_seat[0][-1].target, -by_seat[1][-1].target)
            before = copy.deepcopy(actor.state_dict())
            critic_before = copy.deepcopy(critic.state_dict())
            ao = torch.optim.Adam(actor.parameters(), lr=3e-4)
            co = torch.optim.Adam(critic.parameters(), lr=3e-4)
            warm_critic = copy.deepcopy(critic)
            warm_optimizer = torch.optim.Adam(warm_critic.parameters(), lr=3e-4)
            warm = update(
                actor,
                warm_critic,
                ao,
                warm_optimizer,
                records,
                random.Random(9),
                epochs=1,
                batch_size=128,
                critic_only=True,
            )
            self.assertEqual(warm["actor_steps"], 0)
            self.assertEqual(ao.state_dict()["state"], {})
            for k, v in actor.state_dict().items():
                torch.testing.assert_close(v, before[k], atol=0, rtol=0)
            self.assertTrue(
                any(
                    not torch.equal(critic_before[k], v)
                    for k, v in warm_critic.state_dict().items()
                    if k.startswith("memory.")
                )
            )
            records[0].logp += 0.01
            with self.assertRaisesRegex(ValueError, "replay policy mismatch"):
                update(
                    actor,
                    critic,
                    ao,
                    co,
                    records,
                    random.Random(9),
                    epochs=1,
                    batch_size=128,
                )
            records[0].logp -= 0.01
            result = update(
                actor,
                critic,
                ao,
                co,
                records,
                random.Random(9),
                epochs=1,
                batch_size=128,
            )
            self.assertLess(result["replay_identity_max_logp_error"], 2e-4)
            self.assertLess(result["critic_replay_error"], 2e-4)
            self.assertGreater(result["actor_steps"], 0)
            self.assertTrue(
                any(
                    not torch.equal(before[k], v)
                    for k, v in actor.state_dict().items()
                    if k.startswith("memory.")
                )
            )
            self.assertTrue(
                any(
                    not torch.equal(critic_before[k], v)
                    for k, v in critic.state_dict().items()
                    if k.startswith("memory.")
                )
            )
            self.assertTrue(
                any(
                    not torch.equal(before[k], v) for k, v in actor.state_dict().items()
                )
            )

    def test_population_preserves_human_authority_and_excludes_independent_partner(
        self,
    ):
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            Population({"frozen:103": None})
        pop = Population({"frozen:101": None, "frozen:102": None})
        rng = random.Random(19)
        joint = human = 0
        for _ in range(500):
            ep = pop.episode(4, rng)
            seats = [s for s, c in enumerate(ep.controllers) if c == "learner"]
            joint += len(seats) > 1
            self.assertTrue(all(ep.roles[s] for s in seats))
            friend = (ep.learner + 2) % 4
            if (
                ep.controllers[friend].startswith("script:")
                or ep.controllers[friend] == "reference"
            ):
                human += 1
                self.assertFalse(ep.roles[friend])
            self.assertNotIn("frozen:103", ep.controllers)
        self.assertGreater(joint, 100)
        self.assertGreater(human, 100)

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / "inputs"
            inputs.mkdir()
            manifest = {"parents": {}}
            for seed in (101, 102, 103):
                path = inputs / f"{seed}.pt"
                torch.save({"model": LegacyActor(True).state_dict()}, path)
                manifest["parents"][str(seed)] = {
                    "file": path.name,
                    "sha256": file_hash(path),
                }
            path = inputs / "reference.pt"
            torch.save({"model": Policy().state_dict()}, path)
            manifest["reference"] = {"file": path.name, "sha256": file_hash(path)}
            input_path = inputs / "manifest.json"
            input_path.write_text(json.dumps(manifest))
            base = [
                sys.executable,
                "-m",
                "retruco_ml.rmappo_train",
                "--inputs",
                str(input_path),
                "--simulator",
                str(ROOT / "bin/selfplay"),
                "--cell",
                "gru",
                "--seed",
                "717",
                "--choices",
                "100000",
                "--steps",
                "64",
                "--environments",
                "1",
                "--workers",
                "1",
                "--batch-size",
                "128",
                "--warmups",
                "1",
                "--warmup-choices",
                "32",
            ]

            def run(name, count, resume=False):
                result = subprocess.run(
                    base
                    + ["--output", str(root / name), "--max-updates", str(count)]
                    + (["--resume"] if resume else []),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    result.returncode, 0, result.stdout[-1500:] + result.stderr[-1500:]
                )

            run("whole", 2)
            run("split", 1)
            run("split", 2, True)
            a = torch.load(root / "whole/checkpoint.pt", weights_only=True)
            b = torch.load(root / "split/checkpoint.pt", weights_only=True)
            for section in ("actor", "critic"):
                for key in a[section]:
                    torch.testing.assert_close(
                        a[section][key], b[section][key], atol=0, rtol=0
                    )
            self.assertEqual(a["choices"], b["choices"])
            self.assertEqual(a["python_rng"], b["python_rng"])
            self.assertEqual(a["optimizer_rng"], b["optimizer_rng"])
            self.assertTrue(torch.equal(a["torch_rng"], b["torch_rng"]))

    def test_campaign_rejects_overlapping_development_and_unbounded_budget(self):
        config = json.loads((ROOT / "ml/experiments/rmappo-v1.json").read_text())
        validate(config)
        config["final_seed"] = config["selection_seed"]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate(config)
        config = json.loads((ROOT / "ml/experiments/rmappo-v1.json").read_text())
        config["max_wall_hours"] = 24
        with self.assertRaisesRegex(ValueError, "twelve"):
            validate(config)


if __name__ == "__main__":
    unittest.main()
