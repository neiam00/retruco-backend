"""Coverage, immutable teammates, and soft policy preservation contracts."""

import copy
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch
from test_recurrent_batch import fixture

from retruco_ml.cooperation_diagnostics import DecisionContexts
from retruco_ml.cooperation_practice import CooperationPracticePopulation
from retruco_ml.decision_ppo import update
from retruco_ml.historical_league import HistoricalLeaguePopulation
from retruco_ml.improvement_inputs import migrate_legacy
from retruco_ml.model import Policy
from retruco_ml.partner_pool import prepare_partner_pool
from retruco_ml.recurrent import Actor, Critic, file_hash
from retruco_ml.recurrent_batch import PackedRecords
from retruco_ml.stability import cache_logits, categorical_kl, reference_distribution


class CooperationPracticeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(77)

    def test_partner_practice_preserves_headsup_stream_and_opponent_league(self):
        actor = Actor(True, memory_size=64)
        control = HistoricalLeaguePopulation(
            learned_controllers=("frozen:101", "frozen:102")
        )
        practice = CooperationPracticePopulation(
            learned_controllers=("frozen:101", "frozen:102"), partner_actor=actor
        )
        for population in (control, practice):
            population.add_actor(actor, 0)
        counts = Counter()
        for seed in range(4000):
            left, right = random.Random(seed), random.Random(seed)
            self.assertEqual(control.episode(2, left), practice.episode(2, right))
            self.assertEqual(left.getstate(), right.getstate())
            a, b = (
                control.episode(4, random.Random(seed)),
                practice.episode(4, random.Random(seed)),
            )
            self.assertEqual(a.learner, b.learner)
            for seat in range(4):
                if seat % 2 != a.learner % 2:
                    self.assertEqual(
                        (a.controllers[seat], a.roles[seat]),
                        (b.controllers[seat], b.roles[seat]),
                    )
            partner = (b.learner + 2) % 4
            counts["cpu"] += b.roles[partner]
            counts["learned"] += b.controllers[partner] == "practice:starter"
            self.assertNotEqual(b.controllers[partner], "frozen:103")
        self.assertTrue(0.32 < counts["cpu"] / 4000 < 0.38)
        self.assertTrue(0.12 < counts["learned"] / 4000 < 0.18)

    def test_frozen_partner_is_independent_and_resume_binds_its_weights(self):
        actor = Actor(True, memory_size=64)
        initial = copy.deepcopy(actor.state_dict())
        population = CooperationPracticePopulation(partner_actor=actor)
        population.add_actor(actor, 0)
        state = population.state_dict()
        before = torch.get_rng_state().clone()
        models = population.models(None)
        torch.testing.assert_close(before, torch.get_rng_state(), atol=0, rtol=0)
        for p in actor.parameters():
            p.data.add_(0.1)
        for k, v in models["practice:starter"].state_dict().items():
            torch.testing.assert_close(v, initial[k], atol=0, rtol=0)
        restored = CooperationPracticePopulation(
            partner_actor=models["practice:starter"]
        )
        restored.load_state_dict(state)
        changed = CooperationPracticePopulation(partner_actor=actor)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            changed.load_state_dict(state)

    def test_zero_anchor_does_not_change_training_when_teacher_changes(self):
        actor, critic = Actor(True), Critic(True)
        records = fixture(actor, critic)
        parent = copy.deepcopy(actor)
        other = copy.deepcopy(actor)
        with torch.no_grad():
            other.signal.bias.add_(torch.arange(6))
        trained = []
        for teacher in (parent, other):
            a, c = copy.deepcopy(actor), copy.deepcopy(critic)
            update(
                a,
                c,
                torch.optim.SGD(a.parameters(), lr=0.001),
                torch.optim.SGD(c.parameters(), lr=0.001),
                records,
                random.Random(4),
                teacher,
                epochs=1,
                batch_size=64,
            )
            trained.append(a.state_dict())
        for k, v in trained[0].items():
            torch.testing.assert_close(v, trained[1][k], atol=0, rtol=0)

    def test_anchor_corrects_signal_drift_without_allowing_false_claims(self):
        actor, critic = Actor(True), Critic(True)
        with torch.no_grad():
            actor.signal.weight.zero_()
            actor.signal.bias.zero_()
        records = [
            r for r in fixture(actor, critic) if r.observation["kind"] == "signal"
        ]
        for record in records:
            record.advantage = 0.0
        parent = copy.deepcopy(actor)
        with torch.no_grad():
            parent.signal.bias[2] = 2.0
        packed = PackedRecords(records)
        teacher = cache_logits(parent, packed)

        def divergence():
            with torch.no_grad():
                contexts, indices = packed.contexts(actor, packed.chunks)
                d = packed.distribution(actor, contexts, indices)
                self.assertTrue(
                    torch.equal(
                        d.probs[:, [1, 3, 4, 5]],
                        torch.zeros_like(d.probs[:, [1, 3, 4, 5]]),
                    )
                )
                return float(
                    categorical_kl(
                        reference_distribution(teacher, indices, d.logits.shape), d
                    ).mean()
                )

        before = divergence()
        result = update(
            actor,
            critic,
            torch.optim.SGD(actor.parameters(), lr=0.01),
            torch.optim.SGD(critic.parameters(), lr=0.001),
            records,
            random.Random(3),
            parent,
            epochs=1,
            batch_size=64,
            signal_anchor=1.0,
        )
        self.assertLess(divergence(), before)
        self.assertGreater(result["anchor_loss_mean"], 0.0)
        self.assertEqual(result["response_choices"], 0)
        self.assertTrue(all(p.grad is None for p in parent.parameters()))

    def test_pending_declarations_are_not_counted_as_response_opportunities(self):
        counts = DecisionContexts()
        observation = {
            "kind": "game",
            "seat": 0,
            "view": {"config": {"players": 2}, "pending": {"kind": "truco"}},
            "legal": [{"type": "declare", "declaration": 27}],
        }
        counts.observe(observation, 0)
        self.assertEqual(counts.counts["no_partner_claim"]["pending"], 1)
        self.assertEqual(counts.counts["no_partner_claim"]["response_opportunities"], 0)
        observation["legal"] = [
            {"type": "respond", "kind": "accept"},
            {"type": "respond", "kind": "decline"},
            {"type": "call", "kind": "truco", "amount": 6},
        ]
        for action in (0, 1, 2):
            counts.observe(observation, action)
        bucket = counts.counts["no_partner_claim"]
        self.assertEqual(bucket["response_opportunities"], 3)
        self.assertEqual(
            (
                bucket["response_accepts"],
                bucket["response_declines"],
                bucket["response_other"],
            ),
            (1, 1, 1),
        )
        self.assertEqual(
            counts.by_family["2/truco/0/False/False"],
            {"opportunities": 3, "accepts": 1, "declines": 1, "other": 1},
        )

    def test_combined_intervention_resume_preserves_actual_teacher_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            package = Path(__file__).resolve().parents[1] / "retruco_ml"
            # Other work may edit the checkout; this job's sources stay frozen.
            shutil.copytree(
                package,
                root / "source/retruco_ml",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            manifest = {
                "kind": "improvement-inputs-v1",
                "parents": {},
                "critics": {},
                "candidates": {},
            }
            actor, critic = Actor(True), Critic(True)

            def save(name, payload):
                p = inputs / name
                torch.save(payload, p)
                return {"file": name, "sha256": file_hash(p)}

            for seed in (101, 102, 103):
                manifest["parents"][str(seed)] = save(
                    f"actor-{seed}.pt", {"model": actor.state_dict()}
                )
                manifest["critics"][str(seed)] = save(
                    f"critic-{seed}.pt", {"critic": critic.state_dict()}
                )
            manifest["reference"] = save(
                "reference.pt", {"model": Policy().state_dict()}
            )
            manifest["candidates"]["legacy-champion"] = save(
                "starter.pt", migrate_legacy({"model": Policy().state_dict()})
            )
            manifest["candidates"]["d-parent"] = manifest["parents"]["101"]
            path = inputs / "manifest.json"
            path.write_text(json.dumps(manifest))
            spec = root / "partners.json"
            spec.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "entries": [
                            {
                                "id": "starter",
                                "family": "legacy",
                                "source": str(inputs / "starter.pt"),
                                "lineage": [
                                    manifest["candidates"]["legacy-champion"]["sha256"]
                                ],
                            }
                        ],
                    }
                )
            )
            partner_manifest = prepare_partner_pool(spec, root / "partners")
            hashes = {str(p): file_hash(p) for p in inputs.iterdir()}
            env = dict(os.environ, PYTHONPATH=str(root / "source"))

            def run(name, updates, resume=False, anchor=".1", entropy=".05"):
                command = [
                    sys.executable,
                    "-m",
                    "retruco_ml.improvement_train",
                    "--output",
                    str(root / name),
                    "--inputs",
                    str(path),
                    "--simulator",
                    str(package.parents[1] / "bin/selfplay"),
                    "--variant",
                    "D",
                    "--seed",
                    "401",
                    "--warm-start",
                    "legacy-champion",
                    "--actor-memory",
                    "64",
                    "--critic-hidden",
                    "256",
                    "128",
                    "--self-play-league",
                    "--cooperation-practice",
                    "--heldout-parent",
                    "103",
                    "--league-anchor-fraction",
                    ".2",
                    "--native-fraction",
                    ".5",
                    "--headsup-fraction",
                    ".4",
                    "--gae-lambda",
                    "1",
                    "--partner-manifest",
                    str(partner_manifest),
                    "--partner-policy-version",
                    "3",
                    "--entropy-coef",
                    entropy,
                    "--ppo-epochs",
                    "2",
                    "--extra-critic-epochs",
                    "1",
                    "--sequence-length",
                    "64",
                    "--critic-learning-rate",
                    ".0002",
                    "--parent-anchor",
                    ".01",
                    "--response-anchor",
                    anchor,
                    "--signal-anchor",
                    ".05",
                    "--choices",
                    "200000",
                    "--steps",
                    "32",
                    "--workers",
                    "1",
                    "--environments",
                    "1",
                    "--batch-size",
                    "8",
                    "--calibration-choices",
                    "16",
                    "--warmup-max",
                    "2",
                    "--max-updates",
                    str(updates),
                ]
                if resume:
                    command.append("--resume")
                return subprocess.run(command, env=env, capture_output=True, text=True)

            for name, updates, resume in [
                ("whole", 2, False),
                ("split", 1, False),
                ("split", 1, True),
            ]:
                result = run(name, updates, resume)
                self.assertEqual(result.returncode, 0, result.stderr[-2500:])
            a = torch.load(root / "whole/checkpoint.pt", weights_only=True)
            b = torch.load(root / "split/checkpoint.pt", weights_only=True)
            for field in ("model", "critic"):
                for k, v in a[field].items():
                    torch.testing.assert_close(v, b[field][k], atol=0, rtol=0)
            for field in ("actor_optimizer", "critic_optimizer"):
                for k, state in a[field]["state"].items():
                    for key, v in state.items():
                        torch.testing.assert_close(
                            v, b[field]["state"][k][key], atol=0, rtol=0
                        )
            self.assertEqual(a["random"], b["random"])
            self.assertEqual(a["optimizer_random"], b["optimizer_random"])
            self.assertNotEqual(a["random"], a["optimizer_random"])
            self.assertEqual(a["contracts"]["rng_protocol"], "separate-optimizer-v1")
            torch.testing.assert_close(
                a["torch_random"], b["torch_random"], atol=0, rtol=0
            )
            self.assertEqual(
                a["contracts"]["anchor_source_sha256"],
                manifest["candidates"]["legacy-champion"]["sha256"],
            )
            self.assertEqual(
                a["contracts"]["parent_sha256"],
                manifest["candidates"]["legacy-champion"]["sha256"],
            )
            self.assertNotIn(
                "frozen:D103", a["league"]["configuration"]["learned_controllers"]
            )
            self.assertEqual(
                json.loads(a["contracts"]["recipe"])["native_fraction"], 0.5
            )
            self.assertEqual(a["configuration"]["headsup_fraction"], 0.4)
            bad = run("split", 1, True, ".2")
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("resume requires exact", bad.stderr)
            bad = run("split", 1, True, entropy=".1")
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("resume requires exact", bad.stderr)
            self.assertEqual(hashes, {str(p): file_hash(p) for p in inputs.iterdir()})


if __name__ == "__main__":
    unittest.main()
