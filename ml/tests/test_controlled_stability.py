import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from test_recurrent_batch import fixture

from retruco_ml.controlled_compare import crossed_interval
from retruco_ml.learning_choices import choice_credit, meaningful
from retruco_ml.mappo_rollout import Decision, credit
from retruco_ml.recurrent import Actor, Critic
from retruco_ml.recurrent_batch import PackedRecords
from retruco_ml.recurrent_batch import update as original_update
from retruco_ml.stability import categorical_kl, update


class ControlledStabilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_forced_none_does_not_advance_credit_but_stays_in_memory(self):
        actor = Actor(True)
        torch.nn.init.normal_(actor.projection.weight, std=0.1)
        events = list(np.random.default_rng(9).normal(size=(4, 768)).astype(np.float32))
        records = []
        for i in range(4):
            observation = {
                "features": events[i],
                "kind": "game",
                "actions": np.zeros((2, 48), np.float32),
            }
            if i in (1, 2):
                observation.update(
                    kind="signal",
                    actions=[],
                    signalMask=[True, False, False, False, False, False],
                )
            records.append(
                Decision(
                    events, i + 1, observation, np.zeros(128, np.float32), 0, 0.0, 0.0
                )
            )
        original = copy.deepcopy(records)
        credit(original, 1)
        selected = choice_credit(records, 1)
        self.assertEqual([r.end for r in selected], [1, 4])
        self.assertAlmostEqual(selected[0].advantage, 0.95)
        self.assertAlmostEqual(original[0].advantage, 0.95**3)
        self.assertIs(selected[-1].events, events)
        packed = PackedRecords(selected)
        contexts, indices = packed.contexts(actor, packed.chunks)
        memory = actor.initial()
        for event in events:
            memory = actor.advance(torch.from_numpy(event), memory)
        expected = actor.context(torch.from_numpy(events[-1]), memory)
        torch.testing.assert_close(contexts[indices.index(1)], expected)

    def test_instrumented_control_preserves_original_ppo_distribution(self):
        actor = Actor(True)
        torch.nn.init.normal_(actor.projection.weight, std=0.02)
        critic = Critic(True)
        records = sum([fixture(actor, critic) for _ in range(4)], [])
        a, c = copy.deepcopy(actor), copy.deepcopy(critic)
        b, d = copy.deepcopy(actor), copy.deepcopy(critic)
        original_update(
            a,
            c,
            torch.optim.Adam([*a.parameters(), *c.parameters()], lr=1e-4),
            records,
            random.Random(2),
        )
        stats = update(
            b,
            d,
            torch.optim.Adam(b.parameters(), lr=1e-4),
            torch.optim.Adam(d.parameters(), lr=1e-4),
            records,
            random.Random(2),
            actor,
            stable=False,
        )
        packed = PackedRecords(records)
        left, indices = packed.contexts(a, packed.chunks)
        right, _ = packed.contexts(b, packed.chunks)
        torch.testing.assert_close(
            packed.distribution(a, left, indices).probs,
            packed.distribution(b, right, indices).probs,
            atol=2e-6,
            rtol=2e-5,
        )
        for x, y in zip(c.parameters(), d.parameters()):
            torch.testing.assert_close(x, y)
        self.assertTrue(np.isfinite(stats["all_choices"]["exact_kl"]))
        self.assertEqual(
            stats["all_choices"]["count"], sum(meaningful(r) for r in records)
        )

    def test_oversized_update_rolls_back_actor_and_reduces_learning_rate(self):
        actor = Actor(True)
        critic = Critic(True)
        records = sum([fixture(actor, critic) for _ in range(4)], [])
        parent = copy.deepcopy(actor)
        optimizer = torch.optim.Adam(actor.parameters(), lr=0.1)
        stats = update(
            actor,
            critic,
            optimizer,
            torch.optim.Adam(critic.parameters(), lr=1e-4),
            records,
            random.Random(3),
            parent,
            stable=True,
            target_kl=1e-6,
            epochs=1,
        )
        self.assertTrue(stats["actor_epoch_rolled_back"])
        self.assertLessEqual(stats["all_choices"]["exact_kl"], 1.5e-6)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)
        for x, y in zip(actor.parameters(), parent.parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertFalse(any(torch.isnan(p).any() for p in actor.parameters()))

    def test_critic_warmup_cannot_change_actor_or_create_actor_adam_state(self):
        actor = Actor(True)
        critic = Critic(True)
        parent = copy.deepcopy(actor)
        records = fixture(actor, critic)
        optimizer = torch.optim.Adam(actor.parameters(), lr=3e-5)
        update(
            actor,
            critic,
            optimizer,
            torch.optim.Adam(critic.parameters(), lr=1e-4),
            records,
            random.Random(4),
            parent,
            stable=True,
            policy_weight=0.0,
        )
        self.assertFalse(optimizer.state)
        for x, y in zip(actor.parameters(), parent.parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_masked_kl_stays_finite_for_underflowed_legal_probabilities(self):
        before = torch.distributions.Categorical(
            logits=torch.tensor([0.0, -1.0, -torch.inf])
        )
        logits = torch.tensor([-1200.0, 0.0, -torch.inf], requires_grad=True)
        after = torch.distributions.Categorical(logits=logits)
        divergence = categorical_kl(before, after)
        expected = before.probs[:2] @ (before.logits[:2] - after.logits[:2])
        torch.testing.assert_close(divergence, expected)
        self.assertTrue(torch.isfinite(divergence))
        divergence.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[2]), 0.0)

    def test_resuming_a_parent_campaign_is_rejected_without_mutating_it(self):
        from retruco_ml.controlled_campaign import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configuration.json").write_text('{"version":1}')
            status = root / "status.json"
            status.write_text('{"status":"complete"}')
            with patch.object(
                sys, "argv", ["campaign", "--output", directory, "--resume"]
            ):
                with self.assertRaisesRegex(ValueError, "parent runs are read-only"):
                    main()
            self.assertEqual(status.read_text(), '{"status":"complete"}')
            self.assertFalse((root / "campaign.lock").exists())

    def test_unseen_sparse_truthful_profile_overrides_the_default_signal_rate(self):
        from retruco_ml.cooperation_population import Profile, cooperation_scripted

        observation = {"kind": "signal", "view": {"config": {"players": 4}}}
        sparse = Profile(0.0, True, 28, 32, 0.3)
        with patch(
            "retruco_ml.cooperation_population.truthful_signals", return_value=[1]
        ):
            self.assertEqual(
                cooperation_scripted(
                    observation, "literal-v2", random.Random(5), profile=sparse
                ),
                0,
            )
            self.assertEqual(
                cooperation_scripted(observation, "literal-v2", random.Random(5)), 1
            )

    def test_bootstrap_retains_training_seed_uncertainty(self):
        self.assertEqual(crossed_interval(np.zeros((3, 16))), [0.0, 0.0])
        interval = crossed_interval(np.asarray([[-0.2] * 16, [0.0] * 16, [0.2] * 16]))
        self.assertLess(interval[0], 0)
        self.assertGreater(interval[1], 0)


if __name__ == "__main__":
    unittest.main()
