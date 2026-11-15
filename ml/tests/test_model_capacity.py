import copy
import tempfile
import unittest

import torch

from retruco_ml.mappo_export import export
from retruco_ml.model_capacity import (
    MAX_ACTOR_PARAMETERS,
    actor_from_checkpoint,
    critic_from_checkpoint,
    widen_actor,
    widen_critic,
)
from retruco_ml.recurrent import Actor, Critic


class CapacityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(64)

    def test_widening_preserves_learned_policy_value_and_keeps_new_units_trainable(
        self,
    ):
        old = Actor(True)
        old.projection.weight.data.normal_(std=0.08)
        critic = Critic(True)
        actor = widen_actor(old, 64)
        large = widen_critic(critic, (256, 128))
        self.assertEqual(sum(p.numel() for p in actor.parameters()), 122567)
        self.assertEqual(sum(p.numel() for p in large.parameters()), 279041)
        self.assertLessEqual(
            sum(p.numel() for p in actor.parameters()), MAX_ACTOR_PARAMETERS
        )
        a, b = old.initial(), actor.initial()
        actions = torch.randn(37, 48)
        for _ in range(40):
            x = torch.randn(768)
            private = torch.randn(128)
            with torch.no_grad():
                a = old.advance(x, a)
                b = actor.advance(x, b)
                ca, cb = old.context(x, a), actor.context(x, b)
                torch.testing.assert_close(ca, cb, atol=2e-6, rtol=2e-5)
                for kind in ("game", "signal"):
                    obs = {"version": 2, "kind": kind, "actions": actions}
                    torch.testing.assert_close(
                        old.distribution(ca, obs).probs,
                        actor.distribution(cb, obs).probs,
                        atol=2e-7,
                        rtol=2e-5,
                    )
                torch.testing.assert_close(
                    critic(x, private, ca), large(x, private, cb), atol=2e-6, rtol=2e-5
                )
        b = actor.advance(x, actor.initial())
        context = actor.context(x, b)
        (
            -actor.distribution(
                context, {"version": 2, "kind": "signal", "actions": []}
            ).log_prob(torch.tensor(1))
        ).backward()
        self.assertGreater(float(actor.projection.weight.grad[:, 16:].abs().sum()), 0)
        large(x, private, context).backward()
        self.assertGreater(float(large.layers[4].weight.grad[:, 64:].abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in critic.parameters()))

    def test_checkpoint_capacity_is_inferred_and_validated_without_changing_old_states(
        self,
    ):
        for width, hidden in [(16, (96, 64)), (64, (256, 128))]:
            actor, critic = Actor(True, width), Critic(True, hidden)
            saved = {
                "model": copy.deepcopy(actor.state_dict()),
                "critic": copy.deepcopy(critic.state_dict()),
                "configuration": {"actor_memory": width, "critic_hidden": list(hidden)},
            }
            torch.testing.assert_close(
                actor_from_checkpoint(saved).projection.weight, actor.projection.weight
            )
            torch.testing.assert_close(
                critic_from_checkpoint(saved).layers[2].weight, critic.layers[2].weight
            )
            saved["configuration"]["actor_memory"] = 32
            with self.assertRaisesRegex(ValueError, "metadata"):
                actor_from_checkpoint(saved)
        with self.assertRaisesRegex(ValueError, "discard"):
            widen_actor(Actor(True, 64), 16)

    def test_lstm64_export_stays_within_download_budget_and_binds_architecture(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = export(Actor(True, 64), root)
            self.assertEqual(manifest["format"], 3)
            self.assertEqual(manifest["memory_size"], 64)
            self.assertEqual(manifest["parameters"], 122567)
            self.assertEqual(manifest["bytes"], 126248)
            self.assertLessEqual(manifest["bytes"], 128 * 1024)
            self.assertFalse(manifest["approved"])


if __name__ == "__main__":
    unittest.main()
