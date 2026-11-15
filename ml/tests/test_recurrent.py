import copy
import random
import unittest

import numpy as np
import torch

from retruco_ml.mappo_rollout import Decision, credit
from retruco_ml.mappo_train import update
from retruco_ml.model import Policy
from retruco_ml.recurrent import Actor, Critic, Replay


class RecurrentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_warm_start_preserves_all_legal_logits_before_memory_training(self):
        parent = Policy()
        actor = Actor(True)
        actor.warm_start(parent.state_dict())
        self.assertEqual(sum(p.numel() for p in actor.parameters()), 91463)
        features = torch.randn(768)
        actions = torch.randn(1024, 48)
        memory = actor.initial()
        for _ in range(5):
            memory = actor.advance(torch.randn(768), memory)
        game, signals, _ = parent(features[None], actions[None])
        context = actor.context(features, memory)
        for kind, expected in [("game", game[0]), ("signal", signals[0])]:
            obs = {"version": 2, "kind": kind, "actions": actions.numpy()}
            actual = actor.distribution(context, obs).probs
            torch.testing.assert_close(actual, expected.softmax(-1), check_dtype=False)

    def test_replay_is_exactly_once_and_recovers_after_worker_restart(self):
        actor = Actor(True)
        actor.projection.weight.data.fill_(0.02)
        events = np.random.default_rng(2).normal(size=(40, 768)).astype(np.float32)

        def page(start, end, hand=1):
            return {
                "version": 1,
                "complete": True,
                "hand": hand,
                "start": start,
                "end": end,
                "total": 40,
                "events": events[start:end],
            }

        continuous = Replay(actor)
        continuous.apply(page(0, 32))
        continuous.apply(page(32, 40))
        restored = Replay(actor)
        restored.apply(page(0, 20))
        restored.apply(page(0, 20))
        restored.apply(page(20, 40))
        for a, b in zip(continuous.memory, restored.memory):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(len(restored.events), 40)
        with self.assertRaisesRegex(ValueError, "start at zero"):
            Replay(actor).apply(page(5, 10))
        bad = page(0, 1)
        bad["complete"] = False
        with self.assertRaisesRegex(ValueError, "incomplete"):
            restored.apply(bad)
        restored.apply(page(0, 2, 2))
        self.assertEqual(len(restored.events), 2)

    def test_privileged_critic_cannot_change_actor_or_backpropagate_into_it(self):
        actor = Actor(True)
        critic = Critic(True)
        features = torch.randn(768)
        context = actor.context(features, actor.initial())
        private = torch.randn(128)
        value = critic(features, private, context)
        other = critic(features, private + 1, context)
        self.assertNotEqual(float(value.detach()), float(other.detach()))
        value.backward()
        self.assertTrue(all(p.grad is None for p in actor.parameters()))
        local = Critic(False)
        torch.testing.assert_close(
            local(features, private, context),
            local(features, private + 1, context),
            rtol=0,
            atol=0,
        )

    def test_sequence_update_learns_and_credit_continues_across_hand_resets(self):
        for recurrent in (False, True):
            actor = Actor(recurrent)
            critic = Critic(True)
            records = []
            for hand in range(2):
                events = [
                    np.random.default_rng(hand + i).normal(size=768).astype(np.float32)
                    for i in range(4)
                ]
                memory = actor.initial()
                for i, event in enumerate(events):
                    with torch.no_grad():
                        memory = actor.advance(torch.as_tensor(event), memory)
                        context = actor.context(torch.as_tensor(event), memory)
                        obs = {
                            "version": 2,
                            "kind": "signal",
                            "features": event,
                            "actions": [],
                            "seat": 0,
                        }
                        d = actor.distribution(context, obs)
                        private = np.zeros(128, np.float32)
                        v = critic(
                            torch.as_tensor(event), torch.as_tensor(private), context
                        )
                    records.append(
                        Decision(
                            events,
                            i + 1,
                            obs,
                            private,
                            1,
                            float(d.log_prob(torch.tensor(1))),
                            float(v),
                        )
                    )
            credit(records, 1, lam=1)
            np.testing.assert_allclose([r.target for r in records], 1, atol=1e-6)
            before = copy.deepcopy(actor.state_dict())
            loss = update(
                actor,
                critic,
                torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=3e-4),
                records,
                random.Random(3),
                epochs=2,
            )
            self.assertTrue(np.isfinite(loss))
            self.assertTrue(
                any(
                    not torch.equal(before[k], v) for k, v in actor.state_dict().items()
                )
            )


if __name__ == "__main__":
    unittest.main()
