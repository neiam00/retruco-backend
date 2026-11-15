import copy
import random
import unittest

import numpy as np
import torch

from retruco_ml.mappo_rollout import Decision, credit
from retruco_ml.mappo_train import update as scalar_update
from retruco_ml.recurrent import Actor, Critic
from retruco_ml.recurrent_batch import PackedRecords, update


def fixture(actor, critic):
    records = []
    rng = np.random.default_rng(78)
    for length, ends in [(40, [1, 5, 31, 35, 40]), (7, [2, 7]), (62, [1, 34, 40, 62])]:
        events = list(rng.normal(size=(length, 768)).astype(np.float32))
        memory = actor.initial()
        for i, event in enumerate(events):
            with torch.no_grad():
                memory = actor.advance(torch.from_numpy(event), memory)
                if i + 1 not in ends:
                    continue
                signal = i % 2 == 0
                obs = {
                    "version": 2,
                    "kind": "signal" if signal else "game",
                    "features": event,
                    "actions": []
                    if signal
                    else rng.normal(size=(3 + i % 5, 48)).astype(np.float32),
                    "seat": 0,
                }
                if signal:
                    obs["signalMask"] = [True, False, True, False, False, False]
                context = actor.context(torch.from_numpy(event), memory)
                dist = actor.distribution(context, obs)
                action = 2 if signal else 1
                private = rng.normal(size=128).astype(np.float32)
                value = critic(
                    torch.from_numpy(event), torch.from_numpy(private), context
                )
                records.append(
                    Decision(
                        events,
                        i + 1,
                        obs,
                        private,
                        action,
                        float(dist.log_prob(torch.tensor(action))),
                        float(value),
                    )
                )
    credit(records, 1)
    return records


class RecurrentBatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(5)

    def test_batched_replay_candidates_and_nonzero_memory_gradients_match_scalar(self):
        actor = Actor(True)
        torch.nn.init.normal_(actor.projection.weight, std=0.02)
        critic = Critic(True)
        records = fixture(actor, critic)
        packed = PackedRecords(records)
        contexts, indices = packed.contexts(actor, packed.chunks)
        distribution = packed.distribution(actor, contexts, indices)
        scalar = []
        for i in indices:
            r = records[i]
            memory = actor.initial()
            lower = ((r.end - 1) // 32) * 32
            with torch.no_grad():
                for event in r.events[:lower]:
                    memory = actor.advance(torch.from_numpy(event), memory)
            for event in r.events[lower : r.end]:
                memory = actor.advance(torch.from_numpy(event), memory)
            scalar.append(
                actor.context(torch.from_numpy(r.observation["features"]), memory)
            )
        torch.testing.assert_close(contexts, torch.stack(scalar), atol=2e-7, rtol=2e-5)
        expected = torch.stack(
            [
                actor.distribution(c, records[i].observation).log_prob(
                    torch.tensor(records[i].action)
                )
                for c, i in zip(scalar, indices)
            ]
        )
        torch.testing.assert_close(
            distribution.log_prob(packed.actions[indices]),
            expected,
            atol=5e-7,
            rtol=2e-5,
        )
        old_actor = copy.deepcopy(actor)
        old_critic = copy.deepcopy(critic)
        scalar_update(
            old_actor,
            old_critic,
            torch.optim.Adam(
                [*old_actor.parameters(), *old_critic.parameters()], lr=1e-4
            ),
            records,
            random.Random(3),
            epochs=1,
        )
        update(
            actor,
            critic,
            torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=1e-4),
            records,
            random.Random(3),
            epochs=1,
        )
        for key, value in actor.state_dict().items():
            # The final scorer bias is a common logit offset: its theoretical
            # policy gradient is zero, but tiny reduction errors affect Adam.
            if key != "score.2.bias":
                torch.testing.assert_close(
                    value, old_actor.state_dict()[key], atol=2e-6, rtol=2e-4
                )
        for key, value in critic.state_dict().items():
            torch.testing.assert_close(
                value, old_critic.state_dict()[key], atol=2e-6, rtol=2e-4
            )
        for r in records:
            probabilities = []
            for model in (actor, old_actor):
                memory = model.initial()
                with torch.no_grad():
                    for event in r.events[: r.end]:
                        memory = model.advance(torch.from_numpy(event), memory)
                    probabilities.append(
                        model.distribution(
                            model.context(
                                torch.from_numpy(r.observation["features"]), memory
                            ),
                            r.observation,
                        ).probs
                    )
            torch.testing.assert_close(*probabilities, atol=2e-6, rtol=2e-4)
        self.assertGreater(float(actor.memory.weight_ih.grad.abs().sum()), 0)

    def test_value_warmup_preserves_actor_and_adam_actor_state(self):
        actor = Actor(True)
        critic = Critic(True)
        records = fixture(actor, critic)
        original = copy.deepcopy(actor.state_dict())
        optimizer = torch.optim.Adam(
            [*actor.parameters(), *critic.parameters()], lr=1e-4
        )
        update(
            actor,
            critic,
            optimizer,
            records,
            random.Random(2),
            epochs=1,
            policy_weight=0,
        )
        for key, value in actor.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]))
        self.assertTrue(
            all(parameter not in optimizer.state for parameter in actor.parameters())
        )


if __name__ == "__main__":
    unittest.main()
