import copy
import random
import unittest

import numpy as np
import torch

from retruco_ml.model import PackedObservations, Policy, batch_distribution
from retruco_ml.train import update


def reference_distribution(model, observations):
    """Pre-optimization torch/list implementation as a numerical oracle."""
    maximum = max(6, max(len(o["actions"]) for o in observations))
    candidates = torch.zeros((len(observations), maximum, 48))
    mask = torch.zeros((len(observations), maximum), dtype=torch.bool)
    signals = torch.tensor([o["kind"] == "signal" for o in observations])
    for row, o in enumerate(observations):
        count = 6 if o["kind"] == "signal" else len(o["actions"])
        mask[row, :count] = True
        if o["kind"] != "signal":
            candidates[row, :count] = torch.tensor(o["actions"])
    game, signal, value = model(
        torch.tensor([o["features"] for o in observations]), candidates
    )
    logits = torch.where(
        signals[:, None], torch.nn.functional.pad(signal, (0, maximum - 6)), game
    ).masked_fill(~mask, -torch.inf)
    return torch.distributions.Categorical(logits=logits), value


def reference_update(model, optimizer, records, rng, epochs=4, batch_size=512):
    adv = torch.tensor([r[4] for r in records])
    normalized = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    indices = list(range(len(records)))
    for _ in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            batch = [records[i] for i in selected]
            distribution, value = reference_distribution(model, [r[0] for r in batch])
            logp = distribution.log_prob(torch.tensor([r[1] for r in batch]))
            ratio = (logp - torch.tensor([r[2] for r in batch])).exp()
            advantage = normalized[selected]
            loss = (
                -torch.minimum(
                    ratio * advantage, ratio.clamp(0.8, 1.2) * advantage
                ).mean()
                + 0.5 * (value - torch.tensor([r[5] for r in batch])).square().mean()
                - 0.01 * distribution.entropy().mean()
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
    return float(loss.detach())


def observations():
    rng = np.random.default_rng(731)
    return [
        {
            "version": 2,
            "kind": "signal" if count == 0 else "game",
            "features": rng.normal(size=768).astype(np.float32).tolist(),
            "actions": rng.normal(size=(count, 48)).astype(np.float32).tolist(),
        }
        for count in (0, 1, 3, 19, 0, 7, 1024)
    ]


class PackingTests(unittest.TestCase):
    def test_ragged_arrays_preserve_logits_masks_values_and_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)
        original = Policy()
        optimized = copy.deepcopy(original)
        rows = observations()
        expected, value = reference_distribution(original, rows)
        arrays = [
            {
                **o,
                "features": np.asarray(o["features"], np.float32),
                "actions": np.asarray(o["actions"], np.float32).reshape(-1, 48),
            }
            for o in rows
        ]
        actual, actual_value = batch_distribution(optimized, arrays)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        torch.testing.assert_close(actual_value, value, rtol=0, atol=0)
        (expected.entropy().sum() + value.sum()).backward()
        (actual.entropy().sum() + actual_value.sum()).backward()
        for left, right in zip(original.parameters(), optimized.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
        # Two outstanding forwards must not share a scratch buffer that gets
        # overwritten when a differently sized batch is packed.
        packed = PackedObservations(arrays)
        first, _ = packed.distribution(optimized, [6, 0])
        before = first.logits.detach().clone()
        packed.distribution(optimized, [2, 1, 4])
        torch.testing.assert_close(first.logits, before, rtol=0, atol=0)

    def test_ppo_continuation_preserves_optimizer_and_rng_from_existing_state(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        model = Policy()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        rows = observations()[:-1]
        with torch.no_grad():
            distribution, values = reference_distribution(model, rows)
        records = [
            (
                o,
                0,
                float(
                    distribution.log_prob(torch.zeros(len(rows), dtype=torch.long))[i]
                ),
                float(values[i]),
                (-1) ** i * 0.7,
                float(values[i]) + 0.2,
            )
            for i, o in enumerate(rows)
        ]
        # Warm Adam's moments before the resume comparison.
        reference_update(
            model, optimizer, records, random.Random(4), epochs=1, batch_size=3
        )
        resumed = Policy()
        resumed.load_state_dict(model.state_dict())
        resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=3e-4)
        resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        left_rng = random.Random(18)
        right_rng = random.Random(18)
        expected = reference_update(model, optimizer, records, left_rng, batch_size=3)
        actual = update(resumed, resumed_optimizer, records, right_rng, batch_size=3)
        self.assertEqual(expected, actual)
        self.assertEqual(left_rng.getstate(), right_rng.getstate())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
        for key, values in optimizer.state_dict()["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(
                    value,
                    resumed_optimizer.state_dict()["state"][key][name],
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
