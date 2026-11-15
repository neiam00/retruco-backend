import copy
import math
import unittest

import numpy as np
import torch

from retruco_ml.recurrent import Actor
from retruco_ml.recurrent_batch import PackedRecords
from retruco_ml.rmappo_betting import ResponseAnchor
from retruco_ml.rmappo_rollout import JointDecision


def response_actions():
    # Independent examples of shared Go EncodeAction ABI v1: two accept
    # declarations, one decline, and two legal raise amounts.
    actions = np.zeros((5, 48), np.float32)
    actions[:, 5] = 1
    actions[:2, 11] = 1
    actions[2, 12] = 1
    actions[3:, 13] = 1
    actions[0, 24] = 1
    actions[1, 18] = 1
    actions[2:, 24] = 1
    actions[:, 27] = 1
    actions[3:, 35] = [0.06, 0.09]
    return actions


def record(actions=None, response=True, family="truco", features=None):
    features = np.zeros(768, np.float32) if features is None else features
    actions = response_actions() if actions is None else actions
    observation = {
        "version": 2,
        "kind": "game",
        "features": features,
        "actions": actions,
        "responseChoice": response,
        "responseFamily": family,
        "seat": 0,
    }
    return JointDecision(
        [features.copy()],
        1,
        observation,
        np.zeros(128, np.float32),
        0,
        0.0,
        0.0,
        advantage=1.0,
        target=1.0,
    )


class ResponseAnchorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(59)
        torch.set_num_threads(1)
        self.teacher = Actor(False)
        with torch.no_grad():
            for parameter in self.teacher.parameters():
                parameter.zero_()

    def test_semantic_anchor_keeps_declarations_and_raise_amounts_free(self):
        packed = PackedRecords([record()])
        anchor = (
            ResponseAnchor(self.teacher).prepare(packed).capture_before(self.teacher)
        )
        # Teacher masses are accept=.4, decline=.2, raise=.4. The student's
        # conditional choices differ considerably, but the semantic loss is 0.
        logits = (
            torch.tensor([[0.35, 0.05, 0.2, 0.1, 0.3]], dtype=torch.float64)
            .log()
            .requires_grad_()
        )
        loss = anchor.loss(torch.distributions.Categorical(logits=logits), [0], 1)
        self.assertAlmostEqual(loss.item(), 0.0, places=12)
        loss.backward()
        torch.testing.assert_close(
            logits.grad, torch.zeros_like(logits), atol=1e-12, rtol=0
        )
        result = anchor.diagnostics(self.teacher)
        self.assertEqual(result["response_count"], 1)
        self.assertAlmostEqual(result["teacher_group_mass"]["respond:decline"], 0.2)
        self.assertAlmostEqual(result["semantic_teacher_kl_before"], 0.0)
        self.assertAlmostEqual(result["weighted_anchor_loss"], 0.0)
        self.assertTrue(
            all(
                p.grad is None and not p.requires_grad
                for p in anchor.teacher.parameters()
            )
        )
        self.assertTrue(all(p.requires_grad for p in self.teacher.parameters()))

    def test_weighted_response_mean_is_not_diluted_by_cards_or_minibatches(self):
        actions = response_actions()[[0, 2]]
        card_actions = actions.copy()
        card_actions[:, :18] = 0
        card_actions[:, 1] = 1
        card_actions[:, 17] = 1
        card_actions[:, 32] = [0, 0.5]
        packed = PackedRecords(
            [
                record(actions),
                record(actions, family="envido"),
                record(card_actions, response=False),
            ]
        )
        anchor = ResponseAnchor(self.teacher).prepare(packed)
        logits = (
            torch.tensor([[0.8, 0.2], [0.25, 0.75], [0.3, 0.7]], dtype=torch.float64)
            .log()
            .requires_grad_()
        )
        distribution = torch.distributions.Categorical(logits=logits)
        weights = [1.0, 3.0, 80.0]
        expected = (
            0.1
            * (
                (-0.5 * math.log(4 * 0.8 * 0.2))
                + 3 * (-0.5 * math.log(4 * 0.25 * 0.75))
            )
            / 4
        )
        loss = anchor.loss(distribution, [0, 1, 2], 3, weights=weights)
        self.assertAlmostEqual(loss.item(), expected, places=12)
        # Summing contributions with the same denominator must be equivalent,
        # including a batch containing only an unanchored card decision.
        parts = [
            anchor.loss(
                torch.distributions.Categorical(logits=logits[i : i + 1]),
                [i],
                3,
                weights=weights,
            )
            for i in range(3)
        ]
        self.assertAlmostEqual(sum(parts).item(), expected, places=12)
        loss.backward()
        torch.testing.assert_close(
            logits.grad[2], torch.zeros(2, dtype=torch.float64), atol=0, rtol=0
        )
        self.assertGreater(float(logits.grad[0].abs().sum()), 0.0)
        self.assertTrue(all(p.grad is None for p in anchor.teacher.parameters()))

    def test_teacher_replays_authorized_history_and_ignores_privileged_channel(self):
        teacher = Actor(True)
        with torch.no_grad():
            teacher.projection.weight.fill_(0.1)
        rows = [record(features=np.full(768, 0.1, np.float32))]
        rows[0].events.insert(0, np.full(768, -0.2, np.float32))
        rows[0].end = 2
        packed = PackedRecords(rows)
        anchor = ResponseAnchor(teacher).prepare(packed)
        with torch.no_grad():
            memory = teacher.initial()
            for event in rows[0].events:
                memory = teacher.advance(torch.from_numpy(event), memory)
            context = teacher.context(
                torch.from_numpy(rows[0].observation["features"]), memory
            )
            probabilities = teacher.distribution(context, rows[0].observation).probs
        expected = {
            "respond:accept": float(probabilities[0] + probabilities[1]),
            "respond:decline": float(probabilities[2]),
            "respond:raise": float(probabilities[3] + probabilities[4]),
        }
        measured = anchor.diagnostics(teacher)["teacher_group_mass"]
        for group, value in expected.items():
            self.assertAlmostEqual(measured[group], value, places=9)
        changed = copy.deepcopy(rows)
        changed[0].privileged.fill(987.0)
        other = ResponseAnchor(teacher).prepare(PackedRecords(changed))
        self.assertEqual(measured, other.diagnostics(teacher)["teacher_group_mass"])

    def test_anchor_moves_response_probabilities_without_training_teacher(self):
        packed = PackedRecords([record(), record(family="envido")])
        anchor = ResponseAnchor(self.teacher).prepare(packed)
        logits = (
            torch.tensor(
                [[0.8, 0.1, 0.05, 0.025, 0.025], [0.1, 0.1, 0.7, 0.05, 0.05]],
                dtype=torch.float64,
            )
            .log()
            .requires_grad_()
        )
        optimizer = torch.optim.SGD([logits], lr=0.5)
        before = anchor.loss(
            torch.distributions.Categorical(logits=logits), [0, 1], 2
        ).item()
        for _ in range(12):
            loss = anchor.loss(
                torch.distributions.Categorical(logits=logits), [0, 1], 2
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        after = anchor.loss(
            torch.distributions.Categorical(logits=logits), [0, 1], 2
        ).item()
        self.assertLess(after, before)
        self.assertTrue(all(p.grad is None for p in anchor.teacher.parameters()))

    def test_nonresponse_and_zero_weight_batches_are_safe_with_padding(self):
        row = record(response=False)
        row.observation.update(
            kind="signal",
            actions=np.zeros((0, 48), np.float32),
            signalMask=[True, True, False, False, False, False],
        )
        packed = PackedRecords([row])
        anchor = ResponseAnchor(self.teacher).prepare(packed)
        logits = torch.tensor([[0.0, 0.0, -torch.inf]], requires_grad=True)
        loss = anchor.loss(torch.distributions.Categorical(logits=logits), [0], 1)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(anchor.diagnostics(self.teacher)["response_count"], 0)
        anchor.prepare(PackedRecords([record()]), weights=[0.0])
        loss = anchor.loss(
            torch.distributions.Categorical(logits=torch.zeros(1, 5)), [0], 1
        )
        self.assertEqual(loss.item(), 0.0)

    def test_response_authority_and_weight_contracts_fail_closed(self):
        bad = record()
        bad.observation["actions"] = response_actions()[[0, 1, 3, 4]]
        with self.assertRaisesRegex(ValueError, "accept and decline"):
            ResponseAnchor(self.teacher).prepare(PackedRecords([bad]))
        anchor = ResponseAnchor(self.teacher).prepare(PackedRecords([record()]))
        distribution = torch.distributions.Categorical(logits=torch.zeros(1, 5))
        for weights in ([-1.0], [float("nan")], [1.0, 2.0]):
            with self.assertRaisesRegex(ValueError, "full-rollout weights"):
                anchor.loss(distribution, [0], 1, weights=weights)


if __name__ == "__main__":
    unittest.main()
