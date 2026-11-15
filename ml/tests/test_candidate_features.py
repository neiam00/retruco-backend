import copy
import math
import unittest
from types import SimpleNamespace

import torch

from retruco_ml.candidate_features import augment
from retruco_ml.recurrent_batch import candidate_distribution
from retruco_ml.rmappo_model import RecurrentActor, actor_from_checkpoint
from retruco_ml.semantic_exploration import SemanticExploration


def features(players=4):
    f = torch.zeros(768)
    f[0] = players / 4
    f[1] = 1
    f[3] = 0.5
    f[4] = 0.5
    f[13] = 1
    for i, rank in enumerate((0.2, 0.7, 1.0)):
        f[21 + 5 * i] = 1
        f[24 + 5 * i] = rank
    f[36] = 0.8
    f[37] = 0.7
    return f


def play(f, seat, round, rank, burn=False, hidden=False):
    start = 39 + seat * 22 + 1 + round * 7
    f[start : start + 7] = torch.tensor(
        [1.0, 0.0, 0.0, rank, 1.0, float(burn), float(hidden)]
    )


def action(kind, card=0):
    a = torch.zeros(48)
    a[5 if kind in ("accept", "decline", "raise") else 1] = 1
    a[{"accept": 11, "decline": 12, "raise": 13}.get(kind, 17)] = 1
    a[24] = 1
    a[27] = 1
    a[32] = card / 2
    return a


class CandidateFeatureTests(unittest.TestCase):
    def test_public_certainty_requires_complete_first_trick_and_visible_deciding_perico(
        self,
    ):
        f = features()
        f[127] = 1
        f[128] = 0.2
        f[130] = 0.04
        f[131] = 0.01
        for seat, rank in enumerate((0.9, 0.2, 0.4, 0.3)):
            play(f, seat, 0, rank)
        play(f, 2, 1, 1.0)
        actions = torch.stack([action("accept"), action("decline")])
        x = augment(f.expand(2, -1), actions)
        self.assertTrue(torch.equal(x[:, 44], torch.ones(2)))
        self.assertAlmostEqual(float(x[0, 46]), 4 / 24, places=6)
        self.assertAlmostEqual(float(x[1, 46]), 1 / 24, places=6)
        for change in ("conceal", "ordinary", "missing_first"):
            bad = f.clone()
            if change == "conceal":
                play(bad, 2, 1, 1.0, hidden=True)
            elif change == "ordinary":
                play(bad, 2, 1, 0.94)
            else:
                bad[39 + 1 + 4] = 0
            self.assertEqual(float(augment(bad[None, :], actions[:1])[0, 44]), 0)
        play(f, 2, 1, 0.0, hidden=True)
        play(f, 1, 1, 1.0)
        self.assertEqual(
            float(augment(f[None, :], actions[:1])[0, 44]),
            0,
            "winning second alone does not settle the hand",
        )
        f[4] = 1
        for seat, rank in enumerate((0.1, 0.9, 0.4, 0.2)):
            play(f, seat, 1, rank)
        play(f, 1, 2, 1.0)
        self.assertEqual(float(augment(f[None, :], actions[:1])[0, 44]), -1)

    def test_card_conservation_features_work_in_both_modes_and_ignore_inactive_seats(
        self,
    ):
        for players in (2, 4):
            f = features(players)
            play(f, 1, 1, 0.6)
            play(f, 2, 1, 0.8)
            a = torch.stack([action("play", 0), action("play", 2), action("accept")])
            x = augment(f.expand(3, -1), a)
            torch.testing.assert_close(x[:2, 36], torch.tensor([0.2, 1.0]))
            torch.testing.assert_close(x[:2, 38], torch.tensor([0.8, 0.0]))
            self.assertEqual(float(x[0, 40]), float(players == 4))
            torch.testing.assert_close(x[2, 36:44], torch.zeros(8))
        with self.assertRaises(ValueError):
            augment(torch.zeros(1, 896), action("play")[None, :])

    def test_migration_preserves_initial_probabilities_and_checkpoint_restores_feature_contract(
        self,
    ):
        torch.manual_seed(21)
        actor = RecurrentActor("lstm")
        enhanced = copy.deepcopy(actor).enable_candidate_features()
        f = features()
        a = torch.stack([action("play", i) for i in range(3)])
        obs = {
            "kind": "game",
            "version": 2,
            "features": f.numpy(),
            "actions": a.numpy(),
        }
        context = actor.context(f, actor.initial())
        before = actor.distribution(context, obs)
        after = enhanced.distribution(context, obs)
        torch.testing.assert_close(before.logits, after.logits, atol=0, rtol=0)
        self.assertEqual(
            sum(x.numel() for x in actor.parameters()),
            sum(x.numel() for x in enhanced.parameters()),
        )
        batch = candidate_distribution(enhanced, context[None, :], [obs])
        torch.testing.assert_close(batch.logits[0], after.logits)
        (-batch.log_prob(torch.tensor([2]))).backward()
        self.assertGreater(float(enhanced.action[0].weight.grad[:, 36:].abs().sum()), 0)
        saved = {
            "kind": "rmappo-v1",
            "configuration": {"cell": "lstm", "candidate_features": 1},
            "actor": enhanced.state_dict(),
        }
        loaded = actor_from_checkpoint(saved)
        self.assertEqual(loaded.candidate_features, 1)
        torch.testing.assert_close(
            loaded.distribution(context, obs).logits, after.logits
        )
        saved["configuration"]["candidate_features"] = 2
        with self.assertRaises(ValueError):
            actor_from_checkpoint(saved)


class SemanticExplorationTests(unittest.TestCase):
    def prepared(self, actions):
        obs = {
            "kind": "game",
            "responseChoice": True,
            "actions": torch.stack(actions).numpy(),
        }
        return SemanticExploration(0.02).prepare(
            SimpleNamespace(records=[SimpleNamespace(observation=obs)])
        )

    def test_collapsed_legal_response_keeps_finite_nonvanishing_gradient(self):
        loss = self.prepared([action("accept"), action("decline"), action("raise")])
        logits = torch.tensor(
            [[0.0, -1000.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        value = loss.loss(torch.distributions.Categorical(logits=logits), [0], 1)
        self.assertTrue(torch.isfinite(value))
        value.backward()
        self.assertLess(float(logits.grad[0, 1]), -0.005)

    def test_semantic_loss_is_not_biased_by_duplicate_declaration_variants(self):
        original = self.prepared([action("accept"), action("decline"), action("raise")])
        duplicate = self.prepared(
            [action("accept"), action("accept"), action("decline"), action("raise")]
        )
        a = original.loss(
            torch.distributions.Categorical(
                logits=torch.zeros(1, 3, dtype=torch.float64)
            ),
            [0],
            1,
        )
        b = duplicate.loss(
            torch.distributions.Categorical(
                logits=torch.tensor(
                    [[-math.log(2), -math.log(2), 0.0, 0.0]], dtype=torch.float64
                )
            ),
            [0],
            1,
        )
        self.assertAlmostEqual(float(a), 0, places=12)
        self.assertAlmostEqual(float(b), 0, places=12)


if __name__ == "__main__":
    unittest.main()
