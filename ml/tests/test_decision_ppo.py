import copy
import random
import unittest
from unittest.mock import patch

import torch
from test_recurrent_batch import fixture

from retruco_ml.decision_ppo import decision_batches, update
from retruco_ml.improvement_inputs import migrate_legacy
from retruco_ml.model import Policy
from retruco_ml.recurrent import Actor, Critic
from retruco_ml.recurrent_batch import PackedRecords


class DecisionPPOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_balanced_batches_cover_every_decision_with_identical_weight(self):
        actor, critic = Actor(True), Critic(True)
        records = sum([fixture(actor, critic) for _ in range(10)], [])
        packed = PackedRecords(records)
        batches = decision_batches(packed, random.Random(8), 64)
        indices = [
            index for groups, _ in batches for group in groups for index in group
        ]
        self.assertEqual(sorted(indices), list(range(len(records))))
        sizes = [sum(len(group) for group in groups) for groups, _ in batches]
        self.assertLessEqual(max(sizes) - min(sizes), 1)
        self.assertGreater(min(sizes), 1)
        self.assertEqual(len({denominator for _, denominator in batches}), 1)

    def test_partitioned_recurrent_gradients_equal_uniform_full_decision_objective(
        self,
    ):
        actor, critic = Actor(True), Critic(True)
        torch.nn.init.normal_(actor.projection.weight, std=0.04)
        records = fixture(actor, critic)
        packed = PackedRecords(records)
        a, c = copy.deepcopy(actor), copy.deepcopy(critic)
        b, d = copy.deepcopy(actor), copy.deepcopy(critic)

        def objective(policy, value, groups, denominator):
            contexts, indices = packed.contexts(policy, groups)
            dist = packed.distribution(policy, contexts, indices)
            ratio = (dist.log_prob(packed.actions[indices]) - packed.old[indices]).exp()
            advantage = packed.advantages[indices]
            actor_loss = -torch.minimum(
                ratio * advantage, ratio.clamp(0.8, 1.2) * advantage
            )
            prediction = value(
                packed.features[indices], packed.privileged[indices], contexts
            )
            return (
                (actor_loss - 0.01 * dist.entropy()).sum()
                + 0.5 * (prediction - packed.targets[indices]).square().sum()
            ) / denominator

        objective(a, c, packed.chunks, len(records)).backward()
        batches = decision_batches(packed, random.Random(4), 4)
        for groups, denominator in batches:
            (objective(b, d, groups, denominator) / len(batches)).backward()
        for left, right in zip(
            [*a.parameters(), *c.parameters()], [*b.parameters(), *d.parameters()]
        ):
            torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-4)

    def test_signal_triggered_stop_never_increases_learning_rate(self):
        actor, critic = Actor(True), Critic(True)
        parent = copy.deepcopy(actor)
        records = fixture(actor, critic)
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
        with patch(
            "retruco_ml.decision_ppo.policy_kl",
            return_value={"all": 0.001, "game": 0.001, "signal": 0.04},
        ):
            result = update(
                actor,
                critic,
                optimizer,
                torch.optim.Adam(critic.parameters(), lr=1e-4),
                records,
                random.Random(5),
                parent,
                batch_size=4,
                epochs=1,
            )
        self.assertTrue(result["actor_early_stopped"])
        self.assertLess(result["actor_lr_next"], result["actor_lr_used"])

    def test_replay_log_probability_mismatch_is_rejected_before_update(self):
        actor, critic = Actor(True), Critic(True)
        records = fixture(actor, critic)
        records[0].logp += 0.01
        before = copy.deepcopy(actor.state_dict())
        for invalid in (records[0].logp, float("nan"), float("inf")):
            records[0].logp = invalid
            with (
                self.subTest(logp=invalid),
                self.assertRaisesRegex(ValueError, "replay policy mismatch"),
            ):
                update(
                    actor,
                    critic,
                    torch.optim.Adam(actor.parameters()),
                    torch.optim.Adam(critic.parameters()),
                    records,
                    random.Random(9),
                    actor,
                )
        for name, value in actor.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_response_only_kl_rollback_requires_sufficient_opportunities(self):
        for minimum, expected_rollback in ((1, True), (100, False)):
            with self.subTest(minimum=minimum):
                actor, critic = Actor(True), Critic(True)
                records = fixture(actor, critic)
                for record in records:
                    record.observation["responseChoice"] = (
                        record.observation["kind"] == "game"
                    )
                before = copy.deepcopy(actor.state_dict())
                optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
                divergence = {
                    "all": 0.001,
                    "game": 0.001,
                    "signal": 0.001,
                    "response": 0.08,
                }
                with patch(
                    "retruco_ml.decision_ppo.policy_kl", return_value=divergence
                ):
                    result = update(
                        actor,
                        critic,
                        optimizer,
                        torch.optim.Adam(critic.parameters()),
                        records,
                        random.Random(5),
                        copy.deepcopy(actor),
                        epochs=1,
                        response_min_choices=minimum,
                    )
                self.assertEqual(result["actor_epoch_rolled_back"], expected_rollback)
                self.assertEqual(
                    result["response_kl_guard"]["enabled"], expected_rollback
                )
                self.assertTrue(result["response_kl_guard"]["warning"])
                if expected_rollback:
                    for name, value in actor.state_dict().items():
                        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
                    self.assertFalse(optimizer.state)
                    self.assertEqual(result["actor_lr_next"], 5e-5)
                else:
                    self.assertTrue(
                        any(
                            not torch.equal(value, before[name])
                            for name, value in actor.state_dict().items()
                        )
                    )

    def test_response_kl_affects_learning_rate_below_stop_threshold(self):
        from retruco_ml.stability import diagnostics

        actor, critic = Actor(True), Critic(True)
        records = fixture(actor, critic)
        for record in records:
            record.observation["responseChoice"] = record.observation["kind"] == "game"
        divergence = {"all": 0.001, "game": 0.001, "signal": 0.001, "response": 0.029}

        def measured(*args, **kwargs):
            result = diagnostics(*args, **kwargs)
            for kind in ("all_choices", "game", "signal"):
                result[kind]["exact_kl"] = 0.001
            result["response"]["exact_kl"] = 0.029
            return result

        with (
            patch("retruco_ml.decision_ppo.policy_kl", return_value=divergence),
            patch("retruco_ml.decision_ppo.diagnostics", side_effect=measured),
        ):
            result = update(
                actor,
                critic,
                torch.optim.Adam(actor.parameters(), lr=1e-4),
                torch.optim.Adam(critic.parameters()),
                records,
                random.Random(5),
                copy.deepcopy(actor),
                epochs=1,
                response_min_choices=1,
                response_kl_target=0.01,
            )
        self.assertFalse(result["actor_early_stopped"])
        self.assertLess(result["actor_lr_next"], result["actor_lr_used"])

    def test_entropy_control_changes_policy_when_advantages_are_zero(self):
        original, original_critic = Actor(True), Critic(True)
        with torch.no_grad():
            original.signal.bias[0] = 2.0
        records = fixture(original, original_critic)
        for record in records:
            record.advantage = 0.0
        outcomes = []
        for entropy in (0.0, 0.1):
            actor, critic = copy.deepcopy(original), copy.deepcopy(original_critic)
            result = update(
                actor,
                critic,
                torch.optim.Adam(actor.parameters(), lr=1e-4),
                torch.optim.Adam(critic.parameters()),
                records,
                random.Random(5),
                original,
                epochs=1,
                batch_size=64,
                entropy_coef=entropy,
            )
            outcomes.append(result["all_choices"]["entropy"])
            if entropy == 0.0:
                for name, value in actor.state_dict().items():
                    torch.testing.assert_close(
                        value, original.state_dict()[name], atol=0, rtol=0
                    )
        self.assertGreater(outcomes[1], outcomes[0])

    def test_extra_critic_epochs_improve_fit_without_changing_actor_or_adam(self):
        original, original_critic = Actor(True), Critic(True)
        torch.nn.init.normal_(original.projection.weight, std=0.04)
        records = fixture(original, original_critic)
        results = []
        for extra in (0, 2):
            actor, critic = copy.deepcopy(original), copy.deepcopy(original_critic)
            optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
            result = update(
                actor,
                critic,
                optimizer,
                torch.optim.Adam(critic.parameters(), lr=1e-3),
                records,
                random.Random(5),
                original,
                epochs=1,
                batch_size=64,
                extra_critic_epochs=extra,
                actor_max_grad_norm=0.02,
                critic_max_grad_norm=0.03,
            )
            self.assertLessEqual(
                float(
                    torch.linalg.vector_norm(
                        torch.cat(
                            [
                                p.grad.flatten()
                                for p in actor.parameters()
                                if p.grad is not None
                            ]
                        )
                    )
                ),
                0.020001,
            )
            self.assertLessEqual(
                float(
                    torch.linalg.vector_norm(
                        torch.cat(
                            [
                                p.grad.flatten()
                                for p in critic.parameters()
                                if p.grad is not None
                            ]
                        )
                    )
                ),
                0.030001,
            )
            results.append((actor.state_dict(), optimizer.state_dict(), result))
        for name, value in results[0][0].items():
            torch.testing.assert_close(value, results[1][0][name], atol=0, rtol=0)
        self.assertEqual(results[0][1]["param_groups"], results[1][1]["param_groups"])
        for key, state in results[0][1]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(
                    value, results[1][1]["state"][key][name], atol=0, rtol=0
                )
        self.assertGreater(
            results[1][2]["critic_optimizer_steps"],
            results[0][2]["critic_optimizer_steps"],
        )
        self.assertLess(
            results[1][2]["all_choices"]["value_mse"],
            results[0][2]["all_choices"]["value_mse"],
        )

    def test_invalid_update_controls_fail_before_parameters_or_optimizers_change(self):
        actor, critic = Actor(True), Critic(True)
        records = fixture(actor, critic)
        before = copy.deepcopy(actor.state_dict())
        actor_optimizer, critic_optimizer = (
            torch.optim.Adam(actor.parameters()),
            torch.optim.Adam(critic.parameters()),
        )
        invalid = (
            {"entropy_coef": float("nan")},
            {"entropy_coef": -0.1},
            {"clip_range": 0.0},
            {"clip_range": 1.0},
            {"actor_max_grad_norm": 0.0},
            {"critic_max_grad_norm": float("inf")},
            {"epochs": 1.5},
            {"extra_critic_epochs": -1},
            {"response_min_choices": 0},
            {"response_kl_stop": 0.01},
            {"response_kl_rollback": float("nan")},
            {"sequence": 0},
            {"sequence": 257},
            {"sequence": 32.5},
            {"sequence": True},
        )
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                update(
                    actor,
                    critic,
                    actor_optimizer,
                    critic_optimizer,
                    records,
                    random.Random(5),
                    actor,
                    **settings,
                )
        for name, value in actor.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        self.assertFalse(actor_optimizer.state)
        self.assertFalse(critic_optimizer.state)

    def test_clip_range_affects_reused_samples_after_identical_first_step(self):
        original, original_critic = Actor(True), Critic(True)
        records = fixture(original, original_critic)
        for epochs in (1, 2):
            outcomes = []
            for clip in (1e-5, 0.2):
                actor, critic = copy.deepcopy(original), copy.deepcopy(original_critic)
                update(
                    actor,
                    critic,
                    torch.optim.SGD(actor.parameters(), lr=0.01),
                    torch.optim.SGD(critic.parameters(), lr=0.001),
                    records,
                    random.Random(5),
                    original,
                    epochs=epochs,
                    batch_size=64,
                    entropy_coef=0.0,
                    clip_range=clip,
                )
                outcomes.append(actor.state_dict())
            equal = all(
                torch.equal(value, outcomes[1][name])
                for name, value in outcomes[0].items()
            )
            self.assertEqual(equal, epochs == 1)

    def test_longer_sequence_learns_across_boundary_and_recomputes_authorized_memory(
        self,
    ):
        actor, critic = Actor(True), Critic(True)
        torch.nn.init.normal_(actor.projection.weight, std=0.04)
        record = next(record for record in fixture(actor, critic) if record.end == 35)
        gradients = {}
        for sequence in (32, 64):
            packed = PackedRecords([record], sequence=sequence)
            events = packed.events[id(record.events)].requires_grad_(True)
            contexts, indices = packed.contexts(actor, packed.chunks)
            dist = packed.distribution(actor, contexts, indices)
            self.assertAlmostEqual(
                float(dist.log_prob(packed.actions)[0].detach()),
                record.logp,
                delta=2e-6,
            )
            gradients[sequence] = torch.autograd.grad(
                -dist.log_prob(packed.actions).sum(), events
            )[0][31]
        self.assertEqual(float(gradients[32].abs().sum()), 0.0)
        self.assertGreater(float(gradients[64].abs().sum()), 1e-8)

        packed = PackedRecords([record], sequence=32)
        before, _ = packed.contexts(actor, packed.chunks)
        with torch.no_grad():
            actor.memory.bias_ih.add_(0.1)
        after, _ = packed.contexts(actor, packed.chunks)
        memory = actor.initial()
        with torch.no_grad():
            for event in record.events[: record.end]:
                memory = actor.advance(torch.from_numpy(event), memory)
            expected = actor.context(
                torch.from_numpy(record.observation["features"]), memory
            )
        self.assertFalse(torch.allclose(before, after, atol=1e-7, rtol=1e-6))
        torch.testing.assert_close(after[0], expected, atol=2e-7, rtol=2e-5)

    def test_update_sequence_control_changes_recurrent_learning_not_initial_policy(
        self,
    ):
        original, original_critic = Actor(True), Critic(True)
        torch.nn.init.normal_(original.projection.weight, std=0.04)
        records = fixture(original, original_critic)
        gradients = []
        for sequence in (32, 64):
            actor, critic = copy.deepcopy(original), copy.deepcopy(original_critic)
            result = update(
                actor,
                critic,
                torch.optim.SGD(actor.parameters(), lr=0.001),
                torch.optim.SGD(critic.parameters(), lr=0.001),
                records,
                random.Random(5),
                original,
                epochs=1,
                batch_size=64,
                sequence=sequence,
            )
            self.assertLess(result["replay_identity_max_logp_error"], 2e-4)
            self.assertEqual(result["update_controls"]["sequence"], sequence)
            gradients.append(actor.memory.weight_ih.grad.clone())
        self.assertGreater(float((gradients[0] - gradients[1]).abs().sum()), 1e-7)

    def test_rare_response_family_drift_cannot_hide_in_aggregate_kl(self):
        for minimum, expected in ((1, True), (2, False)):
            with self.subTest(minimum=minimum):
                actor, critic = Actor(True), Critic(True)
                records = fixture(actor, critic)
                responses = [r for r in records if r.observation["kind"] == "game"]
                for index, record in enumerate(responses):
                    record.observation.update(
                        responseChoice=True,
                        responseFamily="truco" if index == 0 else "envido",
                    )
                divergence = {
                    "all": 0.001,
                    "game": 0.001,
                    "signal": 0.001,
                    "response": 0.002,
                    "response:truco": 0.08,
                    "response:envido": 0.001,
                }
                with patch(
                    "retruco_ml.decision_ppo.policy_kl", return_value=divergence
                ):
                    result = update(
                        actor,
                        critic,
                        torch.optim.Adam(actor.parameters(), lr=1e-4),
                        torch.optim.Adam(critic.parameters()),
                        records,
                        random.Random(5),
                        copy.deepcopy(actor),
                        epochs=1,
                        response_min_choices=minimum,
                    )
                self.assertEqual(result["actor_epoch_rolled_back"], expected)
                family = result["response_kl_guard"]["families"]["truco"]
                self.assertEqual(family["choices"], 1)
                self.assertEqual(family["enabled"], expected)
                self.assertEqual(family["max_observed_kl"], 0.08)

    def test_family_kl_matches_its_own_legal_response_distributions(self):
        from retruco_ml.decision_ppo import policy_kl
        from retruco_ml.stability import cache_logits

        actor, critic = Actor(True), Critic(True)
        records = fixture(actor, critic)
        responses = [
            i for i, r in enumerate(records) if r.observation["kind"] == "game"
        ]
        for position, index in enumerate(responses):
            records[index].observation.update(
                responseChoice=True,
                responseFamily="truco" if position < 2 else "envido",
            )
        packed = PackedRecords(records)
        before = cache_logits(actor, packed)
        with torch.no_grad():
            actor.score[2].weight.mul_(2)
        after = cache_logits(actor, packed)
        actual = policy_kl(actor, packed, before)
        for family, indices in (("truco", responses[:2]), ("envido", responses[2:])):
            expected = sum(
                float((before[i].exp() * (before[i] - after[i])).sum()) for i in indices
            ) / len(indices)
            self.assertGreater(expected, 0.0)
            self.assertAlmostEqual(actual["response:" + family], expected, delta=1e-10)

    def test_large_common_scores_preserve_policy_and_replay_across_batch_shapes(self):
        actor, critic = Actor(True), Critic(True)
        torch.nn.init.normal_(actor.projection.weight, std=0.04)
        ordinary = fixture(actor, critic)
        with torch.no_grad():
            actor.score[2].bias.add_(-16384.0)
        offset = fixture(actor, critic)
        self.assertEqual(len(ordinary), len(offset))
        for before, after in zip(ordinary, offset, strict=True):
            self.assertAlmostEqual(before.logp, after.logp, delta=2e-6)
        packed = PackedRecords(offset)
        from retruco_ml.decision_ppo import replay_identity_error
        from retruco_ml.stability import cache_logits

        self.assertLess(
            replay_identity_error(packed, cache_logits(actor, packed)), 2e-4
        )
        context, indices = packed.contexts(actor, packed.chunks)
        distribution = packed.distribution(actor, context, indices)
        (-distribution.log_prob(packed.actions[indices]).mean()).backward()
        self.assertGreater(float(actor.memory.weight_ih.grad.abs().sum()), 0.0)
        self.assertGreater(float(actor.score[0].weight.grad.abs().sum()), 0.0)
        self.assertTrue(all(p.dtype == torch.float32 for p in actor.parameters()))
        self.assertTrue(
            all(
                p.grad is None or p.grad.dtype == torch.float32
                for p in actor.parameters()
            )
        )

    def test_legacy_migration_preserves_probabilities_and_value_through_memory(self):
        legacy = Policy()
        migrated = migrate_legacy({"model": legacy.state_dict()})
        actor, critic = Actor(True), Critic(True)
        actor.load_state_dict(migrated["model"])
        critic.load_state_dict(migrated["critic"])
        memory = actor.initial()
        for step in range(5):
            features = torch.randn(768)
            actions = torch.randn(7, 48)
            memory = actor.advance(features, memory)
            context = actor.context(features, memory)
            old, signal, value = legacy(features[None], actions[None])
            logits = actor.score(
                torch.cat((context.expand(7, -1), actor.action(actions)), 1)
            ).squeeze(-1)
            torch.testing.assert_close(
                logits.softmax(-1), old[0].softmax(-1), atol=1e-7, rtol=1e-6
            )
            torch.testing.assert_close(actor.signal(context), signal[0])
            torch.testing.assert_close(
                critic(features, torch.randn(128), context),
                value[0],
                atol=1e-7,
                rtol=1e-6,
            )
        self.assertEqual(sum(p.numel() for p in actor.parameters()), 91463)


if __name__ == "__main__":
    unittest.main()
