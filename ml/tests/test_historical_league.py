"""Frozen opponent behavior, match roles, and reproducible league checkpoints."""

import copy
import io
import random
import unittest

import torch

from retruco_ml.benchmark_population import NativeLeaguePopulation
from retruco_ml.historical_league import HistoricalLeaguePopulation
from retruco_ml.improvement_population import ImprovementPopulation
from retruco_ml.model import Policy
from retruco_ml.recurrent import Actor


class HistoricalLeagueTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(912)

    def test_frozen_registry_has_no_learner_aliases_gradients_or_mid_match_changes(
        self,
    ):
        learner, reference = Actor(True, 64), Policy()
        torch.nn.init.normal_(learner.projection.weight, std=0.03)
        population = HistoricalLeaguePopulation(max_snapshots=1)
        before_rng = torch.get_rng_state().clone()
        key = population.add_actor(learner, 0)
        registry = population.models(reference)
        torch.testing.assert_close(torch.get_rng_state(), before_rng, atol=0, rtol=0)
        opponent = registry[key]
        self.assertFalse(opponent.training)
        self.assertTrue(
            all(not p.requires_grad and p.grad is None for p in opponent.parameters())
        )
        self.assertTrue(
            all(
                a.data_ptr() != b.data_ptr()
                for a, b in zip(
                    learner.parameters(), opponent.parameters(), strict=True
                )
            )
        )
        event = torch.randn(768)
        with torch.no_grad():
            memory = opponent.advance(event, opponent.initial())
            expected = opponent.signal_logits(opponent.context(event, memory)).clone()
            learner.signal.bias.add_(1)
        new_key = population.add_actor(learner, 50000)
        self.assertNotEqual(new_key, key)
        self.assertNotIn(key, population.models(reference))
        # The already-issued registry survives both mutation and eviction.
        with torch.no_grad():
            memory = opponent.advance(event, opponent.initial())
            actual = opponent.signal_logits(opponent.context(event, memory))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        state = population.state_dict()
        state["actors"][0]["model"]["signal.bias"].add_(10)
        torch.testing.assert_close(
            population.models(reference)[new_key].signal.bias,
            learner.signal.bias,
            atol=0,
            rtol=0,
        )

    def test_sampling_keeps_teammate_roles_and_one_frozen_opponent_for_the_match(self):
        args = {"learned_controllers": ("frozen:D101",), "reactive": True}
        population = HistoricalLeaguePopulation(**args)
        base = ImprovementPopulation(**args)
        keys = [
            population.add_actor(Actor(True, width), index * 50000)
            for index, width in enumerate((16, 32, 64))
        ]
        registry = population.models(Policy())
        self.assertEqual(
            [registry[key].memory.hidden_size for key in keys], [16, 32, 64]
        )
        rng = random.Random(32)
        counts = {key: 0 for key in [*keys, "inherited"]}
        for index in range(5000):
            players = 2 if index % 2 else 4
            base_rng = random.Random()
            base_rng.setstate(rng.getstate())
            expected = base.episode(players, base_rng)
            actual = population.episode(players, rng)
            self.assertEqual(actual.learner, expected.learner)
            for seat in range(players):
                if seat % 2 == actual.learner % 2:
                    self.assertEqual(
                        actual.controllers[seat], expected.controllers[seat]
                    )
                    self.assertEqual(actual.roles[seat], expected.roles[seat])
            opposition = [
                seat for seat in range(players) if seat % 2 != actual.learner % 2
            ]
            controller = actual.controllers[opposition[0]]
            if controller.startswith("league:"):
                counts[controller] += 1
                self.assertTrue(
                    all(
                        actual.controllers[seat] == controller and actual.roles[seat]
                        for seat in opposition
                    )
                )
            else:
                counts["inherited"] += 1
                self.assertEqual(actual, expected)
        for key, expected in zip(
            [*keys, "inherited"], (0.20, 0.20, 0.45, 0.15), strict=True
        ):
            self.assertAlmostEqual(counts[key] / 5000, expected, delta=0.025)

    def test_empty_and_single_version_fallbacks_preserve_valid_assignments(self):
        population, base = (
            HistoricalLeaguePopulation(reactive=True),
            ImprovementPopulation(reactive=True),
        )
        left, right = random.Random(26), random.Random(26)
        for _ in range(50):
            self.assertEqual(population.episode(4, left), base.episode(4, right))
        self.assertEqual(left.getstate(), right.getstate())
        key = population.add_actor(Actor(True, 16), 0)
        for _ in range(100):
            episode = population.episode(4, left)
            for controller in episode.controllers:
                if controller.startswith("league:"):
                    self.assertEqual(controller, key)

    def test_serialized_mixed_width_pool_resumes_same_rng_assignments_and_weights(self):
        reference = Policy()
        args = {
            "snapshots": [reference.state_dict()],
            "learned_controllers": ("frozen:D101",),
            "reactive": True,
        }
        population = HistoricalLeaguePopulation(**args)
        keys = [
            population.add_actor(Actor(True, width), index * 50000)
            for index, width in enumerate((16, 64))
        ]
        rng = random.Random(821)
        for _ in range(11):
            population.episode(4, rng)
        stream = io.BytesIO()
        torch.save(
            {"league": population.state_dict(), "random": rng.getstate()}, stream
        )
        stream.seek(0)
        checkpoint = torch.load(stream, weights_only=True)
        resumed = HistoricalLeaguePopulation(
            learned_controllers=("frozen:D101",), reactive=True
        )
        before_rng = torch.get_rng_state().clone()
        resumed.load_state_dict(checkpoint["league"])
        torch.testing.assert_close(torch.get_rng_state(), before_rng, atol=0, rtol=0)
        second_rng = random.Random()
        second_rng.setstate(checkpoint["random"])
        for index in range(200):
            self.assertEqual(
                population.episode(2 if index % 2 else 4, rng),
                resumed.episode(2 if index % 2 else 4, second_rng),
            )
        self.assertEqual(rng.getstate(), second_rng.getstate())
        actual, expected = resumed.models(reference), population.models(reference)
        for key in [*keys, "history:0"]:
            for name, value in actual[key].state_dict().items():
                torch.testing.assert_close(
                    value, expected[key].state_dict()[name], atol=0, rtol=0
                )
        checkpoint["league"]["actors"][0]["model"]["signal.bias"].add_(10)
        checkpoint["league"]["base_snapshots"][0]["signal.bias"].add_(10)
        for key in [keys[0], "history:0"]:
            torch.testing.assert_close(
                resumed.models(reference)[key].signal.bias,
                expected[key].signal.bias,
                atol=0,
                rtol=0,
            )

    def test_eviction_deduplication_and_invalid_restore_leave_pool_intact(self):
        population, actor = HistoricalLeaguePopulation(), Actor(True, 16)
        keys = []
        for index in range(18):
            with torch.no_grad():
                actor.signal.bias[0].add_(0.1)
            keys.append(population.add_actor(actor, index * 50000))
        self.assertEqual(population.snapshot_count, 16)
        self.assertEqual(
            set(
                key for key in population.models(Policy()) if key.startswith("league:")
            ),
            set(keys[-16:]),
        )
        self.assertEqual(population.add_actor(actor, 900000), keys[-1])
        state = population.state_dict()
        self.assertEqual(population.snapshot_count, 16)
        for mutation in (
            "version",
            "configuration",
            "fingerprint",
            "weights",
            "order",
            "capacity",
        ):
            invalid = copy.deepcopy(state)
            if mutation == "version":
                invalid["version"] += 1
            elif mutation == "configuration":
                invalid["configuration"]["recipe"] = "wrong"
            elif mutation == "fingerprint":
                invalid["actors"][0]["fingerprint"] = "0" * 64
            elif mutation == "weights":
                invalid["actors"][0]["model"]["signal.bias"][0] = float("nan")
            elif mutation == "order":
                invalid["actors"][1]["choices"] = 0
            else:
                invalid["actors"].append(copy.deepcopy(invalid["actors"][-1]))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                population.load_state_dict(invalid)
            self.assertEqual(
                [entry["fingerprint"] for entry in population.state_dict()["actors"]],
                [entry["fingerprint"] for entry in state["actors"]],
            )
        with self.assertRaises(ValueError):
            population.add_actor(Actor(False), 950000)
        with self.assertRaises(ValueError):
            population.add_actor(actor, 0)
        with self.assertRaises(ValueError):
            HistoricalLeaguePopulation(max_snapshots=17)

    def test_persistent_anchor_survives_rotation_within_total_cap_and_is_frozen(self):
        actor = Actor(True, 64)
        original = copy.deepcopy(actor.state_dict())
        population = HistoricalLeaguePopulation(
            anchor_actor=actor, anchor_fraction=0.25, max_snapshots=4
        )
        anchor_key = population.add_actor(actor, 0)
        before_rng = torch.get_rng_state().clone()
        for update in range(1, 25):
            with torch.no_grad():
                actor.signal.bias[0].add_(0.1)
            latest_key = population.add_actor(actor, update * 50000)
            registry = population.models(None)
            self.assertIn(anchor_key, registry)
            self.assertIn(latest_key, registry)
            self.assertLessEqual(population.snapshot_count, 4)
            self.assertEqual(
                len([key for key in registry if key.startswith("league:")]),
                population.snapshot_count,
            )
            for name, value in original.items():
                torch.testing.assert_close(
                    value, registry[anchor_key].state_dict()[name], atol=0, rtol=0
                )
            self.assertTrue(
                all(not p.requires_grad for p in registry[anchor_key].parameters())
            )
        torch.testing.assert_close(before_rng, torch.get_rng_state(), atol=0, rtol=0)
        counts = {"anchor": 0, "league": 0}
        for seed in range(4000):
            episode = population.episode(4, random.Random(seed))
            opponent = episode.controllers[(episode.learner + 1) % 4]
            counts["anchor"] += opponent == anchor_key
            counts["league"] += opponent.startswith("league:")
        self.assertAlmostEqual(counts["league"] / 4000, 0.85, delta=0.025)
        self.assertAlmostEqual(counts["anchor"] / counts["league"], 0.25, delta=0.03)

    def test_anchor_choice_preserves_match_rng_and_native_pairing_after_turnover(self):
        actor = Actor(True)
        control = NativeLeaguePopulation(native_fraction=0.5, max_snapshots=4)
        anchored = NativeLeaguePopulation(
            native_fraction=0.5,
            max_snapshots=4,
            anchor_actor=actor,
            anchor_fraction=0.4,
        )
        disabled = NativeLeaguePopulation(
            native_fraction=0.5,
            max_snapshots=4,
            anchor_actor=actor,
            anchor_fraction=0.0,
        )
        self.assertEqual(control.recipe, disabled.recipe)
        for update in range(12):
            if update:
                with torch.no_grad():
                    actor.signal.bias[0].add_(0.1)
            for population in (control, anchored, disabled):
                population.add_actor(actor, update * 50000)
        for seed in range(500):
            left, right, third = (
                random.Random(seed),
                random.Random(seed),
                random.Random(seed),
            )
            for players in (2, 4, 4, 2):
                expected = control.episode(players, left)
                actual = anchored.episode(players, right)
                self.assertEqual(expected, disabled.episode(players, third))
                self.assertEqual(left.getstate(), right.getstate())
                self.assertEqual(left.getstate(), third.getstate())
                self.assertEqual(expected.learner, actual.learner)
                self.assertEqual(expected.roles, actual.roles)
                for seat in range(players):
                    if seat % 2 == expected.learner % 2:
                        self.assertEqual(
                            expected.controllers[seat], actual.controllers[seat]
                        )
                    else:
                        self.assertEqual(
                            expected.controllers[seat] == "script:baseline",
                            actual.controllers[seat] == "script:baseline",
                        )
        self.assertNotIn("anchor", disabled.state_dict())

    def test_anchor_resume_binds_source_weights_fraction_and_reserved_slot(self):
        source = Actor(True)
        population = HistoricalLeaguePopulation(
            anchor_actor=source, anchor_fraction=0.3
        )
        population.add_actor(Actor(True, 64), 50000)
        state = population.state_dict()
        resumed = HistoricalLeaguePopulation(anchor_actor=source, anchor_fraction=0.3)
        resumed.load_state_dict(state)
        left, right = random.Random(12), random.Random(12)
        for _ in range(100):
            self.assertEqual(population.episode(4, left), resumed.episode(4, right))
        self.assertEqual(left.getstate(), right.getstate())
        for changed in (
            HistoricalLeaguePopulation(anchor_actor=source, anchor_fraction=0.4),
            HistoricalLeaguePopulation(anchor_actor=Actor(True), anchor_fraction=0.3),
        ):
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                changed.load_state_dict(state)
        for mutation in ("weights", "missing", "slot"):
            bad = copy.deepcopy(state)
            if mutation == "weights":
                bad["anchor"]["model"]["signal.bias"][0].add_(1)
            elif mutation == "missing":
                del bad["anchor"]
            else:
                bad["actors"] = [
                    entry
                    for entry in bad["actors"]
                    if entry["fingerprint"] != bad["anchor"]["fingerprint"]
                ]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                resumed.load_state_dict(bad)
            self.assertEqual(
                resumed.state_dict()["anchor"]["fingerprint"],
                state["anchor"]["fingerprint"],
            )
        for options in (
            {"anchor_fraction": 0.2},
            {"anchor_actor": source, "anchor_fraction": 0.2, "max_snapshots": 1},
            {"anchor_actor": source, "anchor_fraction": float("nan")},
            {"anchor_actor": source, "anchor_fraction": 1.1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                HistoricalLeaguePopulation(**options)


if __name__ == "__main__":
    unittest.main()
