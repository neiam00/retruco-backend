"""Observable population and proxy contracts using the Go view encoder."""

import copy
import json
import random
import unittest
from collections import Counter

from test_cooperation_population import card, view_for

from retruco_ml.bridge import Simulator
from retruco_ml.cooperation_population import (
    CooperationPopulation,
    cooperation_scripted,
)
from retruco_ml.improvement_population import (
    ImprovementPopulation,
    improvement_scripted,
    reactive_style,
)


class ImprovementPopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simulator = Simulator(transport="binary")

    @classmethod
    def tearDownClass(cls):
        cls.simulator.close()

    def observe(self, view, kind="game", baseline=0):
        result = self.simulator.call(
            [
                {
                    "op": "encode",
                    "id": 0,
                    "view": view,
                    "seat": 0,
                    "kind": kind,
                    "context": {
                        "signals": [],
                        "cpu": [True] * 4,
                        "complete": True,
                        "signalsComplete": True,
                    },
                }
            ]
        )[0]
        result["baseline"] = baseline
        return result

    def test_control_retains_original_rng_path_and_reactive_preserves_partner(self):
        control = ImprovementPopulation(learned_controllers=("frozen:D101",))
        original = CooperationPopulation(learned_controllers=("frozen:D101",))
        mixed = ImprovementPopulation(
            learned_controllers=("frozen:D101",), reactive=True
        )
        tally = Counter()
        for seed in range(4000):
            for players in (2, 4):
                old = original.episode(players, random.Random(seed))
                self.assertEqual(control.episode(players, random.Random(seed)), old)
                new = mixed.episode(players, random.Random(seed))
                self.assertEqual(new.learner, old.learner)
                if players == 4:
                    partner = (new.learner + 2) % 4
                    self.assertEqual(
                        (new.controllers[partner], new.roles[partner]),
                        (old.controllers[partner], old.roles[partner]),
                    )
                opposite = [s for s in range(players) if s % 2 != new.learner % 2]
                controller = new.controllers[opposite[0]]
                if controller.startswith("script:reactive-"):
                    tally["reactive"] += 1
                    self.assertTrue(
                        all(
                            new.controllers[s] == controller and not new.roles[s]
                            for s in opposite
                        )
                    )
                else:
                    self.assertEqual(new, old)
                    tally["retained"] += 1
                    tally["learned"] += int(not controller.startswith("script:"))
        self.assertTrue(0.18 < tally["reactive"] / 8000 < 0.22)
        self.assertGreater(tally["learned"] / 8000, 0.60)
        self.assertEqual(json.loads(control.recipe)["reactive_fraction"], 0)
        self.assertNotEqual(control.recipe, mixed.recipe)

    def test_visible_weak_play_changes_acceptance_but_not_resistant_or_hidden_views(
        self,
    ):
        view = view_for()
        view["hand"][0]["card"] = card("cups", 3)
        weak = copy.deepcopy(view)
        weak["seats"][1]["plays"][0] = {"present": True, "card": card("coins", 4)}
        style = "reactive-v1:responsive:0.9:0.62:0.6"
        self.assertEqual(
            improvement_scripted(self.observe(view), style, random.Random(4)), 0
        )
        self.assertEqual(
            improvement_scripted(self.observe(weak), style, random.Random(4)), 1
        )
        self.assertEqual(
            improvement_scripted(
                self.observe(weak), "reactive-resistant-v1", random.Random(4)
            ),
            0,
        )
        for hidden in ("burned", "concealed"):
            hidden_view = copy.deepcopy(weak)
            hidden_view["seats"][1]["plays"][0][hidden] = True
            self.assertEqual(
                improvement_scripted(
                    self.observe(hidden_view), style, random.Random(4)
                ),
                0,
            )
        very_weak = copy.deepcopy(weak)
        very_weak["hand"][0]["card"] = card("cups", 4)
        self.assertEqual(
            improvement_scripted(self.observe(very_weak), style, random.Random(4)), 0
        )
        # The held-out response rule requires corroboration, not one weak card.
        self.assertEqual(
            improvement_scripted(
                self.observe(weak), "reactive-heldout-v1", random.Random(4)
            ),
            0,
        )

    def test_reactions_are_legal_bounded_truthful_and_ignore_training_channels(self):
        view = view_for()
        view["pending"] = None
        view["hand"][0]["card"] = card("spades", 12)
        view["hand"][1]["card"] = card("spades", 10)
        view["legal"] = [
            {"type": "call", "kind": "truco", "amount": 9},
            {"type": "play", "card": 0},
            {"type": "play", "card": 1},
            {"type": "call", "kind": "truco", "amount": 3},
        ]
        obs = self.observe(view)

        class Forbidden(dict):
            def __getitem__(self, key):
                raise AssertionError("read privileged information")

            def get(self, key, default=None):
                raise AssertionError("read privileged information")

        alternate = copy.deepcopy(obs)
        alternate["training"] = Forbidden()
        alternate["opponentHands"] = Forbidden()
        for seed in range(100):
            style = reactive_style(random.Random(seed))
            chosen = improvement_scripted(obs, style, random.Random(seed))
            self.assertIn(chosen, (1, 2, 3))
            self.assertEqual(
                chosen, improvement_scripted(alternate, style, random.Random(seed))
            )
        signal = self.observe(view, kind="signal")
        signal["signalMask"] = [True, True, True, False, False, True]
        for seed in range(50):
            action = improvement_scripted(
                signal, reactive_style(random.Random(seed)), random.Random(seed)
            )
            self.assertIn(action, (0, 1, 2, 5))
        # Dispatching existing styles must preserve their behavior and RNG use.
        for style in ("literal-v2", "trusting-v2", "baseline"):
            left, right = random.Random(9), random.Random(9)
            self.assertEqual(
                improvement_scripted(obs, style, left),
                cooperation_scripted(obs, style, right),
            )
            self.assertEqual(left.getstate(), right.getstate())

    def test_real_matches_finish_with_legal_bounded_reactive_actions(self):
        for players in (2, 4):
            for heldout in (False, True):
                rng = random.Random(51 + players)
                styles = [reactive_style(rng, heldout=heldout) for _ in range(players)]
                obs = self.simulator.call(
                    [
                        {
                            "op": "reset",
                            "id": 0,
                            "seed": 51,
                            "truthfulCPU": True,
                            "roles": [True, False, True, False],
                            "config": {
                                "players": players,
                                "target": 12,
                                "florRequired": True,
                            },
                        }
                    ]
                )[0]
                for _ in range(2000):
                    if obs["done"]:
                        break
                    action = improvement_scripted(obs, styles[obs["seat"]], rng)
                    if obs["kind"] == "signal":
                        if obs.get("signalMask"):
                            self.assertTrue(obs["signalMask"][action])
                    else:
                        self.assertTrue(0 <= action < len(obs["legal"]))
                        chosen = obs["legal"][action]
                        if chosen["type"] == "call" and chosen.get("kind") == "truco":
                            self.assertLessEqual(chosen.get("amount", 0), 6)
                    obs = self.simulator.call(
                        [{"op": "step", "id": 0, "action": action}]
                    )[0]
                self.assertTrue(obs["done"], "reactive play must complete real matches")


if __name__ == "__main__":
    unittest.main()
