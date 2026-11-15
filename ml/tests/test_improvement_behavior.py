"""Descriptive diagnostics across private-view changes and public transitions."""

import copy
import unittest

from test_cooperation_population import card, view_for

from retruco_ml.bridge import Simulator
from retruco_ml.improvement_behavior import ImprovementBehavior


class ImprovementBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simulator = Simulator(transport="binary")

    @classmethod
    def tearDownClass(cls):
        cls.simulator.close()

    def observation(self):
        view = view_for()
        view["pending"] = None
        view["scores"] = [2, 3]
        view["hand"][0]["card"] = card("spades", 12)
        view["seats"][2]["plays"][0] = {"present": True, "card": card("coins", 7)}
        view["legal"] = [
            {"type": "play", "card": 0},
            {"type": "play", "card": 1},
            {"type": "play", "card": 2},
        ]
        return self.simulator.call(
            [
                {
                    "op": "encode",
                    "id": 0,
                    "view": view,
                    "seat": 0,
                    "kind": "game",
                    "context": {
                        "signals": [],
                        "cpu": [True] * 4,
                        "complete": True,
                        "signalsComplete": True,
                    },
                }
            ]
        )[0]

    def test_card_conservation_uses_choice_sets_and_real_encoded_strengths(self):
        obs = self.observation()
        counters = ImprovementBehavior(0)
        counters.observe(obs, 1)
        counters.observe(obs, 0)
        counts = counters.counts
        self.assertEqual(counts["card_choice_opportunities"], 2)
        self.assertEqual(counts["strongest_card_plays"], 1)
        self.assertEqual(counts["conserving_card_plays"], 1)
        self.assertEqual(counts["weak_plays_retaining_strong_card"], 1)
        self.assertEqual(counts["card_choices_teammate_winning"], 2)
        self.assertEqual(counts["conserving_plays_teammate_winning"], 1)
        forced = copy.deepcopy(obs)
        forced["legal"] = [
            {"type": "play", "card": 1},
            {"type": "play", "card": 1, "burn": True},
        ]
        counters.observe(forced, 0)
        self.assertEqual(
            counts["card_choice_opportunities"],
            2,
            "two commands for one card are not a card choice",
        )
        other = copy.deepcopy(obs)
        other["seat"] = 1
        counters.observe(other, 1)
        self.assertEqual(counts["card_choice_opportunities"], 2)

    def test_transition_tracks_public_scores_and_opponent_calls_across_seat_changes(
        self,
    ):
        obs = self.observation()
        counters = ImprovementBehavior(0)
        counters.observe(obs, 1)

        class PrivateView(dict):
            def __iter__(self):
                raise AssertionError("read another seat's hand")

        opponent = {
            "seat": 1,
            "kind": "game",
            "legal": [{"type": "call", "kind": "truco", "amount": 3}],
            "view": {"handNumber": 1, "hand": PrivateView(), "scores": [2, 3]},
        }
        response = {
            "seat": 2,
            "kind": "game",
            "view": {"handNumber": 1, "hand": PrivateView(), "scores": [2, 3]},
        }
        counters.transition(obs, 1, opponent)
        counters.transition(opponent, 0, response)
        self.assertEqual(
            counters.counts["opponent_truco_calls_after_learner_weak_play"], 1
        )
        self.assertEqual(
            counters.counts["hands_with_opponent_call_after_learner_weak_play"], 1
        )
        finished = {
            "done": False,
            "scores": [5, 3],
            "view": {"handNumber": 2, "scores": [5, 3]},
        }
        counters.transition(opponent, 0, finished)
        self.assertEqual(counters.counts["points_for"], 3)
        self.assertEqual(counters.counts["net_points"], 3)
        self.assertEqual(counters.counts["hands_completed"], 1)
        opponent["view"]["handNumber"] = 2
        opponent["view"]["scores"] = [5, 3]
        counters.transition(opponent, 0, {"done": True, "scores": [5, 6]})
        self.assertEqual(
            counters.counts["opponent_truco_calls_after_learner_weak_play"],
            2,
            "new hand clears the weak-play context",
        )
        self.assertEqual(counters.counts["points_against"], 3)
        self.assertEqual(counters.counts["net_points"], 0)
        self.assertEqual(counters.counts["hands_completed"], 2)

    def test_hidden_or_non_card_choices_do_not_create_bait_context(self):
        obs = self.observation()
        obs["legal"][1]["burn"] = True
        counters = ImprovementBehavior(0)
        counters.observe(obs, 1)
        self.assertEqual(counters.counts["weak_plays_retaining_strong_card"], 0)
        opponent = {
            "seat": 1,
            "kind": "game",
            "legal": [{"type": "call", "kind": "truco", "amount": 3}],
            "view": {"handNumber": 1, "scores": [2, 3]},
        }
        counters.transition(opponent, 0, {"view": {"handNumber": 1, "scores": [2, 3]}})
        self.assertEqual(
            counters.counts["opponent_truco_calls_after_learner_weak_play"], 0
        )


if __name__ == "__main__":
    unittest.main()
