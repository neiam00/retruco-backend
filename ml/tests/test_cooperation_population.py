"""Behavior contracts against the real Go authorized-view encoder."""

import copy
import json
import random
import unittest
from collections import Counter

from retruco_ml.bridge import Simulator
from retruco_ml.cooperation_population import (
    COOPERATION_STYLES,
    CooperationPopulation,
    cooperation_scripted,
)


def card(suit, number):
    return {"suit": suit, "number": number}


def view_for(seat=0):
    return {
        "config": {"players": 4, "target": 24, "florRequired": False},
        "phase": "playing",
        "round": 0,
        "handNumber": 1,
        "vira": card("spades", 11),
        "turn": seat,
        "hand": [
            {"index": i, "card": c, "used": False}
            for i, c in enumerate((card("coins", 4), card("cups", 5), card("clubs", 6)))
        ],
        "holding": {"envido": 6, "flor": 0, "receta": False},
        "seats": [
            {
                "seat": s,
                "team": s % 2,
                "remaining": 3,
                "plays": [{"present": False} for _ in range(3)],
            }
            for s in range(4)
        ],
        "pending": {"kind": "truco", "value": 3, "caller": 1 - seat % 2},
        "legal": [
            {"type": "respond", "kind": "decline"},
            {"type": "respond", "kind": "accept"},
        ],
    }


class CooperationPopulationTests(unittest.TestCase):
    def test_no_proxy_emits_a_team_signal_in_heads_up(self):
        view = view_for()
        view["config"]["players"] = 2
        view["hand"][0]["card"] = card("spades", 12)
        for style in (
            *COOPERATION_STYLES,
            "literal-v3",
            "trusting-v3",
            "cautious-v3",
            "deceptive",
            "ignore",
        ):
            for seed in range(20):
                self.assertEqual(
                    cooperation_scripted(
                        {"view": view, "kind": "signal"}, style, random.Random(seed)
                    ),
                    0,
                )

    @classmethod
    def setUpClass(cls):
        cls.simulator = Simulator(transport="binary")

    @classmethod
    def tearDownClass(cls):
        cls.simulator.close()

    def observe(self, view, signals=(), seat=0, kind="game", baseline=0):
        obs = self.simulator.call(
            [
                {
                    "op": "encode",
                    "id": 0,
                    "view": view,
                    "seat": seat,
                    "kind": kind,
                    "context": {
                        "signals": list(signals),
                        "cpu": [True] * 4,
                        "complete": True,
                        "signalsComplete": True,
                    },
                }
            ]
        )[0]
        obs["baseline"] = baseline
        return obs

    def choose(self, view, signals=(), style="literal-v2", seat=0):
        return cooperation_scripted(
            self.observe(view, signals, seat), style, random.Random(5)
        )

    def test_every_literal_claim_including_sevens_supports_truco_only(self):
        for kind in ("perico", "perica", "seven_spades", "seven_coins", "receta"):
            for seat in range(4):
                with self.subTest(kind=kind, seat=seat):
                    view = view_for(seat)
                    signals = [{"seat": (seat + 2) % 4, "kind": kind}]
                    self.assertEqual(self.choose(view, signals, seat=seat), 1)
                    for family in ("envido", "ley_envido", "flor", "privo"):
                        view["pending"]["kind"] = family
                        self.assertEqual(self.choose(view, signals, seat=seat), 0)
        view = view_for()
        for source in (0, 1, 3):
            self.assertEqual(self.choose(view, [{"seat": source, "kind": "receta"}]), 0)

    def test_spent_or_contradicted_claims_do_not_count_twice(self):
        view = view_for()
        view["round"] = 1
        signals = [{"seat": 2, "kind": "perico"}] * 3
        view["pending"]["value"] = 9
        self.assertEqual(self.choose(view, signals), 0, "repetition is not three cards")
        view["pending"]["value"] = 3
        view["seats"][2]["plays"][0] = {"present": True, "card": card("spades", 12)}
        self.assertEqual(
            self.choose(view, signals), 0, "vira-eleven replacement piece was spent"
        )
        for hidden in ("burned", "concealed"):
            view["seats"][2]["plays"][0][hidden] = True
            self.assertEqual(
                self.choose(view, signals), 1, "hidden card must not refute a claim"
            )
            view["seats"][2]["plays"][0].pop(hidden)
        view["seats"][2]["plays"][0] = {"present": False}
        view["hand"][0]["card"] = card("spades", 12)
        view["hand"][0]["used"] = True
        self.assertEqual(
            self.choose(view, signals),
            0,
            "observer knows the claimed card is not held by partner",
        )

    def test_receta_loses_only_the_publicly_spent_piece(self):
        view = view_for()
        view["round"] = 1
        view["seats"][2]["plays"][0] = {"present": True, "card": card("spades", 12)}
        signals = [{"seat": 2, "kind": "receta"}]
        self.assertEqual(self.choose(view, signals), 1)
        view["pending"]["value"] = 9
        self.assertEqual(self.choose(view, signals), 0)

    def test_cautious_does_not_blanket_refuse_and_uses_own_point_evidence(self):
        view = view_for()
        view["hand"][0]["card"] = card("spades", 12)
        self.assertEqual(self.choose(view, style="cautious-v2"), 1)
        view["pending"] = {"kind": "envido", "value": 3, "terminal": True}
        view["holding"]["envido"] = 34
        self.assertEqual(self.choose(view, style="cautious-v2"), 1)
        view["holding"]["envido"] = 24
        self.assertEqual(self.choose(view, style="cautious-v2"), 0)

    def test_go_encoded_strengths_conserve_card_for_winning_teammate(self):
        for seat in range(4):
            view = view_for(seat)
            view["pending"] = None
            view["round"] = 1
            view["hand"][0]["card"] = card(
                "spades", 12
            )  # perico, not face-card strength
            view["hand"][1]["card"] = card("coins", 4)
            view["legal"] = [{"type": "play", "card": 0}, {"type": "play", "card": 1}]
            view["seats"][(seat + 2) % 4]["plays"][1] = {
                "present": True,
                "card": card("spades", 7),
            }
            self.assertEqual(self.choose(view, seat=seat), 1)
            # Opposing perica beats that seven; do not act as though partner wins.
            view["seats"][(seat + 1) % 4]["plays"][1] = {
                "present": True,
                "card": card("spades", 10),
            }
            self.assertEqual(self.choose(view, seat=seat), 0)

    def test_signal_changes_card_choice_only_if_partner_can_still_act(self):
        view = view_for()
        view["pending"] = None
        view["legal"] = [{"type": "play", "card": 2}, {"type": "play", "card": 0}]
        signals = [{"seat": 2, "kind": "seven_coins"}]
        self.assertEqual(self.choose(view), 0)
        self.assertEqual(self.choose(view, signals), 1)
        view["seats"][2]["plays"][0] = {"present": True, "concealed": True}
        self.assertEqual(self.choose(view, signals), 0)

    def test_main_proxy_profiles_signal_only_truthful_remaining_cards(self):
        view = view_for()
        view["vira"] = card("spades", 10)
        view["hand"] = [
            {"index": i, "card": c, "used": False}
            for i, c in enumerate(
                (card("spades", 11), card("spades", 12), card("coins", 7))
            )
        ]
        for style in COOPERATION_STYLES:
            if style == "erratic-v2":
                continue
            rng = random.Random(83)
            obs = self.observe(view, kind="signal")
            actions = {cooperation_scripted(obs, style, rng) for _ in range(100)}
            self.assertLessEqual(actions, {0, 1, 2, 4, 5})
            if style == "silent-v2":
                self.assertEqual(actions, {0})
            else:
                self.assertIn(5, actions)
            spent = copy.deepcopy(view)
            spent["hand"][0]["used"] = spent["hand"][1]["used"] = True
            obs = self.observe(spent, kind="signal")
            self.assertLessEqual(
                {cooperation_scripted(obs, style, rng) for _ in range(100)}, {0, 4}
            )

    def test_contextual_partner_does_not_treat_claim_as_a_promised_play(self):
        view = view_for()
        view["pending"] = None
        view["hand"][0]["card"] = card("spades", 12)
        view["legal"] = [{"type": "play", "card": 0}, {"type": "play", "card": 1}]
        signals = [{"seat": 2, "kind": "perica"}]
        self.assertEqual(self.choose(view, signals, style="literal-v2"), 1)
        for style in ("literal-v3", "trusting-v3", "cautious-v3"):
            self.assertEqual(
                self.choose(view, signals, style=style),
                0,
                "possession does not commit the teammate to winning",
            )
        view["seats"][2]["plays"][0] = {"present": True, "card": card("spades", 10)}
        self.assertEqual(
            self.choose(view, signals, style="literal-v3"),
            1,
            "an actually winning teammate still allows conservation",
        )

    def test_contextual_partner_does_not_accept_a_deciding_trick_already_beaten(self):
        view = view_for()
        view["round"] = 2
        view["seats"][1]["plays"][2] = {"present": True, "card": card("spades", 12)}
        signals = [{"seat": 2, "kind": "perica"}]
        self.assertEqual(self.choose(view, signals, style="literal-v2"), 1)
        self.assertEqual(self.choose(view, signals, style="literal-v3"), 0)

    def test_contextual_partner_requires_own_support_before_initiating_from_claims(
        self,
    ):
        view = view_for()
        view["pending"] = None
        view["legal"] = [
            {"type": "play", "card": 0},
            {"type": "call", "kind": "truco", "amount": 3},
        ]
        obs = self.observe(view, [{"seat": 2, "kind": "receta"}])
        self.assertEqual(cooperation_scripted(obs, "literal-v2", random.Random(1)), 1)
        self.assertEqual(cooperation_scripted(obs, "literal-v3", random.Random(1)), 0)

    def test_erratic_human_can_misuse_signals_while_literal_stays_truthful(self):
        view = view_for()
        view["hand"][0]["card"] = card("spades", 12)  # only perico is true
        obs = self.observe(view, kind="signal")
        rng = random.Random(129)
        actions = [cooperation_scripted(obs, "erratic-v2", rng) for _ in range(500)]
        self.assertTrue(set(actions) & {2, 3, 4, 5}, "misused claims must occur")
        self.assertGreater(actions.count(1), sum(a in (2, 3, 4, 5) for a in actions))
        rng = random.Random(129)
        self.assertLessEqual(
            {cooperation_scripted(obs, "literal-v2", rng) for _ in range(500)}, {0, 1}
        )

    def test_population_covers_roles_styles_and_frozen_opponents_reproducibly(self):
        population = CooperationPopulation(
            learned_controllers=("frozen:D101", "frozen:D102", "frozen:D103")
        )
        rng = random.Random(17)
        episodes = [population.episode(4, rng) for _ in range(6000)]
        replay = random.Random(17)
        self.assertEqual(
            episodes[:20], [population.episode(4, replay) for _ in range(20)]
        )
        partners, opponents = Counter(), set()
        for episode in episodes:
            partner = (episode.learner + 2) % 4
            partners[(episode.controllers[partner], episode.roles[partner])] += 1
            opponents.add(episode.controllers[(episode.learner + 1) % 4])
            self.assertEqual(episode.controllers.count("learner"), 1)
            self.assertTrue(episode.roles[episode.learner])
            for seat, controller in enumerate(episode.controllers):
                if controller == "script:erratic-v2":
                    self.assertFalse(
                        episode.roles[seat],
                        "synthetic human misuse must not get a CPU role",
                    )
        self.assertTrue(set(population.learned_controllers) <= opponents)
        for style in COOPERATION_STYLES:
            self.assertGreater(partners[("script:" + style, False)], 100)
        self.assertTrue(1000 < partners[("current", False)] < 1400)
        self.assertTrue(450 < partners[("current", True)] < 750)
        self.assertEqual(json.loads(population.recipe)["version"], "cooperation-v2")
        self.assertEqual(len(population.episode(2, rng).controllers), 2)


if __name__ == "__main__":
    unittest.main()
