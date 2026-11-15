import unittest

from retruco_ml.team_play_probe import team_action


def position(trick, ranks, first=(9, 8, 7, 6), own=(4, 9, 15), flor=False):
    # Player 0 acts last. ranks are the three visible opponents/partner cards in seat order.
    features = [0.0] * 768
    for i, rank in enumerate(own):
        features[24 + 5 * i] = rank / 17
    seats = []
    for seat in range(4):
        plays = []
        for t in range(3):
            rank = (
                first[seat]
                if t == 0 and trick
                else ranks[seat - 1]
                if t == trick and seat
                else None
            )
            play = {"present": rank is not None, "concealed": False, "burned": False}
            if rank is not None:
                play["card"] = {"suit": "coins", "number": 1}
                features[43 + 22 * seat + 7 * t] = rank / 17
            plays.append(play)
        seats.append({"seat": seat, "team": seat % 2, "plays": plays})
    view = {
        "config": {"players": 4, "florRequired": flor},
        "phase": "playing",
        "round": trick,
        "mandatoryStack": False,
        "seats": seats,
        "florValid": flor,
        "holding": {"flor": 34 if flor else 0},
    }
    return view, features


class TeamPlayProbeTests(unittest.TestCase):
    def test_closing_card_wins_with_the_cheapest_sufficient_card_and_keeps_declarations(
        self,
    ):
        v, f = position(0, (8, 5, 6))
        legal = [
            {"type": "play", "card": 2},
            {"type": "play", "card": 2, "declaration": "a_ley"},
            {"type": "play", "card": 1, "declaration": "a_ley"},
            {"type": "play", "declaration": "a_ley"},
        ]
        self.assertEqual(
            team_action(v, f, 0, legal, 3, "closing-card"), (2, True, False)
        )
        self.assertEqual(
            team_action(v, f, 0, legal, 1, "closing-card"), (2, True, False)
        )
        self.assertEqual(
            team_action(v, f, 0, legal, 0, "closing-card"), (0, False, False)
        )

    def test_allied_high_cards_win_only_the_first_trick(self):
        legal = [
            {"type": "play"},
            {"type": "play", "card": 1},
            {"type": "play", "card": 2},
        ]
        view, features = position(0, (7, 9, 6), own=(9, 12, 15))
        self.assertEqual(
            team_action(view, features, 0, legal, 2, "closing-card-and-flor-v2")[0], 0
        )
        self.assertEqual(
            team_action(view, features, 0, legal, 2, "closing-card-and-flor-v1")[0], 1
        )
        view, features = position(1, (9, 6, 5), first=(12, 4, 12, 5), own=(4, 9, 15))
        self.assertEqual(
            team_action(view, features, 0, legal, 0, "closing-card-and-flor-v2")[0], 1
        )
        self.assertEqual(
            team_action(view, features, 0, legal, 0, "closing-card-and-flor-v1")[0], 0
        )

    def test_tying_a_partner_is_a_tie_even_when_the_two_highest_cards_are_teammates(
        self,
    ):
        v, f = position(1, (7, 9, 6), first=(4, 15, 6, 7), own=(4, 9, 15))
        legal = [
            {"type": "play"},
            {"type": "play", "card": 1},
            {"type": "play", "card": 2},
        ]
        # Playing the second 9 ties our partner and loses the hand. Leaving
        # the partner's 9 alone wins the trick and keeps our 15 for trick three.
        self.assertEqual(
            team_action(v, f, 0, legal, 1, "closing-card"), (0, True, False)
        )

    def test_last_card_can_use_a_tie_only_when_our_team_won_the_first_trick(self):
        legal = [{"type": "burn"}, {"type": "play"}, {"type": "play", "card": 1}]
        for first, expected in [((16, 6, 7, 3), 2), ((6, 16, 7, 3), 0)]:
            with self.subTest(first=first):
                v, f = position(2, (9, 5, 7), first=first, own=(4, 9, 15))
                self.assertEqual(
                    team_action(v, f, 0, legal, 0, "closing-card")[0], expected
                )

    def test_incomplete_tricks_hidden_stacks_and_headsup_defer_to_the_actor(self):
        legal = [{"type": "play"}, {"type": "play", "card": 2}]
        for change in ("not_last", "pegado", "mandatory", "headsup"):
            with self.subTest(change=change):
                v, f = position(1, (8, 5, 6))
                if change == "not_last":
                    v["seats"][1]["plays"][1]["present"] = False
                if change == "pegado":
                    v["seats"][2]["plays"][1]["concealed"] = True
                if change == "mandatory":
                    v["mandatoryStack"] = True
                if change == "headsup":
                    v["config"]["players"] = 2
                self.assertEqual(
                    team_action(v, f, 0, legal, 0, "closing-card-and-flor"),
                    (0, False, False),
                )

    def test_flor_uses_the_rules_engines_eligible_spoken_form_for_the_same_move(self):
        v, f = position(0, (8, 4, 3), flor=True)
        legal = [
            {"type": "play", "card": 1},
            {"type": "play", "card": 1, "declaration": "flor"},
            {"type": "play", "declaration": "a_ley"},
            {"type": "play", "card": 1, "declaration": "a_ley"},
        ]
        self.assertEqual(team_action(v, f, 0, legal, 0, "keep-flor"), (3, False, True))
        legal = [
            {"type": "call", "kind": "truco", "amount": 3},
            {
                "type": "call",
                "kind": "truco",
                "amount": 6,
                "declaration": "con_mi_flor",
            },
            {
                "type": "call",
                "kind": "truco",
                "amount": 3,
                "declaration": "con_mi_flor",
            },
        ]
        self.assertEqual(team_action(v, f, 0, legal, 0, "keep-flor"), (2, False, True))
        for key in (
            "already_declared",
            "needs_different_call",
            "human_response",
            "receta",
            "flor_optional",
            "no_valid_flor",
        ):
            with self.subTest(key=key):
                view, features = position(0, (8, 4, 3), flor=True)
                if key == "receta":
                    view["holding"]["receta"] = True
                if key == "flor_optional":
                    view["config"]["florRequired"] = False
                if key == "no_valid_flor":
                    view["florValid"] = False
                actions = legal[:2] if key == "needs_different_call" else legal
                if key == "human_response":
                    actions = [
                        {"type": "respond", "kind": "decline"},
                        {"type": "respond", "kind": "accept"},
                    ]
                chosen = 2 if key == "already_declared" else 0
                self.assertEqual(
                    team_action(view, features, 0, actions, chosen, "keep-flor"),
                    (chosen, False, False),
                )
