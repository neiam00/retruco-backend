"""Descriptive behavior counters, never rewards or causal evidence of baiting."""

from .cooperation_population import _position
from .improvement_population import public_opponent_plays


class ImprovementBehavior:
    def __init__(self, learner_seat):
        self.learner_seat = learner_seat
        self.counts = dict.fromkeys(
            (
                "card_choice_opportunities",
                "strongest_card_plays",
                "conserving_card_plays",
                "weak_plays_retaining_strong_card",
                "card_choices_teammate_winning",
                "conserving_plays_teammate_winning",
                "truco_calls_after_visible_weak_play",
                "truco_accepts_after_visible_weak_play",
                "opponent_truco_calls_after_learner_weak_play",
                "hands_with_opponent_call_after_learner_weak_play",
                "hands_completed",
                "points_for",
                "points_against",
                "net_points",
                "score_events",
                "game_decisions",
                "envido_openings",
                "envido_openings_at_most_seven",
                "vale_partida_offers",
                "vale_partida_accepts",
                "covered_card_wastes",
                "voluntary_stacks",
                "mandatory_stacks",
                "heads_up_voluntary_stacks",
                "heads_up_stack_under_visible_winner",
                "empty_heads_up_ley_cards",
                "team_stacks_with_certain_top_burn",
                "team_truco_raises_after_match_point",
            ),
            0,
        )
        self._bait_hand = None
        self._called_hands = set()
        self._completed_hands = set()

    def observe(self, observation, action):
        if (
            observation.get("seat") != self.learner_seat
            or observation.get("kind") != "game"
        ):
            return
        legal, view = observation["legal"], observation["view"]
        chosen = legal[action]
        self.counts["game_decisions"] += 1
        if chosen["type"] == "call" and chosen.get("kind") in (
            "envido",
            "ley_envido",
            "flor",
        ):
            self.counts["envido_openings"] += 1
            holding = view.get("holding") or {}
            if "envido" in holding and holding["envido"] <= 7:
                self.counts["envido_openings_at_most_seven"] += 1
        if chosen["type"] == "stack":
            mandatory = bool(view.get("mandatoryStack"))
            self.counts["mandatory_stacks" if mandatory else "voluntary_stacks"] += 1
            if view["config"]["players"] == 2 and not mandatory:
                self.counts["heads_up_voluntary_stacks"] += 1
                own, lead, _, _ = _position(observation)
                slot = chosen.get("card", 0)
                if slot in own and lead and own[slot] <= lead:
                    self.counts["heads_up_stack_under_visible_winner"] += 1
            if view["config"]["players"] == 4 and not mandatory:
                own, lead, teammate_leads, _ = _position(observation)
                bottom = own.get(chosen.get("card", 0))
                if bottom is not None and (
                    lead > bottom or lead == bottom and not teammate_leads
                ):
                    self.counts["team_stacks_with_certain_top_burn"] += 1
        if (
            view["config"]["players"] == 2
            and chosen.get("declaration") == "a_ley"
            and not view.get("florValid")
        ):
            self.counts["empty_heads_up_ley_cards"] += 1
        pending = view.get("pending") or {}
        if (
            view["config"]["players"] == 4
            and (
                chosen.get("kind") == "truco"
                or chosen.get("kind") == "raise"
                and pending.get("kind") == "truco"
            )
            and view["trucoValue"]
            >= view["config"]["target"] - view["scores"][self.learner_seat % 2]
        ):
            self.counts["team_truco_raises_after_match_point"] += 1
        if chosen.get("amount") == 100 and (
            chosen.get("kind") == "truco"
            or (chosen.get("kind") == "raise" and pending.get("kind") == "truco")
        ):
            self.counts["vale_partida_offers"] += 1
        if (
            chosen.get("kind") == "accept"
            and pending.get("kind") == "truco"
            and pending.get("value") == 100
        ):
            self.counts["vale_partida_accepts"] += 1
        if chosen.get("kind") == "truco" or (
            chosen.get("kind") == "accept"
            and (view.get("pending") or {}).get("kind") == "truco"
        ):
            weak_visible = any(
                strength < 8 / 17 for _, strength in public_opponent_plays(observation)
            )
            if weak_visible:
                if chosen["type"] == "call":
                    self.counts["truco_calls_after_visible_weak_play"] += 1
                elif chosen["type"] == "respond" and chosen.get("kind") == "accept":
                    self.counts["truco_accepts_after_visible_weak_play"] += 1
        if chosen["type"] != "play":
            return
        own, leading_rank, teammate_winning, _ = _position(observation)
        # Use the original hand, not the filtered candidate count. This narrower
        # counter counts an expensive card strictly below the leading teammate.
        slot = chosen.get("card", 0)
        if (
            view["config"]["players"] == 4
            and teammate_winning
            and slot in own
            and min(own.values()) < own[slot] < leading_rank
            and not view.get("mandatoryStack")
            and not any(p["plays"][1].get("concealed") for p in view["seats"])
        ):
            self.counts["covered_card_wastes"] += 1
        playable = {
            item.get("card", 0)
            for item in legal
            if item["type"] == "play" and item.get("card", 0) in own
        }
        card_index = chosen.get("card", 0)  # Go omits the zero-valued slot.
        if len(playable) < 2 or card_index not in own:
            return
        strength = own[card_index]
        conserve = strength < max(own[index] for index in playable) - 1e-6
        self.counts["card_choice_opportunities"] += 1
        self.counts[
            "conserving_card_plays" if conserve else "strongest_card_plays"
        ] += 1
        teammate_winning = teammate_winning and view["config"]["players"] == 4
        self.counts["card_choices_teammate_winning"] += int(teammate_winning)
        self.counts["conserving_plays_teammate_winning"] += int(
            teammate_winning and conserve
        )
        retaining = any(
            index != card_index and value >= 12 / 17 - 1e-6
            for index, value in own.items()
        )
        # Burned/concealed plays cannot publicly induce the response counted here.
        if (
            strength < 8 / 17
            and retaining
            and not chosen.get("burned")
            and not chosen.get("concealed")
            and not chosen.get("burn")
        ):
            self.counts["weak_plays_retaining_strong_card"] += 1
            self._bait_hand = view.get("handNumber")

    def transition(self, before, action, after):
        """Called for all seats; reads only public commands, hand IDs and scores."""
        view, next_view = before.get("view") or {}, after.get("view") or {}
        hand = view.get("handNumber")
        if before.get("kind") == "game":
            chosen = before["legal"][action]
            if (
                hand is not None
                and hand == self._bait_hand
                and before["seat"] % 2 != self.learner_seat % 2
                and chosen["type"] == "call"
                and chosen.get("kind") == "truco"
            ):
                self.counts["opponent_truco_calls_after_learner_weak_play"] += 1
                if hand not in self._called_hands:
                    self._called_hands.add(hand)
                    self.counts["hands_with_opponent_call_after_learner_weak_play"] += 1
        previous = before.get("scores", view.get("scores"))
        current = after.get("scores", next_view.get("scores"))
        if previous is not None and current is not None:
            team = self.learner_seat % 2
            ours, theirs = (
                current[team] - previous[team],
                current[1 - team] - previous[1 - team],
            )
            self.counts["points_for"] += ours
            self.counts["points_against"] += theirs
            self.counts["net_points"] += ours - theirs
            self.counts["score_events"] += int(ours != 0 or theirs != 0)
        finished = (
            after.get("done", False)
            or next_view.get("handNumber", hand) != hand
            or next_view.get("handWinner", -1) >= 0
        )
        if hand is not None and finished and hand not in self._completed_hands:
            self._completed_hands.add(hand)
            self.counts["hands_completed"] += 1
        if finished:
            self._bait_hand = None
