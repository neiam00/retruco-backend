import unittest

import torch

from retruco_ml.rmappo_play_pretrain import (
    browser_action_indices,
    executed_distribution,
)


class PlayPretrainingTest(unittest.TestCase):
    def test_flor_aliases_preserve_total_probability_and_exclude_unoffered_moves(self):
        legal = [
            {"type": "call", "kind": "truco", "amount": 3},
            {
                "type": "call",
                "kind": "truco",
                "amount": 3,
                "declaration": "con_mi_flor",
            },
            {"type": "call", "kind": "truco", "amount": 9},
            {"type": "respond", "kind": "decline"},
        ]
        offered = [legal[3], legal[0], legal[1]]
        logits = torch.tensor([0.3, 0.2, 0.4, 0.5], dtype=torch.float64).log()
        view = {
            "config": {"players": 4, "florRequired": True},
            "phase": "playing",
            "florValid": True,
            "holding": {"flor": 32},
            "round": 0,
        }
        sources, destinations = browser_action_indices(view, [], 1, legal, offered)
        probabilities = executed_distribution(logits, sources, destinations)
        torch.testing.assert_close(
            probabilities, torch.tensor([0.0, 0.5, 0.0, 0.5], dtype=logits.dtype)
        )
        view["config"]["players"] = 2
        sources, destinations = browser_action_indices(view, [], 1, legal, offered)
        heads_up = executed_distribution(logits, sources, destinations)
        torch.testing.assert_close(
            heads_up, torch.tensor([0.3, 0.2, 0.0, 0.5], dtype=logits.dtype)
        )
