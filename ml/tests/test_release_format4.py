import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from retruco_ml.artifacts import digest
from retruco_ml.model import Policy, batch_distribution
from retruco_ml.release_recurrent import SeatJournals, mask_features
from retruco_ml.release_v4_campaign import attest, specifications


class SummingActor:
    profile = "value-and-team-v7"

    def advance(self, memory, rows, mode):
        memory[0] += sum(mask_features(rows, mode)[:, 352])


def observation(seat, hand, start, values):
    rows = np.zeros((len(values), 768), dtype=np.float32)
    rows[:, 352] = values
    return {
        "done": False,
        "seat": seat,
        "training": {
            "memory": {
                "version": 1,
                "complete": True,
                "hand": hand,
                "start": start,
                "end": start + len(values),
                "total": start + len(values),
                "events": rows,
            }
        },
    }


class ReleaseFormat4Tests(unittest.TestCase):
    def test_hand_events_are_consumed_once_by_their_own_seat_and_ablated_in_memory(
        self,
    ):
        for mode, expected in [("none", (3, 7)), ("adaptation", (0, 0))]:
            journals = SeatJournals(SummingActor(), mode)
            req = {"id": 8}
            journals.reset(req, [0, 2])
            self.assertEqual(req["playingSeats"], [True, False, True, False])
            journals.consume(None, {8: observation(0, 1, 0, [1, 2])})
            journals.consume(None, {8: observation(2, 1, 0, [7])})
            self.assertEqual(
                (journals.memory(8, 0)[0], journals.memory(8, 2)[0]), expected
            )
            with self.assertRaisesRegex(ValueError, "discontinuous"):
                journals.consume(None, {8: observation(0, 1, 0, [1])})
            journals.consume(None, {8: observation(0, 2, 0, [4])})
            self.assertEqual(journals.parameters(8)["memoryCursors"], [1, 0, 0, 0])
            self.assertEqual(journals.memory(8, 0)[0], 4 if mode == "none" else 0)

    def test_fixed_target_opponent_cannot_train_on_a_forbidden_signal(self):
        policy = Policy()
        obs = {
            "features": np.zeros(768, np.float32),
            "actions": [],
            "version": 2,
            "kind": "signal",
            "signalMask": [True, False, False, False, False, False],
        }
        a, _ = policy.distribution(obs)
        b, _ = batch_distribution(policy, [obs])
        self.assertTrue(torch.equal(a.probs, torch.tensor([1.0, 0, 0, 0, 0, 0])))
        self.assertTrue(torch.equal(a.probs, b.probs[0]))

    def test_report_receipt_rejects_changed_output_or_another_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ["reports", "logs", "receipts"]:
                (root / name).mkdir()
            (root / "logs/task.log").write_text("complete\n")
            path = root / "reports/task.json"
            path.write_text('{"evaluation_fingerprint":"one"}')
            proof = {
                "fingerprint": "one",
                "report_sha256": digest(path),
                "log_sha256": digest(root / "logs/task.log"),
                "exit_code": 0,
            }
            (root / "receipts/task.json").write_text(json.dumps(proof))
            self.assertEqual(
                attest(root, "task", "one")["evaluation_fingerprint"], "one"
            )
            with self.assertRaises(ValueError):
                attest(root, "task", "two")
            path.write_text('{"evaluation_fingerprint":"one","changed":true}')
            with self.assertRaises(ValueError):
                attest(root, "task", "one")

    def test_qualification_includes_each_independent_response_and_matched_signal_references(
        self,
    ):
        specs = specifications()
        self.assertEqual(
            len({r["response_seed"] for r in specs.values() if "response_seed" in r}), 6
        )
        for partner in ["literal", "learned"]:
            a = specs["communication-" + partner]
            b = specs["no-communication-" + partner]
            self.assertEqual({**a, "ablation": "signals"}, b)
