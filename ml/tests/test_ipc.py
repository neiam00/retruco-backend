import copy
import json
import random
import struct
import unittest

import numpy as np
import torch

from retruco_ml.bridge import Simulator, decode_binary
from retruco_ml.model import Policy
from retruco_ml.population import Population
from retruco_ml.rollout import SimulatorPool, collect


class IPCTests(unittest.TestCase):
    def compare(self, left, right):
        np.testing.assert_array_equal(
            np.asarray(left["features"], np.float32), right["features"]
        )
        np.testing.assert_array_equal(
            np.asarray(left["actions"] or [], np.float32).reshape(-1, 48),
            right["actions"],
        )
        self.assertEqual(
            {k: v for k, v in left.items() if k not in ("features", "actions")},
            {k: v for k, v in right.items() if k not in ("features", "actions")},
        )

    def test_binary_and_json_replay_complete_matches_identically(self):
        with Simulator() as legacy, Simulator(transport="binary") as binary:
            first = None
            for players in (2, 4):
                for target in (12, 24):
                    for flor in (False, True):
                        requests = [
                            {
                                "op": "reset",
                                "id": i,
                                "seed": 230 + i,
                                "roles": [True, False, True, True],
                                "config": {
                                    "players": players,
                                    "target": target,
                                    "florRequired": flor,
                                },
                            }
                            for i in (7, 2)
                        ]
                        for step in range(10000):
                            left = legacy.call(requests)
                            right = binary.call(requests)
                            for a, b in zip(left, right):
                                self.compare(a, b)
                            if first is None:
                                first = (
                                    right[0]["features"],
                                    right[0]["features"].copy(),
                                )
                            requests = [
                                {
                                    "op": "step",
                                    "id": request["id"],
                                    "action": step % 6
                                    if o["kind"] == "signal"
                                    else o["baseline"],
                                }
                                for request, o in zip(requests, left)
                                if not o["done"]
                            ]
                            if not requests:
                                break
                        self.assertFalse(requests, "complete match stalled")
            np.testing.assert_array_equal(*first)

    def test_capacity_probe_preserves_every_candidate_and_errors_are_recoverable(self):
        with Simulator() as legacy, Simulator(transport="binary") as binary:
            reset = [
                {
                    "op": "reset",
                    "id": 0,
                    "config": {"players": 4, "target": 24},
                    "seed": 42,
                }
            ]
            original = legacy.call(reset)[0]
            binary.call(reset)
            view = copy.deepcopy(original["view"])
            view["legal"] = [
                {"type": "call", "kind": "envido", "amount": i} for i in range(1024)
            ]
            probe = [
                {
                    "op": "encode",
                    "view": view,
                    "context": {},
                    "kind": "game",
                    "seat": original["seat"],
                }
            ]
            self.compare(legacy.call(probe)[0], binary.call(probe)[0])
            with self.assertRaisesRegex(RuntimeError, "unknown environment"):
                binary.call([{"op": "step", "id": 63}])
            self.compare(
                legacy.call([{"op": "observe", "id": 0}])[0],
                binary.call([{"op": "observe", "id": 0}])[0],
            )

    def test_decoder_rejects_corruption_without_allocating_unbounded_arrays(self):
        metadata = json.dumps(
            [{"id": 0, "observation": {"version": 2, "action_count": 0}}]
        ).encode()
        frame = b"RTO1" + struct.pack("<I", len(metadata)) + metadata + bytes(768 * 4)
        self.assertEqual(
            decode_binary(frame)[0]["observation"]["features"].shape, (768,)
        )
        for bad in (
            b"",
            b"FAIL" + frame[4:],
            frame[:-1],
            frame + b"\0",
            b"RTO1" + struct.pack("<I", 2**32 - 1),
        ):
            with self.assertRaises(RuntimeError):
                decode_binary(bad)
        metadata = json.dumps(
            [{"id": 0, "observation": {"version": 2, "action_count": 2**31}}]
        ).encode()
        with self.assertRaises(RuntimeError):
            decode_binary(b"RTO1" + struct.pack("<I", len(metadata)) + metadata)

    def test_transport_does_not_change_sampled_learner_trajectories(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        model = Policy()
        runs = []
        for transport in ("json", "binary"):
            torch.manual_seed(27)
            with SimulatorPool(2, transport=transport) as pool:
                runs.append(
                    collect(model, pool, Population(), random.Random(22), 128, 0, 8)
                )
        self.assertEqual(runs[0][1:], runs[1][1:])
        self.assertEqual(len(runs[0][0]), len(runs[1][0]))
        for a, b in zip(runs[0][0], runs[1][0]):
            self.assertEqual(a[1:], b[1:])
            for field in ("features", "actions"):
                np.testing.assert_array_equal(
                    np.asarray(a[0][field], np.float32).reshape(-1),
                    np.asarray(b[0][field]).reshape(-1),
                )


if __name__ == "__main__":
    unittest.main()
