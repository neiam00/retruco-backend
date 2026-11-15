import unittest

import numpy as np

from retruco_ml.bridge import Simulator


class MappoIPCTests(unittest.TestCase):
    def test_opt_in_binary_journal_and_critic_match_json_through_complete_matches(self):
        with Simulator() as left, Simulator(transport="binary") as right:
            for players in (2, 4):
                cursors = [0] * 4
                hand = 0
                pages = 0
                request = {
                    "op": "reset",
                    "id": 0,
                    "seed": 919,
                    "training": True,
                    "roles": [True] * 4,
                    "config": {"players": players, "target": 12, "florRequired": True},
                }
                for _ in range(10000):
                    a = left.call([request])[0]
                    b = right.call([request])[0]
                    np.testing.assert_array_equal(
                        np.asarray(a["features"], np.float32), b["features"]
                    )
                    np.testing.assert_array_equal(
                        np.asarray(a["actions"] or [], np.float32).reshape(-1, 48),
                        b["actions"],
                    )
                    if a["done"]:
                        break
                    ta, tb = a["training"], b["training"]
                    pa, pb = ta["memory"], tb["memory"]
                    np.testing.assert_array_equal(
                        np.asarray(pa["events"], np.float32).reshape(-1, 768),
                        pb["events"],
                    )
                    np.testing.assert_array_equal(
                        np.asarray(ta["privileged"], np.float32), tb["privileged"]
                    )
                    self.assertEqual(
                        {k: v for k, v in pa.items() if k != "events"},
                        {k: v for k, v in pb.items() if k != "events"},
                    )
                    self.assertLessEqual(len(pa["events"]), 32)
                    pages += 1
                    if pa["hand"] != hand:
                        hand = pa["hand"]
                        cursors = [0] * 4
                    cursors[a["seat"]] = pa["end"]
                    request = {
                        "op": "observe" if pa["end"] < pa["total"] else "step",
                        "id": 0,
                        "memoryHand": hand,
                        "memoryCursors": cursors,
                        "action": pages % 6 if a["kind"] == "signal" else a["baseline"],
                    }
                else:
                    self.fail("match did not finish")
                self.assertGreater(pages, 5)


if __name__ == "__main__":
    unittest.main()
