import unittest

from retruco_ml.release_v4_campaign import specifications
from retruco_ml.selective_bluff import (
    MATCHES,
    MIN_OFFERS,
    PROTOCOL,
    start_seed,
    verify_ordinary,
)


class SelectiveOpeningTests(unittest.TestCase):
    def test_new_protocol_keeps_the_old_fixed_hand_reports_and_disjoint_deals(self):
        old = specifications()
        new = specifications(selective=True)
        self.assertEqual(set(new) - set(old), {"bluff-ordinary"})
        self.assertTrue(all(new[n] == spec for n, spec in old.items()))
        first = start_seed({"seed": 2000000, "pairs": 2000})
        second = start_seed({"seed": 2002000, "pairs": 2000})
        self.assertLessEqual(first + MATCHES, second)

    def test_qualification_requires_real_support_and_positive_matched_outcomes(self):
        reservation = {"seed": 2000000, "pairs": 2000}
        data = {
            "fixture": PROTOCOL,
            "opponent": "baseline",
            "seed": start_seed(reservation),
            "matches": MATCHES,
            "rows": [None] * MATCHES,
        }
        with self.assertRaisesRegex(ValueError, "Too few"):
            verify_ordinary(data, reservation)
        data["rows"][:MIN_OFFERS] = [["a" * 64, 0.03, 0, 0]] * MIN_OFFERS
        with self.assertRaisesRegex(ValueError, "not established"):
            verify_ordinary(data, reservation)
        data["rows"][0] = ["b" * 64, 0.5, 1, 0]
        with self.assertRaisesRegex(ValueError, "probability cap"):
            verify_ordinary(data, reservation)
