"""Append-only reservations prevent changed candidates reusing release deals."""

import fcntl
import json
import re
import time
from pathlib import Path

from .splits import BLUFF_END, BLUFF_START, RELEASE_END, RELEASE_START


def validate_reservations(records):
    if not isinstance(records, dict):
        raise ValueError("invalid release seed ledger")
    for key, row in records.items():
        if not re.fullmatch("[0-9a-f]{64}", key) or row.get("fingerprint") != key:
            raise ValueError("release seed reservation fingerprint mismatch")
    for prefix, lower, upper in (
        ("", RELEASE_START, RELEASE_END),
        ("bluff_", BLUFF_START, BLUFF_END),
    ):
        intervals = []
        for row in records.values():
            start = row[prefix + "seed"]
            count = row[prefix + "pairs"]
            if (
                type(start) is not int
                or type(count) is not int
                or count < 2
                or start < lower
                or start + count > upper
            ):
                raise ValueError("invalid release seed reservation")
            intervals.append((start, start + count))
        intervals.sort()
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("overlapping release seed reservations")


def reserve(path, fingerprint, pairs=2000, bluff_pairs=1000):
    if not re.fullmatch("[0-9a-f]{64}", fingerprint):
        raise ValueError("invalid evaluation fingerprint")
    if (
        type(pairs) is not int
        or type(bluff_pairs) is not int
        or pairs < 2000
        or bluff_pairs < 1000
    ):
        raise ValueError(
            "release reservations require at least 2000 match pairs and 1000 bluff pairs"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = (
            json.loads(path.read_text())
            if path.exists()
            else {"format": 1, "reservations": {}}
        )
        if type(ledger.get("format")) is not int or ledger["format"] != 1:
            raise ValueError("unknown release seed ledger")
        records = ledger["reservations"]
        validate_reservations(records)
        if fingerprint in records:
            previous = records[fingerprint]
            if previous["pairs"] != pairs or previous["bluff_pairs"] != bluff_pairs:
                raise ValueError("reservation counts changed for this evaluation")
            return previous
        row = {
            "seed": max(
                [RELEASE_START, *[r["seed"] + r["pairs"] for r in records.values()]]
            ),
            "pairs": pairs,
            "bluff_seed": max(
                [
                    BLUFF_START,
                    *[r["bluff_seed"] + r["bluff_pairs"] for r in records.values()],
                ]
            ),
            "bluff_pairs": bluff_pairs,
            "fingerprint": fingerprint,
            "reserved_at": time.time(),
        }
        records[fingerprint] = row
        validate_reservations(records)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(ledger, indent=2) + "\n")
        temporary.replace(path)
        return row
