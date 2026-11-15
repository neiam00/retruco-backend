"""Explicit deal-seed partitions. Training never samples validation/test ranges.

The old 1,000,000 range was touched by an early two-pair smoke run and is not a
release holdout. Frozen pilot validation explicitly uses VALIDATION_START.
"""

TRAIN_MIN = 2**32
TRAIN_MAX = 2**53
VALIDATION_START = 100000
RELEASE_START = 2000000
RELEASE_END = 2100000
BLUFF_START = 3000000
BLUFF_END = 3100000
