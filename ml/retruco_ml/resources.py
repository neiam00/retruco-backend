"""Monitor resident memory of the trainer and its owned simulator processes."""

import os
import subprocess
import time

import torch


class MemoryBudget:
    def __init__(self, processes, gib=16):
        if not 0 < gib <= 16:
            raise ValueError("memory target must be in (0,16] GiB")
        self.pids = [os.getpid(), *[process.pid for process in processes]]
        self.limit = gib * 1024**3
        self.peak = 0
        self.checked = 0

    def check(self, force=False):
        now = time.monotonic()
        if not force and now - self.checked < 2:
            return
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", ",".join(map(str, self.pids))],
            capture_output=True,
            text=True,
            check=True,
        )
        resident = sum(int(line) for line in result.stdout.split()) * 1024
        if torch.backends.mps.is_available():
            # Conservative: unified GPU allocations may overlap process RSS.
            resident += torch.mps.driver_allocated_memory()
        self.peak = max(self.peak, resident)
        self.checked = now
        if resident > self.limit:
            raise MemoryError(
                f"training group memory {resident / 1024**3:.2f} GiB exceeds target {self.limit / 1024**3:.2f} GiB; resume from last complete checkpoint with fewer environments"
            )
