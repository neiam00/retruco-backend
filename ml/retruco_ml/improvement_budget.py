"""Cooperative process-wide campaign deadline; never part of actor inputs."""

import os
import time

KEY = "RETRUCO_ML_DEADLINE_MONOTONIC"


class DeadlineReached(RuntimeError):
    pass


def check_deadline():
    value = os.environ.get(KEY)
    if value is not None and time.monotonic() >= float(value):
        raise DeadlineReached("bounded model-improvement campaign deadline reached")


class DeadlineMonitor:
    def __init__(self, memory):
        self.memory = memory

    def check(self, *args, **kwargs):
        check_deadline()
        return self.memory.check(*args, **kwargs)

    @property
    def peak(self):
        return self.memory.peak
