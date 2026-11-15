import os
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, patch

from retruco_ml.rmappo_controls_campaign import (
    BudgetExhausted,
    arm_deadline,
    check_cap,
    reap_group,
)


class LifecycleTests(unittest.TestCase):
    def test_permission_error_requires_proof_that_the_owned_group_has_no_live_members(
        self,
    ):
        process = Mock(pid=4321)
        for listing, code, safe in [
            ("4321 Z\n123 S\n", 0, True),
            ("123 S\n", 0, True),
            ("4321 S\n", 0, False),
            ("", 1, False),
            ("unreadable", 0, False),
        ]:
            with (
                self.subTest(listing=listing, code=code),
                patch("os.killpg", side_effect=PermissionError("denied")),
                patch(
                    "subprocess.run", return_value=Mock(stdout=listing, returncode=code)
                ),
            ):
                if safe:
                    reap_group(process)
                else:
                    with self.assertRaises(PermissionError):
                        reap_group(process)

    def test_deadline_interrupts_comparison_work_outside_child_waits(self):
        previous = signal.getsignal(signal.SIGALRM)
        timer = signal.getitimer(signal.ITIMER_REAL)
        try:
            with self.assertRaises(BudgetExhausted):
                arm_deadline(time.monotonic() + 0.03)
                time.sleep(1)
            with self.assertRaises(BudgetExhausted):
                check_cap(time.monotonic() - 1)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            if timer[0]:
                signal.setitimer(signal.ITIMER_REAL, *timer)

    def test_exited_leader_cannot_leave_live_simulator_descendants(self):
        code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print(p.pid,flush=True)"
        p = subprocess.Popen(
            [sys.executable, "-c", code],
            start_new_session=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            child = int(p.stdout.readline())
            p.wait(timeout=3)

            def live():
                state = subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(child)],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                return bool(state) and not state.startswith("Z")

            self.assertTrue(live())
            # Old leader-only cleanup does not signal a surviving descendant.
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
            self.assertTrue(live())
            reap_group(p)
            until = time.monotonic() + 2
            while live() and time.monotonic() < until:
                time.sleep(0.02)
            self.assertFalse(live())
        finally:
            reap_group(p)
            p.stdout.close()


if __name__ == "__main__":
    unittest.main()
