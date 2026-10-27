"""Exercise Certbot hook failures without touching a live proxy or certificate."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "renew-certificate.sh"


class CertificateRenewalTests(unittest.TestCase):
    def test_hook_scope_and_reload_failures(self):
        # These are commands at the external Compose/NGINX boundary. Reject
        # everything else, including stop/restart, which would disconnect games.
        for lineage, failure, expected_code, expected_events in (
            ("origin", "", 0, ["check", "reload"]),
            ("other-site", "", 0, []),
            (None, "", 2, []),
            ("origin", "check", 1, ["check"]),
            ("origin", "reload", 1, ["check", "reload"]),
        ):
            with self.subTest(lineage=lineage, failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                log = root / "events"
                compose = root / "compose helper"
                compose.write_text("""#!/bin/sh
set -eu
case "$*" in
  'exec -T proxy nginx -t') event=check ;;
  'exec -T proxy nginx -s reload') event=reload ;;
  *) echo 'Unexpected operation' >&2; exit 99 ;;
esac
printf '%s\\n' "$event" >> "$TEST_EVENTS"
[ "$event" != "$TEST_FAILURE" ]
""")
                compose.chmod(0o755)
                env = {**os.environ, "TEST_EVENTS": str(log), "TEST_FAILURE": failure}
                env.pop("RENEWED_LINEAGE", None)
                if lineage is not None:
                    env["RENEWED_LINEAGE"] = str(root / lineage)
                result = subprocess.run(
                    ["sh", str(SCRIPT), str(root / "origin"), str(compose)],
                    env=env, text=True, capture_output=True, timeout=5,
                )
                if lineage is None:
                    self.assertNotEqual(result.returncode, 0, result.stderr)
                else:
                    self.assertEqual(result.returncode, expected_code, result.stderr)
                events = log.read_text().splitlines() if log.exists() else []
                self.assertEqual(events, expected_events)


if __name__ == "__main__":
    unittest.main()
