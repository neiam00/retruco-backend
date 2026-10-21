"""Exercise deployment failures without SSH, Docker, or a real database."""

import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "deploy-vps.sh"
REVISION = "a" * 40


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = self.root / "backend"
        (self.app / "deploy").mkdir(parents=True)
        self.protected = {
            ".env.production": "private server values\n",
            "compose.vps.yaml": "operator override\n",
            "deploy/nginx.vps.conf.template": "certificate config\n",
            ".deployed-revision": "previous-revision\n",
        }
        for name, value in self.protected.items():
            (self.app / name).write_text(value)
        (self.app / "obsolete.go").write_text("obsolete source\n")
        self.log = self.root / "events"
        self.bin = self.root / "bin"
        self.bin.mkdir()

        # Real rsync/tar operate on a fixture directory. Only external services
        # and the wait/lock utilities are substituted.
        self.compose = self.bin / "compose"
        self.executable(
            self.compose,
            """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$MOCK_LOG"
case "$*" in
  "config --services") printf 'db\\nmigrate\\napi\\nproxy\\n' ;;
  "ps -q db") printf 'database-container\\n' ;;
  "build api migrate") [[ "${MOCK_FAIL:-}" != build ]] || exit 1 ;;
  "exec -T db pg_dump -U retruco -d retruco -Fc")
    if [[ "${MOCK_DRAIN_STDIN:-}" == 1 ]]; then cat > /dev/null; fi
    [[ "${MOCK_FAIL:-}" != backup ]] || exit 1
    printf 'database-recovery-snapshot\\n'
    ;;
  "run --rm --no-deps migrate") [[ "${MOCK_FAIL:-}" != migration ]] || exit 1 ;;
  "exec -T api sh -c "*) [[ "${MOCK_FAIL:-}" != local-health ]] || exit 1 ;;
esac
""",
        )
        self.executable(self.bin / "flock", "#!/bin/sh\nexit 0\n")
        self.executable(self.bin / "sleep", "#!/bin/sh\nexit 0\n")
        self.executable(
            self.bin / "curl",
            """#!/bin/sh
printf 'public-health\\n' >> "$MOCK_LOG"
test "${MOCK_FAIL:-}" != public-health
""",
        )
        source = self.root / "source"
        (source / "deploy").mkdir(parents=True)
        (source / "Dockerfile").write_text("FROM scratch\n")
        (source / "compose.yaml").write_text("services: {}\n")
        (source / "new.go").write_text("new source\n")
        # Even if these arrive in an archive, server-owned files must survive.
        for name in self.protected:
            (source / name).write_text("must not replace server data\n")
        self.archive = self.root / "source.tar.gz"
        with tarfile.open(self.archive, "w:gz") as archive:
            archive.add(source, arcname=".")
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "RETRUCO_DEPLOY_DIR": str(self.app),
            "RETRUCO_COMPOSE": str(self.compose),
            "MOCK_LOG": str(self.log),
        }

    @staticmethod
    def executable(path, contents):
        path.write_text(contents)
        path.chmod(0o755)

    def deploy(self, fail="", streamed=False):
        command = ["bash", str(SCRIPT), str(self.archive), REVISION]
        if streamed:
            command = ["bash", "-s", "--", str(self.archive), REVISION]
        result = subprocess.run(
            command,
            input=SCRIPT.read_text() if streamed else None,
            env={**self.env, "MOCK_FAIL": fail, "MOCK_DRAIN_STDIN": "1" if streamed else "0"},
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.events = self.log.read_text().splitlines()
        self.assertFalse(self.archive.exists(), result.stderr)
        self.assertFalse(list(self.root.glob(".deploy-source.*")))
        return result

    def assert_server_files_preserved(self, check_revision=True):
        for name, value in self.protected.items():
            if check_revision or name != ".deployed-revision":
                self.assertEqual((self.app / name).read_text(), value)

    def test_success_preserves_host_data_and_removes_obsolete_source(self):
        result = self.deploy()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_server_files_preserved(check_revision=False)
        self.assertFalse((self.app / "obsolete.go").exists())
        self.assertTrue((self.app / "new.go").exists())
        self.assertEqual((self.app / ".deployed-revision").read_text(), REVISION + "\n")
        backups = list((self.root / "backups").glob("*.dump"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), "database-recovery-snapshot\n")
        stop = self.events.index("stop api")
        backup = self.events.index("exec -T db pg_dump -U retruco -d retruco -Fc")
        migration = self.events.index("run --rm --no-deps migrate")
        self.assertLess(self.events.index("build api migrate"), stop)
        self.assertLess(stop, backup)
        self.assertLess(backup, migration)
        self.assertLess(migration, self.events.index("up -d --no-build --no-deps api"))
        self.assertIn("up -d --no-build --no-deps --force-recreate proxy", self.events)
        self.assertIn("public-health", self.events)

    def test_build_failure_does_not_stop_running_api(self):
        result = self.deploy("build")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("stop api", self.events)
        self.assertNotIn("run --rm --no-deps migrate", self.events)
        self.assert_server_files_preserved()

    def test_streamed_script_finishes_when_backup_reads_stdin(self):
        result = self.deploy(streamed=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.app / ".deployed-revision").read_text(), REVISION + "\n")
        self.assertIn("run --rm --no-deps migrate", self.events)
        self.assertIn("up -d --no-build --no-deps api", self.events)
        self.assertIn("public-health", self.events)
        self.assertEqual(len(list((self.root / "backups").glob("*.dump"))), 1)

    def test_backup_failure_restarts_existing_api_without_migrating(self):
        result = self.deploy("backup")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("up -d --no-build --no-deps --no-recreate api", self.events)
        self.assertNotIn("run --rm --no-deps migrate", self.events)
        self.assertEqual(list((self.root / "backups").iterdir()), [])
        self.assert_server_files_preserved()

    def test_migration_failure_retains_backup_and_does_not_start_api(self):
        result = self.deploy("migration")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("up -d --no-build --no-deps --no-recreate api", self.events)
        self.assertNotIn("up -d --no-build --no-deps api", self.events)
        self.assertEqual(len(list((self.root / "backups").glob("*.dump"))), 1)
        self.assert_server_files_preserved()

    def test_failed_public_health_does_not_record_success(self):
        result = self.deploy("public-health")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("public-health", self.events)
        self.assertEqual((self.app / ".deployed-revision").read_text(), "previous-revision\n")
        self.assertNotIn("health checks passed", result.stdout)

    def test_failed_api_health_does_not_recreate_proxy_or_report_success(self):
        result = self.deploy("local-health")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("up -d --no-build --no-deps --force-recreate proxy", self.events)
        self.assertNotIn("public-health", self.events)
        self.assertEqual((self.app / ".deployed-revision").read_text(), "previous-revision\n")


if __name__ == "__main__":
    unittest.main()
