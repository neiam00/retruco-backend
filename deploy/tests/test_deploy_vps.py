"""Exercise deployment failures without SSH, Docker, or a real database."""

import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "deploy-vps.sh"
REVISION = "a" * 40
IMAGE_ID = "sha256:" + "b" * 64


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = self.root / "backend"
        (self.app / "deploy").mkdir(parents=True)
        self.protected = {
            ".env.production": "ALLOWED_ORIGINS=https://cards.example.test,https://other.example.test\n",
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
if [[ "$*" == *"/internal/deployment"* ]]; then
  exec sh -c "$6" "$7" "${@:8}"
fi
case "$*" in
  "exec -T api printenv ALLOWED_ORIGINS")
    [[ "${MOCK_FAIL:-}" != origins ]] || exit 1
    printf '%s\\n' "$MOCK_ALLOWED_ORIGINS"
    ;;
  "config --services")
    printf 'db\\nmigrate\\napi\\nproxy\\n'
    if [[ "${MOCK_FAIL:-}" == legacy-turn ]]; then printf 'turn\\n'; fi
    ;;
  "ps -q db") printf 'database-container\\n' ;;
  "exec -T db pg_dump -U retruco -d retruco -Fc")
    if [[ "${MOCK_DRAIN_STDIN:-}" == 1 ]]; then cat > /dev/null; fi
    [[ "${MOCK_FAIL:-}" != backup ]] || exit 1
    printf 'database-recovery-snapshot\\n'
    ;;
  "run --rm --pull never --no-deps migrate") [[ "${MOCK_FAIL:-}" != migration ]] || exit 1 ;;
  "exec -T api sh -c "*) [[ "${MOCK_FAIL:-}" != local-health ]] || exit 1 ;;
esac
""",
        )
        self.executable(
            self.bin / "docker",
            """#!/usr/bin/env bash
set -eu
printf 'docker %s\\n' "$*" >> "$MOCK_LOG"
case "$*" in
  "image load --input "*)
    if [[ "${MOCK_DRAIN_STDIN:-}" == 1 ]]; then cat > /dev/null; fi
    [[ "${MOCK_FAIL:-}" != image-load ]] || exit 1
    ;;
  "version --format "*) printf 'linux/amd64\\n' ;;
  "image inspect --format "*)
    [[ "${MOCK_FAIL:-}" != image-missing ]] || exit 1
    image_id="$MOCK_IMAGE_ID"
    revision="$MOCK_REVISION"
    platform=linux/amd64
    case "${MOCK_FAIL:-}" in
      image-id) image_id=sha256:wrong ;;
      image-revision) revision=wrong ;;
      image-platform) platform=linux/arm64 ;;
    esac
    printf '%s %s %s\\n' "$image_id" "$revision" "$platform"
    ;;
  "image tag "*) [[ "${MOCK_FAIL:-}" != image-tag ]] || exit 1 ;;
  *) exit 1 ;;
esac
""",
        )
        self.executable(self.bin / "flock", "#!/bin/sh\nexit 0\n")
        self.executable(self.bin / "sleep", "#!/bin/sh\nprintf 'sleep %s\\n' \"$*\" >> \"$MOCK_LOG\"\n[ \"${MOCK_FAIL:-}\" != notice-wait ]\n")
        self.executable(
            self.bin / "wget",
            """#!/usr/bin/env bash
set -eu
case "$*" in
  *"/api/health"*)
    if [[ "${MOCK_FAIL:-}" == legacy-notices ]]; then
      printf '{"status":"ok","rulesVersion":"2"}'
    else
      printf '{"status":"ok","deploymentNotices":true,"rulesVersion":"2"}'
    fi
    ;;
  *"/internal/deployment"*)
    body=""
    for arg in "$@"; do
      if [[ "$arg" == --post-data=* ]]; then body="${arg#--post-data=}"; fi
    done
    printf 'notice:%s\\n' "$body" >> "$MOCK_LOG"
    [[ "${MOCK_FAIL:-}" != notice || "$body" != '{"updating":true}' ]] || exit 1
    ;;
  *) exit 1 ;;
esac
""",
        )
        self.executable(
            self.bin / "curl",
            """#!/bin/sh
printf 'public-health\\n' >> "$MOCK_LOG"
printf 'public-health-url:%s\\n' "$*" >> "$MOCK_LOG"
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
        self.image_archive = self.root / "image.tar.gz"
        self.image_archive.write_bytes(b"CI-built image archive")
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "RETRUCO_DEPLOY_DIR": str(self.app),
            "RETRUCO_COMPOSE": str(self.compose),
            "MOCK_LOG": str(self.log),
            "MOCK_ALLOWED_ORIGINS": "https://cards.example.test,https://other.example.test",
            "MOCK_IMAGE_ID": IMAGE_ID,
            "MOCK_REVISION": REVISION,
        }

    @staticmethod
    def executable(path, contents):
        path.write_text(contents)
        path.chmod(0o755)

    def deploy(self, fail="", streamed=False):
        arguments = [str(self.archive), REVISION, str(self.image_archive), IMAGE_ID]
        command = ["bash", str(SCRIPT), *arguments]
        if streamed:
            command = ["bash", "-s", "--", *arguments]
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
        self.assertFalse(self.image_archive.exists(), result.stderr)
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
        migration = self.events.index("run --rm --pull never --no-deps migrate")
        load = self.events.index(f"docker image load --input {self.image_archive}")
        tag = self.events.index(f"docker image tag {IMAGE_ID} retruco-backend:production")
        notice = self.events.index('notice:{"updating":true}')
        self.assertLess(load, tag)
        self.assertLess(tag, notice)
        self.assertFalse(any("build api migrate" in event for event in self.events))
        self.assertLess(notice, self.events.index("sleep 10"))
        self.assertLess(self.events.index("sleep 10"), stop)
        self.assertLess(stop, backup)
        self.assertLess(backup, migration)
        self.assertLess(migration, self.events.index("up -d --no-build --pull never --no-deps api"))
        self.assertIn("up -d --no-build --no-deps --force-recreate proxy", self.events)
        self.assertIn("public-health", self.events)

    def test_image_load_failure_does_not_replace_source_or_stop_running_api(self):
        result = self.deploy("image-load")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.app / "obsolete.go").exists())
        self.assertNotIn("stop api", self.events)
        self.assertNotIn("run --rm --pull never --no-deps migrate", self.events)
        self.assertNotIn('notice:{"updating":true}', self.events)
        self.assert_server_files_preserved()

    def assert_image_rejected(self, failure):
        result = self.deploy(failure)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"docker image load --input {self.image_archive}", self.events)
        self.assertTrue((self.app / "obsolete.go").exists())
        self.assertNotIn("stop api", self.events)
        self.assertNotIn('notice:{"updating":true}', self.events)
        self.assertFalse(any(event.startswith("docker image tag ") for event in self.events))
        self.assert_server_files_preserved()

    def test_missing_expected_image_does_not_interrupt_players(self):
        self.assert_image_rejected("image-missing")

    def test_mismatched_image_id_does_not_interrupt_players(self):
        self.assert_image_rejected("image-id")

    def test_wrong_image_commit_does_not_interrupt_players(self):
        self.assert_image_rejected("image-revision")

    def test_incompatible_image_platform_does_not_interrupt_players(self):
        self.assert_image_rejected("image-platform")

    def test_image_tag_failure_keeps_existing_api_and_revision(self):
        result = self.deploy("image-tag")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("stop api", self.events)
        self.assertNotIn('notice:{"updating":true}', self.events)
        self.assert_server_files_preserved()

    def test_legacy_installation_preserves_running_origins_before_source_update(self):
        env_file = self.app / ".env.production"
        env_file.write_text("# private server values\n")
        result = self.deploy()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            env_file.read_text(),
            "# private server values\n\nALLOWED_ORIGINS=https://cards.example.test,https://other.example.test\n",
        )
        self.assertIn("Preserved installed frontend origins", result.stdout)
        self.assertTrue(any(event.endswith(" https://cards.example.test/api/health") for event in self.events))

    def test_custom_health_url_does_not_require_running_origin_lookup(self):
        self.env["RETRUCO_HEALTH_URL"] = "https://another.example.test/api/health"
        result = self.deploy("origins")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("exec -T api printenv ALLOWED_ORIGINS", self.events)
        self.assertTrue(any(event.endswith(" https://another.example.test/api/health") for event in self.events))

    def test_missing_running_origins_blocks_before_replacing_source(self):
        result = self.deploy("origins")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.app / "obsolete.go").exists())
        self.assertNotIn(f"docker image load --input {self.image_archive}", self.events)
        self.assertNotIn("stop api", self.events)
        self.assert_server_files_preserved()

    def test_origin_migration_rejects_multiline_values_before_writing_env(self):
        self.env["MOCK_ALLOWED_ORIGINS"] = "https://cards.example.test\nUNEXPECTED=value"
        env_file = self.app / ".env.production"
        env_file.write_text("# private server values\n")
        result = self.deploy()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(env_file.read_text(), "# private server values\n")
        self.assertTrue((self.app / "obsolete.go").exists())
        self.assertNotIn("stop api", self.events)

    def test_legacy_turn_overlay_blocks_before_source_or_service_changes(self):
        result = self.deploy("legacy-turn")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Retire the local Coturn", result.stderr)
        self.assertTrue((self.app / "obsolete.go").exists())
        self.assertFalse((self.app / "new.go").exists())
        self.assertNotIn(f"docker image load --input {self.image_archive}", self.events)
        self.assertNotIn("stop api", self.events)
        self.assert_server_files_preserved()

    def test_notice_failure_keeps_api_running_and_cancels_the_notice(self):
        result = self.deploy("notice")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("stop api", self.events)
        self.assertIn('notice:{"updating":false}', self.events)
        self.assertNotIn("run --rm --pull never --no-deps migrate", self.events)

    def test_cancelled_notice_resumes_play_without_stopping_api(self):
        result = self.deploy("notice-wait")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('notice:{"updating":true}', self.events)
        self.assertIn('notice:{"updating":false}', self.events)
        self.assertNotIn("stop api", self.events)

    def test_first_rollout_can_upgrade_an_api_without_notice_support(self):
        result = self.deploy("legacy-notices")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bootstrapping", result.stdout)
        self.assertNotIn("sleep 10", self.events)
        self.assertIn("stop api", self.events)

    def test_streamed_script_finishes_when_backup_reads_stdin(self):
        result = self.deploy(streamed=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.app / ".deployed-revision").read_text(), REVISION + "\n")
        self.assertIn("run --rm --pull never --no-deps migrate", self.events)
        self.assertIn("up -d --no-build --pull never --no-deps api", self.events)
        self.assertIn("public-health", self.events)
        self.assertEqual(len(list((self.root / "backups").glob("*.dump"))), 1)

    def test_backup_failure_restarts_existing_api_without_migrating(self):
        result = self.deploy("backup")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("up -d --no-build --no-deps --no-recreate api", self.events)
        self.assertNotIn("run --rm --pull never --no-deps migrate", self.events)
        self.assertEqual(list((self.root / "backups").iterdir()), [])
        self.assert_server_files_preserved()

    def test_migration_failure_retains_backup_and_does_not_start_api(self):
        result = self.deploy("migration")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("up -d --no-build --no-deps --no-recreate api", self.events)
        self.assertNotIn("up -d --no-build --pull never --no-deps api", self.events)
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
