"""Run candidate integration checks with durable, hashed evidence; never approve."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from .artifacts import digest, validate_manifest
from .assessment import load_checkpoint
from .export import serialize


def git(root, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def dirty_fingerprint(root):
    paths = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "--modified",
                "--deleted",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    contents = {
        name: digest(root / name) if (root / name).is_file() else None
        for name in sorted(set(paths))
        if name
    }
    return hashlib.sha256(json.dumps(contents, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="two rematches and 20 latency samples; insufficient for release",
    )
    args = parser.parse_args()
    backend = Path(__file__).resolve().parents[2]
    frontend = args.frontend.resolve()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "verification.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("this verification already has a live runner")
    bundle = args.bundle.resolve()
    checkpoint = args.checkpoint.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    model = load_checkpoint(checkpoint)
    validate_manifest(
        manifest,
        (bundle / "policy.bin").read_bytes(),
        (bundle / "policy.wasm").read_bytes(),
    )
    if (
        model.checkpoint_sha256 != manifest["checkpoint_sha256"]
        or serialize(model)[0] != (bundle / "policy.bin").read_bytes()
    ):
        raise ValueError("candidate checkpoint and bundle differ")
    if (
        digest(bundle / "policy.bin") != manifest["sha256"]
        or digest(bundle / "policy.wasm") != manifest["wasm_sha256"]
    ):
        raise ValueError("candidate artifact checksum mismatch")
    identities = {
        "backend_commit": git(backend, "rev-parse", "HEAD"),
        "frontend_commit": git(frontend, "rev-parse", "HEAD"),
        "source_dirty": bool(
            git(backend, "status", "--porcelain")
            or git(frontend, "status", "--porcelain")
        ),
        "backend_dirty_sha256": dirty_fingerprint(backend),
        "frontend_dirty_sha256": dirty_fingerprint(frontend),
    }
    if not args.smoke and (
        identities["source_dirty"]
        or manifest.get("source_dirty") is not False
        or manifest["exporter_commit"] != identities["backend_commit"]
    ):
        raise ValueError(
            "full verification requires clean commits and a bundle exported from the current backend commit"
        )
    configuration = {
        **identities,
        "checkpoint_sha256": model.checkpoint_sha256,
        "model_sha256": manifest["sha256"],
        "wasm_sha256": manifest["wasm_sha256"],
        "manifest_sha256": digest(bundle / "manifest.json"),
        "smoke": args.smoke,
        "frontend": str(frontend),
    }
    if (root / "configuration.json").exists() and json.loads(
        (root / "configuration.json").read_text()
    ) != configuration:
        raise ValueError(
            "verification source/model/configuration changed; use a new output directory"
        )
    write(root / "configuration.json", configuration)
    report = {**configuration, "approved": False, "complete": False, "checks": []}
    status = {"pid": os.getpid(), "status": "running", "started": time.time()}

    def interrupted(*_):
        raise KeyboardInterrupt("verification interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    environment = dict(
        os.environ,
        APP_ENV="test",
        DATABASE_URL="",
        DATABASE_TEST_URL="",
        API_PROXY_SECRET="",
        API_ORIGIN="",
        BACKEND_ORIGIN="",
        CARD_ASSET_ORIGIN="",
        GOOGLE_ANALYTICS_ID="",
        SITE_ORIGIN="",
        VITE_CPU_ML_ENABLED="true",
        VITE_CPU_EVALUATION="true",
        VITE_CPU_MODEL_MANIFEST="/cpu-verification/manifest.json",
        CPU_SOAK_MATCHES="2" if args.smoke else "10",
        PYTHONPATH=str(backend / "ml"),
    )

    def run(name, command, cwd, artifacts=(), cache=True):
        log = root / (name + ".log")
        receipt = root / (name + ".receipt.json")
        previous = json.loads(receipt.read_text()) if receipt.exists() else None
        intact = previous and all(
            (root / path).is_file() and digest(root / path) == value
            for path, value in previous.get("artifacts", {}).items()
        )
        if (
            cache
            and intact
            and set(previous.get("artifacts", {})) == set(artifacts)
            and previous["command"] == command
            and previous["exit_code"] == 0
            and log.exists()
            and digest(log) == previous["evidence_sha256"]
        ):
            report["checks"].append(previous)
            return
        print(json.dumps({"stage": name}), flush=True)
        with log.open("w") as output:
            child = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status.update(stage=name, child_pid=child.pid)
            write(root / "status.json", status)
            try:
                code = child.wait()
            except BaseException:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
        entry = {
            "name": name,
            "command": command,
            "exit_code": code,
            "evidence_path": log.name,
            "evidence_sha256": digest(log),
            "artifacts": {
                path: digest(root / path)
                for path in artifacts
                if (root / path).is_file()
            },
        }
        write(receipt, entry)
        report["checks"].append(entry)
        write(root / "integration.json", report)
        if code:
            raise RuntimeError(f"{name} failed; inspect {log}")
        if set(entry["artifacts"]) != set(artifacts):
            raise ValueError("verification command did not produce its evidence")

    python = str(backend / "ml/.venv/bin/python")
    try:
        run(
            "build-simulator",
            ["go", "build", "-o", "bin/selfplay", "./cmd/selfplay"],
            backend,
            cache=False,
        )
        run(
            "build-native-runtime",
            ["cargo", "build", "--locked", "--offline", "--release"],
            backend / "ml/runtime",
            cache=False,
        )
        run(
            "build-wasm-runtime",
            [
                "cargo",
                "build",
                "--locked",
                "--offline",
                "--release",
                "--target",
                "wasm32v1-none",
            ],
            backend / "ml/runtime",
            cache=False,
        )
        run("native-tests", ["go", "test", "-race", "-v", "./..."], backend)
        native_log = (root / "native-tests.log").read_text()
        if (
            "--- PASS: TestCPUv2HumanSignalInvalidatesWorkAndAuthorityIsRequired "
            not in native_log
        ):
            raise ValueError("stale-token behavior was not exercised")
        report["checks"].append(
            {
                **next(c for c in report["checks"] if c["name"] == "native-tests"),
                "name": "stale-work",
            }
        )
        run(
            "wasm-tests",
            [python, "-m", "unittest", "discover", "-s", "tests", "-v"],
            backend / "ml",
        )
        run("frontend-tests", ["pnpm", "test"], frontend)
        run("typecheck", ["pnpm", "check"], frontend)
        run("production-build", ["pnpm", "build"], frontend, cache=False)
        run(
            "production-approval-guard",
            ["node", "scripts/check-cpu-production.mjs"],
            frontend,
            cache=False,
        )
        run(
            "runtime-parity",
            [
                python,
                "-m",
                "retruco_ml.parity",
                str(checkpoint),
                str(bundle),
                "--frontend",
                str(frontend),
                "--output",
                str(root / "parity.json"),
            ],
            backend / "ml",
            artifacts=("parity.json", "parity.evidence.json"),
        )
        run(
            "performance-inputs",
            [
                python,
                "-m",
                "retruco_ml.benchmark_cases",
                str(root / "performance-cases.json"),
            ],
            backend / "ml",
            artifacts=("performance-cases.json",),
        )
        run(
            "browser-performance",
            [
                "node",
                "scripts/benchmark-cpu.mjs",
                str(bundle),
                str(root / "performance-cases.json"),
                str(root / "performance.json"),
                "20" if args.smoke else "200",
            ],
            frontend,
            artifacts=("performance.json",),
        )
        run(
            "browser-integration",
            [
                "node",
                "scripts/test-cpu-integration.mjs",
                str(bundle),
                str(root / "browser"),
            ],
            frontend,
            artifacts=("browser/integration.json", "browser/playwright.json"),
        )
        browser = json.loads((root / "browser/integration.json").read_text())
        for key in (
            "checkpoint_sha256",
            "model_sha256",
            "wasm_sha256",
            "backend_commit",
            "frontend_commit",
        ):
            if browser[key] != report[key]:
                raise ValueError("browser evidence identity mismatch: " + key)
        for check in browser["checks"]:
            path = Path("browser") / check["evidence_path"]
            if digest(root / path) != check["evidence_sha256"]:
                raise ValueError("browser raw evidence changed")
            report["checks"].append({**check, "evidence_path": str(path)})
        required = {
            "browser-1v1",
            "browser-2v2",
            "controller-handoff",
            "server-restore",
            "corrupt-artifacts",
            "soak-smoke" if args.smoke else "soak",
        }
        if not required.issubset(
            {c["name"] for c in browser["checks"] if c["exit_code"] == 0}
        ):
            raise ValueError("browser verification coverage incomplete")
        if (
            git(backend, "rev-parse", "HEAD") != identities["backend_commit"]
            or git(frontend, "rev-parse", "HEAD") != identities["frontend_commit"]
        ):
            raise ValueError("source commits changed during verification")
        report["source_dirty"] |= bool(
            git(backend, "status", "--porcelain")
            or git(frontend, "status", "--porcelain")
        )
        if (
            dirty_fingerprint(backend) != identities["backend_dirty_sha256"]
            or dirty_fingerprint(frontend) != identities["frontend_dirty_sha256"]
        ):
            raise ValueError("source edits changed during verification")
        if digest(bundle / "manifest.json") != configuration["manifest_sha256"]:
            raise ValueError("candidate manifest changed during verification")
        report["complete"] = True
        write(root / "integration.json", report)
        write(
            root / "evidence-paths.json",
            {
                "integration": "integration.json",
                "runtime_parity": "parity.json",
                "performance": "performance.json",
            },
        )
        status.update(status="complete", child_pid=None)
    except BaseException as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        write(root / "status.json", status)


if __name__ == "__main__":
    main()
