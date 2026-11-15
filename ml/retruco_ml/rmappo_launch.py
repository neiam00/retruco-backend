"""Launch one prepared RMAPPO campaign under launchd with standard CPU QoS."""

import argparse
import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path

from .snapshot import verify_snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--module",
        choices=(
            "rmappo_campaign",
            "rmappo_controls_campaign",
            "rmappo_target_campaign",
        ),
        default="rmappo_campaign",
    )
    a = p.parse_args()
    root = a.output.resolve()
    if not a.label.startswith("com.retruco.ml-cpu.") or "/" in a.label:
        p.error("invalid training job label")
    if json.loads((root / "status.json").read_text()).get("status") != "prepared":
        p.error("only a newly prepared campaign can launch")
    verify_snapshot(root / "source")
    if not (root / f"source/ml/retruco_ml/{a.module}.py").is_file():
        p.error("missing frozen RMAPPO source")
    loaded = subprocess.run(
        ["launchctl", "list", a.label], capture_output=True, text=True
    )
    if loaded.returncode == 0:
        raise RuntimeError(
            "launchd label already exists; no duplicate or automatic restart"
        )
    job = {
        "Label": a.label,
        "ProgramArguments": [
            "/usr/bin/caffeinate",
            "-i",
            sys.executable,
            "-u",
            "-m",
            "retruco_ml." + a.module,
            "--output",
            str(root),
            "--run",
        ],
        "WorkingDirectory": str(root / "source/ml"),
        "EnvironmentVariables": {
            "PYTHONPATH": str(root / "source/ml"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "RunAtLoad": True,
        "KeepAlive": False,
        "ProcessType": "Standard",
        "StandardOutPath": str(root / "launchd.stdout.log"),
        "StandardErrorPath": str(root / "launchd.stderr.log"),
    }
    path = root / "launchd.plist"
    with path.open("xb") as stream:
        plistlib.dump(job, stream)
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=True
    )
    print(
        json.dumps(
            {
                "launched": True,
                "label": a.label,
                "plist": str(path),
                "process_type": "Standard",
            }
        )
    )


if __name__ == "__main__":
    main()
