"""Launch an explicitly prepared local training campaign with standard QoS."""

import argparse
import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if not args.label.startswith("com.retruco.ml-cpu.") or "/" in args.label:
        parser.error("use a retruco ml-cpu job label")
    if json.loads((root / "status.json").read_text()).get("status") != "prepared":
        parser.error("only a newly prepared campaign may be launched automatically")
    if not (root / "source/ml/retruco_ml/target_campaign.py").is_file():
        parser.error("frozen target campaign source is missing")
    found = subprocess.run(
        ["launchctl", "list", args.label], capture_output=True, text=True
    )
    if found.returncode == 0:
        raise RuntimeError(
            "launchd label already loaded; inspect rather than duplicate or restart it"
        )
    job = {
        "Label": args.label,
        "ProgramArguments": [
            "/usr/bin/caffeinate",
            "-i",
            sys.executable,
            "-u",
            "-m",
            "retruco_ml.target_campaign",
            "--output",
            str(root),
            "--resume",
        ],
        "WorkingDirectory": str(root / "source/ml"),
        "EnvironmentVariables": {"PYTHONPATH": str(root / "source/ml")},
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
                "label": args.label,
                "process_type": "Standard",
                "plist": str(path),
            }
        )
    )


if __name__ == "__main__":
    main()
