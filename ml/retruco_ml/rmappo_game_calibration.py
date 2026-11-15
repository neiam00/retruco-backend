"""Calibrate an actual saved learner on fresh paired deals, never the opponent.

Scaling only the final game-scoring weight multiplies every within-decision
log-odds by the gain (up to FP32 storage rounding). The common bias cancels.
Signal logits, recurrent state, legal masks, partners and the simulator stay
unchanged. Derived files are warm-start initializers, not resumable optimizer
snapshots. The unmodified resumable source is always retained separately.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .controlled_train import atomic_json
from .cooperation_refusal_campaign import validate_inputs
from .cooperation_refusal_compare import clusters
from .recurrent import file_hash
from .rmappo_controls_campaign import reap_group
from .rmappo_controls_compare import _assessment
from .snapshot import verify_snapshot
from .target_compare import verified_report

GAINS = {"parent": 1.0, "soft": 0.75, "sharp": 1.5, "sharper": 2.0}


def initializer(parent, destination, gain):
    if gain not in GAINS.values():
        raise ValueError("undeclared calibration gain")
    if destination.exists():
        raise FileExistsError(destination)
    if gain == 1.0:
        with parent.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        return
    s = torch.load(parent, map_location="cpu", weights_only=True)
    if s.get("kind") != "rmappo-v1" or s["configuration"]["cell"] != "lstm":
        raise ValueError("require an RMAPPO LSTM initializer")
    out = {
        k: s[k]
        for k in (
            "kind",
            "configuration",
            "actor",
            "critic",
            "choices",
            "updates",
            "league",
            "encoder",
            "memory_version",
        )
    }
    out["actor"] = dict(s["actor"])
    out["actor"]["score.2.weight"] = s["actor"]["score.2.weight"] * gain
    out.update(
        approved=False,
        initialization_only=True,
        calibration={
            "game_logit_gain": gain,
            "parent_sha256": file_hash(parent),
            "signal_gain": 1.0,
        },
    )
    with destination.open("xb") as f:
        torch.save(out, f)


def rank(row):
    h, t = (row["modes"][k]["win_rate"] for k in ("2", "4"))
    return (row["acceptable_strength"], min(h / 0.70, t / 0.60), t, h)


def promote(candidate, parent, primary):
    modes = candidate["modes"].values()
    if rank(candidate) <= rank(parent):
        return False
    if any(x["gain_ci95"][0] <= -0.02 for x in modes) or not any(
        x["gain_ci95"][0] > 0 for x in modes
    ):
        return False
    if any(
        parent["modes"][k]["target_met"] and not candidate["modes"][k]["target_met"]
        for k in ("2", "4")
    ):
        return False
    if any(x["gain"] < -0.02 for s, x in candidate["strata"].items() if s in primary):
        return False
    if any(
        x["gain"] < -0.05 and x["gain_ci95"][1] < 0
        for s, x in candidate["strata"].items()
        if s not in primary
    ):
        return False
    return not any(
        x["stratum"] in primary or x["gain"] < -0.05
        for x in candidate["configuration_regressions"]
    )


def run(root, parent, previous, streams):
    root = root.resolve()
    parent = parent.resolve()
    previous = previous.resolve()
    root.mkdir(exist_ok=False, parents=True)
    config = json.loads((previous / "configuration.json").read_text())
    expected = json.loads((previous / "selection.json").read_text())
    if (
        parent != Path(expected["nominee_checkpoint"]).resolve()
        or file_hash(parent) != expected["nominee_checkpoint_sha256"]
    ):
        raise ValueError("calibration source must be the pre-frozen prior nominee")
    inputs = previous / "inputs/manifest.json"
    source = previous / "source"
    digest = file_hash(inputs)
    verify_snapshot(source)
    validate_inputs(inputs.parent)
    models = {}
    (root / "candidates").mkdir()
    started = time.monotonic()
    deadline = started + 1800
    for name, gain in GAINS.items():
        p = root / "candidates" / f"{name}.pt"
        initializer(parent, p, gain)
        models[name] = p
    plans = {
        "gains": GAINS,
        "parent": str(parent),
        "parent_sha256": file_hash(parent),
        "primary": config["primary_specs"],
        "diagnostics": config["diagnostic_specs"],
        "streams": streams,
        "inputs_sha256": digest,
        "model_sha256": {n: file_hash(p) for n, p in models.items()},
        "frozen_evaluation_source": str(source),
        "rule": "Screen native primary modes; nominate one guarded candidate by both targets, weaker normalized mode, teams, heads-up. On separate recheck require a gain CI above zero in at least one mode, both lower gain CIs above -2pp, preserve attained targets, no material primary regression and no confidence-supported diagnostic drop greater than 5pp; otherwise retain original.",
        "scope": "Development calibration on separate deals; not final confirmation or model approval.",
    }
    atomic_json(root / "plan.json", plans)

    def evaluate(names, phase, include_diagnostics):
        specs = config["primary_specs"] + (
            config["diagnostic_specs"] if include_diagnostics else []
        )
        queue = []
        finished = []
        live = {}
        for name in names:
            for spec in specs:
                stream = streams[
                    "calibration_screen"
                    if phase == "screen"
                    else "calibration_recheck"
                    if spec in config["primary_specs"]
                    else "calibration_diagnostic"
                ]
                out = root / phase / name / (spec["id"] + ".json")
                out.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    "-B",
                    "-u",
                    "-m",
                    "retruco_ml.rmappo_evaluate",
                    str(models[name]),
                    "--inputs",
                    str(inputs),
                    "--simulator",
                    str(source / "bin/selfplay"),
                    "--output",
                    str(out),
                    "--partner",
                    spec["partner"],
                    "--opponent",
                    spec["opponent"],
                    "--players",
                    str(spec["players"]),
                    "--pairs",
                    str(stream["pairs"]),
                    "--seed",
                    str(stream["seed"]),
                    "--signals",
                    "normal",
                ]
                queue.append((name, spec, stream, out, cmd))
        last_status = 0.0
        expected_count = len(queue)
        try:
            while queue or live:
                if time.monotonic() >= deadline:
                    raise TimeoutError("30-minute calibration allocation exhausted")
                while queue and len(live) < 4:
                    job = queue.pop(0)
                    _, _, _, out, cmd = job
                    with out.with_suffix(".log").open("xb") as stream:
                        process = subprocess.Popen(
                            cmd,
                            cwd=source / "ml",
                            env={
                                **os.environ,
                                "PYTHONPATH": str(source / "ml"),
                                "PYTHONDONTWRITEBYTECODE": "1",
                            },
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    live[process.pid] = (process, job)
                for pid, (process, job) in list(live.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    reap_group(process)
                    del live[pid]
                    if code:
                        raise RuntimeError(
                            f"evaluator exited {code}: {job[3].with_suffix('.log')}"
                        )
                    finished.append(job)
                if (
                    time.monotonic() - last_status >= 15
                    or len(finished) == expected_count
                ):
                    last_status = time.monotonic()
                    atomic_json(
                        root / "status.json",
                        {
                            "status": "running",
                            "stage": phase,
                            "pid": os.getpid(),
                            "child_pids": list(live),
                            "completed": len(finished),
                            "expected": expected_count,
                        },
                    )
                if live:
                    time.sleep(0.3)
        finally:
            for process, _ in live.values():
                reap_group(process)
        values = {n: {} for n in names}
        configs = {n: {} for n in names}
        for name, spec, st, out, _ in finished:
            report = verified_report(
                out, spec, st["pairs"], st["seed"], models[name], digest, "normal"
            )
            key = spec["id"]
            values[name][key] = clusters(report)
            configs[name][key] = {
                (v["players"], v["target"], v["flor"]): np.asarray(v["pair_scores"])
                for v in report["configurations"]
            }
        result = {
            n: _assessment(
                values[n], values["parent"], configs[n], configs["parent"], specs, 0.02
            )
            for n in names
        }
        atomic_json(root / (phase + "-comparison.json"), result)
        return result

    screen = evaluate(list(models), "screen", False)
    candidate = max(
        (n for n in models if screen[n]["observed_guard_pass"]),
        key=lambda n: rank(screen[n]),
    )
    atomic_json(
        root / "screen-choice.json", {"name": candidate, "frozen_before_recheck": True}
    )
    retained = "parent"
    rechecked = False
    if candidate != "parent":
        result = evaluate(["parent", candidate], "recheck", True)
        rechecked = True
        if promote(
            result[candidate],
            result["parent"],
            {s["id"] for s in config["primary_specs"]},
        ):
            retained = candidate
    if file_hash(parent) != plans["parent_sha256"] or any(
        file_hash(models[n]) != plans["model_sha256"][n] for n in models
    ):
        raise ValueError("calibration checkpoint changed")
    choice = {
        "retained": retained,
        "gain": GAINS[retained],
        "checkpoint": str(models[retained] if retained != "parent" else parent),
        "sha256": plans["model_sha256"][retained],
        "screen_winner": candidate,
        "rechecked": rechecked,
        "authenticated_reports": len(models) * len(config["primary_specs"])
        + (
            2 * (len(config["primary_specs"]) + len(config["diagnostic_specs"]))
            if rechecked
            else 0
        ),
        "elapsed_seconds": time.monotonic() - started,
        "approved": False,
    }
    atomic_json(root / "choice.json", choice)
    atomic_json(
        root / "status.json",
        {
            "status": "complete",
            "pid": os.getpid(),
            "child_pids": [],
            "retained": retained,
        },
    )
    print(json.dumps(choice), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--previous", required=True, type=Path)
    p.add_argument("--parent", required=True, type=Path)
    p.add_argument("--streams", required=True, type=Path)
    a = p.parse_args()
    run(a.output, a.parent, a.previous, json.loads(a.streams.read_text())["streams"])
