"""Bounded fresh RMAPPO comparison with immutable inputs and equal budgets."""

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

from .controlled_train import atomic_json
from .cooperation_refusal_campaign import validate_inputs, verify_preservation
from .improvement_budget import KEY
from .recurrent import file_hash
from .rmappo_compare import choose, compare
from .rmappo_coverage import coverage
from .snapshot import freeze_source, verify_snapshot
from .target_campaign import immutable_inventory
from .target_compare import verified_report


class BudgetExhausted(RuntimeError):
    pass


class Interrupted(RuntimeError):
    pass


def validate(c):
    if c["cells"] != ["gru", "lstm"] or c["mode_win_rate_targets"] != {
        "1v1": 0.7,
        "2v2": 0.6,
    }:
        raise ValueError("invalid cell/target comparison")
    if not c["smoke"] and (len(c["seeds"]) != 3 or c["max_wall_hours"] > 12):
        raise ValueError("three seeds and at most twelve hours required")
    if len(set(c["seeds"])) != len(c["seeds"]) or not c["seeds"]:
        raise ValueError("invalid seeds")
    if c["rounds"] != sorted(set(c["rounds"])) or min(c["rounds"]) < 1:
        raise ValueError("invalid equal-budget rounds")
    for k in (
        "steps",
        "environments",
        "workers",
        "batch_size",
        "warmups",
        "warmup_choices",
        "evaluation_workers",
    ):
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError("invalid capacity " + k)
    if not 0 < c["evaluation_reserve_hours"] < c["max_wall_hours"]:
        raise ValueError("invalid evaluation reserve")
    spans = []
    for phase in ("selection", "final", "ablation"):
        n, s = c[phase + "_pairs"], c[phase + "_seed"]
        if n < 2 or not 100000 <= s < s + n < 200000:
            raise ValueError("invalid development deals")
        spans.append(set(range(s, s + n)))
    if any(spans[i] & spans[j] for i in range(3) for j in range(i)):
        raise ValueError("overlapping development deals")
    specs = c["primary_specs"] + c["diagnostic_specs"]
    if len({s["id"] for s in specs}) != len(specs):
        raise ValueError("duplicate evaluation specs")
    if {s["id"] for s in c["primary_specs"]} != {
        "native-headsup",
        "native-literal-v2",
        "native-trusting-v2",
        "native-cautious-v2",
        "native-reference-human",
    }:
        raise ValueError("native coverage missing")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--run", action="store_true")
    a = p.parse_args()
    root = a.output.resolve()
    repository = Path(__file__).resolve().parents[2]
    if not a.run:
        if root.exists() or not a.config or not a.inputs:
            p.error("prepare needs new output/config/inputs")
        c = json.loads(a.config.read_text())
        validate(c)
        preservation = immutable_inventory(repository, root, 2)
        root.mkdir(parents=True)
        atomic_json(root / "preservation.json", preservation)
        shutil.copytree(a.inputs.resolve().parent, root / "inputs")
        validate_inputs(root / "inputs")
        atomic_json(root / "configuration.json", c)
        freeze_source(root / "source", repository, allow_dirty=c["smoke"])
        atomic_json(
            root / "frozen.json",
            {
                "configuration_sha256": file_hash(root / "configuration.json"),
                "inputs_sha256": file_hash(root / "inputs/manifest.json"),
            },
        )
        atomic_json(
            root / "status.json",
            {"status": "prepared", "pid": None, "child_pids": [], "work_items": []},
        )
        print(
            json.dumps({"prepared": str(root), "preserved": len(preservation["files"])})
        )
        return
    c = json.loads((root / "configuration.json").read_text())
    validate(c)
    frozen = json.loads((root / "frozen.json").read_text())
    if frozen != {
        "configuration_sha256": file_hash(root / "configuration.json"),
        "inputs_sha256": file_hash(root / "inputs/manifest.json"),
    }:
        raise ValueError("frozen contract changed")
    if json.loads((root / "status.json").read_text())["status"] != "prepared":
        raise ValueError("one-shot campaign; no automatic restart")
    verify_snapshot(root / "source")
    validate_inputs(root / "inputs")
    lock = (root / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    deadline = started + c["max_wall_hours"] * 3600
    children = {}
    stage = "starting"
    context = {}
    last_status = 0.0

    def status(state="running", **extra):
        nonlocal last_status
        last_status = time.monotonic()
        atomic_json(
            root / "status.json",
            {
                "status": state,
                "pid": os.getpid(),
                "stage": stage,
                "campaign_seconds": last_status - started,
                "remaining_seconds": max(0, deadline - last_status),
                "child_pids": list(children),
                "work_items": [x[1] for x in children.values()],
                **context,
                **extra,
            },
        )

    def terminate():
        for process, _ in children.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process, _ in children.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        children.clear()

    def interrupted(signum, frame):
        raise Interrupted(f"supervisor signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def run_jobs(jobs, limit=1, until=None):
        queue = list(jobs)
        until = deadline if until is None else min(until, deadline)
        try:
            while queue or children:
                if time.monotonic() >= until:
                    raise BudgetExhausted("campaign allocation exhausted")
                while queue and len(children) < limit:
                    cmd, log = queue.pop(0)
                    log.parent.mkdir(parents=True, exist_ok=True)
                    env = {
                        **os.environ,
                        "PYTHONPATH": str(root / "source/ml"),
                        KEY: str(until),
                    }
                    with log.open("ab") as stream:
                        process = subprocess.Popen(
                            cmd,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            env=env,
                            cwd=root / "source/ml",
                            start_new_session=True,
                        )
                    children[process.pid] = (process, {"command": cmd, "log": str(log)})
                    status()
                for pid, (process, item) in list(children.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    del children[pid]
                    if code == 75:
                        raise BudgetExhausted(
                            "child allocation exhausted: " + item["log"]
                        )
                    if code:
                        raise RuntimeError(f"child exited {code}: {item['log']}")
                if time.monotonic() - last_status >= 30:
                    status()
                if children:
                    time.sleep(0.5)
        finally:
            if children:
                terminate()

    def evaluations(phase, models, signals=("normal",)):
        jobs = []
        expected = []
        specs = (
            c["primary_specs"] + c["diagnostic_specs"]
            if phase != "ablation"
            else [c["primary_specs"][1]]
        )
        for name, cp in models.items():
            for spec in specs:
                for mode in signals:
                    out = (
                        root
                        / phase
                        / name
                        / (
                            spec["id"]
                            + ("" if phase != "ablation" else "-" + mode)
                            + ".json"
                        )
                    )
                    cmd = [
                        sys.executable,
                        "-u",
                        "-m",
                        "retruco_ml.rmappo_evaluate",
                        str(cp),
                        "--inputs",
                        str(root / "inputs/manifest.json"),
                        "--simulator",
                        str(root / "source/bin/selfplay"),
                        "--output",
                        str(out),
                        "--partner",
                        spec["partner"],
                        "--opponent",
                        spec["opponent"],
                        "--players",
                        str(spec["players"]),
                        "--pairs",
                        str(c[phase + "_pairs"]),
                        "--seed",
                        str(c[phase + "_seed"]),
                        "--signals",
                        mode,
                    ]
                    jobs.append((cmd, out.with_suffix(".log")))
                    expected.append((out, spec, cp, mode))
        run_jobs(jobs, c["evaluation_workers"])
        for out, spec, cp, mode in expected:
            verified_report(
                out,
                spec,
                c[phase + "_pairs"],
                c[phase + "_seed"],
                cp,
                frozen["inputs_sha256"],
                mode,
            )

    try:
        stage = "training"
        completed = []
        previous = 0
        training_start = time.monotonic()
        training_deadline = deadline - c["evaluation_reserve_hours"] * 3600
        for budget in c["rounds"]:
            if previous:
                rate = (time.monotonic() - training_start) / (
                    previous * len(c["seeds"]) * len(c["cells"])
                )
                forecast = (
                    rate * (budget - previous) * len(c["seeds"]) * len(c["cells"]) * 1.3
                )
                if time.monotonic() + forecast >= training_deadline:
                    break
            finished = True
            for seed in c["seeds"]:
                for cell in c["cells"]:
                    name = f"{cell}-{seed}"
                    trial = root / "trials" / name
                    context = {
                        "cell": cell,
                        "seed": seed,
                        "target_choices": budget,
                        "completed_rounds": completed,
                    }
                    cmd = [
                        sys.executable,
                        "-u",
                        "-m",
                        "retruco_ml.rmappo_train",
                        "--output",
                        str(trial),
                        "--inputs",
                        str(root / "inputs/manifest.json"),
                        "--simulator",
                        str(root / "source/bin/selfplay"),
                        "--cell",
                        cell,
                        "--seed",
                        str(seed),
                        "--choices",
                        str(budget),
                    ]
                    for key in (
                        "steps",
                        "environments",
                        "workers",
                        "batch_size",
                        "warmups",
                        "warmup_choices",
                    ):
                        cmd += ["--" + key.replace("_", "-"), str(c[key])]
                    if trial.exists():
                        cmd.append("--resume")
                    try:
                        run_jobs(
                            [(cmd, root / "logs" / f"{name}-{budget}.log")],
                            until=training_deadline,
                        )
                    except BudgetExhausted:
                        finished = False
                        break
                    saved = torch.load(
                        trial / "checkpoint.pt", map_location="cpu", weights_only=True
                    )
                    if (
                        saved["choices"] < budget
                        or saved["configuration"]["initialization"]
                        != "random; no warm start"
                    ):
                        raise ValueError("incomplete or warm-started endpoint")
                    endpoint = trial / "endpoints" / f"choices-{budget}.pt"
                    endpoint.parent.mkdir(exist_ok=True)
                    shutil.copy2(trial / "checkpoint.pt", endpoint)
                if not finished:
                    break
            if not finished:
                break
            previous = budget
            completed.append(budget)
            atomic_json(
                root / "training-plan.json",
                {
                    "completed_equal_rounds": completed,
                    "common_budget": previous,
                    "planned_rounds": c["rounds"],
                    "forecast_only": True,
                },
            )
        if not completed:
            raise BudgetExhausted(
                "no complete equal-seed round; partial checkpoints preserved"
            )
        atomic_json(
            root / "training-plan.json",
            {
                "completed_equal_rounds": completed,
                "common_budget": previous,
                "planned_rounds": c["rounds"],
                "skipped_rounds": [b for b in c["rounds"] if b not in completed],
                "forecast_only": True,
            },
        )
        models = {"reference": root / "inputs/legacy-champion-lstm16.pt"}
        for cell in c["cells"]:
            for seed in c["seeds"]:
                name = f"{cell}-{seed}"
                trial = root / "trials" / name
                cp = trial / "candidate.pt"
                shutil.copy2(trial / "endpoints" / f"choices-{previous}.pt", cp)
                cp.with_suffix(".sha256").write_text(file_hash(cp) + "\n")
                models[name] = cp
        context = {"common_budget": previous}
        stage = "selection-validation"
        status()
        evaluations("selection", models)
        selected = compare(root, c, "selection")
        atomic_json(root / "comparison-selection.json", selected)
        nominee = choose(selected)
        nominee["checkpoint_sha256"] = file_hash(models[nominee["exploratory_nominee"]])
        atomic_json(root / "nominee.json", nominee)
        stage = "independent-development-confirmation"
        status()
        evaluations("final", models)
        result = compare(root, c, "final")
        result["selection"] = nominee
        name = nominee["exploratory_nominee"]
        safe = result["individuals"][name]["observed_guard_pass"]
        result["retained_checkpoint"] = str(
            models[name]
            if safe and not nominee["starter_fallback"]
            else models["reference"]
        )
        result["retained_is_trained"] = safe and not nominee["starter_fallback"]
        stage = "signal-ablation"
        status()
        cell = name.split("-")[0]
        ablation_models = {
            n: cp
            for n, cp in models.items()
            if n == "reference" or n.startswith(cell + "-")
        }
        evaluations(
            "ablation",
            ablation_models,
            ("normal", "no-send", "no-receive", "no-partner-delivery"),
        )
        result["ablation_models"] = list(ablation_models)
        result["coverage"] = coverage(root, c)
        if not result["coverage"]["complete"]:
            raise ValueError("incomplete authenticated evaluation coverage")
        result["preservation"] = verify_preservation(
            json.loads((root / "preservation.json").read_text())
        )
        if not result["preservation"]["passed"]:
            raise ValueError("preserved artifact mismatch")
        atomic_json(root / "comparison.json", result)
        stage = "complete"
        status("complete", comparison=str(root / "comparison.json"))
    except (BudgetExhausted, Interrupted) as error:
        terminate()
        state = (
            "budget-exhausted" if isinstance(error, BudgetExhausted) else "interrupted"
        )
        reports = [
            str(p.relative_to(root))
            for p in root.rglob("*.json")
            if p.with_suffix(".sha256").exists()
        ]
        atomic_json(
            root / "partial-results.json",
            {
                "status": state,
                "error": str(error),
                "completed_reports": reports,
                "unfinished_coverage": True,
                "approved": False,
            },
        )
        status(state, error=str(error))
    except BaseException as error:
        terminate()
        status("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
