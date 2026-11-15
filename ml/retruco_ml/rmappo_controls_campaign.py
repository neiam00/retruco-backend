"""Matched-parent factorial trial of partner choice balance and response anchoring."""

import argparse
import fcntl
import json
import math
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
from .rmappo_controls_compare import ablation_comparison, compare, coverage
from .snapshot import freeze_source, verify_snapshot
from .target_campaign import immutable_inventory
from .target_compare import verified_report

ARMS = {
    "A": {"balance": False, "betting": False},
    "B": {"balance": True, "betting": False},
    "C": {"balance": False, "betting": True},
    "D": {"balance": True, "betting": True},
}


class BudgetExhausted(RuntimeError):
    pass


class Interrupted(RuntimeError):
    pass


def reap_group(process):
    # Clean the owned group even if its leader exited before a descendant.
    def send(sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Darwin may return EPERM while the last terminated group member
            # is being reaped. Only accept that outcome after an authoritative
            # process listing proves there are no live members. A failed ps or
            # a real permission failure against live work remains fatal.
            observed = subprocess.run(
                ["ps", "-axo", "pgid=,stat="], capture_output=True, text=True
            )
            if observed.returncode:
                raise
            for line in observed.stdout.splitlines():
                fields = line.split()
                if len(fields) != 2 or not fields[0].isdigit():
                    raise
                if int(fields[0]) == process.pid and not fields[1].startswith("Z"):
                    raise

    send(signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        send(signal.SIGKILL)
    process.wait(timeout=5)


def check_cap(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExhausted("total campaign wall-clock cap reached")


def arm_deadline(deadline):
    check_cap(deadline)

    def expired(signum, frame):
        raise BudgetExhausted("total campaign wall-clock cap reached")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(1e-6, deadline - time.monotonic()))


def validate(c):
    if (
        c["arms"] != ARMS
        or c["cell"] not in ("gru", "lstm")
        or c["mode_win_rate_targets"] != {"1v1": 0.7, "2v2": 0.6}
    ):
        raise ValueError("invalid controlled comparison")
    if (
        len(set(c["seeds"])) != len(c["seeds"])
        or not c["seeds"]
        or any(type(x) is not int or x < 0 for x in c["seeds"])
    ):
        raise ValueError("invalid seed blocks")
    if set(c["parent_sources"]) != {str(s) for s in c["seeds"]}:
        raise ValueError("missing matched parents")
    if any(e["file"] != f"{s}.pt" for s, e in c["parent_sources"].items()):
        raise ValueError("unsafe parent destination")
    if (
        not math.isfinite(c["max_wall_hours"])
        or not 0 < c["max_wall_hours"] <= 12
        or (not c["smoke"] and len(c["seeds"]) != 3)
    ):
        raise ValueError("three blocks and at most twelve hours required")
    for k in (
        "choices",
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
        raise ValueError("invalid reserve")
    spans = []
    for family in ("primary", "diagnostic", "ablation"):
        s, n = c[family + "_seed"], c[family + "_pairs"]
        if type(n) is not int or n < 2 or not 100000 <= s < s + n < 200000:
            raise ValueError("invalid development streams")
        spans.append(set(range(s, s + n)))
    if any(spans[i] & spans[j] for i in range(3) for j in range(i)):
        raise ValueError("overlapping evaluation deals")
    required = {
        "native-headsup",
        "native-literal-v2",
        "native-trusting-v2",
        "native-cautious-v2",
        "native-reference-human",
    }
    if {s["id"] for s in c["primary_specs"]} != required:
        raise ValueError("incomplete target protocol")
    specs = c["primary_specs"] + c["diagnostic_specs"]
    if len({s["id"] for s in specs}) != len(specs):
        raise ValueError("duplicate report specification")


def campaign_input(repository: Path, value: str) -> Path:
    """Resolve a supplied path or a repository-relative experiment input."""
    path = Path(value)
    if path.is_absolute() or path.is_file():
        return path.resolve()
    return (repository / path).resolve()


def verify_parents(root, c):
    for seed, entry in c["parent_sources"].items():
        p = (root / "parents" / entry["file"]).resolve()
        if (
            not p.is_relative_to((root / "parents").resolve())
            or file_hash(p) != entry["sha256"]
        ):
            raise ValueError("frozen matched parent changed")
        saved = torch.load(p, map_location="cpu", weights_only=True)
        if (
            saved.get("kind") != "rmappo-v1"
            or saved["configuration"]["cell"] != c["cell"]
            or saved["configuration"]["seed"] != entry["original_seed"]
            or saved["choices"] != entry["original_choices"]
        ):
            raise ValueError("wrong parent provenance")


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
            p.error("new output/config/inputs required")
        c = json.loads(a.config.read_text())
        validate(c)
        if (
            not c["smoke"]
            and file_hash(campaign_input(repository, c["curve_report"]))
            != c["curve_report_sha256"]
        ):
            raise ValueError("checkpoint audit changed before campaign freeze")
        preservation = immutable_inventory(repository, root, 2)
        root.mkdir(parents=True)
        atomic_json(root / "preservation.json", preservation)
        if c.get("curve_report"):
            shutil.copy2(
                campaign_input(repository, c["curve_report"]),
                root / "checkpoint-audit.json",
            )
        shutil.copytree(a.inputs.resolve().parent, root / "inputs")
        validate_inputs(root / "inputs")
        (root / "parents").mkdir()
        for entry in c["parent_sources"].values():
            source = campaign_input(repository, entry["origin"])
            if file_hash(source) != entry["sha256"]:
                raise ValueError("selected parent changed before copy")
            shutil.copy2(source, root / "parents" / entry["file"])
        verify_parents(root, c)
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
        raise ValueError("frozen configuration changed")
    if json.loads((root / "status.json").read_text())["status"] != "prepared":
        raise ValueError("one-shot campaign; no automatic restart")
    verify_snapshot(root / "source")
    validate_inputs(root / "inputs")
    verify_parents(root, c)
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
            reap_group(process)
        children.clear()

    def interrupted(signum, frame):
        raise Interrupted(f"supervisor signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def jobs_run(jobs, limit=1, until=None):
        queue = list(jobs)
        until = min(deadline, until or deadline)
        try:
            while queue or children:
                if time.monotonic() >= until:
                    raise BudgetExhausted("bounded campaign allocation exhausted")
                while queue and len(children) < limit:
                    cmd, log = queue.pop(0)
                    log.parent.mkdir(parents=True, exist_ok=True)
                    env = {
                        **os.environ,
                        "PYTHONPATH": str(root / "source/ml"),
                        "PYTHONDONTWRITEBYTECODE": "1",
                        KEY: str(until),
                    }
                    with log.open("ab") as stream:
                        process = subprocess.Popen(
                            cmd,
                            cwd=root / "source/ml",
                            env=env,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    children[process.pid] = (process, {"command": cmd, "log": str(log)})
                    status()
                for pid, (process, item) in list(children.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    reap_group(process)
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

    def evaluations(models, ablation=False):
        jobs = []
        expected = []
        for name, cp in models.items():
            for spec in (
                [c["primary_specs"][1]]
                if ablation
                else c["primary_specs"] + c["diagnostic_specs"]
            ):
                family = (
                    "ablation"
                    if ablation
                    else "primary"
                    if spec in c["primary_specs"]
                    else "diagnostic"
                )
                for mode in (
                    ("normal", "no-send", "no-receive", "no-partner-delivery")
                    if ablation
                    else ("normal",)
                ):
                    out = (
                        root
                        / ("ablation" if ablation else "evaluation")
                        / name
                        / (spec["id"] + ("-" + mode if ablation else "") + ".json")
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
                        str(c[family + "_pairs"]),
                        "--seed",
                        str(c[family + "_seed"]),
                        "--signals",
                        mode,
                    ]
                    jobs.append((cmd, out.with_suffix(".log")))
                    expected.append((out, spec, cp, family, mode))
        jobs_run(jobs, c["evaluation_workers"])
        for out, spec, cp, family, mode in expected:
            verified_report(
                out,
                spec,
                c[family + "_pairs"],
                c[family + "_seed"],
                cp,
                frozen["inputs_sha256"],
                mode,
            )

    try:
        arm_deadline(deadline)
        models = {"reference": root / "inputs/legacy-champion-lstm16.pt"}
        for seed in c["seeds"]:
            models[f"parent-{seed}"] = (
                root / "parents" / c["parent_sources"][str(seed)]["file"]
            )
        stage = "training"
        completed = 0
        arms = list(c["arms"])
        for block, seed in enumerate(c["seeds"]):
            parent = models[f"parent-{seed}"]
            for arm in arms[block % 4 :] + arms[: block % 4]:
                treatment = c["arms"][arm]
                name = f"{arm}-{seed}"
                trial = root / "trials" / name
                context = {
                    "arm": arm,
                    "seed": seed,
                    "target_choices": c["choices"],
                    "completed_training_runs": completed,
                }
                cmd = [
                    sys.executable,
                    "-u",
                    "-m",
                    "retruco_ml.rmappo_control_train",
                    "--output",
                    str(trial),
                    "--inputs",
                    str(root / "inputs/manifest.json"),
                    "--simulator",
                    str(root / "source/bin/selfplay"),
                    "--cell",
                    c["cell"],
                    "--seed",
                    str(seed),
                    "--arm",
                    arm,
                    "--choices",
                    str(c["choices"]),
                    "--parent-checkpoint",
                    str(parent),
                    "--teacher",
                    str(models["reference"]),
                    "--response-anchor",
                    str(0.1 if treatment["betting"] else 0.0),
                ]
                if treatment["balance"]:
                    cmd.append("--partner-balance")
                for key in (
                    "steps",
                    "environments",
                    "workers",
                    "batch_size",
                    "warmups",
                    "warmup_choices",
                ):
                    cmd += ["--" + key.replace("_", "-"), str(c[key])]
                jobs_run(
                    [(cmd, root / "logs" / f"{name}.log")],
                    until=deadline - c["evaluation_reserve_hours"] * 3600,
                )
                initial = torch.load(
                    trial / "initial.pt", map_location="cpu", weights_only=True
                )
                origin = torch.load(parent, map_location="cpu", weights_only=True)
                for field in ("actor", "critic"):
                    if initial[field].keys() != origin[field].keys() or any(
                        not torch.equal(v, origin[field][k])
                        for k, v in initial[field].items()
                    ):
                        raise ValueError(
                            "initial actor/critic differs from matched parent"
                        )
                if (
                    initial["choices"]
                    or initial["updates"]
                    or initial["actor_optimizer"]["state"]
                    or initial["critic_optimizer"]["state"]
                ):
                    raise ValueError(
                        "continuation must use fresh optimizer and additional-choice clock"
                    )
                endpoint = torch.load(
                    trial / "checkpoint.pt", map_location="cpu", weights_only=True
                )
                if endpoint["choices"] < c["choices"]:
                    raise ValueError("incomplete trial endpoint")
                cp = trial / "candidate.pt"
                shutil.copy2(trial / "checkpoint.pt", cp)
                cp.with_suffix(".sha256").write_text(file_hash(cp) + "\n")
                models[name] = cp
                completed += 1
        context = {"completed_training_runs": completed}
        stage = "paired-evaluation"
        status()
        evaluations(models)
        check_cap(deadline)
        result = compare(root, c)
        check_cap(deadline)
        atomic_json(root / "comparison-primary.json", result)
        arm = result["exploratory_arm"]
        atomic_json(
            root / "ablation-choice.json",
            {
                "arm": arm,
                "scope": "selected exploratorily on primary development results",
                "approved": False,
            },
        )
        chosen = {
            name: cp
            for name, cp in models.items()
            if name == "reference"
            or name.startswith("parent-")
            or name.startswith(arm + "-")
        }
        stage = "signal-ablation"
        status()
        evaluations(chosen, True)
        check_cap(deadline)
        result["signal_ablations"] = ablation_comparison(root, c, arm)
        check_cap(deadline)
        result["coverage"] = coverage(root, c, ablation_arm=arm)
        if not result["coverage"]["complete"]:
            raise ValueError("incomplete authenticated coverage")
        result["preservation"] = verify_preservation(
            json.loads((root / "preservation.json").read_text())
        )
        check_cap(deadline)
        if not result["preservation"]["passed"]:
            raise ValueError("preserved artifact mismatch")
        result["ablation_models"] = list(chosen)
        atomic_json(root / "comparison.json", result)
        stage = "complete"
        status("complete", comparison=str(root / "comparison.json"))
    except (BudgetExhausted, Interrupted) as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminate()
        state = (
            "budget-exhausted" if isinstance(error, BudgetExhausted) else "interrupted"
        )
        atomic_json(
            root / "partial-results.json",
            {
                "status": state,
                "error": str(error),
                "unfinished_coverage": True,
                "approved": False,
            },
        )
        status(state, error=str(error))
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminate()
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
