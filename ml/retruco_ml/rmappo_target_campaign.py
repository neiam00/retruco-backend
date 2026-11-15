"""Bounded matched-parent continuation with untouched final confirmation."""

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
from .rmappo_controls_campaign import (
    BudgetExhausted,
    Interrupted,
    arm_deadline,
    campaign_input,
    check_cap,
    reap_group,
    verify_parents,
)
from .rmappo_headsup_protection import (
    CALIBRATED_STARTER_SHA256,
    verify_protected_weights,
)
from .rmappo_population import population_config
from .rmappo_target_compare import (
    coverage,
    expected_reports,
    select_candidates,
    summarize,
)
from .snapshot import freeze_source, verify_snapshot
from .target_campaign import immutable_inventory
from .target_compare import verified_report

PROFILES = ("original", "target")
STUDY_PROFILES = {
    1: PROFILES,
    2: ("gae99", "mc"),
    3: ("control", "targeted"),
    4: ("protected", "focused"),
}


def training_population(version, profile):
    if version == 4:
        return {"protected": "target", "focused": "reference_focus"}[profile]
    return "target" if version >= 2 else profile


def training_command(root, config, profile, seed, budget, resume=False):
    trial = root / "trials" / f"{profile}-{seed}"
    parent = root / "parents" / config["parent_sources"][str(seed)]["file"]
    version = config["version"]
    command = [
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
        config["cell"],
        "--seed",
        str(seed),
        "--arm",
        "A",
        "--population-profile",
        training_population(version, profile),
        "--save-every",
        str(config["save_every"]),
        "--choices",
        str(budget),
        "--parent-checkpoint",
        str(parent),
        "--teacher",
        str(root / "inputs/legacy-champion-lstm16.pt"),
        "--response-anchor",
        "0",
    ]
    for key in (
        "steps",
        "environments",
        "workers",
        "batch_size",
        "warmups",
        "warmup_choices",
    ):
        command.extend(("--" + key.replace("_", "-"), str(config[key])))
    if version >= 2:
        command.extend(
            (
                "--gae-lambda",
                str(1.0 if version == 2 and profile == "mc" else 0.99),
                "--response-kl-stop",
                str(config["response_kl_stop"]),
                "--response-kl-rollback",
                str(config["response_kl_rollback"]),
            )
        )
    if version == 3:
        command.extend(
            (
                "--candidate-features",
                str(int(profile == "targeted")),
                "--semantic-exploration",
                str(config["semantic_exploration"] if profile == "targeted" else 0.0),
                "--semantic-anneal-choices",
                str(config["semantic_anneal_choices"]),
            )
        )
    if version == 4:
        command.append("--protect-headsup")
    if resume:
        command.append("--resume")
    return command


def validate(c):
    if (
        c.get("version") not in STUDY_PROFILES
        or c.get("cell") != "lstm"
        or tuple(c.get("profiles", ())) != STUDY_PROFILES[c["version"]]
        or c.get("mode_win_rate_targets") != {"1v1": 0.7, "2v2": 0.6}
    ):
        raise ValueError("invalid target continuation protocol")
    if c["version"] >= 2 and (
        c.get("response_kl_stop") != 0.03 or c.get("response_kl_rollback") != 0.05
    ):
        raise ValueError("retraining requires the frozen response guard")
    if c["version"] == 3 and (
        c.get("candidate_features") != 1
        or c.get("semantic_exploration") != 0.02
        or c.get("semantic_anneal_choices") != 1000000
    ):
        raise ValueError("invalid frozen contextual exploration bundle")
    if c["version"] in (3, 4) and (
        not isinstance(c.get("prior_nominee"), dict)
        or not isinstance(c["prior_nominee"].get("origin"), str)
        or len(c["prior_nominee"].get("sha256", "")) != 64
    ):
        raise ValueError("prior nominee must remain a registered baseline")
    seeds = c.get("seeds", [])
    smoke = c.get("smoke", False)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or len(seeds) != (1 if smoke else 3)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("one smoke block or three production blocks required")
    if set(c["parent_sources"]) != {str(s) for s in seeds}:
        raise ValueError("missing matched parents")
    if any(e["file"] != f"{s}.pt" for s, e in c["parent_sources"].items()):
        raise ValueError("unsafe parent destination")
    budgets = c.get("budgets", [])
    production_budgets = {3: [500000, 1000000, 1500000, 2000000], 4: [1000000]}
    if (
        not budgets
        or any(type(n) is not int or n < 1 for n in budgets)
        or budgets != sorted(set(budgets))
        or budgets[-1] > 2000000
        or (
            not smoke
            and budgets
            != production_budgets.get(c["version"], [500000, 1000000, 2000000])
        )
    ):
        raise ValueError("invalid common continuation milestones")
    for key in (
        "save_every",
        "steps",
        "environments",
        "workers",
        "batch_size",
        "warmups",
        "warmup_choices",
        "evaluation_workers",
    ):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError("invalid capacity " + key)
    if not smoke and c["save_every"] != 250000:
        raise ValueError("save every 250000 choices")
    if c["evaluation_workers"] > 4:
        raise ValueError("at most four concurrent evaluators")
    if (
        not math.isfinite(c["max_wall_hours"])
        or not 0 < c["max_wall_hours"] <= 12
        or not math.isfinite(c["evaluation_reserve_hours"])
        or not 0 < c["evaluation_reserve_hours"] < c["max_wall_hours"]
    ):
        raise ValueError("invalid campaign cap or evaluation reserve")
    if not smoke and c["evaluation_reserve_hours"] < 4:
        raise ValueError("reserve at least four hours")
    carried = c.get("carried_seconds", 0.0)
    if (
        not isinstance(carried, (float, int))
        or not math.isfinite(carried)
        or not 0 <= carried < c["max_wall_hours"] * 3600
    ):
        raise ValueError("invalid carried screening runtime")
    if (
        c["version"] >= 2
        and len({e["sha256"] for e in c["parent_sources"].values()}) != 1
    ):
        raise ValueError("retraining seeds must share the screened parent")
    spans = []
    for family in ("selection", "selection_diagnostic", "final", "final_diagnostic"):
        seed, pairs = c[family + "_seed"], c[family + "_pairs"]
        if (
            type(seed) is not int
            or type(pairs) is not int
            or pairs < 2
            or not 100000 <= seed < seed + pairs < 200000
        ):
            raise ValueError("invalid evaluation stream " + family)
        spans.append((seed, seed + pairs))
    if any(
        max(a[0], b[0]) < min(a[1], b[1])
        for i, a in enumerate(spans)
        for b in spans[:i]
    ):
        raise ValueError("overlapping evaluation deal streams")
    if not smoke and c["final_pairs"] < 2000:
        raise ValueError("confirmation requires 2000 paired deals")
    if c["version"] == 4:
        if any(
            entry["sha256"] != CALIBRATED_STARTER_SHA256
            for entry in c["parent_sources"].values()
        ):
            raise ValueError(
                "heads-up protection requires the registered calibrated starter"
            )
        expected = {
            name: population_config(training_population(4, name))
            for name in c["profiles"]
        }
        if (
            c.get("headsup_protection") != 1
            or c.get("players") != 4
            or c.get("populations") != expected
        ):
            raise ValueError("protected team population or mode contract changed")
        if not smoke and (
            c["selection_pairs"] < 1024
            or c["selection_diagnostic_pairs"] < 64
            or c["evaluation_reserve_hours"] < 6
        ):
            raise ValueError(
                "protected selection requires 1024 paired deals, 64 diagnostics and a six-hour reserve"
            )
    required = {
        "native-headsup",
        "native-literal-v2",
        "native-trusting-v2",
        "native-cautious-v2",
        "native-reference-human",
    }
    specs = c["primary_specs"] + c["diagnostic_specs"]
    if (
        {s["id"] for s in c["primary_specs"]} != required
        or len(c["primary_specs"]) != 5
        or len({s["id"] for s in specs}) != len(specs)
    ):
        raise ValueError("incomplete or duplicate target coverage")
    for spec in c["primary_specs"]:
        headsup = spec["id"] == "native-headsup"
        partner = "reference-cpu" if headsup else spec["id"][7:]
        if (spec["opponent"], spec["partner"], spec["players"]) != (
            "native",
            partner,
            2 if headsup else 4,
        ):
            raise ValueError("native target specification changed")


def main_training_rate(trial):
    rows = [
        json.loads(line) for line in (trial / "metrics.jsonl").read_text().splitlines()
    ]
    rows = [r for r in rows if r["stage"] == "training"]
    choices = sum(r["choices_this_update"] for r in rows)
    seconds = math.fsum(r["elapsed_seconds"] for r in rows)
    if choices <= 0 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("missing training throughput")
    return seconds / choices


def distinct_evaluations(expected):
    """Identical weights on identical deals are one observation, not new data."""
    unique = []
    aliases = []
    seen = {}
    hashes = {}
    for item in expected:
        checkpoint = item["checkpoint"]
        if checkpoint not in hashes:
            hashes[checkpoint] = file_hash(checkpoint)
        key = (
            hashes[checkpoint],
            item["pairs"],
            item["seed"],
            json.dumps(item["spec"], sort_keys=True),
        )
        if key in seen:
            aliases.append((item, seen[key]))
        else:
            seen[key] = item
            unique.append(item)
    return unique, aliases


def verify_initial(trial, parent):
    initial = torch.load(trial / "initial.pt", map_location="cpu", weights_only=True)
    origin = torch.load(parent, map_location="cpu", weights_only=True)
    if (
        initial["configuration"].get("candidate_features", 0) == 1
        and origin["configuration"].get("candidate_features", 0) == 0
    ):
        origin["actor"]["action.0.weight"] = origin["actor"]["action.0.weight"].clone()
        origin["actor"]["action.0.weight"][:, 36:] = 0
    for field in ("actor", "critic"):
        if initial[field].keys() != origin[field].keys() or any(
            not torch.equal(v, origin[field][k]) for k, v in initial[field].items()
        ):
            raise ValueError("initial actor/critic differs from matched parent")
    if (
        initial["choices"]
        or initial["updates"]
        or initial["actor_optimizer"]["state"]
        or initial["critic_optimizer"]["state"]
    ):
        raise ValueError("continuation requires fresh optimizers and choice clock")
    if len(initial["league"]) != len(origin["league"]):
        raise ValueError("initial historical league differs from parent")
    for a, b in zip(initial["league"], origin["league"], strict=True):
        if (
            a["configuration"]["cell"] != b["configuration"]["cell"]
            or a["configuration"].get("candidate_features", 0)
            != b["configuration"].get("candidate_features", 0)
            or a["actor"].keys() != b["actor"].keys()
            or any(not torch.equal(v, b["actor"][k]) for k, v in a["actor"].items())
        ):
            raise ValueError("initial historical league differs from parent")


def verify_paired_rng(first, second):
    a, b = (
        torch.load(p / "initial.pt", map_location="cpu", weights_only=True)
        for p in (first, second)
    )
    if (
        a["python_rng"] != b["python_rng"]
        or a["optimizer_rng"] != b["optimizer_rng"]
        or not torch.equal(a["torch_rng"], b["torch_rng"])
    ):
        raise ValueError("matched seed initial RNG differs")


def verify_endpoint(root, trial, c, profile, seed, budget):
    checkpoint = trial / "endpoints" / f"choices-{budget}.pt"
    if (
        not checkpoint.is_file()
        or not checkpoint.with_suffix(".sha256").is_file()
        or checkpoint.with_suffix(".sha256").read_text().strip()
        != file_hash(checkpoint)
    ):
        raise ValueError("missing or changed immutable milestone")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    settings = saved["configuration"]
    if (
        saved.get("kind") != "rmappo-v1"
        or saved.get("endpoint_nominal_choices") != budget
        or saved["choices"] < budget
        or settings.get("seed") != seed
        or settings.get("cell") != c["cell"]
        or settings.get("population_profile")
        != training_population(c["version"], profile)
        or settings.get("partner_balance") is not False
        or settings.get("response_anchor") != 0.0
        or settings.get("parent_sha256") != c["parent_sources"][str(seed)]["sha256"]
        or settings.get("inputs_sha256") != file_hash(root / "inputs/manifest.json")
    ):
        raise ValueError("milestone provenance or treatment mismatch")
    if c["version"] >= 2 and (
        settings.get("gae_lambda")
        != (1.0 if c["version"] == 2 and profile == "mc" else 0.99)
        or settings.get("response_kl_stop") != c["response_kl_stop"]
        or settings.get("response_kl_rollback") != c["response_kl_rollback"]
    ):
        raise ValueError("retraining treatment mismatch")
    if c["version"] == 3 and (
        settings.get("candidate_features") != int(profile == "targeted")
        or settings.get("semantic_exploration")
        != (c["semantic_exploration"] if profile == "targeted" else 0.0)
        or settings.get("semantic_anneal_choices") != c["semantic_anneal_choices"]
        or settings.get("simulator_sha256") != file_hash(root / "source/bin/selfplay")
    ):
        raise ValueError("targeted continuation treatment mismatch")
    if c["version"] == 4:
        if (
            settings.get("headsup_protection") != 1
            or settings.get("players") != 4
            or settings.get("headsup_fraction") != 0
            or settings.get("candidate_features") != 0
            or settings.get("semantic_exploration") != 0
            or settings.get("population") != c["populations"][profile]
            or settings.get("simulator_sha256")
            != file_hash(root / "source/bin/selfplay")
        ):
            raise ValueError("protected continuation treatment mismatch")
        parent_path = root / "parents" / c["parent_sources"][str(seed)]["file"]
        if file_hash(parent_path) != CALIBRATED_STARTER_SHA256:
            raise ValueError("calibrated starter changed before endpoint verification")
        parent = torch.load(parent_path, map_location="cpu", weights_only=True)
        verify_protected_weights(parent["actor"], saved["actor"])
    return checkpoint


def record_frozen_nominee(result, chosen, config):
    final_row = result["individuals"][chosen["nominee_final_name"]]
    if final_row["checkpoint_sha256"] != chosen["nominee_checkpoint_sha256"]:
        raise ValueError("final nominee does not match frozen selection")
    result["frozen_nominee_result"] = {
        "name": chosen["nominee"],
        "final_name": chosen["nominee_final_name"],
        "checkpoint": final_row["checkpoint"],
        "checkpoint_sha256": final_row["checkpoint_sha256"],
        "starter_fallback": chosen["nominee_is_starter_fallback"],
        "modes": final_row["modes"],
        "acceptable_strength": final_row["acceptable_strength"],
        "approved": False,
    }
    if config["version"] == 4:
        admitted = (
            not chosen["nominee_is_starter_fallback"]
            and final_row["protected_gate"]["qualified"]
        )
        starter = result["individuals"][f"parent-{config['seeds'][0]}"]
        retained = final_row if admitted else starter
        result["frozen_nominee_result"]["protected_gate"] = final_row["protected_gate"]
        result["retained_for_review"] = {
            "checkpoint": retained["checkpoint"],
            "checkpoint_sha256": retained["checkpoint_sha256"],
            "starter_fallback": not admitted,
            "approved": False,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    repository = Path(__file__).resolve().parents[2]
    if not args.run:
        if root.exists() or not args.config or not args.inputs:
            parser.error("new output/config/inputs required")
        c = json.loads(args.config.read_text())
        validate(c)
        preservation = immutable_inventory(repository, root, 2)
        root.mkdir(parents=True)
        atomic_json(root / "preservation.json", preservation)
        shutil.copytree(args.inputs.resolve().parent, root / "inputs")
        validate_inputs(root / "inputs")
        (root / "parents").mkdir()
        for entry in c["parent_sources"].values():
            source = campaign_input(repository, entry["origin"])
            if file_hash(source) != entry["sha256"]:
                raise ValueError("selected parent changed before copy")
            shutil.copy2(source, root / "parents" / entry["file"])
        if c["version"] in (3, 4):
            prior = campaign_input(repository, c["prior_nominee"]["origin"])
            if file_hash(prior) != c["prior_nominee"]["sha256"]:
                raise ValueError("prior nominee changed before copy")
            shutil.copy2(prior, root / "parents/prior-nominee.pt")
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
    started = time.monotonic() - c.get("carried_seconds", 0.0)
    profiles = tuple(c["profiles"])
    deadline = started + c["max_wall_hours"] * 3600
    training_deadline = deadline - c["evaluation_reserve_hours"] * 3600
    frozen = json.loads((root / "frozen.json").read_text())
    if frozen != {
        "configuration_sha256": file_hash(root / "configuration.json"),
        "inputs_sha256": file_hash(root / "inputs/manifest.json"),
    }:
        raise ValueError("frozen configuration changed")
    lock = (root / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if json.loads((root / "status.json").read_text())["status"] != "prepared":
        raise ValueError("one-shot campaign; no automatic restart")
    children = {}
    stage = "verifying"
    context = {}
    last_status = 0.0
    plan = {
        "requested_budgets": c["budgets"],
        "completed_common_budgets": [],
        "completed_trial_budgets": {},
        "rounds": [],
        "forecast_multiplier": 1.35,
        "evaluation_reserve_hours": c["evaluation_reserve_hours"],
    }

    def save_plan():
        atomic_json(root / "training-plan.json", plan)

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
                "completed_common_budgets": plan["completed_common_budgets"],
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
                    command, log = queue.pop(0)
                    log.parent.mkdir(parents=True, exist_ok=True)
                    env = {
                        **os.environ,
                        "PYTHONPATH": str(root / "source/ml"),
                        "PYTHONDONTWRITEBYTECODE": "1",
                        KEY: str(until),
                    }
                    with log.open("ab") as stream:
                        process = subprocess.Popen(
                            command,
                            cwd=root / "source/ml",
                            env=env,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    children[process.pid] = (
                        process,
                        {"command": command, "log": str(log)},
                    )
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

    def evaluations(models, phase):
        expected = expected_reports(root, c, models, phase)
        jobs = []
        unique, aliases = distinct_evaluations(expected)
        for item in unique:
            spec, output = item["spec"], item["path"]
            command = [
                sys.executable,
                "-u",
                "-m",
                "retruco_ml.rmappo_evaluate",
                str(item["checkpoint"]),
                "--inputs",
                str(root / "inputs/manifest.json"),
                "--simulator",
                str(root / "source/bin/selfplay"),
                "--output",
                str(output),
                "--partner",
                spec["partner"],
                "--opponent",
                spec["opponent"],
                "--players",
                str(spec["players"]),
                "--pairs",
                str(item["pairs"]),
                "--seed",
                str(item["seed"]),
                "--signals",
                "normal",
            ]
            jobs.append((command, output.with_suffix(".log")))
        jobs_run(jobs, c["evaluation_workers"])
        for item, source in aliases:
            verified_report(
                source["path"],
                item["spec"],
                item["pairs"],
                item["seed"],
                item["checkpoint"],
                frozen["inputs_sha256"],
                "normal",
            )
            output = item["path"]
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as stream:
                stream.write(source["path"].read_bytes())
            with output.with_suffix(".sha256").open("x") as stream:
                stream.write(file_hash(output) + "\n")
        atomic_json(
            root / (phase + "-report-reuse.json"),
            {
                "unique_evaluations": len(unique),
                "expected_report_paths": len(expected),
                "aliases": [
                    {
                        "path": str(a["path"]),
                        "same_checkpoint_and_deals_as": str(b["path"]),
                    }
                    for a, b in aliases
                ],
                "scope": "Byte-identical checkpoint and identical protocol/deals reused; not independent samples.",
            },
        )
        for item in expected:
            verified_report(
                item["path"],
                item["spec"],
                item["pairs"],
                item["seed"],
                item["checkpoint"],
                frozen["inputs_sha256"],
                "normal",
            )
        authenticated = coverage(root, c, models, phase)
        if not authenticated["complete"]:
            raise ValueError("incomplete authenticated " + phase + " coverage")
        return authenticated

    try:
        arm_deadline(deadline)
        status()
        verify_snapshot(root / "source")
        validate_inputs(root / "inputs")
        verify_parents(root, c)
        save_plan()
        baselines = {"reference": root / "inputs/legacy-champion-lstm16.pt"}
        baselines.update(
            {
                f"parent-{seed}": root
                / "parents"
                / c["parent_sources"][str(seed)]["file"]
                for seed in c["seeds"]
            }
        )
        if c["version"] in (3, 4):
            prior = root / "parents/prior-nominee.pt"
            if file_hash(prior) != c["prior_nominee"]["sha256"]:
                raise ValueError("frozen prior nominee changed")
            baselines["prior-nominee"] = prior
        stage = "training"
        for round_index, budget in enumerate(c["budgets"]):
            previous = (
                plan["completed_common_budgets"][-1]
                if plan["completed_common_budgets"]
                else 0
            )
            forecast = None
            if previous:
                forecast = (
                    1.35
                    * (budget - previous)
                    * math.fsum(
                        main_training_rate(root / "trials" / f"{profile}-{seed}")
                        for seed in c["seeds"]
                        for profile in profiles
                    )
                )
            round_plan = {
                "budget": budget,
                "previous_budget": previous,
                "forecast_seconds": forecast,
                "status": "planned",
            }
            plan["rounds"].append(round_plan)
            if (
                forecast is not None
                and time.monotonic() + forecast >= training_deadline
            ):
                round_plan.update(
                    status="skipped",
                    reason="whole common round would consume evaluation reserve",
                )
                plan["skipped_budgets"] = c["budgets"][round_index:]
                save_plan()
                break
            save_plan()
            rotation = round_index % len(c["seeds"])
            seeds = c["seeds"][rotation:] + c["seeds"][:rotation]
            for block, seed in enumerate(seeds):
                shift = (block + round_index) % len(profiles)
                for profile in profiles[shift:] + profiles[:shift]:
                    name = f"{profile}-{seed}"
                    trial = root / "trials" / name
                    parent = baselines[f"parent-{seed}"]
                    context = {
                        "profile": profile,
                        "seed": seed,
                        "target_choices": budget,
                    }
                    command = training_command(
                        root, c, profile, seed, budget, resume=bool(previous)
                    )
                    jobs_run(
                        [(command, root / "logs" / f"{name}-c{budget}.log")],
                        until=training_deadline,
                    )
                    if not previous:
                        verify_initial(trial, parent)
                    verify_endpoint(root, trial, c, profile, seed, budget)
                    plan["completed_trial_budgets"].setdefault(name, []).append(budget)
                    save_plan()
                if not previous and c["version"] in (3, 4):
                    verify_paired_rng(
                        *(root / "trials" / f"{p}-{seed}" for p in profiles)
                    )
            round_plan["status"] = "complete"
            plan["completed_common_budgets"].append(budget)
            save_plan()
        if not plan["completed_common_budgets"]:
            raise BudgetExhausted("no complete common training milestone")
        models = dict(baselines)
        for budget in plan["completed_common_budgets"]:
            for seed in c["seeds"]:
                for profile in profiles:
                    trial = root / "trials" / f"{profile}-{seed}"
                    models[f"{profile}-{seed}-c{budget}"] = verify_endpoint(
                        root, trial, c, profile, seed, budget
                    )
        context = {"selection_candidates": len(models) - len(baselines)}
        stage = "selection-evaluation"
        status()
        evaluations(models, "selection")
        check_cap(deadline)
        selection_summary = summarize(root, c, models, "selection")
        atomic_json(root / "comparison-selection.json", selection_summary)
        chosen = select_candidates(selection_summary, c)
        atomic_json(root / "selection.json", chosen)
        selection_hash = file_hash(root / "selection.json")
        (root / "selection.sha256").write_text(selection_hash + "\n")
        final_models = dict(baselines)
        for seed in c["seeds"]:
            entry = chosen["selected"][str(seed)]
            checkpoint = Path(entry["checkpoint"]).resolve()
            if (
                not checkpoint.is_relative_to(root)
                or entry["name"] not in models
                or checkpoint != models[entry["name"]].resolve()
                or file_hash(checkpoint) != entry["checkpoint_sha256"]
            ):
                raise ValueError(
                    "selected checkpoint outside authenticated common milestones"
                )
            final_models[f"selected-{seed}"] = checkpoint
        stage = "final-evaluation"
        context = {
            "nominee": chosen["nominee"],
            "nominee_checkpoint": chosen["nominee_checkpoint"],
        }
        status()
        evaluations(final_models, "final")
        if file_hash(root / "selection.json") != selection_hash:
            raise ValueError("frozen nominee changed during confirmation")
        check_cap(deadline)
        result = summarize(root, c, final_models, "final")
        result.update(
            selection=chosen, selection_sha256=selection_hash, training_plan=plan
        )
        if c["version"] in (3, 4):
            record_frozen_nominee(result, chosen, c)
        result["preservation"] = verify_preservation(
            json.loads((root / "preservation.json").read_text())
        )
        check_cap(deadline)
        if not result["preservation"]["passed"]:
            raise ValueError("preserved artifact mismatch")
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
                "training_plan": plan,
                "unfinished_coverage": True,
                "approved": False,
            },
        )
        status(state, error=str(error))
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminate()
        atomic_json(
            root / "partial-results.json",
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "training_plan": plan,
                "unfinished_coverage": True,
                "approved": False,
            },
        )
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
