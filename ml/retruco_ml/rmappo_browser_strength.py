"""Paired float/int8 playing-strength check; never trains or approves a model."""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from .artifacts import digest
from .bridge import Simulator
from .improvement_evaluate import evaluate, frozen_models
from .mappo_export import load_actor, quantized
from .statistics import interval
from .team_play_probe import CHOICE_PROFILES


def work(
    checkpoint,
    inputs,
    simulator,
    output,
    spec,
    use_int8,
    pairs,
    seed,
    card_policy="policy",
):
    torch.set_num_threads(1)
    actor = load_actor(checkpoint)
    if use_int8:
        actor = quantized(actor)
    models = frozen_models(json.loads(inputs.read_text()), inputs.parent)
    started = time.monotonic()
    with Simulator(simulator, "binary") as env:
        report = evaluate(
            actor,
            models,
            env,
            pairs,
            seed,
            spec["partner"],
            "native",
            "normal",
            spec["players"],
            card_policy,
        )
    report.update(
        checkpoint_sha256=digest(checkpoint),
        inputs_sha256=digest(inputs),
        quantized=use_int8,
        elapsed_seconds=time.monotonic() - started,
    )
    output.write_text(json.dumps(report) + "\n")
    output.with_suffix(".sha256").write_text(digest(output) + "\n")
    return output.name, report["elapsed_seconds"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("bundle", type=Path)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--simulator",
        type=Path,
        help="simulator built with the selected browser candidate policy",
    )
    p.add_argument("--pairs", type=int, default=512)
    p.add_argument("--seed", type=int, default=122000)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((a.campaign / "configuration.json").read_text())
    manifest = json.loads((a.bundle / "manifest.json").read_text())
    if manifest["checkpoint_sha256"] != digest(a.checkpoint):
        raise ValueError("checkpoint mismatch")
    if manifest.get("team_play") not in (
        None,
        "closing-card-and-flor-v1",
        "closing-card-and-flor-v2",
    ):
        raise ValueError("unsupported team decision rule")
    card_policy = manifest.get("team_play", "policy")
    if manifest.get("choice_policy") is not None:
        if (
            manifest["choice_policy"] not in CHOICE_PROFILES
            or card_policy != "closing-card-and-flor-v2"
        ):
            raise ValueError("unsupported browser choice policy")
        card_policy = manifest["choice_policy"]
    inputs = a.campaign / "inputs/manifest.json"
    simulator = a.simulator or a.campaign / "source/bin/selfplay"
    plan = {
        "pairs": a.pairs,
        "seed": a.seed,
        "specs": config["primary_specs"],
        "checkpoint_sha256": digest(a.checkpoint),
        "model_sha256": manifest["sha256"],
        "inputs_sha256": digest(inputs),
        "simulator_sha256": digest(simulator),
        "loss_upper_ci95_max": 0.01,
        "targets": {"1v1": 0.70, "2v2": 0.60},
        "approved": False,
    }
    if manifest.get("choice_policy"):
        plan["choice_policy"] = card_policy
    planpath = a.output / "plan.json"
    if manifest.get("team_play"):
        plan["team_play"] = manifest["team_play"]
    if planpath.exists() and json.loads(planpath.read_text()) != plan:
        raise ValueError("immutable evaluation plan differs")
    planpath.write_text(json.dumps(plan, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = []
        for spec in config["primary_specs"]:
            for int8 in (False, True):
                output = a.output / f"{spec['id']}-{'int8' if int8 else 'float'}.json"
                if output.exists():
                    if (
                        digest(output)
                        != output.with_suffix(".sha256").read_text().strip()
                    ):
                        raise ValueError("changed receipt")
                    continue
                futures.append(
                    pool.submit(
                        work,
                        a.checkpoint,
                        inputs,
                        simulator,
                        output,
                        spec,
                        int8,
                        a.pairs,
                        a.seed,
                        card_policy,
                    )
                )
        for future in as_completed(futures):
            print(json.dumps({"completed": future.result()}), flush=True)
    groups = {2: {"float": [], "int8": []}, 4: {"float": [], "int8": []}}
    configurations = []
    for spec in config["primary_specs"]:
        reports = {
            name: json.loads((a.output / f"{spec['id']}-{name}.json").read_text())
            for name in ("float", "int8")
        }
        for name, report in reports.items():
            if (
                report["card_policy"] != card_policy
                or report["checkpoint_sha256"] != plan["checkpoint_sha256"]
                or report["quantized"] != (name == "int8")
                or report["pairs"] != a.pairs
                or report["seed"] != a.seed
                or report["partner"] != spec["partner"]
                or report["opponent"] != "native"
            ):
                raise ValueError(
                    "unpaired policy, weights or protocol in quantized strength report"
                )
        for original, exported in zip(
            reports["float"]["configurations"],
            reports["int8"]["configurations"],
            strict=True,
        ):
            for key in ("players", "target", "flor", "pairs"):
                if original[key] != exported[key]:
                    raise ValueError("unpaired reports")
            f = np.array(original["pair_scores"])
            q = np.array(exported["pair_scores"])
            loss = f - q
            groups[spec["players"]]["float"].append(f)
            groups[spec["players"]]["int8"].append(q)
            configurations.append(
                {
                    "spec": spec["id"],
                    **{k: original[k] for k in ("players", "target", "flor")},
                    "float": float(f.mean()),
                    "int8": float(q.mean()),
                    "loss": float(loss.mean()),
                    "loss_ci95": interval(loss.tolist()),
                }
            )
    modes = {}
    for count, group in groups.items():
        f = np.mean(group["float"], axis=0)
        q = np.mean(group["int8"], axis=0)
        loss = f - q
        modes[f"{count // 2}v{count // 2}"] = {
            "float": float(f.mean()),
            "float_ci95": interval(f.tolist()),
            "int8": float(q.mean()),
            "int8_ci95": interval(q.tolist()),
            "loss": float(loss.mean()),
            "loss_ci95": interval(loss.tolist()),
        }
    report = {
        "plan": plan,
        "modes": modes,
        "configurations": configurations,
        "quantization_gate_pass": all(
            row["loss_ci95"][1] <= 0.01 for row in modes.values()
        ),
        "targets_pass": all(
            row["int8"] >= plan["targets"][name] and row["int8_ci95"][0] > 0.5
            for name, row in modes.items()
        ),
        "approved": False,
        "scope": "paired export development validation; original pre-frozen final benchmark remains separate",
    }
    (a.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
