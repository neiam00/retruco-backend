"""Paired comparisons of four fixed continuation endpoints per parent block.

Continuation seeds are paired with distinct frozen parent training seeds. They
are not fresh replicates from a single parent. All confidence intervals are
exploratory, unadjusted development intervals; this module never approves a
checkpoint for release.
"""

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .controlled_compare import crossed_interval
from .cooperation_refusal_compare import PARTNERS, clusters
from .recurrent import file_hash
from .statistics import interval
from .target_compare import endpoint_receipt, verified_report

ARMS = {
    "A": {"balance": False, "betting": False},
    "B": {"balance": True, "betting": False},
    "C": {"balance": False, "betting": True},
    "D": {"balance": True, "betting": True},
}
SIGNAL_MODES = ("normal", "no-send", "no-receive", "no-partner-delivery")
MODE_TARGETS = {2: 0.70, 4: 0.60}


def _validate_config(config):
    if config.get("arms") != ARMS:
        raise ValueError("comparison requires the declared A/B/C/D factorial controls")
    seeds = config.get("seeds", [])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("distinct continuation seed blocks are required")
    specs = config["primary_specs"] + config["diagnostic_specs"]
    names = [spec["id"] for spec in specs]
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        raise ValueError("evaluation specs require unique simple identifiers")
    primary = {spec["id"] for spec in config["primary_specs"]}
    if primary != {"native-headsup"} | {"native-" + partner for partner in PARTNERS}:
        raise ValueError(
            "complete equally weighted native partner coverage is required"
        )
    for phase in ("primary", "diagnostic", "ablation"):
        if config[phase + "_pairs"] < 2:
            raise ValueError("paired comparisons require at least two deals")
    if config["choices"] < 1:
        raise ValueError("positive additional training choices are required")
    sources = config["parent_sources"]
    if set(sources) != {str(seed) for seed in seeds}:
        raise ValueError("each continuation block requires one declared matched parent")
    for seed in seeds:
        source = sources[str(seed)]
        if source["file"] != f"{seed}.pt":
            raise ValueError(
                "matched parent files must be the frozen parents/{seed}.pt copies"
            )


def _model_paths(root, config):
    result = {"reference": root / "inputs/legacy-champion-lstm16.pt"}
    result.update(
        {f"parent-{seed}": root / "parents" / f"{seed}.pt" for seed in config["seeds"]}
    )
    result.update(
        {
            f"{arm}-{seed}": root / "trials" / f"{arm}-{seed}" / "candidate.pt"
            for arm in ARMS
            for seed in config["seeds"]
        }
    )
    return result


def expected_reports(root, config, ablation_arm=None):
    """Exact report jobs, including checkpoint and protocol inputs for receipts."""
    root = Path(root)
    _validate_config(config)
    models = _model_paths(root, config)
    jobs = []
    for name, checkpoint in models.items():
        for phase in ("primary", "diagnostic"):
            for spec in config[phase + "_specs"]:
                jobs.append(
                    {
                        "path": root / "evaluation" / name / (spec["id"] + ".json"),
                        "model": name,
                        "phase": phase,
                        "spec": spec,
                        "checkpoint": checkpoint,
                        "pairs": config[phase + "_pairs"],
                        "seed": config[phase + "_seed"],
                        "signals": "normal",
                    }
                )
    if ablation_arm is not None:
        if ablation_arm not in ARMS:
            raise ValueError("unknown ablation arm")
        spec = next(
            s for s in config["primary_specs"] if s["id"] == "native-literal-v2"
        )
        names = ["reference"] + [f"parent-{s}" for s in config["seeds"]]
        names += [f"{ablation_arm}-{s}" for s in config["seeds"]]
        for name in names:
            for mode in SIGNAL_MODES:
                jobs.append(
                    {
                        "path": root / "ablation" / name / f"{spec['id']}-{mode}.json",
                        "model": name,
                        "phase": "ablation",
                        "spec": spec,
                        "checkpoint": models[name],
                        "pairs": config["ablation_pairs"],
                        "seed": config["ablation_seed"],
                        "signals": mode,
                    }
                )
    return jobs


def _reports(root, config, ablation_arm=None):
    jobs = expected_reports(root, config, ablation_arm)
    inputs_sha = file_hash(root / "inputs/manifest.json")
    reports, missing, invalid = {}, [], []
    for job in jobs:
        path = job["path"]
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            reports[str(path)] = verified_report(
                path,
                job["spec"],
                job["pairs"],
                job["seed"],
                job["checkpoint"],
                inputs_sha,
                job["signals"],
            )
        except (ValueError, OSError, KeyError, TypeError) as error:
            invalid.append({"path": str(path), "error": str(error)})
    expected = {job["path"] for job in jobs}
    present = set((root / "evaluation").glob("*/*.json"))
    if ablation_arm is not None:
        present |= set((root / "ablation").glob("*/*.json"))
    unexpected = sorted(str(path) for path in present - expected)
    coverage_result = {
        "expected": len(jobs),
        "authenticated": len(reports),
        "complete": not missing and not invalid and not unexpected,
        "missing": missing,
        "invalid": invalid,
        "unexpected": unexpected,
        "normal_expected": sum(j["phase"] != "ablation" for j in jobs),
        "ablation_expected": sum(j["phase"] == "ablation" for j in jobs),
        "inputs_sha256": inputs_sha,
    }
    return reports, coverage_result


def coverage(root, config, ablation_arm=None):
    """Authenticate exact evaluation paths; return missing/invalid provenance."""
    return _reports(Path(root), config, ablation_arm)[1]


def _load_checkpoint(path):
    # Campaign checkpoints are local frozen inputs, including optimizer/RNG state.
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("kind") != "rmappo-v1":
        raise ValueError("expected a full RMAPPO checkpoint: " + str(path))
    if not isinstance(saved.get("actor"), dict) or not saved["actor"]:
        raise ValueError("checkpoint has no actor state: " + str(path))
    if not isinstance(saved.get("critic"), dict) or not saved["critic"]:
        raise ValueError("checkpoint has no critic state: " + str(path))
    return saved


def _same_actor(left, right):
    return left.keys() == right.keys() and all(
        isinstance(left[key], torch.Tensor)
        and isinstance(right[key], torch.Tensor)
        and torch.equal(left[key], right[key])
        for key in left
    )


def _finite_metrics(value):
    if isinstance(value, dict):
        return all(_finite_metrics(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite_metrics(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def _training(root, config, name, arm, seed, parent):
    folder = root / "trials" / name
    checkpoint = folder / "candidate.pt"
    endpoint_receipt(checkpoint)
    saved = _load_checkpoint(checkpoint)
    initial = _load_checkpoint(folder / "initial.pt")
    settings = saved["configuration"]
    source = config["parent_sources"][str(seed)]
    coefficient = 0.1 if ARMS[arm]["betting"] else 0.0
    expected = {
        "arm": arm,
        "seed": seed,
        "cell": config["cell"],
        "parent_sha256": source["sha256"],
        "parent_choices": source["original_choices"],
        "partner_balance": ARMS[arm]["balance"],
        "response_anchor": coefficient,
        "teacher_sha256": file_hash(root / "inputs/legacy-champion-lstm16.pt"),
        "inputs_sha256": file_hash(root / "inputs/manifest.json"),
    }
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "endpoint initialization/arm/control configuration differs: " + name
        )
    if json.loads((folder / "configuration.json").read_text()) != settings:
        raise ValueError("endpoint configuration file differs from checkpoint: " + name)
    if initial["configuration"] != settings or initial.get("choices") != 0:
        raise ValueError(
            "initial continuation checkpoint configuration/count differs: " + name
        )
    if not _same_actor(initial["actor"], parent["actor"]):
        raise ValueError("initial actor does not match its frozen parent: " + name)
    if not _same_actor(initial["critic"], parent["critic"]):
        raise ValueError("initial critic does not match its frozen parent: " + name)
    if any(
        initial.get(key, {}).get("state") != {}
        for key in ("actor_optimizer", "critic_optimizer")
    ):
        raise ValueError("continuation must start with fresh empty Adam state: " + name)
    status = json.loads((folder / "status.json").read_text())
    if (
        status.get("status") != "chunk-complete"
        or status.get("target_choices") != config["choices"]
        or status.get("total_choices") != saved.get("choices")
        or saved.get("choices", -1) < config["choices"]
        or status.get("update") != saved.get("updates")
    ):
        raise ValueError(
            "endpoint is not a completed declared continuation budget: " + name
        )
    rows = [
        json.loads(line)
        for line in (folder / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not rows or any(
        not isinstance(row, dict) or not _finite_metrics(row) for row in rows
    ):
        raise ValueError("missing or nonfinite endpoint training metrics: " + name)
    total, updates, elapsed = 0, 0, 0.0
    objective_mass = {}
    for row in rows:
        if row.get("stage") not in ("training", "critic-warmup"):
            raise ValueError("unknown endpoint training metric stage: " + name)
        if row.get("stage") == "critic-warmup":
            if row.get("total_choices") != total:
                raise ValueError(
                    "critic warmup counted as additional learner choices: " + name
                )
            continue
        count = row.get("choices_this_update", 0)
        if type(count) is not int or count <= 0:
            raise ValueError("training metrics lack genuine-choice counts: " + name)
        total += count
        updates += 1
        if row.get("total_choices") != total or row.get("update") != updates:
            raise ValueError(
                "training metrics do not reconstruct the endpoint budget: " + name
            )
        balancing = row.get("partner_balance", {})
        anchor = row.get("response_anchor", {})
        if balancing.get("applied") is not ARMS[arm]["balance"]:
            raise ValueError("actual balance application differs from arm: " + name)
        if anchor.get("coefficient") != coefficient:
            raise ValueError(
                "actual response-anchor coefficient differs from arm: " + name
            )
        for group, mass in balancing.get("actual_objective_masses", {}).items():
            objective_mass[group] = objective_mass.get(group, 0.0) + count * mass
        elapsed += row.get("elapsed_seconds", 0.0)
    if total != saved["choices"] or updates != saved["updates"]:
        raise ValueError("metrics and checkpoint endpoint counts differ: " + name)
    return {
        "status": status,
        "checkpoint_sha256": file_hash(checkpoint),
        "initial_checkpoint_sha256": file_hash(folder / "initial.pt"),
        "configuration": settings,
        "parent_sha256": source["sha256"],
        "parent_original_seed": source["original_seed"],
        "parent_original_choices": source["original_choices"],
        "initial_actor_matches_parent": True,
        "initial_critic_matches_parent": True,
        "initial_optimizers_empty": True,
        "additional_choices": total,
        "updates": updates,
        "metric_rows": len(rows),
        "training_seconds_from_metrics": elapsed,
        "mean_actual_objective_masses": {
            key: value / total for key, value in objective_mass.items()
        },
        "last_training_metrics": next(
            row for row in reversed(rows) if row["stage"] == "training"
        ),
    }


@lru_cache(maxsize=4096)
def _cached_ci(shape, data, crossed):
    values = np.frombuffer(data, dtype=np.float64).reshape(shape)
    return tuple(crossed_interval(values) if crossed else interval(values))


def _ci(values, crossed):
    values = np.ascontiguousarray(values, dtype=np.float64)
    return list(_cached_ci(values.shape, values.tobytes(), crossed))


def _summary(values, baseline, crossed=False, target=None):
    values, baseline = np.asarray(values), np.asarray(baseline)
    delta = values - baseline
    result = {
        "win_rate": float(values.mean()),
        "win_ci95": _ci(values, crossed),
        "gain": float(delta.mean()),
        "gain_ci95": _ci(delta, crossed),
    }
    if crossed:
        result["per_block_win_rates"] = values.mean(axis=1).tolist()
        result["per_block_gains"] = delta.mean(axis=1).tolist()
    if target is not None:
        result.update(
            target_win_rate=target,
            target_met=result["win_rate"] >= target,
            confidence_above_chance=result["win_ci95"][0] > 0.5,
        )
    return result


def _mode_values(values):
    return {
        2: values["native-headsup"],
        4: np.stack([values["native-" + p] for p in PARTNERS]).mean(axis=0),
    }


def _score(modes):
    return 0.5 * (modes[2] / MODE_TARGETS[2] + modes[4] / MODE_TARGETS[4])


def _rank(row):
    modes = row["modes"]
    return (
        row["acceptable_strength"],
        min(modes["2"]["win_rate"] / 0.70, modes["4"]["win_rate"] / 0.60),
        modes["4"]["win_rate"],
    )


def _assessment(
    current,
    before,
    configurations,
    baseline_configurations,
    specs,
    margin,
    crossed=False,
):
    modes, old_modes = _mode_values(current), _mode_values(before)
    mode_summary = {
        str(mode): _summary(modes[mode], old_modes[mode], crossed, target)
        for mode, target in MODE_TARGETS.items()
    }
    strata, regressions = {}, []
    for spec in specs:
        name = spec["id"]
        row = _summary(current[name], before[name], crossed)
        row["configurations"] = []
        for key in sorted(configurations[name]):
            result = _summary(
                configurations[name][key], baseline_configurations[name][key], crossed
            )
            result.update(players=key[0], target=key[1], flor=key[2])
            row["configurations"].append(result)
            if result["gain"] < -margin and result["gain_ci95"][1] < 0:
                regressions.append({"stratum": name, **result})
        strata[name] = row
    score, previous_score = _score(modes), _score(old_modes)
    guard = (
        all(row["gain"] >= -margin for row in mode_summary.values())
        and all(row["gain"] >= -margin for row in strata.values())
        and not regressions
    )
    return {
        "modes": mode_summary,
        "strata": strata,
        "balanced_score": float(score.mean()),
        "balanced_gain": float((score - previous_score).mean()),
        "balanced_gain_ci95": _ci(score - previous_score, crossed),
        "observed_guard_pass": bool(guard),
        "configuration_regressions": regressions,
        "guard_margin": margin,
        "guard_version": "matched-parent-and-champion-strata-configuration-v1",
        "acceptable_strength": all(
            row["target_met"] and row["confidence_above_chance"]
            for row in mode_summary.values()
        ),
    }


def _stack(values, names):
    return {
        key: np.stack([values[name][key] for name in names]) for key in values[names[0]]
    }


def _stack_configurations(configurations, names):
    return {
        spec: {
            key: np.stack([configurations[name][spec][key] for name in names])
            for key in configurations[names[0]][spec]
        }
        for spec in configurations[names[0]]
    }


def _contrasts(matrices):
    a, b, c, d = (matrices[arm] for arm in ARMS)
    effects = {
        "balance": 0.5 * ((b - a) + (d - c)),
        "betting": 0.5 * ((c - a) + (d - b)),
        "interaction": d - c - b + a,
        "D_minus_A": d - a,
    }
    return {
        name: {
            "estimate": float(effect.mean()),
            "crossed_ci95": _ci(effect, True),
            "per_block_effects": effect.mean(axis=1).tolist(),
        }
        for name, effect in effects.items()
    }


def ablation_comparison(root, config, arm):
    """Compare signal-removal losses on their own paired development deal set."""
    root = Path(root)
    reports, authenticated = _reports(root, config, arm)
    if not authenticated["complete"]:
        raise ValueError(
            "ablation coverage/provenance is incomplete: " + json.dumps(authenticated)
        )
    jobs = [
        job for job in expected_reports(root, config, arm) if job["phase"] == "ablation"
    ]
    values = {}
    for job in jobs:
        values.setdefault(job["model"], {})[job["signals"]] = clusters(
            reports[str(job["path"])]
        )
    individuals, losses = {}, {}
    for name, modes in values.items():
        losses[name] = {
            mode: modes["normal"] - modes[mode] for mode in SIGNAL_MODES[1:]
        }
        individuals[name] = {
            "normal_win_rate": float(modes["normal"].mean()),
            "normal_win_ci95": _ci(modes["normal"], False),
            "signal_removals": {
                mode: {
                    "win_rate": float(modes[mode].mean()),
                    "normal_minus_removed": float(loss.mean()),
                    "paired_ci95": _ci(loss, False),
                }
                for mode, loss in losses[name].items()
            },
        }
    comparisons = {}
    for mode in SIGNAL_MODES[1:]:
        current = np.stack([losses[f"{arm}-{seed}"][mode] for seed in config["seeds"]])
        parents = np.stack([losses[f"parent-{seed}"][mode] for seed in config["seeds"]])
        champion = losses["reference"][mode][None]
        comparisons[mode] = {
            "mean_normal_minus_removed": float(current.mean()),
            "crossed_ci95": _ci(current, True),
            "loss_change_vs_matched_parent": float((current - parents).mean()),
            "loss_change_vs_matched_parent_ci95": _ci(current - parents, True),
            "loss_change_vs_champion": float((current - champion).mean()),
            "loss_change_vs_champion_ci95": _ci(current - champion, True),
        }
    return {
        "arm": arm,
        "individuals": individuals,
        "comparisons": comparisons,
        "coverage": authenticated,
        "approved": False,
        "scope": "signal-removal sensitivity on separate paired deals; descriptive, not a release guard",
    }


def compare(root, config):
    """Validate and compare every predeclared endpoint, without selecting snapshots."""
    root = Path(root)
    _validate_config(config)
    reports, authenticated = _reports(root, config)
    if not authenticated["complete"]:
        raise ValueError(
            "normal evaluation coverage/provenance is incomplete: "
            + json.dumps(authenticated)
        )
    models = _model_paths(root, config)
    specs = config["primary_specs"] + config["diagnostic_specs"]
    values, configurations, parent_payloads = {}, {}, {}
    for seed in config["seeds"]:
        source = config["parent_sources"][str(seed)]
        path = models[f"parent-{seed}"]
        if file_hash(path) != source["sha256"]:
            raise ValueError("matched parent hash changed: " + str(seed))
        saved = _load_checkpoint(path)
        settings = saved["configuration"]
        if (
            saved.get("choices") != source["original_choices"]
            or settings.get("seed") != source["original_seed"]
            or settings.get("cell") != config["cell"]
        ):
            raise ValueError(
                "matched parent seed/choice/cell provenance differs: " + str(seed)
            )
        parent_payloads[seed] = saved
    for name in models:
        values[name], configurations[name] = {}, {}
        for spec in specs:
            report = reports[str(root / "evaluation" / name / (spec["id"] + ".json"))]
            values[name][spec["id"]] = clusters(report)
            configurations[name][spec["id"]] = {
                (row["players"], row["target"], row["flor"]): np.asarray(
                    row["pair_scores"]
                )
                for row in report["configurations"]
            }
    margin = config.get("regression_margin", 0.02)
    if not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
        raise ValueError("regression margin must be finite and nonnegative")
    individuals, variants, arm_values, arm_modes = {}, {}, {}, {}
    for arm in ARMS:
        for seed in config["seeds"]:
            name = f"{arm}-{seed}"
            training = _training(root, config, name, arm, seed, parent_payloads[seed])
            comparisons = {}
            for label, baseline in (
                ("matched_parent", f"parent-{seed}"),
                ("champion", "reference"),
                ("control_A", f"A-{seed}"),
            ):
                comparisons[label] = _assessment(
                    values[name],
                    values[baseline],
                    configurations[name],
                    configurations[baseline],
                    specs,
                    margin,
                )
            champion = comparisons["champion"]
            evaluation_behavior = {}
            for spec in specs:
                report = reports[
                    str(root / "evaluation" / name / (spec["id"] + ".json"))
                ]
                evaluation_behavior[spec["id"]] = {
                    "net_points_per_hand": float(
                        clusters(report, "pair_net_points_per_hand").mean()
                    ),
                    **{
                        key: report[key]
                        for key in (
                            "behavior",
                            "decision_contexts",
                            "response_families",
                        )
                        if key in report
                    },
                }
            individuals[name] = {
                "arm": arm,
                "continuation_seed": seed,
                "parent_model": f"parent-{seed}",
                "checkpoint": str(models[name]),
                "checkpoint_sha256": training["checkpoint_sha256"],
                "initialization": "continuation from its matched frozen parent; not fresh training",
                "training": training,
                "comparisons": comparisons,
                "evaluation_behavior": evaluation_behavior,
                "modes": champion["modes"],
                "balanced_score": champion["balanced_score"],
                "acceptable_strength": champion["acceptable_strength"],
                "observed_guard_pass": all(
                    comparisons[label]["observed_guard_pass"]
                    for label in ("matched_parent", "champion")
                ),
                "approved": False,
            }
        names = [f"{arm}-{seed}" for seed in config["seeds"]]
        arm_values[arm] = _stack(values, names)
        arm_modes[arm] = _mode_values(arm_values[arm])
        current_configs = _stack_configurations(configurations, names)
        comparisons = {}
        baseline_names = {
            "matched_parent": [f"parent-{seed}" for seed in config["seeds"]],
            "champion": ["reference"] * len(config["seeds"]),
            "control_A": [f"A-{seed}" for seed in config["seeds"]],
        }
        for label, baselines in baseline_names.items():
            comparisons[label] = _assessment(
                arm_values[arm],
                _stack(values, baselines),
                current_configs,
                _stack_configurations(configurations, baselines),
                specs,
                margin,
                crossed=True,
            )
        champion = comparisons["champion"]
        blocks_safe = all(
            individuals[name]["comparisons"]["matched_parent"]["observed_guard_pass"]
            for name in names
        )
        variants[arm] = {
            "flags": ARMS[arm],
            "members": names,
            "comparisons": comparisons,
            "modes": champion["modes"],
            "diagnostics": champion["strata"],
            "balanced_score": champion["balanced_score"],
            "acceptable_strength": champion["acceptable_strength"],
            "all_matched_parent_blocks_guard_pass": blocks_safe,
            "observed_guard_pass": blocks_safe
            and all(
                comparisons[label]["observed_guard_pass"]
                for label in ("matched_parent", "champion")
            ),
            "approved": False,
        }
    contrasts = {
        str(mode): _contrasts({arm: arm_modes[arm][mode] for arm in ARMS})
        for mode in MODE_TARGETS
    }
    contrasts["balanced_score"] = _contrasts(
        {arm: _score(arm_modes[arm]) for arm in ARMS}
    )
    contrasts["diagnostics"] = {
        spec["id"]: _contrasts({arm: arm_values[arm][spec["id"]] for arm in ARMS})
        for spec in specs
    }
    exploratory_arm = max(ARMS, key=lambda arm: _rank(variants[arm]))
    nominee = max(
        variants[exploratory_arm]["members"], key=lambda name: _rank(individuals[name])
    )
    safe = [
        name
        for name, value in individuals.items()
        if value["observed_guard_pass"]
        and variants[value["arm"]]["observed_guard_pass"]
    ]
    retained = (
        max(safe, key=lambda name: _rank(individuals[name])) if safe else "reference"
    )
    blocks = [
        {
            "continuation_seed": seed,
            "parent_original_seed": config["parent_sources"][str(seed)][
                "original_seed"
            ],
            "parent_original_choices": config["parent_sources"][str(seed)][
                "original_choices"
            ],
            "parent_sha256": config["parent_sources"][str(seed)]["sha256"],
        }
        for seed in config["seeds"]
    ]
    return {
        "protocol": "rmappo-controls-comparison-v1",
        "individuals": individuals,
        "variants": variants,
        "contrasts": contrasts,
        "blocks": blocks,
        "selected": exploratory_arm,
        "exploratory_arm": exploratory_arm,
        "exploratory_nominee": nominee,
        "nominee": nominee,
        "retained_for_review": retained,
        "retained_checkpoint": str(models[retained]),
        "starter_fallback": retained == "reference",
        "coverage": authenticated,
        "model_sha256": {name: file_hash(path) for name, path in models.items()},
        "selection": {
            "endpoint_policy": "all predeclared continuation endpoints; no snapshot search",
            "ablation_arm_policy": "both targets/confidence first, then weaker normalized mode, then 2v2",
            "retention_policy": "individual and arm guards against matched parents and champion",
            "regression_margin": margin,
            "approved": False,
        },
        "scope": (
            "paired development deals; crossed matched-parent/continuation-block and deal bootstrap; "
            "parent-seed blocks are not fresh runs from one parent; unadjusted multiple comparisons; "
            "factorial effects describe these complete training controls, not isolated mechanistic causes; "
            "the frozen response teacher is an experimental anchor, not a proven fix; no release approval"
        ),
        "approved": False,
    }
