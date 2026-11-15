"""Authenticated target comparisons for longer, paired RMAPPO continuation.

The caller supplies a frozen mapping of model names to checkpoints. Selection
may inspect predeclared milestones; final reports must evaluate the selection
frozen before those deals are opened. Neither this module nor target attainment
approves a model for release.
"""

import json
import math
import re
from pathlib import Path

import numpy as np

from .cooperation_refusal_compare import PARTNERS, clusters
from .recurrent import file_hash
from .rmappo_controls_compare import _assessment, _rank, _stack, _stack_configurations
from .target_compare import verified_report

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_CANDIDATE = re.compile(
    r"(original|target|gae99|mc|control|targeted|protected|focused|selected)-(\d+)(?:-c(\d+))?\Z"
)
_TARGETS = {"1v1": 0.70, "2v2": 0.60}
_CONTRASTS = {
    1: ("target", "original", "original_control"),
    2: ("mc", "gae99", "gae_control"),
    3: ("targeted", "control", "continuation_control"),
    4: ("focused", "protected", "matched_partner_population"),
}


def _inside(root, path):
    path = Path(path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("campaign artifact escapes root: " + str(path))
    return path


def _settings(root, config, models, phase, family):
    root = Path(root).resolve()
    family = family or ("final" if phase == "final" else "selection")
    if not _NAME.fullmatch(phase) or family not in ("selection", "final"):
        raise ValueError("invalid evaluation phase/family")
    seeds = config["seeds"]
    if (
        not seeds
        or any(type(s) is not int or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("distinct integer continuation seeds required")
    if config.get("mode_win_rate_targets", _TARGETS) != _TARGETS:
        raise ValueError("native targets must remain 70% heads-up and 60% teams")
    specs = config["primary_specs"] + config["diagnostic_specs"]
    ids = [s["id"] for s in specs]
    if len(set(ids)) != len(ids) or any(not _NAME.fullmatch(n) for n in ids):
        raise ValueError("unique simple report identifiers required")
    primary = {s["id"]: s for s in config["primary_specs"]}
    required = {"native-headsup"} | {"native-" + p for p in PARTNERS}
    if set(primary) != required:
        raise ValueError("complete equally weighted native target coverage required")
    for name, spec in primary.items():
        partner = (
            "reference-cpu" if name == "native-headsup" else name[len("native-") :]
        )
        players = 2 if name == "native-headsup" else 4
        if (spec["opponent"], spec["partner"], spec["players"]) != (
            "native",
            partner,
            players,
        ):
            raise ValueError("native target specification changed: " + name)
    for suffix in ("", "_diagnostic"):
        pairs, seed = (
            config[family + suffix + "_pairs"],
            config[family + suffix + "_seed"],
        )
        if type(pairs) is not int or pairs < 2 or type(seed) is not int or seed < 0:
            raise ValueError("invalid paired evaluation stream")
    baseline_names = {"reference", *(f"parent-{s}" for s in seeds)}
    if config.get("version") in (3, 4):
        baseline_names.add("prior-nominee")
    if not baseline_names.issubset(models):
        raise ValueError("reference and every matched parent must be evaluated")
    paths = {}
    for name, path in models.items():
        if not _NAME.fullmatch(name):
            raise ValueError("invalid model name")
        match = _CANDIDATE.fullmatch(name)
        if name not in baseline_names and not match:
            raise ValueError("unknown model identity: " + name)
        if (
            match
            and match[1] != "selected"
            and match[1] not in config.get("profiles", ("original", "target"))
        ):
            raise ValueError("candidate uses undeclared recipe")
        if match and int(match[2]) not in seeds:
            raise ValueError("candidate has undeclared continuation seed: " + name)
        paths[name] = _inside(root, path)
    margin = config.get("regression_margin", 0.02)
    if not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
        raise ValueError("invalid native regression margin")
    return root, family, paths


def expected_reports(root, config, models, phase="selection", family=None):
    """Return exact report jobs, safely resolved inside the campaign root."""
    root, family, paths = _settings(root, config, models, phase, family)
    jobs = []
    for name, checkpoint in paths.items():
        for diagnostic in (False, True):
            suffix = "_diagnostic" if diagnostic else ""
            specs = config["diagnostic_specs" if diagnostic else "primary_specs"]
            for spec in specs:
                jobs.append(
                    {
                        "path": _inside(
                            root, root / phase / name / (spec["id"] + ".json")
                        ),
                        "model": name,
                        "checkpoint": checkpoint,
                        "spec": spec,
                        "pairs": config[family + suffix + "_pairs"],
                        "seed": config[family + suffix + "_seed"],
                        "signals": "normal",
                        "diagnostic": diagnostic,
                    }
                )
    return jobs


def _reports(root, config, models, phase, family):
    root = Path(root).resolve()
    jobs = expected_reports(root, config, models, phase, family)
    inputs_sha = file_hash(_inside(root, root / "inputs/manifest.json"))
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
        except (OSError, ValueError, KeyError, TypeError) as error:
            invalid.append({"path": str(path), "error": str(error)})
    expected = {job["path"] for job in jobs}
    present = {p.resolve() for p in (root / phase).glob("*/*.json")}
    unexpected = sorted(str(p) for p in present - expected)
    return reports, {
        "expected": len(jobs),
        "authenticated": len(reports),
        "complete": not missing and not invalid and not unexpected,
        "missing": missing,
        "invalid": invalid,
        "unexpected": unexpected,
        "primary_expected": sum(not j["diagnostic"] for j in jobs),
        "diagnostic_expected": sum(j["diagnostic"] for j in jobs),
        "inputs_sha256": inputs_sha,
    }


def coverage(root, config, models, phase="selection", family=None):
    """Read-only exact-path, protocol and checkpoint/input hash validation."""
    return _reports(root, config, models, phase, family)[1]


def _native_guard(assessment, primary_ids, margin):
    # Separate diagnostics remain visible; they do not silently redefine the
    # explicit native benchmark objective of this continuation campaign.
    return (
        all(x["gain"] >= -margin for x in assessment["modes"].values())
        and all(assessment["strata"][name]["gain"] >= -margin for name in primary_ids)
        and not any(
            x["stratum"] in primary_ids for x in assessment["configuration_regressions"]
        )
    )


def _protected_gate(row, candidate_report, starter_report):
    # Compare the paired deal and point arrays in each configuration, plus
    # actual actions/counters. Identical aggregate win rates alone can hide drift.
    fields = ("configurations", "behavior", "decision_contexts", "response_families")
    exact = all(
        key in candidate_report
        and key in starter_report
        and candidate_report[key] == starter_report[key]
        for key in fields
    )
    matched = row["comparisons"]["matched_parent"]
    reference = matched["strata"]["native-reference-human"]
    both_targets = all(
        mode["target_met"] and mode["confidence_above_chance"]
        for mode in row["modes"].values()
    )
    return {
        "identical_headsup_results": exact,
        "both_targets": both_targets,
        "reference_human_gain": reference["gain"],
        "reference_human_gain_ci95": reference["gain_ci95"],
        "qualified": bool(
            exact
            and both_targets
            and row["native_guard_pass"]
            and matched["modes"]["4"]["gain"] >= 0
            and reference["gain"] > 0
        ),
    }


def _reference_rank(row):
    gate = row["protected_gate"]
    return (
        gate["qualified"],
        gate["identical_headsup_results"],
        row["modes"]["2"]["target_met"],
        row["modes"]["4"]["target_met"],
        gate["reference_human_gain_ci95"][0],
        row["modes"]["4"]["win_rate"],
        gate["reference_human_gain"],
    )


def summarize(root, config, models, phase="selection", family=None):
    """Compare every supplied endpoint against its parent and the champion.

    Output has ``individuals`` (including explicit reference/parent rows),
    ``groups`` (three matched blocks per recipe/milestone), and authenticated
    ``coverage``. Candidate names are original-SEED-cBUDGET,
    target-SEED-cBUDGET, or selected-SEED. No files are written.
    """
    root, family, paths = _settings(root, config, models, phase, family)
    reports, authenticated = _reports(root, config, paths, phase, family)
    if not authenticated["complete"]:
        raise ValueError(
            "incomplete authenticated evaluation coverage: " + json.dumps(authenticated)
        )
    specs = config["primary_specs"] + config["diagnostic_specs"]
    primary_ids = {s["id"] for s in config["primary_specs"]}
    margin = config.get("regression_margin", 0.02)
    values, configurations, behavior = {}, {}, {}
    for name in paths:
        values[name], configurations[name], behavior[name] = {}, {}, {}
        for spec in specs:
            report = reports[str(root / phase / name / (spec["id"] + ".json"))]
            key = spec["id"]
            values[name][key] = clusters(report)
            configurations[name][key] = {
                (x["players"], x["target"], x["flor"]): np.asarray(x["pair_scores"])
                for x in report["configurations"]
            }
            behavior[name][key] = {
                "net_points_per_hand": float(
                    clusters(report, "pair_net_points_per_hand").mean()
                ),
                "behavior": report["behavior"],
                "decision_contexts": report["decision_contexts"],
                "response_families": report.get("response_families", {}),
            }
    individuals, members = {}, {}
    treatment, control_profile, label = _CONTRASTS[config.get("version", 1)]
    for name, checkpoint in paths.items():
        match = _CANDIDATE.fullmatch(name)
        seed = (
            int(match[2])
            if match
            else int(name[7:])
            if name.startswith("parent-")
            else None
        )
        parent = f"parent-{seed}" if seed is not None else name
        baselines = {"matched_parent": parent, "champion": "reference"}
        if "prior-nominee" in values:
            baselines["prior_nominee"] = "prior-nominee"
        if match and match[1] == treatment:
            control = (
                (control_profile + "-")
                + match[2]
                + ("-c" + match[3] if match[3] else "")
            )
            if control in values:
                baselines[label] = control
        comparisons = {
            label: _assessment(
                values[name],
                values[baseline],
                configurations[name],
                configurations[baseline],
                specs,
                margin,
            )
            for label, baseline in baselines.items()
        }
        current = comparisons["matched_parent"]
        row = {
            "kind": "trained_candidate"
            if match
            else "prior_nominee"
            if name == "prior-nominee"
            else "reference"
            if seed is None
            else "matched_parent",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_hash(checkpoint),
            "seed": seed,
            "recipe": match[1] if match else None,
            "nominal_additional_choices": int(match[3]) if match and match[3] else None,
            "parent_model": parent,
            "modes": current["modes"],
            "acceptable_strength": current["acceptable_strength"],
            "native_guard_pass": bool(_native_guard(current, primary_ids, margin)),
            "all_diagnostic_guard_pass": current["observed_guard_pass"],
            "comparisons": comparisons,
            "evaluation_behavior": behavior[name],
            "approved": False,
        }
        if config.get("version") == 4:
            current_report = reports[str(root / phase / name / "native-headsup.json")]
            parent_report = reports[str(root / phase / parent / "native-headsup.json")]
            row["protected_gate"] = _protected_gate(row, current_report, parent_report)
        individuals[name] = row
        if match:
            group = match[1] + ("-c" + match[3] if match[3] else "")
            members.setdefault(group, {})[seed] = name
    groups = {}
    if len(config["seeds"]) == 3:
        for group, by_seed in members.items():
            if set(by_seed) != set(config["seeds"]):
                continue
            names = [by_seed[s] for s in config["seeds"]]
            current, configs = (
                _stack(values, names),
                _stack_configurations(configurations, names),
            )
            baselines = {
                "matched_parent": [f"parent-{s}" for s in config["seeds"]],
                "champion": ["reference"] * len(names),
            }
            if "prior-nominee" in values:
                baselines["prior_nominee"] = ["prior-nominee"] * len(names)
            original = group.replace(treatment, control_profile, 1)
            if group.startswith(treatment) and set(members.get(original, {})) == set(
                config["seeds"]
            ):
                baselines[label] = [members[original][s] for s in config["seeds"]]
            comparisons = {
                label: _assessment(
                    current,
                    _stack(values, before),
                    configs,
                    _stack_configurations(configurations, before),
                    specs,
                    margin,
                    True,
                )
                for label, before in baselines.items()
            }
            matched = comparisons["matched_parent"]
            groups[group] = {
                "members": names,
                "comparisons": comparisons,
                "modes": matched["modes"],
                "acceptable_strength": matched["acceptable_strength"],
                "native_guard_pass": bool(_native_guard(matched, primary_ids, margin)),
                "all_diagnostic_guard_pass": matched["observed_guard_pass"],
                "approved": False,
            }
    block_scope = (
        "three fresh continuation seeds from one preserved parent; public candidate features and semantic exploration are a bundle; "
        if config.get("version") == 3
        else "three fresh continuation seeds from one screened parent; "
        if config.get("version") == 2
        else "three blocks are not fresh replicates from one parent; "
    )
    if config.get("version") == 4:
        block_scope = (
            "three matched seeds from the calibrated starter, with its heads-up actor protected; "
            "arms compare 30% and 60% reference-human teammate matches; "
        )
    return {
        "protocol": "rmappo-target-comparison-v" + str(config.get("version", 1)),
        "phase": phase,
        "family": family,
        "individuals": individuals,
        "groups": groups,
        "coverage": authenticated,
        "model_paths": {name: str(p) for name, p in paths.items()},
        "mode_win_rate_targets": _TARGETS,
        "approved": False,
        "scope": "paired deals; crossed continuation-seed and deal intervals; "
        + block_scope
        + "starter/checkpoint selection is exploratory; "
        "unadjusted comparisons; native guards exclude separately reported diagnostics; no release approval",
    }


def select_candidates(summary, config):
    """Select one actual endpoint per seed; never disguise a fallback as training.

    Prefer native-guarded candidates. If a seed has none, retain the best actual
    trained candidate for explicitly exploratory confirmation. The caller freezes
    this result before final evaluation and handles any reference fallback.
    """
    selected = {}
    protected = config.get("version") == 4
    rank = _reference_rank if protected else _rank

    def eligible(row):
        return (
            row["protected_gate"]["qualified"]
            if protected
            else row["native_guard_pass"]
        )

    for seed in config["seeds"]:
        candidates = {
            name: row
            for name, row in summary["individuals"].items()
            if row["kind"] == "trained_candidate" and row["seed"] == seed
        }
        if not candidates:
            raise ValueError("no actual trained candidate for seed " + str(seed))
        guarded = {name: row for name, row in candidates.items() if eligible(row)}
        pool = guarded or candidates
        name = max(sorted(pool), key=lambda n: rank(pool[n]))
        row = pool[name]
        selected[str(seed)] = {
            "name": name,
            "checkpoint": row["checkpoint"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "recipe": row["recipe"],
            "nominal_additional_choices": row["nominal_additional_choices"],
            "native_guard_pass": row["native_guard_pass"],
            "acceptable_strength": row["acceptable_strength"],
            "exploratory": not bool(guarded),
            "starter_fallback": False,
            "approved": False,
        }
        if protected:
            selected[str(seed)]["protected_gate"] = row["protected_gate"]
    names = [x["name"] for x in selected.values()]
    guarded = [n for n in names if eligible(summary["individuals"][n])]
    nominee = max(
        sorted(guarded or names), key=lambda n: rank(summary["individuals"][n])
    )
    final_name = next(
        "selected-" + s for s, v in selected.items() if v["name"] == nominee
    )
    starter_fallback = False
    if config.get("version") in (3, 4):
        # All seed blocks have byte-identical screened starters. Keep each
        # selected training endpoint visible even if the starter wins selection.
        starter = "parent-" + str(config["seeds"][0])
        if not guarded or rank(summary["individuals"][starter]) >= rank(
            summary["individuals"][nominee]
        ):
            nominee = final_name = starter
            starter_fallback = True
    return {
        "selected": selected,
        "nominee": nominee,
        "nominee_final_name": final_name,
        "nominee_is_starter_fallback": starter_fallback,
        "nominee_checkpoint": summary["individuals"][nominee]["checkpoint"],
        "nominee_checkpoint_sha256": summary["individuals"][nominee][
            "checkpoint_sha256"
        ],
        "scope": "selection-only frozen choices; confirm on untouched deals; no release approval",
        "approved": False,
    }
