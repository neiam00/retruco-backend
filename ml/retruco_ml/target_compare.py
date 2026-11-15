"""Development comparisons for native exposure x undiscounted credit trials.

Selection and final confirmation use disjoint deal seeds. Starter fallbacks are
explicitly identified and never presented as independent trained policies.
"""

import json
from pathlib import Path

import numpy as np

from .controlled_compare import crossed_interval
from .cooperation_refusal_compare import (
    PARTNERS,
    clusters,
    read,
    summary,
    validate_report,
)
from .recurrent import file_hash
from .statistics import interval
from .target_protocol import (
    confirmation_budgets,
    rank_improves,
    selection_rank,
    targets,
)


def verified_report(path, spec, pairs, seed, checkpoint, inputs_sha, signals="normal"):
    receipt = path.with_suffix(".sha256")
    if not receipt.is_file() or receipt.read_text().strip() != file_hash(path):
        raise ValueError("evaluation receipt changed or missing: " + str(path))
    report = read(path)
    validate_report(
        report, spec, pairs, seed, file_hash(checkpoint), inputs_sha, signals
    )
    return report


def endpoint_receipt(path):
    receipt = path.with_suffix(".sha256")
    if not receipt.is_file() or receipt.read_text().strip() != file_hash(path):
        raise ValueError("endpoint receipt changed or missing")


def score_clusters(modes, config=None):
    goals = targets(config)
    return 0.5 * (modes[2] / goals[2] + modes[4] / goals[4])


def configuration_regressions(strata, margin=0.02):
    """Flag a material observed drop whose development paired CI excludes zero.

    This is a retention safeguard, not a multiple-testing-adjusted release test.
    Aggregate improvements must not hide a reproducible configuration failure.
    """
    return [
        {"stratum": name, **row}
        for name, value in strata.items()
        for row in value.get("configurations", [])
        if row["gain"] < -margin and row["gain_ci95"][1] < 0
    ]


def assessment(
    folder, baseline, specs, pairs, seed, checkpoint, starter, inputs_sha, config=None
):
    current, before, strata = {}, {}, {}
    for spec in specs:
        a = verified_report(
            folder / f"{spec['id']}.json", spec, pairs, seed, checkpoint, inputs_sha
        )
        b = verified_report(
            baseline / f"{spec['id']}.json", spec, pairs, seed, starter, inputs_sha
        )
        if (config or {}).get("version", 1) >= 3 and spec["players"] == 2:
            if (
                a["configurations"] != b["configurations"]
                or a["behavior"] != b["behavior"]
                or a["decision_contexts"] != b["decision_contexts"]
            ):
                raise ValueError("team specialist changed protected heads-up behavior")
        current[spec["id"]], before[spec["id"]] = clusters(a), clusters(b)
        strata[spec["id"]] = {
            **summary(current[spec["id"]], before[spec["id"]]),
            "net_points_per_hand": float(
                clusters(a, "pair_net_points_per_hand").mean()
            ),
            "behavior": a["behavior"],
            "decision_contexts": a["decision_contexts"],
            "configurations": [],
        }
        if "response_families" in a:
            strata[spec["id"]]["response_families"] = a["response_families"]
        parents = {(row["target"], row["flor"]): row for row in b["configurations"]}
        for row in a["configurations"]:
            delta = (
                np.asarray(row["pair_scores"])
                - parents[row["target"], row["flor"]]["pair_scores"]
            )
            strata[spec["id"]]["configurations"].append(
                {
                    "players": row["players"],
                    "target": row["target"],
                    "flor": row["flor"],
                    "win_rate": row["win_rate"],
                    "gain": float(delta.mean()),
                    "gain_ci95": interval(delta),
                }
            )
    required = {"native-headsup"} | {"native-" + p for p in PARTNERS}
    if not required.issubset(current):
        raise ValueError("complete native target coverage is required")
    modes = {
        2: current["native-headsup"],
        4: np.stack([current["native-" + p] for p in PARTNERS]).mean(0),
    }
    parent_modes = {
        2: before["native-headsup"],
        4: np.stack([before["native-" + p] for p in PARTNERS]).mean(0),
    }
    balanced, parent_balanced = (
        score_clusters(modes, config),
        score_clusters(parent_modes, config),
    )
    values = {
        str(mode): summary(modes[mode], parent_modes[mode], target=target)
        for mode, target in targets(config).items()
    }
    regressions = configuration_regressions(strata)
    guard = (
        all(row["gain"] >= -0.02 for row in values.values())
        and all(row["gain"] >= -0.02 for row in strata.values())
        and not regressions
    )
    return {
        "modes": values,
        "balanced_score": float(balanced.mean()),
        "balanced_gain": float((balanced - parent_balanced).mean()),
        "balanced_gain_ci95": interval(balanced - parent_balanced),
        "observed_guard_pass": bool(guard),
        "strata": strata,
        "configuration_regressions": regressions,
        "guard_version": "strata-and-configuration-v2",
        "acceptable_strength": all(
            v["target_met"] and v["confidence_above_chance"] for v in values.values()
        ),
    }, modes


def training_summary(folder):
    state = read(folder / "status.json")
    records = [
        json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()
    ]
    updates = [row for row in records if row["phase"] == "training"]
    elapsed = sum(row["collect_seconds"] + row["optimize_seconds"] for row in updates)
    return {
        "status": state["status"],
        "choices": state["choices"],
        "update": state["update"],
        "warmups": state["warmups"],
        "choices_per_second": sum(row["choices"] for row in updates) / elapsed,
        "replay_max_logp_error": max(
            row["diagnostics"]["replay_identity_max_logp_error"] for row in updates
        ),
        "last_policy_diagnostics": updates[-1]["diagnostics"],
        "latest_checkpoint": str(folder / "checkpoint.pt"),
    }


def discovery_comparison(root, config):
    root = Path(root)
    inputs_sha = file_hash(root / "inputs/manifest.json")
    starter = root / "inputs/legacy-champion-lstm16.pt"
    variants, individual, matrices = {}, {}, {}
    for recipe in config["recipes"]:
        matrices[recipe] = {2: [], 4: []}
        for seed in config["discovery_seeds"]:
            name = f"{recipe}-seed-{seed}"
            folder = root / "discovery" / name
            checkpoint = folder / "candidate.pt"
            endpoint_receipt(checkpoint)
            primary, modes = assessment(
                folder / "primary",
                root / "discovery-baseline/primary",
                config["primary_specs"],
                config["discovery_pairs"],
                config["discovery_seed"],
                checkpoint,
                starter,
                inputs_sha,
                config,
            )
            diagnostic = {}
            for spec in config["diagnostic_specs"]:
                a = verified_report(
                    folder / "diagnostic" / f"{spec['id']}.json",
                    spec,
                    config["diagnostic_pairs"],
                    config["diagnostic_seed"],
                    checkpoint,
                    inputs_sha,
                )
                b = verified_report(
                    root / "discovery-baseline/diagnostic" / f"{spec['id']}.json",
                    spec,
                    config["diagnostic_pairs"],
                    config["diagnostic_seed"],
                    starter,
                    inputs_sha,
                )
                if (
                    config["version"] >= 3
                    and spec["players"] == 2
                    and (
                        a["configurations"] != b["configurations"]
                        or a["behavior"] != b["behavior"]
                    )
                ):
                    raise ValueError(
                        "team specialist changed protected diagnostic heads-up behavior"
                    )
                diagnostic[spec["id"]] = {
                    **summary(clusters(a), clusters(b)),
                    "behavior": a["behavior"],
                    "decision_contexts": a["decision_contexts"],
                }
            primary["diagnostics"] = diagnostic
            primary["checkpoint"] = str(checkpoint)
            primary["training"] = training_summary(folder)
            individual[name] = primary
            for mode in (2, 4):
                matrices[recipe][mode].append(modes[mode])
        variants[recipe] = {"modes": {}, "diagnostics": {}}
        base_reports = [
            read(root / "discovery-baseline/primary" / f"native-{p}.json")
            for p in PARTNERS
        ]
        baseline_modes = {
            2: clusters(read(root / "discovery-baseline/primary/native-headsup.json")),
            4: np.stack([clusters(x) for x in base_reports]).mean(0),
        }
        for mode, target in targets(config).items():
            matrices[recipe][mode] = np.stack(matrices[recipe][mode])
            variants[recipe]["modes"][str(mode)] = summary(
                matrices[recipe][mode], baseline_modes[mode][None], True, target
            )
        scores = score_clusters(matrices[recipe], config)
        parent_scores = score_clusters(baseline_modes, config)
        variants[recipe].update(
            balanced_score=float(scores.mean()),
            balanced_gain=float((scores - parent_scores[None]).mean()),
            balanced_gain_ci95=crossed_interval(scores - parent_scores[None]),
        )
        for spec in config["diagnostic_specs"]:
            # Diagnostic clusters are disjoint from primary clusters and never
            # inserted into the native target or balanced score.
            values = np.stack(
                [
                    clusters(
                        read(
                            root
                            / "discovery"
                            / f"{recipe}-seed-{seed}"
                            / "diagnostic"
                            / f"{spec['id']}.json"
                        )
                    )
                    for seed in config["discovery_seeds"]
                ]
            )
            before = clusters(
                read(root / "discovery-baseline/diagnostic" / f"{spec['id']}.json")
            )
            variants[recipe]["diagnostics"][spec["id"]] = summary(
                values, before[None], True
            )
        configuration_strata = {}
        for spec in config["primary_specs"] + config["diagnostic_specs"]:
            family = "primary" if spec in config["primary_specs"] else "diagnostic"
            baseline = read(root / "discovery-baseline" / family / f"{spec['id']}.json")
            after = [
                read(
                    root
                    / "discovery"
                    / f"{recipe}-seed-{seed}"
                    / family
                    / f"{spec['id']}.json"
                )
                for seed in config["discovery_seeds"]
            ]
            configuration_strata[spec["id"]] = {"configurations": []}
            for cell in baseline["configurations"]:
                key = (cell["target"], cell["flor"])
                rows = [
                    next(
                        x
                        for x in report["configurations"]
                        if (x["target"], x["flor"]) == key
                    )
                    for report in after
                ]
                delta = (
                    np.asarray([x["pair_scores"] for x in rows])
                    - np.asarray(cell["pair_scores"])[None]
                )
                configuration_strata[spec["id"]]["configurations"].append(
                    {
                        "players": cell["players"],
                        "target": cell["target"],
                        "flor": cell["flor"],
                        "gain": float(delta.mean()),
                        "gain_ci95": crossed_interval(delta),
                    }
                )
        regressions = configuration_regressions(configuration_strata)
        variants[recipe]["configuration_regressions"] = regressions
        variants[recipe]["partners"] = {}
        for partner in PARTNERS:
            values = np.stack(
                [
                    clusters(
                        read(
                            root
                            / "discovery"
                            / f"{recipe}-seed-{seed}"
                            / "primary"
                            / f"native-{partner}.json"
                        )
                    )
                    for seed in config["discovery_seeds"]
                ]
            )
            before = clusters(
                read(root / "discovery-baseline/primary" / f"native-{partner}.json")
            )
            variants[recipe]["partners"][partner] = summary(values, before[None], True)
        native_safe = all(
            v["gain"] >= -0.02 for v in variants[recipe]["modes"].values()
        )
        diagnostic_safe = all(
            v["gain"] >= -0.02 for v in variants[recipe]["diagnostics"].values()
        )
        variants[recipe]["observed_guard_pass"] = (
            native_safe
            and diagnostic_safe
            and not regressions
            and all(v["gain"] >= -0.02 for v in variants[recipe]["partners"].values())
        )
    eligible = [
        name for name, value in variants.items() if value["observed_guard_pass"]
    ]
    chosen = max(
        eligible or list(variants),
        key=lambda name: selection_rank(variants[name], config),
    )
    effects = {}
    contrasts = config.get(
        "factorial_contrasts",
        {
            "native_exposure": {"N99": 0.5, "L99": -0.5, "N100": 0.5, "L100": -0.5},
            "lambda_one": {"L100": 0.5, "L99": -0.5, "N100": 0.5, "N99": -0.5},
            "interaction": {"N100": 1, "N99": -1, "L100": -1, "L99": 1},
        },
    )
    for mode in (2, 4):
        differences = {
            name: sum(
                weight * matrices[recipe][mode] for recipe, weight in weights.items()
            )
            for name, weights in contrasts.items()
        }
        effects[str(mode)] = {
            key: {"gain": float(value.mean()), "gain_ci95": crossed_interval(value)}
            for key, value in differences.items()
        }
    return {
        "selected_recipe": chosen,
        "exploratory_unguarded_continuation": not eligible,
        "variants": variants,
        "individual": individual,
        "effects": effects,
        "approved": False,
        "selection": (
            "both-target attainment, then weaker normalized mode, then 2v2, among mode/partner/configuration/diagnostic-preserving recipes"
            if config["version"] >= 2
            else "highest normalized balanced score"
        ),
        "mode_win_rate_targets": config["mode_win_rate_targets"],
        "limitations": [
            "three training seeds; exploratory unadjusted intervals",
            "diagnostics are separate from native acceptance target",
        ],
    }


def select_checkpoint(best, current, checkpoint, seed, budget, starter, config=None):
    config = config or {"version": 1}
    rank, before = (
        selection_rank(current, config),
        selection_rank(best["assessment"], config),
    )
    improved = rank_improves(rank, before)
    if current["observed_guard_pass"] and improved:
        return {
            "checkpoint": str(checkpoint),
            "sha256": file_hash(checkpoint),
            "assessment": current,
            "seed": seed,
            "origin": f"budget-{budget}",
            "trained": True,
            "selection_budget": budget,
        }
    return best


def verify_frozen_choices(root, config, selections, nominee):
    for name, expected in [
        ("nominee.json", nominee),
        *[
            (f"selection-{seed}.json", selections[str(seed)])
            for seed in config["confirmation_seeds"]
        ],
    ]:
        path = root / name
        receipt = path.with_suffix(".sha256")
        if (
            not receipt.is_file()
            or receipt.read_text().strip() != file_hash(path)
            or read(path) != expected
        ):
            raise ValueError(
                "final confirmation requires unchanged frozen selections and nominee"
            )
        checkpoint = Path(expected["checkpoint"]).resolve()
        if (
            not checkpoint.is_relative_to(root.resolve())
            or file_hash(checkpoint) != expected["sha256"]
        ):
            raise ValueError("frozen selection checkpoint changed or outside campaign")


def final_comparison(root, config, recipe_choice, selections, nominee):
    root = Path(root)
    inputs_sha = file_hash(root / "inputs/manifest.json")
    starter = root / "inputs/legacy-champion-lstm16.pt"
    verify_frozen_choices(root, config, selections, nominee)
    baseline = root / "final/starter"
    specs = config["primary_specs"] + config["diagnostic_specs"]
    baseline_assessment, _ = assessment(
        baseline,
        baseline,
        specs,
        config["final_pairs"],
        config["final_seed"],
        starter,
        starter,
        inputs_sha,
        config,
    )
    individual, matrices = {}, {2: [], 4: []}
    for seed in config["confirmation_seeds"]:
        selected = selections[str(seed)]
        label = f"seed-{seed}"
        checkpoint = Path(selected["checkpoint"])
        if selected["sha256"] != file_hash(checkpoint):
            raise ValueError("frozen selection checkpoint changed")
        measured, modes = assessment(
            root / "final" / label,
            baseline,
            specs,
            config["final_pairs"],
            config["final_seed"],
            checkpoint,
            starter,
            inputs_sha,
            config,
        )
        measured.update(
            checkpoint=str(checkpoint),
            origin=selected["origin"],
            trained=selected["trained"],
            selection_budget=selected["selection_budget"],
            training=training_summary(root / "confirmation" / label),
        )
        individual[label] = measured
        for mode in (2, 4):
            matrices[mode].append(modes[mode])
    nominee_label = nominee["evaluation_label"]
    chosen = (
        baseline_assessment if nominee_label == "starter" else individual[nominee_label]
    )
    train_count = sum(row["trained"] for row in individual.values())
    average = {}
    for mode, target in targets(config).items():
        # Average selected outcomes can include repeated starter policies. State
        # that explicitly instead of claiming independent trained replication.
        values = np.stack(matrices[mode])
        before = (
            clusters(read(baseline / "native-headsup.json"))
            if mode == 2
            else np.stack(
                [clusters(read(baseline / f"native-{p}.json")) for p in PARTNERS]
            ).mean(0)
        )
        average[str(mode)] = summary(values, before[None], True, target)
    retain = (
        nominee["trained"]
        and chosen["observed_guard_pass"]
        and chosen["balanced_gain_ci95"][0] > 0
        and chosen["modes"]["2"]["gain_ci95"][0] >= -0.02
        and chosen["modes"]["4"]["gain_ci95"][0] >= -0.02
    )
    return {
        "protocol": f"target-improvement-v{config['version']}",
        "recipe_choice": recipe_choice,
        "selections": selections,
        "nominee": nominee,
        "baseline": baseline_assessment,
        "individual": individual,
        "selected_outcome_average": average,
        "trained_selected_count": train_count,
        "fallback_count": len(individual) - train_count,
        "selected_average_scope": "includes explicit starter fallbacks; not an estimate from three independent trained policies",
        "nominee_confirmation": chosen,
        "strength_targets": chosen["modes"],
        "acceptable_strength": chosen["acceptable_strength"],
        "retain_nominee_for_review": bool(retain),
        "retained_checkpoint": nominee["checkpoint"] if retain else str(starter),
        "approved": False,
        "release_holdouts_used": False,
        "confirmation_planned_checkpoints": config["confirmation_checkpoints"],
        "confirmation_completed_checkpoints": confirmation_budgets(root, config),
        "selection_checkpoint_assessments": {
            str(seed): {
                str(budget): read(
                    root / "confirmation" / f"seed-{seed}" / f"assessment-{budget}.json"
                )
                for budget in confirmation_budgets(root, config)
            }
            for seed in config["confirmation_seeds"]
        },
        "limitations": [
            "three fresh training seeds and exploratory multiple comparisons",
            f"selection{config['selection_seed']} is disjoint from frozen-nominee confirmation{config['final_seed']}",
            f"native target point estimates {config['mode_win_rate_targets']} and confidence above chance; intervals wholly above targets are a stronger claim",
            "synthetic partners do not establish human compatibility",
            "final releaseholdout, communication/bluffing, browser and human-review gates remain",
        ],
    }
