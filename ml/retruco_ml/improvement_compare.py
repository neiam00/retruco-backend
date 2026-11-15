"""Paired, opponent-balanced development comparisons and independent confirmation."""

import json

import numpy as np

from .controlled_compare import crossed_interval
from .statistics import interval


def read(path):
    return json.loads(path.read_text())


def clusters(report, field="pair_scores"):
    return np.asarray(
        [row[field] for row in report["configurations"]], dtype=np.float64
    ).mean(0)


def bundle(folder, specs, field="pair_scores"):
    weighted, weight = None, 0.0
    for spec in specs:
        if spec.get("weight", 0) <= 0:
            continue
        report = read(folder / f"{spec['id']}.json")
        value = clusters(report, field) * spec["weight"]
        weighted = value if weighted is None else weighted + value
        weight += spec["weight"]
    if weighted is None or not weight:
        raise ValueError("empty primary evaluation bundle")
    return weighted / weight


def assess(folder, baseline, specs):
    current, before = bundle(folder, specs), bundle(baseline, specs)
    gain = current - before
    strata = {}
    safe = True
    for spec in specs:
        candidate, parent = (
            read(folder / f"{spec['id']}.json"),
            read(baseline / f"{spec['id']}.json"),
        )
        if (
            candidate["seed"] != parent["seed"]
            or candidate["inputs_sha256"] != parent["inputs_sha256"]
        ):
            raise ValueError("unpaired improvement reports")
        if [
            (x["players"], x["target"], x["flor"], x["pairs"])
            for x in candidate["configurations"]
        ] != [
            (x["players"], x["target"], x["flor"], x["pairs"])
            for x in parent["configurations"]
        ]:
            raise ValueError("different evaluation coverage")
        changes = clusters(candidate) - clusters(parent)
        gain_ci = interval(changes)
        strata[spec["id"]] = {
            "win_rate": float(clusters(candidate).mean()),
            "gain": float(changes.mean()),
            "gain_ci95": gain_ci,
        }
        if spec.get("guard", False):
            safe &= changes.mean() >= -0.02
    points = bundle(folder, specs, "pair_net_points_per_hand")
    return {
        "win_rate": float(current.mean()),
        "gain": float(gain.mean()),
        "gain_ci95": interval(gain),
        "net_points_per_hand": float(points.mean()),
        "observed_guard_pass": bool(safe),
        "strata": strata,
        "approved": False,
    }


def aggregate(folders, baseline, specs):
    values = np.stack([bundle(folder, specs) for folder in folders])
    parent = bundle(baseline, specs)
    delta = values - parent[None]
    return {
        "win_rate": float(values.mean()),
        "per_seed_win_rates": values.mean(1).tolist(),
        "gain": float(delta.mean()),
        "gain_ci95": crossed_interval(delta),
        "method": "crossed training-seed and paired-deal-seed bootstrap; exploratory",
    }


def discovery_comparison(root, config, source_name):
    variants = {}
    matrices = {}
    for name in config["recipes"]:
        folders = [
            root / "discovery" / f"{name}-seed-{seed}" / "validation"
            for seed in config["discovery_seeds"]
        ]
        variants[name] = aggregate(
            folders, root / "discovery-baseline", config["screen_specs"]
        )
        matrices[name] = np.stack(
            [bundle(folder, config["screen_specs"]) for folder in folders]
        )
    u95, u99, r95, r99 = (matrices[name] for name in ("U95", "U99", "R95", "R99"))
    contrasts = {
        "longer_credit": ((u99 - u95) + (r99 - r95)) / 2,
        "reactive_population": ((r95 - u95) + (r99 - u99)) / 2,
        "interaction": r99 - r95 - u99 + u95,
    }
    effects = {
        name: {"gain": float(value.mean()), "gain_ci95": crossed_interval(value)}
        for name, value in contrasts.items()
    }
    chosen = max(variants, key=lambda name: variants[name]["win_rate"])
    return {
        "variants": variants,
        "effects": effects,
        "selected_recipe": chosen,
        "starter": source_name,
        "selection": "highest fixed-endpoint mean opponent-balanced development score; not release approval",
        "approved": False,
    }


def final_comparison(root, config, source_choice, recipe_choice, selections, nominee):
    specs = config["final_specs"]
    baseline = root / "confirmation-evaluation/d-parent"
    starter = root / "confirmation-evaluation/starter"
    folders = [
        root / "confirmation-evaluation" / f"seed-{seed}"
        for seed in config["confirmation_seeds"]
    ]
    report = {
        "approved": False,
        "source_choice": source_choice,
        "recipe_choice": recipe_choice,
        "nominee": nominee,
        "selections": selections,
        "confirmation": aggregate(folders, baseline, specs),
        "starter_confirmation": assess(starter, baseline, specs),
        "individual": {
            str(seed): assess(folder, baseline, specs)
            for seed, folder in zip(config["confirmation_seeds"], folders)
        },
        "configuration_results": {},
        "signal_ablations": {},
        "release_holdouts_used": False,
        "scope": "development discovery followed by frozen-nominee independent development confirmation; no model approval",
        "limitations": [
            "three fresh training seeds",
            "synthetic reactive humans are proxies",
            "exploratory intervals without multiple-comparison correction",
            "candidate and recipe chosen on separate development deals before confirmation",
        ],
    }
    delta_from_starter = (
        np.stack([bundle(folder, specs) for folder in folders])
        - bundle(starter, specs)[None]
    )
    report["confirmation_vs_starter"] = {
        "gain": float(delta_from_starter.mean()),
        "gain_ci95": crossed_interval(delta_from_starter),
    }
    for spec in specs:
        parent = read(baseline / f"{spec['id']}.json")
        candidates = [read(folder / f"{spec['id']}.json") for folder in folders]
        report["configuration_results"][spec["id"]] = []
        for index, row in enumerate(parent["configurations"]):
            values = np.stack(
                [
                    candidate["configurations"][index]["pair_scores"]
                    for candidate in candidates
                ]
            )
            delta = values - np.asarray(row["pair_scores"])[None]
            report["configuration_results"][spec["id"]].append(
                {
                    "players": row["players"],
                    "target": row["target"],
                    "flor": row["flor"],
                    "win_rate": float(values.mean()),
                    "gain": float(delta.mean()),
                    "gain_ci95": crossed_interval(delta),
                }
            )
    for label in ("d-parent", "nominee"):
        folder = root / "ablations" / label
        for family in ("native", "reactive"):
            for partner in ("literal-v2", "trusting-v2"):
                key = f"{family}-{partner}"
                normal = read(folder / f"{key}-normal.json")
                report["signal_ablations"][f"{label}/{key}"] = {}
                for mode in ("no-send", "no-receive"):
                    ablated = read(folder / f"{key}-{mode}.json")
                    delta = clusters(normal) - clusters(ablated)
                    report["signal_ablations"][f"{label}/{key}"][mode] = {
                        "benefit": float(delta.mean()),
                        "benefit_ci95": interval(delta),
                    }
    report["card_selection_ablations"] = {}
    for label in ("d-parent", "nominee"):
        for family in ("native", "reactive"):
            for partner in ("literal-v2", "trusting-v2"):
                key = f"{family}-{partner}"
                normal = read(root / "card-ablations" / label / f"{key}-policy.json")
                highest = read(root / "card-ablations" / label / f"{key}-highest.json")
                delta = clusters(normal) - clusters(highest)
                points = clusters(normal, "pair_net_points_per_hand") - clusters(
                    highest, "pair_net_points_per_hand"
                )
                report["card_selection_ablations"][f"{label}/{key}"] = {
                    "win_benefit": float(delta.mean()),
                    "win_benefit_ci95": interval(delta),
                    "net_points_per_hand_benefit": float(points.mean()),
                    "net_points_per_hand_benefit_ci95": interval(points),
                }
    report["card_ablation_scope"] = (
        "learned card selection versus highest legal compatible card, preserving chosen bid/declaration; paired separate development deals"
    )
    primary_nominee = root / "confirmation-evaluation" / nominee["evaluation_label"]
    report["nominee_confirmation"] = assess(primary_nominee, baseline, specs)
    report["behavior_results"] = {}
    for label, folder in [("d-parent", baseline), ("nominee", primary_nominee)]:
        report["behavior_results"][label] = {}
        for spec in specs:
            result = read(folder / f"{spec['id']}.json")
            report["behavior_results"][label][spec["id"]] = {
                "behavior": result["behavior"],
                "decision_contexts": result["decision_contexts"],
                "net_points_per_hand": float(
                    clusters(result, "pair_net_points_per_hand").mean()
                ),
            }
    report["retain_nominee_for_review"] = report["nominee_confirmation"]["gain_ci95"][
        0
    ] > 0 and all(
        value["gain_ci95"][0] >= -0.03
        for key, value in report["nominee_confirmation"]["strata"].items()
        if next(spec for spec in specs if spec["id"] == key).get("guard", False)
    )
    report["retained_checkpoint"] = (
        nominee["checkpoint"]
        if report["retain_nominee_for_review"]
        else str(
            root
            / "inputs"
            / read(root / "inputs/manifest.json")["parents"]["101"]["file"]
        )
    )
    if config.get("mode_win_rate_targets"):
        from .strength_targets import summarize

        expected = {"1v1": 0.70, "2v2": 0.65}
        if config["mode_win_rate_targets"] != expected:
            raise ValueError("unexpected user strength targets")
        rows = read(primary_nominee / "native-headsup.json")["configurations"][:]
        for partner in ("literal-v2", "reference-human", "trusting-v2", "cautious-v2"):
            rows += read(primary_nominee / f"native-{partner}.json")["configurations"]
        report["strength_targets"] = {
            "opponent": "native",
            "modes": summarize(rows),
            "two_player_partner": None,
            "four_player_partners": [
                "literal-v2",
                "reference-human",
                "trusting-v2",
                "cautious-v2",
            ],
            "aggregation": "equal paired-seed average across both score targets/flor settings; equal four-player partner weights",
            "scope": "development acceptance measurement; release requires untouched holdout and all remaining gates",
        }
        report["acceptable_strength"] = all(
            v["target_met"] and v["confidence_above_chance"]
            for v in report["strength_targets"]["modes"].values()
        )

    return report
