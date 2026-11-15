"""Paired development comparisons of fresh GRU/LSTM runs and frozen champion."""

import json

import numpy as np

from .controlled_compare import crossed_interval
from .cooperation_refusal_compare import PARTNERS, clusters, summary
from .recurrent import file_hash
from .target_compare import assessment


def compare(root, config, phase):
    specs = config["primary_specs"] + config["diagnostic_specs"]
    inputs_sha = file_hash(root / "inputs/manifest.json")
    baseline = root / "inputs/legacy-champion-lstm16.pt"
    before = root / phase / "reference"
    pairs, seed = config[phase + "_pairs"], config[phase + "_seed"]
    individuals = {}
    matrices = {}
    variants = {}
    for cell in config["cells"]:
        matrices[cell] = {2: [], 4: []}
        for train_seed in config["seeds"]:
            name = f"{cell}-{train_seed}"
            folder = root / phase / name
            cp = root / "trials" / name / "candidate.pt"
            value, modes = assessment(
                folder,
                before,
                specs,
                pairs,
                seed,
                cp,
                baseline,
                inputs_sha,
                {"version": 2, "mode_win_rate_targets": {"1v1": 0.7, "2v2": 0.6}},
            )
            value["checkpoint"] = str(cp)
            value["initialization"] = "random, no warm start"
            value["training"] = json.loads(
                (root / "trials" / name / "status.json").read_text()
            )
            individuals[name] = value
            for mode in (2, 4):
                matrices[cell][mode].append(modes[mode])
        variants[cell] = {"modes": {}, "diagnostics": {}}
        for mode, target in ((2, 0.7), (4, 0.6)):
            ids = ["native-headsup"] if mode == 2 else ["native-" + p for p in PARTNERS]
            base = np.stack(
                [
                    clusters(json.loads((before / (i + ".json")).read_text()))
                    for i in ids
                ]
            ).mean(0)
            matrices[cell][mode] = np.stack(matrices[cell][mode])
            variants[cell]["modes"][str(mode)] = summary(
                matrices[cell][mode], base[None], True, target
            )
        for spec in specs:
            values = np.stack(
                [
                    clusters(
                        json.loads(
                            (
                                root / phase / f"{cell}-{s}" / (spec["id"] + ".json")
                            ).read_text()
                        )
                    )
                    for s in config["seeds"]
                ]
            )
            base = clusters(json.loads((before / (spec["id"] + ".json")).read_text()))
            variants[cell]["diagnostics"][spec["id"]] = summary(
                values, base[None], True
            )
        variants[cell]["acceptable_strength"] = all(
            r["target_met"] and r["confidence_above_chance"]
            for r in variants[cell]["modes"].values()
        )
    effects = {
        str(mode): {
            "lstm_minus_gru": float(
                (matrices["lstm"][mode] - matrices["gru"][mode]).mean()
            ),
            "crossed_ci95": crossed_interval(
                matrices["lstm"][mode] - matrices["gru"][mode]
            ),
        }
        for mode in (2, 4)
    }
    return {
        "phase": phase,
        "individuals": individuals,
        "variants": variants,
        "contrasts": effects,
        "scope": "paired development deals; crossed training-seed/deal bootstrap; unadjusted multiple comparisons; no release approval",
        "approved": False,
    }


def choose(report):
    def key(row):
        m = row["modes"]
        return (
            row["observed_guard_pass"] and row["acceptable_strength"],
            row["observed_guard_pass"],
            min(m["2"]["win_rate"] / 0.7, m["4"]["win_rate"] / 0.6),
            m["4"]["win_rate"],
        )

    name = max(report["individuals"], key=lambda n: key(report["individuals"][n]))
    safe = report["individuals"][name]["observed_guard_pass"]
    return {
        "exploratory_nominee": name,
        "retained_for_review": name if safe else "reference",
        "starter_fallback": not safe,
        "approved": False,
        "selection_before_independent_development": True,
    }
