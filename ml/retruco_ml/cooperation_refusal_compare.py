"""Paired fixed-endpoint comparisons; never substitute parent for a trial."""

import json
from pathlib import Path

import numpy as np

from .controlled_compare import crossed_interval
from .recurrent import file_hash
from .statistics import interval

PARTNERS = ("literal-v2", "trusting-v2", "cautious-v2", "reference-human")


def read(path):
    return json.loads(Path(path).read_text())


def validate_report(
    report, spec, pairs, seed, checkpoint_sha, inputs_sha, signals="normal"
):
    expected = {
        (spec["players"], target, flor) for target in (12, 24) for flor in (False, True)
    }
    rows = report.get("configurations", [])
    actual = [(c["players"], c["target"], c["flor"]) for c in rows]
    if (
        report.get("protocol") != "improvement-evaluation-v1"
        or len(actual) != 4
        or set(actual) != expected
        or report.get("seed") != seed
        or report.get("pairs") != pairs
        or report.get("partner") != spec["partner"]
        or report.get("opponent") != spec["opponent"]
        or report.get("signals") != signals
        or report.get("card_policy") != "policy"
        or report.get("checkpoint_sha256") != checkpoint_sha
        or report.get("inputs_sha256") != inputs_sha
    ):
        raise ValueError("evaluation protocol, provenance or coverage differs")
    for row in rows:
        values = row.get("pair_scores", [])
        points = row.get("pair_net_points_per_hand", [])
        if (
            row["pairs"] != pairs
            or len(values) != pairs
            or len(points) != pairs
            or row.get("matches") != pairs * spec["players"]
            or not all(
                isinstance(x, (int, float)) and np.isfinite(x) and 0 <= x <= 1
                for x in values
            )
            or not all(isinstance(x, (int, float)) and np.isfinite(x) for x in points)
            or not isinstance(row.get("win_rate"), (int, float))
            or not np.isfinite(row["win_rate"])
            or abs(row["win_rate"] - np.mean(values)) > 1e-12
        ):
            raise ValueError("invalid paired evaluation values")


def clusters(report, field="pair_scores"):
    return np.asarray(
        [row[field] for row in report["configurations"]], dtype=np.float64
    ).mean(0)


def mode_clusters(folder, mode):
    if mode == 2:
        return clusters(read(folder / "native-headsup.json"))
    return np.stack(
        [clusters(read(folder / f"native-{p}.json")) for p in PARTNERS]
    ).mean(0)


def summary(values, baseline, crossed=False, target=None):
    values = np.asarray(values, dtype=np.float64)
    gain = values - baseline
    ci = crossed_interval if crossed else interval
    result = {
        "win_rate": float(values.mean()),
        "win_ci95": ci(values),
        "gain": float(gain.mean()),
        "gain_ci95": ci(gain),
    }
    if crossed:
        result["per_seed_win_rates"] = values.mean(1).tolist()
    if target is not None:
        result.update(
            target_win_rate=target,
            target_met=result["win_rate"] >= target,
            confidence_above_chance=result["win_ci95"][0] > 0.5,
        )
    return result


def compare(root, config):
    root = Path(root)
    primary = root / "baseline/primary"
    inputs_sha = file_hash(root / "inputs/manifest.json")
    starter = root / "inputs/legacy-champion-lstm16.pt"
    starter_sha = file_hash(starter)
    specs = config["primary_specs"] + config["diagnostic_specs"]

    def verified(path, spec, family, digest):
        receipt = path.with_suffix(".sha256")
        if not receipt.exists() or receipt.read_text().strip() != file_hash(path):
            raise ValueError("changed or missing comparison receipt")
        report = read(path)
        validate_report(
            report,
            spec,
            config[family + "_pairs"],
            config[family + "_seed"],
            digest,
            inputs_sha,
        )
        return report

    for spec in specs:
        family = "primary" if spec in config["primary_specs"] else "diagnostic"
        verified(
            root / "baseline" / family / f"{spec['id']}.json", spec, family, starter_sha
        )
    baseline = {mode: mode_clusters(primary, mode) for mode in (2, 4)}
    individual, variants, matrices, configurations = {}, {}, {}, {}
    for arm in config["arms"]:
        matrices[arm] = {mode: [] for mode in (2, 4)}
        rows = []
        for seed in config["seeds"]:
            name = f"{arm}-seed-{seed}"
            folder = root / "trials" / name
            candidate = folder / "candidate.pt"
            receipt = candidate.with_suffix(".sha256")
            if not receipt.exists() or receipt.read_text().strip() != file_hash(
                candidate
            ):
                raise ValueError("changed or missing endpoint receipt")
            candidate_sha = file_hash(candidate)
            state = read(folder / "status.json")
            if state["status"] != "complete" or state["choices"] < config["choices"]:
                raise ValueError("comparison requires all completed endpoint trials")
            modes = {}
            for mode, target in ((2, 0.70), (4, 0.65)):
                values = mode_clusters(folder / "primary", mode)
                matrices[arm][mode].append(values)
                modes[str(mode)] = summary(values, baseline[mode], target=target)
            strata, contexts = {}, {}
            safe = True
            for spec in specs:
                family = "primary" if spec in config["primary_specs"] else "diagnostic"
                current = verified(
                    folder / family / f"{spec['id']}.json", spec, family, candidate_sha
                )
                before = verified(
                    root / "baseline" / family / f"{spec['id']}.json",
                    spec,
                    family,
                    starter_sha,
                )
                gain = clusters(current) - clusters(before)
                strata[spec["id"]] = summary(clusters(current), clusters(before))
                strata[spec["id"]]["net_points_per_hand"] = float(
                    clusters(current, "pair_net_points_per_hand").mean()
                )
                safe &= gain.mean() >= -0.02
                contexts[spec["id"]] = {
                    "behavior": current["behavior"],
                    "decision_contexts": current["decision_contexts"],
                }
                previous = {
                    (c["target"], c["flor"]): c for c in before["configurations"]
                }
                configurations[name + "/" + spec["id"]] = []
                for c in current["configurations"]:
                    changes = (
                        np.asarray(c["pair_scores"])
                        - previous[c["target"], c["flor"]]["pair_scores"]
                    )
                    configurations[name + "/" + spec["id"]].append(
                        {
                            "players": c["players"],
                            "target": c["target"],
                            "flor": c["flor"],
                            "win_rate": c["win_rate"],
                            "gain": float(changes.mean()),
                            "gain_ci95": interval(changes),
                        }
                    )
            logs = [
                json.loads(line)
                for line in (folder / "metrics.jsonl").read_text().splitlines()
            ]
            training = [x for x in logs if x["phase"] == "training"]
            elapsed = sum(
                x["collect_seconds"] + x["optimize_seconds"] for x in training
            )
            row = {
                "arm": arm,
                "seed": seed,
                "modes": modes,
                "strata": strata,
                "contexts": contexts,
                "observed_all_strata_guard_pass": bool(safe),
                "choices": state["choices"],
                "update": state["update"],
                "warmups": state["warmups"],
                "choices_per_second": sum(x["choices"] for x in training) / elapsed,
                "replay_max_logp_error": max(
                    x["diagnostics"]["replay_identity_max_logp_error"] for x in training
                ),
                "last_policy_diagnostics": {
                    key: training[-1]["diagnostics"].get(key)
                    for key in (
                        "anchor_loss_mean",
                        "anchor_coefficients",
                        "response",
                        "response_choices",
                        "signal_choices",
                        "all_choices",
                    )
                },
                "checkpoint": str(folder / "candidate.pt"),
                "latest_checkpoint": str(folder / "checkpoint.pt"),
            }
            individual[name] = row
            rows.append(row)
        variants[arm] = {
            "modes": {},
            "guard_pass_seeds": sum(
                row["observed_all_strata_guard_pass"] for row in rows
            ),
        }
        for mode, target in ((2, 0.70), (4, 0.65)):
            matrices[arm][mode] = np.stack(matrices[arm][mode])
            variants[arm]["modes"][str(mode)] = summary(
                matrices[arm][mode], baseline[mode][None], True, target
            )
    effects = {}
    for mode in (2, 4):
        a, b, c, d = (matrices[x][mode] for x in "ABCD")
        contrasts = {
            "cooperation_population_bundle": ((b - a) + (d - c)) / 2,
            "targeted_anchor_bundle": ((c - a) + (d - b)) / 2,
            "interaction": d - b - c + a,
            "combined_vs_control": d - a,
        }
        effects[str(mode)] = {
            name: {"gain": float(v.mean()), "gain_ci95": crossed_interval(v)}
            for name, v in contrasts.items()
        }
    exploratory = max(variants, key=lambda arm: variants[arm]["modes"]["4"]["win_rate"])
    acceptable = {
        name: all(
            v["target_met"] and v["confidence_above_chance"]
            for v in row["modes"].values()
        )
        for name, row in individual.items()
    }
    eligible = [
        name
        for name, row in individual.items()
        if row["observed_all_strata_guard_pass"]
        and row["modes"]["4"]["gain_ci95"][0] > 0
        and row["modes"]["2"]["gain_ci95"][0] >= -0.02
    ]
    retained = (
        max(eligible, key=lambda name: individual[name]["modes"]["4"]["win_rate"])
        if eligible
        else "starter"
    )
    return {
        "protocol": "cooperation-refusal-v1",
        "baseline": {
            str(mode): summary(values, values, target={2: 0.70, 4: 0.65}[mode])
            for mode, values in baseline.items()
        },
        "variants": variants,
        "individual": individual,
        "effects": effects,
        "configuration_results": configurations,
        "exploratory_best_arm": exploratory,
        "acceptable_strength": acceptable,
        "retained": retained,
        "retained_checkpoint": individual[retained]["checkpoint"]
        if retained != "starter"
        else str(root / "inputs/legacy-champion-lstm16.pt"),
        "approved": False,
        "release_holdouts_used": False,
        "aggregation": "native benchmark; equal four human-role partners in2v2, target12/24/floroff/on, paired deal clusters",
        "limitations": [
            f"{len(config['seeds'])} training seeds; unadjusted exploratory comparisons",
            "fixed endpoints; no fallback copied into unsuccessful seeds",
            "population and anchor changes are bundles, not isolated causal mechanisms",
            "synthetic partners do not establish human compatibility",
            "acceptance rates are descriptive; more accepts alone do not demonstrate improvement",
        ],
    }


def ablation_comparison(root, labels):
    result = {}
    for label in labels:
        folder = Path(root) / "ablations" / label
        normal = read(folder / "normal.json")
        result[label] = {}
        for mode in ("no-send", "no-receive"):
            gain = clusters(normal) - clusters(read(folder / f"{mode}.json"))
            result[label][mode] = {
                "benefit": float(gain.mean()),
                "benefit_ci95": interval(gain),
            }
    return result
