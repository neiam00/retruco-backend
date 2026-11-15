"""Crossed training-seed/deal-seed comparisons for a fixed factorial screen."""

import argparse
import json
from pathlib import Path

import numpy as np

from .controlled_train import atomic_json

PRIMARY = ("reference-human", "literal-v2", "trusting-v2", "cautious-v2")


def read(path):
    return json.loads(path.read_text())


def cluster_scores(report, players=4):
    rows = [row for row in report["configurations"] if row["players"] == players]
    if not rows:
        raise ValueError("missing rules configurations")
    expected = {
        (players, target, flor) for target in (12, 24) for flor in (False, True)
    }
    if {(row["players"], row["target"], row["flor"]) for row in rows} != expected:
        raise ValueError("incomplete rules coverage")
    if any(len(row["pair_scores"]) != row["pairs"] for row in rows):
        raise ValueError("invalid paired scores")
    return np.asarray([row["pair_scores"] for row in rows], dtype=np.float64).mean(0)


def crossed_interval(values, draws=5000):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.size:
        raise ValueError("requires training-seed x deal-seed matrix")
    rng = np.random.default_rng(3041)
    rows = rng.integers(values.shape[0], size=(draws, values.shape[0]))
    columns = rng.integers(values.shape[1], size=(draws, values.shape[1]))
    means = values[rows[:, :, None], columns[:, None, :]].mean((1, 2))
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def summarize(values, baseline):
    delta = values - baseline[None, :]
    return {
        "win_rate": float(values.mean()),
        "per_seed_win_rates": values.mean(1).tolist(),
        "gain": float(delta.mean()),
        "gain_ci95": crossed_interval(delta),
        "method": "crossed training-seed and paired-deal-seed bootstrap; exploratory, three seeds",
    }


def compare(root):
    config = read(root / "configuration.json")
    seeds = config["seeds"]
    partners = config["primary_partners"]
    parent = {p: read(root / "baseline" / f"{p}-normal.json") for p in partners}
    expected_seed = config["validation_seed"]
    parent_sha = read(root / "inputs/manifest.json")["parents"]["101"]["sha256"]
    for report in parent.values():
        if report["seed"] != expected_seed or report["checkpoint_sha256"] != parent_sha:
            raise ValueError("unpaired baseline or wrong original parent")
    metrics, matrices, individual = {}, {}, []
    for arm in config["arms"]:
        reports = [
            {
                p: read(root / f"{arm}-seed-{seed}" / "validation" / f"{p}-normal.json")
                for p in partners
            }
            for seed in seeds
        ]
        for row in reports:
            for p, report in row.items():
                if (
                    report["seed"] != expected_seed
                    or report["partner"] != p
                    or report["signals"] != "normal"
                    or report["reference_sha256"] != parent[p]["reference_sha256"]
                ):
                    raise ValueError(
                        "comparison requires identical evaluation protocol"
                    )
                if [
                    (c["players"], c["target"], c["flor"], c["pairs"])
                    for c in report["configurations"]
                ] != [
                    (c["players"], c["target"], c["flor"], c["pairs"])
                    for c in parent[p]["configurations"]
                ]:
                    raise ValueError("comparison coverage differs")
        values = {
            p: np.stack([cluster_scores(row[p]) for row in reports]) for p in PRIMARY
        }
        values["headsup"] = np.stack(
            [cluster_scores(row["reference-cpu"], 2) for row in reports]
        )
        values["reference-cpu"] = np.stack(
            [cluster_scores(row["reference-cpu"]) for row in reports]
        )
        values["cooperation"] = np.stack([values[p] for p in PRIMARY]).mean(0)
        baseline = {p: cluster_scores(parent[p]) for p in PRIMARY}
        baseline.update(
            headsup=cluster_scores(parent["reference-cpu"], 2),
            **{"reference-cpu": cluster_scores(parent["reference-cpu"])},
        )
        baseline["cooperation"] = np.stack([baseline[p] for p in PRIMARY]).mean(0)
        metrics[arm] = {p: summarize(value, baseline[p]) for p, value in values.items()}
        per_configuration = {}
        for partner in partners:
            per_configuration[partner] = []
            for index, row in enumerate(parent[partner]["configurations"]):
                matched = np.stack(
                    [
                        item[partner]["configurations"][index]["pair_scores"]
                        for item in reports
                    ]
                )
                per_configuration[partner].append(
                    {
                        "players": row["players"],
                        "target": row["target"],
                        "flor": row["flor"],
                        **summarize(matched, np.asarray(row["pair_scores"])),
                    }
                )
        metrics[arm]["configurations"] = per_configuration
        matrices[arm] = values
        for row, seed in zip(reports, seeds, strict=True):
            folder = root / f"{arm}-seed-{seed}"
            state = read(folder / "status.json")
            logs = [
                json.loads(line)
                for line in (folder / "metrics.jsonl").read_text().splitlines()
            ]
            training = [entry for entry in logs if entry["phase"] == "training"]
            elapsed = sum(
                entry["collect_seconds"] + entry["optimize_seconds"]
                for entry in training
            )
            individual.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "choices": state["choices"],
                    "decisions": state["decisions"],
                    "update": state["update"],
                    "cooperation": float(
                        np.stack([cluster_scores(row[p]) for p in PRIMARY]).mean()
                    ),
                    "headsup": float(cluster_scores(row["reference-cpu"], 2).mean()),
                    "choices_per_second": sum(entry["choices"] for entry in training)
                    / elapsed,
                    "warmups": state["warmups"],
                    "checkpoint": str(folder / "checkpoint.pt"),
                    "last_diagnostics": training[-1]["diagnostics"],
                    "behavior": {p: row[p]["behavior"] for p in partners},
                    "decision_contexts": {
                        p: row[p]["decision_contexts"] for p in partners
                    },
                }
            )
    effects = {}
    if set(matrices) == {"A", "B", "C", "D"}:
        for metric in ("cooperation", "headsup", *PRIMARY):
            a, b, c, d = (matrices[k][metric] for k in "ABCD")
            changes = {
                "stability": ((b - a) + (d - c)) / 2,
                "choice_credit": ((c - a) + (d - b)) / 2,
                "interaction": d - b - c + a,
                "combined_vs_control": d - a,
            }
            effects[metric] = {
                key: {"gain": float(delta.mean()), "gain_ci95": crossed_interval(delta)}
                for key, delta in changes.items()
            }
    best = max(metrics, key=lambda arm: metrics[arm]["cooperation"]["win_rate"])
    return {
        "approved": False,
        "parent_sha256": parent_sha,
        "configuration": config,
        "arms": metrics,
        "individual_runs": individual,
        "effects": effects,
        "exploratory_best_arm": best,
        "scope": "short factorial development screen; no model release approval",
        "limitations": [
            "three training seeds",
            "exploratory unadjusted intervals and multiple comparisons",
            "fixed meaningful-choice budget; complete-match overshoot retained",
            "synthetic partners do not establish human compatibility",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = compare(args.root.resolve())
    atomic_json(args.root / "comparison.json", report)
    print(
        json.dumps(
            {"best_observed_arm": report["exploratory_best_arm"], "approved": False}
        )
    )


if __name__ == "__main__":
    main()
