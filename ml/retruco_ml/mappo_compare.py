"""Report factorial contrasts, clustering both training seeds and shared deals."""

import argparse
import json
from pathlib import Path

import numpy as np

from .recurrent import file_hash

CONTRASTS = {
    "memory_local": {"B": 1, "A": -1},
    "critic_feedforward": {"C": 1, "A": -1},
    "memory_centralized": {"D": 1, "C": -1},
    "critic_recurrent": {"D": 1, "B": -1},
    "interaction": {"D": 1, "C": -1, "B": -1, "A": 1},
}


def contrast(values, seed=917):
    x = np.asarray(values, np.float64)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("three training seeds and paired deals required")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(10000):
        rows = rng.integers(x.shape[0], size=x.shape[0])
        columns = rng.integers(x.shape[1], size=x.shape[1])
        means.append(float(x[np.ix_(rows, columns)].mean()))
    return {
        "difference": float(x.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "per_training_seed": x.mean(1).tolist(),
        "method": "crossed training-seed/deal-seed bootstrap; only three training seeds",
    }


def summarize(root, budget="decisions"):
    config = json.loads((root / "configuration.json").read_text())
    result = {"approved": False, "budget": budget, "comparisons": {}}
    if len(config["seeds"]) < 3:
        raise ValueError("smoke campaign is not strength evidence")
    for partner in config["validation_partners"]:
        reports = {}
        for arm in config["arms"]:
            for seed in config["seeds"]:
                folder = root / f"{arm}-seed-{seed}"
                suffix = "" if budget == "decisions" else "-wall"
                report = json.loads(
                    (folder / f"validation-{partner}{suffix}.json").read_text()
                )
                checkpoint = (
                    folder / "checkpoint.pt"
                    if budget == "decisions"
                    else Path(
                        json.loads((root / "wall-checkpoints.json").read_text())[
                            f"{arm}-seed-{seed}"
                        ]["checkpoint"]
                    )
                )
                if report["checkpoint_sha256"] != file_hash(checkpoint):
                    raise ValueError("validation checkpoint changed")
                if (
                    report["seed"] != config["validation_seed"]
                    or report["partner"] != partner
                ):
                    raise ValueError("validation protocol mismatch")
                expected = {
                    (p, t, f) for p in (2, 4) for t in (12, 24) for f in (False, True)
                }
                if {
                    (c["players"], c["target"], c["flor"])
                    for c in report["configurations"]
                } != expected or len(report["configurations"]) != 8:
                    raise ValueError("rules coverage mismatch")
                if any(
                    c["pairs"] != config["validation_pairs"]
                    or len(c["pair_scores"]) != config["validation_pairs"]
                    for c in report["configurations"]
                ):
                    raise ValueError("paired deal count mismatch")
                reports[arm, seed] = report
        for mode in (2, 4):
            metrics = {}
            for name, weights in CONTRASTS.items():
                values = []
                for seed in config["seeds"]:
                    row = np.zeros(config["validation_pairs"])
                    for arm, weight in weights.items():
                        rows = [
                            c["pair_scores"]
                            for c in reports[arm, seed]["configurations"]
                            if c["players"] == mode
                        ]
                        row += weight * np.asarray(rows).mean(0)
                    values.append(row)
                metrics[name] = contrast(values)
            result["comparisons"][f"{partner}/{mode}players"] = metrics
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--budget", choices=("decisions", "wall"), default="decisions")
    args = parser.parse_args()
    report = summarize(args.campaign, args.budget)
    (args.campaign / f"comparison-{args.budget}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
