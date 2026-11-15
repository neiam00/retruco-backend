"""Seed-cluster confidence intervals, preserving seat/configuration dependence."""

import numpy as np


def interval(values, seed=9017, samples=10000):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("need at least two finite independent seed clusters")
    rng = np.random.default_rng(seed)
    # Bound memory rather than constructing a samples-by-full-tournament matrix.
    means = []
    for start in range(0, samples, 100):
        indices = rng.integers(
            len(values), size=(min(100, samples - start), len(values))
        )
        means.extend(values[indices].mean(axis=1).tolist())
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def pooled(configurations):
    if not configurations:
        raise ValueError("missing configurations")
    sizes = {len(c["pair_scores"]) for c in configurations}
    if len(sizes) != 1:
        raise ValueError("configuration seed clusters must align")
    # The same seed is reused across configurations: average within seed first.
    clusters = np.asarray(
        [c["pair_scores"] for c in configurations], dtype=np.float64
    ).mean(axis=0)
    return {
        "win_rate": float(clusters.mean()),
        "cluster_ci95": interval(clusters),
        "seed_clusters": len(clusters),
    }


def paired_loss(reference, candidate):
    if reference["seed"] != candidate["seed"]:
        raise ValueError("paired reports must use identical deal seeds")
    left = {
        (c["players"], c["target"], c["flor"]): c for c in reference["configurations"]
    }
    right = {
        (c["players"], c["target"], c["flor"]): c for c in candidate["configurations"]
    }
    if left.keys() != right.keys() or not left:
        raise ValueError("configuration mismatch")
    losses = []
    for key, c in left.items():
        if (
            c["pairs"] != right[key]["pairs"]
            or len(c["pair_scores"]) != c["pairs"]
            or len(right[key]["pair_scores"]) != c["pairs"]
        ):
            raise ValueError("pair count mismatch")
        losses.append(
            np.asarray(c["pair_scores"]) - np.asarray(right[key]["pair_scores"])
        )
    clusters = np.asarray(losses).mean(axis=0)
    return {
        "loss": float(clusters.mean()),
        "cluster_ci95": interval(clusters),
        "seed_clusters": len(clusters),
        "positive_clusters": int((clusters > 0).sum()),
        "negative_clusters": int((clusters < 0).sum()),
        "method": "paired seed-cluster bootstrap",
    }
