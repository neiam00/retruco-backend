"""Read-only paired development audit of a frozen RMAPPO campaign.

Copy this executable into a new output directory before running it. Evaluators
import only from the authenticated old source; checkpoints remain untouched.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

MILESTONES = (250000, 500000, 1000000, 2000000)
TRAINING_SEEDS = (1101, 1102, 1103)
CELLS = ("gru", "lstm")
EXTRA_SPECS = ("parent-headsup", "native-independent-partner")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_suffix(path.suffix + ".tmp") if replace else path
    with target.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if replace:
        target.replace(path)


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def verify_receipts(entries):
    for entry in entries:
        path = Path(entry["path"])
        if not path.is_file() or digest(path) != entry["sha256"]:
            raise ValueError(f"protected artifact changed: {path}")


def stream_inventory(ml_root, seed, pairs, exclude):
    """Reject recorded evaluation ranges overlapping the proposed new deals."""
    overlaps, scanned = [], 0
    for path in sorted(Path(ml_root).glob("**/*.json")):
        if path.is_relative_to(exclude) or any(
            part in (".venv", "source", "__pycache__") for part in path.parts
        ):
            continue
        if path.name in (
            "preservation.json",
            "comparison.json",
            "comparison-selection.json",
        ):
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        scanned += 1
        for start_key, count_key in [
            ("seed", "pairs"),
            ("selection_seed", "selection_pairs"),
            ("final_seed", "final_pairs"),
            ("ablation_seed", "ablation_pairs"),
            ("evaluation_seed", "evaluation_pairs"),
        ]:
            start, count = data.get(start_key), data.get(count_key)
            if isinstance(start, int) and isinstance(count, int) and count > 0:
                if seed < start + count and start < seed + pairs:
                    overlaps.append(
                        {"path": str(path.resolve()), "seed": start, "pairs": count}
                    )
    if overlaps:
        raise ValueError(f"development deal range already used: {overlaps}")
    return {
        "json_objects_scanned": scanned,
        "overlapping_recorded_ranges": overlaps,
        "seed": seed,
        "pairs": pairs,
        "deal_range_inclusive": [seed, seed + pairs - 1],
        "method": "top-level paired reports and campaign seed/pair ranges; frozen source copies excluded",
    }


def clusters(report):
    """Average seats/configurations within a deal, preserving dependence."""
    pairs = report["pairs"]
    if not isinstance(pairs, int) or pairs < 2:
        raise ValueError("need at least two paired deal clusters")
    rows, keys = [], set()
    for config in report["configurations"]:
        key = (config["players"], config["target"], config["flor"])
        values = config["pair_scores"]
        if key in keys or config["pairs"] != pairs or len(values) != pairs:
            raise ValueError("duplicate configuration or mismatched deal counts")
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
            raise ValueError("invalid paired score")
        keys.add(key)
        rows.append(values)
    if not rows:
        raise ValueError("missing paired configurations")
    return np.asarray(rows, dtype=np.float64).mean(axis=0)


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("need at least two finite clusters")
    rng = np.random.default_rng(9017)
    means = []
    for _ in range(100):
        indices = rng.integers(len(values), size=(100, len(values)))
        means.extend(values[indices].mean(axis=1).tolist())
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def paired_delta(candidate, reference):
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("paired comparison requires corresponding reports")

    def configurations(report):
        return {
            (c["players"], c["target"], c["flor"]) for c in report["configurations"]
        }

    rows = []
    for left, right in zip(candidate, reference):
        identity = ("seed", "pairs", "partner", "opponent", "signals", "card_policy")
        if any(left.get(k) != right.get(k) for k in identity):
            raise ValueError("paired comparison protocol/deal mismatch")
        if configurations(left) != configurations(right):
            raise ValueError("paired comparison configuration mismatch")
        rows.append(clusters(left) - clusters(right))
    if len({r["seed"] for r in candidate}) != 1:
        raise ValueError("all reports must share the same deal stream")
    values = np.asarray(rows).mean(axis=0)
    return {
        "delta": float(values.mean()),
        "cluster_ci95": interval(values),
        "deal_clusters": len(values),
        "method": "candidate minus reference; paired deal-cluster bootstrap",
        "positive_clusters": int((values > 0).sum()),
        "negative_clusters": int((values < 0).sum()),
    }


def acceptance(reports):
    """Opportunity-weighted descriptive acceptance by canto and partner claim."""
    families = {}
    for report in reports:
        for name, counts in report.get("response_families", {}).items():
            players, canto, _value, _terminal, claim = name.split("/")
            key = "/".join((players, canto, claim))
            bucket = families.setdefault(
                key, dict.fromkeys(("opportunities", "accepts", "declines", "other"), 0)
            )
            for field in bucket:
                bucket[field] += counts[field]
    for bucket in families.values():
        opportunities = bucket["opportunities"]
        if sum(bucket[x] for x in ("accepts", "declines", "other")) != opportunities:
            raise ValueError("response family counts do not exhaust opportunities")
        bucket["acceptance_rate"] = (
            bucket["accepts"] / opportunities if opportunities else None
        )
        bucket["decline_rate"] = (
            bucket["declines"] / opportunities if opportunities else None
        )
    return {
        "key": "players/canto/visible_partner_claim",
        "families": families,
        "interpretation": "descriptive conditional opportunities, not causal claim effects; supports differ by policy",
    }


def metric(reports, reference=None):
    if not reports or len({(r["seed"], r["pairs"]) for r in reports}) != 1:
        raise ValueError("aggregate needs aligned nonempty deal streams")
    values = np.asarray([clusters(r) for r in reports]).mean(axis=0)
    result = {
        "win_rate": float(values.mean()),
        "cluster_ci95": interval(values),
        "deal_clusters": len(values),
        "reports": len(reports),
        "acceptance": acceptance(reports),
    }
    if reference is not None:
        result["paired_champion_delta"] = paired_delta(reports, reference)
    return result


def prepare(source_root, output_root, pairs, seed, workers):
    if (
        workers not in range(1, 5)
        or not 100000 <= seed < seed + pairs < 200000
        or pairs < 2
    ):
        raise ValueError("development pairs and 1–4 evaluators required")
    original = json.loads((source_root / "configuration.json").read_text())
    forbidden = [
        (original[k], original[k.replace("seed", "pairs")])
        for k in ("selection_seed", "final_seed", "ablation_seed")
    ]
    if any(seed < start + size and start < seed + pairs for start, size in forbidden):
        raise ValueError("original selection/final/ablation deal stream is forbidden")
    source = source_root / "source"
    frozen = json.loads((source / "manifest.json").read_text())
    protected = []
    for relative, expected in frozen["files"].items():
        item = receipt(source / relative)
        if item["sha256"] != expected:
            raise ValueError(f"frozen evaluator source changed: {relative}")
        protected.append(item)
    for path in sorted((source_root / "inputs").iterdir()):
        if path.is_file():
            protected.append(receipt(path))
    protected.extend(
        receipt(source_root / x)
        for x in ("configuration.json", "frozen.json", "source/manifest.json")
    )
    inputs = json.loads((source_root / "inputs/manifest.json").read_text())
    champion = inputs["candidates"]["legacy-champion"]
    champion_path = source_root / "inputs" / champion["file"]
    if digest(champion_path) != champion["sha256"]:
        raise ValueError("legacy champion differs from frozen input receipt")
    specs = original["primary_specs"] + [
        s for s in original["diagnostic_specs"] if s["id"] in EXTRA_SPECS
    ]
    if len(specs) != 7 or len({s["id"] for s in specs}) != 7:
        raise ValueError("expected five primary and two transfer specifications")
    checkpoints = [
        {
            "id": "legacy-champion",
            "cell": "legacy",
            "training_seed": None,
            "choices": None,
            **receipt(champion_path),
        }
    ]
    for cell in CELLS:
        for training_seed in TRAINING_SEEDS:
            for choices in MILESTONES:
                path = (
                    source_root
                    / f"trials/{cell}-{training_seed}/endpoints/choices-{choices}.pt"
                )
                item = receipt(path)
                protected.append(item)
                checkpoints.append(
                    {
                        "id": f"{cell}-{training_seed}-choices-{choices}",
                        "cell": cell,
                        "training_seed": training_seed,
                        "choices": choices,
                        **item,
                    }
                )
    inventory = stream_inventory(source_root.parents[1], seed, pairs, output_root)
    return {
        "version": 1,
        "kind": "rmappo-checkpoint-audit-v1",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_commit": frozen["source_commit"],
        "source_dirty": frozen["source_dirty"],
        "seed": seed,
        "pairs": pairs,
        "workers": workers,
        "specs": specs,
        "primary_spec_ids": [s["id"] for s in original["primary_specs"]],
        "checkpoints": checkpoints,
        "protected_artifacts": protected,
        "stream_inventory": inventory,
        "runner": receipt(Path(__file__)),
        "created_unix": time.time(),
        "expected_evaluations": len(checkpoints) * len(specs),
        "scope": "new development deals only; no model selection, training, holdout reuse or promotion",
    }


def validate_result(path, checkpoint, spec, manifest):
    path = Path(path)
    if path.with_suffix(".sha256").read_text().strip() != digest(path):
        raise ValueError(f"evaluation report receipt mismatch: {path}")
    report = json.loads(path.read_text())
    expected = {
        "protocol": "improvement-evaluation-v1",
        "seed": manifest["seed"],
        "pairs": manifest["pairs"],
        "partner": spec["partner"],
        "opponent": spec["opponent"],
        "signals": "normal",
        "card_policy": "policy",
        "checkpoint_sha256": checkpoint["sha256"],
        "inputs_sha256": digest(Path(manifest["source_root"]) / "inputs/manifest.json"),
    }
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError(f"evaluation identity mismatch: {path}")
    keys = {(c["players"], c["target"], c["flor"]) for c in report["configurations"]}
    if keys != {(spec["players"], t, f) for t in (12, 24) for f in (False, True)}:
        raise ValueError(f"evaluation configurations mismatch: {path}")
    values = clusters(report)
    if not math.isclose(
        float(values.mean()), report["pooled"]["win_rate"], abs_tol=1e-12
    ):
        raise ValueError(f"evaluation pooled scores disagree: {path}")
    return report


def run_evaluation(checkpoint, spec, manifest):
    root, original = Path(manifest["output_root"]), Path(manifest["source_root"])
    output = root / "evaluations" / checkpoint["id"] / (spec["id"] + ".json")
    log = root / "logs" / (checkpoint["id"] + "--" + spec["id"] + ".log")
    if output.exists():
        validate_result(output, checkpoint, spec, manifest)
        return receipt(output)
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "retruco_ml.rmappo_evaluate",
        checkpoint["path"],
        "--inputs",
        str(original / "inputs/manifest.json"),
        "--simulator",
        str(original / "source/bin/selfplay"),
        "--output",
        str(output),
        "--partner",
        spec["partner"],
        "--opponent",
        spec["opponent"],
        "--players",
        str(spec["players"]),
        "--pairs",
        str(manifest["pairs"]),
        "--seed",
        str(manifest["seed"]),
    ]
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(original / "source/ml"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    started = time.time()
    with log.open("a") as stream:
        stream.write(json.dumps({"command": command, "started_unix": started}) + "\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=original / "source",
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"evaluator exited {result.returncode}; see {log}")
    validate_result(output, checkpoint, spec, manifest)
    return {**receipt(output), "seconds": time.time() - started}


def summarize(manifest):
    root = Path(manifest["output_root"])
    reports, receipts = {}, []
    for checkpoint in manifest["checkpoints"]:
        for spec in manifest["specs"]:
            path = root / "evaluations" / checkpoint["id"] / (spec["id"] + ".json")
            reports[checkpoint["id"], spec["id"]] = validate_result(
                path, checkpoint, spec, manifest
            )
            receipts.append(receipt(path))
    groups = {
        "primary_1v1": [
            s["id"]
            for s in manifest["specs"]
            if s["id"] in manifest["primary_spec_ids"] and s["players"] == 2
        ],
        "primary_2v2": [
            s["id"]
            for s in manifest["specs"]
            if s["id"] in manifest["primary_spec_ids"] and s["players"] == 4
        ],
    }
    groups.update({s["id"]: [s["id"]] for s in manifest["specs"]})
    champion = {
        name: metric([reports["legacy-champion", s] for s in ids])
        for name, ids in groups.items()
    }
    individual, curves = {}, {}
    for cell in CELLS:
        for training_seed in TRAINING_SEEDS:
            run = f"{cell}-{training_seed}"
            individual[run] = {}
            for index, choices in enumerate(MILESTONES):
                identifier = f"{run}-choices-{choices}"
                measures = {}
                for name, ids in groups.items():
                    current = [reports[identifier, s] for s in ids]
                    result = metric(
                        current, [reports["legacy-champion", s] for s in ids]
                    )
                    result["paired_previous_milestone_delta"] = (
                        None
                        if index == 0
                        else paired_delta(
                            current,
                            [
                                reports[f"{run}-choices-{MILESTONES[index - 1]}", s]
                                for s in ids
                            ],
                        )
                    )
                    result["paired_250k_delta"] = paired_delta(
                        current, [reports[f"{run}-choices-250000", s] for s in ids]
                    )
                    measures[name] = result
                individual[run][str(choices)] = measures
        curves[cell] = {}
        for index, choices in enumerate(MILESTONES):
            measures = {}
            for name, ids in groups.items():
                current = [
                    reports[f"{cell}-{seed}-choices-{choices}", s]
                    for seed in TRAINING_SEEDS
                    for s in ids
                ]
                result = metric(
                    current,
                    [
                        reports["legacy-champion", s]
                        for _ in TRAINING_SEEDS
                        for s in ids
                    ],
                )
                seed_rates = {
                    str(seed): individual[f"{cell}-{seed}"][str(choices)][name][
                        "win_rate"
                    ]
                    for seed in TRAINING_SEEDS
                }
                result["training_seed_win_rates"] = seed_rates
                result["training_seed_min_max"] = [
                    min(seed_rates.values()),
                    max(seed_rates.values()),
                ]
                result["paired_previous_milestone_delta"] = (
                    None
                    if index == 0
                    else paired_delta(
                        current,
                        [
                            reports[f"{cell}-{seed}-choices-{MILESTONES[index - 1]}", s]
                            for seed in TRAINING_SEEDS
                            for s in ids
                        ],
                    )
                )
                result["paired_250k_delta"] = paired_delta(
                    current,
                    [
                        reports[f"{cell}-{seed}-choices-250000", s]
                        for seed in TRAINING_SEEDS
                        for s in ids
                    ],
                )
                measures[name] = result
            curves[cell][str(choices)] = measures
    verify_receipts(manifest["protected_artifacts"])
    return {
        "version": 1,
        "kind": manifest["kind"],
        "complete": True,
        "approved": False,
        "source_commit": manifest["source_commit"],
        "manifest": receipt(root / "manifest.json"),
        "evaluation_count": len(receipts),
        "evaluation_receipts": receipts,
        "deal_stream": manifest["stream_inventory"],
        "legacy_champion": champion,
        "learning_curves": curves,
        "individual_runs": individual,
        "preservation": {
            "verified_artifacts": len(manifest["protected_artifacts"]),
            "all_unchanged": True,
        },
        "interpretation": [
            "All milestones and the exact legacy champion share the new development deal stream.",
            "Each deal averages learner seats, target/flor configurations, then specifications and training seeds.",
            "Three-seed curves retain 32 independent deal clusters; training seeds are not extra deals.",
            "Confidence intervals condition on these three training seeds; training variance is shown by per-seed rates.",
            "Primary 1v1 and 2v2 remain separate; parent-headsup and frozen independent D103 partner are diagnostics.",
            "Conditional acceptance uses response opportunities by canto and visible partner claim; it is descriptive.",
            "No lucky seed selection, final holdout reuse or deployment is authorized by this development audit.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=149000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    original, root = args.source_root.resolve(), args.output_root.resolve()
    if (
        root == original
        or root.is_relative_to(original)
        or args.report.resolve().is_relative_to(original)
    ):
        raise ValueError("audit outputs must be outside the preserved campaign")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["source_root"],
            manifest["output_root"],
            manifest["pairs"],
            manifest["seed"],
            manifest["workers"],
        ) != (str(original), str(root), args.pairs, args.seed, args.workers):
            raise ValueError("resume must retain the original audit configuration")
        verify_receipts(manifest["protected_artifacts"] + [manifest["runner"]])
    else:
        if args.report.exists() or manifest_path.exists():
            raise FileExistsError(
                "refusing to overwrite an audit; use --resume for interrupted evaluations"
            )
        manifest = prepare(original, root, args.pairs, args.seed, args.workers)
        write_json(manifest_path, manifest)
    complete, failures = [], []
    status = root / "status.json"
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(run_evaluation, checkpoint, spec, manifest): (
                checkpoint["id"],
                spec["id"],
            )
            for checkpoint in manifest["checkpoints"]
            for spec in manifest["specs"]
        }
        for future in as_completed(jobs):
            identity = jobs[future]
            try:
                complete.append(
                    {"checkpoint": identity[0], "spec": identity[1], **future.result()}
                )
            except Exception as error:
                failures.append(
                    {
                        "checkpoint": identity[0],
                        "spec": identity[1],
                        "error": str(error),
                    }
                )
            write_json(
                status,
                {
                    "phase": "evaluating",
                    "completed": len(complete),
                    "expected": len(jobs),
                    "failed": failures,
                    "elapsed_seconds": time.time() - started,
                    "last_finished": list(identity),
                },
                replace=status.exists(),
            )
            print(
                json.dumps(
                    {
                        "completed": len(complete),
                        "failed": len(failures),
                        "expected": len(jobs),
                        "last_finished": identity,
                    }
                ),
                flush=True,
            )
    if failures:
        raise RuntimeError(
            f"{len(failures)} evaluations failed; inspect status.json and logs; resume preserves completed reports"
        )
    report = summarize(manifest)
    write_json(args.report, report)
    write_json(root / "summary.json", report)
    write_json(
        root / "completion.json",
        {
            "report": receipt(args.report),
            "summary": receipt(root / "summary.json"),
            "elapsed_seconds": time.time() - started,
            "evaluations": complete,
        },
    )
    write_json(
        status,
        {
            "phase": "complete",
            "completed": len(complete),
            "expected": len(complete),
            "failed": [],
            "elapsed_seconds": time.time() - started,
        },
        replace=True,
    )
    print(
        json.dumps({"complete": True, "report": str(args.report.resolve())}), flush=True
    )


if __name__ == "__main__":
    main()
