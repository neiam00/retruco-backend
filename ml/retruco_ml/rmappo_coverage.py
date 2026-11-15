"""Exact authenticated campaign receipts; unrelated files cannot inflate progress."""

import json

from .recurrent import file_hash
from .target_compare import verified_report


def coverage(root, config):
    models = {"reference": root / "inputs/legacy-champion-lstm16.pt"}
    for cell in config["cells"]:
        for seed in config["seeds"]:
            models[f"{cell}-{seed}"] = (
                root / "trials" / f"{cell}-{seed}" / "candidate.pt"
            )
    specs = config["primary_specs"] + config["diagnostic_specs"]
    expected = []
    for phase in ("selection", "final"):
        for name, cp in models.items():
            for spec in specs:
                expected.append(
                    (
                        root / phase / name / (spec["id"] + ".json"),
                        phase,
                        spec,
                        cp,
                        "normal",
                    )
                )
    ablation_count = 4 * (len(config["seeds"]) + 1)
    nominee = root / "nominee.json"
    if nominee.exists():
        cell = json.loads(nominee.read_text())["exploratory_nominee"].split("-")[0]
        for name, cp in models.items():
            if name != "reference" and not name.startswith(cell + "-"):
                continue
            spec = config["primary_specs"][1]
            for mode in ("normal", "no-send", "no-receive", "no-partner-delivery"):
                expected.append(
                    (
                        root / "ablation" / name / (spec["id"] + "-" + mode + ".json"),
                        "ablation",
                        spec,
                        cp,
                        mode,
                    )
                )
    count = 2 * len(models) * len(specs) + ablation_count
    completed = []
    invalid = []
    missing = []
    inputs_sha = file_hash(root / "inputs/manifest.json")
    for path, phase, spec, cp, mode in expected:
        if not path.exists() or not path.with_suffix(".sha256").exists():
            missing.append(str(path.relative_to(root)))
            continue
        try:
            verified_report(
                path,
                spec,
                config[phase + "_pairs"],
                config[phase + "_seed"],
                cp,
                inputs_sha,
                mode,
            )
        except (ValueError, OSError, KeyError) as error:
            invalid.append({"path": str(path.relative_to(root)), "error": str(error)})
        else:
            completed.append(str(path.relative_to(root)))
    return {
        "expected": count,
        "authenticated": len(completed),
        "complete": len(completed) == count and not invalid,
        "invalid": invalid,
        "missing_known_paths": missing,
        "completed_paths": completed,
    }
