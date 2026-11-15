"""Frozen campaign settings shared by launch, endpoint checks and comparison."""

import math
from pathlib import Path

from .decision_ppo import validate_update_options

CONTROL_KEYS = frozenset(
    (
        "native_fraction",
        "gae_lambda",
        "entropy_coef",
        "sequence_length",
        "ppo_epochs",
        "target_kl",
        "clip_range",
        "actor_learning_rate",
        "critic_learning_rate",
        "actor_max_grad_norm",
        "critic_max_grad_norm",
        "extra_critic_epochs",
        "response_kl_target",
        "response_kl_stop",
        "response_kl_rollback",
        "response_min_choices",
        "league_anchor_fraction",
        "partner_policy_version",
    )
)
CAPACITY_KEYS = (
    "steps",
    "workers",
    "environments",
    "batch_size",
    "warmup_max",
    "calibration_choices",
    "memory_gib",
    "parent_anchor",
    "response_anchor",
    "signal_anchor",
)


def targets(config=None):
    # Historical callers retain their explicit legacy default. New campaigns
    # always pass their frozen configuration, not a process-global target.
    values = (config or {}).get("mode_win_rate_targets", {"1v1": 0.70, "2v2": 0.65})
    if set(values) != {"1v1", "2v2"} or any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0.5 < v < 1
        for v in values.values()
    ):
        raise ValueError("invalid separate native win-rate targets")
    return {2: values["1v1"], 4: values["2v2"]}


def selection_specs(config):
    return config["primary_specs"] + (
        config["diagnostic_specs"] if config["version"] >= 2 else []
    )


def ablation_modes(config):
    return config.get("ablation_modes", ["normal", "no-send", "no-receive"])


def confirmation_budgets(root, config):
    import json

    from .recurrent import file_hash

    path = Path(root) / "confirmation-plan.json"
    if config["version"] < 2 or not path.exists():
        return config["confirmation_checkpoints"]
    if path.with_suffix(".sha256").read_text().strip() != file_hash(path):
        raise ValueError("confirmation budget receipt changed")
    budgets = json.loads(path.read_text())["completed_checkpoints"]
    if not budgets or budgets != config["confirmation_checkpoints"][: len(budgets)]:
        raise ValueError("confirmation budgets must be a nonempty common prefix")
    return budgets


def training_settings(config, recipe, root):
    settings = {key: config[key] for key in CAPACITY_KEYS}
    settings.update(
        headsup_fraction=config["headsup_fraction"],
        actor_memory=config["actor_memory"],
        critic_hidden=config["critic_hidden"],
        league_interval=config["league_interval"],
        heldout_parent=config["heldout_parent"],
    )
    if config["version"] == 1:
        settings.update(config["recipes"][recipe])
    else:
        settings.update(config["trainer_controls"])
        settings.update(
            {k: v for k, v in config["recipes"][recipe].items() if k != "partner_pool"}
        )
        settings["partner_manifest"] = (
            str((Path(root) / "partners/manifest.json").resolve())
            if config["recipes"][recipe]["partner_pool"]
            else None
        )
    return settings


def training_command(config, recipe, root, folder, seed, budget, simulator):
    command = [
        "retruco_ml.improvement_train",
        "--output",
        str(folder),
        "--inputs",
        str(root / "inputs/manifest.json"),
        "--simulator",
        str(simulator),
        "--variant",
        recipe,
        "--seed",
        str(seed),
        "--warm-start",
        "legacy-champion",
        "--choices",
        str(budget),
        "--cooperation-practice",
        "--self-play-league",
    ]
    for key, value in training_settings(config, recipe, root).items():
        if value is None:
            continue
        command += [
            "--" + key.replace("_", "-"),
            *[str(x) for x in (value if isinstance(value, list) else [value])],
        ]
    return command


def validate_controls(config):
    controls = config.get("trainer_controls", {})
    allowed = CONTROL_KEYS | ({"team_adapter"} if config["version"] >= 3 else set())
    if set(controls) != allowed:
        raise ValueError("complete explicit trainer controls required")
    if any(
        type(v) not in (int, float) or not math.isfinite(v)
        for k, v in controls.items()
        if k != "team_adapter"
    ):
        raise ValueError("trainer controls must be finite numeric values")
    recipes = config.get("recipes", {})
    if not 2 <= len(recipes) <= 8:
        raise ValueError("bounded controlled recipes required")
    for name, overrides in recipes.items():
        if (
            not isinstance(name, str)
            or not name.replace("_", "").isalnum()
            or len(name) > 48
        ):
            raise ValueError("invalid recipe path component")
        if (
            not isinstance(overrides, dict)
            or type(overrides.get("partner_pool")) is not bool
            or set(overrides) - allowed - {"partner_pool"}
        ):
            raise ValueError("unknown or missing recipe intervention")
        values = {
            **controls,
            **{k: v for k, v in overrides.items() if k != "partner_pool"},
        }
        validate_update_options(
            **{
                k: v
                for k, v in values.items()
                if k
                not in (
                    "native_fraction",
                    "gae_lambda",
                    "actor_learning_rate",
                    "critic_learning_rate",
                    "league_anchor_fraction",
                    "partner_policy_version",
                    "sequence_length",
                    "ppo_epochs",
                    "team_adapter",
                )
            },
            sequence=values["sequence_length"],
            epochs=values["ppo_epochs"],
        )
        if (
            not 0 <= values["native_fraction"] <= 1
            or not 0 <= values["league_anchor_fraction"] <= 1
            or values["gae_lambda"] not in (0.95, 0.99, 1.0)
            or type(values["partner_policy_version"]) is not int
            or values["partner_policy_version"] not in (2, 3)
            or not 1e-5 <= values["actor_learning_rate"] <= 3e-4
            or not 0 < values["critic_learning_rate"] < 1
        ):
            raise ValueError("invalid trainer population or learning rates")
        if config["version"] >= 3 and values.get("team_adapter") not in (
            "residual",
            "contextual",
        ):
            raise ValueError("team specialist requires an explicit supported adapter")
    contrasts = config.get("factorial_contrasts", {})
    if not contrasts or any(
        not isinstance(weights, dict)
        or not weights
        or set(weights) - set(recipes)
        or any(
            type(x) not in (int, float) or not math.isfinite(x)
            for x in weights.values()
        )
        or abs(sum(weights.values())) > 1e-12
        or not any(weights.values())
        for weights in contrasts.values()
    ):
        raise ValueError("contrasts must name recipes with finite zero-sum weights")
    if ablation_modes(config) != [
        "normal",
        "no-send",
        "no-receive",
        "no-partner-delivery",
    ]:
        raise ValueError("full signal intervention coverage required")
    if (
        not isinstance(config.get("partner_pool_source"), str)
        or not config["partner_pool_source"]
    ):
        raise ValueError("explicit immutable partner pool required")


def selection_rank(assessment, config):
    if config.get("version", 1) == 1:
        return (assessment["balanced_score"],)
    modes = assessment["modes"]
    qualified = all(
        v["target_met"] and v["confidence_above_chance"] for v in modes.values()
    )
    bottleneck = min(v["win_rate"] / v["target_win_rate"] for v in modes.values())
    return (qualified, bottleneck, modes["4"]["win_rate"], assessment["balanced_score"])


def rank_improves(after, before, tolerance=0.002):
    if len(after) == 1:
        return after[0] > before[0] + tolerance
    if after[0] != before[0]:
        return after[0] > before[0]
    for a, b in zip(after[1:], before[1:]):
        if abs(a - b) > 1e-12:
            return a > b + tolerance
    return False


def estimate_confirmation_round(root, config, recipe, budget, previous_budget):
    """Conservative forecast from persisted real work, never cached replay time."""
    import json

    root = Path(root)
    rates = []
    warmups = []
    for seed in config["discovery_seeds"]:
        path = root / "discovery" / f"{recipe}-seed-{seed}" / "metrics.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        updates = [r for r in records if r["phase"] == "training"]
        duration = sum(r["collect_seconds"] + r["optimize_seconds"] for r in updates)
        if not updates or duration <= 0:
            raise ValueError(
                "no measured discovery throughput for confirmation forecast"
            )
        rates.append(sum(r["choices"] for r in updates) / duration)
        warmups.append(sum(r["seconds"] for r in records if r["phase"] == "warmup"))
    costs = {}
    for line in (root / "work-completions.jsonl").read_text().splitlines():
        row = json.loads(line)
        command = row["command"]
        if command[0] != "retruco_ml.improvement_evaluate":
            continue
        name = Path(command[command.index("--output") + 1]).stem
        pairs = int(command[command.index("--pairs") + 1])
        costs[name] = max(costs.get(name, 0.0), row["seconds"] / pairs)
    if any(spec["id"] not in costs for spec in selection_specs(config)):
        raise ValueError("missing measured evaluation cost for confirmation forecast")
    # Count evaluator cost serially, deliberately overestimating parallel wall time.
    evaluation = sum(
        costs[s["id"]] * config["selection_pairs"] for s in selection_specs(config)
    )
    training = (budget - previous_budget) / min(rates) + (
        max(warmups) if previous_budget == 0 else 0
    )
    return 1.25 * len(config["confirmation_seeds"]) * (training + evaluation)
