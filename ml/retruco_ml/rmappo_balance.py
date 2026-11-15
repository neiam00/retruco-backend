"""Explicit objective masses over genuine learner choices.

For stratum s with n_s choices in a rollout of N choices, an uncapped weight
is N * target_s / n_s. Thus ``sum(weight * loss) / N`` is the requested
mixture of within-stratum mean losses, independent of match/seat length.
Compute weights once per complete rollout and keep the common decision-batch
denominator for every loss term, rather than a batch-specific weight sum.
Labels are scheduling metadata, never actor inputs.
"""

import math
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

from .learning_choices import meaningful


@dataclass(frozen=True, order=True)
class Stratum:
    mode: str
    partner_role: str
    partner_family: str

    def __post_init__(self):
        if self.mode not in ("1v1", "2v2"):
            raise ValueError("balance mode must be 1v1 or 2v2")
        if self.mode == "1v1":
            if (self.partner_role, self.partner_family) != ("none", "none"):
                raise ValueError("1v1 has no partner")
        elif self.partner_role not in ("human", "cpu") or not self.partner_family:
            raise ValueError("2v2 requires a human/cpu partner role and family")
        if "/" in self.partner_family:
            raise ValueError("partner family cannot contain /")

    @property
    def key(self):
        return f"{self.mode}/{self.partner_role}/{self.partner_family}"


def stratum_for_episode(players, controllers, roles, learner_seat):
    """Label one learner seat using only assigned controllers and roles.

    ``roles`` follows the simulator convention: True is the CPU role. In a
    joint-learning episode each learner seat gets its own teammate's label.
    Fixed checkpoint identities collapse into a family, retaining checkpoint
    diversity without fragmenting the balancing objective.
    """
    if players not in (2, 4) or len(controllers) != players:
        raise ValueError("balance requires a 2- or 4-player episode")
    if not 0 <= learner_seat < players or controllers[learner_seat] != "learner":
        raise ValueError("balance label requires a learner-controlled seat")
    if any(controller == "frozen:103" for controller in controllers):
        raise ValueError("D103 is evaluation-only")
    if players == 2:
        return Stratum("1v1", "none", "none")
    if len(roles) < players or any(type(role) is not bool for role in roles[:players]):
        raise ValueError("balance requires explicit boolean CPU roles")
    friend = (learner_seat + 2) % players
    controller = controllers[friend]
    if controller.startswith("script:"):
        family = controller.removeprefix("script:")
    elif controller.startswith("frozen:"):
        family = "frozen"
    elif controller.startswith("league:"):
        family = "league"
    else:
        family = controller
    return Stratum("2v2", "cpu" if roles[friend] else "human", family)


@dataclass(frozen=True)
class BalanceConfig:
    targets: Mapping[Stratum, float]
    max_weight: float = 8.0
    min_choices_per_stratum: int = 8
    coverage: str = "error"

    def __post_init__(self):
        targets = dict(self.targets)
        if not targets or any(not isinstance(s, Stratum) for s in targets):
            raise ValueError("balance targets require explicit Stratum keys")
        if any(not math.isfinite(v) or v < 0 for v in targets.values()):
            raise ValueError("balance targets must be finite nonnegative masses")
        if not math.isclose(math.fsum(targets.values()), 1.0, abs_tol=1e-12, rel_tol=0):
            raise ValueError("balance target masses must sum to 1")
        if not math.isfinite(self.max_weight) or self.max_weight < 1:
            raise ValueError("max_weight must be finite and at least 1")
        if (
            type(self.min_choices_per_stratum) is not int
            or self.min_choices_per_stratum < 1
        ):
            raise ValueError("minimum choices per stratum must be a positive integer")
        if self.coverage not in ("error", "warn"):
            raise ValueError("coverage must be error or warn")
        # Do not retain the caller's mutable target dictionary.
        object.__setattr__(self, "targets", targets)

    def to_dict(self):
        return {
            "targets": {s.key: v for s, v in sorted(self.targets.items())},
            "max_weight": self.max_weight,
            "min_choices_per_stratum": self.min_choices_per_stratum,
            "coverage": self.coverage,
        }

    @classmethod
    def from_dict(cls, value):
        """Read the same JSON-friendly representation emitted in run metadata."""
        return cls(
            targets={
                Stratum(*key.split("/")): mass for key, mass in value["targets"].items()
            },
            max_weight=value.get("max_weight", 8.0),
            min_choices_per_stratum=value.get("min_choices_per_stratum", 8),
            coverage=value.get("coverage", "error"),
        )


def native_partner_balance():
    """The campaign objective; performance gates are separate from these masses."""
    return BalanceConfig(
        {
            Stratum("1v1", "none", "none"): 0.40,
            Stratum("2v2", "human", "literal-v3"): 0.10,
            Stratum("2v2", "human", "trusting-v3"): 0.10,
            Stratum("2v2", "human", "cautious-v3"): 0.10,
            Stratum("2v2", "human", "reference"): 0.10,
            Stratum("2v2", "cpu", "learner"): 0.18,
            Stratum("2v2", "cpu", "frozen"): 0.02,
        }
    )


@dataclass(frozen=True)
class BalanceResult:
    weights: tuple[float, ...]
    metrics: dict


def balance_choices(records: Sequence, config: BalanceConfig):
    """Return aligned loss weights and honest coverage/achieved-mass metrics.

    Records must be genuine choices with collector-assigned ``balance_stratum``.
    Missing/undercovered positive target strata fail by default. Explicit
    ``coverage='warn'`` renormalizes available target mass but reports the
    original target and ``target_met=False`` for absent strata. Capping
    redistributes excess mass proportionally across uncapped targets; it never
    changes the common loss denominator or exceeds ``max_weight``.
    """
    if not records:
        raise ValueError("cannot balance an empty genuine-choice rollout")
    if any(not meaningful(record) for record in records):
        raise ValueError("balance accepts genuine choices only")
    labels = [getattr(record, "balance_stratum", None) for record in records]
    if any(not isinstance(label, Stratum) for label in labels):
        raise ValueError("every choice requires a collector-assigned balance_stratum")
    counts = Counter(labels)
    unknown = counts.keys() - config.targets.keys()
    if unknown:
        raise ValueError(
            "unconfigured balance strata: " + ", ".join(s.key for s in sorted(unknown))
        )
    undercovered = [
        s
        for s, target in config.targets.items()
        if target > 0 and counts[s] < config.min_choices_per_stratum
    ]
    if undercovered:
        message = "insufficient genuine-choice coverage: " + ", ".join(
            f"{s.key}={counts[s]}<{config.min_choices_per_stratum}"
            for s in sorted(undercovered)
        )
        if config.coverage == "error":
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    n = len(records)
    present = {s for s, target in config.targets.items() if target > 0 and counts[s]}
    available_target_mass = math.fsum(config.targets[s] for s in present)
    if not present:
        raise ValueError("no genuine choices in any positive-target stratum")
    capacity = math.fsum(counts[s] * config.max_weight for s in present)
    if capacity < n - 1e-10:
        raise ValueError(
            "weight cap cannot preserve the common loss normalization with available coverage"
        )

    # Bounded water filling: fix overflowing groups at capacity and distribute
    # the remaining N total weight according to the remaining target masses.
    group_weights = {s: 0.0 for s in config.targets}
    remaining, weight_budget = set(present), float(n)
    capped = set()
    while remaining:
        target_sum = math.fsum(config.targets[s] for s in remaining)
        proposed = {
            s: weight_budget * config.targets[s] / target_sum / counts[s]
            for s in remaining
        }
        overflowing = {s for s in remaining if proposed[s] > config.max_weight}
        if not overflowing:
            group_weights.update(proposed)
            break
        for s in overflowing:
            group_weights[s] = config.max_weight
            weight_budget -= counts[s] * config.max_weight
        remaining -= overflowing
        capped |= overflowing

    weights = tuple(group_weights[s] for s in labels)
    total_weight = math.fsum(weights)
    if not math.isclose(total_weight, n, abs_tol=1e-8, rel_tol=1e-12):
        raise ValueError("balance could not preserve the common loss normalization")
    squared_weight = math.fsum(w * w for w in weights)
    strata = {}
    for s, target in sorted(config.targets.items()):
        count, weight = counts[s], group_weights[s]
        strata[s.key] = {
            "sampled_choices": count,
            "sampled_fraction": count / n,
            "target_mass": target,
            "effective_mass": count * weight / n,
            "loss_weight": weight,
            "uncapped_weight": (
                n * target / available_target_mass / count if count else None
            ),
            "weight_sum": count * weight,
            "effective_sample_size": float(count) if weight > 0 else 0.0,
            "capped": s in capped,
            "coverage_ok": target == 0 or count >= config.min_choices_per_stratum,
        }
    maximum_error = max(
        abs(m["effective_mass"] - m["target_mass"]) for m in strata.values()
    )
    metrics = {
        "sampled_choices": n,
        "weight_sum": total_weight,
        "mean_loss_weight": total_weight / n,
        "max_loss_weight": max(weights),
        "effective_sample_size": total_weight * total_weight / squared_weight,
        "effective_sample_fraction": total_weight * total_weight / squared_weight / n,
        "available_target_mass": available_target_mass,
        "coverage_ok": not undercovered,
        "target_met": maximum_error <= 1e-12,
        "maximum_target_mass_error": maximum_error,
        "missing_strata": [s.key for s in sorted(undercovered) if counts[s] == 0],
        "undercovered_strata": [s.key for s in sorted(undercovered)],
        "capped_strata": [s.key for s in sorted(capped)],
        "strata": strata,
    }
    return BalanceResult(weights=weights, metrics=metrics)


def normalize_advantages(advantages, weights, epsilon=1e-8):
    """Normalize raw advantages using the same rollout-level loss measure.

    This helper accepts plain numerical sequences, does not mutate records,
    and returns a tuple ready for conversion to the caller's tensor type.
    """
    if len(advantages) != len(weights) or not weights:
        raise ValueError(
            "weighted advantage normalization requires aligned nonempty inputs"
        )
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("advantage weights must be finite and nonnegative")
    if any(not math.isfinite(a) for a in advantages):
        raise ValueError("advantages must be finite")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("advantage epsilon must be finite and positive")
    total = math.fsum(weights)
    if total <= 0:
        raise ValueError("advantage weights must have positive total mass")
    mean = math.fsum(a * w for a, w in zip(advantages, weights, strict=True)) / total
    variance = (
        math.fsum(w * (a - mean) ** 2 for a, w in zip(advantages, weights, strict=True))
        / total
    )
    scale = max(math.sqrt(variance), epsilon)
    return tuple((a - mean) / scale for a in advantages)
