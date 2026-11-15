"""Match-stable frozen recurrent opponents, separate from legacy Policy history.

Only call add_actor between complete collection batches. A models() registry owns
independent frozen models, so later additions cannot change an active match.
"""

import hashlib
import json
import math
import random

import torch

from .improvement_population import ImprovementPopulation
from .model import Policy
from .model_capacity import actor_from_checkpoint, team_adapter_configuration
from .recurrent import Actor

VERSION = 1
MAX_SNAPSHOTS = 16


def _copy_weights(weights):
    if not isinstance(weights, dict) or not weights:
        raise ValueError("invalid league weights")
    result = {}
    for name, value in weights.items():
        if (
            not isinstance(name, str)
            or not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.layout != torch.strided
            or not torch.isfinite(value).all()
        ):
            raise ValueError("league weights require finite float32 tensors")
        result[name] = value.detach().cpu().contiguous().clone()
    return result


def _fingerprint(weights):
    digest = hashlib.sha256()
    for name in sorted(weights):
        value = weights[name]
        header = json.dumps(
            [name, str(value.dtype), list(value.shape)], separators=(",", ":")
        ).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _actor(weights):
    # Construction initializes parameters before loading. It must not perturb the
    # learner's sampling stream merely because an opponent registry is rebuilt.
    with torch.random.fork_rng(devices=[]):
        metadata = team_adapter_configuration(weights)
        model = actor_from_checkpoint(
            {"model": weights, **({"configuration": metadata} if metadata else {})}
        )
    if not model.recurrent:
        raise ValueError("league requires recurrent actors")
    return model.eval().requires_grad_(False)


def _legacy(weights):
    with torch.random.fork_rng(devices=[]):
        model = Policy()
        model.load_state_dict(weights, strict=True)


class HistoricalLeaguePopulation(ImprovementPopulation):
    def __init__(
        self,
        snapshots=None,
        fixed_opponent=None,
        learned_controllers=(),
        reactive=False,
        max_snapshots=MAX_SNAPSHOTS,
        anchor_actor=None,
        anchor_fraction=0.0,
    ):
        if type(max_snapshots) is not int or not 1 <= max_snapshots <= MAX_SNAPSHOTS:
            raise ValueError("league capacity must be between 1 and 16")
        if not math.isfinite(anchor_fraction) or not 0 <= anchor_fraction <= 1:
            raise ValueError("anchor fraction must be finite and between zero and one")
        if anchor_fraction and (
            max_snapshots < 2
            or not isinstance(anchor_actor, Actor)
            or not anchor_actor.recurrent
        ):
            raise ValueError(
                "persistent anchor requires a recurrent actor and at least two slots"
            )
        super().__init__(snapshots, fixed_opponent, learned_controllers, reactive)
        self.snapshots = [_copy_weights(state) for state in self.snapshots]
        self.fixed_opponent = (
            None if fixed_opponent is None else _copy_weights(fixed_opponent)
        )
        self.max_snapshots = max_snapshots
        self._actors = []
        self._anchor = None
        self.anchor_fraction = anchor_fraction
        self.recipe = json.dumps(
            {
                "version": "historical-league-v1",
                "base": json.loads(self.recipe),
                "latest_fraction": 0.45,
                "earlier_fraction": 0.40,
                "inherited_fraction": 0.15,
                "max_snapshots": max_snapshots,
                "empty_league": "inherited",
                "missing_earlier": "latest",
                "opponent_role": "cpu",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if anchor_fraction:
            weights = _copy_weights(anchor_actor.state_dict())
            _actor(weights)
            self._anchor = {"fingerprint": _fingerprint(weights), "model": weights}
            self._actors.append({**self._anchor, "choices": 0})
            recipe = json.loads(self.recipe)
            recipe["persistent_anchor"] = {
                "fingerprint": self._anchor["fingerprint"],
                "fraction_within_league": anchor_fraction,
                "sampling": "domain-separated replacement after ordinary opponent draw",
                "capacity": "one reserved slot within max_snapshots",
                "only_anchor": "use anchor",
            }
            self.recipe = json.dumps(recipe, sort_keys=True, separators=(",", ":"))

    @property
    def snapshot_count(self):
        return len(self._actors)

    def add_actor(self, actor, choices):
        """Freeze an update-boundary actor; repeated weights occupy one slot."""
        if not isinstance(actor, Actor) or not actor.recurrent:
            raise ValueError("league requires recurrent actors")
        if (
            type(choices) is not int
            or choices < 0
            or (self._actors and choices < self._actors[-1]["choices"])
        ):
            raise ValueError("league choices cannot move backwards")
        weights = _copy_weights(actor.state_dict())
        _actor(weights)  # Validate the serialized architecture before admitting it.
        fingerprint = _fingerprint(weights)
        entry = {"fingerprint": fingerprint, "choices": choices, "model": weights}
        self._actors = [
            old for old in self._actors if old["fingerprint"] != fingerprint
        ]
        self._actors.append(entry)
        if self._anchor is None:
            self._actors = self._actors[-self.max_snapshots :]
        else:
            while len(self._actors) > self.max_snapshots:
                oldest = next(
                    index
                    for index, old in enumerate(self._actors)
                    if old["fingerprint"] != self._anchor["fingerprint"]
                )
                del self._actors[oldest]
        return "league:" + fingerprint

    def models(self, reference):
        result = super().models(reference)
        for entry in self._actors:
            result["league:" + entry["fingerprint"]] = _actor(entry["model"])
        return result

    def episode(self, players, rng):
        episode = super().episode(players, rng)
        if not self._actors:
            return episode
        draw = rng.random()
        if draw >= 0.85:
            return episode
        entry = (
            self._actors[-1]
            if draw < 0.45 or len(self._actors) == 1
            else rng.choice(self._actors[:-1])
        )
        if self._anchor is not None:
            # Retaining the anchor inside the ordinary capped list preserves
            # its draw cardinality through rotation. The extra choice must not
            # advance the stream used by native mixtures and subsequent deals.
            seed = hashlib.sha256(
                b"persistent-league-anchor-v1\0" + repr(rng.getstate()).encode()
            ).digest()
            anchor_rng = random.Random(seed)
            if anchor_rng.random() < self.anchor_fraction:
                entry = self._anchor
            elif entry["fingerprint"] == self._anchor["fingerprint"]:
                # Exclude natural anchor hits from the unanchored branch so
                # anchor_fraction means the actual share of league choices.
                candidates = self._actors[:-1] if draw >= 0.45 else self._actors
                candidates = [
                    old
                    for old in candidates
                    if old["fingerprint"] != self._anchor["fingerprint"]
                ]
                if not candidates:
                    candidates = [
                        old
                        for old in self._actors
                        if old["fingerprint"] != self._anchor["fingerprint"]
                    ]
                if candidates:
                    entry = (
                        candidates[-1] if draw < 0.45 else anchor_rng.choice(candidates)
                    )
        for seat in range(players):
            if seat % 2 != episode.learner % 2:
                episode.controllers[seat] = "league:" + entry["fingerprint"]
                episode.roles[seat] = True
        return episode

    def _configuration(self):
        return {
            "recipe": self.recipe,
            "learned_controllers": list(self.learned_controllers),
        }

    def state_dict(self):
        result = {
            "version": VERSION,
            "configuration": self._configuration(),
            "base_snapshots": [_copy_weights(state) for state in self.snapshots],
            "fixed_opponent": None
            if self.fixed_opponent is None
            else _copy_weights(self.fixed_opponent),
            "actors": [
                {
                    "fingerprint": entry["fingerprint"],
                    "choices": entry["choices"],
                    "model": _copy_weights(entry["model"]),
                }
                for entry in self._actors
            ],
        }
        if self._anchor is not None:
            result["anchor"] = {
                "fingerprint": self._anchor["fingerprint"],
                "model": _copy_weights(self._anchor["model"]),
            }
        return result

    def load_state_dict(self, state):
        """Validate completely before replacing any pool; caller restores its RNG."""
        if (
            not isinstance(state, dict)
            or type(state.get("version")) is not int
            or state["version"] != VERSION
            or state.get("configuration") != self._configuration()
        ):
            raise ValueError("league version or configuration differs")
        anchor = state.get("anchor")
        if self._anchor is None:
            if anchor is not None:
                raise ValueError("unexpected persistent anchor")
        else:
            if not isinstance(anchor, dict) or set(anchor) != {"fingerprint", "model"}:
                raise ValueError("missing or invalid persistent anchor")
            weights = _copy_weights(anchor["model"])
            if (
                _fingerprint(weights) != anchor["fingerprint"]
                or anchor["fingerprint"] != self._anchor["fingerprint"]
            ):
                raise ValueError("persistent anchor weights differ")
            _actor(weights)
        actors, base = state.get("actors"), state.get("base_snapshots")
        if (
            not isinstance(actors, list)
            or len(actors) > self.max_snapshots
            or not isinstance(base, list)
            or len(base) > MAX_SNAPSHOTS
        ):
            raise ValueError("invalid league snapshot count")
        restored, fingerprints, previous = [], set(), -1
        for entry in actors:
            if not isinstance(entry, dict) or set(entry) != {
                "fingerprint",
                "choices",
                "model",
            }:
                raise ValueError("invalid league entry")
            choices = entry["choices"]
            if type(choices) is not int or choices < 0 or choices < previous:
                raise ValueError("invalid league update order")
            weights = _copy_weights(entry["model"])
            fingerprint = _fingerprint(weights)
            if fingerprint != entry["fingerprint"] or fingerprint in fingerprints:
                raise ValueError("league fingerprint mismatch or duplicate")
            _actor(weights)
            restored.append(
                {"fingerprint": fingerprint, "choices": choices, "model": weights}
            )
            fingerprints.add(fingerprint)
            previous = choices
        if self._anchor is not None and self._anchor["fingerprint"] not in fingerprints:
            raise ValueError("persistent anchor missing from reserved league slot")
        restored_base = [_copy_weights(weights) for weights in base]
        fixed = state.get("fixed_opponent")
        restored_fixed = None if fixed is None else _copy_weights(fixed)
        if self.reactive and restored_fixed is not None:
            raise ValueError("reactive league cannot restore a fixed opponent")
        for weights in [
            *restored_base,
            *([] if restored_fixed is None else [restored_fixed]),
        ]:
            _legacy(weights)
        self._actors, self.snapshots, self.fixed_opponent = (
            restored,
            restored_base,
            restored_fixed,
        )
