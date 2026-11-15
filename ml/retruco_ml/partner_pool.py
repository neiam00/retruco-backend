"""Immutable, provenance-bound training partners; never actor input features."""

import argparse
import copy
import hashlib
import json
import random
import re
from pathlib import Path

import torch

from .historical_league import _copy_weights, _fingerprint
from .model_capacity import actor_from_checkpoint, team_adapter_configuration
from .recurrent import file_hash

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_HASH = re.compile(r"[0-9a-f]{64}")
MAX_PARTNERS = 32


def _identity(entry):
    if any(
        not isinstance(entry.get(key), str) or not _NAME.fullmatch(entry[key])
        for key in ("id", "family")
    ):
        raise ValueError("partner id and family require simple nonempty names")
    lineage = entry.get("lineage")
    if (
        not isinstance(lineage, list)
        or not lineage
        or any(
            not isinstance(value, str) or not _HASH.fullmatch(value)
            for value in lineage
        )
    ):
        raise ValueError(
            "partner lineage requires declared source or ancestor SHA256 hashes"
        )
    return sorted(set(lineage))


def _entries(document):
    entries = document.get("entries") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or type(document.get("version")) is not int
        or document["version"] != 1
        or not isinstance(entries, list)
        or not 1 <= len(entries) <= MAX_PARTNERS
        or any(not isinstance(entry, dict) for entry in entries)
    ):
        raise ValueError("partner manifest requires version 1 and one to 32 entries")
    return entries


def _actor(saved):
    if (
        not isinstance(saved, dict)
        or saved.get("encoder", 2) != 2
        or saved.get("memory_version", 1) != 1
    ):
        raise ValueError("incompatible partner observation contract")
    weights = _copy_weights(saved.get("model"))
    with torch.random.fork_rng(devices=[]):
        model = actor_from_checkpoint({**saved, "model": weights})
    if not model.recurrent:
        raise ValueError("partner pool requires recurrent actors")
    return model.eval().requires_grad_(False), _fingerprint(weights)


class FrozenPartnerPool:
    """Sample families uniformly, then a checkpoint uniformly within its family."""

    def __init__(self, manifest, manifest_sha256, entries, models):
        self._manifest = manifest
        self._manifest_sha256 = manifest_sha256
        self._entries = copy.deepcopy(entries)
        self._models = models
        self._families = {}
        for entry in entries:
            self._families.setdefault(entry["family"], []).append(
                "practice:pool:" + entry["id"]
            )
        self._family_names = tuple(sorted(self._families))

    @property
    def configuration(self):
        return {
            "version": 1,
            "manifest_sha256": self._manifest_sha256,
            "sampling": "uniform-family-then-uniform-checkpoint",
            "match_rng": "domain-separated snapshot without advancing environment RNG",
            "entries": copy.deepcopy(self._entries),
        }

    def verify(self):
        if file_hash(self._manifest) != self._manifest_sha256:
            raise ValueError("partner manifest changed")
        for entry in self._entries:
            path = (self._manifest.parent / entry["file"]).resolve()
            if (
                not path.is_relative_to(self._manifest.parent)
                or file_hash(path) != entry["sha256"]
            ):
                raise ValueError(
                    "partner checkpoint changed or outside manifest directory"
                )

    def sample(self, rng):
        return rng.choice(self._families[rng.choice(self._family_names)])

    def sample_match(self, rng):
        # Native-opponent substitution may run after partner assignment. Do not
        # shift that draw (or later deals) merely by enabling teammate diversity.
        seed = hashlib.sha256(
            b"frozen-teammate-v1\0" + repr(rng.getstate()).encode()
        ).digest()
        return self.sample(random.Random(seed))

    def models(self):
        self.verify()
        return {
            name: copy.deepcopy(actor).eval().requires_grad_(False)
            for name, actor in self._models.items()
        }


def load_partner_pool(path, forbidden_hashes=(), forbidden_fingerprints=()):
    manifest = Path(path).resolve()
    digest = file_hash(manifest)
    document = json.loads(manifest.read_text())
    entries, models, identities, fingerprints = [], {}, set(), set()
    forbidden_hashes, forbidden_fingerprints = (
        set(forbidden_hashes),
        set(forbidden_fingerprints),
    )
    for entry in _entries(document):
        lineage = _identity(entry)
        name = entry["id"]
        relative = entry.get("file")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("partner files must be confined relative paths")
        checkpoint = (manifest.parent / relative).resolve()
        if not checkpoint.is_relative_to(manifest.parent):
            raise ValueError("partner checkpoint outside manifest directory")
        expected = entry.get("sha256")
        if (
            not isinstance(expected, str)
            or not _HASH.fullmatch(expected)
            or file_hash(checkpoint) != expected
        ):
            raise ValueError("partner checkpoint SHA256 differs")
        if name in identities:
            raise ValueError("duplicate partner identity")
        if forbidden_hashes.intersection([expected, *lineage]):
            raise ValueError("held-out partner checkpoint or declared lineage")
        actor, fingerprint = _actor(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        if fingerprint in forbidden_fingerprints:
            raise ValueError("held-out partner weights")
        if fingerprint in fingerprints:
            raise ValueError("duplicate partner weights do not add diversity")
        if entry.get("fingerprint", fingerprint) != fingerprint:
            raise ValueError("partner weight fingerprint differs")
        identities.add(name)
        fingerprints.add(fingerprint)
        models["practice:pool:" + name] = actor
        entries.append(
            {
                "id": name,
                "family": entry["family"],
                "file": relative,
                "sha256": expected,
                "lineage": lineage,
                "fingerprint": fingerprint,
                "actor_memory": actor.memory.hidden_size,
            }
        )
    result = FrozenPartnerPool(manifest, digest, entries, models)
    result.verify()
    return result


def prepare_partner_pool(
    spec_path, output, forbidden_hashes=(), forbidden_fingerprints=()
):
    """Create new actor-only copies; source files and existing outputs are immutable."""
    spec_path, output = Path(spec_path).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("partner preparation requires a new output directory")
    spec_sha = file_hash(spec_path)
    rows, provenance, identities, fingerprints = [], [], set(), set()
    forbidden_hashes, forbidden_fingerprints = (
        set(forbidden_hashes),
        set(forbidden_fingerprints),
    )
    for entry in _entries(json.loads(spec_path.read_text())):
        lineage = _identity(entry)
        source_name = entry.get("source")
        if not isinstance(source_name, str) or not source_name:
            raise ValueError("partner source must be an explicit checkpoint path")
        source = (spec_path.parent / source_name).resolve()
        digest = file_hash(source)
        if entry.get("source_sha256", digest) != digest:
            raise ValueError("partner source differs from the declared checkpoint hash")
        lineage = sorted(set([digest, *lineage]))
        if forbidden_hashes.intersection(lineage):
            raise ValueError("held-out partner source or declared lineage")
        actor, fingerprint = _actor(
            torch.load(source, map_location="cpu", weights_only=True)
        )
        if fingerprint in forbidden_fingerprints:
            raise ValueError("held-out partner weights")
        if entry["id"] in identities or fingerprint in fingerprints:
            raise ValueError("duplicate partner identity or weights")
        identities.add(entry["id"])
        fingerprints.add(fingerprint)
        rows.append((entry, actor, fingerprint, lineage))
        provenance.append(
            {
                "id": entry["id"],
                "source": str(source),
                "source_sha256": digest,
                "fingerprint": fingerprint,
                "lineage": lineage,
            }
        )
    # Validate the complete source set before creating anything.
    output.mkdir(parents=True, exist_ok=False)
    manifest_entries = []
    for entry, actor, fingerprint, lineage in rows:
        destination = output / (entry["id"] + ".pt")
        torch.save(
            {
                "model": _copy_weights(actor.state_dict()),
                "encoder": 2,
                "memory_version": 1,
                "configuration": {
                    "actor_memory": actor.memory.hidden_size,
                    **team_adapter_configuration(actor.state_dict()),
                },
                "approved": False,
                "signal_policy": "truthful-cpu-v1",
            },
            destination,
        )
        manifest_entries.append(
            {
                "id": entry["id"],
                "family": entry["family"],
                "file": destination.name,
                "sha256": file_hash(destination),
                "fingerprint": fingerprint,
                "lineage": lineage,
            }
        )
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps({"version": 1, "entries": manifest_entries}, indent=2) + "\n"
    )
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "version": 1,
                "spec_sha256": spec_sha,
                "manifest_sha256": file_hash(manifest),
                "sources": provenance,
                "lineage_scope": "declared ancestry, not independently reconstructed",
            },
            indent=2,
        )
        + "\n"
    )
    if file_hash(spec_path) != spec_sha or any(
        file_hash(Path(row["source"])) != row["source_sha256"] for row in provenance
    ):
        raise ValueError("partner sources changed during preparation")
    load_partner_pool(manifest, forbidden_hashes, forbidden_fingerprints)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forbidden-sha256", action="append", default=[])
    parser.add_argument("--forbidden-fingerprint", action="append", default=[])
    args = parser.parse_args()
    print(
        prepare_partner_pool(
            args.spec, args.output, args.forbidden_sha256, args.forbidden_fingerprint
        )
    )


if __name__ == "__main__":
    main()
