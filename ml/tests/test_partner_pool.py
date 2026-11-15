"""Immutable partner provenance, role preservation, and reproducible populations."""

import copy
import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch

from retruco_ml.benchmark_population import NativePracticePopulation
from retruco_ml.cooperation_practice import CooperationPracticePopulation
from retruco_ml.historical_league import HistoricalLeaguePopulation, _fingerprint
from retruco_ml.partner_pool import load_partner_pool, prepare_partner_pool
from retruco_ml.recurrent import Actor, file_hash


class PartnerPoolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(84)

    def test_preparation_preserves_specialist_mode_and_trainable_head_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = Actor(True, 64, team_adapter="contextual")
            with torch.no_grad():
                actor.team_head[2].weight.fill_(0.15)
            source = root / "source.pt"
            torch.save(
                {
                    "model": actor.state_dict(),
                    "configuration": {
                        "actor_memory": 64,
                        "team_adapter": "contextual",
                        "team_adapter_version": 1,
                    },
                },
                source,
            )
            digest = file_hash(source)
            spec = root / "spec.json"
            spec.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "entries": [
                            {
                                "id": "team",
                                "family": "adapted",
                                "source": str(source),
                                "lineage": [digest],
                            }
                        ],
                    }
                )
            )
            manifest = prepare_partner_pool(spec, root / "copies")
            loaded = load_partner_pool(manifest).models()["practice:pool:team"]
            self.assertEqual(loaded.team_adapter, "contextual")
            for key, value in actor.state_dict().items():
                torch.testing.assert_close(
                    loaded.state_dict()[key], value, atol=0, rtol=0
                )
            self.assertEqual(file_hash(source), digest)

    def prepare(self, root, name="pool", offset=0):
        sources = root / ("sources-" + name)
        sources.mkdir()
        entries, actors, hashes = [], {}, {}
        for index, size in enumerate((16, 64, 16)):
            actor = Actor(True, memory_size=size)
            with torch.no_grad():
                actor.signal.bias.add_(offset + index * 0.1)
            path = sources / f"actor-{index}.pt"
            torch.save(
                {"model": actor.state_dict(), "configuration": {"actor_memory": size}},
                path,
            )
            digest = file_hash(path)
            entries.append(
                {
                    "id": f"partner-{index}",
                    "family": "family-a" if index == 0 else "family-b",
                    "source": str(path),
                    "lineage": [digest],
                }
            )
            actors[f"partner-{index}"] = actor
            hashes[path] = digest
        spec = sources / "sources.json"
        spec.write_text(json.dumps({"version": 1, "entries": entries}))
        hashes[spec] = file_hash(spec)
        manifest = prepare_partner_pool(spec, root / name)
        return manifest, actors, hashes, spec

    def test_preparation_preserves_sources_tensors_and_rng_with_independent_frozen_models(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, actors, hashes, spec = self.prepare(root)
            before = torch.get_rng_state().clone()
            pool = load_partner_pool(manifest)
            registry = pool.models()
            torch.testing.assert_close(before, torch.get_rng_state(), atol=0, rtol=0)
            for name, actor in actors.items():
                frozen = registry["practice:pool:" + name]
                self.assertFalse(frozen.training)
                self.assertTrue(
                    all(
                        not parameter.requires_grad for parameter in frozen.parameters()
                    )
                )
                for key, value in actor.state_dict().items():
                    torch.testing.assert_close(
                        value, frozen.state_dict()[key], atol=0, rtol=0
                    )
                self.assertEqual(frozen.memory.hidden_size, actor.memory.hidden_size)
                with torch.no_grad():
                    frozen.signal.bias.add_(2)
                torch.testing.assert_close(
                    pool.models()["practice:pool:" + name].signal.bias,
                    actor.signal.bias,
                    atol=0,
                    rtol=0,
                )
            for path, digest in hashes.items():
                self.assertEqual(file_hash(path), digest)
            with self.assertRaisesRegex(ValueError, "new output"):
                prepare_partner_pool(spec, root / "pool")
            configuration = pool.configuration
            configuration["entries"][0]["family"] = "modified"
            self.assertNotEqual(pool.configuration, configuration)

    def test_pool_changes_only_starter_partner_slots_and_samples_every_family(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _, _, _ = self.prepare(Path(directory))
            pool = load_partner_pool(manifest)
            actor = Actor(True, memory_size=64)
            base = NativePracticePopulation(partner_actor=actor, native_fraction=0.5)
            mixed = NativePracticePopulation(
                partner_actor=actor, partner_pool=pool, native_fraction=0.5
            )
            for population in (base, mixed):
                population.add_actor(actor, 0)
            counts = Counter()
            for seed in range(2000):
                left, right = random.Random(seed), random.Random(seed)
                self.assertEqual(base.episode(2, left), mixed.episode(2, right))
                self.assertEqual(left.getstate(), right.getstate())
                left, right = random.Random(seed), random.Random(seed)
                original = base.episode(4, left)
                changed = mixed.episode(4, right)
                self.assertEqual(left.getstate(), right.getstate())
                partner = (original.learner + 2) % 4
                self.assertEqual(original.learner, changed.learner)
                self.assertEqual(original.roles, changed.roles)
                for seat in range(4):
                    if (
                        seat != partner
                        or original.controllers[partner] != "practice:starter"
                    ):
                        self.assertEqual(
                            original.controllers[seat], changed.controllers[seat]
                        )
                if original.controllers[partner] == "practice:starter":
                    counts[changed.controllers[partner]] += 1
            self.assertEqual(set(counts), set(pool.models()))
            self.assertTrue(all(count > 20 for count in counts.values()))
            self.assertTrue(
                0.4 < counts["practice:pool:partner-0"] / sum(counts.values()) < 0.6
            )

    def test_policy_version_remaps_only_declared_teammates_without_rng_or_role_changes(
        self,
    ):
        actor = Actor(True)
        legacy = CooperationPracticePopulation(partner_actor=actor)
        explicit = CooperationPracticePopulation(
            partner_actor=actor, partner_pool=None, partner_policy_version=2
        )
        updated = CooperationPracticePopulation(
            partner_actor=actor, partner_policy_version=3
        )
        historical = HistoricalLeaguePopulation()
        self.assertEqual(legacy.recipe, explicit.recipe)
        self.assertNotIn("partner_pool", json.loads(legacy.recipe))
        changed = set()
        for seed in range(600):
            a, b, c = random.Random(seed), random.Random(seed), random.Random(seed)
            self.assertEqual(legacy.episode(2, a), historical.episode(2, b))
            self.assertEqual(a.getstate(), b.getstate())
            original, same = legacy.episode(4, a), explicit.episode(4, b)
            # Compare v3 against the same initial stream, including the 1v1 call.
            c = random.Random(seed)
            updated.episode(2, c)
            new = updated.episode(4, c)
            self.assertEqual(original, same)
            self.assertEqual(a.getstate(), b.getstate())
            self.assertEqual(a.getstate(), c.getstate())
            self.assertEqual(original.roles, new.roles)
            partner = (original.learner + 2) % 4
            for seat, controller in enumerate(original.controllers):
                expected = controller
                if seat == partner and controller in (
                    "script:literal-v2",
                    "script:trusting-v2",
                    "script:cautious-v2",
                ):
                    expected = controller[:-1] + "3"
                    changed.add(expected)
                self.assertEqual(new.controllers[seat], expected)
        self.assertEqual(
            changed, {"script:literal-v3", "script:trusting-v3", "script:cautious-v3"}
        )
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            updated.load_state_dict(legacy.state_dict())

    def test_heldout_hash_lineage_and_repacked_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, actors, hashes, spec = self.prepare(root)
            document = json.loads(manifest.read_text())
            entry = document["entries"][0]
            for forbidden in (entry["sha256"], entry["lineage"][0]):
                with (
                    self.subTest(forbidden=forbidden),
                    self.assertRaisesRegex(ValueError, "held-out"),
                ):
                    load_partner_pool(manifest, forbidden_hashes=[forbidden])
            with self.assertRaisesRegex(ValueError, "held-out partner weights"):
                load_partner_pool(
                    manifest,
                    forbidden_fingerprints=[
                        _fingerprint(actors["partner-0"].state_dict())
                    ],
                )
            with self.assertRaisesRegex(ValueError, "held-out"):
                prepare_partner_pool(
                    spec, root / "blocked", forbidden_hashes=[entry["lineage"][0]]
                )
            self.assertFalse((root / "blocked").exists())
            for path, digest in hashes.items():
                self.assertEqual(file_hash(path), digest)
            changed = json.loads(spec.read_text())
            changed["entries"][0]["source_sha256"] = "0" * 64
            bad_spec = root / "wrong-source.json"
            bad_spec.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "declared checkpoint hash"):
                prepare_partner_pool(bad_spec, root / "wrong-source-pool")
            self.assertFalse((root / "wrong-source-pool").exists())

    def test_tampered_files_paths_lineage_and_resume_configuration_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _, _, _ = self.prepare(root)
            pool = load_partner_pool(manifest)
            actor = Actor(True)
            population = CooperationPracticePopulation(
                partner_actor=actor, partner_pool=pool
            )
            state = population.state_dict()
            restored = CooperationPracticePopulation(
                partner_actor=actor, partner_pool=load_partner_pool(manifest)
            )
            restored.load_state_dict(state)
            other, _, _, _ = self.prepare(root, "other", offset=1)
            changed = CooperationPracticePopulation(
                partner_actor=actor, partner_pool=load_partner_pool(other)
            )
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                changed.load_state_dict(state)
            original = json.loads(manifest.read_text())
            for field, value in (
                ("file", "../outside.pt"),
                ("file", str(manifest)),
                ("lineage", []),
            ):
                altered = copy.deepcopy(original)
                altered["entries"][0][field] = value
                manifest.write_text(json.dumps(altered))
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    load_partner_pool(manifest)
            outside = root / "outside.pt"
            outside.write_bytes(
                (manifest.parent / original["entries"][0]["file"]).read_bytes()
            )
            (manifest.parent / "escape.pt").symlink_to(outside)
            altered = copy.deepcopy(original)
            altered["entries"][0]["file"] = "escape.pt"
            manifest.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "outside"):
                load_partner_pool(manifest)
            manifest.write_text(json.dumps(original))
            pool = load_partner_pool(manifest)
            checkpoint = manifest.parent / original["entries"][0]["file"]
            checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                pool.models()
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_partner_pool(manifest)


if __name__ == "__main__":
    unittest.main()
