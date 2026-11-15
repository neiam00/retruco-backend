"""Behavioral protocol guards for a checkpoint-preserving factorial study."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from retruco_ml.cooperation_refusal_campaign import (
    coverage,
    preservation_snapshot,
    validate_config,
    validate_inputs,
    verify_preservation,
)
from retruco_ml.cooperation_refusal_compare import compare, validate_report

CONFIG = Path(__file__).resolve().parents[1] / "experiments/cooperation-refusal-v1.json"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def protect(path):
    path.with_suffix(".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest()
    )


def report(spec, value=0.5, pairs=8, seed=181000):
    return {
        "protocol": "improvement-evaluation-v1",
        "seed": seed,
        "pairs": pairs,
        "partner": spec["partner"],
        "opponent": spec["opponent"],
        "signals": "normal",
        "card_policy": "policy",
        "checkpoint_sha256": "actor",
        "inputs_sha256": "inputs",
        "behavior": {"truthful_claims": 8, "claim_opportunities": 10},
        "decision_contexts": {"partner_claim": {"accepts": 1, "declines": 2}},
        "configurations": [
            {
                "players": spec["players"],
                "target": target,
                "flor": flor,
                "pairs": pairs,
                "pair_scores": [value] * pairs,
                "pair_net_points_per_hand": [0.0] * pairs,
                "win_rate": value,
                "matches": pairs * spec["players"],
            }
            for target in (12, 24)
            for flor in (False, True)
        ],
    }


class CooperationRefusalCampaignTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text())

    def test_rejects_mislabeled_factorial_arms_mixed_benchmarks_and_overlapping_deals(
        self,
    ):
        validate_config(self.config)
        mutations = [
            lambda c: c["arms"]["B"].update(anchor=True),
            lambda c: c["primary_specs"][0].update(opponent="parent"),
            lambda c: c.update(diagnostic_seed=c["primary_seed"] + 1),
            lambda c: c.update(seeds=[401, 402]),
            lambda c: c.update(parent_anchor=float("nan")),
            lambda c: c.update(environments=0),
        ]
        for change in mutations:
            bad = copy.deepcopy(self.config)
            change(bad)
            with self.subTest(config=bad), self.assertRaises(ValueError):
                validate_config(bad)

    def test_report_guard_rejects_partial_coverage_changed_parent_and_unpaired_deals(
        self,
    ):
        spec = self.config["primary_specs"][0]
        original = report(spec)
        validate_report(original, spec, 8, 181000, "actor", "inputs")
        mutations = [
            lambda r: r["configurations"].pop(),
            lambda r: r.update(checkpoint_sha256="other"),
            lambda r: r.update(seed=181001),
            lambda r: r["configurations"][0].update(pair_scores=[1.0]),
            lambda r: r.update(signals="no-send"),
            lambda r: r["configurations"].__setitem__(1, r["configurations"][0]),
            lambda r: r.update(protocol="different-protocol"),
            lambda r: r["configurations"][0].update(matches=1),
            lambda r: r["configurations"][0].update(win_rate=0.99),
            lambda r: r["configurations"][0]["pair_net_points_per_hand"].__setitem__(
                0, float("nan")
            ),
        ]
        for change in mutations:
            bad = copy.deepcopy(original)
            change(bad)
            with self.subTest(report=bad), self.assertRaises(ValueError):
                validate_report(bad, spec, 8, 181000, "actor", "inputs")

    def test_preservation_checks_bytes_and_missing_files_without_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            old = repository / "ml/runs/old/checkpoint.pt"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"preserved checkpoint")
            output = repository / "ml/runs/new"
            output.mkdir()
            (output / "new.pt").write_bytes(b"new data")
            manifest = preservation_snapshot(repository, output)
            self.assertEqual(len(manifest["files"]), 1)
            self.assertTrue(verify_preservation(manifest)["passed"])
            old.write_bytes(b"corrupted checkpoint")
            self.assertEqual(
                verify_preservation(manifest)["mismatches"], [str(old.resolve())]
            )
            self.assertEqual(old.read_bytes(), b"corrupted checkpoint")
            old.unlink()
            self.assertFalse(verify_preservation(manifest)["passed"])

    def test_inputs_guard_rejects_outside_paths_and_changed_stronger_starter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"fixed parent"
            (root / "legacy-champion-lstm16.pt").write_bytes(data)
            entry = {
                "file": "legacy-champion-lstm16.pt",
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            manifest = {"candidates": {"legacy-champion": entry}, "reference": entry}
            write(root / "manifest.json", manifest)
            validate_inputs(root)
            (root / "legacy-champion-lstm16.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                validate_inputs(root)
            (root / "legacy-champion-lstm16.pt").write_bytes(data)
            manifest["reference"] = {"file": "../outside.pt", "sha256": "invalid"}
            write(root / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                validate_inputs(root)

    def test_comparison_keeps_every_endpoint_separates_native_targets_and_identifies_factorial_effects(
        self,
    ):
        config = copy.deepcopy(self.config)
        config.update(primary_pairs=8, diagnostic_pairs=8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "inputs/manifest.json", {"kind": "test-frozen-inputs"})
            starter = root / "inputs/legacy-champion-lstm16.pt"
            starter.write_bytes(b"independent parent artifact")
            starter_sha = hashlib.sha256(starter.read_bytes()).hexdigest()
            inputs_sha = hashlib.sha256(
                (root / "inputs/manifest.json").read_bytes()
            ).hexdigest()
            for family in ("primary", "diagnostic"):
                for spec in config[family + "_specs"]:
                    value = 0.75 if spec["players"] == 2 else 0.5
                    result = report(spec, value, seed=config[family + "_seed"])
                    result.update(
                        checkpoint_sha256=starter_sha, inputs_sha256=inputs_sha
                    )
                    path = root / "baseline" / family / (spec["id"] + ".json")
                    write(path, result)
                    protect(path)
            for arm, value in zip("ABCD", (0.5, 0.625, 0.75, 0.875)):
                for seed in config["seeds"]:
                    folder = root / "trials" / f"{arm}-seed-{seed}"
                    write(
                        folder / "status.json",
                        {
                            "status": "complete",
                            "choices": 200100,
                            "update": 25,
                            "warmups": 2,
                        },
                    )
                    metric = {
                        "phase": "training",
                        "choices": 200100,
                        "collect_seconds": 100,
                        "optimize_seconds": 100,
                        "diagnostics": {"replay_identity_max_logp_error": 1e-5},
                    }
                    (folder / "metrics.jsonl").write_text(json.dumps(metric) + "\n")
                    candidate = folder / "candidate.pt"
                    candidate.write_bytes(f"trained endpoint {arm} {seed}".encode())
                    protect(candidate)
                    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
                    for family in ("primary", "diagnostic"):
                        for spec in config[family + "_specs"]:
                            result = report(
                                spec,
                                0.75 if spec["players"] == 2 else value,
                                seed=config[family + "_seed"],
                            )
                            result.update(
                                checkpoint_sha256=candidate_sha,
                                inputs_sha256=inputs_sha,
                            )
                            path = folder / family / (spec["id"] + ".json")
                            write(path, result)
                            protect(path)
            compared = compare(root, config)
            self.assertEqual(len(compared["individual"]), 12)
            self.assertEqual(
                compared["individual"]["B-seed-401"]["modes"]["4"]["win_rate"], 0.625
            )
            self.assertTrue(
                compared["individual"]["B-seed-401"]["modes"]["2"]["target_met"]
            )
            self.assertFalse(
                compared["individual"]["B-seed-401"]["modes"]["4"]["target_met"]
            )
            self.assertEqual(
                compared["effects"]["4"]["cooperation_population_bundle"]["gain"], 0.125
            )
            self.assertEqual(
                compared["effects"]["4"]["targeted_anchor_bundle"]["gain"], 0.25
            )
            self.assertEqual(compared["effects"]["4"]["interaction"]["gain"], 0.0)
            self.assertEqual(compared["exploratory_best_arm"], "D")
            self.assertFalse(compared["approved"])
            path = root / "trials/A-seed-401/primary/native-headsup.json"
            data = json.loads(path.read_text())
            data["checkpoint_sha256"] = "different valid-looking actor"
            write(path, data)
            protect(path)
            with self.assertRaises(ValueError):
                compare(root, config)
            data["checkpoint_sha256"] = hashlib.sha256(
                (root / "trials/A-seed-401/candidate.pt").read_bytes()
            ).hexdigest()
            write(path, data)
            protect(path)
            write(
                root / "trials/A-seed-401/status.json",
                {"status": "failed", "choices": 200100},
            )
            with self.assertRaises(ValueError):
                compare(root, config)

    def test_coverage_requires_valid_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "baseline/primary/native-headsup.json"
            write(path, {"complete": True})
            self.assertEqual(
                coverage(root, self.config)["completed_primary_and_diagnostic"], 0
            )
            path.with_suffix(".sha256").write_text(
                hashlib.sha256(path.read_bytes()).hexdigest()
            )
            self.assertEqual(
                coverage(root, self.config)["completed_primary_and_diagnostic"], 1
            )
            write(path, {"complete": False})
            self.assertEqual(
                coverage(root, self.config)["completed_primary_and_diagnostic"], 0
            )


if __name__ == "__main__":
    unittest.main()
