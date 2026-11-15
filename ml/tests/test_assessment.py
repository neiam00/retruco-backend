import hashlib
import random
import tempfile
import unittest
from pathlib import Path

import torch

from retruco_ml.artifacts import read_quantized, validate_manifest
from retruco_ml.assessment import ablate, assess, assess_parallel
from retruco_ml.evaluation_styles import UNSEEN_STYLES, opponent_for_pair
from retruco_ml.export import export
from retruco_ml.model import Policy
from retruco_ml.population import Population
from retruco_ml.statistics import paired_loss, pooled


class AssessmentTests(unittest.TestCase):
    def test_held_out_scripted_styles_are_not_used_in_training(self):
        rng = random.Random(19)
        population = Population()
        seen = set()
        for _ in range(5000):
            episode = population.episode(4, rng)
            seen.update(
                c.split(":", 1)[1]
                for c in episode.controllers
                if c.startswith("script:")
            )
        self.assertTrue(seen)
        self.assertTrue(set(UNSEEN_STYLES).isdisjoint(seen))
        evaluation = {opponent_for_pair("population", 230000, i) for i in range(1000)}
        self.assertTrue(set(UNSEEN_STYLES).issubset(evaluation))

    def test_manifest_binds_compatibility_types_and_actual_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = export(Policy(), temporary)
            data = (Path(temporary) / "policy.bin").read_bytes()
            wasm = b"runtime-bytes"
            manifest.update(
                wasm_sha256=hashlib.sha256(wasm).hexdigest(),
                wasm_bytes=len(wasm),
                actor_parameters=85127,
            )
            validate_manifest(manifest, data, wasm)
            for key, value in (
                ("encoder", 999),
                ("action_encoder", 999),
                ("format", True),
                ("approved", "false"),
                ("rules", "unknown"),
                ("bytes", len(data) + 1),
                ("sha256", "wrong"),
            ):
                with self.assertRaises(ValueError):
                    validate_manifest({**manifest, key: value}, data, wasm)
            with self.assertRaises(ValueError):
                validate_manifest(manifest, data, wasm + b"changed")

    def test_parallel_assessment_preserves_deals_roles_streams_and_counts(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        model = Policy()
        opponent = Policy()
        partner = Policy()
        for options in (
            {"players": 2, "opponent": "baseline"},
            {
                "players": 4,
                "opponent": "deceptive",
                "partner": "literal",
                "ablation": "signals",
            },
            {
                "players": 4,
                "opponent_model": opponent,
                "partner_model": partner,
                "ablation": "adaptation",
            },
            {"players": 4, "opponent": "patient", "partner": "literal"},
            {"players": 2, "opponent": "counter-raiser"},
            {"players": 4, "opponent": "population", "ablation": "adaptation"},
        ):
            args = dict(
                model=model, pairs=4, target=12, flor=True, seed=35000, **options
            )
            reference = assess(**args)
            for workers, environments in ((1, 1), (2, 8)):
                batch = assess_parallel(
                    **args, workers=workers, environments=environments
                )
                for key in ("pair_scores", "actions", "win_rate"):
                    self.assertEqual(
                        batch[key],
                        reference[key],
                        (options, workers, environments, key),
                    )

    def test_binary_decoder_checks_format_and_preserves_actor_outputs(self):
        torch.manual_seed(2)
        model = Policy()
        with tempfile.TemporaryDirectory() as temporary:
            export(model, temporary)
            data = (Path(temporary) / "policy.bin").read_bytes()
            quantized = read_quantized(data, model)
            with torch.no_grad():
                state = torch.randn(1, 768)
                actions = torch.randn(1, 17, 48)
                reference = model(state, actions)[0]
                decoded = quantized(state, actions)[0]
            self.assertLess(float((reference - decoded).abs().max()), 0.01)
            for invalid in (data[:-1], data + b"\x00", b"FAIL" + data[4:]):
                with self.assertRaises(ValueError):
                    read_quantized(invalid)

    def test_pairing_does_not_treat_configurations_as_independent_seeds(self):
        configurations = [
            {
                "players": 2,
                "target": target,
                "flor": False,
                "pairs": 4,
                "pair_scores": [0, 1, 0, 1],
            }
            for target in (12, 24)
        ]
        summary = pooled(configurations)
        self.assertEqual(summary["seed_clusters"], 4)
        self.assertEqual(summary["win_rate"], 0.5)
        reference = {"seed": 123, "configurations": configurations}
        self.assertEqual(paired_loss(reference, reference)["loss"], 0)
        with self.assertRaises(ValueError):
            paired_loss(reference, dict(reference, seed=124))

    def test_ablations_keep_current_hand_and_private_card_features(self):
        observation = {"features": [1.0] * 768}
        adapted = ablate(observation, "adaptation")["features"]
        self.assertEqual(adapted[:192], [1.0] * 192)
        self.assertEqual(adapted[192:352], [1.0] * 160)
        self.assertEqual(adapted[512:716], [0.0] * 204)
        self.assertEqual(observation["features"], [1.0] * 768)

    def test_assessment_runs_real_matches_with_literal_partners_and_deceptive_opponents(
        self,
    ):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        report = assess(
            Policy(),
            2,
            4,
            12,
            True,
            35000,
            opponent="deceptive",
            partner="literal",
            ablation="signals",
        )
        self.assertEqual(len(report["pair_scores"]), 2)
        self.assertTrue(all(0 <= score <= 1 for score in report["pair_scores"]))
        self.assertNotIn("false_signal_claim", report["actions"])
        self.assertGreater(report["actions"].get("signal:none", 0), 0)


if __name__ == "__main__":
    unittest.main()
