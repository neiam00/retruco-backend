import copy
import unittest

from retruco_ml.assessment import ASSESSMENT_VERSION, SEAT_SCHEDULE
from retruco_ml.curriculum import phase_remaining
from retruco_ml.release_audit import (
    CONFIGURATIONS,
    configurations,
    quality,
    quantization,
    strict_json,
    teacher_preservation,
)


def report(quantized=True):
    return {
        "seed": 2000000,
        "assessment_version": ASSESSMENT_VERSION,
        "seat_schedule": SEAT_SCHEDULE,
        "checkpoint_sha256": "checkpoint",
        "quantized_sha256": "model" if quantized else None,
        "opponent": "baseline",
        "partner": "learned",
        "ablation": "none",
        "configurations": [
            {
                "players": p,
                "target": t,
                "flor": f,
                "pairs": 2000,
                "pair_scores": [1.0] * 1200 + [0.5] * 600 + [0.0] * 200,
                "win_rate": 0.01,
            }
            for p, t, f in sorted(CONFIGURATIONS)
        ],
    }


class ReleaseAuditTests(unittest.TestCase):
    def test_qat_must_preserve_the_original_teacher_not_only_its_updated_float_weights(
        self,
    ):
        teacher = report(False)
        teacher["checkpoint_sha256"] = "teacher"
        candidate = report()
        candidate["checkpoint_sha256"] = "student"
        self.assertEqual(
            teacher_preservation(teacher, candidate, "teacher", "student", "model")[
                "loss"
            ],
            0,
        )
        for row in candidate["configurations"]:
            row["pair_scores"] = [0.5] * 2000
        with self.assertRaisesRegex(ValueError, "original teacher"):
            teacher_preservation(teacher, candidate, "teacher", "student", "model")
        teacher["checkpoint_sha256"] = "student"
        with self.assertRaisesRegex(ValueError, "original float32 teacher"):
            teacher_preservation(teacher, candidate, "teacher", "student", "model")

    def test_nonfinite_evidence_is_rejected_including_json_exponent_overflow(self):
        for value in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999"):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                strict_json('{"duration":' + value + "}")

    def test_strength_recomputes_scores_instead_of_trusting_green_metrics(self):
        good = report()
        self.assertAlmostEqual(
            quality(good, "checkpoint", "model")["pooled"]["win_rate"], 0.75
        )
        bad = copy.deepcopy(good)
        for row in bad["configurations"]:
            row["win_rate"] = 1
            row["pair_scores"] = [0.0] * 2000
        with self.assertRaisesRegex(ValueError, "70%"):
            quality(bad, "checkpoint", "model")

    def test_separate_mode_targets_cannot_be_hidden_by_a_high_pooled_rate(self):
        for weak_mode, rate, text in [(2, 0.69, "70%"), (4, 0.59, "60%")]:
            candidate = report()
            for row in candidate["configurations"]:
                wins = int((rate if row["players"] == weak_mode else 0.95) * 2000)
                row["pair_scores"] = [1.0] * wins + [0.0] * (2000 - wins)
            with self.subTest(mode=weak_mode), self.assertRaisesRegex(ValueError, text):
                quality(candidate, "checkpoint", "model")
        acceptable = report()
        for row in acceptable["configurations"]:
            wins = 1400 if row["players"] == 2 else 1200
            row["pair_scores"] = [1.0] * wins + [0.0] * (2000 - wins)
        modes = quality(acceptable, "checkpoint", "model")["modes"]
        self.assertEqual(modes["2"]["target_win_rate"], 0.70)
        self.assertEqual(modes["4"]["target_win_rate"], 0.60)

    def test_release_rejects_smoke_seeds_missing_modes_and_nonfinite_scores(self):
        wrong = report()
        wrong["seed"] = 1000000
        with self.assertRaisesRegex(ValueError, "holdout"):
            configurations(wrong)
        missing = report()
        missing["configurations"].pop()
        with self.assertRaisesRegex(ValueError, "configurations"):
            configurations(missing)
        corrupt = report()
        corrupt["configurations"][0]["pair_scores"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "actual wins"):
            configurations(corrupt)

    def test_both_modes_must_improve_and_quantization_must_be_paired(self):
        bad = report()
        for row in bad["configurations"]:
            if row["players"] == 4:
                row["pair_scores"] = [0.5] * 2000
        with self.assertRaisesRegex(ValueError, "4-player"):
            quality(bad, "checkpoint", "model")
        reference = report(False)
        quantized = report()
        self.assertEqual(
            quantization(reference, quantized, "checkpoint", "model")["loss"], 0
        )
        quantized["seed"] += 1
        with self.assertRaisesRegex(ValueError, "unpaired"):
            quantization(reference, quantized, "checkpoint", "model")

    def test_phase_resume_counts_only_training_after_warm_start(self):
        self.assertEqual(phase_remaining(3600 + 7200, 3600, 12), 10)
        self.assertEqual(phase_remaining(3600 + 13 * 3600, 3600, 12), 0)
        with self.assertRaises(ValueError):
            phase_remaining(100, 3600, 12)
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                phase_remaining(value, 0, 12)


if __name__ == "__main__":
    unittest.main()
