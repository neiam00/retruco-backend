import json
import tempfile
import unittest
from pathlib import Path

from test_rmappo_controls_compare import digest, report_for, write_json

from retruco_ml.rmappo_target_campaign import record_frozen_nominee
from retruco_ml.rmappo_target_compare import (
    coverage,
    expected_reports,
    select_candidates,
    summarize,
)

REPOSITORY = Path(__file__).resolve().parents[2]


def fixture(root, seeds=(3101, 3102, 3103), rates=None, version=1):
    previous = json.loads((REPOSITORY / "ml/experiments/rmappo-v1.json").read_text())
    config = {
        "seeds": list(seeds),
        "primary_specs": previous["primary_specs"],
        "diagnostic_specs": previous["diagnostic_specs"],
        "selection_pairs": 2,
        "selection_seed": 135000,
        "selection_diagnostic_pairs": 2,
        "selection_diagnostic_seed": 136000,
        "final_pairs": 2,
        "final_seed": 137000,
        "final_diagnostic_pairs": 2,
        "final_diagnostic_seed": 138000,
    }
    profiles = {
        4: ("protected", "focused"),
        3: ("control", "targeted"),
        2: ("gae99", "mc"),
    }.get(version, ("original", "target"))
    config.update(version=version, profiles=list(profiles))
    write_json(root / "inputs/manifest.json", {"immutable": True})
    names = ["reference"] + [f"parent-{s}" for s in seeds]
    if version in (3, 4):
        names.append("prior-nominee")
    names += [f"{recipe}-{s}-c500000" for recipe in profiles for s in seeds]
    models = {}
    for name in names:
        path = root / "checkpoints" / (name + ".pt")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode())
        models[name] = path
    for job in expected_reports(root, config, models):
        spec = job["spec"]
        value = (rates or {}).get(job["model"], (0.55, 0.55))
        score = value[0 if spec["players"] == 2 else 1]
        report = report_for(
            spec,
            job["pairs"],
            job["seed"],
            job["checkpoint"],
            digest(root / "inputs/manifest.json"),
            score,
        )
        if version == 4:
            report["response_families"] = {}
        write_json(job["path"], report, receipt=True)
    return config, models


class RMAPPOObjectiveCompareTests(unittest.TestCase):
    def test_final_miss_retains_calibrated_starter_and_preserves_the_frozen_nominee(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rates = {
                "parent-3101": (0.72, 0.58),
                "protected-3101-c500000": (0.72, 0.59),
                "focused-3101-c500000": (0.72, 0.62),
            }
            config, models = fixture(root, seeds=(3101,), rates=rates, version=4)
            choice = select_candidates(summarize(root, config, models), config)
            final = {
                name: path
                for name, path in models.items()
                if not name.endswith("c500000")
            }
            final["selected-3101"] = models[choice["nominee"]]
            for job in expected_reports(root, config, final, "final"):
                s = job["spec"]
                score = (
                    0.695
                    if s["players"] == 2
                    else 0.62
                    if job["model"] == "selected-3101"
                    else 0.58
                )
                report = report_for(
                    s,
                    job["pairs"],
                    job["seed"],
                    job["checkpoint"],
                    digest(root / "inputs/manifest.json"),
                    score,
                )
                report["response_families"] = {}
                write_json(job["path"], report, receipt=True)
            result = summarize(root, config, final, "final")
            record_frozen_nominee(result, choice, config)
            self.assertFalse(result["frozen_nominee_result"]["starter_fallback"])
            self.assertFalse(
                result["frozen_nominee_result"]["protected_gate"]["qualified"]
            )
            self.assertEqual(
                result["retained_for_review"]["checkpoint"],
                str(final["parent-3101"].resolve()),
            )
            self.assertTrue(result["retained_for_review"]["starter_fallback"])

    def test_reference_focus_nomination_requires_both_targets_and_same_deal_results(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rates = {
                "parent-3101": (0.75, 0.58),
                "protected-3101-c500000": (0.75, 0.59),
                "focused-3101-c500000": (0.75, 0.62),
            }
            config, models = fixture(root, seeds=(3101,), rates=rates, version=4)

            def select():
                return select_candidates(summarize(root, config, models), config)

            self.assertEqual(select()["nominee"], "focused-3101-c500000")
            # Same heads-up win rate, different outcomes on the paired deals.
            path = root / "selection/focused-3101-c500000/native-headsup.json"
            report = json.loads(path.read_text())
            report["configurations"][0]["pair_scores"] = [0.5, 1.0]
            write_json(path, report, receipt=True)
            decision = select()
            self.assertEqual(decision["nominee"], "parent-3101")
            self.assertTrue(decision["nominee_is_starter_fallback"])
            self.assertFalse(
                decision["selected"]["3101"]["protected_gate"]["qualified"]
            )
            self.assertFalse(decision["selected"]["3101"]["starter_fallback"])

    def test_better_team_score_cannot_buy_a_missed_headsup_floor_or_reference_regression(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rates = {
                "parent-3101": (0.69, 0.58),
                "protected-3101-c500000": (0.69, 0.75),
                "focused-3101-c500000": (0.69, 0.80),
            }
            config, models = fixture(root, seeds=(3101,), rates=rates, version=4)
            outcome = select_candidates(summarize(root, config, models), config)
            self.assertTrue(outcome["nominee_is_starter_fallback"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rates = {
                "parent-3101": (0.72, 0.58),
                "protected-3101-c500000": (0.72, 0.61),
                "focused-3101-c500000": (0.72, 0.68),
            }
            config, models = fixture(root, seeds=(3101,), rates=rates, version=4)
            path = root / "selection/focused-3101-c500000/native-reference-human.json"
            report = json.loads(path.read_text())
            for conf in report["configurations"]:
                conf.update(win_rate=0.57, pair_scores=[0.57, 0.57])
            write_json(path, report, receipt=True)
            summary = summarize(root, config, models)
            self.assertTrue(
                summary["individuals"]["focused-3101-c500000"]["acceptable_strength"]
            )
            self.assertEqual(
                select_candidates(summary, config)["nominee"], "protected-3101-c500000"
            )

    def test_screened_starter_can_remain_nominee_without_becoming_a_trained_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, models = fixture(
                root,
                seeds=(3101,),
                version=3,
                rates={
                    "parent-3101": (0.72, 0.61),
                    "control-3101-c500000": (0.71, 0.605),
                    "targeted-3101-c500000": (0.70, 0.60),
                },
            )
            selection = select_candidates(summarize(root, config, models), config)
            self.assertTrue(selection["nominee_is_starter_fallback"])
            self.assertEqual(selection["nominee_final_name"], "parent-3101")
            self.assertEqual(
                selection["selected"]["3101"]["name"], "control-3101-c500000"
            )
            self.assertFalse(selection["selected"]["3101"]["starter_fallback"])

    def test_contextual_exploration_bundle_is_compared_to_matched_ordinary_continuation(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rates = {}
            for seed in (3101, 3102, 3103):
                rates[f"control-{seed}-c500000"] = (0.70, 0.58)
                rates[f"targeted-{seed}-c500000"] = (0.72, 0.62)
            config, models = fixture(root, rates=rates, version=3)
            result = summarize(root, config, models)
            self.assertEqual(result["protocol"], "rmappo-target-comparison-v3")
            contrast = result["groups"]["targeted-c500000"]["comparisons"][
                "continuation_control"
            ]["modes"]
            self.assertAlmostEqual(contrast["2"]["gain"], 0.02)
            self.assertAlmostEqual(contrast["4"]["gain"], 0.04)
            self.assertIn(
                "prior_nominee", result["groups"]["targeted-c500000"]["comparisons"]
            )
            self.assertEqual(
                result["individuals"]["prior-nominee"]["kind"], "prior_nominee"
            )
            self.assertTrue(
                select_candidates(result, config)["nominee"].startswith("targeted-")
            )

    def test_retraining_return_estimators_use_declared_profiles_and_paired_seed_contrasts(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rates = {}
            for seed in (3101, 3102, 3103):
                rates[f"gae99-{seed}-c500000"] = (0.71, 0.61)
                rates[f"mc-{seed}-c500000"] = (0.74, 0.64)
            config, models = fixture(root, rates=rates, version=2)
            result = summarize(root, config, models)
            self.assertIn(
                "fresh continuation seeds from one screened parent", result["scope"]
            )
            for mode in ("2", "4"):
                contrast = result["groups"]["mc-c500000"]["comparisons"]["gae_control"][
                    "modes"
                ][mode]
                self.assertAlmostEqual(contrast["gain"], 0.03)
            self.assertTrue(
                select_candidates(result, config)["nominee"].startswith("mc-")
            )
            with self.assertRaisesRegex(ValueError, "undeclared"):
                expected_reports(
                    root,
                    config,
                    {**models, "target-3101-c500000": models["mc-3101-c500000"]},
                )

    def test_actual_matched_blocks_and_equal_partner_macro_drive_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rates = {"reference": (0.4, 0.4)}
            for seed, parent in zip((3101, 3102, 3103), (0.45, 0.55, 0.65)):
                rates[f"parent-{seed}"] = (parent, parent)
                rates[f"original-{seed}-c500000"] = (parent + 0.02, parent + 0.02)
                rates[f"target-{seed}-c500000"] = (parent + 0.10, parent + 0.10)
            config, models = fixture(root, rates=rates)
            result = summarize(root, config, models)
            self.assertEqual(result["coverage"]["authenticated"], 140)
            self.assertFalse(result["approved"])
            group = result["groups"]["target-c500000"]
            for mode in ("2", "4"):
                paired = group["comparisons"]["matched_parent"]["modes"][mode]
                self.assertAlmostEqual(paired["gain"], 0.10)
                for bound in paired["gain_ci95"]:
                    self.assertAlmostEqual(bound, 0.10)
                control = group["comparisons"]["original_control"]["modes"][mode]
                self.assertAlmostEqual(control["gain"], 0.08)
            # One partner cannot become the overall 2v2 score.
            path = root / "selection/target-3101-c500000/native-reference-human.json"
            report = json.loads(path.read_text())
            for row in report["configurations"]:
                row.update(win_rate=0.95, pair_scores=[0.95, 0.95])
            write_json(path, report, receipt=True)
            result = summarize(root, config, models)
            self.assertAlmostEqual(
                result["individuals"]["target-3101-c500000"]["modes"]["4"]["win_rate"],
                0.65,
            )

    def test_incomplete_stale_extra_and_wrong_protocol_reports_cannot_be_compared(self):
        for failure in ("missing", "stale", "wrong_seed", "extra", "checkpoint"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config, models = fixture(root, seeds=(3101,))
                path = root / "selection/target-3101-c500000/native-headsup.json"
                if failure == "missing":
                    path.unlink()
                elif failure == "stale":
                    path.write_text(path.read_text() + " ")
                elif failure == "wrong_seed":
                    row = json.loads(path.read_text())
                    row["seed"] = config["final_seed"]
                    write_json(path, row, receipt=True)
                elif failure == "extra":
                    write_json(path.with_name("unplanned.json"), {})
                else:
                    models["target-3101-c500000"].write_bytes(b"replaced weights")
                self.assertFalse(coverage(root, config, models)["complete"])
                with self.assertRaisesRegex(ValueError, "incomplete authenticated"):
                    summarize(root, config, models)

    def test_both_targets_and_weaker_mode_beat_high_compensating_single_mode(self):
        cases = (((0.95, 0.59), (0.71, 0.61)), ((0.85, 0.62), (0.73, 0.64)))
        for original, target in cases:
            with self.subTest(original=original), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config, models = fixture(
                    root,
                    seeds=(3101,),
                    rates={
                        "original-3101-c500000": original,
                        "target-3101-c500000": target,
                    },
                )
                result = summarize(root, config, models)
                choice = select_candidates(result, config)
                self.assertEqual(choice["nominee"], "target-3101-c500000")
                self.assertFalse(choice["selected"]["3101"]["starter_fallback"])
                self.assertFalse(choice["approved"])

    def test_native_regression_guard_and_diagnostics_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, models = fixture(
                root,
                seeds=(3101,),
                rates={
                    "original-3101-c500000": (0.65, 0.60),
                    "target-3101-c500000": (0.75, 0.70),
                },
            )
            path = root / "selection/target-3101-c500000/parent-headsup.json"
            report = json.loads(path.read_text())
            for row in report["configurations"]:
                row.update(win_rate=0.1, pair_scores=[0.1, 0.1])
            write_json(path, report, receipt=True)
            result = summarize(root, config, models)
            row = result["individuals"]["target-3101-c500000"]
            self.assertTrue(row["native_guard_pass"])
            self.assertFalse(row["all_diagnostic_guard_pass"])
            self.assertEqual(
                len(row["comparisons"]["matched_parent"]["configuration_regressions"]),
                4,
            )
            self.assertEqual(
                select_candidates(result, config)["nominee"], "target-3101-c500000"
            )
            # A primary configuration failure is not hidden by strong mode means.
            path = root / "selection/target-3101-c500000/native-literal-v2.json"
            report = json.loads(path.read_text())
            report["configurations"][0].update(win_rate=0.1, pair_scores=[0.1, 0.1])
            write_json(path, report, receipt=True)
            result = summarize(root, config, models)
            self.assertFalse(
                result["individuals"]["target-3101-c500000"]["native_guard_pass"]
            )
            self.assertEqual(
                select_candidates(result, config)["nominee"], "original-3101-c500000"
            )

    def test_no_guarded_candidate_remains_explicitly_exploratory_not_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, models = fixture(
                root,
                seeds=(3101,),
                rates={
                    "parent-3101": (0.9, 0.9),
                    "original-3101-c500000": (0.65, 0.65),
                    "target-3101-c500000": (0.72, 0.61),
                },
            )
            choice = select_candidates(summarize(root, config, models), config)
            self.assertTrue(choice["selected"]["3101"]["exploratory"])
            self.assertEqual(choice["selected"]["3101"]["name"], "target-3101-c500000")
            self.assertFalse(choice["selected"]["3101"]["starter_fallback"])
            self.assertFalse(choice["approved"])

    def test_checkpoint_and_report_paths_and_target_semantics_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, models = fixture(root, seeds=(3101,))
            for phase in ("../outside", "/tmp", "."):
                with self.subTest(phase=phase), self.assertRaises(ValueError):
                    expected_reports(root, config, models, phase)
            with self.assertRaisesRegex(ValueError, "escapes root"):
                expected_reports(
                    root,
                    config,
                    {**models, "target-3101-c500000": root.parent / "elsewhere.pt"},
                )
            config["primary_specs"][0]["opponent"] = "parent"
            with self.assertRaisesRegex(ValueError, "specification changed"):
                expected_reports(root, config, models)


if __name__ == "__main__":
    unittest.main()
