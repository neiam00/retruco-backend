"""Protocol and provenance behavior for staged native-target training."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from retruco_ml.recurrent import file_hash
from retruco_ml.target_campaign import (
    BudgetExhausted,
    check_deadline,
    coverage,
    freeze_json,
    validate_config,
    validate_endpoint,
)
from retruco_ml.target_compare import (
    assessment,
    final_comparison,
    select_checkpoint,
    verify_frozen_choices,
)
from retruco_ml.target_protocol import (
    confirmation_budgets,
    estimate_confirmation_round,
    selection_rank,
)

CONFIG = Path(__file__).resolve().parents[1] / "experiments/target-improvement-v1.json"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def protect(path):
    path.with_suffix(".sha256").write_text(file_hash(path) + "\n")


def evaluation(path, spec, value, pairs, seed, checkpoint, inputs_sha):
    report = {
        "protocol": "improvement-evaluation-v1",
        "seed": seed,
        "pairs": pairs,
        "partner": spec["partner"],
        "opponent": spec["opponent"],
        "signals": "normal",
        "card_policy": "policy",
        "checkpoint_sha256": file_hash(checkpoint),
        "inputs_sha256": inputs_sha,
        "behavior": {"truthful_claims": 4},
        "decision_contexts": {},
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
    write(path, report)
    protect(path)


class TargetCampaignTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text())

    def test_requires_factorial_recipes_fresh_training_seeds_and_disjoint_selection_final_deals(
        self,
    ):
        validate_config(self.config)
        changes = [
            lambda c: c["recipes"]["N100"].update(gae_lambda=0.99),
            lambda c: c.update(final_seed=c["selection_seed"]),
            lambda c: c.update(confirmation_seeds=c["discovery_seeds"]),
            lambda c: c["primary_specs"][0].update(opponent="parent"),
            lambda c: c.update(confirmation_checkpoints=[1000000, 500000]),
            lambda c: c.update(max_wall_hours=float("inf")),
        ]
        for change in changes:
            bad = copy.deepcopy(self.config)
            change(bad)
            with self.subTest(config=bad), self.assertRaises(ValueError):
                validate_config(bad)

    def test_deadline_carries_prior_runtime_and_rejects_exact_boundary(self):
        check_deadline(100.0, 30.0, 120.0, now=189.999)
        with self.assertRaises(BudgetExhausted):
            check_deadline(100.0, 30.0, 120.0, now=190.0)
        with self.assertRaises(BudgetExhausted):
            check_deadline(100.0, 121.0, 120.0, now=100.0)

    def test_v2_acceptance_and_selection_require_both_user_targets(self):
        config = json.loads(CONFIG.with_name("target-improvement-v2.json").read_text())
        validate_config(config)
        for mutate in (
            lambda c: c["trainer_controls"].update(entropy_coef=float("nan")),
            lambda c: c["recipes"]["E05_POOL"].update(unknown_knob=1),
            lambda c: c["trainer_controls"].update(gae_lambda=0.5),
            lambda c: c["trainer_controls"].update(partner_policy_version=3.0),
            lambda c: c.update(validation_reserve_hours=12),
            lambda c: c["mode_win_rate_targets"].update({"2v2": 0.65}),
        ):
            broken = copy.deepcopy(config)
            mutate(broken)
            with self.assertRaises(ValueError):
                validate_config(broken)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            starter = root / "starter.pt"
            starter.write_bytes(b"parent")
            candidate = root / "candidate.pt"
            candidate.write_bytes(b"trained")
            specs = config["primary_specs"]
            for spec in specs:
                evaluation(
                    root / "before" / (spec["id"] + ".json"),
                    spec,
                    0.7 if spec["players"] == 2 else 0.55,
                    8,
                    151000,
                    starter,
                    "inputs",
                )
                evaluation(
                    root / "after" / (spec["id"] + ".json"),
                    spec,
                    0.7 if spec["players"] == 2 else 0.60,
                    8,
                    151000,
                    candidate,
                    "inputs",
                )
            measured, _ = assessment(
                root / "after",
                root / "before",
                specs,
                8,
                151000,
                candidate,
                starter,
                "inputs",
                config,
            )
            self.assertTrue(measured["acceptable_strength"])
            self.assertEqual(measured["modes"]["4"]["target_win_rate"], 0.60)
            self.assertAlmostEqual(measured["balanced_score"], 1.0)
            high_headsup = copy.deepcopy(measured)
            high_headsup["modes"]["2"].update(win_rate=0.8)
            high_headsup["modes"]["4"].update(win_rate=0.55, target_met=False)
            high_headsup["balanced_score"] = 0.5 * (0.8 / 0.7 + 0.55 / 0.6)
            self.assertGreater(
                high_headsup["balanced_score"], measured["balanced_score"]
            )
            self.assertGreater(
                selection_rank(measured, config), selection_rank(high_headsup, config)
            )
            best = {"checkpoint": str(starter), "assessment": high_headsup}
            chosen = select_checkpoint(
                best, measured, candidate, 801, 500000, starter, config
            )
            self.assertTrue(chosen["trained"])
            better_team = copy.deepcopy(measured)
            better_team["modes"]["4"].update(win_rate=0.65)
            better_team["balanced_score"] = 0.5 * (1 + 0.65 / 0.6)
            tied_bottleneck = select_checkpoint(
                {"assessment": measured},
                better_team,
                candidate,
                801,
                1000000,
                starter,
                config,
            )
            self.assertTrue(tied_bottleneck["trained"])

    def test_confirmation_forecast_uses_persisted_work_even_after_cached_resume(self):
        config = json.loads(CONFIG.with_name("target-improvement-v2.json").read_text())
        recipe = "E05_POOL"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in config["discovery_seeds"]:
                p = root / "discovery" / f"{recipe}-seed-{seed}" / "metrics.jsonl"
                p.parent.mkdir(parents=True)
                p.write_text(
                    json.dumps({"phase": "warmup", "seconds": 30})
                    + "\n"
                    + json.dumps(
                        {
                            "phase": "training",
                            "choices": 250000,
                            "collect_seconds": 150,
                            "optimize_seconds": 100,
                        }
                    )
                    + "\n"
                )
            rows = [
                {
                    "command": [
                        "retruco_ml.improvement_evaluate",
                        "--output",
                        str(root / (spec["id"] + ".json")),
                        "--pairs",
                        "64",
                    ],
                    "seconds": 6.4,
                }
                for spec in config["primary_specs"] + config["diagnostic_specs"]
            ]
            (root / "work-completions.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n"
            )
            first = estimate_confirmation_round(root, config, recipe, 500000, 0)
            self.assertAlmostEqual(first, 2323.5)
            self.assertGreater(
                first, 1000
            )  # first round cannot fit 1,000s before cutoff
            for seed in config["confirmation_seeds"]:
                write(
                    root / "confirmation" / f"seed-{seed}" / "assessment-500000.json",
                    {"cached": True},
                )
            # Replaying those cached files in milliseconds cannot shrink the
            # independently measured forecast for the next real extension.
            resumed = estimate_confirmation_round(root, config, recipe, 1000000, 500000)
            self.assertAlmostEqual(resumed, 2211.0)

    def test_specialist_campaign_requires_team_only_training_and_identical_headsup(
        self,
    ):
        config = json.loads(CONFIG.with_name("team-specialist-v1.json").read_text())
        validate_config(config)
        mixed = copy.deepcopy(config)
        mixed["headsup_fraction"] = 0.4
        with self.assertRaises(ValueError):
            validate_config(mixed)
        bad = copy.deepcopy(config)
        bad["recipes"]["CONTEXTUAL"]["team_adapter"] = "unknown"
        with self.assertRaises(ValueError):
            validate_config(bad)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            starter = root / "starter.pt"
            starter.write_bytes(b"parent")
            candidate = root / "candidate.pt"
            candidate.write_bytes(b"team-only")
            specs = config["primary_specs"]
            for spec in specs:
                value = 0.70 if spec["players"] == 2 else 0.55
                evaluation(
                    root / "before" / (spec["id"] + ".json"),
                    spec,
                    value,
                    8,
                    177000,
                    starter,
                    "inputs",
                )
                evaluation(
                    root / "after" / (spec["id"] + ".json"),
                    spec,
                    value if spec["players"] == 2 else 0.61,
                    8,
                    177000,
                    candidate,
                    "inputs",
                )
            measured, _ = assessment(
                root / "after",
                root / "before",
                specs,
                8,
                177000,
                candidate,
                starter,
                "inputs",
                config,
            )
            self.assertTrue(measured["acceptable_strength"])
            # Even an apparent improvement must expose accidental changes to
            # the architecturally protected heads-up policy.
            evaluation(
                root / "after/native-headsup.json",
                specs[0],
                0.75,
                8,
                177000,
                candidate,
                "inputs",
            )
            with self.assertRaisesRegex(ValueError, "protected heads-up"):
                assessment(
                    root / "after",
                    root / "before",
                    specs,
                    8,
                    177000,
                    candidate,
                    starter,
                    "inputs",
                    config,
                )

    def test_v2_coverage_rejects_unexpected_paths_and_respects_frozen_equal_rounds(
        self,
    ):
        config = json.loads(CONFIG.with_name("target-improvement-v2.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "inputs/manifest.json", {"frozen": True})
            sha = file_hash(root / "inputs/manifest.json")
            starter = root / "inputs/legacy-champion-lstm16.pt"
            starter.write_bytes(b"parent")
            spec = config["primary_specs"][0]
            evaluation(
                root / "unrelated.json",
                spec,
                0.75,
                64,
                config["discovery_seed"],
                starter,
                sha,
            )
            result = coverage(root, config)
            self.assertEqual(result["expected_evaluation_reports"], 394)
            self.assertEqual(result["completed_evaluation_reports"], 0)
            evaluation(
                root / "discovery-baseline/primary/native-headsup.json",
                spec,
                0.75,
                64,
                config["discovery_seed"],
                starter,
                sha,
            )
            self.assertEqual(coverage(root, config)["completed_evaluation_reports"], 1)
            freeze_json(
                root / "confirmation-plan.json", {"completed_checkpoints": [500000]}
            )
            self.assertEqual(confirmation_budgets(root, config), [500000])
            self.assertEqual(coverage(root, config)["expected_evaluation_reports"], 310)
            write(root / "confirmation-plan.json", {"completed_checkpoints": [1000000]})
            protect(root / "confirmation-plan.json")
            with self.assertRaises(ValueError):
                confirmation_budgets(root, config)

    def test_endpoint_rejects_wrong_native_fraction_old_budget_or_source(self):
        config = self.config
        budget = 500000
        recipe = "N100"
        seed = 601
        settings = {
            "variant": recipe,
            "seed": seed,
            "choices": budget,
            "warm_start": "legacy-champion",
            "actor_memory": 64,
            "critic_hidden": [256, 128],
            "self_play_league": True,
            "league_interval": config["league_interval"],
            "heldout_parent": 103,
            "cooperation_practice": True,
            "headsup_fraction": 0.4,
            "native_fraction": 0.5,
            "gae_lambda": 1.0,
            **{
                key: config[key]
                for key in ("parent_anchor", "response_anchor", "signal_anchor")
            },
        }
        saved = {
            "training_kind": "improvement-mappo-v1",
            "configuration": settings,
            "choices": 500100,
            "contracts": {"starter_sha256": "parent", "inputs_sha256": "inputs"},
        }
        validate_endpoint(saved, config, recipe, seed, budget, "parent", "inputs")
        for change in [
            lambda x: x["configuration"].update(native_fraction=0.0),
            lambda x: x.update(choices=499999),
            lambda x: x["contracts"].update(starter_sha256="different"),
            lambda x: x["configuration"].update(choices=1000000),
        ]:
            bad = copy.deepcopy(saved)
            change(bad)
            with self.assertRaises(ValueError):
                validate_endpoint(bad, config, recipe, seed, budget, "parent", "inputs")

    def test_native_target_equal_partner_weights_and_authenticated_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            starter = root / "starter.pt"
            starter.write_bytes(b"parent")
            candidate = root / "candidate.pt"
            candidate.write_bytes(b"trained")
            specs = self.config["primary_specs"]
            values = {
                "literal-v2": 0.9,
                "trusting-v2": 0.7,
                "cautious-v2": 0.5,
                "reference-human": 0.5,
            }
            for spec in specs:
                evaluation(
                    root / "before" / (spec["id"] + ".json"),
                    spec,
                    0.7 if spec["players"] == 2 else 0.5,
                    4,
                    171000,
                    starter,
                    "inputs",
                )
                evaluation(
                    root / "after" / (spec["id"] + ".json"),
                    spec,
                    0.7 if spec["players"] == 2 else values[spec["partner"]],
                    4,
                    171000,
                    candidate,
                    "inputs",
                )
            result, modes = assessment(
                root / "after",
                root / "before",
                specs,
                4,
                171000,
                candidate,
                starter,
                "inputs",
            )
            self.assertAlmostEqual(result["modes"]["4"]["win_rate"], 0.65)
            self.assertTrue(result["acceptable_strength"])
            self.assertAlmostEqual(result["balanced_score"], 1.0)
            # A self-consistent rewritten receipt cannot authorize another model.
            path = root / "after/native-headsup.json"
            report = json.loads(path.read_text())
            report["checkpoint_sha256"] = "other"
            write(path, report)
            protect(path)
            with self.assertRaises(ValueError):
                assessment(
                    root / "after",
                    root / "before",
                    specs,
                    4,
                    171000,
                    candidate,
                    starter,
                    "inputs",
                )

    def test_checkpoint_selection_preserves_starter_when_guard_fails_and_freezes_choices(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            starter = root / "starter.pt"
            starter.write_bytes(b"parent")
            candidate = root / "trained.pt"
            candidate.write_bytes(b"candidate")
            best = {"checkpoint": str(starter), "assessment": {"balanced_score": 0.9}}
            bad = {"observed_guard_pass": False, "balanced_score": 1.1}
            self.assertIs(
                select_checkpoint(best, bad, candidate, 601, 500000, starter), best
            )
            good = {"observed_guard_pass": True, "balanced_score": 0.91}
            chosen = select_checkpoint(best, good, candidate, 601, 500000, starter)
            self.assertTrue(chosen["trained"])
            self.assertEqual(chosen["selection_budget"], 500000)
            path = root / "selection.json"
            freeze_json(path, chosen)
            freeze_json(path, chosen)
            changed = copy.deepcopy(chosen)
            changed["trained"] = False
            with self.assertRaises(ValueError):
                freeze_json(path, changed)

    def test_configuration_regression_cannot_hide_behind_improved_partner_average(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            starter = root / "starter.pt"
            starter.write_bytes(b"parent")
            candidate = root / "candidate.pt"
            candidate.write_bytes(b"trained")
            specs = self.config["primary_specs"]
            for spec in specs:
                evaluation(
                    root / "before" / (spec["id"] + ".json"),
                    spec,
                    0.5,
                    8,
                    171000,
                    starter,
                    "inputs",
                )
                evaluation(
                    root / "after" / (spec["id"] + ".json"),
                    spec,
                    0.8,
                    8,
                    171000,
                    candidate,
                    "inputs",
                )
            path = root / "after/native-literal-v2.json"
            report = json.loads(path.read_text())
            report["configurations"][0].update(pair_scores=[0.25] * 8, win_rate=0.25)
            write(path, report)
            protect(path)
            measured, _ = assessment(
                root / "after",
                root / "before",
                specs,
                8,
                171000,
                candidate,
                starter,
                "inputs",
            )
            self.assertGreater(measured["strata"]["native-literal-v2"]["gain"], 0)
            self.assertFalse(measured["observed_guard_pass"])
            self.assertEqual(
                measured["configuration_regressions"][0]["stratum"], "native-literal-v2"
            )
            best = {"checkpoint": str(starter), "assessment": {"balanced_score": 0.5}}
            self.assertIs(
                select_checkpoint(best, measured, candidate, 601, 500000, starter), best
            )

    def test_final_comparison_labels_starter_fallbacks_and_requires_frozen_nominee_first(
        self,
    ):
        config = copy.deepcopy(self.config)
        config.update(final_pairs=4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "inputs/manifest.json", {"immutable": True})
            inputs_sha = file_hash(root / "inputs/manifest.json")
            starter = root / "inputs/legacy-champion-lstm16.pt"
            starter.write_bytes(b"parent")
            trained = root / "confirmation/seed-601/candidate-budget-000500000.pt"
            trained.parent.mkdir(parents=True)
            trained.write_bytes(b"trained")
            protect(trained)
            selections = {}
            for seed in config["confirmation_seeds"]:
                checkpoint = trained if seed == 601 else starter
                selections[str(seed)] = {
                    "checkpoint": str(checkpoint),
                    "sha256": file_hash(checkpoint),
                    "origin": "budget-500000" if seed == 601 else "starter",
                    "trained": seed == 601,
                    "selection_budget": 500000 if seed == 601 else 0,
                    "seed": seed,
                }
                folder = root / "confirmation" / f"seed-{seed}"
                write(
                    folder / "status.json",
                    {
                        "status": "complete",
                        "choices": 2000100,
                        "update": 240,
                        "warmups": 2,
                    },
                )
                metric = {
                    "phase": "training",
                    "choices": 2000100,
                    "collect_seconds": 100,
                    "optimize_seconds": 100,
                    "diagnostics": {"replay_identity_max_logp_error": 1e-5},
                }
                (folder / "metrics.jsonl").write_text(json.dumps(metric) + "\n")
                for budget in config["confirmation_checkpoints"]:
                    write(folder / f"assessment-{budget}.json", {"endpoint": budget})
            nominee = {**selections["601"], "evaluation_label": "seed-601"}
            with self.assertRaises(ValueError):
                verify_frozen_choices(root, config, selections, nominee)
            for seed, row in selections.items():
                freeze_json(root / f"selection-{seed}.json", row)
            freeze_json(root / "nominee.json", nominee)
            verify_frozen_choices(root, config, selections, nominee)
            for label, checkpoint in [
                ("starter", starter),
                *[
                    (f"seed-{s}", Path(row["checkpoint"]))
                    for s, row in selections.items()
                ],
            ]:
                for spec in config["primary_specs"] + config["diagnostic_specs"]:
                    value = (
                        0.7
                        if spec["players"] == 2
                        else 0.66
                        if label == "seed-601"
                        else 0.55
                    )
                    evaluation(
                        root / "final" / label / (spec["id"] + ".json"),
                        spec,
                        value,
                        4,
                        174000,
                        checkpoint,
                        inputs_sha,
                    )
            result = final_comparison(
                root, config, {"selected_recipe": "N100"}, selections, nominee
            )
            self.assertEqual(result["trained_selected_count"], 1)
            self.assertEqual(result["fallback_count"], 2)
            self.assertFalse(result["individual"]["seed-602"]["trained"])
            self.assertEqual(
                result["individual"]["seed-602"]["checkpoint"], str(starter)
            )
            self.assertTrue(result["acceptable_strength"])
            self.assertFalse(result["approved"])
            wrong = copy.deepcopy(nominee)
            wrong["checkpoint"] = str(starter)
            with self.assertRaises(ValueError):
                final_comparison(root, config, {}, selections, wrong)

    def test_expected_coverage_counts_each_stage_and_requires_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                coverage(root, self.config)["expected_evaluation_reports"], 249
            )
            spec = self.config["primary_specs"][0]
            checkpoint = root / "actor.pt"
            checkpoint.write_bytes(b"model")
            path = root / "partial.json"
            evaluation(path, spec, 0.75, 4, 171000, checkpoint, "inputs")
            self.assertEqual(
                coverage(root, self.config)["completed_evaluation_reports"], 1
            )
            path.with_suffix(".sha256").unlink()
            self.assertEqual(
                coverage(root, self.config)["completed_evaluation_reports"], 0
            )


if __name__ == "__main__":
    unittest.main()
