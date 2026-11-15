import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from retruco_ml.rmappo_controls_compare import (
    ablation_comparison,
    compare,
    coverage,
    expected_reports,
)

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value, receipt=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    if receipt:
        path.with_suffix(".sha256").write_text(digest(path))


def report_for(spec, pairs, seed, checkpoint, inputs_sha, score, signals="normal"):
    return {
        "protocol": "improvement-evaluation-v1",
        "seed": seed,
        "pairs": pairs,
        "partner": spec["partner"],
        "opponent": spec["opponent"],
        "signals": signals,
        "card_policy": "policy",
        "checkpoint_sha256": digest(checkpoint),
        "inputs_sha256": inputs_sha,
        "behavior": {},
        "decision_contexts": {},
        "configurations": [
            {
                "players": spec["players"],
                "target": target,
                "flor": flor,
                "pairs": pairs,
                "matches": pairs * spec["players"],
                "win_rate": score,
                "pair_scores": [score] * pairs,
                "pair_net_points_per_hand": [0.0] * pairs,
            }
            for target in (12, 24)
            for flor in (False, True)
        ],
    }


def campaign(root, seeds=(2101, 2102), reference=0.45, parents=None, scores=None):
    original = json.loads((ROOT / "ml/experiments/rmappo-v1.json").read_text())
    config = {
        "cell": "lstm",
        "seeds": list(seeds),
        "choices": 50,
        "arms": {
            "A": {"balance": False, "betting": False},
            "B": {"balance": True, "betting": False},
            "C": {"balance": False, "betting": True},
            "D": {"balance": True, "betting": True},
        },
        "parent_sources": {},
        "primary_specs": original["primary_specs"],
        "diagnostic_specs": original["diagnostic_specs"],
        "primary_pairs": 2,
        "primary_seed": 142000,
        "diagnostic_pairs": 2,
        "diagnostic_seed": 143000,
        "ablation_pairs": 2,
        "ablation_seed": 144000,
    }
    (root / "inputs").mkdir()
    (root / "parents").mkdir()
    write_json(root / "inputs/manifest.json", {"fixture": "frozen inputs"})
    reference_path = root / "inputs/legacy-champion-lstm16.pt"
    torch.save({"legacy": True}, reference_path)
    models, model_scores = {"reference": reference_path}, {"reference": reference}
    for index, seed in enumerate(seeds):
        parent_path = root / "parents" / f"{seed}.pt"
        actor = {"weight": torch.tensor([float(index + 1)])}
        critic = {"weight": torch.tensor([float(index + 2)])}
        parent = {
            "kind": "rmappo-v1",
            "configuration": {"seed": 100 + index, "cell": "lstm"},
            "actor": actor,
            "critic": critic,
            "choices": 1000 + 100 * index,
        }
        torch.save(parent, parent_path)
        config["parent_sources"][str(seed)] = {
            "file": f"{seed}.pt",
            "sha256": digest(parent_path),
            "original_seed": 100 + index,
            "original_choices": parent["choices"],
        }
        models[f"parent-{seed}"] = parent_path
        model_scores[f"parent-{seed}"] = (
            parents[index] if parents is not None else 0.45 + 0.2 * index
        )
        for arm, flags in config["arms"].items():
            name = f"{arm}-{seed}"
            folder = root / "trials" / name
            folder.mkdir(parents=True)
            settings = {
                "arm": arm,
                "seed": seed,
                "cell": "lstm",
                "parent_sha256": digest(parent_path),
                "parent_choices": parent["choices"],
                "partner_balance": flags["balance"],
                "response_anchor": 0.1 if flags["betting"] else 0.0,
                "teacher_sha256": digest(reference_path),
                "inputs_sha256": digest(root / "inputs/manifest.json"),
            }
            write_json(folder / "configuration.json", settings)
            torch.save(
                {
                    "kind": "rmappo-v1",
                    "configuration": settings,
                    "actor": actor,
                    "critic": critic,
                    "actor_optimizer": {"state": {}},
                    "critic_optimizer": {"state": {}},
                    "choices": 0,
                    "updates": 0,
                },
                folder / "initial.pt",
            )
            candidate = {
                "kind": "rmappo-v1",
                "configuration": settings,
                "actor": {"weight": torch.tensor([float(index + 10)])},
                "critic": critic,
                "choices": 50,
                "updates": 1,
            }
            torch.save(candidate, folder / "candidate.pt")
            (folder / "candidate.sha256").write_text(digest(folder / "candidate.pt"))
            write_json(
                folder / "status.json",
                {
                    "status": "chunk-complete",
                    "update": 1,
                    "total_choices": 50,
                    "target_choices": 50,
                },
            )
            metrics = {
                "stage": "training",
                "update": 1,
                "total_choices": 50,
                "choices_this_update": 50,
                "elapsed_seconds": 2.0,
                "partner_balance": {
                    "applied": flags["balance"],
                    "actual_objective_masses": {},
                },
                "response_anchor": {"coefficient": settings["response_anchor"]},
            }
            (folder / "metrics.jsonl").write_text(json.dumps(metrics) + "\n")
            models[name] = folder / "candidate.pt"
            model_scores[name] = (
                scores[arm][index]
                if scores is not None
                else 0.4 + 0.2 * index + {"A": 0.0, "B": 0.1, "C": 0.15, "D": 0.3}[arm]
            )
    for name, checkpoint in models.items():
        for phase in ("primary", "diagnostic"):
            for spec in config[phase + "_specs"]:
                report = report_for(
                    spec,
                    config[phase + "_pairs"],
                    config[phase + "_seed"],
                    checkpoint,
                    digest(root / "inputs/manifest.json"),
                    model_scores[name],
                )
                write_json(
                    root / "evaluation" / name / f"{spec['id']}.json",
                    report,
                    receipt=True,
                )
    return config


class RMAPPOControlsCompareTests(unittest.TestCase):
    def test_matched_blocks_and_factorial_effects_use_the_actual_fixed_endpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = campaign(root)
            result = compare(root, config)
            self.assertEqual(result["coverage"]["authenticated"], 154)
            self.assertEqual(len(result["individuals"]), 8)
            self.assertEqual(result["exploratory_arm"], "D")
            self.assertEqual(result["retained_for_review"], "D-2102")
            self.assertFalse(result["approved"])
            self.assertFalse(result["variants"]["A"]["observed_guard_pass"])
            self.assertEqual(
                [block["parent_original_seed"] for block in result["blocks"]],
                [100, 101],
            )
            for seed in config["seeds"]:
                row = result["individuals"][f"D-{seed}"]
                self.assertAlmostEqual(
                    row["comparisons"]["matched_parent"]["modes"]["2"]["gain"], 0.25
                )
                self.assertAlmostEqual(
                    row["comparisons"]["control_A"]["modes"]["2"]["gain"], 0.30
                )
                self.assertEqual(row["training"]["additional_choices"], 50)
            self.assertAlmostEqual(result["variants"]["D"]["modes"]["2"]["gain"], 0.35)
            for mode in ("2", "4"):
                for effect, expected in {
                    "balance": 0.125,
                    "betting": 0.175,
                    "interaction": 0.05,
                    "D_minus_A": 0.30,
                }.items():
                    row = result["contrasts"][mode][effect]
                    self.assertAlmostEqual(row["estimate"], expected)
                    for bound in row["crossed_ci95"]:
                        self.assertAlmostEqual(bound, expected)

    def test_selection_prioritizes_both_targets_then_the_weaker_mode(self):
        cases = (
            {"B": (0.95, 0.59), "D": (0.71, 0.61)},
            {"B": (0.85, 0.62), "D": (0.73, 0.64)},
        )
        for mode_rates in cases:
            with (
                self.subTest(mode_rates=mode_rates),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                config = campaign(
                    root,
                    seeds=(2101,),
                    reference=0.4,
                    parents=[0.4],
                    scores={arm: [0.5] for arm in ("A", "B", "C", "D")},
                )
                for arm, rates in mode_rates.items():
                    for spec in config["primary_specs"]:
                        path = (
                            root / "evaluation" / f"{arm}-2101" / f"{spec['id']}.json"
                        )
                        report = json.loads(path.read_text())
                        score = rates[0 if spec["players"] == 2 else 1]
                        for row in report["configurations"]:
                            row.update(win_rate=score, pair_scores=[score, score])
                        write_json(path, report, receipt=True)
                result = compare(root, config)
                self.assertGreater(
                    result["variants"]["B"]["balanced_score"],
                    result["variants"]["D"]["balanced_score"],
                )
                self.assertEqual(result["exploratory_arm"], "D")
                self.assertEqual(result["exploratory_nominee"], "D-2101")
                self.assertEqual(result["retained_for_review"], "D-2101")

    def test_retention_requires_safety_against_both_parent_and_champion(self):
        for parent, champion in ((0.8, 0.4), (0.4, 0.8)):
            with (
                self.subTest(parent=parent, champion=champion),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                config = campaign(
                    root,
                    seeds=(2101,),
                    reference=champion,
                    parents=[parent],
                    scores={arm: [0.6] for arm in ("A", "B", "C", "D")},
                )
                result = compare(root, config)
                self.assertEqual(result["retained_for_review"], "reference")
                self.assertEqual(
                    result["retained_checkpoint"],
                    str(root / "inputs/legacy-champion-lstm16.pt"),
                )
                self.assertTrue(result["starter_fallback"])
                self.assertTrue(
                    all(
                        not row["observed_guard_pass"]
                        for row in result["individuals"].values()
                    )
                )

    def test_strong_mode_means_cannot_hide_partner_or_configuration_regressions(self):
        for failure in ("partner", "configuration"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config = campaign(
                    root,
                    seeds=(2101,),
                    reference=0.6,
                    parents=[0.6],
                    scores={arm: [0.8] for arm in ("A", "B", "C", "D")},
                )
                for arm in config["arms"]:
                    spec = (
                        "native-cautious-v3"
                        if failure == "partner"
                        else "native-literal-v2"
                    )
                    path = root / "evaluation" / f"{arm}-2101" / f"{spec}.json"
                    report = json.loads(path.read_text())
                    for index, row in enumerate(report["configurations"]):
                        score = 0.3 if failure == "partner" or index == 0 else 0.9
                        row.update(win_rate=score, pair_scores=[score, score])
                    write_json(path, report, receipt=True)
                result = compare(root, config)
                self.assertEqual(result["retained_for_review"], "reference")
                self.assertTrue(result["individuals"]["D-2101"]["acceptable_strength"])
                guard = result["individuals"]["D-2101"]["comparisons"]["matched_parent"]
                if failure == "configuration":
                    self.assertGreater(guard["strata"]["native-literal-v2"]["gain"], 0)
                    self.assertEqual(len(guard["configuration_regressions"]), 1)
                    self.assertEqual(
                        guard["configuration_regressions"][0]["target"], 12
                    )

    def test_actual_training_and_initial_parent_provenance_are_required(self):
        for failure, expected_error in (
            ("initial", "initial actor"),
            ("critic", "initial critic"),
            ("optimizer", "fresh empty Adam"),
            ("receipt", "endpoint receipt"),
            ("teacher", "endpoint initialization/arm/control"),
            ("flags", "actual balance application"),
            ("metrics", "reconstruct the endpoint budget"),
            ("status", "completed declared continuation budget"),
        ):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config = campaign(root, seeds=(2101,))
                folder = root / "trials/A-2101"
                if failure in ("initial", "critic", "optimizer"):
                    path = folder / "initial.pt"
                    saved = torch.load(path, weights_only=True)
                    if failure == "optimizer":
                        saved["actor_optimizer"]["state"] = {0: {"step": 1}}
                    else:
                        saved["critic" if failure == "critic" else "actor"][
                            "weight"
                        ] += 99
                    torch.save(saved, path)
                elif failure == "receipt":
                    (folder / "candidate.sha256").write_text("0" * 64)
                elif failure == "teacher":
                    path = folder / "candidate.pt"
                    saved = torch.load(path, weights_only=True)
                    saved["configuration"]["teacher_sha256"] = "0" * 64
                    torch.save(saved, path)
                    (folder / "candidate.sha256").write_text(digest(path))
                    # Authenticate the modified endpoint to reach the separate
                    # initialization provenance check, not the receipt guard.
                    for report_path in (root / "evaluation/A-2101").glob("*.json"):
                        report = json.loads(report_path.read_text())
                        report["checkpoint_sha256"] = digest(path)
                        write_json(report_path, report, receipt=True)
                elif failure == "status":
                    path = folder / "status.json"
                    row = json.loads(path.read_text())
                    row["status"] = "training"
                    write_json(path, row)
                else:
                    path = folder / "metrics.jsonl"
                    row = json.loads(path.read_text())
                    if failure == "flags":
                        row["partner_balance"]["applied"] = True
                    else:
                        row["choices_this_update"] = 49
                    path.write_text(json.dumps(row) + "\n")
                with self.assertRaisesRegex(ValueError, expected_error):
                    compare(root, config)

    def test_receipts_and_phase_specific_deal_provenance_are_not_inferred_from_paths(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = campaign(root, seeds=(2101,))
            path = root / "evaluation/D-2101/native-cautious-v3.json"
            report = json.loads(path.read_text())
            report["seed"] = config["primary_seed"]
            write_json(path, report, receipt=True)
            result = coverage(root, config)
            self.assertFalse(result["complete"])
            self.assertEqual(len(result["invalid"]), 1)
            self.assertIn("provenance", result["invalid"][0]["error"])
            report["seed"] = config["diagnostic_seed"]
            write_json(path, report, receipt=True)
            path.with_suffix(".sha256").unlink()
            self.assertIn("receipt", coverage(root, config)["invalid"][0]["error"])

    def test_ablation_inventory_and_losses_include_reference_matched_parents_and_selected_arm(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = campaign(root)
            jobs = expected_reports(root, config, "D")
            self.assertEqual(len(jobs), 174)
            ablations = [job for job in jobs if job["phase"] == "ablation"]
            self.assertEqual(len(ablations), 20)
            self.assertEqual(
                {job["model"] for job in ablations},
                {"reference", "parent-2101", "parent-2102", "D-2101", "D-2102"},
            )
            for job in ablations:
                loss = (
                    0.0
                    if job["signals"] == "normal"
                    else (0.2 if job["model"].startswith("D-") else 0.05)
                )
                report = report_for(
                    job["spec"],
                    job["pairs"],
                    job["seed"],
                    job["checkpoint"],
                    digest(root / "inputs/manifest.json"),
                    0.8 - loss,
                    job["signals"],
                )
                write_json(job["path"], report, receipt=True)
            result = ablation_comparison(root, config, "D")
            self.assertTrue(result["coverage"]["complete"])
            self.assertFalse(result["approved"])
            row = result["comparisons"]["no-send"]
            self.assertAlmostEqual(row["mean_normal_minus_removed"], 0.2)
            self.assertAlmostEqual(row["loss_change_vs_matched_parent"], 0.15)
            self.assertAlmostEqual(row["loss_change_vs_champion"], 0.15)


if __name__ == "__main__":
    unittest.main()
