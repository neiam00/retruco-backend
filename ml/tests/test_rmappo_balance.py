import math
import unittest
from types import SimpleNamespace

from retruco_ml.rmappo_balance import (
    BalanceConfig,
    Stratum,
    balance_choices,
    normalize_advantages,
    stratum_for_episode,
)

HEADSUP = Stratum("1v1", "none", "none")
HUMAN = Stratum("2v2", "human", "reference")


def choice(stratum, loss=0.0):
    return SimpleNamespace(
        observation={"kind": "game", "actions": [[1], [2]]},
        balance_stratum=stratum,
        loss=loss,
    )


class RMAPPOBalanceTests(unittest.TestCase):
    def test_objective_uses_actual_choice_counts_and_keeps_common_normalization(self):
        config = BalanceConfig.from_dict(
            {
                "targets": {"1v1/none/none": 0.4, "2v2/human/reference": 0.6},
                "min_choices_per_stratum": 1,
            }
        )
        records = [choice(HEADSUP, value) for value in (1, 3, 5, 7, 9, 11)]
        records += [choice(HUMAN, value) for value in (20, 40)]
        balanced = balance_choices(records, config)

        for weight in balanced.weights[:6]:
            self.assertAlmostEqual(weight, 8 / 15)
        for weight in balanced.weights[6:]:
            self.assertAlmostEqual(weight, 12 / 5)
        loss = sum(r.loss * w for r, w in zip(records, balanced.weights)) / 8
        self.assertAlmostEqual(
            loss, 20.4
        )  # .4 * mean(1..11, step 2) + .6 * mean(20,40)
        self.assertAlmostEqual(balanced.metrics["weight_sum"], 8)
        self.assertAlmostEqual(balanced.metrics["effective_sample_size"], 150 / 31)
        self.assertEqual(balanced.metrics["strata"][HUMAN.key]["sampled_choices"], 2)
        self.assertAlmostEqual(
            balanced.metrics["strata"][HUMAN.key]["effective_mass"], 0.6
        )
        self.assertTrue(balanced.metrics["target_met"])

        # Long human-partner games must not silently change the objective.
        replicated = records[:6] + records[6:] * 7
        repeat = balance_choices(replicated, config)
        repeat_loss = sum(r.loss * w for r, w in zip(replicated, repeat.weights)) / 20
        self.assertAlmostEqual(repeat_loss, 20.4)
        self.assertAlmostEqual(repeat.metrics["mean_loss_weight"], 1)
        self.assertAlmostEqual(
            repeat.metrics["strata"][HUMAN.key]["effective_mass"], 0.6
        )

    def test_weight_cap_reports_the_achievable_objective_without_shrinking_loss(self):
        config = BalanceConfig(
            {HEADSUP: 0.4, HUMAN: 0.6}, max_weight=2, min_choices_per_stratum=1
        )
        result = balance_choices([choice(HEADSUP)] * 9 + [choice(HUMAN)], config)
        for weight in result.weights[:9]:
            self.assertAlmostEqual(weight, 8 / 9)
        self.assertEqual(result.weights[-1], 2)
        self.assertAlmostEqual(sum(result.weights), 10)
        self.assertAlmostEqual(result.metrics["effective_sample_size"], 9)
        self.assertAlmostEqual(
            result.metrics["strata"][HUMAN.key]["effective_mass"], 0.2
        )
        self.assertEqual(result.metrics["capped_strata"], [HUMAN.key])
        self.assertTrue(result.metrics["coverage_ok"])
        self.assertFalse(result.metrics["target_met"])

    def test_missing_or_sparse_target_strata_fail_unless_explicitly_warned(self):
        config = BalanceConfig({HEADSUP: 0.4, HUMAN: 0.6}, min_choices_per_stratum=2)
        for human_choices in (0, 1):
            with self.subTest(human_choices=human_choices):
                records = [choice(HEADSUP)] * 4 + [choice(HUMAN)] * human_choices
                with self.assertRaisesRegex(
                    ValueError, "insufficient genuine-choice coverage"
                ):
                    balance_choices(records, config)

        warned = BalanceConfig(
            {HEADSUP: 0.4, HUMAN: 0.6}, coverage="warn", min_choices_per_stratum=2
        )
        with self.assertWarnsRegex(RuntimeWarning, "2v2/human/reference=0<2"):
            result = balance_choices([choice(HEADSUP)] * 4, warned)
        self.assertEqual(result.weights, (1, 1, 1, 1))
        self.assertFalse(result.metrics["coverage_ok"])
        self.assertFalse(result.metrics["target_met"])
        self.assertEqual(result.metrics["missing_strata"], [HUMAN.key])
        self.assertEqual(result.metrics["strata"][HUMAN.key]["sampled_choices"], 0)
        self.assertEqual(
            result.metrics["strata"][HUMAN.key]["effective_sample_size"], 0
        )
        self.assertEqual(result.metrics["strata"][HUMAN.key]["effective_mass"], 0)

    def test_forced_choices_missing_labels_and_unconfigured_families_are_rejected(self):
        config = BalanceConfig({HEADSUP: 1.0}, min_choices_per_stratum=1)
        forced_game = choice(HEADSUP)
        forced_game.observation["actions"] = [[1]]
        forced_signal = choice(HEADSUP)
        forced_signal.observation = {
            "kind": "signal",
            "signalMask": [True, False, False],
        }
        for record in (forced_game, forced_signal):
            with self.subTest(observation=record.observation):
                with self.assertRaisesRegex(ValueError, "genuine choices only"):
                    balance_choices([record], config)
        with self.assertRaisesRegex(ValueError, "collector-assigned"):
            balance_choices([choice(None)], config)
        with self.assertRaisesRegex(ValueError, "unconfigured balance strata"):
            balance_choices([choice(HUMAN)], config)
        with self.assertRaisesRegex(ValueError, "empty"):
            balance_choices([], config)

    def test_zero_target_support_cannot_defeat_the_weight_cap(self):
        config = BalanceConfig(
            {HEADSUP: 1, HUMAN: 0}, max_weight=2, min_choices_per_stratum=1
        )
        with self.assertRaisesRegex(ValueError, "common loss normalization"):
            balance_choices([choice(HEADSUP)] + [choice(HUMAN)] * 9, config)

    def test_labels_follow_partner_role_and_seat_and_exclude_evaluation_checkpoint(
        self,
    ):
        controllers = ["learner", "frozen:101", "reference", "frozen:101"]
        human_roles = [True, True, False, True]
        self.assertEqual(stratum_for_episode(4, controllers, human_roles, 0), HUMAN)
        self.assertEqual(
            stratum_for_episode(4, controllers, [True] * 4, 0),
            Stratum("2v2", "cpu", "reference"),
        )
        # A learner on seat 2 has seat 0 as its partner, not the episode anchor.
        controllers = ["frozen:102", "frozen:101", "learner", "frozen:101"]
        self.assertEqual(
            stratum_for_episode(4, controllers, [True] * 4, 2),
            Stratum("2v2", "cpu", "frozen"),
        )
        self.assertEqual(
            stratum_for_episode(2, ["learner", "frozen:102"], [True] * 4, 0), HEADSUP
        )
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            stratum_for_episode(2, ["learner", "frozen:103"], [True] * 4, 0)

    def test_weighting_reads_only_scheduling_labels_and_legal_choice_cardinality(self):
        class RestrictedRecord:
            def __init__(self, label):
                self.balance_stratum = label
                self.observation = {"kind": "game", "actions": [[1], [2]]}

            def __getattribute__(self, name):
                if name not in ("balance_stratum", "observation"):
                    raise AssertionError(f"weighting read non-scheduling data: {name}")
                return object.__getattribute__(self, name)

        records = [RestrictedRecord(HEADSUP), RestrictedRecord(HUMAN)]
        before = [dict(record.observation) for record in records]
        result = balance_choices(
            records,
            BalanceConfig({HEADSUP: 0.4, HUMAN: 0.6}, min_choices_per_stratum=1),
        )
        self.assertEqual(result.weights, (0.8, 1.2))
        self.assertEqual([record.observation for record in records], before)

    def test_advantages_are_centered_in_the_same_loss_measure(self):
        result = normalize_advantages([0] * 6 + [10] * 2, [8 / 15] * 6 + [12 / 5] * 2)
        for value in result[:6]:
            self.assertAlmostEqual(value, -math.sqrt(1.5))
        for value in result[6:]:
            self.assertAlmostEqual(value, math.sqrt(2 / 3))
        self.assertEqual(normalize_advantages([4, 4], [1, 1]), (0, 0))
        with self.assertRaisesRegex(ValueError, "positive total mass"):
            normalize_advantages([1, 2], [0, 0])


if __name__ == "__main__":
    unittest.main()
