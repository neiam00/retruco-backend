"""Exercise heads-up preservation with native observations and untrained stress weights."""

import copy
import unittest
from pathlib import Path

import numpy as np
import torch

from retruco_ml.mappo_export import quantized
from retruco_ml.rmappo_browser_parity import collect_cases, torch_results
from retruco_ml.rmappo_headsup_protection import (
    HeadsUpProtection,
    verify_protected_weights,
)
from retruco_ml.rmappo_model import RecurrentActor, actor_from_checkpoint

ROOT = Path(__file__).resolve().parents[2]


class HeadsUpProtectionTests(unittest.TestCase):
    def test_team_weight_changes_preserve_native_headsup_decisions_memory_and_quantization(
        self,
    ):
        torch.set_num_threads(1)
        torch.manual_seed(61)
        starter = RecurrentActor("lstm")
        cases = [
            case
            for case in collect_cases(starter, ROOT / "bin/selfplay")
            if not case["synthetic"]
        ]
        trial = copy.deepcopy(starter)
        constraint = HeadsUpProtection(trial, starter.state_dict())
        optimizer = constraint.optimizer()
        self.assertEqual(
            [name for name, p in trial.named_parameters() if p.requires_grad],
            ["state.0.weight"],
        )
        # Inject an invalid change with gradients absent. Adam makes no update;
        # its actual registered post-step hook must restore the protected state.
        with torch.no_grad():
            trial.state[0].weight.fill_(1000.0)
        with self.assertRaises(ValueError):
            verify_protected_weights(starter.state_dict(), trial.state_dict())
        optimizer.step()
        self.assertFalse(optimizer.state)
        verify_protected_weights(starter.state_dict(), trial.state_dict())
        loaded = actor_from_checkpoint(
            {
                "kind": "rmappo-v1",
                "configuration": {"cell": "lstm"},
                "actor": trial.state_dict(),
            }
        )
        for first, second in (
            (starter, loaded),
            (quantized(starter), quantized(loaded)),
        ):
            original, old_memory = torch_results(first, cases)
            changed, new_memory = torch_results(second, cases)
            changed_team = False
            for i, case in enumerate(cases):
                if case["view"]["config"]["players"] == 2:
                    np.testing.assert_array_equal(original[i], changed[i])
                    np.testing.assert_array_equal(old_memory[i], new_memory[i])
                elif not np.array_equal(original[i], changed[i]):
                    changed_team = True
            self.assertTrue(changed_team)

    def test_loss_gradient_and_checkpoint_verifier_enforce_the_same_contract(self):
        torch.manual_seed(62)
        starter = RecurrentActor("lstm")
        trial = copy.deepcopy(starter)
        constraint = HeadsUpProtection(trial, starter.state_dict())
        local = torch.randn(768)
        with torch.enable_grad():
            context = trial.context(local, trial.initial())
            (grad,) = torch.autograd.grad(
                trial.signal_logits(context)[0], constraint.weight
            )
        self.assertTrue(torch.any(grad[constraint.allowed] != 0))
        self.assertFalse(torch.any(grad[~constraint.allowed] != 0))
        with torch.no_grad():
            trial.state[0].weight[constraint.allowed] = 1000.0
        with self.assertRaisesRegex(ValueError, "row scale"):
            verify_protected_weights(starter.state_dict(), trial.state_dict())
        constraint.enforce_bounds()
        with torch.no_grad():
            trial.memory.core.bias_hh_l0[0] += 1
        with self.assertRaisesRegex(ValueError, "memory"):
            verify_protected_weights(starter.state_dict(), trial.state_dict())
