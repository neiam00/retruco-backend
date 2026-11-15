import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from retruco_ml.improvement_evaluate import evaluate, highest_compatible_play
from retruco_ml.model import Policy
from retruco_ml.recurrent import Actor, Critic
from retruco_ml.rollout import Simulator


class ImprovementIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.simulator = Path(__file__).resolve().parents[2] / "bin/selfplay"
        self.assertTrue(self.simulator.exists())

    def test_highest_card_ablation_preserves_bids_and_play_declarations(self):
        features = [0.0] * 768
        features[24], features[29], features[34] = 0.9, 0.2, 0.7
        observation = {
            "features": features,
            "view": {"hand": [{"index": i, "used": False} for i in range(3)]},
            "legal": [
                {"type": "play", "mode": "ordinary"},
                {"type": "play", "mode": "ordinary", "card": 1},
                {"type": "play", "mode": "ordinary", "card": 2},
                {"type": "play", "mode": "declare", "declaration": 27, "card": 1},
                {"type": "play", "mode": "declare", "declaration": 27, "card": 2},
                {"type": "truco", "amount": 3},
            ],
        }
        self.assertEqual(highest_compatible_play(observation, 1), 0)
        self.assertEqual(highest_compatible_play(observation, 3), 4)
        self.assertEqual(highest_compatible_play(observation, 5), 5)

    def test_paired_learned_opposition_is_deterministic_and_never_reads_privileged_channel(
        self,
    ):
        actor = Actor(True)
        models = {
            "reference": Policy(),
            "frozen:101": Actor(True),
            "frozen:102": Actor(True),
            "frozen:103": Actor(True),
        }
        with Simulator(self.simulator, "binary") as simulator:
            ordinary = evaluate(
                actor,
                models,
                simulator,
                pairs=2,
                seed=193100,
                opponent="parent",
                players=0,
            )

            class PoisonCriticOnlyChannel:
                def call(self, requests):
                    observations = simulator.call(requests)
                    for observation in observations:
                        if observation.get("training"):
                            observation["training"]["privileged"] = [float("nan")] * 128
                    return observations

            poisoned = evaluate(
                actor,
                models,
                PoisonCriticOnlyChannel(),
                pairs=2,
                seed=193100,
                opponent="parent",
                players=0,
            )
            reactive = evaluate(
                actor,
                models,
                simulator,
                pairs=2,
                seed=193101,
                opponent="reactive",
                players=4,
            )
        self.assertEqual(ordinary["configurations"], poisoned["configurations"])
        self.assertEqual(ordinary["behavior"], poisoned["behavior"])
        self.assertEqual(len(ordinary["configurations"]), 8)
        self.assertEqual(len(reactive["configurations"]), 4)
        self.assertGreater(ordinary["behavior"]["hands_completed"], 0)
        self.assertEqual(
            ordinary["behavior"]["net_points"],
            ordinary["behavior"]["points_for"] - ordinary["behavior"]["points_against"],
        )
        self.assertFalse(
            any(
                parameter.grad is not None
                for model in [actor, *models.values()]
                for parameter in model.parameters()
            )
        )

    def test_extended_budget_resume_keeps_models_optimizers_and_random_streams_exact(
        self,
    ):
        for memory_size, critic_hidden in ((16, [96, 64]), (64, [256, 128])):
            with (
                self.subTest(memory=memory_size),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                inputs = root / "inputs"
                inputs.mkdir()
                actor, critic = Actor(True), Critic(True)
                manifest = {
                    "kind": "improvement-inputs-v1",
                    "parents": {},
                    "critics": {},
                }

                def save(name, payload):
                    path = inputs / name
                    torch.save(payload, path)
                    return {
                        "file": name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }

                for seed in (101, 102, 103):
                    manifest["parents"][str(seed)] = save(
                        f"actor-{seed}.pt", {"model": actor.state_dict()}
                    )
                    manifest["critics"][str(seed)] = save(
                        f"critic-{seed}.pt", {"critic": critic.state_dict()}
                    )
                manifest["reference"] = save(
                    "reference.pt", {"model": Policy().state_dict()}
                )
                manifest["candidates"] = {"d-parent": manifest["parents"]["101"]}
                path = inputs / "manifest.json"
                path.write_text(json.dumps(manifest))

                def run(folder, updates, budget, resume=False):
                    command = [
                        sys.executable,
                        "-m",
                        "retruco_ml.improvement_train",
                        "--output",
                        str(folder),
                        "--inputs",
                        str(path),
                        "--simulator",
                        str(self.simulator),
                        "--variant",
                        "R99",
                        "--gae-lambda",
                        ".99",
                        "--reactive",
                        "--seed",
                        "201",
                        "--choices",
                        str(budget),
                        "--steps",
                        "32",
                        "--workers",
                        "1",
                        "--environments",
                        "1",
                        "--batch-size",
                        "8",
                        "--calibration-choices",
                        "16",
                        "--warmup-max",
                        "2",
                        "--max-updates",
                        str(updates),
                    ]
                    command += [
                        "--actor-memory",
                        str(memory_size),
                        "--critic-hidden",
                        *map(str, critic_hidden),
                    ]
                    if memory_size == 64:
                        command += ["--self-play-league", "--league-interval", "1"]
                    if resume:
                        command.append("--resume")
                    result = subprocess.run(command, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr[-2400:])
                    return torch.load(folder / "checkpoint.pt", weights_only=True)

                whole = run(root / "whole", 2, 100000)
                run(root / "split", 1, 100000)
                continued = run(root / "split", 1, 200000, True)
                for name in ("model", "critic"):
                    for key, value in whole[name].items():
                        torch.testing.assert_close(
                            value, continued[name][key], atol=0, rtol=0
                        )
                for name in ("actor_optimizer", "critic_optimizer"):
                    for key, values in whole[name]["state"].items():
                        for parameter, value in values.items():
                            torch.testing.assert_close(
                                value,
                                continued[name]["state"][key][parameter],
                                atol=0,
                                rtol=0,
                            )
                self.assertEqual(whole["choices"], continued["choices"])
                self.assertEqual(whole["random"], continued["random"])
                if memory_size == 64:
                    self.assertGreaterEqual(len(continued["league"]["actors"]), 2)
                    self.assertEqual(
                        [x["fingerprint"] for x in whole["league"]["actors"]],
                        [x["fingerprint"] for x in continued["league"]["actors"]],
                    )
                    for left, right in zip(
                        whole["league"]["actors"],
                        continued["league"]["actors"],
                        strict=True,
                    ):
                        for key, value in left["model"].items():
                            torch.testing.assert_close(
                                value, right["model"][key], atol=0, rtol=0
                            )
                torch.testing.assert_close(
                    whole["torch_random"], continued["torch_random"], atol=0, rtol=0
                )
                overshot = run(root / "overshot", 1, 1)
                self.assertGreater(overshot["choices"], 2)
                relabeled = run(root / "overshot", 1, 2, True)
                self.assertEqual(relabeled["configuration"]["choices"], 2)
                self.assertEqual(overshot["choices"], relabeled["choices"])
                self.assertEqual(overshot["update"], relabeled["update"])
                for key, value in overshot["model"].items():
                    torch.testing.assert_close(
                        value, relabeled["model"][key], atol=0, rtol=0
                    )
                probe = json.loads((root / "split/critic-probe.json").read_text())
                self.assertIn("brier", probe["overall"])
                self.assertIn("checkpoint_sha256", probe)
                for entry in [
                    *manifest["parents"].values(),
                    *manifest["critics"].values(),
                    manifest["reference"],
                ]:
                    self.assertEqual(
                        hashlib.sha256(
                            (inputs / entry["file"]).read_bytes()
                        ).hexdigest(),
                        entry["sha256"],
                    )

    def test_native_opponent_signal_delivery_is_a_negative_control(self):
        actor = Actor(True)
        models = {
            "reference": Policy(),
            "frozen:101": Actor(True),
            "frozen:102": Actor(True),
            "frozen:103": Actor(True),
        }
        with Simulator(self.simulator, "binary") as simulator:
            normal = evaluate(
                actor,
                models,
                simulator,
                pairs=2,
                seed=193110,
                partner="literal-v2",
                players=4,
            )
            blocked = evaluate(
                actor,
                models,
                simulator,
                pairs=2,
                seed=193110,
                partner="literal-v2",
                players=4,
                signals="no-opponent-delivery",
            )
            # Native cpu.Decide reads rules.View only, never delivered signals.
            # Filtering opponents must not change sender/partner state or RNG.
            self.assertEqual(normal["configurations"], blocked["configurations"])
            self.assertEqual(normal["behavior"], blocked["behavior"])
            self.assertGreater(normal["behavior"]["truthful_claims"], 0)
            self.assertEqual(
                normal["response_families"],
                {},
                "human-role teammate owns pending canto responses",
            )


if __name__ == "__main__":
    unittest.main()
