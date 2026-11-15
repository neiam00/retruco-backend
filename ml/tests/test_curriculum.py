import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from retruco_ml import curriculum


class CurriculumTests(unittest.TestCase):
    def test_training_cannot_discard_a_stronger_phase_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilots = root / "pilots"
            output = root / "output"
            (pilots / "source/ml").mkdir(parents=True)
            (pilots / "source/manifest.json").write_text(json.dumps({"files": {}}))
            initial = pilots / "initial.pt"
            torch.save({"generation": 0, "total_seconds": 0.0}, initial)
            selected = {
                "checkpoint": str(initial),
                "checkpoint_sha256": hashlib.sha256(initial.read_bytes()).hexdigest(),
                "seed": 43,
            }

            class SimulationResult:
                """External trainer/evaluator contract; selection stays real."""

                pid = 98765

                def __init__(self, command, **_):
                    self.command = command

                def wait(self, timeout=None):
                    command = self.command

                    def argument(flag):
                        return command[command.index(flag) + 1]

                    destination = Path(argument("--output"))
                    if command[3] == "retruco_ml.train":
                        path = destination / "checkpoint.pt"
                        state = torch.load(path, weights_only=True)
                        state["generation"] += 1
                        state["total_seconds"] += 1
                        torch.save(state, path)
                    else:
                        candidate = Path(command[4])
                        state = torch.load(candidate, weights_only=True)
                        values = [1, 0.5] if state["generation"] == 0 else [0.5, 0]
                        players = int(argument("--players"))
                        report = {
                            "checkpoint_sha256": hashlib.sha256(
                                candidate.read_bytes()
                            ).hexdigest(),
                            "seed": 100000,
                            "configurations": [
                                {
                                    "players": p,
                                    "target": target,
                                    "flor": flor,
                                    "pairs": 2,
                                    "pair_scores": values,
                                    "win_rate": sum(values) / 2,
                                }
                                for p in ((players,) if players else (2, 4))
                                for target in (12, 24)
                                for flor in (False, True)
                            ],
                        }
                        destination.write_text(json.dumps(report))
                    return 0

            arguments = [
                "curriculum",
                "--pilots",
                str(pilots),
                "--output",
                str(output),
                "--smoke",
                "--heads-up-hours",
                ".00001",
                "--mixed-hours",
                ".00001",
                "--chunk-hours",
                ".00001",
                "--validation-pairs",
                "2",
            ]
            with (
                patch("sys.argv", arguments),
                patch.object(curriculum, "selection_if_ready", return_value=selected),
                patch.object(curriculum.subprocess, "Popen", SimulationResult),
                patch.object(curriculum.signal, "signal"),
            ):
                curriculum.main()
            candidate = json.loads((output / "candidate.json").read_text())
            self.assertEqual(
                torch.load(candidate["checkpoint"], weights_only=True)["generation"], 0
            )
            for phase in ("heads-up", "mixed"):
                self.assertEqual(
                    json.loads((output / phase / "phase.json").read_text())[
                        "best_score"
                    ],
                    0.75,
                )


if __name__ == "__main__":
    unittest.main()
