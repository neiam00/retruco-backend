import copy
import tempfile
import unittest
from pathlib import Path

from retruco_ml.artifacts import digest
from retruco_ml.release_audit import REQUIRED_INTEGRATION, integration_evidence


class IntegrationAuditTests(unittest.TestCase):
    def test_silent_auxiliary_steps_are_valid_but_test_output_is_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tests.log").write_text("tests passed\n")
            (root / "build.log").write_text("")

            def receipt(name, file):
                return {
                    "name": name,
                    "command": ["tool", name],
                    "exit_code": 0,
                    "evidence_path": file,
                    "evidence_sha256": digest(root / file),
                }

            report = {
                "complete": True,
                "smoke": False,
                "source_dirty": False,
                "model_sha256": "model",
                "backend_commit": "backend",
                "frontend_commit": "a" * 40,
                "checks": [
                    receipt(name, "tests.log") for name in sorted(REQUIRED_INTEGRATION)
                ]
                + [receipt("build-simulator", "build.log")],
            }
            self.assertIn(
                "build-simulator",
                integration_evidence(report, root, "model", "backend"),
            )
            invalid = []
            r = copy.deepcopy(report)
            r["complete"] = False
            invalid.append(r)
            r = copy.deepcopy(report)
            r["smoke"] = True
            invalid.append(r)
            r = copy.deepcopy(report)
            r["checks"][0] = receipt(r["checks"][0]["name"], "build.log")
            invalid.append(r)
            r = copy.deepcopy(report)
            r["checks"][-1]["exit_code"] = 1
            invalid.append(r)
            r = copy.deepcopy(report)
            r["checks"][-1]["evidence_sha256"] = "changed"
            invalid.append(r)
            r = copy.deepcopy(report)
            r["checks"].pop(0)
            invalid.append(r)
            for r in invalid:
                with self.subTest(report=r), self.assertRaises(ValueError):
                    integration_evidence(r, root, "model", "backend")
            (root / "build.log").unlink()
            with self.assertRaises(ValueError):
                integration_evidence(report, root, "model", "backend")


if __name__ == "__main__":
    unittest.main()
