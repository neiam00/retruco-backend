import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from retruco_ml.release_audit import Audit


class EvidenceReceiptTests(unittest.TestCase):
    def test_audit_rejects_another_campaign_or_changed_raw_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in ("reports", "receipts", "logs"):
                (root / name).mkdir()
            evidence = root / "evidence.json"
            evidence.write_text(json.dumps({"quantized": "reports/quantized.json"}))
            output = root / "reports/quantized.json"
            log = root / "logs/quantized.log"
            log.write_text("completed assessment\n")
            report = {
                "configurations": [],
                "evaluation_fingerprint": "a" * 64,
                "evaluation_source_commit": "b" * 40,
            }
            output.write_text(json.dumps(report))
            (root / "receipts/quantized.json").write_text(
                json.dumps(
                    {
                        "exit_code": 0,
                        "output_sha256": hashlib.sha256(
                            output.read_bytes()
                        ).hexdigest(),
                        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                    }
                )
            )
            audit = Audit(evidence)
            audit.evaluation_fingerprint = "a" * 64
            audit.evaluation_source_commit = "b" * 40
            audit.evaluation_base = root
            self.assertEqual(audit.read("quantized"), report)
            report["evaluation_fingerprint"] = "c" * 64
            output.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "different frozen"):
                audit.read("quantized")
            report["evaluation_fingerprint"] = "a" * 64
            report["configurations"] = [{"claimed_win_rate": 1}]
            output.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "raw output changed"):
                audit.read_report(output)


if __name__ == "__main__":
    unittest.main()
