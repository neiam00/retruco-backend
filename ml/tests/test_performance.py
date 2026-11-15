import copy
import unittest

from retruco_ml.release_audit import browser_performance


def fixture():
    profiles = []
    for rate in (1, 6, 10):
        profiles.append(
            {
                "rate": rate,
                "workerThrottleVerified": False,
                "workerCalibrationRatio": 1,
                "pageThrottleVerified": True,
                "pageCalibrationRatio": rate,
                "cases": [
                    {
                        "actions": 1024,
                        "history": {"signals": 20, "cantos": 64},
                        "estimatedComplete": {"samples": 200, "p95Ms": 15, "p99Ms": 19},
                    }
                ],
                "startup": {"worker": {"memoryBytes": 1769472}},
                "cold": {
                    "pageThrottleVerified": True,
                    "pageCalibrationRatio": rate,
                    **{
                        target: {"readyMs": 100, "totalMs": 200, "memoryBytes": 1769472}
                        for target in ("page", "worker")
                    },
                },
            }
        )
    return {
        "measurementVersion": 2,
        "decisionPath": "actual-worker-and-decision-session",
        "sourceSha256": {
            name: "a" * 64
            for name in ("model", "fallback", "decision", "load-policy", "worker")
        },
        "modelSha256": "model",
        "wasmSha256": "wasm",
        "profiles": profiles,
        "assets": {"totalGzipBytes": 95000},
        "productionGuard": {
            "productionEvaluationDisabled": True,
            "unapprovedBinaryDownloads": 0,
        },
    }


class PerformanceTests(unittest.TestCase):
    def test_recurrent_evidence_requires_full_replay_and_javascript_memory(self):
        valid = fixture()
        valid["recurrentScope"] = "authorized paginated replay"
        valid["sourceSha256"]["memory"] = "b" * 64
        for profile in valid["profiles"]:
            profile["cases"][0].update(
                events=2048, reconnect={"estimatedCompleteMs": 700, "roundTripMs": 100}
            )
            profile["memory"] = {
                "usedSize": 1500000,
                "backingStorageSize": 25200000,
                "embedderHeapUsedSize": 40000,
            }
        browser_performance(valid, "model", "wasm")
        invalid = []
        bad = copy.deepcopy(valid)
        del bad["sourceSha256"]["memory"]
        invalid.append(bad)
        bad = copy.deepcopy(valid)
        bad["profiles"][-1]["cases"][0]["events"] = 32
        invalid.append(bad)
        bad = copy.deepcopy(valid)
        bad["profiles"][-1]["cases"][0]["reconnect"]["estimatedCompleteMs"] = 1001
        invalid.append(bad)
        bad = copy.deepcopy(valid)
        bad["profiles"][-1]["memory"]["backingStorageSize"] = 65 * 1024 * 1024
        invalid.append(bad)
        for report in invalid:
            with self.subTest(report=report), self.assertRaises(ValueError):
                browser_performance(report, "model", "wasm")

    def test_complete_decisions_and_cold_calibration_are_required(self):
        valid = fixture()
        browser_performance(valid, "model", "wasm")
        invalid = []
        old = copy.deepcopy(valid)
        old["measurementVersion"] = 1
        invalid.append(old)
        old = copy.deepcopy(valid)
        del old["sourceSha256"]["fallback"]
        invalid.append(old)
        old = copy.deepcopy(valid)
        old["profiles"][-1]["cases"][0]["estimatedComplete"]["p95Ms"] = 21
        invalid.append(old)
        old = copy.deepcopy(valid)
        old["profiles"][-1]["cases"][0]["history"] = {}
        invalid.append(old)
        old = copy.deepcopy(valid)
        old["profiles"][-1]["cold"]["pageCalibrationRatio"] = 1
        invalid.append(old)
        old = copy.deepcopy(valid)
        old["profiles"][-1]["cold"]["page"]["totalMs"] = 501
        invalid.append(old)
        old = copy.deepcopy(valid)
        old["profiles"][1]["cases"][0]["estimatedComplete"]["samples"] = 20
        invalid.append(old)
        old = copy.deepcopy(valid)
        old["profiles"][-1]["workerThrottleVerified"] = True
        old["profiles"][-1]["workerCalibrationRatio"] = 10
        old["profiles"][-1]["cases"][0]["roundTrip"] = {
            "samples": 200,
            "p95Ms": 21,
            "p99Ms": 22,
        }
        invalid.append(old)
        for report in invalid:
            with self.subTest(report=report), self.assertRaises(ValueError):
                browser_performance(report, "model", "wasm")


if __name__ == "__main__":
    unittest.main()
