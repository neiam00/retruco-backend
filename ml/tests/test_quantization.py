import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from retruco_ml.artifacts import read_quantized
from retruco_ml.export import serialize
from retruco_ml.model import DEPLOY_LAYERS, Policy, batch_distribution
from retruco_ml.quantization import policy_kl, quantization_aware
from retruco_ml.snapshot import verify_snapshot


class QuantizationTests(unittest.TestCase):
    def test_training_snapshot_rejects_changes_and_unrecorded_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "train.py"
            path.write_bytes(b"original source")
            metadata = {
                "files": {"train.py": hashlib.sha256(path.read_bytes()).hexdigest()},
                "torch": str(torch.__version__),
            }
            (root / "manifest.json").write_text(json.dumps(metadata))
            verify_snapshot(root)
            path.write_bytes(b"changed source")
            with self.assertRaisesRegex(ValueError, "source changed"):
                verify_snapshot(root)
            path.write_bytes(b"original source")
            (root / "extra.py").write_text("unrecorded source")
            with self.assertRaisesRegex(ValueError, "inventory"):
                verify_snapshot(root)
            (root / "extra.py").unlink()
            metadata["torch"] = "different version"
            (root / "manifest.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "environment"):
                verify_snapshot(root)

    def test_fake_quantized_actor_matches_the_actual_export_and_keeps_value_head_out(
        self,
    ):
        torch.manual_seed(41)
        torch.set_num_threads(2)
        reference = Policy()
        student = quantization_aware(reference)
        deployed = read_quantized(serialize(student)[0], reference)
        self.assertEqual(set(student.state_dict()), set(reference.state_dict()))
        self.assertFalse(student.value.weight.requires_grad)
        with torch.no_grad():
            for candidates in (1, 17, 1024):
                state = torch.randn(1, 768)
                actions = torch.randn(1, candidates, 48)
                actual = student(state, actions)
                expected = deployed(state, actions)
                for index in (0, 1):
                    torch.testing.assert_close(
                        actual[index], expected[index], atol=1e-6, rtol=1e-5
                    )
        for name in DEPLOY_LAYERS:
            self.assertTrue(student.get_submodule(name).weight.requires_grad)

    def test_ragged_distillation_has_finite_gradients_and_preserves_teacher(self):
        torch.manual_seed(42)
        teacher = Policy()
        student = quantization_aware(teacher)
        before = {k: v.clone() for k, v in teacher.state_dict().items()}
        observations = [
            {
                "features": torch.randn(768).tolist(),
                "actions": torch.randn(n, 48).tolist(),
                "kind": kind,
                "version": 2,
            }
            for n, kind in ((1, "game"), (11, "game"), (0, "signal"))
        ]
        with torch.no_grad():
            reference, _ = batch_distribution(teacher, observations)
        actual, _ = batch_distribution(student, observations)
        loss = policy_kl(reference, actual)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        gradients = [p.grad for p in student.parameters() if p.requires_grad]
        self.assertTrue(
            all(g is not None and torch.isfinite(g).all() for g in gradients)
        )
        self.assertTrue(any(torch.count_nonzero(g) > 0 for g in gradients))
        for key, value in teacher.state_dict().items():
            self.assertTrue(torch.equal(before[key], value))


if __name__ == "__main__":
    unittest.main()
