"""Integration test: exporter → Node WASM → quantized PyTorch reference.

Build the runtime first. This exercises real byte layout and inference, not mocks.
"""

import ctypes
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from retruco_ml.export import export
from retruco_ml.model import DEPLOY_LAYERS, Policy
from retruco_ml.parity import differences

ROOT = Path(__file__).resolve().parents[1]
WASM = ROOT / "runtime/target/wasm32v1-none/release/retruco_policy.wasm"
NATIVE = ROOT / (
    "runtime/target/release/libretruco_policy.dylib"
    if sys.platform == "darwin"
    else "runtime/target/release/libretruco_policy.so"
)


class RuntimeTests(unittest.TestCase):
    def test_parity_rejects_nonfinite_and_misaligned_evidence(self):
        for actual in ([[float("nan"), 0]], [[0]], []):
            with self.assertRaises(ValueError):
                differences([[0, 1]], actual)
        # Logits may differ by a common offset while defining the same policy.
        result = differences([[0, 1]], [[100, 101]])
        self.assertEqual(result["max_probability_error"], 0)
        self.assertEqual(result["max_logit_error"], 100)

    def test_real_wasm_matches_exported_network(self):
        self.assertTrue(WASM.exists(), "build the Rust WASM runtime before testing")
        torch.manual_seed(18)
        torch.set_num_threads(2)
        model = Policy()
        with tempfile.TemporaryDirectory() as temporary:
            manifest = export(model, temporary)
            self.assertLessEqual(manifest["bytes"], 128 * 1024)
            self.assertFalse(manifest["approved"])
            # Match only storage quantization; Torch independently evaluates layers.
            with torch.no_grad():
                for name in DEPLOY_LAYERS:
                    weight = model.get_submodule(name).weight
                    scale = weight.abs().amax(dim=1).clamp(min=1.27e-10) / 127
                    weight.copy_(
                        (weight / scale[:, None]).round().clamp(-127, 127)
                        * scale[:, None]
                    )
            cases = []
            expected = []
            for count, signal in [(1, False), (11, False), (1024, False), (0, True)]:
                state = torch.randn(1, 768)
                actions = torch.randn(1, max(count, 1), 48)
                with torch.no_grad():
                    game, signals, _ = model(state, actions)
                expected.append((signals if signal else game)[0].tolist())
                cases.append(
                    {
                        "features": state[0].tolist(),
                        "actions": actions[0][:count].flatten().tolist(),
                        "count": count,
                        "signal": int(signal),
                    }
                )
            script = r"""
const fs = require('fs');
(async () => {
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const module = await WebAssembly.compile(fs.readFileSync(input.wasm));
if (WebAssembly.Module.imports(module).length) throw Error('unexpected host imports');
const instance = await WebAssembly.instantiate(module, {});
const x = instance.exports;
const bytes = fs.readFileSync(input.model);
new Uint8Array(x.memory.buffer, x.model_ptr(), bytes.length).set(bytes);
if (x.load_model(bytes.length) !== 0) throw Error('load failed');
const results = input.cases.map(c => {
new Float32Array(x.memory.buffer, x.state_ptr(), 768).set(c.features);
new Float32Array(x.memory.buffer, x.actions_ptr(), c.actions.length).set(c.actions);
const count = x.infer(c.count, c.signal);
if (count < 0) throw Error('inference failed');
return Array.from(new Float32Array(x.memory.buffer, x.output_ptr(), count));
});
if (x.load_model(bytes.length-1) >= 0 || x.infer(1,0) >= 0) throw Error('invalid model remained usable');
let capped = false;
try { x.memory.grow(65); } catch (error) { capped = error instanceof RangeError; }
if (!capped) throw Error('WASM memory maximum not enforced');
process.stdout.write(JSON.stringify({results,memory:x.memory.buffer.byteLength}));
})().catch(e => {console.error(e);process.exit(1)});
"""
            response = subprocess.run(
                ["node", "-e", script],
                input=json.dumps(
                    {
                        "wasm": str(WASM),
                        "model": str(Path(temporary) / "policy.bin"),
                        "cases": cases,
                    }
                ),
                text=True,
                capture_output=True,
                check=True,
            )
            actual = json.loads(response.stdout)
            self.assertLessEqual(actual["memory"], 4 * 1024 * 1024)
            for want, got in zip(expected, actual["results"], strict=True):
                np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-4)
            self.assertTrue(
                NATIVE.exists(), "build the native Rust runtime before testing"
            )
            native = ctypes.CDLL(str(NATIVE))
            for name in ("model_ptr", "state_ptr", "actions_ptr", "output_ptr"):
                getattr(native, name).restype = ctypes.c_void_p
            native.load_model.argtypes = [ctypes.c_size_t]
            native.infer.argtypes = [ctypes.c_size_t, ctypes.c_uint32]
            blob = (Path(temporary) / "policy.bin").read_bytes()
            ctypes.memmove(native.model_ptr(), blob, len(blob))
            self.assertEqual(native.load_model(len(blob)), 0)
            for case, want in zip(cases, expected, strict=True):
                state = (ctypes.c_float * 768)(*case["features"])
                ctypes.memmove(native.state_ptr(), state, ctypes.sizeof(state))
                actions = (ctypes.c_float * len(case["actions"]))(*case["actions"])
                ctypes.memmove(native.actions_ptr(), actions, ctypes.sizeof(actions))
                count = native.infer(case["count"], case["signal"])
                self.assertEqual(count, len(want))
                got = list((ctypes.c_float * count).from_address(native.output_ptr()))
                np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
