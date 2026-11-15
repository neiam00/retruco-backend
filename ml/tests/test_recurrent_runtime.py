"""Quantized recurrent artifacts must agree across PyTorch, native, and WASM."""

import ctypes
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from retruco_ml.mappo_export import export, quantized
from retruco_ml.recurrent import Actor
from retruco_ml.rmappo_model import RecurrentActor

RUNTIME = Path(__file__).resolve().parents[1] / "runtime/target"
WASM = RUNTIME / "wasm32v1-none/release/retruco_policy.wasm"
SCORE_EVENTS = (0, 31, 127)


# Exercise the browser ABI in a real WebAssembly engine, including load failure
# and retry/replay lifecycle behavior that the PyTorch oracle cannot observe.
WASM_SCRIPT = r"""
const fs = require('fs');
(async () => {
    const q = JSON.parse(fs.readFileSync(0, 'utf8'));
    const module = await WebAssembly.compile(fs.readFileSync(q.wasm));
    if (WebAssembly.Module.imports(module).length) throw Error('host import');
    const x = (await WebAssembly.instantiate(module, {})).exports;
    const bytes = fs.readFileSync(q.model), before = x.memory.buffer;
    const load = data => {
        new Uint8Array(before, x.model_ptr(), data.length).set(data);
        return x.load_model(data.length);
    };
    if (load(bytes)) throw Error('load');
    if (x.memory_size() !== q.hidden) throw Error('memory size');
    const memory = () => Array.from(new Float32Array(before, x.memory_ptr(), 2 * q.hidden));
    const advance = row => {
        new Float32Array(before, x.state_ptr(), 768).set(row);
        if (x.advance_memory()) throw Error('advance');
    };
    const score = signal => {
        const count = x.infer(1024, signal);
        if (count !== (signal ? 6 : 1024)) throw Error('infer count');
        const logits = Array.from(new Float32Array(before, x.output_ptr(), count));
        const maximum = Math.max(...logits), weights = logits.map(v => Math.exp(v - maximum));
        const total = weights.reduce((a, b) => a + b, 0);
        return {logits, probabilities: weights.map(v => v / total)};
    };
    new Float32Array(before, x.actions_ptr(), q.actions.length).set(q.actions);
    const states = [], outputs = [];
    for (let i = 0; i < q.events.length; i++) {
        advance(q.events[i]);
        states.push(memory());
        if (q.score_events.includes(i)) {
            const output = score(i === 31 ? 1 : 0);
            outputs.push(output);
            if (JSON.stringify(score(i === 31 ? 1 : 0)) !== JSON.stringify(output))
                throw Error('score retry drift');
            if (JSON.stringify(memory()) !== JSON.stringify(states.at(-1)))
                throw Error('score advanced memory');
        }
    }
    x.reset_memory();
    if (memory().some(value => value !== 0)) throw Error('reset');
    for (const row of q.events) advance(row);
    if (JSON.stringify(memory()) !== JSON.stringify(states.at(-1))) throw Error('restart drift');
    if (JSON.stringify(score(0)) !== JSON.stringify(outputs.at(-1))) throw Error('replay score drift');
    for (const invalid of q.invalid) {
        const data = Buffer.from(bytes);
        for (const [offset, value] of invalid.words) data.writeUInt32LE(value, offset);
        if (load(data.subarray(0, invalid.length)) >= 0) throw Error('accepted ' + invalid.name);
        if (x.memory_size() !== 0 || x.infer(1, 0) >= 0 || x.advance_memory() >= 0)
            throw Error('failed load retained model');
        if (load(bytes) || x.memory_size() !== q.hidden || memory().some(value => value !== 0))
            throw Error('load recovery');
    }
    if (x.memory.buffer !== before) throw Error('memory growth');
    process.stdout.write(JSON.stringify({states, outputs, memory: before.byteLength}));
})().catch(error => { console.error(error); process.exit(1); });
"""


def malformed_models(data, hidden):
    """Mutate valid external artifacts at ABI boundaries, keeping other bytes valid."""
    cases = [
        {"name": "truncated payload", "words": [], "length": len(data) - 1},
        {"name": "unsupported format", "words": [(4, 99)], "length": len(data)},
    ]
    if hidden == 16:
        return cases
    cases.extend(
        {
            "name": f"unsupported hidden size {size}",
            "words": [(12, size)],
            "length": len(data),
        }
        for size in (0, 16, 31, 33, 65, 0xFFFFFFFF)
    )
    cases.append({"name": "truncated width header", "words": [], "length": 15})
    offset = 16
    for _ in range(6):
        rows, cols = struct.unpack_from("<II", data, offset)
        offset += 8 + rows * 8 + rows * cols
    rows, cols = struct.unpack_from("<II", data, offset)
    projection = offset + 8 + rows * 8 + rows * cols
    for name, word_offset, value in (
        ("wrong gate width", offset, 4 * hidden - 1),
        ("wrong gate input", offset + 4, 64 + hidden - 1),
        ("nonfinite gate scale", offset + 8, 0x7FC00000),
        ("wrong projection width", projection + 4, hidden - 1),
    ):
        cases.append(
            {"name": name, "words": [(word_offset, value)], "length": len(data)}
        )
    return cases


class RecurrentRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        filename = (
            "libretruco_policy.dylib"
            if sys.platform == "darwin"
            else "libretruco_policy.so"
        )
        cls.library = ctypes.CDLL(str(RUNTIME / "release" / filename))
        cls.library.model_ptr.restype = ctypes.POINTER(ctypes.c_ubyte)
        cls.library.memory_size.restype = ctypes.c_size_t
        cls.library.load_model.argtypes = (ctypes.c_size_t,)
        cls.library.infer.argtypes = (ctypes.c_size_t, ctypes.c_uint32)
        for name in ("state_ptr", "memory_ptr", "actions_ptr", "output_ptr"):
            getattr(cls.library, name).restype = ctypes.POINTER(ctypes.c_float)

    def load_native(self, data):
        ctypes.memmove(self.library.model_ptr(), data, len(data))
        return self.library.load_model(len(data))

    def native_memory(self, hidden):
        return np.ctypeslib.as_array(
            self.library.memory_ptr(), shape=(2 * hidden,)
        ).copy()

    def advance_native(self, event):
        np.ctypeslib.as_array(self.library.state_ptr(), shape=(768,))[:] = event
        self.assertEqual(self.library.advance_memory(), 0)

    def score_native(self, signal):
        count = self.library.infer(1024, int(signal))
        self.assertEqual(count, 6 if signal else 1024)
        logits = np.ctypeslib.as_array(self.library.output_ptr(), shape=(count,)).copy()
        weights = np.exp(logits.astype(np.float64) - logits.max())
        return {"logits": logits, "probabilities": weights / weights.sum()}

    def assert_output_matches(self, actual, expected):
        np.testing.assert_allclose(
            actual["logits"], expected["logits"], atol=2e-6, rtol=2e-5
        )
        np.testing.assert_allclose(
            actual["probabilities"], expected["probabilities"], atol=2e-6, rtol=2e-4
        )

    def test_full_sequence_states_logits_and_replay_match_native_and_wasm(self):
        events = np.random.default_rng(7).normal(size=(128, 768)).astype(np.float32)
        actions = np.random.default_rng(8).normal(size=(1024, 48)).astype(np.float32)
        for hidden, rmappo in (
            (16, False),
            (32, False),
            (64, False),
            (64, True),
            (64, "context"),
        ):
            with (
                self.subTest(hidden=hidden, rmappo=rmappo),
                tempfile.TemporaryDirectory() as folder,
            ):
                torch.manual_seed(45)
                actor = (
                    RecurrentActor("lstm")
                    if rmappo
                    else Actor(True, memory_size=hidden)
                )
                if rmappo == "context":
                    from test_candidate_features import action, features, play

                    actor.enable_candidate_features()
                    torch.nn.init.normal_(actor.action[0].weight[:, 36:], std=0.05)
                    events = []
                    for step in range(128):
                        f = features(4 if step % 2 else 2)
                        f[1] = 0.5 if step % 3 else 1
                        for seat, rank in enumerate((0.9, 0.2, 0.4, 0.3)):
                            play(f, seat, 0, rank)
                        play(f, 2 if step % 2 else 0, 1, 1.0)
                        f[127] = 1
                        f[128] = (0.2, 0.4, 0.6, 0.8)[step % 4]
                        f[130] = 0.04
                        f[131] = 0.01
                        events.append(f.numpy())
                    events = np.asarray(events, np.float32)
                    commands = [action("play", i) for i in range(3)] + [
                        action(k) for k in ("accept", "decline", "raise")
                    ]
                    actions = np.asarray(
                        [commands[i % len(commands)].numpy() for i in range(1024)],
                        np.float32,
                    )
                torch.nn.init.normal_(actor.projection.weight, std=0.03)
                if rmappo:
                    # A large common score must not erase action differences
                    # through a premature FP32 cast in the deployment runtime.
                    with torch.no_grad():
                        actor.score[2].bias.fill_(1e7)
                        actor.signal.bias.fill_(1e7)
                reference = quantized(actor)
                memory = reference.initial()
                states, outputs = [], []
                with torch.no_grad():
                    for i, event in enumerate(events):
                        features = torch.as_tensor(event)
                        memory = reference.advance(features, memory)
                        states.append(torch.cat(memory).numpy().copy())
                        if i in SCORE_EVENTS:
                            context = reference.context(features, memory)
                            logits = (
                                reference.signal_logits(context)
                                if i == 31
                                else reference.game_logits(
                                    context.expand(len(actions), -1),
                                    torch.as_tensor(actions),
                                    features.expand(len(actions), -1)
                                    if rmappo == "context"
                                    else None,
                                )
                            )
                            observation = {
                                "version": 2,
                                "kind": "signal" if i == 31 else "game",
                                "actions": actions,
                                "features": event,
                            }
                            outputs.append(
                                {
                                    "logits": (
                                        logits - logits.max() if rmappo else logits
                                    ).numpy(),
                                    "probabilities": reference.distribution(
                                        context, observation
                                    ).probs.numpy(),
                                }
                            )
                manifest = export(actor, folder)
                self.assertLessEqual(manifest["bytes"], 128 * 1024)
                self.assertFalse(manifest["approved"])
                data = (Path(folder) / "policy.bin").read_bytes()
                self.assertEqual(
                    struct.unpack_from("<I", data, 4)[0],
                    5
                    if rmappo == "context"
                    else 4
                    if rmappo
                    else 2
                    if hidden == 16
                    else 3,
                )
                invalid = malformed_models(data, hidden)
                self.assertEqual(self.load_native(data), 0)
                self.assertEqual(self.library.memory_size(), hidden)
                np.ctypeslib.as_array(self.library.actions_ptr(), shape=(1024 * 48,))[
                    :
                ] = actions.ravel()
                output_index = 0
                for i, (event, expected) in enumerate(zip(events, states, strict=True)):
                    self.advance_native(event)
                    np.testing.assert_allclose(
                        self.native_memory(hidden), expected, atol=2e-6, rtol=2e-5
                    )
                    if i in SCORE_EVENTS:
                        before = self.native_memory(hidden)
                        actual = self.score_native(i == 31)
                        self.assert_output_matches(actual, outputs[output_index])
                        np.testing.assert_array_equal(
                            self.score_native(i == 31)["logits"], actual["logits"]
                        )
                        np.testing.assert_array_equal(
                            self.native_memory(hidden), before
                        )
                        output_index += 1
                final_memory = self.native_memory(hidden)
                final_output = self.score_native(False)
                self.library.reset_memory()
                np.testing.assert_array_equal(
                    self.native_memory(hidden), np.zeros(2 * hidden)
                )
                for event in events:
                    self.advance_native(event)
                np.testing.assert_array_equal(self.native_memory(hidden), final_memory)
                np.testing.assert_array_equal(
                    self.score_native(False)["logits"], final_output["logits"]
                )
                for case in invalid:
                    with self.subTest(corruption=case["name"]):
                        corrupt = bytearray(data)
                        for offset, value in case["words"]:
                            struct.pack_into("<I", corrupt, offset, value)
                        self.assertLess(
                            self.load_native(bytes(corrupt[: case["length"]])), 0
                        )
                        self.assertEqual(self.library.memory_size(), 0)
                        self.assertLess(self.library.infer(1, 0), 0)
                        self.assertLess(self.library.advance_memory(), 0)
                        self.assertEqual(self.load_native(data), 0)
                        self.assertEqual(self.library.memory_size(), hidden)
                        np.testing.assert_array_equal(
                            self.native_memory(hidden), np.zeros(2 * hidden)
                        )
                result = subprocess.run(
                    ["node", "-e", WASM_SCRIPT],
                    capture_output=True,
                    text=True,
                    check=True,
                    input=json.dumps(
                        {
                            "wasm": str(WASM),
                            "model": str(Path(folder) / "policy.bin"),
                            "hidden": hidden,
                            "events": events.tolist(),
                            "actions": actions.ravel().tolist(),
                            "score_events": SCORE_EVENTS,
                            "invalid": invalid,
                        }
                    ),
                )
                actual = json.loads(result.stdout)
                self.assertLessEqual(actual["memory"], 4 * 1024 * 1024)
                np.testing.assert_allclose(
                    actual["states"], states, atol=2e-6, rtol=2e-5
                )
                for actual_output, expected_output in zip(
                    actual["outputs"], outputs, strict=True
                ):
                    self.assert_output_matches(actual_output, expected_output)

    def test_gru_export_is_rejected_instead_of_silently_using_lstm_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "LSTM, not GRU"):
                export(RecurrentActor("gru"), folder)


if __name__ == "__main__":
    unittest.main()
