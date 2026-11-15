"""Candidate-bound PyTorch/native Rust/Node/browser numerical verification."""

import argparse
import ctypes
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .artifacts import digest, read_quantized
from .assessment import load_checkpoint
from .bridge import Simulator
from .bundle import runtime_sources
from .export import serialize
from .model import ENCODER_VERSION


def collect_cases():
    cases = []
    with Simulator() as simulator:
        for players in (2, 4):
            observation = simulator.call(
                [
                    {
                        "op": "reset",
                        "id": 0,
                        "seed": 81813,
                        "roles": [True] * 4,
                        "config": {
                            "players": players,
                            "target": 24,
                            "florRequired": True,
                        },
                    }
                ]
            )[0]
            count = 0
            for step in range(10000):
                if observation["done"]:
                    break
                if observation["version"] != ENCODER_VERSION:
                    raise ValueError("simulator encoder mismatch")
                signal = observation["kind"] == "signal"
                if step % 3 == 0 and count < 24:
                    cases.append(
                        {
                            "name": f"{players}-player-{step}",
                            "features": observation["features"],
                            "actions": observation["actions"],
                            "signal": signal,
                            "synthetic": False,
                        }
                    )
                    count += 1
                observation = simulator.call(
                    [
                        {
                            "op": "step",
                            "id": 0,
                            "action": 1 if signal else observation["baseline"],
                        }
                    ]
                )[0]
            if not observation["done"]:
                raise RuntimeError("parity fixture match stalled")
    base = max((c for c in cases if not c["signal"]), key=lambda c: len(c["actions"]))
    cases.append(
        {
            **base,
            "name": "synthetic-1024-candidate-capacity",
            "synthetic": True,
            "actions": [base["actions"][i % len(base["actions"])] for i in range(1024)],
        }
    )
    if len(cases) < 20 or not any(c["signal"] for c in cases):
        raise RuntimeError("insufficient parity fixtures")
    return cases


def torch_logits(model, cases):
    results = []
    with torch.no_grad():
        for case in cases:
            state = torch.tensor([case["features"]], dtype=torch.float32)
            actions = torch.tensor(
                [case["actions"] or [[0.0] * 48]], dtype=torch.float32
            )
            game, signals, _ = model(state, actions)
            results.append((signals if case["signal"] else game)[0].numpy().copy())
    return results


def native_logits(library, blob, cases):
    native = ctypes.CDLL(str(library))
    for name in ("model_ptr", "state_ptr", "actions_ptr", "output_ptr"):
        getattr(native, name).restype = ctypes.c_void_p
    native.load_model.argtypes = [ctypes.c_size_t]
    native.infer.argtypes = [ctypes.c_size_t, ctypes.c_uint32]
    ctypes.memmove(native.model_ptr(), blob, len(blob))
    if native.load_model(len(blob)) != 0:
        raise ValueError("native model load failed")
    results = []
    for case in cases:
        state = np.asarray(case["features"], dtype=np.float32)
        actions = np.asarray(case["actions"], dtype=np.float32)
        ctypes.memmove(native.state_ptr(), state.ctypes.data, state.nbytes)
        ctypes.memmove(native.actions_ptr(), actions.ctypes.data, actions.nbytes)
        count = native.infer(len(case["actions"]), int(case["signal"]))
        if count != (6 if case["signal"] else len(case["actions"])):
            raise ValueError("native inference failed")
        results.append(
            np.array(
                (ctypes.c_float * count).from_address(native.output_ptr()), copy=True
            )
        )
    return results


def probabilities(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("invalid or nonfinite inference outputs")
    weights = np.exp(values - values.max())
    return weights / weights.sum()


def differences(reference, actual):
    if len(reference) != len(actual):
        raise ValueError("inference case count mismatch")
    logits = probability = variation = 0.0
    for expected, result in zip(reference, actual, strict=True):
        expected = np.asarray(expected)
        result = np.asarray(result)
        if expected.shape != result.shape:
            raise ValueError("inference output shape mismatch")
        delta = np.abs(probabilities(expected) - probabilities(result))
        logits = max(logits, float(np.max(np.abs(expected - result))))
        probability = max(probability, float(np.max(delta)))
        variation = max(variation, float(np.sum(delta) / 2))
    return {
        "max_logit_error": logits,
        "max_probability_error": probability,
        "max_total_variation": variation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_checkpoint(args.checkpoint)
    folder = args.bundle.resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    blob = (folder / "policy.bin").read_bytes()
    data, reference = serialize(model)
    if blob != data or (folder / "reference.f32").read_bytes() != reference:
        raise ValueError("bundle is not the exact export of this checkpoint")
    if (
        manifest["checkpoint_sha256"] != model.checkpoint_sha256
        or manifest["sha256"] != digest(folder / "policy.bin")
        or manifest["wasm_sha256"] != digest(folder / "policy.wasm")
    ):
        raise ValueError("bundle manifest checksum mismatch")
    runtime = Path(__file__).resolve().parents[1] / "runtime"
    source_hash = runtime_sources(runtime)
    if source_hash != manifest["runtime_source_sha256"]:
        raise ValueError("native runtime source differs from bundle")
    subprocess.run(
        ["cargo", "build", "--locked", "--offline", "--release"],
        cwd=runtime,
        check=True,
    )
    if runtime_sources(runtime) != source_hash:
        raise ValueError("runtime source changed during native build")
    library = (
        runtime
        / "target/release"
        / (
            "libretruco_policy.dylib"
            if sys.platform == "darwin"
            else "libretruco_policy.so"
        )
    )
    cases = collect_cases()
    full = torch_logits(model, cases)
    quantized = torch_logits(read_quantized(blob, model), cases)
    native = native_logits(library, blob, cases)
    frontend = args.frontend.resolve()
    with tempfile.TemporaryDirectory(prefix="retruco-parity-") as temporary:
        casefile = Path(temporary) / "cases.json"
        resultfile = Path(temporary) / "results.json"
        casefile.write_text(json.dumps({"encoder": ENCODER_VERSION, "cases": cases}))
        subprocess.run(
            [
                "node",
                "scripts/verify-cpu-parity.mjs",
                str(folder),
                str(casefile),
                str(resultfile),
            ],
            cwd=frontend,
            check=True,
        )
        result = json.loads(resultfile.read_text())
        measured = {
            name: differences(quantized, values)
            for name, values in (
                ("native-rust", native),
                ("node-wasm", result["node"]),
                ("browser-wasm", result["browser"]),
            )
        }
        report = {
            "approved": False,
            "checkpoint_sha256": model.checkpoint_sha256,
            "model_sha256": manifest["sha256"],
            "wasm_sha256": manifest["wasm_sha256"],
            "runtime_source_sha256": source_hash,
            "native_binary_sha256": digest(library),
            "cases": len(cases),
            "authorized_cases": sum(not c["synthetic"] for c in cases),
            "max_candidates": max(len(c["actions"]) for c in cases),
            "engines": ["pytorch", *measured],
            "max_probability_error": max(
                v["max_probability_error"] for v in measured.values()
            ),
            "nonfinite_outputs": 0,
            "comparisons_to_dequantized_pytorch": measured,
            "quantization_vs_float32": differences(full, quantized),
            "cases_sha256": digest(casefile),
            "browser": {
                key: value
                for key, value in result.items()
                if key not in ("node", "browser")
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Retain exact input/output evidence so summaries can be independently recomputed.
        evidence = args.output.with_suffix(".evidence.json")
        evidence.write_text(
            json.dumps(
                {
                    "cases": cases,
                    "float32": [x.tolist() for x in full],
                    "quantized": [x.tolist() for x in quantized],
                    "native": [x.tolist() for x in native],
                    "node": result["node"],
                    "browser": result["browser"],
                }
            )
            + "\n"
        )
        report["evidence_path"] = evidence.name
        report["evidence_sha256"] = digest(evidence)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("cases", "max_probability_error", "quantization_vs_float32")
            }
        )
    )
    if report["max_probability_error"] > 1e-4:
        raise SystemExit("runtime probability parity failed")


if __name__ == "__main__":
    main()
