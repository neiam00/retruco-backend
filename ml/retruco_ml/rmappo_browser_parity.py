"""Authorized recurrent replay parity. No checkpoint writes or approval."""

import argparse
import copy
import ctypes
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .artifacts import digest
from .bridge import Simulator
from .bundle import runtime_sources
from .mappo_export import load_actor, quantized
from .parity import differences
from .recurrent import Replay, limit_opening_bluff
from .team_play_probe import CHOICE_PROFILES, team_action


def collect_cases(actor, executable, choice_policy=None):
    if choice_policy not in (None, *CHOICE_PROFILES):
        raise ValueError("unsupported browser choices")
    options = (
        {"playingSeats": [True] * 4, "choicePolicy": choice_policy}
        if choice_policy
        else {}
    )
    cases = []
    with Simulator(executable, "binary") as simulator, torch.no_grad():
        if choice_policy in (
            "value-and-team-v8",
            "value-and-team-v9",
            "value-and-team-v10",
            "value-and-team-v11",
            "value-and-team-v12",
            "value-and-team-v13",
            "value-and-team-v14",
        ):
            obs = simulator.call(
                [
                    {
                        **options,
                        "op": "reset",
                        "id": 0,
                        "fixture": "weak-opening-v1",
                        "training": True,
                        "truthfulCPU": True,
                        "seed": 124999,
                        "config": {"players": 2, "target": 12, "florRequired": False},
                    }
                ]
            )[0]
            if choice_policy in (
                "value-and-team-v11",
                "value-and-team-v12",
                "value-and-team-v13",
                "value-and-team-v14",
            ):
                if (obs["actions"][:, 47] == -1).any() or any(
                    c["type"] == "call"
                    and c.get("kind") == "truco"
                    and c.get("amount") == 3
                    for c in obs["legal"]
                ):
                    raise ValueError("The zero-cover opening passed the v11 filter")
                # This ordinary development deal retains a bounded opening. Test
                # its real legal actions and journal in each numerical runtime.
                cut = simulator.call(
                    [
                        {
                            **options,
                            "op": "reset",
                            "training": True,
                            "truthfulCPU": True,
                            "id": 0,
                            "seed": 378034,
                            "roles": [True] * 4,
                            "config": {
                                "players": 2,
                                "target": 12,
                                "florRequired": True,
                            },
                        }
                    ]
                )[0]
                if cut["legal"][0]["type"] != "cut":
                    raise ValueError("Expected a fresh hand before the parity opening")
                obs = simulator.call([{"op": "step", "id": 0, "action": 0}])[0]
            page = obs["training"]["memory"]
            if (
                page["start"] != 0
                or page["end"] != page["total"]
                or (obs["actions"][:, 47] == -1).sum() != 1
            ):
                raise ValueError(
                    "Missing the authentic weak opening and its complete journal"
                )
            cases.append(
                {
                    "name": "heads-up-weak-opening",
                    "features": obs["features"].tolist(),
                    "actions": obs["actions"].tolist(),
                    "signal": False,
                    "signalMask": None,
                    "view": obs["view"],
                    "seat": obs["seat"],
                    "synthetic": False,
                    "events": np.asarray(page["events"]).reshape(-1, 768).tolist(),
                    "hand": page["hand"],
                    "entropy": 124999,
                }
            )
        for players in (2, 4):
            for target in (12, 24):
                for flor in (False, True):
                    seed = 124000 + len(cases)
                    torch.manual_seed(seed)
                    obs = simulator.call(
                        [
                            {
                                **options,
                                "op": "reset",
                                "training": True,
                                "truthfulCPU": True,
                                "id": 0,
                                "seed": seed,
                                "roles": [True] * 4,
                                "config": {
                                    "players": players,
                                    "target": target,
                                    "florRequired": flor,
                                },
                            }
                        ]
                    )[0]
                    replays = {seat: Replay(actor) for seat in range(players)}
                    cursors, hand, kept = [0] * 4, 0, 0
                    for step in range(10000):
                        if kept >= 24:
                            break
                        if obs["done"]:
                            seed += 1000
                            obs = simulator.call(
                                [
                                    {
                                        **options,
                                        "op": "reset",
                                        "training": True,
                                        "truthfulCPU": True,
                                        "id": 0,
                                        "seed": seed,
                                        "roles": [True] * 4,
                                        "config": {
                                            "players": players,
                                            "target": target,
                                            "florRequired": flor,
                                        },
                                    }
                                ]
                            )[0]
                            replays = {s: Replay(actor) for s in range(players)}
                            cursors, hand = [0] * 4, 0
                        if (
                            choice_policy
                            and obs["kind"] == "game"
                            and obs.get("choicePolicy") != choice_policy
                        ):
                            raise ValueError(
                                "simulator did not supply the browser choice policy"
                            )
                        seat = obs["seat"]
                        page = obs["training"]["memory"]
                        if page["hand"] != hand:
                            hand, cursors = page["hand"], [0] * 4
                        while True:
                            replays[seat].apply(page)
                            cursors[seat] = page["end"]
                            if page["end"] == page["total"]:
                                break
                            obs = simulator.call(
                                [
                                    {
                                        "op": "observe",
                                        "id": 0,
                                        "memoryHand": hand,
                                        "memoryCursors": cursors,
                                    }
                                ]
                            )[0]
                            page = obs["training"]["memory"]
                        if step % 7 == 0:
                            cases.append(
                                {
                                    "name": f"{players}p-{target}-{int(flor)}-{step}",
                                    "features": obs["features"].tolist(),
                                    "actions": obs["actions"].tolist(),
                                    "signal": obs["kind"] == "signal",
                                    "signalMask": obs.get("signalMask"),
                                    "view": obs["view"],
                                    "seat": seat,
                                    "synthetic": False,
                                    "events": [
                                        event.tolist() for event in replays[seat].events
                                    ],
                                    "hand": hand,
                                    "entropy": (seed * 17 + step) & 0xFFFFFFFF,
                                }
                            )
                            kept += 1
                        context = actor.context(
                            torch.as_tensor(obs["features"]), replays[seat].memory
                        )
                        action = int(actor.distribution(context, obs).sample())
                        obs = simulator.call(
                            [
                                {
                                    "op": "step",
                                    "id": 0,
                                    "action": action,
                                    "memoryHand": hand,
                                    "memoryCursors": cursors,
                                }
                            ]
                        )[0]
    base = max((c for c in cases if not c["signal"]), key=lambda c: len(c["events"]))
    stress = copy.deepcopy(base)
    stress.update(
        name="synthetic-1024-candidates",
        synthetic=True,
        actions=[base["actions"][i % len(base["actions"])] for i in range(1024)],
    )
    stress["view"]["legal"] = [
        base["view"]["legal"][i % len(base["view"]["legal"])] for i in range(1024)
    ]
    cases.append(stress)
    long = copy.deepcopy(base)
    long.update(
        name="synthetic-2048-events",
        synthetic=True,
        events=[base["events"][i % len(base["events"])] for i in range(2048)],
    )
    cases.append(long)
    return cases


def torch_results(actor, cases):
    logits, states = [], []
    with torch.no_grad():
        for case in cases:
            memory = actor.initial()
            for event in case["events"]:
                memory = actor.advance(torch.tensor(event, dtype=torch.float32), memory)
            context = actor.context(torch.tensor(case["features"]), memory)
            if case["signal"]:
                result = actor.signal_logits(context)
            else:
                actions = torch.tensor(case["actions"], dtype=torch.float32)
                local = (
                    torch.tensor(case["features"]).expand(len(actions), -1)
                    if actor.candidate_features
                    else None
                )
                result = actor.game_logits(
                    context.expand(len(actions), -1), actions, local
                )
                result = limit_opening_bluff(result, actions[:, 47] == -1)
            logits.append((result - result.max()).numpy())
            states.append(torch.cat(memory).numpy())
    return logits, states


def benchmark_cases(cases, executable):
    """Include public-history and recurrent-history limits together, not just separately."""
    real = [c for c in cases if not c["synthetic"] and c["view"]["phase"] == "playing"]
    selected = [
        next(c for c in real if c["signal"]),
        max((c for c in real if not c["signal"]), key=lambda c: len(c["actions"])),
        max(real, key=lambda c: len(c["events"])),
        *cases[-2:],
    ]
    base = max(
        (c for c in real if not c["signal"] and c["view"]["config"]["players"] == 4),
        key=lambda c: len(c["events"]),
    )
    context = {
        "cpu": [True] * 4,
        "complete": True,
        "signalsComplete": True,
        "cantos": [
            {
                "hand": base["hand"],
                "event": {
                    "type": "call",
                    "seat": i % 4,
                    "kind": ("truco", "envido", "flor", "privo")[i % 4],
                    "amount": i % 9 + 1,
                },
            }
            for i in range(64)
        ],
        "signals": [
            {
                "seat": i % 4,
                "kind": ("perico", "perica", "seven_spades", "seven_coins", "receta")[
                    i % 5
                ],
            }
            for i in range(20)
        ],
        "signalCounts": [[0] * 5 for _ in range(4)],
    }
    for i in range(20):
        context["signalCounts"][i % 4][i % 5] += 1
    with Simulator(executable, "binary") as simulator:
        for count, depth in ((len(base["actions"]), 32), (1024, 2048)):
            view = copy.deepcopy(base["view"])
            view["legal"] = [
                base["view"]["legal"][i % len(base["actions"])] for i in range(count)
            ]
            encoded = simulator.call(
                [
                    {
                        "op": "encode",
                        "id": 0,
                        "view": view,
                        "seat": base["seat"],
                        "kind": "game",
                        "context": context,
                    }
                ]
            )[0]
            selected.append(
                {
                    **base,
                    "name": f"synthetic-full-public-history-{depth}-events-{count}-candidates",
                    "synthetic": True,
                    "view": view,
                    "features": encoded["features"].tolist(),
                    "actions": encoded["actions"].tolist(),
                    "events": [encoded["features"].tolist() for _ in range(depth)],
                    "history": {"signals": 20, "cantos": 64, "encoder": "shared-go"},
                }
            )
    return list({c["name"]: c for c in selected}.values())


def native_results(library, blob, cases, profile=None):
    native = ctypes.CDLL(str(library))
    for name in ("model_ptr", "state_ptr", "actions_ptr", "output_ptr", "memory_ptr"):
        getattr(native, name).restype = ctypes.c_void_p
    native.load_model.argtypes = [ctypes.c_size_t]
    native.infer.argtypes = [ctypes.c_size_t, ctypes.c_uint32]
    ctypes.memmove(native.model_ptr(), blob, len(blob))
    if native.load_model(len(blob)) != 0:
        raise ValueError("native load failed")
    logits, states = [], []
    for case in cases:
        native.reset_memory()
        for event in case["events"]:
            row = np.asarray(event, dtype=np.float32)
            ctypes.memmove(native.state_ptr(), row.ctypes.data, row.nbytes)
            if native.advance_memory() != 0:
                raise ValueError("native replay failed")
        row = np.asarray(case["features"], dtype=np.float32)
        actions = np.asarray(case["actions"], dtype=np.float32)
        ctypes.memmove(native.state_ptr(), row.ctypes.data, row.nbytes)
        ctypes.memmove(native.actions_ptr(), actions.ctypes.data, actions.nbytes)
        count = native.infer(len(actions), int(case["signal"]))
        if count != (6 if case["signal"] else len(actions)):
            raise ValueError("native inference failed")
        if "team_actions_v1" in case:
            adjust = (
                native.team_action_v2
                if profile == "closing-card-and-flor-v2"
                else native.team_action_v1
            )
            mapped = [adjust(len(actions), i) for i in range(len(actions))]
            if mapped != case["team_actions_v1"]:
                raise ValueError(
                    "native team decisions differ from the frozen evaluation rule"
                )
        logits.append(
            np.array(
                (ctypes.c_float * count).from_address(native.output_ptr()), copy=True
            )
        )
        states.append(
            np.array(
                (ctypes.c_float * 128).from_address(native.memory_ptr()), copy=True
            )
        )
    return logits, states


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("bundle", type=Path)
    p.add_argument("--simulator", type=Path, required=True)
    p.add_argument("--frontend", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads((a.bundle / "manifest.json").read_text())
    blob = (a.bundle / "policy.bin").read_bytes()
    if (
        digest(a.checkpoint) != manifest["checkpoint_sha256"]
        or digest(a.bundle / "policy.bin") != manifest["sha256"]
        or digest(a.bundle / "policy.wasm") != manifest["wasm_sha256"]
    ):
        raise ValueError("candidate artifact mismatch")
    runtime = Path(__file__).resolve().parents[1] / "runtime"
    if runtime_sources(runtime) != manifest["runtime_source_sha256"]:
        raise ValueError("runtime source mismatch")
    subprocess.run(
        ["cargo", "build", "--locked", "--offline", "--release"],
        cwd=runtime,
        check=True,
    )
    library = (
        runtime
        / "target/release"
        / (
            "libretruco_policy.dylib"
            if sys.platform == "darwin"
            else "libretruco_policy.so"
        )
    )
    actor = load_actor(a.checkpoint)
    dequantized = quantized(actor)
    casefile = a.output.with_suffix(".cases.json")
    if casefile.exists():
        saved_cases = json.loads(casefile.read_text())
        if saved_cases.get("choice_policy") != manifest.get("choice_policy"):
            raise ValueError(
                "parity cases were collected with a different choice policy"
            )
        cases = saved_cases["cases"]
    else:
        cases = collect_cases(actor, a.simulator, manifest.get("choice_policy"))
        casefile.write_text(
            json.dumps(
                {
                    "format": 3,
                    "encoder": 2,
                    "action_encoder": 1,
                    "choice_policy": manifest.get("choice_policy"),
                    "cases": cases,
                }
            )
            + "\n"
        )
    webcasefile = casefile
    if manifest.get("team_play"):
        if manifest["team_play"] not in (
            "closing-card-and-flor-v1",
            "closing-card-and-flor-v2",
        ):
            raise ValueError("unsupported team decision rule")
        for case in cases:
            if case["signal"]:
                continue
            legal = case["view"]["legal"]
            if len(legal) != len(case["actions"]):
                raise ValueError(
                    "missing authorized legal actions in team decision cases"
                )
            case["team_actions_v1"] = [
                team_action(
                    case["view"],
                    case["features"],
                    case["seat"],
                    legal,
                    i,
                    manifest["team_play"],
                )[0]
                for i in range(len(legal))
            ]
        webcasefile = a.output.with_suffix(".team-cases.json")
        webcasefile.write_text(
            json.dumps(
                {
                    "format": 3,
                    "encoder": 2,
                    "action_encoder": 1,
                    "choice_policy": manifest.get("choice_policy"),
                    "cases": cases,
                }
            )
            + "\n"
        )
    full, fullstates = torch_results(actor, cases)
    quantized_logits, quantized_states = torch_results(dequantized, cases)
    native, native_states = native_results(
        library, blob, cases, manifest.get("team_play")
    )
    webfile = a.output.with_suffix(".web.json")
    subprocess.run(
        [
            "node",
            "scripts/verify-cpu-parity.mjs",
            str(a.bundle.resolve()),
            str(webcasefile.resolve()),
            str(webfile.resolve()),
        ],
        cwd=a.frontend,
        check=True,
    )
    web = json.loads(webfile.read_text())
    measured = {
        name: differences(quantized_logits, values)
        for name, values in [
            ("native", native),
            ("node", web["node"]),
            ("browser", web["browser"]),
        ]
    }
    memory_errors = {
        name: float(np.max(np.abs(np.asarray(quantized_states) - np.asarray(values))))
        for name, values in [
            ("native", native_states),
            ("node", web["node_states"]),
            ("browser", web["browser_states"]),
        ]
    }
    report = {
        "approved": False,
        "checkpoint_sha256": digest(a.checkpoint),
        "model_sha256": manifest["sha256"],
        "wasm_sha256": manifest["wasm_sha256"],
        "runtime_source_sha256": runtime_sources(runtime),
        "native_sha256": digest(library),
        "cases": len(cases),
        "authorized_cases": sum(not c["synthetic"] for c in cases),
        "max_events": max(len(c["events"]) for c in cases),
        "max_candidates": max(len(c["actions"]) for c in cases),
        "runtime_comparisons": measured,
        "memory_max_absolute_errors": memory_errors,
        "quantization_vs_original": differences(full, quantized_logits),
        "max_probability_error": max(
            v["max_probability_error"] for v in measured.values()
        ),
        "browser_version": web["browser_version"],
        "wasm_memory_bytes": web["memory_bytes"],
        "cases_sha256": digest(casefile),
        "web_evidence_sha256": digest(webfile),
        "scope": "numerical and recurrent replay evidence; playing-strength quantization gate is separate",
    }
    if manifest.get("choice_policy"):
        report["choice_policy"] = manifest["choice_policy"]
    if manifest.get("team_play"):
        report["team_play"] = manifest["team_play"]
        report["team_cases_sha256"] = digest(webcasefile)
        report["team_decisions"] = web["team_decisions"]
    evidence = a.output.with_suffix(".evidence.json")
    evidence.write_text(
        json.dumps(
            {
                "float": [x.tolist() for x in full],
                "quantized": [x.tolist() for x in quantized_logits],
                "native": [x.tolist() for x in native],
                "float_states": [x.tolist() for x in fullstates],
                "quantized_states": [x.tolist() for x in quantized_states],
                "native_states": [x.tolist() for x in native_states],
            }
        )
        + "\n"
    )
    report["evidence_sha256"] = digest(evidence)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if report["max_probability_error"] > 1e-4:
        raise SystemExit("runtime probability parity failed")


if __name__ == "__main__":
    main()
