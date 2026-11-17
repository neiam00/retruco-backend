"""Seat-local inference for a frozen format-4 browser actor during release assessment."""

import ctypes
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from .artifacts import digest
from .bundle import runtime_sources
from .mappo_export import export, load_actor
from .recurrent import limit_opening_bluff
from .team_play_probe import CHOICE_PROFILES, team_action


def mask_features(rows, mode):
    if mode == "none":
        return rows
    if mode not in ("signals", "adaptation", "both"):
        raise ValueError("Unknown release ablation")
    rows = np.array(rows, dtype=np.float32, copy=True)
    if mode in ("adaptation", "both"):
        rows[..., 352:384] = 0
        rows[..., 512:716] = 0
    if mode in ("signals", "both"):
        rows[..., 416:476] = 0
        rows[..., 672:717] = 0
    return rows


class BrowserActor:
    """One native runtime instance per process. Each match/seat owns its memory."""

    def __init__(self, checkpoint, folder, library, *, fp32=False):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / "manifest.json").read_text())
        self.checkpoint_sha256 = digest(checkpoint)
        m = self.manifest
        self.profile = m["choice_policy"]
        if (
            self.profile not in CHOICE_PROFILES
            or m["team_play"] != "closing-card-and-flor-v2"
            or (
                m["format"],
                m["encoder"],
                m["action_encoder"],
                m["memory_size"],
                m["memory_version"],
                m["memory_reset"],
            )
            != (4, 2, 1, 64, 1, "hand")
            or m["checkpoint_sha256"] != self.checkpoint_sha256
            or m["source_dirty"] is not False
        ):
            raise ValueError("Unsupported release actor or provenance")
        self.blob = (self.folder / "policy.bin").read_bytes()
        if (
            digest(self.folder / "policy.bin") != m["sha256"]
            or digest(self.folder / "policy.wasm") != m["wasm_sha256"]
        ):
            raise ValueError("Model or WASM hash mismatch")
        if (
            runtime_sources(Path(__file__).resolve().parents[1] / "runtime")
            != m["runtime_source_sha256"]
        ):
            raise ValueError("Runtime source differs from the browser bundle")
        model = load_actor(Path(checkpoint)).eval()
        with TemporaryDirectory() as name:
            other = export(model, Path(name))
            if (
                (Path(name) / "policy.bin").read_bytes() != self.blob
                or digest(Path(name) / "reference.f32")
                != digest(self.folder / "reference.f32")
                or other["parameters"] != m["actor_parameters"]
                or m["actor_parameters"] > 150000
                or len(self.blob) > 128 * 1024
            ):
                raise ValueError("Artifact is not the exact checked checkpoint export")
        self.float_actor = model if fp32 else None
        self.native = ctypes.CDLL(str(Path(library).resolve()))
        if self.profile in (
            "value-and-team-v8",
            "value-and-team-v9",
            "value-and-team-v10",
            "value-and-team-v11",
            "value-and-team-v12",
            "value-and-team-v13",
            "value-and-team-v14",
            "value-and-team-v15",
            "value-and-team-v16",
            "value-and-team-v17",
        ) and (
            not hasattr(self.native, "opening_bluff_version")
            or self.native.opening_bluff_version() != 1
        ):
            raise ValueError("Missing bounded opening-bluff runtime")
        for name in (
            "model_ptr",
            "state_ptr",
            "actions_ptr",
            "output_ptr",
            "memory_ptr",
        ):
            getattr(self.native, name).restype = ctypes.c_void_p
        self.native.load_model.argtypes = [ctypes.c_size_t]
        self.native.infer.argtypes = [ctypes.c_size_t, ctypes.c_uint32]
        self.native.team_action_v2.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        ctypes.memmove(self.native.model_ptr(), self.blob, len(self.blob))
        if self.native.load_model(len(self.blob)) or self.native.memory_size() != 64:
            raise ValueError("Native release evaluator rejected the checked model")

        def view(ptr, n):
            return np.ctypeslib.as_array((ctypes.c_float * n).from_address(ptr()))

        self.state = view(self.native.state_ptr, 768)
        self.actions = view(self.native.actions_ptr, 1024 * 48).reshape(1024, 48)
        self.memory = view(self.native.memory_ptr, 128)
        self.output = view(self.native.output_ptr, 1024)

    def advance(self, memory, events, mode):
        events = mask_features(events, mode)
        if self.float_actor is not None:
            hidden, cell = torch.from_numpy(memory.copy()).chunk(2)
            with torch.no_grad():
                for row in events:
                    hidden, cell = self.float_actor.advance(
                        torch.from_numpy(row), (hidden, cell)
                    )
            memory[:] = torch.cat((hidden, cell)).numpy()
        else:
            self.memory[:] = memory
            for row in events:
                self.state[:] = row
                if self.native.advance_memory():
                    raise ValueError("Native recurrent replay failed")
            memory[:] = self.memory

    def probabilities(self, observation, memory, mode="none"):
        signal = observation["kind"] == "signal"
        if not signal and observation.get("choicePolicy") != self.profile:
            raise ValueError(
                "Acting seat did not receive the frozen browser choice policy"
            )
        features = mask_features(np.asarray(observation["features"], np.float32), mode)
        actions = np.asarray(observation["actions"], np.float32).reshape(-1, 48)
        if (
            features.shape != (768,)
            or len(actions) > 1024
            or (not signal and not len(actions))
        ):
            raise ValueError("Malformed authorized features")
        self.state[:] = features
        self.memory[:] = memory
        self.actions[: len(actions)] = actions
        if self.float_actor is None:
            count = self.native.infer(len(actions), int(signal))
            if count != (6 if signal else len(actions)):
                raise ValueError("Native release inference failed")
            logits = self.output[:count].astype(np.float64)
        else:
            with torch.no_grad():
                h, c = torch.from_numpy(memory).chunk(2)
                x = torch.from_numpy(features)
                context = self.float_actor.context(x, (h, c))
                if signal:
                    raw = self.float_actor.signal_logits(context)
                else:
                    raw = self.float_actor.game_logits(
                        context.expand(len(actions), -1), torch.from_numpy(actions)
                    )
                    raw = limit_opening_bluff(
                        raw, torch.from_numpy(actions[:, 47] == -1)
                    )
                logits = raw.detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite release logits")
        if signal:
            mask = np.asarray(observation["signalMask"])
            if mask.shape != (6,) or mask.dtype != np.bool_ or not mask[0]:
                raise ValueError("Invalid truthful CPU signal mask")
        else:
            mask = np.ones(len(logits), dtype=bool)
        weights = np.zeros(len(logits), dtype=np.float64)
        weights[mask] = np.exp(logits[mask] - logits[mask].max())
        return weights / weights.sum()

    def choose(self, observation, memory, rng, mode="none"):
        signal = observation["kind"] == "signal"
        if not signal and observation["view"]["phase"] == "cutting":
            return 0
        if signal and mode in ("both", "signals"):
            return 0
        weights = self.probabilities(observation, memory, mode)
        threshold = rng.getrandbits(32) / 4294967296
        chosen = int(np.searchsorted(weights.cumsum(), threshold, side="right"))
        if chosen == len(weights):
            chosen = int(np.flatnonzero(weights)[-1])
        if not signal and observation["features"][0] == 1:
            if self.float_actor is None:
                chosen = self.native.team_action_v2(len(weights), chosen)
            else:
                chosen = team_action(
                    observation["view"],
                    mask_features(observation["features"], mode),
                    observation["seat"],
                    observation["legal"],
                    chosen,
                    self.profile,
                )[0]
        if not 0 <= chosen < len(weights) or (
            signal and not observation["signalMask"][chosen]
        ):
            raise ValueError("Invalid deployed action")
        return chosen


class SeatJournals:
    """Synchronous simulator decisions consume exactly the authorized delta per seat."""

    def __init__(self, actor, mode="none"):
        self.actor = actor
        self.mode = mode
        self.games = {}

    def reset(self, request, seats):
        seats = frozenset(seats)
        request.update(
            training=True,
            truthfulCPU=True,
            choicePolicy=self.actor.profile,
            playingSeats=[seat in seats for seat in range(4)],
        )
        self.games[request["id"]] = {
            "hand": 0,
            "cursors": [0] * 4,
            "seats": seats,
            "memory": {},
        }

    def parameters(self, identity):
        game = self.games[identity]
        return {"memoryHand": game["hand"], "memoryCursors": game["cursors"].copy()}

    def consume(self, pool, observations):
        todo = {i: o for i, o in observations.items() if not o["done"]}
        while todo:
            requests = []
            for identity, obs in todo.items():
                game = self.games[identity]
                page = obs["training"]["memory"]
                hand = page["hand"]
                if hand != game["hand"]:
                    if hand <= game["hand"] or page["start"] != 0:
                        raise ValueError("Out-of-order hand journal")
                    game.update(hand=hand, cursors=[0] * 4, memory={})
                seat = obs["seat"]
                if seat not in game["seats"]:
                    game["cursors"][seat] = page["total"]
                    continue
                start, end, total = page["start"], page["end"], page["total"]
                rows = np.asarray(page["events"], dtype=np.float32).reshape(-1, 768)
                if (
                    page["version"] != 1
                    or not page["complete"]
                    or start != game["cursors"][seat]
                    or not 0 <= start <= end <= total <= 2048
                    or end - start != len(rows)
                ):
                    raise ValueError("Incomplete or discontinuous authorized journal")
                if seat not in game["memory"]:
                    game["memory"][seat] = np.zeros(128, dtype=np.float32)
                self.actor.advance(game["memory"][seat], rows, self.mode)
                game["cursors"][seat] = end
                if end < total:
                    if not len(rows):
                        raise ValueError("Empty nonfinal memory page")
                    requests.append(
                        {"op": "observe", "id": identity, **self.parameters(identity)}
                    )
            if not requests:
                break
            next_obs = pool.call(requests)
            todo = {q["id"]: o for q, o in zip(requests, next_obs, strict=True)}
            observations.update(todo)

    def memory(self, identity, seat):
        return self.games[identity]["memory"][seat]

    def choose(self, identity, observation, rng):
        return self.actor.choose(
            observation, self.memory(identity, observation["seat"]), rng, self.mode
        )
