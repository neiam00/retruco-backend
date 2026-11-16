"""Prospective paired test of the deployed policy's first bounded opening per match."""

import hashlib
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .controlled_train import atomic_json
from .evaluation_styles import evaluation_scripted
from .release_recurrent import BrowserActor, SeatJournals
from .rollout import SimulatorPool
from .splits import RELEASE_START
from .statistics import interval

MATCHES = 196608
CHUNK = 2048
MIN_OFFERS = 2000
PROTOCOL = "first-bounded-native-opening-v1"


# Each original 2,000-deal reservation owns at least 256,000 supplementary
# seeds. This namespace is disjoint from the old fixed-hand and training ranges.
def start_seed(reservation):
    if reservation["pairs"] < 2000:
        raise ValueError("Invalid primary reservation")
    return 100_000_000 + (reservation["seed"] - RELEASE_START) * 128


def play(actor, pool, seed, index, quiet):
    seat = (index // 4) % 2
    request = {
        "op": "reset",
        "id": 0,
        "seed": seed,
        "roles": [True] * 4,
        "config": {
            "players": 2,
            "target": (12, 24)[index % 2],
            "florRequired": bool((index // 2) % 2),
        },
    }
    journals = SeatJournals(actor)
    journals.reset(request, [seat])
    obs = pool.call([request])[0]
    rng = random.Random(seed * 11 + seat)
    rival_rng = random.Random(seed * 11 + 1 - seat)
    offer = None
    for _ in range(10000):
        if obs["done"]:
            return int(obs["winner"] == seat), offer
        journals.consume(pool, {0: obs})
        if obs["seat"] != seat:
            action = evaluation_scripted(obs, "baseline", rival_rng)
        else:
            marked = (
                np.flatnonzero(obs["actions"][:, 47] == -1)
                if offer is None and obs["kind"] == "game" and len(obs["actions"])
                else []
            )
            if len(marked):
                action = int(marked[0])
                cmd = obs["legal"][action]
                if len(marked) != 1 or cmd != {
                    "type": "call",
                    "kind": "truco",
                    "amount": 3,
                }:
                    raise ValueError("Invalid marked opening")
                memory = journals.memory(0, seat)
                probs = actor.probabilities(obs, memory)
                signature = hashlib.sha256()
                for tensor in (obs["features"], obs["actions"], memory):
                    signature.update(np.asarray(tensor, dtype="<f4").tobytes())
                signature.update(json.dumps(obs["legal"], sort_keys=True).encode())
                offer = (signature.hexdigest(), float(probs[action]))
                if quiet:
                    actions = [
                        i
                        for i, c in enumerate(obs["legal"])
                        if c["type"] == "play" and not c.get("declaration")
                    ]
                    action = max(actions, key=lambda i: probs[i])
            else:
                action = journals.choose(0, obs, rng)
        obs = pool.call(
            [{"op": "step", "id": 0, "action": action, **journals.parameters(0)}]
        )[0]
    raise ValueError("Opening assessment match did not finish")


def run_chunk(args):
    root, ident, first, offset = args
    folder = root / "partial" / "bluff-ordinary"
    path = folder / f"{offset}.json"
    if path.exists():
        part = json.loads(path.read_text())
        if (part["fingerprint"], part["first"], len(part["rows"])) != (
            ident,
            first + offset,
            CHUNK,
        ):
            raise ValueError("Mismatched ordinary-opening chunk")
        return part["rows"]
    torch.set_num_threads(1)
    actor = BrowserActor(
        root / "inputs/candidate.pt",
        root / "inputs/bundle",
        root / "bin/libretruco_policy.dylib",
    )
    rows = []
    with SimulatorPool(1, "binary", str(root / "bin/selfplay")) as pool:
        for idx in range(offset, offset + CHUNK):
            raised, offer = play(actor, pool, first + idx, idx, False)
            if offer is None:
                rows.append(None)
                continue
            quiet, replayed = play(actor, pool, first + idx, idx, True)
            if replayed != offer:
                raise ValueError("Unmatched decision across opening branches")
            rows.append([offer[0], offer[1], raised, quiet])
    atomic_json(path, {"fingerprint": ident, "first": first + offset, "rows": rows})
    return rows


def diagnose_ordinary(root, ident, reservation):
    first = start_seed(reservation)
    (root / "partial" / "bluff-ordinary").mkdir(parents=True, exist_ok=True)
    tasks = [(root, ident, first, i) for i in range(0, MATCHES, CHUNK)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = [row for block in pool.map(run_chunk, tasks) for row in block]
    return {
        "fixture": PROTOCOL,
        "opponent": "baseline",
        "seed": first,
        "matches": MATCHES,
        "rows": rows,
    }


def verify_ordinary(data, reservation):
    if (
        data["fixture"],
        data["opponent"],
        data["seed"],
        data["matches"],
        len(data["rows"]),
    ) != (PROTOCOL, "baseline", start_seed(reservation), MATCHES, MATCHES):
        raise ValueError("Incorrect ordinary opening schedule")
    gains = []
    for row in data["rows"]:
        if row is None:
            continue
        digest, probability, raised, quiet = row
        if (
            len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or not (
                0 < probability <= 1 / (1 + math.exp(3)) + 1e-6
                and type(raised) is int
                and raised in (0, 1)
                and type(quiet) is int
                and quiet in (0, 1)
            )
        ):
            raise ValueError(
                "Invalid paired opening observation, outcome, or probability cap"
            )
        gains.append(raised - quiet)
    if len(gains) < MIN_OFFERS:
        raise ValueError(f"Too few genuine openings: {len(gains)}/{MATCHES}")
    bounds = interval(gains)
    if bounds[0] <= 0:
        raise ValueError(
            f"Positive match-win benefit not established on {len(gains)} offered openings: {bounds}"
        )
    return {
        "offered_matches": len(gains),
        "scheduled_matches": MATCHES,
        "gain": float(np.mean(gains)),
        "ci95": bounds,
    }
