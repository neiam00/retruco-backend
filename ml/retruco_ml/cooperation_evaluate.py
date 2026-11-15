"""Versioned cooperation validation, including authority and signal ablations."""

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
import torch

from .bridge import Simulator
from .cooperation_population import COOPERATION_STYLES, cooperation_scripted
from .model import Policy
from .model_capacity import actor_from_checkpoint
from .population import scripted
from .recurrent import Replay
from .statistics import interval, pooled

PARTNERS = ("reference-cpu", "reference-human", "literal-legacy", *COOPERATION_STYLES)


def sample_legacy(model, obs):
    distribution, _ = model.distribution(obs)
    mask = obs.get("signalMask") or obs.get("truthMask")
    if mask:
        distribution = torch.distributions.Categorical(
            logits=distribution.logits.masked_fill(~torch.tensor(mask), -torch.inf)
        )
    return int(distribution.sample())


def evaluate(
    actor,
    reference,
    simulator,
    pairs=64,
    seed=188000,
    partner="literal-v2",
    signals="normal",
    players=0,
    observer=None,
    partner_policy=None,
):
    if (
        pairs < 2
        or not 100000 <= seed
        or seed + pairs >= 200000
        or partner not in PARTNERS
        or signals not in ("normal", "no-send", "no-receive")
    ):
        raise ValueError("invalid cooperation validation settings")
    configurations = []
    diagnostics = {
        "signal_decisions": 0,
        "truthful_claims": 0,
        "claim_opportunities": 0,
        "pending_decisions": 0,
        "accepts": 0,
        "declines": 0,
    }
    for count in (players,) if players else (2, 4):
        for target in (12, 24):
            for flor in (False, True):
                scores = []
                for deal in range(pairs):
                    wins = []
                    for learner in range(count):
                        torch.manual_seed(seed + deal * 17 + learner)
                        rng = random.Random(seed + deal * 17 + learner)
                        roles = [True] * 4
                        if count == 4 and partner != "reference-cpu":
                            roles[(learner + 2) % 4] = False
                        request = {
                            "op": "reset",
                            "id": 0,
                            "training": True,
                            "truthfulCPU": True,
                            "seed": seed + deal,
                            "config": {
                                "players": count,
                                "target": target,
                                "florRequired": flor,
                            },
                            "roles": roles,
                        }
                        if signals == "no-receive" and count == 4:
                            request["ignorePartnerSignalsFor"] = learner
                        obs = simulator.call([request])[0]
                        replay = Replay(actor)
                        cursors = [0] * 4
                        hand = 0
                        for _ in range(10000):
                            if obs["done"]:
                                wins.append(float(obs["winner"] == learner % 2))
                                break
                            page = obs["training"]["memory"]
                            seat = obs["seat"]
                            if hand != page["hand"]:
                                hand = page["hand"]
                                cursors = [0] * 4
                            if seat == learner:
                                while True:
                                    replay.apply(page)
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
                            else:
                                cursors[seat] = page["total"]
                            if (
                                obs["kind"] == "game"
                                and obs["legal"][0]["type"] == "cut"
                            ):
                                action = 0
                            elif seat == learner:
                                with torch.no_grad():
                                    features = torch.from_numpy(
                                        np.asarray(obs["features"], np.float32)
                                    )
                                    context = actor.context(features, replay.memory)
                                    action = int(
                                        actor.distribution(context, obs).sample()
                                    )
                                if observer is not None:
                                    observer(obs, action)
                                if obs["kind"] == "signal":
                                    if signals == "no-send":
                                        action = 0
                                    diagnostics["signal_decisions"] += 1
                                    diagnostics["truthful_claims"] += int(action != 0)
                                    diagnostics["claim_opportunities"] += int(
                                        any(obs["signalMask"][1:])
                                    )
                                    if not obs["signalMask"][action]:
                                        raise RuntimeError("false learner claim")
                                elif obs["view"].get("pending"):
                                    diagnostics["pending_decisions"] += 1
                                    kind = obs["legal"][action].get("kind")
                                    diagnostics["accepts"] += int(kind == "accept")
                                    diagnostics["declines"] += int(kind == "decline")
                            elif count == 4 and seat == (learner + 2) % 4:
                                if partner.startswith("reference"):
                                    with torch.no_grad():
                                        action = sample_legacy(reference, obs)
                                else:
                                    action = (
                                        partner_policy(obs, rng)
                                        if partner_policy is not None
                                        else cooperation_scripted(
                                            obs,
                                            "literal"
                                            if partner == "literal-legacy"
                                            else partner,
                                            rng,
                                        )
                                    )
                            else:
                                action = scripted(obs, "baseline", rng)
                            if obs.get("signalMask") and not obs["signalMask"][action]:
                                action = 0
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
                        else:
                            raise RuntimeError(
                                "cooperation evaluation match exceeded bound"
                            )
                    scores.append(sum(wins) / len(wins))
                configurations.append(
                    {
                        "players": count,
                        "target": target,
                        "flor": flor,
                        "pairs": pairs,
                        "pair_scores": scores,
                        "win_rate": sum(scores) / pairs,
                        "cluster_ci95": interval(scores),
                    }
                )
    return {
        "protocol": "cooperation-evaluation-v1",
        "seed": seed,
        "partner": partner,
        "signals": signals,
        "truthful_cpu": True,
        "configurations": configurations,
        "pooled": pooled(configurations),
        "behavior": diagnostics,
        "approved": False,
        "scope": "synthetic partner validation; not human compatibility or final release evidence",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--partner", choices=PARTNERS, required=True)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=188000)
    parser.add_argument(
        "--signals", choices=("normal", "no-send", "no-receive"), default="normal"
    )
    parser.add_argument("--players", type=int, choices=(0, 2, 4), default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    payload = args.checkpoint.read_bytes()
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if (
        saved.get("encoder") != 2
        or saved.get("memory_version") != 1
        or saved["configuration"]["arm"] != "D"
    ):
        raise ValueError("requires compatible Model D actor")
    actor = actor_from_checkpoint(saved)
    actor.eval()
    reference_bytes = args.reference.read_bytes()
    parent = torch.load(
        io.BytesIO(reference_bytes), map_location="cpu", weights_only=True
    )
    reference = Policy()
    reference.load_state_dict(parent["model"])
    reference.eval()
    with Simulator(args.simulator, "binary") as simulator:
        report = evaluate(
            actor,
            reference,
            simulator,
            args.pairs,
            args.seed,
            args.partner,
            args.signals,
            args.players,
        )
    report.update(
        checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
        reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
    )
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"pooled": report["pooled"], "behavior": report["behavior"]}))


if __name__ == "__main__":
    main()
