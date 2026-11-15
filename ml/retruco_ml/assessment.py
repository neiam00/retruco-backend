"""Reference/quantized, opponent, partner and ablation comparisons on held-out deals."""

import argparse
import hashlib
import io
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch

from .artifacts import digest, read_quantized
from .bridge import Simulator
from .evaluation_styles import (
    EVALUATION_STYLES,
    MIXTURE_STYLES,
    POPULATION_SCHEDULE,
    evaluation_scripted,
    opponent_for_pair,
)
from .export import serialize
from .model import ENCODER_VERSION, Policy, batch_distribution
from .population import SIGNALS, scripted, truthful_signals
from .rollout import SimulatorPool
from .splits import RELEASE_START
from .statistics import interval, paired_loss, pooled

ASSESSMENT_VERSION = 2
SEAT_SCHEDULE = "swap-teams-alternate-partner-seats-v2"


def load_checkpoint(path):
    data = path.read_bytes()
    checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if checkpoint.get("encoder") != ENCODER_VERSION:
        raise ValueError("incompatible checkpoint encoder")
    if checkpoint.get("training_kind", "ppo") not in ("ppo", "qat"):
        raise ValueError("unsupported checkpoint training kind")
    model = Policy()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.checkpoint_sha256 = hashlib.sha256(data).hexdigest()
    model.training_kind = checkpoint.get("training_kind", "ppo")
    model.qat_origin = {
        key: checkpoint.get("qat", {}).get(key)
        for key in (
            "teacher_checkpoint_sha256",
            "teacher_training_source_commit",
            "teacher_training_snapshot_sha256",
        )
    }
    return model


def ablate(observation, mode):
    if mode == "none":
        return observation
    result = dict(observation)
    features = list(result["features"])
    if mode in ("adaptation", "both"):
        features[352:384] = [0.0] * 32
        features[512:716] = [0.0] * 204
    if mode in ("signals", "both"):
        features[416:476] = [0.0] * 60
        features[672:717] = [0.0] * 45
    result["features"] = features
    return result


def choose(model, observation, rng, ablation="none"):
    if observation["kind"] == "game" and observation["legal"][0]["type"] == "cut":
        return 0
    if observation["kind"] == "signal" and ablation in ("signals", "both"):
        return 0
    with torch.no_grad():
        distribution, _ = model.distribution(ablate(observation, ablation))
        probabilities = distribution.probs.tolist()
    return sample(probabilities, rng)


def sample(probabilities, rng):
    threshold = rng.random()
    for index, probability in enumerate(probabilities):
        threshold -= probability
        if threshold < 0:
            return index
    return len(probabilities) - 1


def assess(
    model,
    pairs,
    players,
    target,
    flor,
    seed,
    opponent="baseline",
    opponent_model=None,
    partner="learned",
    partner_model=None,
    ablation="none",
):
    pair_scores = []
    counts = Counter()
    with Simulator() as simulator:
        for pair in range(pairs):
            style = opponent_for_pair(opponent, seed, pair)
            wins = 0
            for team in (0, 1):
                learner = team + (2 * (pair % 2) if players == 4 else 0)
                partner_seat = (learner + 2) % 4
                roles = [True] * 4
                if players == 4 and partner != "learned":
                    roles[partner_seat] = False
                streams = [
                    random.Random((seed + pair) * 11 + seat) for seat in range(players)
                ]
                observation = simulator.call(
                    [
                        {
                            "op": "reset",
                            "id": 0,
                            "seed": seed + pair,
                            "config": {
                                "players": players,
                                "target": target,
                                "florRequired": flor,
                            },
                            "roles": roles,
                        }
                    ]
                )[0]
                for _ in range(10000):
                    if observation["done"]:
                        break
                    seat = observation["seat"]
                    rng = streams[seat]
                    if seat % 2 != team:
                        action = (
                            choose(opponent_model, observation, rng)
                            if opponent_model is not None
                            else evaluation_scripted(observation, style, rng)
                        )
                    elif players == 4 and seat == partner_seat and partner != "learned":
                        action = scripted(observation, partner, rng)
                    else:
                        acting_model = (
                            partner_model
                            if players == 4
                            and seat == partner_seat
                            and partner_model is not None
                            else model
                        )
                        action = choose(acting_model, observation, rng, ablation)
                        if observation["kind"] == "signal":
                            counts["signal:" + SIGNALS[action]] += 1
                            if action and action not in truthful_signals(
                                observation["view"]
                            ):
                                counts["false_signal_claim"] += 1
                        else:
                            command = observation["legal"][action]
                            counts[command["type"] + ":" + command.get("kind", "")] += 1
                    observation = simulator.call(
                        [{"op": "step", "id": 0, "action": action}]
                    )[0]
                if not observation["done"]:
                    raise RuntimeError("assessment match stalled")
                wins += int(observation["winner"] == team)
            pair_scores.append(wins / 2)
    return {
        "players": players,
        "target": target,
        "flor": flor,
        "pairs": pairs,
        "pair_scores": pair_scores,
        "win_rate": sum(pair_scores) / pairs,
        "cluster_ci95": interval(pair_scores),
        "actions": dict(counts),
    }


def assess_parallel(
    model,
    pairs,
    players,
    target,
    flor,
    seed,
    opponent="baseline",
    opponent_model=None,
    partner="learned",
    partner_model=None,
    ablation="none",
    workers=8,
    environments=64,
):
    """Batch inference and native IPC; keep independent RNG streams per deal/seat.

    Complete paired matches retain equal weight regardless of their duration.
    No rollout trajectories or expanded historical action matrices are retained.
    """
    if pairs < 2 or not 1 <= environments <= 64:
        raise ValueError("invalid assessment size")
    tasks = [(pair, team) for pair in range(pairs) for team in (0, 1)]
    wins = [0] * pairs
    counts = Counter()
    decisions = 0
    started = time.monotonic()
    with SimulatorPool(workers) as pool:
        for start in range(0, len(tasks), environments):
            episodes = {}
            requests = []
            for identity, (pair, team) in enumerate(
                tasks[start : start + environments]
            ):
                roles = [True] * 4
                partner_seat = (team + 2 * (pair % 2) + 2) % 4
                if players == 4 and partner != "learned":
                    roles[partner_seat] = False
                episodes[identity] = (
                    pair,
                    team,
                    [random.Random((seed + pair) * 11 + s) for s in range(players)],
                    partner_seat,
                    opponent_for_pair(opponent, seed, pair),
                )
                requests.append(
                    {
                        "op": "reset",
                        "id": identity,
                        "seed": seed + pair,
                        "config": {
                            "players": players,
                            "target": target,
                            "florRequired": flor,
                        },
                        "roles": roles,
                    }
                )
            observations = dict(enumerate(pool.call(requests)))
            for _ in range(10000):
                if not observations:
                    break
                chosen = {}
                groups = {}
                recorded = set()
                for identity, observation in observations.items():
                    if observation["version"] != ENCODER_VERSION:
                        raise ValueError("simulator encoder mismatch")
                    _, team, streams, partner_seat, style = episodes[identity]
                    seat = observation["seat"]
                    rng = streams[seat]
                    mode = "none"
                    if seat % 2 != team:
                        actor = opponent_model
                        if actor is None:
                            chosen[identity] = evaluation_scripted(
                                observation, style, rng
                            )
                            continue
                    elif players == 4 and seat == partner_seat and partner != "learned":
                        chosen[identity] = scripted(observation, partner, rng)
                        continue
                    else:
                        actor = (
                            partner_model
                            if players == 4
                            and seat == partner_seat
                            and partner_model is not None
                            else model
                        )
                        mode = ablation
                        recorded.add(identity)
                    if (
                        observation["kind"] == "game"
                        and observation["legal"][0]["type"] == "cut"
                    ):
                        chosen[identity] = 0
                    elif observation["kind"] == "signal" and mode in (
                        "signals",
                        "both",
                    ):
                        chosen[identity] = 0
                    else:
                        groups.setdefault((id(actor), mode), [actor, []])[1].append(
                            identity
                        )
                with torch.no_grad():
                    for (_, mode), (actor, identities) in groups.items():
                        batch = [ablate(observations[i], mode) for i in identities]
                        distribution, _ = batch_distribution(actor, batch)
                        probabilities = distribution.probs.tolist()
                        for identity, values in zip(
                            identities, probabilities, strict=True
                        ):
                            observation = observations[identity]
                            count = (
                                6
                                if observation["kind"] == "signal"
                                else len(observation["legal"])
                            )
                            chosen[identity] = sample(
                                values[:count],
                                episodes[identity][2][observation["seat"]],
                            )
                for identity in recorded:
                    observation = observations[identity]
                    action = chosen[identity]
                    if observation["kind"] == "signal":
                        counts["signal:" + SIGNALS[action]] += 1
                        if action and action not in truthful_signals(
                            observation["view"]
                        ):
                            counts["false_signal_claim"] += 1
                    else:
                        command = observation["legal"][action]
                        counts[command["type"] + ":" + command.get("kind", "")] += 1
                requests = [
                    {"op": "step", "id": i, "action": chosen[i]} for i in observations
                ]
                decisions += len(requests)
                for request, observation in zip(
                    requests, pool.call(requests), strict=True
                ):
                    identity = request["id"]
                    if observation["done"]:
                        pair, team, _, _, _ = episodes[identity]
                        wins[pair] += int(observation["winner"] == team)
                        del observations[identity]
                    else:
                        observations[identity] = observation
            if observations:
                raise RuntimeError("parallel assessment match stalled")
    scores = [win / 2 for win in wins]
    elapsed = time.monotonic() - started
    return {
        "players": players,
        "target": target,
        "flor": flor,
        "pairs": pairs,
        "pair_scores": scores,
        "win_rate": sum(scores) / pairs,
        "cluster_ci95": interval(scores),
        "actions": dict(counts),
        "elapsed_seconds": elapsed,
        "decisions_per_second": decisions / elapsed,
        "matches_per_hour": 2 * pairs * 3600 / elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--quantized", type=Path)
    parser.add_argument("--opponent", choices=EVALUATION_STYLES, default="baseline")
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument(
        "--partner",
        choices=("learned", "literal", "cautious", "call-heavy", "silent", "ignore"),
        default="learned",
    )
    parser.add_argument("--partner-checkpoint", type=Path)
    parser.add_argument(
        "--ablation", choices=("none", "adaptation", "signals", "both"), default="none"
    )
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=RELEASE_START)
    parser.add_argument("--players", type=int, choices=(0, 2, 4), default=0)
    parser.add_argument("--compare-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--environments", type=int, default=64)
    args = parser.parse_args()
    if args.pairs < 2:
        parser.error("at least two pairs required")
    if not 1 <= args.workers <= 64 or not 1 <= args.environments <= 64:
        parser.error("workers and environments must be in [1,64]")
    torch.set_num_threads(2)
    model = load_checkpoint(args.checkpoint)
    if args.quantized:
        data = args.quantized.read_bytes()
        expected, _ = serialize(model)
        if data != expected:
            raise ValueError("quantized binary is not the export of this checkpoint")
        model = read_quantized(data, model)
    opponent = (
        load_checkpoint(args.opponent_checkpoint) if args.opponent_checkpoint else None
    )
    partner = (
        load_checkpoint(args.partner_checkpoint) if args.partner_checkpoint else None
    )
    report = {
        "approved": False,
        "assessment_version": ASSESSMENT_VERSION,
        "seat_schedule": SEAT_SCHEDULE,
        "checkpoint_sha256": model.checkpoint_sha256,
        "seed": args.seed,
        "quantized_sha256": digest(args.quantized) if args.quantized else None,
        "opponent": opponent.checkpoint_sha256
        if opponent is not None
        else args.opponent,
        "opponent_schedule": POPULATION_SCHEDULE
        if args.opponent == "population" and opponent is None
        else "fixed-opponent-v1",
        "opponent_styles": list(MIXTURE_STYLES)
        if args.opponent == "population" and opponent is None
        else None,
        "partner": partner.checkpoint_sha256 if partner is not None else args.partner,
        "ablation": args.ablation,
        "workers": args.workers,
        "environments": args.environments,
        "configurations": [],
    }
    for players in (args.players,) if args.players else (2, 4):
        for target in (12, 24):
            for flor in (False, True):
                result = assess_parallel(
                    model,
                    args.pairs,
                    players,
                    target,
                    flor,
                    args.seed,
                    args.opponent,
                    opponent,
                    args.partner,
                    partner,
                    args.ablation,
                    args.workers,
                    args.environments,
                )
                report["configurations"].append(result)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in result.items()
                            if k not in ("pair_scores", "actions")
                        }
                    ),
                    flush=True,
                )
    report["pooled"] = pooled(report["configurations"])
    report["modes"] = {
        str(players): pooled(
            [c for c in report["configurations"] if c["players"] == players]
        )
        for players in {c["players"] for c in report["configurations"]}
    }
    if args.compare_reference:
        reference = json.loads(args.compare_reference.read_text())
        for key in (
            "checkpoint_sha256",
            "opponent",
            "partner",
            "assessment_version",
            "seat_schedule",
        ):
            if reference[key] != report[key]:
                raise ValueError("reference comparison mismatch: " + key)
        for key in ("opponent_schedule", "opponent_styles"):
            if reference.get(key) != report.get(key):
                raise ValueError("reference opponent schedule mismatch")
        if reference["ablation"] != report["ablation"]:
            if reference["quantized_sha256"] != report["quantized_sha256"]:
                raise ValueError(
                    "ablation comparison must use the same weight precision"
                )
            report["ablation_loss"] = paired_loss(reference, report)
        else:
            report["quantization_loss"] = paired_loss(reference, report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
