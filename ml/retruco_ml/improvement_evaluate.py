"""Paired matches against native, learned and held-out reactive opposition."""

import argparse
import json
import random
from pathlib import Path

import torch

from .controlled_train import atomic_json
from .cooperation_diagnostics import DecisionContexts, redundant_signal
from .improvement_behavior import ImprovementBehavior
from .improvement_budget import DeadlineReached, check_deadline
from .improvement_population import improvement_scripted, reactive_style
from .model import Policy
from .model_capacity import actor_from_checkpoint
from .recurrent import Actor, Replay, file_hash
from .rollout import Simulator
from .statistics import interval, pooled
from .team_play_probe import CHOICE_PROFILES, PROBES, team_action

FAMILIES = ("native", "parent", "independent", "reactive", "resistant")
PARTNERS = (
    "reference-cpu",
    "reference-human",
    "literal-v2",
    "trusting-v2",
    "cautious-v2",
    "silent-v2",
    "erratic-v2",
    "independent-partner",
    "literal-v3",
    "trusting-v3",
    "cautious-v3",
)
SIGNAL_MODES = (
    "normal",
    "no-send",
    "no-receive",
    "no-partner-delivery",
    "no-opponent-delivery",
)


def sample(distribution, rng):
    probabilities = distribution.probs.detach().double()
    cumulative = probabilities.cumsum(-1)
    uniform = rng.getrandbits(32) / 4294967296 * float(cumulative[-1])
    choices = (cumulative > uniform).nonzero()
    if not len(choices):
        raise ValueError("invalid policy sampling distribution")
    return int(choices[0])


def highest_compatible_play(observation, action):
    chosen = observation["legal"][action]
    if chosen["type"] != "play":
        return action
    identity = {k: v for k, v in chosen.items() if k != "card"}
    candidates = [
        i
        for i, c in enumerate(observation["legal"])
        if {k: v for k, v in c.items() if k != "card"} == identity
    ]
    ranks = {
        c["index"]: observation["features"][24 + 5 * c["index"]]
        for c in observation["view"]["hand"]
        if not c["used"]
    }
    best = max(ranks[observation["legal"][i].get("card", 0)] for i in candidates)
    if ranks[chosen.get("card", 0)] == best:
        return action
    return next(
        i for i in candidates if ranks[observation["legal"][i].get("card", 0)] == best
    )


def frozen_models(manifest, root):
    result = {}
    entries = {
        **{f"frozen:{seed}": entry for seed, entry in manifest["parents"].items()},
        "reference": manifest["reference"],
    }
    for name, entry in entries.items():
        path = (root / entry["file"]).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or file_hash(path) != entry["sha256"]
        ):
            raise ValueError("frozen evaluation input changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        policy = Policy() if name == "reference" else Actor(True)
        policy.load_state_dict(saved["model"])
        policy.eval()
        result[name] = policy
    return result


def evaluate(
    actor,
    models,
    simulator,
    pairs=16,
    seed=194000,
    partner="literal-v2",
    opponent="native",
    signals="normal",
    players=4,
    card_policy="policy",
):
    if (
        pairs < 2
        or signals not in SIGNAL_MODES
        or card_policy not in ("policy", "highest", *PROBES)
        or opponent not in FAMILIES
        or partner not in PARTNERS
        or not 100000 <= seed < seed + pairs < 200000
        or players not in (0, 2, 4)
    ):
        raise ValueError("unsupported development evaluation")
    configurations, behavior, response_contexts, response_families = [], {}, {}, {}
    for count in (2, 4) if not players else (players,):
        for target in (12, 24):
            for flor in (False, True):
                scores, point_scores, config_counts, matches = [], [], {}, 0
                for deal in range(pairs):
                    wins, net_per_hand = [], []
                    for learner in range(count):
                        profile_rng = random.Random(seed + deal * 17 + learner)
                        script_rng = {
                            seat: random.Random(
                                seed + deal * 101 + learner * 13 + seat * 1237
                            )
                            for seat in range(count)
                        }
                        action_rng = {
                            seat: random.Random(
                                seed + deal * 1009 + learner * 31 + seat * 2437
                            )
                            for seat in range(count)
                        }
                        opposing = {
                            "native": "script:baseline",
                            "parent": "frozen:101",
                            "independent": "frozen:102",
                            "reactive": "script:"
                            + reactive_style(profile_rng, heldout=True),
                            "resistant": "script:reactive-resistant-v1",
                        }[opponent]
                        controllers = [opposing] * count
                        roles = [True] * 4
                        if opponent in ("reactive", "resistant"):
                            for seat in range(count):
                                if seat % 2 != learner % 2:
                                    roles[seat] = False
                        controllers[learner] = "learner"
                        if count == 4:
                            friend = (learner + 2) % 4
                            controllers[friend] = (
                                "reference"
                                if partner.startswith("reference")
                                else "frozen:103"
                                if partner == "independent-partner"
                                else "script:" + partner
                            )
                            roles[friend] = partner in (
                                "reference-cpu",
                                "independent-partner",
                            )
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
                        if card_policy in CHOICE_PROFILES:
                            request["playingSeats"] = [
                                seat == learner for seat in range(4)
                            ]
                            request["choicePolicy"] = card_policy
                        if signals == "no-receive" and count == 4:
                            request["ignorePartnerSignalsFor"] = learner
                        if signals == "no-partner-delivery" and count == 4:
                            request["suppressSignalDelivery"] = [
                                [learner, (learner + 2) % 4]
                            ]
                        if signals == "no-opponent-delivery":
                            request["suppressSignalDelivery"] = [
                                [learner, seat]
                                for seat in range(count)
                                if seat % 2 != learner % 2
                            ]
                        local_models = {**models, "learner": actor}
                        replays = {
                            seat: Replay(local_models[name])
                            for seat, name in enumerate(controllers)
                            if name in local_models
                            and isinstance(local_models[name], Actor)
                        }
                        observation = simulator.call([request])[0]
                        cursors, hand = [0] * 4, 0
                        economy, contexts = (
                            ImprovementBehavior(learner),
                            DecisionContexts(),
                        )
                        for _ in range(10000):
                            check_deadline()
                            if observation["done"]:
                                wins.append(float(observation["winner"] == learner % 2))
                                break
                            seat = observation["seat"]
                            controller = controllers[seat]
                            if (
                                card_policy in CHOICE_PROFILES
                                and seat == learner
                                and observation["kind"] == "game"
                                and observation.get("choicePolicy") != card_policy
                            ):
                                raise ValueError(
                                    "simulator did not apply the requested CPU choices"
                                )
                            page = observation["training"]["memory"]
                            if hand != page["hand"]:
                                hand, cursors = page["hand"], [0] * 4
                            if seat in replays:
                                while True:
                                    replays[seat].apply(page)
                                    cursors[seat] = page["end"]
                                    if page["end"] == page["total"]:
                                        break
                                    observation = simulator.call(
                                        [
                                            {
                                                "op": "observe",
                                                "id": 0,
                                                "memoryHand": hand,
                                                "memoryCursors": cursors,
                                            }
                                        ]
                                    )[0]
                                    page = observation["training"]["memory"]
                            else:
                                cursors[seat] = page["total"]
                            if (
                                observation["kind"] == "game"
                                and observation["legal"][0]["type"] == "cut"
                            ):
                                action = 0
                            elif controller.startswith("script:"):
                                action = improvement_scripted(
                                    observation,
                                    controller.split(":", 1)[1],
                                    script_rng[seat],
                                )
                            else:
                                policy = local_models[controller]
                                with torch.no_grad():
                                    if isinstance(policy, Actor):
                                        feature = torch.tensor(
                                            observation["features"], dtype=torch.float32
                                        )
                                        distribution = policy.distribution(
                                            policy.context(
                                                feature, replays[seat].memory
                                            ),
                                            observation,
                                        )
                                    else:
                                        distribution, _ = policy.distribution(
                                            observation
                                        )
                                    mask = observation.get(
                                        "signalMask"
                                    ) or observation.get("truthMask")
                                    if mask:
                                        distribution = torch.distributions.Categorical(
                                            logits=distribution.logits.masked_fill(
                                                ~torch.tensor(mask), -torch.inf
                                            )
                                        )
                                    action = sample(distribution, action_rng[seat])
                            if seat == learner:
                                if (
                                    card_policy == "highest"
                                    and observation["kind"] == "game"
                                ):
                                    action = highest_compatible_play(
                                        observation, action
                                    )
                                if (
                                    card_policy in PROBES
                                    and observation["kind"] == "game"
                                ):
                                    action, card_changed, flor_changed = team_action(
                                        observation["view"],
                                        observation["features"],
                                        seat,
                                        observation["legal"],
                                        action,
                                        card_policy,
                                    )
                                    if card_changed:
                                        behavior["closing_card_changes"] = (
                                            behavior.get("closing_card_changes", 0) + 1
                                        )
                                    if flor_changed:
                                        behavior["flor_kept_on_same_move"] = (
                                            behavior.get("flor_kept_on_same_move", 0)
                                            + 1
                                        )
                                if (
                                    signals == "no-send"
                                    and observation["kind"] == "signal"
                                ):
                                    action = 0
                                visible = observation
                                if signals == "no-receive" and count == 4:
                                    visible = {
                                        **observation,
                                        "signals": [
                                            s
                                            for s in observation.get("signals", [])
                                            if s["seat"] != (learner + 2) % 4
                                        ],
                                    }
                                economy.observe(visible, action)
                                contexts.observe(visible, action)
                                if observation["kind"] == "signal":
                                    behavior["signal_decisions"] = (
                                        behavior.get("signal_decisions", 0) + 1
                                    )
                                    behavior["truthful_claims"] = behavior.get(
                                        "truthful_claims", 0
                                    ) + int(action != 0)
                                    behavior["claim_opportunities"] = behavior.get(
                                        "claim_opportunities", 0
                                    ) + int(any(observation["signalMask"][1:]))
                                    behavior["redundant_claims"] = behavior.get(
                                        "redundant_claims", 0
                                    ) + int(redundant_signal(observation, action))
                            if (
                                observation.get("signalMask")
                                and not observation["signalMask"][action]
                            ):
                                raise ValueError(
                                    "false CPU signal or invalid masked policy selection"
                                )
                            following = simulator.call(
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
                            economy.transition(observation, action, following)
                            observation = following
                        else:
                            raise RuntimeError(
                                "improvement evaluation exceeded complete-match bound"
                            )
                        matches += 1
                        net_per_hand.append(
                            economy.counts["net_points"]
                            / max(1, economy.counts["hands_completed"])
                        )
                        for key, value in economy.counts.items():
                            behavior[key] = behavior.get(key, 0) + value
                            config_counts[key] = config_counts.get(key, 0) + value
                        for name, counts in contexts.counts.items():
                            bucket = response_contexts.setdefault(
                                name, dict.fromkeys(counts, 0)
                            )
                            for key, value in counts.items():
                                bucket[key] += value
                        for name, counts in contexts.by_family.items():
                            bucket = response_families.setdefault(
                                name, dict.fromkeys(counts, 0)
                            )
                            for key, value in counts.items():
                                bucket[key] += value
                    scores.append(sum(wins) / len(wins))
                    point_scores.append(sum(net_per_hand) / len(net_per_hand))
                configurations.append(
                    {
                        "players": count,
                        "target": target,
                        "flor": flor,
                        "pairs": pairs,
                        "pair_scores": scores,
                        "win_rate": sum(scores) / pairs,
                        "cluster_ci95": interval(scores),
                        "pair_net_points_per_hand": point_scores,
                        "net_points_per_hand": sum(point_scores) / pairs,
                        "matches": matches,
                        "behavior": config_counts,
                    }
                )
    return {
        "protocol": "improvement-evaluation-v1",
        "seed": seed,
        "pairs": pairs,
        "partner": partner,
        "opponent": opponent,
        "signals": signals,
        "card_policy": card_policy,
        "configurations": configurations,
        "pooled": pooled(configurations),
        "behavior": behavior,
        "decision_contexts": response_contexts,
        "response_families": response_families,
        "response_family_key": "players/canto/value/terminal/visible_partner_claim",
        "approved": False,
        "scope": "development study; behavior counters are descriptive, not causal bluff evidence",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partner", choices=PARTNERS, default="literal-v2")
    parser.add_argument("--opponent", choices=FAMILIES, default="native")
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=194000)
    parser.add_argument("--players", type=int, choices=(0, 2, 4), default=4)
    parser.add_argument("--signals", choices=SIGNAL_MODES, default="normal")
    parser.add_argument(
        "--card-policy", choices=("policy", "highest", *PROBES), default="policy"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("encoder") != 2 or saved.get("memory_version") != 1:
        raise ValueError("incompatible recurrent observation version")
    actor = actor_from_checkpoint(saved)
    actor.eval()
    manifest = json.loads(args.inputs.read_text())
    models = frozen_models(manifest, args.inputs.resolve().parent)
    with Simulator(args.simulator, "binary") as simulator:
        report = evaluate(
            actor,
            models,
            simulator,
            args.pairs,
            args.seed,
            args.partner,
            args.opponent,
            args.signals,
            args.players,
            args.card_policy,
        )
    report.update(
        checkpoint_sha256=file_hash(args.checkpoint),
        inputs_sha256=file_hash(args.inputs),
    )
    atomic_json(args.output, report)
    receipt = args.output.with_suffix(".sha256")
    temporary = receipt.with_suffix(".sha256.tmp")
    temporary.write_text(file_hash(args.output) + "\n")
    temporary.replace(receipt)
    print(
        json.dumps(
            {"win_rate": report["pooled"]["win_rate"], "behavior": report["behavior"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except DeadlineReached:
        raise SystemExit(75)
