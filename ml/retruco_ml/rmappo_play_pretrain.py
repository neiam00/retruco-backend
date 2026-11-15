"""Pretrain the recurrent actor's game head on the live CPU's public-information choices."""

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from .bridge import Simulator
from .controlled_train import atomic_json
from .cooperation_population import cooperation_scripted
from .recurrent import Replay, file_hash
from .rmappo_model import actor_from_checkpoint
from .splits import TRAIN_MAX, TRAIN_MIN
from .team_play_probe import LIVE_CHOICES, team_action


@dataclass
class Example:
    context: torch.Tensor
    actions: torch.Tensor
    allowed: torch.Tensor
    teacher_probs: torch.Tensor
    family: str
    players: int
    sources: torch.Tensor
    destinations: torch.Tensor


def command_key(command):
    return tuple(sorted(command.items()))


def browser_action_indices(view, features, seat, legal, candidates):
    """Map the browser's sampled and adjusted commands back to the full rules list."""
    slots = {command_key(command): i for i, command in enumerate(legal)}
    sources = [slots[command_key(command)] for command in candidates]
    if not sources or len(set(sources)) != len(sources) or len(slots) != len(legal):
        raise ValueError(
            "teacher choices must be a nonempty, unique subset of legal moves"
        )
    destinations = sources
    if view["config"]["players"] == 4:
        destinations = [
            sources[team_action(view, features, seat, candidates, i, LIVE_CHOICES)[0]]
            for i in range(len(candidates))
        ]
    return torch.tensor(sources, dtype=torch.long), torch.tensor(
        destinations, dtype=torch.long
    )


def executed_distribution(logits, sources, destinations):
    """Probabilities of actions actually executed, including mapped card/declaration aliases."""
    weights = logits[sources].softmax(0)
    probabilities = torch.zeros_like(logits)
    return probabilities.index_add(0, destinations, weights)


def collect_examples(actor, simulator, count, seed):
    """Replay real training deals. A separate stateless query labels the full legal list."""
    rng = random.Random(seed)
    # Reserve a few genuine compulsory perico decisions from independent training deals.
    # Their frequency in this teaching set is deliberate and reported separately.
    reserve = min(12, count // 256)
    ordinary, perico_examples = [], []
    matches = 0
    with torch.no_grad():
        while len(ordinary) < count - reserve or len(perico_examples) < reserve:
            players = 2 if matches % 2 == 0 else 4
            learner = rng.randrange(players)
            partner = (learner + 2) % 4 if players == 4 else None
            roles = [True] * 4
            partner_style = None
            if players == 4 and matches % 4 == 3:
                roles[partner] = False
                partner_style = ("literal-v2", "trusting-v2", "cautious-v2")[
                    (matches // 4) % 3
                ]
            replayer = Replay(actor)
            hand, cursors = 0, [0] * 4
            reset = {
                "op": "reset",
                "id": 0,
                "training": True,
                "truthfulCPU": True,
                "seed": rng.randrange(TRAIN_MIN, TRAIN_MAX),
                "roles": roles,
                "config": {
                    "players": players,
                    "target": rng.choice((12, 24)),
                    "florRequired": bool(rng.randrange(2)),
                },
            }
            obs = simulator.call([reset])[0]
            while not obs["done"]:
                seat = obs["seat"]
                page = obs["training"]["memory"]
                if page["hand"] != hand:
                    hand, cursors = page["hand"], [0] * 4
                if seat == learner:
                    while True:
                        replayer.apply(page)
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
                if seat == partner and partner_style:
                    action = cooperation_scripted(obs, partner_style, rng)
                elif obs["kind"] == "signal":
                    if seat == learner:
                        context = actor.context(
                            torch.as_tensor(obs["features"]), replayer.memory
                        )
                        weights = actor.distribution(context, obs).probs.tolist()
                        action = rng.choices(range(len(weights)), weights=weights)[0]
                    else:
                        action = 0
                elif obs["view"]["phase"] == "cutting" or seat != learner:
                    action = obs.get("baseline", 0)
                else:
                    legal = obs["legal"]
                    flags = [s == learner for s in range(4)]
                    target = simulator.call(
                        [
                            {
                                "op": "encode",
                                "id": 0,
                                "kind": "game",
                                "seat": seat,
                                "view": obs["view"],
                                "context": {},
                                "playingSeats": flags,
                                "choicePolicy": LIVE_CHOICES,
                            }
                        ]
                    )[0]
                    if target.get("choicePolicy") != LIVE_CHOICES:
                        raise ValueError(
                            "simulator did not apply the requested teaching policy"
                        )
                    state = torch.as_tensor(obs["features"])
                    context = actor.context(state, replayer.memory)
                    actions = torch.as_tensor(obs["actions"])
                    logits = actor.game_logits(context.expand(len(legal), -1), actions)
                    sources, destinations = browser_action_indices(
                        obs["view"], obs["features"], seat, legal, target["legal"]
                    )
                    probabilities = executed_distribution(logits, sources, destinations)
                    action = rng.choices(
                        range(len(legal)), weights=probabilities.tolist()
                    )[0]
                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    mask[destinations] = True
                    types = {c["type"] for c in legal}
                    if "stack" in types or not mask.all() or rng.random() < 0.12:
                        family = (
                            "forced_pegado"
                            if "stack" in types and obs["view"]["mandatoryStack"]
                            else "voluntary_pegado"
                            if "stack" in types
                            else "ordinary"
                        )
                        if family == "forced_pegado" and any(
                            not h["used"] and state[24 + 5 * h["index"]] == 1
                            for h in obs["view"]["hand"]
                        ):
                            family = "forced_pegado_perico"
                        example = Example(
                            context,
                            actions,
                            mask,
                            probabilities,
                            family,
                            players,
                            sources,
                            destinations,
                        )
                        if (
                            family == "forced_pegado_perico"
                            and len(perico_examples) < reserve
                        ):
                            perico_examples.append(example)
                        elif len(ordinary) < count - reserve:
                            ordinary.append(example)
                        if len(ordinary) + len(perico_examples) == count:
                            break
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
            matches += 1
    return ordinary + perico_examples


def batch_objective(actor, examples, live_kl_weight):
    """Teach the raw head while preserving the parent's conditional live decisions."""
    lengths = [len(e.actions) for e in examples]
    contexts = torch.cat(
        [e.context.expand(n, -1) for e, n in zip(examples, lengths, strict=True)]
    )
    logits = actor.game_logits(
        contexts, torch.cat([e.actions for e in examples])
    ).split(lengths)
    losses, rejected, divergences = [], [], []
    for scores, e in zip(logits, examples, strict=True):
        logp = scores.log_softmax(0)
        positive = e.teacher_probs > 0
        target = e.teacher_probs[positive]
        live = executed_distribution(scores, e.sources, e.destinations)[positive]
        live_kl = (
            target * (target.log() - live.clamp_min(torch.finfo(live.dtype).tiny).log())
        ).sum()
        losses.append(-(target * logp[positive]).sum() + live_kl_weight * live_kl)
        divergences.append(live_kl)
        rejected.append(logp[~e.allowed].exp().sum())
    return torch.stack(losses).mean(), torch.stack(rejected), torch.stack(divergences)


@torch.no_grad()
def assess(actor, examples):
    result = {}
    groups = {
        "all": examples,
        "1v1": [e for e in examples if e.players == 2],
        "2v2": [e for e in examples if e.players == 4],
    }
    for family in ("voluntary_pegado", "forced_pegado_perico", "ordinary"):
        groups[family] = [e for e in examples if e.family == family]
    groups["forced_pegado"] = [
        e for e in examples if e.family.startswith("forced_pegado")
    ]
    for family, rows in groups.items():
        if not rows:
            result[family] = {"examples": 0}
            continue
        batches = [
            batch_objective(actor, rows[i : i + 32], 0) for i in range(0, len(rows), 32)
        ]
        mass = torch.cat([b[1] for b in batches])
        live_kl = torch.cat([b[2] for b in batches])
        result[family] = {
            "examples": len(rows),
            "mean_mass_outside_teacher_choices": float(mass.mean()),
            "mean_executed_policy_kl": float(live_kl.mean()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=2048)
    parser.add_argument("--validation-examples", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7211)
    parser.add_argument(
        "--live-kl-weight",
        type=float,
        default=8.0,
        help="weight on forward KL from the frozen parent after browser action selection",
    )
    args = parser.parse_args()
    if min(args.examples, args.validation_examples, args.epochs) < 1:
        parser.error("positive training and validation budgets are required")
    if not 0 <= args.live_kl_weight < float("inf"):
        parser.error("live KL weight must be finite and nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    parent_hash = file_hash(args.parent)
    saved = torch.load(args.parent, map_location="cpu", weights_only=True)
    teacher = actor_from_checkpoint(saved).eval().requires_grad_(False)
    if teacher.candidate_features:
        raise ValueError("calibrated base action encoding required")
    trainee = copy.deepcopy(teacher)
    for module in (trainee.action, trainee.score):
        module.requires_grad_(True)
    params = [p for p in trainee.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-4, eps=1e-5)
    with Simulator(args.simulator, "binary") as simulator:
        train = collect_examples(teacher, simulator, args.examples, args.seed)
        validation = collect_examples(
            teacher, simulator, args.validation_examples, args.seed + 1000003
        )
    before = assess(trainee, validation)
    order = list(range(len(train)))
    rng = random.Random(args.seed)
    updates = 0
    for _ in range(args.epochs):
        rng.shuffle(order)
        for start in range(0, len(order), 32):
            optimizer.zero_grad()
            loss, _, _ = batch_objective(
                trainee,
                [train[i] for i in order[start : start + 32]],
                args.live_kl_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            updates += 1
    after = assess(trainee, validation)
    meta = {
        "parent_sha256": parent_hash,
        "simulator_sha256": file_hash(args.simulator),
        "choice_policy": LIVE_CHOICES,
        "seed": args.seed,
        "validation_seed": args.seed + 1000003,
        "examples": len(train),
        "validation_examples": len(validation),
        "epochs": args.epochs,
        "supervised_updates": updates,
        "live_kl_weight": args.live_kl_weight,
        "training_forced_pegado_perico_examples": sum(
            e.family == "forced_pegado_perico" for e in train
        ),
        "trainable_modules": ["action", "score"],
        "validation_before": before,
        "validation_after": after,
    }
    # This is an initializer, never a resume of its parent's Adam or RNG state.
    payload = {
        k: saved[k]
        for k in (
            "kind",
            "configuration",
            "encoder",
            "memory_version",
            "critic",
            "league",
            "choices",
        )
        if k in saved
    }
    payload.update(actor=trainee.state_dict(), play_pretraining=meta, approved=False)
    candidate = args.output / "candidate.pt"
    with candidate.open("xb") as stream:
        torch.save(payload, stream)
    if file_hash(args.parent) != parent_hash:
        raise ValueError("source checkpoint changed during pretraining")
    atomic_json(
        args.output / "report.json",
        {**meta, "candidate_sha256": file_hash(candidate), "approved": False},
    )
    print(
        json.dumps({"candidate": str(candidate), "before": before, "after": after}),
        flush=True,
    )


if __name__ == "__main__":
    main()
