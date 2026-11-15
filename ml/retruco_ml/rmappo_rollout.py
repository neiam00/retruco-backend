"""Joint-seat RMAPPO collection; independent histories and team reward."""

from dataclasses import dataclass

import numpy as np
import torch

from .cooperation_diagnostics import redundant_signal
from .cooperation_population import cooperation_scripted
from .learning_choices import meaningful
from .mappo_rollout import Decision, credit
from .model import batch_distribution
from .recurrent import Actor, Replay
from .recurrent_batch import candidate_distribution
from .rmappo_balance import stratum_for_episode
from .splits import TRAIN_MAX, TRAIN_MIN
from .team_play_probe import CHOICE_PROFILES, LIVE_CHOICES


@dataclass
class JointDecision(Decision):
    critic_events: list = None
    critic_end: int = 0
    match: int = 0
    learner_seat: int = 0
    balance_stratum: object = None


def team_credit(trajectories, winner, lam=0.99):
    records = []
    for seat, trajectory in trajectories.items():
        chosen = [r for r in trajectory if meaningful(r)]
        credit(chosen, 1.0 if seat % 2 == winner else -1.0, lam)
        records.extend(chosen)
    return records


def advance_group(actor, replays, pages):
    events = [r.prepare(p) for r, p in zip(replays, pages, strict=True)]
    lengths = [len(rows) for rows in events]
    maximum = max(lengths, default=0)
    if actor.recurrent and maximum:
        features = np.zeros((len(events), maximum, 768), np.float32)
        for i, rows in enumerate(events):
            features[i, : len(rows)] = rows
        with torch.no_grad():
            encoded = actor.state(torch.from_numpy(features).reshape(-1, 768)).reshape(
                len(events), maximum, 64
            )
            h = torch.stack([r.memory[0] for r in replays])
            c = torch.stack([r.memory[1] for r in replays])
            for step in range(maximum):
                nh, nc = actor.memory(encoded[:, step], (h, c))
                valid = torch.tensor([step < n for n in lengths])[:, None]
                h, c = torch.where(valid, nh, h), torch.where(valid, nc, c)
            for row, replay in enumerate(replays):
                replay.memory = (h[row].clone(), c[row].clone())
    for replay, rows in zip(replays, events, strict=True):
        replay.events.extend(event.copy() for event in rows)
    return sum(lengths)


def collect(
    actor,
    critic,
    reference,
    pool,
    population,
    rng,
    minimum,
    players=0,
    environments=32,
    monitor=None,
    frozen=None,
    gae_lambda=0.99,
    script_fn=cooperation_scripted,
    headsup_fraction=0.2,
    choice_policy=LIVE_CHOICES,
):
    if choice_policy is not None and choice_policy not in CHOICE_PROFILES:
        raise ValueError("unknown learner choice policy")
    if not 0 <= headsup_fraction <= 1:
        raise ValueError("invalid heads-up fraction")
    models = population.models(reference)
    models.update(frozen or {})
    models["learner"] = actor
    records = []
    matches = events = choices = 0
    diagnostics = {
        "signal_decisions": 0,
        "truthful_claims": 0,
        "claim_opportunities": 0,
        "accepts": 0,
        "declines": 0,
    }
    while choices < minimum:
        episodes = {}
        requests = []
        replays = {}
        match_base = matches
        for identity in range(environments):
            count = players or (2 if rng.random() < headsup_fraction else 4)
            episode = population.episode(count, rng)
            episodes[identity] = episode
            opponents = {
                controller
                for seat, controller in enumerate(episode.controllers)
                if seat % 2 != episode.learner % 2
            }
            family = (
                "native"
                if opponents == {"script:baseline"}
                else "scripted"
                if all(c.startswith("script:") for c in opponents)
                else "learned"
            )
            for key in (f"matches_{count}p", f"opponent_matches_{family}"):
                diagnostics[key] = diagnostics.get(key, 0) + 1
            for seat, controller in enumerate(episode.controllers):
                if controller in models and isinstance(models[controller], Actor):
                    replays[identity, seat] = Replay(models[controller])
            # Only learner seats use the deployed decision support. Historical league
            # opponents and human-role partners retain their original distributions.
            support = (
                {
                    "choicePolicy": choice_policy,
                    "playingSeats": [
                        s < count and episode.controllers[s] == "learner"
                        for s in range(4)
                    ],
                }
                if choice_policy
                else {}
            )
            requests.append(
                {
                    **support,
                    "op": "reset",
                    "id": identity,
                    "training": True,
                    "truthfulCPU": True,
                    "seed": rng.randrange(TRAIN_MIN, TRAIN_MAX),
                    "config": {
                        "players": count,
                        "target": rng.choice((12, 24)),
                        "florRequired": bool(rng.randrange(2)),
                    },
                    "roles": episode.roles,
                }
            )
        observations = dict(enumerate(pool.call(requests)))
        trajectories = {
            i: {seat: [] for seat, c in enumerate(ep.controllers) if c == "learner"}
            for i, ep in episodes.items()
        }
        critic_states = {}
        critic_histories = {}
        hands = {i: 0 for i in episodes}
        cursors = {i: [0] * 4 for i in episodes}
        for _ in range(10000):
            if monitor:
                monitor.check()
            if not observations:
                break
            grouped = {}
            decisions = {}
            for identity, obs in list(observations.items()):
                if obs["kind"] == "signal" and obs["view"]["config"]["players"] != 4:
                    raise ValueError(
                        "heads-up signal opportunity: use the corrected simulator"
                    )
                training = obs.get("training")
                if not training or training["version"] != 1:
                    raise ValueError("training channel missing")
                page = training["memory"]
                seat = obs["seat"]
                controller = episodes[identity].controllers[seat]
                if choice_policy and controller == "learner" and obs["kind"] == "game":
                    if obs.get("choicePolicy") != choice_policy:
                        raise ValueError(
                            "learner decisions were not collected with the configured CPU choice policy"
                        )
                    diagnostics["choice_policy_decisions"] = (
                        diagnostics.get("choice_policy_decisions", 0) + 1
                    )
                if (
                    controller == "learner"
                    and obs["kind"] == "signal"
                    and "signalMask" not in obs
                ):
                    raise ValueError("truthful learner mask missing")
                if not page["complete"]:
                    raise ValueError("authorized journal overflow")
                if page["hand"] != hands[identity]:
                    hands[identity] = page["hand"]
                    cursors[identity] = [0] * 4
                if (identity, seat) in replays:
                    # Pages larger than the ordinary incremental window are
                    # recovered explicitly, never silently shortened.
                    replay = replays[identity, seat]
                    while page["end"] < page["total"]:
                        events += advance_group(models[controller], [replay], [page])
                        cursors[identity][seat] = page["end"]
                        obs = pool.call(
                            [
                                {
                                    "op": "observe",
                                    "id": identity,
                                    "memoryHand": hands[identity],
                                    "memoryCursors": cursors[identity],
                                }
                            ]
                        )[0]
                        page = obs["training"]["memory"]
                        observations[identity] = obs
                cursors[identity][seat] = page["total"]
                if obs["kind"] == "game" and obs["legal"][0]["type"] == "cut":
                    if (identity, seat) in replays:
                        events += advance_group(
                            models[controller], [replays[identity, seat]], [page]
                        )
                    decisions[identity] = 0
                elif controller.startswith("script:"):
                    action = script_fn(obs, controller.split(":", 1)[1], rng)
                    if obs.get("signalMask") and not obs["signalMask"][action]:
                        action = 0
                    decisions[identity] = action
                else:
                    grouped.setdefault(controller, []).append(identity)
            with torch.no_grad():
                for controller, identities in grouped.items():
                    model = models[controller]
                    batch = [observations[i] for i in identities]
                    if isinstance(model, Actor):
                        rs = [replays[i, observations[i]["seat"]] for i in identities]
                        events += advance_group(
                            model, rs, [o["training"]["memory"] for o in batch]
                        )
                        features = torch.from_numpy(
                            np.asarray([o["features"] for o in batch], np.float32)
                        )
                        contexts = model.state(features)
                        if model.recurrent:
                            contexts = model.projected_context(
                                contexts,
                                features,
                                torch.stack([r.memory[0] for r in rs]),
                            )
                        # Candidate scoring is batched independently of event
                        # replay; only actual legal commands enter the logits.
                        distribution = candidate_distribution(model, contexts, batch)
                        actions = distribution.sample()
                        logp = distribution.log_prob(actions)
                        values = None
                        if controller == "learner":
                            keys = [
                                (i, o["seat"], o["training"]["memory"]["hand"])
                                for i, o in zip(identities, batch, strict=True)
                            ]
                            for key in keys:
                                if key not in critic_states:
                                    critic_states[key] = critic.initial()
                                    critic_histories[key] = []
                            memory = tuple(
                                torch.stack([critic_states[k][j] for k in keys])
                                for j in (0, 1)
                            )
                            private = torch.from_numpy(
                                np.asarray(
                                    [o["training"]["privileged"] for o in batch],
                                    np.float32,
                                )
                            )
                            values, new_memory = critic.step(features, private, memory)
                            for row, key in enumerate(keys):
                                critic_states[key] = tuple(
                                    x[row].clone() for x in new_memory
                                )
                                critic_histories[key].append(
                                    np.concatenate(
                                        (features[row].numpy(), private[row].numpy())
                                    )
                                )
                        for row, (identity, obs) in enumerate(
                            zip(identities, batch, strict=True)
                        ):
                            action = int(actions[row])
                            decisions[identity] = action
                            if controller == "learner":
                                replay = rs[row]
                                compact = {
                                    k: obs[k]
                                    for k in (
                                        "features",
                                        "actions",
                                        "kind",
                                        "version",
                                        "seat",
                                    )
                                }
                                if "choicePolicy" in obs:
                                    compact["choicePolicy"] = obs["choicePolicy"]
                                if "signalMask" in obs:
                                    compact["signalMask"] = obs["signalMask"]
                                # Training metadata only: a pending wager alone
                                # need not give this seat response authority.
                                kinds = {
                                    a.get("kind")
                                    for a in (obs.get("legal") or [])
                                    if a.get("type") == "respond"
                                }
                                compact["responseChoice"] = (
                                    obs["kind"] == "game"
                                    and {"accept", "decline"} <= kinds
                                )
                                if compact["responseChoice"]:
                                    compact["responseFamily"] = (
                                        obs["view"].get("pending") or {}
                                    ).get("kind", "unknown")
                                key = keys[row]
                                history = critic_histories[key]
                                trajectories[identity][obs["seat"]].append(
                                    JointDecision(
                                        replay.events,
                                        len(replay.events),
                                        compact,
                                        np.asarray(
                                            obs["training"]["privileged"], np.float32
                                        ),
                                        action,
                                        float(logp[row]),
                                        float(values[row]),
                                        critic_events=history,
                                        critic_end=len(history),
                                        match=match_base + identity,
                                        learner_seat=obs["seat"],
                                        balance_stratum=stratum_for_episode(
                                            len(episodes[identity].controllers),
                                            episodes[identity].controllers,
                                            episodes[identity].roles,
                                            obs["seat"],
                                        ),
                                    )
                                )
                                if obs["kind"] == "signal":
                                    diagnostics["signal_decisions"] += 1
                                    diagnostics["truthful_claims"] += int(action != 0)
                                    diagnostics["claim_opportunities"] += int(
                                        any(obs["signalMask"][1:])
                                    )
                                    diagnostics["redundant_claims"] = diagnostics.get(
                                        "redundant_claims", 0
                                    ) + int(redundant_signal(obs, action))
                                else:
                                    kind = obs["legal"][action].get("kind")
                                    diagnostics["accepts"] += int(kind == "accept")
                                    diagnostics["declines"] += int(kind == "decline")
                    else:
                        dist, _ = batch_distribution(model, batch)
                        logits = dist.logits.clone()
                        for row, obs in enumerate(batch):
                            mask = obs.get("signalMask") or obs.get("truthMask")
                            if mask:
                                logits[row, :6] = logits[row, :6].masked_fill(
                                    ~torch.tensor(mask), -torch.inf
                                )
                        for identity, action in zip(
                            identities,
                            torch.distributions.Categorical(logits=logits).sample(),
                            strict=True,
                        ):
                            decisions[identity] = int(action)
            requests = [
                {
                    "op": "step",
                    "id": i,
                    "action": decisions[i],
                    "memoryHand": hands[i],
                    "memoryCursors": cursors[i],
                }
                for i in observations
            ]
            for request, obs in zip(requests, pool.call(requests), strict=True):
                identity = request["id"]
                if obs["done"]:
                    selected = team_credit(
                        trajectories[identity], obs["winner"], gae_lambda
                    )
                    choices += len(selected)
                    records.extend(selected)
                    diagnostics["learner_trajectories"] = diagnostics.get(
                        "learner_trajectories", 0
                    ) + len(trajectories[identity])
                    diagnostics["joint_matches"] = diagnostics.get(
                        "joint_matches", 0
                    ) + int(len(trajectories[identity]) > 1)
                    del observations[identity]
                    matches += 1
                else:
                    observations[identity] = obs
        if observations:
            raise RuntimeError("cooperation rollout exceeded match bound")
    return records, matches, events, diagnostics
