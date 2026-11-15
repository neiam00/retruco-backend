"""Complete-match, single learner-seat collection with authorized off-turn memory."""

from dataclasses import dataclass

import numpy as np
import torch

from .model import batch_distribution
from .population import scripted
from .recurrent import Replay
from .splits import TRAIN_MAX, TRAIN_MIN


@dataclass
class Decision:
    events: list
    end: int
    observation: dict
    privileged: np.ndarray
    action: int
    logp: float
    value: float
    advantage: float = 0
    target: float = 0


def credit(trajectory, reward, lam=0.95):
    next_value = gae = 0.0
    for i in reversed(range(len(trajectory))):
        r = trajectory[i]
        gae = (
            (reward if i == len(trajectory) - 1 else 0)
            + next_value
            - r.value
            + lam * gae
        )
        r.advantage, r.target = gae, gae + r.value
        next_value = r.value


def collect(
    actor,
    critic,
    reference,
    pool,
    population,
    rng,
    minimum,
    players=0,
    environments=8,
    monitor=None,
):
    # The opponent/partner policies are frozen legacy models in all four arms.
    # Only the designated learner seat changes; no collection-mode confound.
    models = population.models(reference)
    records, matches, event_count = [], 0, 0
    while len(records) < minimum:
        episodes, requests = {}, []
        for identity in range(environments):
            count = players or rng.choice((2, 4))
            episode = population.episode(count, rng)
            episodes[identity] = episode
            requests.append(
                {
                    "op": "reset",
                    "id": identity,
                    "training": True,
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
        trajectories = {i: [] for i in episodes}
        replay = {i: Replay(actor) for i in episodes}
        cursors = {i: [0] * 4 for i in episodes}
        hands = {i: 0 for i in episodes}
        for _ in range(10000):
            if monitor:
                monitor.check()
            if not observations:
                break
            decisions, grouped = {}, {}
            for identity, obs in observations.items():
                training = obs.get("training")
                if not training or training["version"] != 1:
                    raise ValueError(
                        "rebuild simulator: training-only memory/critic channel unavailable"
                    )
                page = training["memory"]
                if not page["complete"]:
                    raise ValueError(
                        "authorized hand journal exceeded experiment capacity"
                    )
                if hands[identity] != page["hand"]:
                    hands[identity], cursors[identity] = page["hand"], [0] * 4
                seat = obs["seat"]
                controller = episodes[identity].controllers[seat]
                while True:
                    cursors[identity][seat] = page["end"]
                    if controller == "learner":
                        before = (
                            len(replay[identity].events)
                            if replay[identity].hand == page["hand"]
                            else 0
                        )
                        replay[identity].apply(page)
                        event_count += len(replay[identity].events) - before
                    if page["end"] == page["total"]:
                        break
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
                    training, page = obs["training"], obs["training"]["memory"]
                if obs["kind"] == "game" and obs["legal"][0]["type"] == "cut":
                    decisions[identity] = 0
                elif controller.startswith("script:"):
                    decisions[identity] = scripted(obs, controller.split(":")[1], rng)
                elif controller != "learner":
                    grouped.setdefault(controller, []).append(identity)
                else:
                    r = replay[identity]
                    features = torch.as_tensor(np.asarray(obs["features"], np.float32))
                    privileged = np.asarray(training["privileged"], np.float32)
                    with torch.no_grad():
                        context = actor.context(features, r.memory)
                        distribution = actor.distribution(context, obs)
                        action = distribution.sample()
                        value = critic(features, torch.as_tensor(privileged), context)
                    decisions[identity] = int(action)
                    compact = {
                        k: obs[k]
                        for k in ("features", "actions", "kind", "version", "seat")
                    }
                    trajectories[identity].append(
                        Decision(
                            r.events,
                            len(r.events),
                            compact,
                            privileged,
                            int(action),
                            float(distribution.log_prob(action)),
                            float(value),
                        )
                    )
            with torch.no_grad():
                for controller, identities in grouped.items():
                    dist, _ = batch_distribution(
                        models[controller], [observations[i] for i in identities]
                    )
                    for identity, action in zip(identities, dist.sample(), strict=True):
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
                i = request["id"]
                if obs["done"]:
                    credit(
                        trajectories[i],
                        1 if episodes[i].learner % 2 == obs["winner"] else -1,
                    )
                    records.extend(trajectories[i])
                    del observations[i]
                    matches += 1
                else:
                    observations[i] = obs
        if observations:
            raise RuntimeError("controlled rollout exceeded complete-match bound")
    return records, matches, event_count
