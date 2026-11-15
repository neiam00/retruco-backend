"""Parallel native simulation, batched inference, complete-match on-policy data."""

from concurrent.futures import ThreadPoolExecutor

import torch

from .bridge import Simulator
from .model import ENCODER_VERSION, batch_distribution
from .population import scripted
from .splits import TRAIN_MAX, TRAIN_MIN


class SimulatorPool:
    def __init__(self, workers=8, transport="json", executable=None):
        if workers < 1 or workers > 64:
            raise ValueError("workers must be in [1,64]")
        self.simulators = []
        self.executor = ThreadPoolExecutor(max_workers=workers)
        try:
            for _ in range(workers):
                self.simulators.append(Simulator(executable, transport=transport))
        except Exception:
            self.close()
            raise

    def call(self, requests):
        groups = [[] for _ in self.simulators]
        for request in requests:
            groups[request["id"] % len(groups)].append(request)
        pending = []
        for worker, group in enumerate(groups):
            if group:
                pending.append(
                    (group, self.executor.submit(self.simulators[worker].call, group))
                )
        results = {}
        for group, future in pending:
            for request, observation in zip(group, future.result(), strict=True):
                results[request["id"]] = observation
        return [results[request["id"]] for request in requests]

    def close(self):
        self.executor.shutdown(wait=True)
        for simulator in self.simulators:
            simulator.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def seat_returns(trajectory, reward, lam=0.95):
    result = []
    next_value = gae = 0.0
    for index in reversed(range(len(trajectory))):
        record = trajectory[index]
        terminal_reward = reward if index == len(trajectory) - 1 else 0.0
        gae = terminal_reward + next_value - record[3] + lam * gae
        result.append((*record, gae, gae + record[3]))
        next_value = record[3]
    return list(reversed(result))


def collect(
    model, pool, population, rng, minimum, players=0, environments=64, monitor=None
):
    if not 1 <= environments <= 64:
        raise ValueError("environments must be in [1,64]")
    models = population.models(model)
    records = []
    matches = 0
    controller_counts = {}
    # Complete waves ensure no trajectories straddle policy updates. Models and
    # every seat's opponent/partner assignment remain frozen throughout a match.
    while len(records) < minimum:
        episodes, requests = {}, []
        for identity in range(environments):
            count = players or rng.choice((2, 4))
            episode = population.episode(count, rng)
            episodes[identity] = episode
            for controller in episode.controllers:
                controller_counts[controller.split(":")[0]] = (
                    controller_counts.get(controller.split(":")[0], 0) + 1
                )
            requests.append(
                {
                    "op": "reset",
                    "id": identity,
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
        trajectories = {identity: [] for identity in episodes}
        for _ in range(10000):
            if monitor:
                monitor.check()
            if not observations:
                break
            decisions, grouped = {}, {}
            for identity, observation in observations.items():
                if observation["version"] != ENCODER_VERSION:
                    raise ValueError(
                        "simulator encoder version mismatch: rebuild bin/selfplay"
                    )
                controller = episodes[identity].controllers[observation["seat"]]
                cut = (
                    observation["kind"] == "game"
                    and observation["legal"][0]["type"] == "cut"
                )
                if cut:
                    decisions[identity] = 0
                elif controller.startswith("script:"):
                    decisions[identity] = scripted(
                        observation, controller.split(":")[1], rng
                    )
                else:
                    grouped.setdefault(controller, []).append(identity)
            with torch.no_grad():
                for controller, identities in grouped.items():
                    batch = [observations[i] for i in identities]
                    distribution, values = batch_distribution(models[controller], batch)
                    actions = distribution.sample()
                    logp = distribution.log_prob(actions)
                    for row, identity in enumerate(identities):
                        action = int(actions[row])
                        decisions[identity] = action
                        if controller == "learner":
                            obs = observations[identity]
                            compact = {
                                key: obs[key]
                                for key in (
                                    "features",
                                    "actions",
                                    "kind",
                                    "version",
                                    "seat",
                                )
                            }
                            trajectories[identity].append(
                                (compact, action, float(logp[row]), float(values[row]))
                            )
            requests = [
                {"op": "step", "id": identity, "action": decisions[identity]}
                for identity in observations
            ]
            updated = pool.call(requests)
            for request, observation in zip(requests, updated, strict=True):
                identity = request["id"]
                if observation["done"]:
                    reward = (
                        1.0
                        if episodes[identity].learner % 2 == observation["winner"]
                        else -1.0
                    )
                    records.extend(seat_returns(trajectories[identity], reward))
                    del observations[identity]
                    matches += 1
                else:
                    observations[identity] = observation
        if observations:
            raise RuntimeError("rollout wave exceeded complete-match step budget")
    return records, matches, controller_counts
