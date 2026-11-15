"""Learning clocks count genuine choices; the authorized memory clock never changes."""

from .mappo_rollout import credit


def choice_count(observation):
    if observation["kind"] == "signal":
        return sum(observation.get("signalMask", [True] * 6))
    return len(observation["actions"])


def meaningful(record):
    return choice_count(record.observation) > 1


def choice_credit(trajectory, reward):
    selected = [record for record in trajectory if meaningful(record)]
    credit(selected, reward)
    return selected


def monte_carlo_credit(trajectory, reward):
    for record in trajectory:
        record.target = float(reward)
        record.advantage = float(reward) - record.value
    return trajectory
