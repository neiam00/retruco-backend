"""Match-stable opponents and human-like partners; styles never enter actor inputs."""

from dataclasses import dataclass

from .model import Policy

STYLES = (
    "baseline",
    "cautious",
    "aggressive",
    "call-heavy",
    "literal",
    "deceptive",
    "silent",
)
SIGNALS = ("none", "perico", "perica", "seven_spades", "seven_coins", "receta")


@dataclass
class Episode:
    controllers: list[str]
    roles: list[bool]
    learner: int


class Population:
    def __init__(self, snapshots=None, fixed_opponent=None):
        self.snapshots = list(snapshots or [])[-16:]
        self.fixed_opponent = fixed_opponent

    def add(self, policy):
        self.snapshots.append(
            {
                key: value.detach().cpu().clone()
                for key, value in policy.state_dict().items()
            }
        )
        self.snapshots = self.snapshots[-16:]

    def models(self, current):
        result = {"current": current, "learner": current}
        for index, state in enumerate(self.snapshots):
            model = Policy()
            model.load_state_dict(state)
            model.eval()
            result[f"history:{index}"] = model
        if self.fixed_opponent is not None:
            fixed = Policy()
            fixed.load_state_dict(self.fixed_opponent)
            fixed.eval()
            result["fixed"] = fixed
        return result

    def historical(self, rng):
        return (
            f"history:{rng.randrange(len(self.snapshots))}"
            if self.snapshots
            else "current"
        )

    def episode(self, players, rng):
        learner = rng.randrange(players)
        draw = rng.random()
        opponent = (
            "current"
            if draw < 0.7
            else self.historical(rng)
            if draw < 0.9
            else f"script:{rng.choice(STYLES)}"
        )
        if self.fixed_opponent is not None:
            opponent = "fixed"
        controllers = [opponent] * players
        roles = [True] * 4
        controllers[learner] = "learner"
        if players == 4:
            partner = (learner + 2) % 4
            draw = rng.random()
            if draw < 0.5:
                controllers[partner] = rng.choice(("current", self.historical(rng)))
            else:
                controllers[partner] = (
                    f"script:{rng.choice(('literal', 'cautious', 'call-heavy'))}"
                    if draw < 0.85
                    else f"script:{rng.choice(('silent', 'ignore'))}"
                )
                roles[partner] = False  # Human proxy owns pending team cantos.
        return Episode(controllers, roles, learner)


def truthful_signals(view):
    if view.get("config", {}).get("players") != 4:
        return []
    vira = view.get("vira")
    if not vira:
        return []
    cards = {
        (c["card"]["suit"], c["card"]["number"]) for c in view["hand"] if not c["used"]
    }
    perico = (vira["suit"], 12 if vira["number"] == 11 else 11)
    perica = (vira["suit"], 12 if vira["number"] == 10 else 10)
    result = []
    for index, card in (
        (1, perico),
        (2, perica),
        (3, ("spades", 7)),
        (4, ("coins", 7)),
    ):
        if card in cards:
            result.append(index)
    if perico in cards and perica in cards:
        result.append(5)
    return result


def scripted(observation, style, rng):
    view = observation["view"]
    if observation["kind"] == "signal":
        if view.get("config", {}).get("players") != 4:
            return 0
        if style in ("baseline", "silent"):
            return 0
        truth = truthful_signals(view)
        if style == "deceptive" and rng.random() < 0.6:
            return rng.choice([i for i in range(1, 6) if i not in truth] or [0])
        if style == "ignore":
            return rng.randrange(6) if rng.random() < 0.2 else 0
        return rng.choice(truth) if truth and rng.random() < 0.6 else 0
    legal = observation["legal"]
    if legal[0]["type"] == "cut":
        return 0  # Simulator applies its independent uniform cut stream.

    def choices(predicate):
        return [i for i, action in enumerate(legal) if predicate(action)]

    if style == "aggressive" and rng.random() < 0.75:
        bids = choices(lambda a: a["type"] == "call" or a.get("kind") == "raise")
        if bids:
            return max(bids, key=lambda i: legal[i].get("amount", 0))
    if style == "cautious":
        declines = choices(lambda a: a.get("kind") == "decline")
        if declines:
            return declines[0]
    if style in ("call-heavy", "literal"):
        willing = style == "call-heavy" or any(
            signal["seat"] != observation["seat"]
            and signal["seat"] % 2 == observation["seat"] % 2
            and signal["kind"] in ("perico", "perica", "receta")
            for signal in observation.get("signals", [])
        )
        accepts = choices(lambda a: a.get("kind") == "accept")
        if willing and accepts:
            return accepts[0]
    return observation["baseline"]
