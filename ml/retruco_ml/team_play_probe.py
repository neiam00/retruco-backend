"""Optional evaluation interventions using only a player's public view and own cards."""

LIVE_CHOICES = "value-and-team-v17"
CHOICE_PROFILES = (
    "value-and-team-v1",
    "value-and-team-v2",
    "value-and-team-v3",
    "value-and-team-v4",
    "value-and-team-v5",
    "value-and-team-v6",
    "value-and-team-v7",
    "value-and-team-v8",
    "value-and-team-v9",
    "value-and-team-v10",
    "value-and-team-v11",
    "value-and-team-v12",
    "value-and-team-v13",
    "value-and-team-v14",
    "value-and-team-v15",
    "value-and-team-v16",
    LIVE_CHOICES,
)

PROBES = (
    "closing-card",
    "keep-flor",
    "closing-card-and-flor",
    "closing-card-and-flor-v1",
    "closing-card-and-flor-v2",
    *CHOICE_PROFILES,
)
FLOR_WORDS = {"flor", "flor_flor", "presento_flor", "con_mi_flor", "con_flor"}


def _other_ranks(view, features, seat, trick):
    ranks = []
    for player in view["seats"]:
        if player["seat"] == seat:
            continue
        play = player["plays"][trick]
        if (
            not play["present"]
            or play["concealed"]
            or (not play["burned"] and not play.get("card"))
        ):
            return None
        if not play["burned"]:
            offset = (player["seat"] - seat) % 4
            ranks.append((features[43 + 22 * offset + 7 * trick], player["team"]))
    return ranks


def _winner(ranks, allied_high_wins):
    if not ranks:
        return None
    best = max(rank for rank, _ in ranks)
    winners = [team for rank, team in ranks if rank == best]
    if len(winners) == 1 or (allied_high_wins and len(set(winners)) == 1):
        return winners[0]
    return None


def _closing_card(view, features, seat, legal, action, allied_first):
    chosen = legal[action]
    if (
        chosen["type"] not in ("play", "burn")
        or view.get("pending")
        or view["mandatoryStack"]
    ):
        return action
    trick = view["round"]
    # A voluntary pegado has a hidden bottom and a conditional top. Leave that
    # decision to the actor, even if the top is already visible in trick three.
    if trick and any(player["plays"][1]["concealed"] for player in view["seats"]):
        return action
    others = _other_ranks(view, features, seat, trick)
    if others is None:
        return action
    team = seat % 2
    first_winner = None
    if trick:
        first = _other_ranks(view, features, seat, 0)
        own_first = next(p for p in view["seats"] if p["seat"] == seat)["plays"][0]
        if first is None or not own_first["present"] or not own_first.get("card"):
            return action
        first_winner = _winner(first + [(features[43], team)], allied_first)
        if first_winner is None:
            return action

    def outcome(ranks):
        winner = _winner(ranks, allied_first and trick == 0)
        if winner is None:
            if not trick:
                return 0
            winner = first_winner
        return 1 if winner == team else -1

    # Match the whole command apart from physical card/type. A different canto,
    # mode, amount or declaration is never introduced by the card intervention.
    def modifiers(command):
        return {
            key: value for key, value in command.items() if key not in ("card", "type")
        }

    original = modifiers(chosen)
    plays = [
        i
        for i, command in enumerate(legal)
        if command["type"] == "play" and modifiers(command) == original
    ]
    if not plays:
        return action

    def strength(i):
        return features[24 + 5 * legal[i].get("card", 0)]

    def result(i):
        return outcome(others + [(strength(i), team)])

    best = max(result(i) for i in plays)
    actual = outcome(others) if chosen["type"] == "burn" else result(action)
    # No change when every legal card loses or the burn already gives the best result.
    if best < 0 or actual > best or (actual == best and chosen["type"] == "burn"):
        return action
    sufficient = [i for i in plays if result(i) == best]
    cheapest = min(sufficient, key=strength)
    if actual == best and strength(action) == strength(cheapest):
        return action
    return cheapest


def _keep_flor(view, legal, action):
    chosen = legal[action]
    if (
        not view["config"]["florRequired"]
        or not view.get("florValid")
        or not view.get("holding", {}).get("flor")
        or view["holding"].get("receta")
        or chosen["type"] not in ("play", "burn", "stack", "call")
        or chosen.get("declaration")
    ):
        return action
    # Look only among commands emitted by the rules engine for the very same
    # move. On the first card, a ley also keeps a teammate's possible flor alive.
    flor = action
    for i, alternate in enumerate(legal):
        declaration = alternate.get("declaration")
        if (
            not declaration
            or {k: v for k, v in alternate.items() if k != "declaration"} != chosen
        ):
            continue
        if declaration == "a_ley" and chosen["type"] == "play" and view["round"] == 0:
            return i
        if declaration in FLOR_WORDS:
            flor = i
    return flor


def team_action(view, features, seat, legal, action, probe):
    """Return an existing legal index and which interventions changed it."""
    if probe not in PROBES:
        raise ValueError("unknown team decision probe")
    if view["config"]["players"] != 4 or view["phase"] != "playing":
        return action, False, False
    allied_first = probe != "closing-card-and-flor-v1"
    card = (
        _closing_card(view, features, seat, legal, action, allied_first)
        if probe != "keep-flor"
        else action
    )
    final = _keep_flor(view, legal, card) if probe != "closing-card" else card
    return final, card != action, final != card
