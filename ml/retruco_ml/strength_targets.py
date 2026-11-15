"""User acceptance targets against the frozen native CPU, separated by mode."""

from .statistics import pooled

MODE_WIN_TARGETS = {2: 0.70, 4: 0.60}


def summarize(rows):
    result = {}
    for players, target in MODE_WIN_TARGETS.items():
        selected = [row for row in rows if row["players"] == players]
        if not selected:
            raise ValueError("missing strength target mode")
        measured = pooled(selected)
        result[str(players)] = {
            **measured,
            "target_win_rate": target,
            "target_met": measured["win_rate"] >= target,
            "confidence_above_chance": measured["cluster_ci95"][0] > 0.5,
        }
    return result
