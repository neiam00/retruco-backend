"""Use the unchanged paired Truco benchmark for new recurrent MAPPO actors."""

import argparse
import json
from pathlib import Path

import torch

from .controlled_train import atomic_json
from .improvement_budget import DeadlineReached
from .improvement_evaluate import (
    FAMILIES,
    PARTNERS,
    SIGNAL_MODES,
    evaluate,
    frozen_models,
)
from .model_capacity import actor_from_checkpoint as legacy_actor
from .recurrent import file_hash
from .rmappo_model import actor_from_checkpoint
from .rollout import Simulator
from .team_play_probe import PROBES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--simulator", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--partner", choices=PARTNERS, required=True)
    p.add_argument("--opponent", choices=FAMILIES, required=True)
    p.add_argument("--players", type=int, choices=(2, 4), required=True)
    p.add_argument("--pairs", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--signals", choices=SIGNAL_MODES, default="normal")
    p.add_argument(
        "--card-policy", choices=("policy", "highest", *PROBES), default="policy"
    )
    a = p.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    actor = (
        actor_from_checkpoint(saved)
        if saved.get("kind") == "rmappo-v1"
        else legacy_actor(saved)
    )
    actor.eval()
    manifest = json.loads(a.inputs.read_text())
    models = frozen_models(manifest, a.inputs.resolve().parent)
    with Simulator(a.simulator, "binary") as simulator:
        report = evaluate(
            actor,
            models,
            simulator,
            a.pairs,
            a.seed,
            a.partner,
            a.opponent,
            a.signals,
            a.players,
            a.card_policy,
        )
    report.update(
        checkpoint_sha256=file_hash(a.checkpoint),
        inputs_sha256=file_hash(a.inputs),
        architecture=saved.get("configuration", {}),
        initialization=saved.get("configuration", {}).get(
            "initialization", "historical-reference"
        ),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(a.output, report)
    receipt = a.output.with_suffix(".sha256")
    temp = receipt.with_suffix(".tmp")
    temp.write_text(file_hash(a.output) + "\n")
    temp.replace(receipt)


if __name__ == "__main__":
    try:
        main()
    except DeadlineReached:
        raise SystemExit(75)
