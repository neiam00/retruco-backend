"""Same Go-authorized evaluation, with instrumented and unseen partner profiles."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .controlled_train import atomic_json
from .cooperation_diagnostics import DecisionContexts
from .cooperation_evaluate import PARTNERS, evaluate
from .cooperation_population import Profile, cooperation_scripted
from .model import Policy
from .model_capacity import actor_from_checkpoint
from .rollout import Simulator

# Evaluation-only truthful partners: lower/missed signals and occasional mistakes.
# A skipped early opportunity can yield a claim in a later trick; no extra
# communication opportunities are invented and named seña meanings never change.
UNSEEN = {
    "sparse-truthful": Profile(0.35, True, 28, 32, 0.3, 0.1),
    "hesitant-truthful": Profile(0.6, True, 30, 33, 0.2, 0.05),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--partner", choices=(*PARTNERS, *UNSEEN), required=True)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=190000)
    parser.add_argument(
        "--signals", choices=("normal", "no-send", "no-receive"), default="normal"
    )
    parser.add_argument("--players", type=int, choices=(0, 2, 4), default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("encoder") != 2 or saved.get("memory_version") != 1:
        raise ValueError("requires a Model D-compatible recurrent checkpoint")
    actor = actor_from_checkpoint(saved)
    actor.eval()
    ref = torch.load(args.reference, map_location="cpu", weights_only=True)
    reference = Policy()
    reference.load_state_dict(ref["model"])
    reference.eval()
    contexts = DecisionContexts()
    profile = UNSEEN.get(args.partner)
    partner_policy = (
        None
        if profile is None
        else lambda obs, rng: cooperation_scripted(
            obs, "literal-v2", rng, profile=profile
        )
    )

    def observe(observation, action):
        if (
            args.signals == "no-receive"
            and observation["view"]["config"]["players"] == 4
        ):
            partner_seat = (observation["seat"] + 2) % 4
            observation = {
                **observation,
                "signals": [
                    s
                    for s in observation.get("signals", [])
                    if s["seat"] != partner_seat
                ],
            }
        contexts.observe(observation, action)

    with Simulator(args.simulator, "binary") as simulator:
        report = evaluate(
            actor,
            reference,
            simulator,
            args.pairs,
            args.seed,
            "literal-v2" if profile else args.partner,
            args.signals,
            args.players,
            observer=observe,
            partner_policy=partner_policy,
        )
    report.update(
        protocol="controlled-cooperation-evaluation-v1",
        partner=args.partner,
        decision_contexts=contexts.counts,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
    )
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "pooled": report["pooled"],
                "behavior": report["behavior"],
                "contexts": contexts.counts,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
