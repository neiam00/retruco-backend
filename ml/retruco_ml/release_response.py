"""Train one fresh empirical best response against the exact recurrent browser artifact."""

import argparse
import fcntl
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch

from .artifacts import digest
from .controlled_train import atomic_json
from .model import Policy
from .population import Population
from .release_recurrent import BrowserActor
from .rollout import SimulatorPool, collect
from .train import update


class ResponsePopulation(Population):
    def episode(self, players, rng):
        episode = super().episode(players, rng)
        for seat in range(players):
            if seat % 2 != episode.learner % 2:
                episode.controllers[seat] = "release"
        return episode


def train(
    checkpoint,
    bundle,
    library,
    simulator,
    out,
    seed,
    players,
    seconds=3600,
    steps=8192,
    environments=32,
):
    if players not in (2, 4) or min(steps, environments, seconds) <= 0:
        raise ValueError("Invalid best-response parameters")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "trial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        rng = random.Random(seed)
        model = Policy()
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        actor = BrowserActor(checkpoint, bundle, library)
        initial = {"kind": "random", "seed": seed}
        init_hash = hashlib.sha256(
            b"".join(t.numpy().tobytes() for t in model.state_dict().values())
        ).hexdigest()
        config = {
            "players": players,
            "seed": seed,
            "steps": steps,
            "environments": environments,
            "profile": actor.profile,
            "simulator_sha256": digest(Path(simulator)),
            "runtime_sha256": digest(Path(library)),
            "manifest_sha256": digest(Path(bundle) / "manifest.json"),
            "source_files": {
                name: digest(Path(__file__).with_name(name))
                for name in (
                    "release_response.py",
                    "release_recurrent.py",
                    "rollout.py",
                    "train.py",
                    "model.py",
                    "population.py",
                )
            },
        }
        pop = ResponsePopulation()
        elapsed = cpu_time = 0.0
        iteration = matches = decisions = 0
        paths = sorted(out.glob("update-*.pt"))
        if paths:
            last = torch.load(paths[-1], weights_only=True, map_location="cpu")
            if (
                last["config"] != config
                or last["initialization"] != initial
                or last["initial_model_sha256"] != init_hash
                or last["opponent_checkpoint_sha256"] != actor.checkpoint_sha256
                or bytes(last["fixed_opponent_blob"].tolist()) != actor.blob
            ):
                raise ValueError(
                    "Recovery checkpoint belongs to a different training run"
                )
            model.load_state_dict(last["model"])
            opt.load_state_dict(last["optimizer"])
            rng.setstate(last["random"])
            torch.set_rng_state(last["torch_random"])
            pop = ResponsePopulation(last["population"])
            elapsed = last["total_seconds"]
            cpu_time = last["process_cpu_seconds"]
            iteration = last["update"]
            matches = last["matches"]
            decisions = last["decisions"]
        with SimulatorPool(1, "binary", str(simulator)) as pool:
            last_saved = elapsed
            while elapsed < seconds:
                start = time.monotonic()
                cpu_start = time.process_time()
                records, num_matches, controllers = collect(
                    model,
                    pool,
                    pop,
                    rng,
                    steps,
                    players,
                    environments,
                    release_target=actor,
                )
                if not controllers.get("release") or not records:
                    raise ValueError("No training samples against the frozen release")
                loss = update(model, opt, records, rng)
                if not torch.isfinite(torch.tensor(loss)):
                    raise ValueError("Nonfinite opponent update")
                iteration += 1
                matches += num_matches
                decisions += len(records)
                if iteration % 10 == 0 or not pop.snapshots:
                    pop.add(model)
                elapsed += time.monotonic() - start
                cpu_time += time.process_time() - cpu_start
                metrics = {
                    "update": iteration,
                    "training_seconds": elapsed,
                    "process_cpu_seconds": cpu_time,
                    "matches": matches,
                    "decisions": decisions,
                    "loss": loss,
                    "controllers": controllers,
                    "pid": os.getpid(),
                }
                if iteration == 1 or elapsed - last_saved >= 600 or elapsed >= seconds:
                    file = out / f"update-{iteration:06d}.pt"
                    if file.exists():
                        raise FileExistsError(file)
                    tmp = file.with_suffix(".tmp")
                    torch.save(
                        {
                            "encoder": 2,
                            "model": model.state_dict(),
                            "optimizer": opt.state_dict(),
                            "random": rng.getstate(),
                            "torch_random": torch.get_rng_state(),
                            "population": pop.snapshots,
                            "initialization": initial,
                            "initial_seed": seed,
                            "initial_model_sha256": init_hash,
                            "config": config,
                            "matches": matches,
                            "decisions": decisions,
                            "total_seconds": elapsed,
                            "process_cpu_seconds": cpu_time,
                            "update": iteration,
                            "fixed_opponent_blob": torch.tensor(
                                list(actor.blob), dtype=torch.uint8
                            ),
                            "opponent_checkpoint_sha256": actor.checkpoint_sha256,
                            "opponent_model_sha256": actor.manifest["sha256"],
                            "wasm_sha256": actor.manifest["wasm_sha256"],
                            "approved": False,
                        },
                        tmp,
                    )
                    os.replace(tmp, file)
                    last_saved = elapsed
                    atomic_json(
                        out / "latest.json",
                        {"file": file.name, "sha256": digest(file), **metrics},
                    )
                with (out / "metrics.jsonl").open("a") as log:
                    log.write(json.dumps(metrics) + "\n")
                atomic_json(
                    out / "status.json",
                    {
                        "stage": "complete" if elapsed >= seconds else "training",
                        **metrics,
                    },
                )
        return json.loads((out / "latest.json").read_text())


def main():
    p = argparse.ArgumentParser()
    for n in ["checkpoint", "bundle", "library", "simulator", "output"]:
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--players", type=int, choices=[2, 4], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--seconds", type=int, default=3600)
    p.add_argument("--steps", type=int, default=8192)
    p.add_argument("--environments", type=int, default=32)
    a = p.parse_args()
    print(
        json.dumps(
            train(
                a.checkpoint,
                a.bundle,
                a.library,
                a.simulator,
                a.output,
                a.seed,
                a.players,
                a.seconds,
                a.steps,
                a.environments,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
