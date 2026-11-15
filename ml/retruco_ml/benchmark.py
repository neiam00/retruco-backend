"""Compare actual tensor packing + training steps on CPU and Apple MPS."""

import copy
import json
import time

import torch

from .model import ENCODER_VERSION, Policy, batch_distribution


def benchmark():
    cpu_rng = torch.get_rng_state()
    mps_rng = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    observations = [
        {
            "version": ENCODER_VERSION,
            "kind": "game",
            "features": [0.1] * 768,
            "actions": [[0.2] * 48] * 32,
        }
        for _ in range(512)
    ]
    result = {}
    try:
        reference = Policy()
        devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
        for device in devices:
            model = copy.deepcopy(reference).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            durations = []
            for repeat in range(5):
                if device == "mps":
                    torch.mps.synchronize()
                start = time.perf_counter()
                distribution, value = batch_distribution(model, observations, device)
                loss = (
                    -distribution.log_prob(
                        torch.zeros(512, dtype=torch.int64, device=device)
                    ).mean()
                    + value.square().mean()
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if device == "mps":
                    torch.mps.synchronize()
                if repeat:
                    durations.append(time.perf_counter() - start)
            result[device] = sum(durations) / len(durations)
        # Require a meaningful win to offset GPU setup and variable batch sizes.
        result["selected"] = (
            "mps" if result.get("mps", float("inf")) < result["cpu"] * 0.9 else "cpu"
        )
        return result
    finally:
        torch.set_rng_state(cpu_rng)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)


if __name__ == "__main__":
    torch.set_num_threads(2)
    print(json.dumps(benchmark()))
