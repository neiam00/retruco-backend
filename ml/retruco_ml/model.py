"""Fixed deployment architecture. Candidate order is the engine's legal order."""

import numpy as np
import torch
from torch import nn

STATE_SIZE = 768
ACTION_SIZE = 48
ENCODER_VERSION = 2
ACTION_ENCODER_VERSION = 1  # The 48 candidate features in shared Go encoder v2.


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = nn.Sequential(
            nn.Linear(768, 96), nn.ReLU(), nn.Linear(96, 64), nn.ReLU()
        )
        self.action = nn.Sequential(nn.Linear(48, 32), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 1))
        self.signal = nn.Linear(64, 6)
        self.value = nn.Linear(64, 1)

    def forward(self, state, actions):
        hidden = self.state(state)
        candidates = self.action(actions)
        expanded = hidden.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        logits = self.score(torch.cat((expanded, candidates), dim=-1)).squeeze(-1)
        return logits, self.signal(hidden), self.value(hidden).squeeze(-1)

    def distribution(self, observation, device="cpu"):
        if observation["version"] != ENCODER_VERSION:
            raise ValueError("unsupported observation encoder")
        state = torch.from_numpy(
            np.asarray(observation["features"], dtype=np.float32)[None, :]
        ).to(device)
        actions = observation["actions"]
        if len(actions) == 0:
            actions = np.zeros((1, ACTION_SIZE), dtype=np.float32)
        action = torch.from_numpy(np.asarray(actions, dtype=np.float32)[None, :, :]).to(
            device
        )
        game, signal, value = self(state, action)
        if observation["kind"] == "signal" and "signalMask" in observation:
            limit = torch.as_tensor(
                observation["signalMask"], dtype=torch.bool, device=device
            )
            if limit.shape != (6,) or not limit[0]:
                raise ValueError("invalid truthful signal mask")
            signal = signal.masked_fill(~limit, -torch.inf)
        return torch.distributions.Categorical(
            logits=(signal if observation["kind"] == "signal" else game)[0]
        ), value[0]


DEPLOY_LAYERS = ("state.0", "state.2", "action.0", "score.0", "score.2", "signal")


class PackedObservations:
    """Convert once, retain ragged actions, and pad only the selected minibatch.

    PPO reuses this storage across epochs. Binary IPC already supplies float32
    arrays, so action storage can be reused without a Python-number conversion.
    Each batch owns its arrays until autograd has finished; no reusable buffer
    can overwrite inputs retained by an earlier forward pass.
    """

    def __init__(self, observations):
        if not observations or any(
            o["version"] != ENCODER_VERSION for o in observations
        ):
            raise ValueError("unsupported or empty observation batch")
        self.states = np.asarray(
            [o["features"] for o in observations], dtype=np.float32
        )
        if self.states.shape != (len(observations), STATE_SIZE):
            raise ValueError("invalid state features")
        self.signals = np.asarray([o["kind"] == "signal" for o in observations])
        self.signal_masks = None
        if any("signalMask" in o for o in observations):
            supplied = np.asarray(
                [o.get("signalMask", [True] * 6) for o in observations]
            )
            if (
                supplied.shape != (len(observations), 6)
                or supplied.dtype != np.bool_
                or not supplied[:, 0].all()
            ):
                raise ValueError("invalid truthful signal masks")
            self.signal_masks = supplied
        self.actions = [
            np.asarray(o["actions"], dtype=np.float32).reshape(-1, ACTION_SIZE)
            for o in observations
        ]
        self.counts = np.asarray(
            [
                6 if signal else len(actions)
                for signal, actions in zip(self.signals, self.actions)
            ],
            dtype=np.int64,
        )
        if np.any(self.counts < 1):
            raise ValueError("nonterminal observation without legal actions")

    def distribution(self, model, indices=None, device="cpu"):
        if indices is None:
            indices = np.arange(len(self.states))
        else:
            indices = np.asarray(indices, dtype=np.int64)
        counts = self.counts[indices]
        maximum = max(6, int(counts.max()))
        candidates = np.zeros((len(indices), maximum, ACTION_SIZE), dtype=np.float32)
        for row, index in enumerate(indices):
            if not self.signals[index]:
                candidates[row, : counts[row]] = self.actions[index]
        mask = torch.from_numpy(np.arange(maximum)[None, :] < counts[:, None]).to(
            device
        )
        signals = torch.from_numpy(self.signals[indices]).to(device)
        if self.signal_masks is not None:
            legal_signals = torch.from_numpy(self.signal_masks[indices]).to(device)
            mask[:, :6] &= ~signals[:, None] | legal_signals
        states = torch.from_numpy(self.states[indices]).to(device)
        game, signal, value = model(states, torch.from_numpy(candidates).to(device))
        signal = torch.nn.functional.pad(signal, (0, maximum - 6))
        logits = torch.where(signals[:, None], signal, game).masked_fill(
            ~mask, -torch.inf
        )
        return torch.distributions.Categorical(logits=logits), value


def batch_distribution(model, observations, device="cpu"):
    return PackedObservations(observations).distribution(model, device=device)
