"""Experimental actors and CTDE critics. Privileged tensors never enter actors."""

import hashlib

import numpy as np
import torch
from torch import nn

from .model import ACTION_SIZE, ENCODER_VERSION, Policy

MEMORY_VERSION = 1
CHECKPOINT_KIND = "controlled-mappo-v1"
TEAM_ADAPTER_VERSION = 1
ARMS = {"A": (False, False), "B": (True, False), "C": (False, True), "D": (True, True)}


class Actor(nn.Module):
    def __init__(self, recurrent=False, memory_size=16, team_adapter=None):
        super().__init__()
        if type(memory_size) is not int or memory_size not in (16, 32, 64):
            raise ValueError("unsupported actor memory size")
        base = Policy()
        self.state, self.action, self.score, self.signal = (
            base.state,
            base.action,
            base.score,
            base.signal,
        )
        self.recurrent = recurrent
        self.candidate_features = 0
        self.team_adapter = None
        if recurrent:
            self.memory = nn.LSTMCell(64, memory_size)
            self.projection = nn.Linear(memory_size, 64)
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        if team_adapter is not None:
            self.enable_team_adapter(team_adapter)

    def enable_team_adapter(self, mode):
        """Enable an experimental 4-player specialist on a zero-projection base."""
        if mode not in ("residual", "contextual"):
            raise ValueError("unsupported team adapter mode")
        if self.team_adapter is not None or not self.recurrent:
            raise ValueError("team adapter requires an unadapted recurrent actor")
        if torch.count_nonzero(self.projection.weight) or torch.count_nonzero(
            self.projection.bias
        ):
            raise ValueError("team adapter requires zero parent projection")
        self.team_adapter = mode
        self.register_buffer(
            "team_adapter_mode",
            self.projection.weight.new_tensor(1.0 if mode == "residual" else 2.0),
        )
        self.team_head = nn.Sequential(nn.Linear(128, 32), nn.Tanh(), nn.Linear(32, 1))
        self.team_head.to(
            device=self.projection.weight.device, dtype=self.projection.weight.dtype
        )
        nn.init.zeros_(self.team_head[2].weight)
        nn.init.zeros_(self.team_head[2].bias)
        for module in (self.state, self.action, self.score, self.signal):
            module.requires_grad_(False)
        return self

    def projected_context(self, encoded, features, hidden):
        if not self.recurrent:
            return encoded
        projection = self.projection(hidden)
        if self.team_adapter is not None:
            projection = torch.where(
                features[..., :1] == 1, projection, torch.zeros_like(projection)
            )
        return encoded + projection

    def warm_start(self, state):
        expected = {
            k for k in self.state_dict() if not k.startswith(("memory.", "projection."))
        }
        if expected != {k for k in state if not k.startswith("value.")}:
            raise ValueError("warm-start requires an exact legacy actor architecture")
        self.load_state_dict(
            {**self.state_dict(), **{k: state[k] for k in expected}}, strict=True
        )

    def initial(self):
        p = next(self.parameters())
        size = self.memory.hidden_size if self.recurrent else 16
        return (p.new_zeros(size), p.new_zeros(size))

    def advance(self, features, memory):
        z = self.state(features)
        return self.memory(z, memory) if self.recurrent else memory

    def context(self, features, memory):
        z = self.state(features)
        return self.projected_context(z, features, memory[0])

    def game_logits(self, contexts, actions, features=None):
        # Trained scores can have a large common offset. Evaluate only the
        # small scoring head in double precision so batch-shaped GEMMs do not
        # quantize meaningful action differences before normalization. Casts
        # remain differentiable; stored parameters/optimizer states stay FP32.
        if self.candidate_features:
            from .candidate_features import augment

            actions = augment(features, actions)
        encoded = torch.nn.functional.linear(
            actions.double(),
            self.action[0].weight.double(),
            self.action[0].bias.double(),
        ).relu()
        hidden = torch.nn.functional.linear(
            torch.cat((contexts.double(), encoded), -1),
            self.score[0].weight.double(),
            self.score[0].bias.double(),
        ).relu()
        logits = torch.nn.functional.linear(
            hidden, self.score[2].weight.double(), self.score[2].bias.double()
        ).squeeze(-1)
        if self.team_adapter is not None:
            if features is None or features.shape != (*actions.shape[:-1], 768):
                raise ValueError("team adapter requires aligned authorized features")
            from .team_features import contextual_card_features

            extra = (
                contextual_card_features(features, actions)
                if self.team_adapter == "contextual"
                else actions.new_zeros((*actions.shape[:-1], 16))
            )
            inputs = torch.cat(
                (contexts.double().tanh(), actions.double(), extra.double()), -1
            )
            residual_hidden = torch.nn.functional.linear(
                inputs,
                self.team_head[0].weight.double(),
                self.team_head[0].bias.double(),
            ).tanh()
            residual = torch.nn.functional.linear(
                residual_hidden,
                self.team_head[2].weight.double(),
                self.team_head[2].bias.double(),
            ).squeeze(-1)
            logits = logits + torch.where(
                features[..., 0] == 1, residual, torch.zeros_like(residual)
            )
        return logits

    def signal_logits(self, contexts):
        return torch.nn.functional.linear(
            contexts.double(), self.signal.weight.double(), self.signal.bias.double()
        )

    def distribution(self, context, observation):
        if observation["version"] != ENCODER_VERSION:
            raise ValueError("actor encoder mismatch")
        if observation["kind"] == "signal":
            logits = self.signal_logits(context)
            if "signalMask" in observation:
                mask = torch.as_tensor(
                    observation["signalMask"], dtype=torch.bool, device=logits.device
                )
                if mask.shape != (6,) or not mask[0]:
                    raise ValueError("invalid truthful signal mask")
                logits = logits.masked_fill(~mask, -torch.inf)
        else:
            actions = torch.as_tensor(
                np.asarray(observation["actions"], np.float32).reshape(-1, ACTION_SIZE),
                device=context.device,
            )
            if not 1 <= len(actions) <= 1024:
                raise ValueError("invalid legal candidate count")
            features = None
            if self.team_adapter is not None or self.candidate_features:
                features = torch.as_tensor(
                    np.asarray(observation["features"], np.float32),
                    device=context.device,
                ).expand(len(actions), -1)
            logits = self.game_logits(
                context.expand(len(actions), -1), actions, features
            )
        return centered_categorical(logits)


def centered_categorical(logits):
    # Center before logsumexp subtracts a large absolute score. Masked -inf
    # candidates remain masked and all probability computations stay FP64.
    return torch.distributions.Categorical(
        logits=logits - logits.amax(-1, keepdim=True)
    )


class Critic(nn.Module):
    """Identical topology in every arm; local arms mask the privileged channel."""

    def __init__(self, centralized=False, hidden=(96, 64)):
        super().__init__()
        if tuple(hidden) not in ((96, 64), (256, 128)):
            raise ValueError("unsupported critic architecture")
        self.centralized = centralized
        self.layers = nn.Sequential(
            nn.Linear(768 + 128 + 64, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, features, privileged, actor_context):
        private = privileged if self.centralized else torch.zeros_like(privileged)
        return self.layers(
            torch.cat((features, private, actor_context.detach()), -1)
        ).squeeze(-1)


class Replay:
    """Exactly-once authorized hand replay. Never resets silently mid-hand."""

    def __init__(self, actor):
        self.actor = actor
        self.hand = None
        self.events = []
        self.memory = actor.initial()

    def prepare(self, page):
        if page.get("version") != MEMORY_VERSION or not page.get("complete"):
            raise ValueError("incomplete or incompatible memory journal")
        if (
            type(page.get("hand")) is not int
            or page["hand"] < 1
            or (self.hand is not None and page["hand"] < self.hand)
        ):
            raise ValueError("obsolete or invalid memory hand")
        if page["hand"] != self.hand:
            if page["start"] != 0:
                raise ValueError("hand replay must start at zero")
            self.hand, self.events, self.memory = page["hand"], [], self.actor.initial()
        events = np.asarray(page["events"], np.float32).reshape(-1, 768)
        start, end, total = page["start"], page["end"], page["total"]
        if (
            not (0 <= start <= end <= total <= 2048)
            or end - start != len(events)
            or not np.isfinite(events).all()
        ):
            raise ValueError("invalid memory page")
        if total < len(self.events):
            raise ValueError("obsolete memory cursor")
        if end <= len(self.events):
            if len(events) and not np.array_equal(
                np.asarray(self.events[start:end]), events
            ):
                raise ValueError("conflicting memory replay")
            return events[:0]
        if start != len(self.events):
            raise ValueError("memory cursor gap")
        return events

    def apply(self, page):
        events = self.prepare(page)
        with torch.no_grad():
            for event in events:
                self.memory = self.actor.advance(torch.as_tensor(event), self.memory)
                self.events.append(event.copy())


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
