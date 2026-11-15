"""Fresh recurrent MAPPO networks with separate execution and training inputs.

The deployed actor only accepts the 768 authorized observation features.  The
critic consumes those features plus 128 privileged features and never supplies
an actor context.  Both use (h, c) state tuples for the existing journal API;
for GRU, c is a zero compatibility placeholder, not a second recurrent state.
"""

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .recurrent import Actor

CHECKPOINT_KIND = "rmappo-v1"
LOCAL_SIZE = 768
PRIVILEGED_SIZE = 128
ACTOR_MEMORY_SIZE = 64
CRITIC_MEMORY_SIZE = 128


def _cell_name(cell):
    if cell not in ("gru", "lstm"):
        raise ValueError("RMAPPO cell must be 'gru' or 'lstm'")
    return cell


def _mask(mask, shape, device, name):
    if mask is None:
        return None
    if mask.shape != shape or mask.dtype != torch.bool or mask.device != device:
        raise ValueError(
            f"{name} must be a boolean tensor with shape {tuple(shape)} on {device}"
        )
    return mask


def _select_state(mask, selected, other):
    return tuple(torch.where(mask.unsqueeze(-1), a, b) for a, b in zip(selected, other))


def detach_state(state):
    """Cut a TBPTT boundary without modifying either recurrent component."""
    return tuple(part.detach() for part in state)


class _Memory(nn.Module):
    """A sequence kernel with the legacy cell-call and tuple-state interface."""

    def __init__(self, input_size, hidden_size, cell):
        super().__init__()
        self.cell = _cell_name(cell)
        self.hidden_size = hidden_size
        cls = nn.GRU if cell == "gru" else nn.LSTM
        self.core = cls(input_size, hidden_size, batch_first=True)

    def _native(self, state):
        h, c = (part.unsqueeze(0) for part in state)
        return h if self.cell == "gru" else (h, c)

    def _tuple(self, state):
        if self.cell == "gru":
            h = state.squeeze(0)
            return h, torch.zeros_like(h)
        return tuple(part.squeeze(0) for part in state)

    def forward(self, encoded, state):
        # A direct cell avoids launching a sequence backend for each collector
        # event, while sharing the exact parameters used by fused replay.
        h, c = state
        inputs = nn.functional.linear(
            encoded, self.core.weight_ih_l0, self.core.bias_ih_l0
        )
        hidden = nn.functional.linear(h, self.core.weight_hh_l0, self.core.bias_hh_l0)
        if self.cell == "gru":
            ir, iz, candidate = inputs.chunk(3, dim=-1)
            hr, hz, hn = hidden.chunk(3, dim=-1)
            reset, update = (ir + hr).sigmoid(), (iz + hz).sigmoid()
            candidate = (candidate + reset * hn).tanh()
            h = (1 - update) * candidate + update * h
            return h, torch.zeros_like(h)
        input_gate, forget_gate, candidate, output_gate = (inputs + hidden).chunk(
            4, dim=-1
        )
        c = forget_gate.sigmoid() * c + input_gate.sigmoid() * candidate.tanh()
        return output_gate.sigmoid() * c.tanh(), c

    def replay(self, encoded, state, valid=None, reset=None):
        batch, length, _ = encoded.shape
        valid = _mask(valid, encoded.shape[:2], encoded.device, "valid")
        reset = _mask(reset, encoded.shape[:2], encoded.device, "reset")
        if length == 0:
            return encoded.new_zeros(batch, 0, self.hidden_size), state
        if reset is None and valid is None:
            output, result = self.core(encoded, self._native(state))
            return output, self._tuple(result)
        # Right-padded batches use the fused sequence kernel. Empty rows receive
        # a dummy step internally, but neither its output nor state is exposed.
        if reset is None and not bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            lengths = valid.sum(1)
            packed = pack_padded_sequence(
                encoded,
                lengths.clamp_min(1).cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            output, result = self.core(packed, self._native(state))
            output, _ = pad_packed_sequence(
                output, batch_first=True, total_length=length
            )
            result = _select_state(lengths > 0, self._tuple(result), state)
            return output.masked_fill(~valid.unsqueeze(-1), 0), result
        # Interior gaps and hand resets require explicit state transitions.
        # The encoder is still batched; only the small recurrent cell is looped.
        outputs = []
        zero = tuple(torch.zeros_like(part) for part in state)
        for index in range(length):
            active = valid[:, index] if valid is not None else None
            previous = state
            if reset is not None:
                state = _select_state(reset[:, index], zero, state)
            state = self(encoded[:, index], state)
            if active is not None:
                state = _select_state(active, state, previous)
                outputs.append(state[0].masked_fill(~active.unsqueeze(-1), 0))
            else:
                outputs.append(state[0])
        return torch.stack(outputs, dim=1), state


def _initial(module, size, batch_size):
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    shape = (size,) if batch_size is None else (batch_size, size)
    parameter = next(module.parameters())
    return parameter.new_zeros(shape), parameter.new_zeros(shape)


def _check_state(state, leading_shape, size, reference):
    shape = (*leading_shape, size)
    if not isinstance(state, tuple) or len(state) != 2:
        raise ValueError("recurrent state must be an (h, c) tuple")
    if any(
        part.shape != shape
        or part.device != reference.device
        or part.dtype != reference.dtype
        for part in state
    ):
        raise ValueError(
            f"recurrent state components must match input device/dtype and shape {shape}"
        )


class RecurrentActor(Actor):
    """Shared GRU64/LSTM64 policy, initialized from scratch, under 150k params.

    Journal replay advances memory only on journal events. A decision calls
    context(local, memory) without advancing it again. step/replay are convenient
    event-sequence APIs and return contexts *after* each event is consumed.
    Legal action and truthful signal masks remain the caller's responsibility
    (the inherited observation distribution applies its supplied signal mask).
    """

    def __init__(self, cell="gru", candidate_features=0):
        cell = _cell_name(cell)
        super().__init__(recurrent=True, memory_size=ACTOR_MEMORY_SIZE)
        self.cell = cell
        self.memory = _Memory(64, ACTOR_MEMORY_SIZE, cell)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        if candidate_features:
            self.enable_candidate_features(candidate_features)

    def enable_candidate_features(self, version=1):
        if version != 1 or self.candidate_features:
            raise ValueError("unsupported or already enabled candidate features")
        # Raw action-v1 columns 36..47 are always zero. Resetting their weights
        # and deriving public features there preserves every initial logit.
        with torch.no_grad():
            self.action[0].weight[:, 36:] = 0
        self.candidate_features = version
        return self

    def initial(self, batch_size=None):
        return _initial(self, ACTOR_MEMORY_SIZE, batch_size)

    def encode(self, features):
        if features.shape[-1] != LOCAL_SIZE:
            raise ValueError("actor accepts only 768 authorized local features")
        return self.state(features)

    def advance(self, features, memory):
        if features.ndim not in (1, 2):
            raise ValueError("advance expects one local event or a batch of events")
        _check_state(memory, features.shape[:-1], ACTOR_MEMORY_SIZE, features)
        return self.memory(self.encode(features), memory)

    def context(self, features, memory):
        _check_state(memory, features.shape[:-1], ACTOR_MEMORY_SIZE, features)
        return self.projected_context(self.encode(features), features, memory[0])

    def step(self, features, state=None, *, valid=None, reset=None):
        if features.ndim not in (1, 2):
            raise ValueError("step expects local features shaped [768] or [B,768]")
        if state is None:
            state = self.initial(None if features.ndim == 1 else features.shape[0])
        _check_state(state, features.shape[:-1], ACTOR_MEMORY_SIZE, features)
        valid = _mask(valid, features.shape[:-1], features.device, "valid")
        reset = _mask(reset, features.shape[:-1], features.device, "reset")
        previous = state
        if reset is not None:
            state = _select_state(
                reset, tuple(torch.zeros_like(p) for p in state), state
            )
        clean = (
            features if valid is None else features.masked_fill(~valid.unsqueeze(-1), 0)
        )
        encoded = self.encode(clean)
        state = self.memory(encoded, state)
        if valid is not None:
            state = _select_state(valid, state, previous)
        context = self.projected_context(encoded, clean, state[0])
        if valid is not None:
            context = context.masked_fill(~valid.unsqueeze(-1), 0)
        return context, state

    def replay(self, features, state=None, *, valid=None, reset=None):
        if features.ndim != 3:
            raise ValueError("replay expects local features shaped [B,T,768]")
        if state is None:
            state = self.initial(features.shape[0])
        _check_state(state, features.shape[:1], ACTOR_MEMORY_SIZE, features)
        valid = _mask(valid, features.shape[:2], features.device, "valid")
        clean = (
            features if valid is None else features.masked_fill(~valid.unsqueeze(-1), 0)
        )
        encoded = self.encode(clean)
        hidden, state = self.memory.replay(encoded, state, valid, reset)
        contexts = self.projected_context(encoded, clean, hidden)
        if valid is not None:
            contexts = contexts.masked_fill(~valid.unsqueeze(-1), 0)
        return contexts, state

    @torch.no_grad()
    def recompute_prefix(self, features, state=None, *, valid=None, reset=None):
        """Recompute a burn-in prefix with current weights before each PPO pass."""
        return detach_state(self.replay(features, state, valid=valid, reset=reset)[1])

    def warm_start(self, state):
        raise ValueError(
            "RMAPPO actors start from scratch; load an rmappo-v1 checkpoint to resume"
        )

    def architecture(self):
        return {
            "cell": self.cell,
            "local_size": LOCAL_SIZE,
            "memory_size": ACTOR_MEMORY_SIZE,
            "candidate_features": self.candidate_features,
        }


class RecurrentCritic(nn.Module):
    """Centralized recurrent value network with no actor parameter dependency."""

    def __init__(self, cell="gru"):
        super().__init__()
        self.cell = _cell_name(cell)
        self.state = nn.Sequential(
            nn.Linear(LOCAL_SIZE + PRIVILEGED_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.memory = _Memory(128, CRITIC_MEMORY_SIZE, cell)
        self.value = nn.Linear(CRITIC_MEMORY_SIZE, 1)

    def initial(self, batch_size=None):
        return _initial(self, CRITIC_MEMORY_SIZE, batch_size)

    def _inputs(self, local, privileged):
        if (
            local.shape[-1] != LOCAL_SIZE
            or privileged.shape[-1] != PRIVILEGED_SIZE
            or local.shape[:-1] != privileged.shape[:-1]
        ):
            raise ValueError(
                "critic requires aligned local[...,768] and privileged[...,128] features"
            )
        return torch.cat((local, privileged), dim=-1)

    def step(self, local, privileged, state=None, *, valid=None, reset=None):
        if local.ndim not in (1, 2):
            raise ValueError("step expects local features shaped [768] or [B,768]")
        if state is None:
            state = self.initial(None if local.ndim == 1 else local.shape[0])
        _check_state(state, local.shape[:-1], CRITIC_MEMORY_SIZE, local)
        valid = _mask(valid, local.shape[:-1], local.device, "valid")
        reset = _mask(reset, local.shape[:-1], local.device, "reset")
        inputs = self._inputs(local, privileged)
        if valid is not None:
            inputs = inputs.masked_fill(~valid.unsqueeze(-1), 0)
        previous = state
        if reset is not None:
            state = _select_state(
                reset, tuple(torch.zeros_like(p) for p in state), state
            )
        state = self.memory(self.state(inputs), state)
        if valid is not None:
            state = _select_state(valid, state, previous)
        value = self.value(state[0]).squeeze(-1)
        return (value if valid is None else value.masked_fill(~valid, 0)), state

    def replay(self, local, privileged, state=None, *, valid=None, reset=None):
        if local.ndim != 3:
            raise ValueError("replay expects local features shaped [B,T,768]")
        if state is None:
            state = self.initial(local.shape[0])
        _check_state(state, local.shape[:1], CRITIC_MEMORY_SIZE, local)
        valid = _mask(valid, local.shape[:2], local.device, "valid")
        inputs = self._inputs(local, privileged)
        if valid is not None:
            inputs = inputs.masked_fill(~valid.unsqueeze(-1), 0)
        hidden, state = self.memory.replay(self.state(inputs), state, valid, reset)
        values = self.value(hidden).squeeze(-1)
        return (values if valid is None else values.masked_fill(~valid, 0)), state

    @torch.no_grad()
    def recompute_prefix(
        self, local, privileged, state=None, *, valid=None, reset=None
    ):
        return detach_state(
            self.replay(local, privileged, state, valid=valid, reset=reset)[1]
        )

    def architecture(self):
        return {
            "cell": self.cell,
            "local_size": LOCAL_SIZE,
            "privileged_size": PRIVILEGED_SIZE,
            "memory_size": CRITIC_MEMORY_SIZE,
        }


def _checkpoint_cell(saved):
    if not isinstance(saved, Mapping) or saved.get("kind") != CHECKPOINT_KIND:
        raise ValueError("expected an rmappo-v1 checkpoint")
    configuration = saved.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("RMAPPO checkpoint requires configuration.cell")
    return _cell_name(configuration.get("cell"))


def actor_from_checkpoint(saved):
    """Reconstruct actor architecture explicitly; legacy checkpoints are rejected."""
    version = saved.get("configuration", {}).get("candidate_features", 0)
    if type(version) is not int or version not in (0, 1):
        raise ValueError("unsupported candidate feature version")
    actor = RecurrentActor(_checkpoint_cell(saved), version)
    actor.load_state_dict(saved["actor"], strict=True)
    return actor


def critic_from_checkpoint(saved):
    critic = RecurrentCritic(_checkpoint_cell(saved))
    critic.load_state_dict(saved["critic"], strict=True)
    return critic
