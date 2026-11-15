"""Explicit capacity limits and function-preserving recurrent warm starts."""

import torch

from .recurrent import TEAM_ADAPTER_VERSION, Actor, Critic

MAX_ACTOR_PARAMETERS = 150000
MAX_MODEL_BYTES = 128 * 1024
MEMORY_SIZES = (16, 32, 64)
LARGE_CRITIC = (256, 128)


def team_adapter_configuration(state):
    """Internal weight-only copies retain a validated, fingerprinted mode."""
    marker = state.get("team_adapter_mode")
    has_head = any(k.startswith("team_head.") for k in state)
    if marker is None:
        if has_head:
            raise ValueError("team adapter weights lack mode marker")
        return {}
    if (
        not has_head
        or marker.shape != torch.Size([])
        or marker.dtype != torch.float32
        or float(marker) not in (1.0, 2.0)
    ):
        raise ValueError("invalid team adapter mode marker")
    return {
        "team_adapter": "residual" if float(marker) == 1.0 else "contextual",
        "team_adapter_version": TEAM_ADAPTER_VERSION,
    }


def actor_from_checkpoint(saved):
    state = saved["model"]
    recurrent = "memory.weight_hh" in state
    size = int(state["memory.weight_hh"].shape[1]) if recurrent else 16
    declared = saved.get("configuration", {}).get("actor_memory", size)
    if type(declared) is not int or declared != size:
        raise ValueError("actor memory metadata differs from weights")
    configuration = saved.get("configuration", {})
    mode = configuration.get("team_adapter")
    version = configuration.get("team_adapter_version")
    derived = team_adapter_configuration(state)
    if mode != derived.get("team_adapter") or version != derived.get(
        "team_adapter_version"
    ):
        raise ValueError("team adapter metadata missing or inconsistent")
    if mode is not None and type(version) is not int:
        raise ValueError("unsupported team adapter version")
    actor = Actor(recurrent, memory_size=size, team_adapter=mode)
    actor.load_state_dict(state, strict=True)
    if sum(p.numel() for p in actor.parameters()) > MAX_ACTOR_PARAMETERS:
        raise ValueError("actor exceeds parameter budget")
    if any(not torch.isfinite(p).all() for p in actor.parameters()):
        raise ValueError("nonfinite actor weights")
    return actor


def critic_from_checkpoint(saved):
    state = saved["critic"]
    hidden = (
        int(state["layers.0.weight"].shape[0]),
        int(state["layers.2.weight"].shape[0]),
    )
    if tuple(saved.get("configuration", {}).get("critic_hidden", hidden)) != hidden:
        raise ValueError("critic metadata differs from weights")
    critic = Critic(True, hidden)
    critic.load_state_dict(state, strict=True)
    if any(not torch.isfinite(p).all() for p in critic.parameters()):
        raise ValueError("nonfinite critic weights")
    return critic


def widen_actor(actor, memory_size):
    if not actor.recurrent:
        raise ValueError("recurrent warm start required")
    old = actor.memory.hidden_size
    if memory_size < old:
        raise ValueError("warm start cannot discard memory units")
    if memory_size == old:
        return actor
    if actor.team_adapter is not None:
        raise ValueError("widen actor before enabling team adapter")
    result = Actor(True, memory_size)
    # New units retain random, trainable activations; old units cannot depend
    # on them until learning changes the initially zero cross-connections.
    with torch.no_grad():
        for name in ("state", "action", "score", "signal"):
            getattr(result, name).load_state_dict(getattr(actor, name).state_dict())
        for gate in range(4):
            a, b = (
                slice(gate * old, (gate + 1) * old),
                slice(gate * memory_size, gate * memory_size + old),
            )
            result.memory.weight_ih[b].copy_(actor.memory.weight_ih[a])
            result.memory.weight_hh[b].zero_()
            result.memory.weight_hh[b, :old].copy_(actor.memory.weight_hh[a])
            result.memory.bias_ih[b].copy_(actor.memory.bias_ih[a])
            result.memory.bias_hh[b].copy_(actor.memory.bias_hh[a])
        result.projection.weight.zero_()
        result.projection.weight[:, :old].copy_(actor.projection.weight)
        result.projection.bias.copy_(actor.projection.bias)
    return result


def widen_critic(critic, hidden):
    hidden = tuple(hidden)
    old = (critic.layers[0].out_features, critic.layers[2].out_features)
    if hidden == old:
        return critic
    if any(a < b for a, b in zip(hidden, old, strict=True)):
        raise ValueError("warm start cannot discard critic units")
    result = Critic(critic.centralized, hidden)
    with torch.no_grad():
        result.layers[0].weight[: old[0]].copy_(critic.layers[0].weight)
        result.layers[0].bias[: old[0]].copy_(critic.layers[0].bias)
        result.layers[2].weight[: old[1]].zero_()
        result.layers[2].weight[: old[1], : old[0]].copy_(critic.layers[2].weight)
        result.layers[2].bias[: old[1]].copy_(critic.layers[2].bias)
        result.layers[4].weight.zero_()
        result.layers[4].weight[:, : old[1]].copy_(critic.layers[4].weight)
        result.layers[4].bias.copy_(critic.layers[4].bias)
    return result
