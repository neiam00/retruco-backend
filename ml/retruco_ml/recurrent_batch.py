"""Batched recurrent PPO with the original hand-prefix/BPTT semantics.

Only numeric features are cached across optimizer steps. Hidden states and
encodings are recomputed with current weights, never reused after an update.
"""

import numpy as np
import torch

from .recurrent import centered_categorical, limit_opening_bluff


class PackedRecords:
    def __init__(self, records, sequence=32):
        self.records = records
        self.sequence = sequence
        self.events = {}
        self.chunks = {}
        self.features = torch.from_numpy(
            np.asarray([r.observation["features"] for r in records], np.float32)
        )
        self.privileged = torch.from_numpy(
            np.asarray([r.privileged for r in records], np.float32)
        )
        self.actions = torch.tensor([r.action for r in records])
        self.old = torch.tensor([r.logp for r in records])
        self.targets = torch.tensor([r.target for r in records])
        advantage = torch.tensor([r.advantage for r in records])
        self.advantages = (advantage - advantage.mean()) / (
            advantage.std(unbiased=False) + 1e-8
        )
        self.candidates = []
        for i, r in enumerate(records):
            identity = id(r.events)
            if identity not in self.events:
                self.events[identity] = torch.from_numpy(
                    np.asarray(r.events, np.float32)
                )
            if not torch.equal(self.events[identity][r.end - 1], self.features[i]):
                raise ValueError("decision differs from authorized journal")
            self.chunks.setdefault((identity, (r.end - 1) // sequence), []).append(i)
            self.candidates.append(
                torch.from_numpy(
                    np.asarray(r.observation["actions"], np.float32).reshape(-1, 48)
                )
            )
        self.chunks = list(self.chunks.values())

    def contexts(self, actor, chunks):
        indices = [i for chunk in chunks for i in chunk]
        if not actor.recurrent:
            return actor.state(self.features[indices]), indices
        size = len(chunks)
        sequence = self.sequence
        starts = [((self.records[c[0]].end - 1) // sequence) * sequence for c in chunks]
        lengths = [
            min(sequence, len(self.events[id(self.records[c[0]].events)]) - start)
            for c, start in zip(chunks, starts, strict=True)
        ]
        prefix = torch.zeros(size, max(starts), 768)
        window = torch.zeros(size, sequence, 768)
        for row, (chunk, start, length) in enumerate(
            zip(chunks, starts, lengths, strict=True)
        ):
            events = self.events[id(self.records[chunk[0]].events)]
            prefix[row, :start] = events[:start]
            window[row, :length] = events[start : start + length]
        h = torch.zeros(size, actor.memory.hidden_size)
        c = torch.zeros_like(h)
        with torch.no_grad():
            encoded = actor.state(prefix.reshape(-1, 768)).reshape(
                size, max(starts), 64
            )
            for step in range(max(starts)):
                nh, nc = actor.memory(encoded[:, step], (h, c))
                valid = torch.tensor([step < s for s in starts])[:, None]
                h, c = torch.where(valid, nh, h), torch.where(valid, nc, c)
        encoded = actor.state(window.reshape(-1, 768)).reshape(size, sequence, 64)
        contexts = []
        wanted = {}
        for row, (chunk, start) in enumerate(zip(chunks, starts, strict=True)):
            for i in chunk:
                wanted[i] = (row, self.records[i].end - start - 1)
        hs = []
        for step in range(max(lengths)):
            nh, nc = actor.memory(encoded[:, step], (h, c))
            valid = torch.tensor([step < n for n in lengths])[:, None]
            h, c = torch.where(valid, nh, h), torch.where(valid, nc, c)
            hs.append(h)
        # A decision's direct encoder path and recurrent path share the same
        # feature encoding; autograd accumulates both contributions normally.
        for i in indices:
            row, step = wanted[i]
            contexts.append(
                actor.projected_context(
                    encoded[row, step], self.features[i], hs[step][row]
                )
            )
        return torch.stack(contexts), indices

    def distribution(self, actor, contexts, indices):
        return candidate_distribution(
            actor,
            contexts,
            [self.records[i].observation for i in indices],
            [self.candidates[i] for i in indices],
        )


def candidate_distribution(actor, contexts, observations, candidates=None):
    if candidates is None:
        candidates = [
            torch.from_numpy(np.asarray(o["actions"], np.float32).reshape(-1, 48))
            for o in observations
        ]
    counts = [
        6 if o["kind"] == "signal" else len(candidates[i])
        for i, o in enumerate(observations)
    ]
    maximum = max(counts)
    logits = contexts.new_full(
        (len(observations), maximum), -torch.inf, dtype=torch.float64
    )
    game_rows = [row for row, o in enumerate(observations) if o["kind"] != "signal"]
    signal_rows = [row for row, o in enumerate(observations) if o["kind"] == "signal"]
    if game_rows:
        packed_candidates = torch.cat([candidates[row] for row in game_rows])
        rows = torch.repeat_interleave(
            torch.tensor(game_rows), torch.tensor([counts[row] for row in game_rows])
        )
        columns = torch.cat([torch.arange(counts[row]) for row in game_rows])
        features = None
        if actor.team_adapter is not None or getattr(actor, "candidate_features", 0):
            features = torch.as_tensor(
                np.asarray([o["features"] for o in observations], np.float32),
                device=contexts.device,
            )[rows]
        scores = actor.game_logits(contexts[rows], packed_candidates, features)
        logits[rows, columns] = scores
        marked = packed_candidates[:, 47] == -1
        if marked.any():
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[rows, columns] = marked
            logits[game_rows] = limit_opening_bluff(logits[game_rows], mask[game_rows])
    if signal_rows:
        signals = actor.signal_logits(contexts[signal_rows])
        masks = torch.tensor(
            [observations[row].get("signalMask", [True] * 6) for row in signal_rows],
            dtype=torch.bool,
        )
        if not masks[:, 0].all():
            raise ValueError("none must remain legal")
        logits[signal_rows, :6] = signals.masked_fill(~masks, -torch.inf)
    return centered_categorical(logits)


def update(
    actor,
    critic,
    optimizer,
    records,
    rng,
    epochs=4,
    batch_size=512,
    sequence=32,
    monitor=None,
    policy_weight=1.0,
):
    packed = PackedRecords(records, sequence)
    chunks = packed.chunks
    last = 0.0
    for _ in range(epochs):
        rng.shuffle(chunks)
        for start in range(0, len(chunks), max(1, batch_size // sequence)):
            if monitor:
                monitor.check()
            contexts, indices = packed.contexts(
                actor, chunks[start : start + max(1, batch_size // sequence)]
            )
            distribution = packed.distribution(actor, contexts, indices)
            ratio = (
                distribution.log_prob(packed.actions[indices]) - packed.old[indices]
            ).exp()
            advantage = packed.advantages[indices]
            policy = -torch.minimum(
                ratio * advantage, ratio.clamp(0.8, 1.2) * advantage
            ).mean()
            values = critic(
                packed.features[indices], packed.privileged[indices], contexts
            )
            loss = (
                policy_weight * (policy - 0.01 * distribution.entropy().mean())
                + 0.5 * (values - packed.targets[indices]).square().mean()
            )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite batched PPO loss")
            optimizer.zero_grad()
            loss.backward()
            if policy_weight == 0:
                for parameter in actor.parameters():
                    parameter.grad = None
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            optimizer.step()
            last = float(loss.detach())
    return last
