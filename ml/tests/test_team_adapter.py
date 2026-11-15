import copy
import io
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from retruco_ml.cooperation_rollout import advance_group
from retruco_ml.decision_ppo import update
from retruco_ml.historical_league import HistoricalLeaguePopulation
from retruco_ml.mappo_export import export
from retruco_ml.mappo_rollout import Decision
from retruco_ml.model import Policy
from retruco_ml.model_capacity import (
    actor_from_checkpoint,
    team_adapter_configuration,
    widen_actor,
)
from retruco_ml.recurrent import Actor, Critic, Replay
from retruco_ml.recurrent_batch import PackedRecords, candidate_distribution
from retruco_ml.team_features import contextual_card_features


def observation(players=4, strengths=(0.2, 0.5, 0.8), signal=False, round_index=0):
    f = np.zeros(768, np.float32)
    f[0], f[3], f[4] = players / 4, 0.5, round_index / 2
    for index, strength in enumerate(strengths):
        f[21 + index * 5 : 26 + index * 5] = [1, 0.25, 0.5, strength, 0]
    for offset in range(players):
        f[39 + 22 * offset] = 1
    actions = np.zeros((3, 48), np.float32)
    actions[:, [1, 17, 24, 27]] = 1
    actions[:, 32] = [0, 0.5, 1]
    return {
        "version": 2,
        "kind": "signal" if signal else "game",
        "features": f,
        "actions": [] if signal else actions,
        "signalMask": [True, False, True, False, False, False],
        "seat": 0,
    }


def scalar(actor, obs, events=None):
    memory = actor.initial()
    for event in events if events is not None else [obs["features"]]:
        memory = actor.advance(torch.as_tensor(event), memory)
    return actor.distribution(
        actor.context(torch.as_tensor(obs["features"]), memory), obs
    )


class TeamAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(29)

    def test_zero_initialization_preserves_parent_and_enforces_safe_enable(self):
        parent = Actor(True, 64)
        for mode in ("residual", "contextual"):
            actor = copy.deepcopy(parent).enable_team_adapter(mode)
            self.assertLess(sum(p.numel() for p in actor.parameters()), 150000)
            self.assertEqual(
                float(actor.team_adapter_mode), 1 if mode == "residual" else 2
            )
            for players in (2, 4):
                for signal in (False, True):
                    obs = observation(players, signal=signal)
                    torch.testing.assert_close(
                        scalar(parent, obs).probs,
                        scalar(actor, obs).probs,
                        atol=0,
                        rtol=0,
                    )
            for name, p in actor.named_parameters():
                if name.startswith(("state.", "action.", "score.", "signal.")):
                    self.assertFalse(p.requires_grad)
            with self.assertRaisesRegex(ValueError, "unadapted"):
                actor.enable_team_adapter(mode)
            with self.assertRaisesRegex(ValueError, "aligned authorized"):
                actor.game_logits(torch.zeros(3, 64), torch.zeros(3, 48))
        parent.projection.bias.data[0] = 0.01
        with self.assertRaisesRegex(ValueError, "zero parent projection"):
            parent.enable_team_adapter("contextual")
        with self.assertRaisesRegex(ValueError, "unadapted"):
            Actor().enable_team_adapter("residual")

    def test_real_team_optimizer_steps_preserve_one_v_one_and_base_weights_exactly(
        self,
    ):
        for mode in ("residual", "contextual"):
            parent = Actor(True, 16)
            actor = copy.deepcopy(parent).enable_team_adapter(mode)
            before = copy.deepcopy(actor.state_dict())
            optimizer = torch.optim.Adam(actor.parameters(), lr=0.02)
            obs = observation(4)
            initial = float(scalar(actor, obs).probs[2].detach())
            for _ in range(20):
                loss = -scalar(actor, obs).log_prob(torch.tensor(2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.assertGreater(
                float(scalar(actor, obs).probs[2].detach()), initial + 0.2
            )
            self.assertFalse(
                torch.equal(actor.projection.weight, before["projection.weight"])
            )
            self.assertFalse(
                torch.equal(actor.memory.weight_ih, before["memory.weight_ih"])
            )
            for name, value in actor.state_dict().items():
                if name.startswith(("state.", "action.", "score.", "signal.")):
                    self.assertTrue(torch.equal(value, before[name]), name)
            for signal in (False, True):
                obs = observation(2, signal=signal)
                history = [
                    observation(2, strengths=(0.8, 0.2, 0.5))["features"]
                ] * 40 + [obs["features"]]
                torch.testing.assert_close(
                    scalar(parent, obs, history).probs,
                    scalar(actor, obs, history).probs,
                    atol=0,
                    rtol=0,
                )
            optimizer.zero_grad()
            (-scalar(actor, observation(2)).log_prob(torch.tensor(0))).backward()
            for parameter in actor.parameters():
                if parameter.grad is not None:
                    self.assertEqual(torch.count_nonzero(parameter.grad), 0)

    def test_compact_snapshot_round_trip_preserves_adapter_contract(self):
        from retruco_ml.improvement_train import compact_actor_snapshot

        actor = Actor(True, 64, team_adapter="contextual")
        config = {
            "actor_memory": 64,
            "team_adapter": "contextual",
            "team_adapter_version": 1,
        }
        payload = compact_actor_snapshot(actor, config, {}, 50000, 7)
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        restored = actor_from_checkpoint(torch.load(stream, weights_only=True))
        torch.testing.assert_close(
            scalar(actor, observation()).probs,
            scalar(restored, observation()).probs,
            atol=0,
            rtol=0,
        )
        broken = {
            k: v
            for k, v in payload.items()
            if k not in ("configuration", "encoder", "memory_version")
        }
        with self.assertRaises(ValueError):
            actor_from_checkpoint(broken)

    def test_context_features_bind_own_card_and_only_visible_current_trick(self):
        obs = observation(round_index=1)
        f = torch.tensor(obs["features"]).repeat(3, 1)
        a = torch.tensor(obs["actions"])
        # Partner leads publicly. Old tricks and concealed cards are excluded.
        partner = 39 + 2 * 22 + 1 + 7
        opponent = 39 + 1 * 22 + 1 + 7
        f[:, partner : partner + 7] = torch.tensor([1, 0.25, 0.5, 0.6, 1, 0, 0])
        f[:, opponent : opponent + 7] = torch.tensor([1, 0.25, 0.5, 0.4, 1, 0, 0])
        f[:, 39 + 3 * 22 + 1 + 7 : 39 + 3 * 22 + 1 + 14] = torch.tensor(
            [0, 0, 0, 0.99, 1, 0, 1]
        )
        x = contextual_card_features(f, a)
        torch.testing.assert_close(x[:, 1], torch.tensor([0.2, 0.5, 0.8]))
        torch.testing.assert_close(x[:, 10], torch.full((3,), 0.6))
        self.assertTrue((x[:, 12] == 1).all())
        torch.testing.assert_close(x[:, 13], torch.tensor([0.0, 0.0, 1.0]))
        changed = f.clone()
        changed[:, 39 + 3 * 22 + 1 + 7 + 3] = 0.05
        changed[:, 39 + 1 + 3] = 0.95  # Previous trick.
        changed[:, 192:] = torch.randn_like(changed[:, 192:])
        torch.testing.assert_close(
            contextual_card_features(changed, a), x, atol=0, rtol=0
        )
        changed = f.clone()
        changed[:, opponent + 3] = 0.6
        self.assertTrue((contextual_card_features(changed, a)[:, 12] == 0).all())
        changed[:, partner + 5] = 1
        self.assertTrue((contextual_card_features(changed, a)[:, 11] == 0).all())
        changed = f.clone()
        changed[:, 0] = 0.5
        self.assertEqual(torch.count_nonzero(contextual_card_features(changed, a)), 0)
        call = a.clone()
        call[:, 1] = 0
        call[:, 4] = 1
        self.assertEqual(torch.count_nonzero(contextual_card_features(f, call)), 0)
        stack = a.clone()
        stack[:, 1] = 0
        stack[:, 2] = 1
        stack[:, 33] = 1
        torch.testing.assert_close(
            contextual_card_features(f, stack)[:, 6], torch.full((3,), 0.8)
        )

    def test_contextual_branch_can_learn_card_binding_without_base_state_information(
        self,
    ):
        actor = Actor(True, 16, team_adapter="contextual")
        with torch.no_grad():
            for p in actor.state.parameters():
                p.zero_()
            for p in actor.score.parameters():
                p.zero_()
        observations = [
            observation(strengths=v)
            for v in ((0.9, 0.2, 0.1), (0.1, 0.9, 0.2), (0.2, 0.1, 0.9))
        ]
        contexts = torch.zeros(3, 64)
        targets = torch.tensor([0, 1, 2])
        optimizer = torch.optim.Adam(actor.team_head.parameters(), lr=0.03)
        for _ in range(60):
            loss = (
                -candidate_distribution(actor, contexts, observations)
                .log_prob(targets)
                .mean()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        probabilities = candidate_distribution(actor, contexts, observations).probs
        self.assertTrue((probabilities[torch.arange(3), targets] > 0.9).all())
        # Removing candidate context makes these observations indistinguishable.
        control = Actor(True, 16, team_adapter="residual")
        control.state.load_state_dict(actor.state.state_dict())
        control.score.load_state_dict(actor.score.state_dict())
        control.action.load_state_dict(actor.action.state_dict())
        control.team_head.load_state_dict(actor.team_head.state_dict())
        p = candidate_distribution(control, contexts, observations).probs
        torch.testing.assert_close(p[0], p[1], atol=0, rtol=0)
        torch.testing.assert_close(p[1], p[2], atol=0, rtol=0)

    def test_scalar_batch_collection_replay_agree_after_adapter_changes_and_ppo(self):
        actor = Actor(True, 16, team_adapter="contextual")
        with torch.no_grad():
            actor.team_head[2].weight.normal_(std=0.08)
            actor.projection.weight.normal_(std=0.03)
        records, observations, pages = [], [], []
        for players, signal in ((2, False), (4, False), (4, True)):
            obs = observation(players, signal=signal)
            events = [
                observation(players, strengths=(0.8, 0.5, 0.2))["features"]
                for _ in range(36)
            ] + [obs["features"]]
            dist = scalar(actor, obs, events)
            record = Decision(
                events,
                len(events),
                obs,
                np.zeros(128, np.float32),
                2,
                float(dist.log_prob(torch.tensor(2)).detach()),
                0.0,
            )
            record.target, record.advantage = 1.0, 1.0 if signal else -1.0
            records.append(record)
            observations.append(obs)
            pages.append(
                {
                    "version": 1,
                    "complete": True,
                    "hand": 1,
                    "start": 0,
                    "end": 37,
                    "total": 37,
                    "events": events,
                }
            )
        packed = PackedRecords(records)
        contexts, indices = packed.contexts(actor, packed.chunks)
        dist = packed.distribution(actor, contexts, indices)
        expected = torch.stack(
            [
                scalar(actor, records[i].observation, records[i].events).log_prob(
                    torch.tensor(2)
                )
                for i in indices
            ]
        )
        torch.testing.assert_close(
            dist.log_prob(packed.actions[indices]), expected, atol=1e-7, rtol=1e-6
        )
        replays = [Replay(actor) for _ in records]
        advance_group(actor, replays, pages)
        features = torch.tensor(np.asarray([o["features"] for o in observations]))
        collected = actor.projected_context(
            actor.state(features), features, torch.stack([r.memory[0] for r in replays])
        )
        torch.testing.assert_close(collected, contexts, atol=1e-7, rtol=1e-6)
        parent = copy.deepcopy(actor)
        critic = Critic(True)
        update(
            actor,
            critic,
            torch.optim.Adam(actor.parameters(), lr=1e-4),
            torch.optim.Adam(critic.parameters()),
            records,
            random.Random(1),
            parent,
            epochs=1,
            response_min_choices=100,
        )
        self.assertFalse(
            torch.equal(actor.team_head[2].weight, parent.team_head[2].weight)
        )
        torch.testing.assert_close(
            scalar(actor, observation(2)).probs,
            scalar(parent, observation(2)).probs,
            atol=0,
            rtol=0,
        )

    def test_serialized_mode_strictly_bound_and_frozen_after_load(self):
        for mode in ("residual", "contextual"):
            actor = Actor(True, 64, team_adapter=mode)
            with torch.no_grad():
                actor.projection.weight.normal_(std=0.02)
                actor.team_head[2].weight.normal_(std=0.04)
            saved = {
                "model": actor.state_dict(),
                "configuration": {
                    "actor_memory": 64,
                    **team_adapter_configuration(actor.state_dict()),
                },
            }
            stream = io.BytesIO()
            torch.save(saved, stream)
            stream.seek(0)
            restored = actor_from_checkpoint(torch.load(stream, weights_only=True))
            torch.testing.assert_close(
                scalar(actor, observation()).probs,
                scalar(restored, observation()).probs,
                atol=0,
                rtol=0,
            )
            self.assertFalse(restored.state[0].weight.requires_grad)
            for configuration in (
                {"actor_memory": 64},
                {"team_adapter": mode, "team_adapter_version": 2},
                {
                    "team_adapter": "residual"
                    if mode == "contextual"
                    else "contextual",
                    "team_adapter_version": 1,
                },
            ):
                with self.assertRaisesRegex(ValueError, "metadata"):
                    actor_from_checkpoint({**saved, "configuration": configuration})
            with self.assertRaisesRegex(ValueError, "widen actor before"):
                widen_actor(Actor(True, 16, team_adapter=mode), 64)

    def test_frozen_league_turnover_retains_adapter_mode_and_export_fails_before_writes(
        self,
    ):
        reference = Policy()
        population = HistoricalLeaguePopulation(max_snapshots=1)
        for mode in ("residual", "contextual"):
            actor = Actor(True, 64, team_adapter=mode)
            with torch.no_grad():
                actor.team_head[2].weight.normal_(std=0.1)
            rng = torch.get_rng_state().clone()
            key = population.add_actor(actor, 0 if mode == "residual" else 100)
            resumed = HistoricalLeaguePopulation(max_snapshots=1)
            resumed.load_state_dict(population.state_dict())
            frozen = resumed.models(reference)[key]
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(frozen.team_adapter, mode)
            self.assertTrue(all(not p.requires_grad for p in frozen.parameters()))
            torch.testing.assert_close(
                scalar(actor, observation()).probs,
                scalar(frozen, observation()).probs,
                atol=0,
                rtol=0,
            )
            self.assertEqual(population.snapshot_count, 1)
            with tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "unsupported-export"
                with self.assertRaisesRegex(ValueError, "browser runtime"):
                    export(actor, destination)
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
