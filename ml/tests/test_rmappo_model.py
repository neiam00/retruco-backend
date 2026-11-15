"""Behavior contracts for fresh recurrent policies and centralized critics."""

import copy
import io
import unittest
from types import SimpleNamespace

import torch

from retruco_ml.recurrent import Actor, centered_categorical
from retruco_ml.rmappo_model import (
    CHECKPOINT_KIND,
    RecurrentActor,
    RecurrentCritic,
    actor_from_checkpoint,
    critic_from_checkpoint,
)
from retruco_ml.rmappo_ppo import CriticRecords


class RecurrentMAPPOModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)

    def assertStateClose(self, left, right):
        for first, second in zip(left, right):
            torch.testing.assert_close(first, second, atol=2e-6, rtol=1e-5)

    def test_batched_sequence_matches_independent_streaming_seats(self):
        """Fused replay must reproduce collection, including LSTM cell state."""
        local, privileged = torch.randn(3, 7, 768), torch.randn(3, 7, 128)
        for cell in ("gru", "lstm"):
            for model in (RecurrentActor(cell), RecurrentCritic(cell)):
                with self.subTest(cell=cell, model=type(model).__name__):
                    args = (local,) if isinstance(model, Actor) else (local, privileged)
                    output, state = model.replay(*args)
                    for seat in range(3):
                        memory = model.initial()
                        steps = []
                        for index in range(7):
                            step_args = tuple(value[seat, index] for value in args)
                            value, memory = model.step(*step_args, memory)
                            steps.append(value)
                        torch.testing.assert_close(
                            output[seat], torch.stack(steps), atol=2e-6, rtol=1e-5
                        )
                        self.assertStateClose(
                            tuple(part[seat] for part in state), memory
                        )
                    if cell == "gru":
                        self.assertEqual(torch.count_nonzero(state[1]).item(), 0)
                    else:
                        self.assertGreater(torch.count_nonzero(state[1]).item(), 0)

    def test_padding_and_hand_resets_do_not_change_other_seat_history(self):
        """Padding never advances a seat, and resets discard both LSTM states."""
        local, privileged = torch.randn(3, 6, 768), torch.randn(3, 6, 128)
        masks = (
            (
                torch.tensor(
                    [[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                    dtype=torch.bool,
                ),
                None,
            ),
            (
                torch.tensor(
                    [[1, 1, 0, 1, 1, 1], [0, 1, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0]],
                    dtype=torch.bool,
                ),
                torch.tensor(
                    [[0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]],
                    dtype=torch.bool,
                ),
            ),
        )
        for cell in ("gru", "lstm"):
            for model in (RecurrentActor(cell), RecurrentCritic(cell)):
                for valid, reset in masks:
                    with self.subTest(
                        cell=cell, model=type(model).__name__, resets=reset is not None
                    ):
                        initial = tuple(
                            torch.randn_like(part) for part in model.initial(3)
                        )
                        if cell == "gru":
                            initial = initial[0], torch.zeros_like(initial[1])
                        poisoned_local = local.masked_fill(
                            ~valid.unsqueeze(-1), float("nan")
                        )
                        args = (
                            (poisoned_local,)
                            if isinstance(model, Actor)
                            else (
                                poisoned_local,
                                privileged.masked_fill(
                                    ~valid.unsqueeze(-1), float("nan")
                                ),
                            )
                        )
                        output, final = model.replay(
                            *args, initial, valid=valid, reset=reset
                        )
                        self.assertTrue(torch.isfinite(output).all())
                        self.assertEqual(torch.count_nonzero(output[~valid]).item(), 0)
                        for seat in range(3):
                            memory = tuple(part[seat] for part in initial)
                            for index in range(6):
                                if not valid[seat, index]:
                                    continue
                                if reset is not None and reset[seat, index]:
                                    memory = model.initial()
                                step_args = tuple(value[seat, index] for value in args)
                                expected, memory = model.step(*step_args, memory)
                                torch.testing.assert_close(
                                    output[seat, index], expected, atol=2e-6, rtol=1e-5
                                )
                            self.assertStateClose(
                                tuple(part[seat] for part in final), memory
                            )
                        output.sum().backward()
                        self.assertTrue(
                            all(
                                p.grad is None or torch.isfinite(p.grad).all()
                                for p in model.parameters()
                            )
                        )

    def test_prefix_is_recomputed_with_current_weights_and_detached_for_tbptt(self):
        """A PPO epoch may not train from a stale rollout recurrent state."""
        for cell in ("gru", "lstm"):
            for model in (RecurrentActor(cell), RecurrentCritic(cell)):
                with self.subTest(cell=cell, model=type(model).__name__):
                    local = torch.randn(2, 8, 768, requires_grad=True)
                    privileged = torch.randn(2, 8, 128)
                    args = (local,) if isinstance(model, Actor) else (local, privileged)
                    prefix_args = tuple(value[:, :5] for value in args)
                    old_state = model.recompute_prefix(*prefix_args)
                    with torch.no_grad():
                        model.memory.core.bias_ih_l0.add_(0.2)
                    prefix = model.recompute_prefix(*prefix_args)
                    self.assertFalse(torch.allclose(old_state[0], prefix[0]))
                    self.assertTrue(all(not part.requires_grad for part in prefix))
                    full, full_state = model.replay(*args)
                    suffix, suffix_state = model.replay(
                        *(value[:, 5:] for value in args), prefix
                    )
                    torch.testing.assert_close(
                        suffix, full[:, 5:], atol=2e-6, rtol=1e-5
                    )
                    self.assertStateClose(suffix_state, full_state)
                    suffix.sum().backward()
                    self.assertEqual(torch.count_nonzero(local.grad[:, :5]).item(), 0)
                    self.assertGreater(torch.count_nonzero(local.grad[:, 5:]).item(), 0)

    def test_critic_training_and_privileged_inputs_cannot_change_actor(self):
        """The centralized value learner has no path into deployed policy state."""
        for cell in ("gru", "lstm"):
            with self.subTest(cell=cell):
                actor, critic = RecurrentActor(cell), RecurrentCritic(cell)
                self.assertLessEqual(
                    sum(p.numel() for p in actor.parameters()), 150_000
                )
                self.assertTrue(
                    {id(p) for p in actor.parameters()}.isdisjoint(
                        id(p) for p in critic.parameters()
                    )
                )
                self.assertGreater(
                    torch.count_nonzero(actor.projection.weight).item(), 0
                )
                local, privileged = torch.randn(2, 4, 768), torch.randn(2, 4, 128)
                original = copy.deepcopy(actor.state_dict())
                before, memory = actor.replay(local)
                value, _ = critic.replay(local, privileged)
                changed, _ = critic.replay(local, privileged + 4)
                self.assertFalse(torch.allclose(value, changed))
                optimizer = torch.optim.Adam(critic.parameters(), lr=0.01)
                optimizer.zero_grad()
                (value - 2).square().mean().backward()
                self.assertTrue(all(p.grad is None for p in actor.parameters()))
                optimizer.step()
                after, new_memory = actor.replay(local)
                torch.testing.assert_close(before, after, rtol=0, atol=0)
                self.assertStateClose(memory, new_memory)
                for name, tensor in actor.state_dict().items():
                    torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "authorized"):
                    actor.step(torch.cat((local[0, 0], privileged[0, 0])))

    def test_critic_training_windows_preserve_scalar_values_and_record_order(self):
        """Mixed hand lengths and TBPTT offsets retain the collected values."""
        histories = [torch.randn(length, 896).numpy() for length in (45, 70, 3)]
        locations = ((0, 1), (0, 33), (0, 45), (1, 5), (1, 65), (1, 70), (2, 3))
        records = [
            SimpleNamespace(critic_events=histories[hand], critic_end=end)
            for hand, end in locations
        ]
        indices = [5, 0, 6, 2, 4, 1, 3]
        for cell in ("gru", "lstm"):
            with self.subTest(cell=cell):
                critic = RecurrentCritic(cell)
                expected = {}
                with torch.no_grad():
                    for hand, history in enumerate(histories):
                        state = critic.initial()
                        for offset, event in enumerate(torch.from_numpy(history), 1):
                            value, state = critic.step(event[:768], event[768:], state)
                            expected[hand, offset] = value
                actual = CriticRecords(records, sequence=32).values(critic, indices)
                torch.testing.assert_close(
                    actual,
                    torch.stack([expected[locations[i]] for i in indices]),
                    atol=2e-6,
                    rtol=1e-5,
                )
                actual.square().sum().backward()
                self.assertGreater(
                    critic.memory.core.weight_hh_l0.grad.abs().sum().item(), 0
                )

    def test_checkpoint_restores_architecture_and_recurrent_continuation(self):
        """Both cell types survive the actual tensor checkpoint boundary."""
        for cell in ("gru", "lstm"):
            with self.subTest(cell=cell):
                actor, critic = RecurrentActor(cell), RecurrentCritic(cell)
                local, privileged = torch.randn(1, 5, 768), torch.randn(1, 5, 128)
                _, actor_state = actor.replay(local[:, :3])
                _, critic_state = critic.replay(local[:, :3], privileged[:, :3])
                buffer = io.BytesIO()
                torch.save(
                    {
                        "kind": CHECKPOINT_KIND,
                        "configuration": {"cell": cell},
                        "actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                        "actor_state": actor_state,
                        "critic_state": critic_state,
                    },
                    buffer,
                )
                buffer.seek(0)
                saved = torch.load(buffer, weights_only=True)
                restored_actor, restored_critic = (
                    actor_from_checkpoint(saved),
                    critic_from_checkpoint(saved),
                )
                expected, expected_state = actor.replay(local[:, 3:], actor_state)
                actual, actual_state = restored_actor.replay(
                    local[:, 3:], saved["actor_state"]
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertStateClose(actual_state, expected_state)
                expected, expected_state = critic.replay(
                    local[:, 3:], privileged[:, 3:], critic_state
                )
                actual, actual_state = restored_critic.replay(
                    local[:, 3:], privileged[:, 3:], saved["critic_state"]
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertStateClose(actual_state, expected_state)
                with self.assertRaisesRegex(ValueError, "rmappo-v1"):
                    actor_from_checkpoint({**saved, "kind": "recurrent-v1"})
                with self.assertRaisesRegex(ValueError, "cell"):
                    actor_from_checkpoint({**saved, "configuration": {"cell": "rnn"}})

    def test_ppo_log_probabilities_match_journal_collection_after_burn_in(self):
        """FP64 centering preserves ratios for varied candidate counts/offsets."""
        for cell in ("gru", "lstm"):
            with self.subTest(cell=cell):
                actor = RecurrentActor(cell)
                with torch.no_grad():
                    actor.score[2].bias.fill_(1e8)
                    actor.score[2].weight.mul_(100)
                    actor.signal.bias.add_(1e8)
                events = torch.randn(3, 7, 768)
                local = torch.randn(3, 768)
                actions = [torch.randn(count, 48) for count in (1, 5, 11)]
                collected = []
                for seat in range(3):
                    state = actor.initial()
                    for event in events[seat]:
                        state = actor.advance(event, state)
                    context = actor.context(local[seat], state)
                    logits = actor.game_logits(
                        context.expand(len(actions[seat]), -1), actions[seat]
                    )
                    collected.append(
                        centered_categorical(logits).log_prob(torch.tensor(0))
                    )
                prefix = actor.recompute_prefix(events[:, :4])
                _, replayed = actor.replay(events[:, 4:], prefix)
                contexts = actor.context(local, replayed)
                selected = []
                for seat in range(3):
                    logits = actor.game_logits(
                        contexts[seat].expand(len(actions[seat]), -1), actions[seat]
                    )
                    self.assertEqual(logits.dtype, torch.float64)
                    selected.append(
                        centered_categorical(logits).log_prob(torch.tensor(0))
                    )
                ratios = (torch.stack(selected) - torch.stack(collected)).exp()
                torch.testing.assert_close(
                    ratios, torch.ones_like(ratios), atol=2e-4, rtol=0
                )
                truthful = torch.tensor([True, False, True, False, False, True])
                logits = actor.signal_logits(contexts).masked_fill(
                    ~truthful, -torch.inf
                )
                distribution = centered_categorical(logits)
                self.assertEqual(
                    distribution.probs[:, ~truthful].count_nonzero().item(), 0
                )
                self.assertTrue(
                    torch.isfinite(
                        distribution.log_prob(torch.zeros(3, dtype=torch.long))
                    ).all()
                )


if __name__ == "__main__":
    unittest.main()
