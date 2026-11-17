package game_test

import (
	"encoding/json"
	"io"
	"log/slog"
	"testing"

	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRestoreDistinguishesLegacyALeyFromConfirmedFlor(t *testing.T) {
	for _, declared := range []bool{false, true} {
		for _, legacyHistory := range []bool{false, true} {
			name := map[bool]string{false: "only A ley", true: "confirmed flor then A ley"}[declared]
			name += map[bool]string{false: "/full history", true: "/recent events only"}[legacyHistory]
			t.Run(name, func(t *testing.T) {
				state, err := rules.New(rules.Config{Players: 2, Target: 12}, 0, rules.Deck())
				require.NoError(t, err)
				state, _, err = rules.Apply(state, state.Turn, rules.Command{Type: "cut", Cut: 3})
				require.NoError(t, err)
				require.False(t, state.Players[0].FlorValid)
				require.True(t, state.Players[1].FlorValid)
				var history []game.GameEvent
				play := func(seat int, command rules.Command) {
					var events []rules.Event
					state, events, err = rules.Apply(state, seat, command)
					require.NoError(t, err)
					for _, event := range events {
						history = append(history, game.GameEvent{ID: uint64(len(history) + 1), Hand: state.HandNumber, Event: event})
					}
				}
				if declared {
					play(1, rules.Command{Type: "call", Kind: "truco", Declaration: "con_mi_flor"})
					play(0, rules.Command{Type: "respond", Kind: "accept"})
				}
				play(1, rules.Command{Type: "play", Card: 0, Declaration: "a_ley"})
				// Reproduce the old persisted flag, which A ley incorrectly set.
				state.Players[1].FlorDeclared = true
				match := game.Match{Code: "FLOR-1234", Config: state.Config, State: &state,
					Seats:  []game.Seat{{ID: "ana", Name: "Ana"}, {ID: "beto", Name: "Beto"}},
					Events: history, CantoHistory: history, Sequence: 10}
				if legacyHistory {
					match.CantoHistory = nil
				} else {
					// A confirmed flor from a previous hand must not affect this hand.
					match.CantoHistory = append([]game.GameEvent{{Hand: state.HandNumber - 1,
						Event: rules.Event{Type: "play", Seat: 1, Text: "flor"}}}, history...)
				}
				payload, err := json.Marshal(match)
				require.NoError(t, err)
				repository := &memoryRepository{payloads: map[string][]byte{match.Code: payload}}
				host := game.New(repository, slog.New(slog.NewTextHandler(io.Discard, nil)))
				require.NoError(t, host.Restore(t.Context()))
				before, err := host.Snapshot(match.Code, "ana", 0)
				require.NoError(t, err)
				command := rules.Command{Type: "call", Kind: "envido"}
				if declared {
					assert.NotContains(t, before.State.Legal, command)
					require.Contains(t, before.State.Legal, rules.Command{Type: "call", Kind: "ley_envido"})
				} else {
					require.Contains(t, before.State.Legal, command)
					assert.NotContains(t, before.State.Legal, rules.Command{Type: "call", Kind: "flor"})
				}
				err = host.Handle(t.Context(), match.Code, store.User{ID: "ana"}, game.Input{
					ID: "old-browser-envido", Revision: state.Revision, Command: command,
				})
				if declared {
					require.Error(t, err, "old clients cannot bypass the declared-flor restriction")
				} else {
					require.NoError(t, err, "a saved A ley must not prevent envido after deployment")
				}
			})
		}
	}
}

func TestRestoreVoidsEnvidoOnlyForAFlorAfterTheWager(t *testing.T) {
	for _, later := range []bool{false, true} {
		t.Run(map[bool]string{false: "previous flor", true: "later flor"}[later], func(t *testing.T) {
			state, err := rules.New(rules.Config{Players: 4, Target: 12}, 0, rules.Deck())
			require.NoError(t, err)
			state, _, err = rules.Apply(state, state.Turn, rules.Command{Type: "cut", Cut: 3})
			require.NoError(t, err)
			match := game.Match{Code: "FLOR-1234", Config: state.Config, State: &state,
				Seats: []game.Seat{{ID: "ana"}, {ID: "beto"}, {ID: "carla"}, {ID: "daniel"}}}
			play := func(seat int, cmd rules.Command) {
				next, events, err := rules.Apply(state, seat, cmd)
				require.NoError(t, err)
				state = next
				for _, event := range events {
					match.CantoHistory = append(match.CantoHistory, game.GameEvent{Hand: state.HandNumber, Event: event})
				}
			}
			announce := func() {
				play(1, rules.Command{Type: "call", Kind: "truco"})
				play(0, rules.Command{Type: "respond", Kind: "accept", Declaration: "con_mi_flor"})
			}
			if !later {
				announce()
			}
			play(1, rules.Command{Type: "call", Kind: "ley_envido"})
			play(2, rules.Command{Type: "respond", Kind: "accept", Mode: "sin_ley"})
			wager := state.Envido
			if later {
				announce()
				// Older saves retained the accepted wager after this declaration.
				state.Envido, state.EnvidoAccepted, state.EnvidoCanceled = wager, true, false
			}
			state.EnvidoOpener = 0 // Missing in older snapshots.
			payload, err := json.Marshal(match)
			require.NoError(t, err)
			host := game.New(&memoryRepository{payloads: map[string][]byte{match.Code: payload}}, slog.New(slog.NewTextHandler(io.Discard, nil)))
			require.NoError(t, host.Restore(t.Context()))
			view, err := host.Snapshot(match.Code, "beto", 1)
			require.NoError(t, err)
			ley := rules.Command{Type: "call", Kind: "ley_envido"}
			if later {
				assert.Contains(t, view.State.Legal, ley)
			} else {
				assert.NotContains(t, view.State.Legal, ley)
			}
		})
	}
}

func TestRestoreWithdrawsAnUnansweredLeyOpeningAfterOwnFlor(t *testing.T) {
	for _, interrupted := range []bool{false, true} {
		t.Run(map[bool]string{false: "card turn", true: "interrupted retruco"}[interrupted], func(t *testing.T) {
			s, err := rules.New(rules.Config{Players: 2, Target: 12}, 0, rules.Deck())
			require.NoError(t, err)
			s, _, err = rules.Apply(s, s.Turn, rules.Command{Type: "cut", Cut: 3})
			require.NoError(t, err)
			var history []game.GameEvent
			step := func(seat int, command rules.Command) {
				var events []rules.Event
				s, events, err = rules.Apply(s, seat, command)
				require.NoError(t, err)
				for _, event := range events {
					history = append(history, game.GameEvent{Hand: s.HandNumber, Event: event})
				}
			}
			step(1, rules.Command{Type: "call", Kind: "truco", Declaration: "con_mi_flor"})
			if interrupted {
				step(0, rules.Command{Type: "respond", Kind: "raise", Amount: 6})
			} else {
				step(0, rules.Command{Type: "respond", Kind: "accept"})
			}
			before := s
			// Former engines accepted this wager and suspended the pending Truco.
			s.Suspended = s.Pending
			s.Pending = &rules.Wager{Kind: "ley_envido", Caller: 1, Value: 2, Previous: 1}
			s.Ley, s.DoubleFlor[1] = true, true
			s.Players[1].FlorDeclared = false // Older snapshots can lack the flag.
			history = append(history, game.GameEvent{Hand: s.HandNumber, Event: rules.Event{Seat: 1, Type: "call", Kind: "ley_envido"}})
			match := game.Match{Code: "FLOR-4321", Config: s.Config, State: &s, CantoHistory: history, Seats: []game.Seat{{ID: "ana"}, {ID: "beto"}}}
			payload, err := json.Marshal(match)
			require.NoError(t, err)
			host := game.New(&memoryRepository{payloads: map[string][]byte{match.Code: payload}}, slog.New(slog.NewTextHandler(io.Discard, nil)))
			require.NoError(t, host.Restore(t.Context()))
			view, err := host.Snapshot(match.Code, "beto", 1)
			require.NoError(t, err)
			assert.Equal(t, before.Pending, view.State.Pending)
			assert.Nil(t, view.State.Suspended)
			assert.True(t, view.State.Seats[1].FlorDeclared)
			assert.Equal(t, rules.LegalActions(before, 1), view.State.Legal)
			assert.Greater(t, view.State.Revision, s.Revision)
		})
	}
}
