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
				state, _, err = rules.Apply(state, state.Turn, rules.Command{Type: "cut", Cut: 39})
				require.NoError(t, err)
				require.True(t, state.Players[0].FlorValid)
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
					require.Contains(t, before.State.Legal, rules.Command{Type: "call", Kind: "flor"})
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
