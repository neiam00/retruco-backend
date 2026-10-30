package game_test

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"testing"

	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRematchRetainsTableAndStartsOnlyAfterEveryHumanConsents(t *testing.T) {
	for _, scenario := range []struct {
		name    string
		players int
		cpu     []int
	}{
		{name: "two humans", players: 2},
		{name: "four humans", players: 4},
		{name: "solo CPU", players: 2, cpu: []int{1}},
		{name: "human partners against CPU", players: 4, cpu: []int{1, 3}},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			repository := &memoryRepository{payloads: make(map[string][]byte)}
			log := slog.New(slog.NewTextHandler(io.Discard, nil))
			host := game.New(repository, log)
			owner := store.User{ID: "owner", Name: "Ana"}
			config := rules.Config{Players: scenario.players, Target: 12, FlorRequired: true}
			options := game.RoomOptions{CPUSeats: scenario.cpu, SpectatorsCanSpeak: true}
			code, err := host.Create(t.Context(), owner, config, true, options)
			require.NoError(t, err)
			defer host.ConnectPlayer(code, owner.ID, "owner-tab")()
			humans := []store.User{owner}
			initial, err := host.Snapshot(code, owner.ID, 0)
			require.NoError(t, err)
			for seat := 1; seat < scenario.players; seat++ {
				if initial.Seats[seat].CPU {
					continue
				}
				player := store.User{ID: fmt.Sprintf("human-%d", seat), Name: fmt.Sprintf("Player %d", seat)}
				require.NoError(t, host.Join(t.Context(), code, player, &seat))
				humans = append(humans, player)
			}
			before, err := host.Snapshot(code, owner.ID, 0)
			require.NoError(t, err)
			require.Error(t, host.Handle(t.Context(), code, owner, game.Input{ID: "too-early", Revision: before.State.Revision, Command: rules.Command{Type: "rematch"}}))
			require.NoError(t, host.Handle(t.Context(), code, owner, game.Input{ID: "chat", Chat: "Seguimos juntos"}))
			require.NoError(t, host.Handle(t.Context(), code, owner, game.Input{ID: "finish", Revision: before.State.Revision, Command: rules.Command{Type: "leave"}}))
			finished, err := host.Snapshot(code, owner.ID, 0)
			require.NoError(t, err)
			require.Equal(t, rules.MatchOver, finished.State.Phase)
			require.Error(t, host.Handle(t.Context(), code, store.User{ID: "watcher"}, game.Input{ID: "spectator-vote", Revision: finished.State.Revision, Command: rules.Command{Type: "rematch"}}))
			require.Error(t, host.Handle(t.Context(), code, owner, game.Input{ID: "stale-vote", Revision: before.State.Revision, Command: rules.Command{Type: "rematch"}}))
			var last game.Input
			for index, player := range humans {
				last = game.Input{ID: fmt.Sprintf("vote-%d", index), Revision: finished.State.Revision, Command: rules.Command{Type: "rematch"}}
				if index == len(humans)-1 {
					repository.fail = true
					require.Error(t, host.Handle(t.Context(), code, player, last))
					unchanged, snapshotErr := host.Snapshot(code, owner.ID, 0)
					require.NoError(t, snapshotErr)
					assert.Equal(t, finished.State, unchanged.State, "failed storage must retain the result")
					repository.fail = false
				}
				require.NoError(t, host.Handle(t.Context(), code, player, last))
				require.NoError(t, host.Handle(t.Context(), code, player, last), "retrying the same vote is idempotent")
				if index < len(humans)-1 {
					waiting, snapshotErr := host.Snapshot(code, owner.ID, 0)
					require.NoError(t, snapshotErr)
					assert.Equal(t, rules.MatchOver, waiting.State.Phase)
					assert.Len(t, waiting.RematchVotes, index+1)
				}
			}
			after, err := host.Snapshot(code, owner.ID, 0)
			require.NoError(t, err)
			assert.Equal(t, 2, after.GameNumber)
			assert.Equal(t, rules.Cutting, after.State.Phase)
			assert.Greater(t, after.State.Revision, finished.State.Revision)
			assert.Equal(t, before.Seats, after.Seats)
			assert.Equal(t, before.Config, after.Config)
			assert.Equal(t, before.Options, after.Options)
			assert.True(t, after.Private)
			assert.Equal(t, [2]int{}, after.State.Scores)
			assert.Equal(t, finished.Chat, after.Chat)
			assert.Empty(t, after.RematchVotes)
			assert.Empty(t, after.CantoHistory)
			require.Len(t, repository.payloads, 2, "the original game owns its own durable record")
			var original game.Match
			require.NoError(t, json.Unmarshal(repository.payloads[code], &original))
			assert.Equal(t, rules.MatchOver, original.State.Phase)
			originalBytes := append([]byte(nil), repository.payloads[code]...)
			restored := game.New(repository, log)
			require.NoError(t, restored.Restore(t.Context()))
			defer restored.ConnectPlayer(code, owner.ID, "restored-tab")()
			view, err := restored.Snapshot(code, owner.ID, 0)
			require.NoError(t, err)
			assert.Equal(t, 2, view.GameNumber)
			actor := store.User{ID: view.Seats[view.State.Turn].ID}
			if !view.Seats[view.State.Turn].CPU {
				require.NoError(t, restored.Handle(t.Context(), code, actor, game.Input{ID: "next-game-cut", Revision: view.State.Revision, Command: rules.Command{Type: "cut", Cut: 20}}))
			}
			assert.Equal(t, originalBytes, repository.payloads[code], "new-game moves must never overwrite the previous result")
			resume := restored.Resumable(owner.ID)
			require.Len(t, resume, 1)
			assert.Equal(t, code, resume[0].Code)
			assert.Empty(t, restored.Resumable("watcher"))
		})
	}
}
