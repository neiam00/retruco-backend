package store_test

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

func TestStatisticsCountLifetimeResultsAndSeparateCPUAndPartnerIdentities(t *testing.T) {
	pool, _ := migrationDatabase(t, 2)
	repository := store.New(pool)
	for _, user := range []store.User{{ID: "ana", Name: "Ana"}, {ID: "partner-a", Name: "Alex"}, {ID: "partner-b", Name: "Alex"}} {
		require.NoError(t, repository.CreateUser(t.Context(), user, "", ""))
	}
	write := func(index, winner int, cpu, expired, finished bool, partner string) {
		t.Helper()
		seats := []game.Seat{{ID: "ana", Name: "Ana"}, {ID: "opponent", Name: "Opponent", CPU: cpu}, {ID: partner, Name: "Alex"}, {ID: "other", Name: "Other"}}
		payload, err := json.Marshal(game.Match{Code: "SAME-1234", GameNumber: index + 1, Seats: seats, Config: rules.Config{Players: 4, Target: 12}, Expired: expired, State: &rules.State{Scores: [2]int{12, 6}, Winner: winner, Phase: rules.MatchOver, HandNumber: 8}})
		require.NoError(t, err)
		key := fmt.Sprintf("result-%d", index)
		require.NoError(t, repository.CreateMatch(t.Context(), key, payload, []string{"ana", partner}))
		require.NoError(t, repository.SaveMatch(t.Context(), key, payload, finished, nil))
	}
	for index := range 51 {
		write(index, 0, false, false, true, "partner-a")
	}
	write(51, 1, false, false, true, "partner-b")
	write(52, 1, true, false, true, "partner-a")
	write(53, 0, false, true, true, "partner-a")
	write(54, 0, false, false, false, "partner-a")
	stats, err := repository.Statistics(t.Context(), "ana")
	require.NoError(t, err)
	assert.Equal(t, store.Performance{Games: 53, Wins: 51, Losses: 2, WinRate: 96}, stats.Overall)
	assert.Equal(t, store.Performance{Games: 52, Wins: 51, Losses: 1, WinRate: 98}, stats.Human)
	assert.Equal(t, store.Performance{Games: 1, Losses: 1}, stats.CPU)
	require.Len(t, stats.Partners, 2, "equal display names must not merge different partners")
	assert.Equal(t, "partner-a", stats.Partners[0].ID)
	assert.Equal(t, 51, stats.Partners[0].Human.Games)
	assert.Equal(t, 1, stats.Partners[0].CPU.Losses)
	assert.Equal(t, "partner-b", stats.Partners[1].ID)
	history, err := repository.History(t.Context(), "ana")
	require.NoError(t, err)
	require.Len(t, history, 50)
	for _, entry := range history {
		assert.Equal(t, "SAME-1234", entry.RoomCode)
		assert.Equal(t, 8, entry.Hands)
		assert.Equal(t, 4, entry.Config.Players)
	}
	outsider, err := repository.Statistics(t.Context(), "outsider")
	require.NoError(t, err)
	assert.Zero(t, outsider.Overall.Games)
	assert.Empty(t, outsider.Partners)
}

func TestRematchKeepsEveryPlayersPreviousResultAcrossRestart(t *testing.T) {
	pool, _ := migrationDatabase(t, 2)
	repository := store.New(pool)
	players := []store.User{{ID: "ana", Name: "Ana"}, {ID: "beto", Name: "Beto"}}
	for _, user := range players {
		require.NoError(t, repository.CreateUser(t.Context(), user, "", ""))
	}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	host := game.New(repository, log)
	code, err := host.Create(t.Context(), players[0], rules.Config{Players: 2, Target: 12}, false, game.RoomOptions{})
	require.NoError(t, err)
	require.NoError(t, host.Join(t.Context(), code, players[1], nil))
	view, err := host.Snapshot(code, players[0].ID, 0)
	require.NoError(t, err)
	require.NoError(t, host.Handle(t.Context(), code, players[0], game.Input{ID: "finish", Revision: view.State.Revision, Command: rules.Command{Type: "leave"}}))
	view, err = host.Snapshot(code, players[0].ID, 0)
	require.NoError(t, err)
	for _, user := range players {
		require.NoError(t, host.Handle(t.Context(), code, user, game.Input{ID: "vote", Revision: view.State.Revision, Command: rules.Command{Type: "rematch"}}))
	}
	for _, user := range players {
		history, historyErr := repository.History(t.Context(), user.ID)
		require.NoError(t, historyErr)
		require.Len(t, history, 2)
		assert.NotEqual(t, history[0].Code, history[1].Code)
		assert.Equal(t, code, history[0].RoomCode)
		assert.Equal(t, 2, history[0].GameNumber)
		assert.False(t, history[0].Finished)
		assert.True(t, history[1].Finished)
		stats, statsErr := repository.Statistics(t.Context(), user.ID)
		require.NoError(t, statsErr)
		assert.Equal(t, 1, stats.Overall.Games)
	}
	restored := game.New(repository, log)
	require.NoError(t, restored.Restore(t.Context()))
	defer restored.ConnectPlayer(code, players[0].ID, "returned")()
	current, err := restored.Snapshot(code, players[0].ID, 0)
	require.NoError(t, err)
	require.Equal(t, 2, current.GameNumber)
	require.NoError(t, restored.Handle(t.Context(), code, players[current.State.Turn], game.Input{ID: "cut", Revision: current.State.Revision, Command: rules.Command{Type: "cut", Cut: 20}}))
	history, err := repository.History(t.Context(), players[0].ID)
	require.NoError(t, err)
	require.Len(t, history, 2)
	assert.True(t, history[1].Finished)
	assert.Equal(t, 1, history[1].Winner)
}
