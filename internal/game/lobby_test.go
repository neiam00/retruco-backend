package game_test

import (
	"context"
	"io"
	"log/slog"
	"testing"
	"testing/synctest"
	"time"

	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLastPlayerLeavingDeletesWaitingRoomDurably(t *testing.T) {
	repository := &memoryRepository{payloads: make(map[string][]byte)}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	host := game.New(repository, log)
	owner := store.User{ID: "owner", Name: "Owner"}
	code, err := host.Create(t.Context(), owner, rules.Config{Players: 4, Target: 12}, false, game.RoomOptions{})
	require.NoError(t, err)
	// A disconnected seat must not keep the room joinable after its last live player leaves.
	require.NoError(t, host.Join(t.Context(), code, store.User{ID: "offline", Name: "Offline"}, nil))
	disconnect := host.ConnectPlayer(code, owner.ID, "owner-tab")
	defer disconnect()
	require.NoError(t, host.Handle(t.Context(), code, owner, game.Input{ID: "leave", Command: rules.Command{Type: "leave"}}))
	assert.Empty(t, host.List())
	_, err = host.Snapshot(code, owner.ID, 0)
	assert.Error(t, err)
	assert.Error(t, host.Join(t.Context(), code, owner, nil), "even the previous owner cannot rejoin")
	assert.NotContains(t, repository.payloads, code)
	restarted := game.New(repository, log)
	require.NoError(t, restarted.Restore(t.Context()))
	_, err = restarted.Snapshot(code, owner.ID, 0)
	assert.Error(t, err)
}

func TestWaitingRoomDisconnectGraceCountsPlayersAndAllTheirTabs(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		repository := &memoryRepository{payloads: make(map[string][]byte)}
		log := slog.New(slog.NewTextHandler(io.Discard, nil))
		host := game.New(repository, log)
		owner := store.User{ID: "owner", Name: "Owner"}
		code, err := host.Create(t.Context(), owner, rules.Config{Players: 4, Target: 12}, false, game.RoomOptions{})
		require.NoError(t, err)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		go host.Run(ctx)
		firstTab := host.ConnectPlayer(code, owner.ID, "first-tab")
		secondTab := host.ConnectPlayer(code, owner.ID, "second-tab")
		watcher := host.ConnectPlayer(code, "spectator", "watcher-tab")
		defer watcher()
		firstTab()
		time.Sleep(11 * time.Second)
		synctest.Wait()
		_, err = host.Snapshot(code, owner.ID, 0)
		require.NoError(t, err, "the second player tab keeps the room alive")
		secondTab()
		time.Sleep(9 * time.Second)
		synctest.Wait()
		_, err = host.Snapshot(code, owner.ID, 0)
		require.NoError(t, err, "reconnection is allowed within the grace period")
		reconnected := host.ConnectPlayer(code, owner.ID, "reconnected-tab")
		time.Sleep(11 * time.Second)
		synctest.Wait()
		_, err = host.Snapshot(code, owner.ID, 0)
		require.NoError(t, err, "reconnecting cancels the old deletion deadline")
		reconnected()
		time.Sleep(11 * time.Second)
		synctest.Wait()
		assert.Empty(t, host.List(), "spectators do not keep a vacant waiting room alive")
		_, err = host.Snapshot(code, owner.ID, 0)
		assert.Error(t, err)
		assert.Error(t, host.Join(t.Context(), code, store.User{ID: "late", Name: "Late"}, nil))
		cancel()
		synctest.Wait()
		assert.NotContains(t, repository.payloads, code)
	})
}

func TestLeavingKeepsOtherPlayersWaitingAndStartedMatchesSurviveDisconnects(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		repository := &memoryRepository{payloads: make(map[string][]byte)}
		host := game.New(repository, slog.New(slog.NewTextHandler(io.Discard, nil)))
		owner := store.User{ID: "owner", Name: "Owner"}
		other := store.User{ID: "other", Name: "Other"}
		code, err := host.Create(t.Context(), owner, rules.Config{Players: 4, Target: 12}, false, game.RoomOptions{})
		require.NoError(t, err)
		require.NoError(t, host.Join(t.Context(), code, other, nil))
		disconnect := host.ConnectPlayer(code, other.ID, "other-tab")
		defer disconnect()
		require.NoError(t, host.Handle(t.Context(), code, owner, game.Input{ID: "leave", Command: rules.Command{Type: "leave"}}))
		view, err := host.Snapshot(code, other.ID, 0)
		require.NoError(t, err)
		require.Len(t, view.Seats, 4)
		assert.Empty(t, view.Seats[0].ID)
		assert.Equal(t, other.ID, view.Seats[1].ID, "leaving must not change the remaining player's team")

		startedCode, err := host.Create(t.Context(), owner, rules.Config{Players: 2, Target: 12}, false, game.RoomOptions{})
		require.NoError(t, err)
		require.NoError(t, host.Join(t.Context(), startedCode, other, nil))
		host.ConnectPlayer(startedCode, owner.ID, "started-tab")()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		go host.Run(ctx)
		time.Sleep(11 * time.Second)
		synctest.Wait()
		_, err = host.Snapshot(code, other.ID, 0)
		assert.NoError(t, err)
		view, err = host.Snapshot(startedCode, owner.ID, 0)
		require.NoError(t, err)
		assert.NotNil(t, view.State)
	})
}

func TestFailedLobbyDeletionCanBeRetried(t *testing.T) {
	repository := &memoryRepository{payloads: make(map[string][]byte)}
	host := game.New(repository, slog.New(slog.NewTextHandler(io.Discard, nil)))
	owner := store.User{ID: "owner", Name: "Owner"}
	code, err := host.Create(t.Context(), owner, rules.Config{Players: 2, Target: 12}, false, game.RoomOptions{})
	require.NoError(t, err)
	repository.fail = true
	leave := game.Input{ID: "leave", Command: rules.Command{Type: "leave"}}
	require.Error(t, host.Handle(t.Context(), code, owner, leave))
	_, err = host.Snapshot(code, owner.ID, 0)
	require.NoError(t, err)
	repository.fail = false
	require.NoError(t, host.Handle(t.Context(), code, owner, leave))
	_, err = host.Snapshot(code, owner.ID, 0)
	assert.Error(t, err)
}

func TestCPUSeatsDoNotKeepAnEmptyWaitingRoomAlive(t *testing.T) {
	repository := &memoryRepository{payloads: make(map[string][]byte)}
	host := game.New(repository, slog.New(slog.NewTextHandler(io.Discard, nil)))
	owner := store.User{ID: "owner", Name: "Owner"}
	code, err := host.Create(t.Context(), owner, rules.Config{Players: 4, Target: 12}, false, game.RoomOptions{CPUSeats: []int{3}})
	require.NoError(t, err)
	require.NoError(t, host.Handle(t.Context(), code, owner, game.Input{ID: "leave", Command: rules.Command{Type: "leave"}}))
	_, err = host.Snapshot(code, owner.ID, 0)
	assert.Error(t, err)
	assert.Empty(t, host.List())
}
