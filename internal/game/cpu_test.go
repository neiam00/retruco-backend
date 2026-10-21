package game_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCPUSeatSelectionKeepsHumanSlotsAndPersists(t *testing.T) {
	for _, cpuSeats := range [][]int{{1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}} {
		t.Run(fmt.Sprint(cpuSeats), func(t *testing.T) {
			repository := &memoryRepository{payloads: make(map[string][]byte)}
			log := slog.New(slog.NewTextHandler(io.Discard, nil))
			host := game.New(repository, log)
			creator := store.User{ID: "creator", Name: "Ana"}
			code, err := host.Create(t.Context(), creator, rules.Config{Players: 4, Target: 12}, false, game.RoomOptions{CPUSeats: cpuSeats})
			require.NoError(t, err)
			list := host.List()
			require.Len(t, list, 1)
			assert.Equal(t, len(cpuSeats), list[0].CPUPlayers)
			assert.Equal(t, len(cpuSeats)+1, list[0].Players)
			assert.Equal(t, len(cpuSeats) == 3, list[0].Started)
			view, err := host.Snapshot(code, creator.ID, 0)
			require.NoError(t, err)
			require.Len(t, view.Seats, 4)
			for seat := 1; seat < 4; seat++ {
				expectedCPU := false
				for _, selected := range cpuSeats {
					expectedCPU = expectedCPU || selected == seat
				}
				assert.Equal(t, expectedCPU, view.Seats[seat].CPU)
				if !expectedCPU {
					assert.Empty(t, view.Seats[seat].ID)
					user := store.User{ID: fmt.Sprintf("human-%d", seat), Name: "Humano"}
					require.NoError(t, host.Join(t.Context(), code, user, nil))
					joined, snapshotErr := host.Snapshot(code, user.ID, 0)
					require.NoError(t, snapshotErr)
					assert.Equal(t, seat, joined.Seat)
				}
			}
			restored := game.New(repository, log)
			require.NoError(t, restored.Restore(t.Context()))
			saved, err := restored.Snapshot(code, creator.ID, 0)
			require.NoError(t, err)
			require.NotNil(t, saved.State)
			assert.Equal(t, cpuSeats, saved.Options.CPUSeats)
			assert.True(t, saved.CPUPaused)
			assert.Equal(t, -1, saved.CPUController)
		})
	}
}

func TestCPUControllerTransfersAndRejectsStaleOrHumanSeatCommands(t *testing.T) {
	repository := &memoryRepository{payloads: make(map[string][]byte)}
	host := game.New(repository, slog.New(slog.NewTextHandler(io.Discard, nil)))
	ana, beto := store.User{ID: "ana", Name: "Ana"}, store.User{ID: "beto", Name: "Beto"}
	code, err := host.Create(t.Context(), ana, rules.Config{Players: 4, Target: 12}, false, game.RoomOptions{CPUSeats: []int{1, 3}})
	require.NoError(t, err)
	require.NoError(t, host.Join(t.Context(), code, beto, nil))
	disconnectAna := host.ConnectPlayer(code, ana.ID, "ana-tab")
	defer disconnectAna()
	disconnectSecondTab := host.ConnectPlayer(code, ana.ID, "ana-other-tab")
	defer disconnectSecondTab()
	disconnectBeto := host.ConnectPlayer(code, beto.ID, "beto-tab")
	defer disconnectBeto()
	disconnectWatcher := host.ConnectPlayer(code, "watcher", "watcher-tab")
	defer disconnectWatcher()
	assert.True(t, host.CPUWork(code, ana.ID, "ana-tab").Controlled)
	assert.False(t, host.CPUWork(code, ana.ID, "ana-other-tab").Controlled)
	assert.False(t, host.CPUWork(code, beto.ID, "beto-tab").Controlled)
	assert.False(t, host.CPUWork(code, "watcher", "watcher-tab").Controlled)

	// Reach a genuine CPU turn without depending on the random dealer.
	players := map[int]store.User{0: ana, 2: beto}
	for step := 0; step < 5 && host.CPUWork(code, ana.ID, "ana-tab").View == nil; step++ {
		view, snapshotErr := host.Snapshot(code, ana.ID, 0)
		require.NoError(t, snapshotErr)
		player := players[view.State.Turn]
		active, snapshotErr := host.Snapshot(code, player.ID, 0)
		require.NoError(t, snapshotErr)
		require.NotEmpty(t, active.State.Legal)
		require.NoError(t, host.Handle(t.Context(), code, player, game.Input{
			ID: fmt.Sprint(step), Revision: active.State.Revision, Command: active.State.Legal[0],
		}))
	}
	work := host.CPUWork(code, ana.ID, "ana-tab")
	require.NotNil(t, work.View)
	require.NotEmpty(t, work.View.Legal)
	snapshot, err := host.Snapshot(code, ana.ID, 0)
	require.NoError(t, err)
	cpuView, err := host.Snapshot(code, snapshot.Seats[work.Seat].ID, 0)
	require.NoError(t, err)
	assert.Equal(t, cpuView.State.Hand, work.View.Hand, "send the acting CPU's hand, never the controlling human's")
	input := game.Input{ID: "cpu-move", CPUSeat: &work.Seat, Revision: work.View.Revision, Command: work.View.Legal[0]}
	for _, seat := range []int{-1, 0, 2, 4} {
		bad := input
		bad.CPUSeat = &seat
		require.Error(t, host.HandleFromConnection(t.Context(), code, ana, bad, "ana-tab"))
	}
	require.Error(t, host.Handle(t.Context(), code, ana, input))
	require.Error(t, host.HandleFromConnection(t.Context(), code, beto, input, "ana-tab"))
	disconnectAna()
	assert.True(t, host.CPUWork(code, ana.ID, "ana-other-tab").Controlled)
	disconnectSecondTab()
	assert.True(t, host.CPUWork(code, beto.ID, "beto-tab").Controlled)
	require.Error(t, host.HandleFromConnection(t.Context(), code, ana, input, "ana-tab"))
	require.NoError(t, host.HandleFromConnection(t.Context(), code, beto, input, "beto-tab"))
	after, err := host.Snapshot(code, beto.ID, 0)
	require.NoError(t, err)
	assert.Equal(t, work.View.Revision+1, after.State.Revision)
	require.NoError(t, host.HandleFromConnection(t.Context(), code, beto, input, "beto-tab"))
	duplicate, err := host.Snapshot(code, beto.ID, 0)
	require.NoError(t, err)
	assert.Equal(t, after.State.Revision, duplicate.State.Revision)

	disconnectBeto()
	paused, err := host.Snapshot(code, ana.ID, 0)
	require.NoError(t, err)
	assert.True(t, paused.CPUPaused)
	assert.Equal(t, -1, paused.CPUController)
	assert.False(t, host.CPUWork(code, "watcher", "watcher-tab").Controlled)
	reconnect := host.ConnectPlayer(code, ana.ID, "ana-returned")
	defer reconnect()
	resumed, err := host.Snapshot(code, ana.ID, 0)
	require.NoError(t, err)
	assert.False(t, resumed.CPUPaused)
	assert.Equal(t, paused.State.Revision, resumed.State.Revision)
	assert.WithinDuration(t, time.Now().Add(30*time.Second), resumed.Deadline, time.Second)
	require.NoError(t, host.HandleFromConnection(t.Context(), code, ana, game.Input{
		ID: "finish", Revision: resumed.State.Revision, Command: rules.Command{Type: "leave"},
	}, "ana-returned"))
	assert.False(t, host.CPUWork(code, ana.ID, "ana-returned").Controlled)
	finished, err := host.Snapshot(code, ana.ID, 0)
	require.NoError(t, err)
	assert.False(t, finished.CPUPaused)
	assert.Equal(t, -1, finished.CPUController)
}

func TestUnattendedCPURoomDoesNotAdvanceExpiredDeadline(t *testing.T) {
	repository := &memoryRepository{payloads: make(map[string][]byte)}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	host := game.New(repository, log)
	ana := store.User{ID: "ana", Name: "Ana"}
	code, err := host.Create(t.Context(), ana, rules.Config{Players: 2, Target: 12}, false, game.RoomOptions{CPUSeats: []int{1}})
	require.NoError(t, err)
	var persisted game.Match
	require.NoError(t, json.Unmarshal(repository.payloads[code], &persisted))
	persisted.Deadline = time.Now().Add(-time.Hour)
	repository.payloads[code], err = json.Marshal(persisted)
	require.NoError(t, err)
	restored := game.New(repository, log)
	require.NoError(t, restored.Restore(t.Context()))
	ctx, cancel := context.WithCancel(t.Context())
	stopped := make(chan struct{})
	go func() { defer close(stopped); restored.Run(ctx) }()
	t.Cleanup(func() { cancel(); <-stopped })
	require.Never(t, func() bool {
		view, snapshotErr := restored.Snapshot(code, ana.ID, 0)
		return snapshotErr != nil || view.State.Revision != persisted.State.Revision
	}, 600*time.Millisecond, 20*time.Millisecond)
}

func TestCPUSeatValidationAndLobbyLeaveDoNotMoveReservedSeats(t *testing.T) {
	repository := &memoryRepository{payloads: make(map[string][]byte)}
	host := game.New(repository, slog.New(slog.NewTextHandler(io.Discard, nil)))
	ana := store.User{ID: "ana", Name: "Ana"}
	for _, seats := range [][]int{{0}, {-1}, {4}, {1, 1}} {
		_, err := host.Create(t.Context(), ana, rules.Config{Players: 4, Target: 12}, false, game.RoomOptions{CPUSeats: seats})
		require.Error(t, err)
	}
	_, err := host.Create(t.Context(), ana, rules.Config{Players: 2, Target: 12}, false, game.RoomOptions{CPUSeats: []int{2}})
	require.Error(t, err)
	code, err := host.Create(t.Context(), ana, rules.Config{Players: 4, Target: 12}, false, game.RoomOptions{CPUSeats: []int{3}})
	require.NoError(t, err)
	// Another human keeps this waiting room alive when Ana leaves.
	require.NoError(t, host.Join(t.Context(), code, store.User{ID: "beto", Name: "Beto"}, nil))
	disconnect := host.ConnectPlayer(code, "beto", "beto-tab")
	defer disconnect()
	require.NoError(t, host.Handle(t.Context(), code, ana, game.Input{ID: "leave", Command: rules.Command{Type: "leave"}}))
	view, err := host.Snapshot(code, ana.ID, 0)
	require.NoError(t, err)
	assert.Empty(t, view.Seats[0].ID)
	assert.True(t, view.Seats[3].CPU)
	require.NoError(t, host.Join(t.Context(), code, ana, nil))
	again, err := host.Snapshot(code, ana.ID, 0)
	require.NoError(t, err)
	assert.Equal(t, 0, again.Seat)
	assert.True(t, again.Seats[3].CPU)
}
