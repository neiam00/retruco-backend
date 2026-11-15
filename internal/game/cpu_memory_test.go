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

func memoryCPUFixture(t *testing.T) (*game.Host, *memoryRepository, store.User, string) {
	t.Helper()
	repo := &memoryRepository{payloads: map[string][]byte{}}
	host := game.New(repo, slog.New(slog.NewTextHandler(io.Discard, nil)))
	user := store.User{ID: "owner", Name: "Ana"}
	code, err := host.Create(t.Context(), user, rules.Config{Players: 4, Target: 12}, false, game.RoomOptions{CPUSeats: []int{1, 2, 3}})
	require.NoError(t, err)
	t.Cleanup(host.ConnectPlayer(code, user.ID, "controller", 2, 1))
	for i := 0; i < 8; i++ {
		work := host.CPUWork(code, user.ID, "controller")
		if work.Kind == "signal" {
			return host, repo, user, code
		}
		if work.View != nil {
			require.NoError(t, host.HandleFromConnection(t.Context(), code, user, game.Input{ID: fmt.Sprint("setup-", i), CPUSeat: &work.Seat, CPUKind: work.Kind, CPUToken: work.Token, Revision: work.View.Revision, Command: work.View.Legal[0]}, "controller"))
		} else {
			snapshot, err := host.Snapshot(code, user.ID, 0)
			require.NoError(t, err)
			require.NotEmpty(t, snapshot.State.Legal)
			require.NoError(t, host.Handle(t.Context(), code, user, game.Input{ID: fmt.Sprint("setup-", i), Revision: snapshot.State.Revision, Command: snapshot.State.Legal[0]}))
		}
	}
	t.Fatal("could not reach signal window")
	return nil, nil, user, code
}

func TestCPUMemoryPagesAreAuthorizedAndDoNotChangeGame(t *testing.T) {
	host, repo, user, code := memoryCPUFixture(t)
	for i := 0; i < 45; i++ {
		kind := "perico"
		if i%2 == 1 {
			kind = "perica"
		}
		require.NoError(t, host.Handle(t.Context(), code, user, game.Input{ID: fmt.Sprint("human-", i), Signal: kind}))
	}
	work := host.CPUWork(code, user.ID, "controller")
	require.NoError(t, host.HandleFromConnection(t.Context(), code, user, signalInput(work, "none-1", "none"), "controller"))
	work = host.CPUWork(code, user.ID, "controller")
	require.Equal(t, 2, work.Seat)
	require.NotNil(t, work.Memory)
	require.True(t, work.Memory.Complete)
	require.Equal(t, 32, work.Memory.End)
	require.Greater(t, work.Memory.Total, 32)
	before, err := host.Snapshot(code, user.ID, 0)
	require.NoError(t, err)
	persisted := append([]byte(nil), repo.payloads[code]...)
	input := game.Input{ID: "page", CPUSeat: &work.Seat, CPUKind: "memory", CPUToken: work.Token, MemoryHand: &work.Memory.Hand, MemoryCursor: &work.Memory.End}
	require.Error(t, host.HandleFromConnection(t.Context(), code, user, input, "other"))
	require.NoError(t, host.HandleFromConnection(t.Context(), code, user, input, "controller"))
	next := host.CPUWork(code, user.ID, "controller")
	require.Equal(t, 32, next.Memory.Start)
	assert.NotEqual(t, work.Token, next.Token)
	after, err := host.Snapshot(code, user.ID, 0)
	require.NoError(t, err)
	assert.Equal(t, before.Sequence, after.Sequence)
	assert.Equal(t, before.Deadline, after.Deadline)
	assert.Equal(t, persisted, repo.payloads[code])
	var stored game.Match
	require.NoError(t, json.Unmarshal(persisted, &stored))
	assert.Equal(t, stored.CPUMemory.Seats[2], append(work.Memory.Events, next.Memory.Events...))
	encoded, err := json.Marshal(after)
	require.NoError(t, err)
	assert.NotContains(t, string(encoded), "CPUMemory")
	zero := 0
	input.ID = "reset"
	input.CPUToken = next.Token
	input.MemoryCursor = &zero
	require.NoError(t, host.HandleFromConnection(t.Context(), code, user, input, "controller"))
	assert.Equal(t, work.Memory, host.CPUWork(code, user.ID, "controller").Memory)
	input = signalInput(next, "stale-action", "none")
	require.Error(t, host.HandleFromConnection(t.Context(), code, user, input, "controller"))
}

func TestCPUMemorySilentDecisionAndReconnect(t *testing.T) {
	host, repo, user, code := memoryCPUFixture(t)
	work := host.CPUWork(code, user.ID, "controller")
	var before game.Match
	require.NoError(t, json.Unmarshal(repo.payloads[code], &before))
	input := signalInput(work, "none", "none")
	input.MemoryHand = &work.Memory.Hand
	input.MemoryCursor = &work.Memory.End
	repo.fail = true
	require.Error(t, host.HandleFromConnection(t.Context(), code, user, input, "controller"))
	assert.Equal(t, work, host.CPUWork(code, user.ID, "controller"), "failed persistence cannot consume memory or its cursor")
	repo.fail = false
	require.NoError(t, host.HandleFromConnection(t.Context(), code, user, input, "controller"))
	var after game.Match
	require.NoError(t, json.Unmarshal(repo.payloads[code], &after))
	for seat := 0; seat < 4; seat++ {
		if seat == work.Seat {
			assert.Len(t, after.CPUMemory.Seats[seat], len(before.CPUMemory.Seats[seat])+1)
		} else {
			assert.Equal(t, before.CPUMemory.Seats[seat], after.CPUMemory.Seats[seat])
		}
	}
	current := host.CPUWork(code, user.ID, "controller")
	restored := game.New(repo, slog.New(slog.NewTextHandler(io.Discard, nil)))
	require.NoError(t, restored.Restore(t.Context()))
	t.Cleanup(restored.ConnectPlayer(code, user.ID, "new-controller", 2, 1))
	replayed := restored.CPUWork(code, user.ID, "new-controller")
	assert.Equal(t, current.Memory, replayed.Memory)
	assert.NotEqual(t, current.MemoryKey, replayed.MemoryKey)
	require.Error(t, restored.HandleFromConnection(t.Context(), code, user, signalInput(current, "old-lease", "none"), "new-controller"))
	after.CPUMemoryVersion = 0
	repo.payloads[code], _ = json.Marshal(after)
	legacy := game.New(repo, slog.New(slog.NewTextHandler(io.Discard, nil)))
	require.NoError(t, legacy.Restore(t.Context()))
	t.Cleanup(legacy.ConnectPlayer(code, user.ID, "legacy", 2, 1))
	assert.False(t, legacy.CPUWork(code, user.ID, "legacy").Memory.Complete)
}
