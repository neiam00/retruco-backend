package game_test

import (
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

func TestHeadsUpHostRejectsSignalsAndDoesNotDelegateThem(t *testing.T) {
	repo := &memoryRepository{payloads: map[string][]byte{}}
	host := game.New(repo, slog.New(slog.NewTextHandler(io.Discard, nil)))
	user := store.User{ID: "owner", Name: "Audit"}
	code, err := host.Create(t.Context(), user, rules.Config{Players: 2, Target: 24}, false, game.RoomOptions{CPUSeats: []int{1}})
	require.NoError(t, err)
	t.Cleanup(host.ConnectPlayer(code, user.ID, "controller", 2))
	for step := 0; step < 3; step++ {
		snapshot, err := host.Snapshot(code, user.ID, 0)
		require.NoError(t, err)
		if snapshot.State.Phase == rules.Playing {
			break
		}
		work := host.CPUWork(code, user.ID, "controller")
		input := game.Input{ID: fmt.Sprint(step), Revision: snapshot.State.Revision}
		if work.View != nil {
			input.CPUSeat, input.CPUKind, input.CPUToken = &work.Seat, work.Kind, work.Token
			input.Command = work.View.Legal[0]
		} else {
			input.Command = snapshot.State.Legal[0]
		}
		require.NoError(t, host.HandleFromConnection(t.Context(), code, user, input, "controller"))
	}
	before, err := host.Snapshot(code, user.ID, 0)
	require.NoError(t, err)
	require.Equal(t, rules.Playing, before.State.Phase)
	assert.NotEqual(t, "signal", host.CPUWork(code, user.ID, "controller").Kind)
	persisted := append([]byte(nil), repo.payloads[code]...)
	require.ErrorContains(t, host.Handle(t.Context(), code, user, game.Input{ID: "signal", Signal: "perico"}), "2v2")
	after, err := host.Snapshot(code, user.ID, 0)
	require.NoError(t, err)
	assert.Equal(t, before.Sequence, after.Sequence)
	assert.Equal(t, persisted, repo.payloads[code])
	assert.Empty(t, after.Signals)
}
