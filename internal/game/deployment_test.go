package game_test

import (
	"context"
	"testing"
	"testing/synctest"
	"time"

	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDeploymentNoticePausesPlayAndVacancyTimersUntilCancelled(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		host, _, code, players := newHost(t, false)
		before := cut(t, host, code, players)
		disconnect := host.ConnectPlayer(code, players[0].ID, "before-update")
		updates, unsubscribe := host.Subscribe(code)
		defer unsubscribe()
		host.SetUpdating(true)
		require.Len(t, updates, 1, "connected players must receive the notice immediately")
		<-updates
		paused, err := host.Snapshot(code, players[0].ID, 0)
		require.NoError(t, err)
		assert.True(t, paused.Updating)
		assert.Empty(t, paused.State.Legal)
		assert.Equal(t, before.State.Revision, paused.State.Revision)
		require.ErrorContains(t, host.Handle(t.Context(), code, players[before.State.Turn], game.Input{
			ID: "paused-play", Revision: before.State.Revision, Command: rules.Command{Type: "play", Card: 0},
		}), "pausa")
		disconnect()
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		go host.Run(ctx)
		time.Sleep(time.Minute)
		synctest.Wait()
		stillPaused, err := host.Snapshot(code, players[0].ID, 0)
		require.NoError(t, err, "a deployment must not close the room when all sockets disconnect")
		assert.Equal(t, paused.State, stillPaused.State)
		host.SetUpdating(false)
		time.Sleep(9 * time.Second)
		synctest.Wait()
		defer host.ConnectPlayer(code, players[0].ID, "cancelled-update-return")()
		after, err := host.Snapshot(code, players[0].ID, 0)
		require.NoError(t, err)
		assert.False(t, after.Updating)
		assert.Equal(t, before.State, after.State)
		assert.WithinDuration(t, time.Now().Add(21*time.Second), after.Deadline, time.Millisecond)
		assert.NotEmpty(t, updates, "cancelling the notice must notify returning players")
	})
}
