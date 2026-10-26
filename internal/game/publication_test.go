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

func TestPublicationPreservesAudienceAndCommittedChanges(t *testing.T) {
	repository := &memoryRepository{payloads: make(map[string][]byte)}
	host := game.New(repository, slog.New(slog.NewTextHandler(io.Discard, nil)))
	owner := store.User{ID: "owner", Name: "Owner"}
	code, err := host.Create(t.Context(), owner, rules.Config{Players: 2, Target: 12}, false, game.RoomOptions{CPUSeats: []int{1}})
	require.NoError(t, err)
	read := func(user string, perspective int) (*game.Publication, game.Snapshot) {
		p, err := host.Publication(code, user, perspective)
		require.NoError(t, err)
		var value game.Snapshot
		require.NoError(t, json.Unmarshal(p.Data, &value))
		return p, value
	}
	before, seated := read(owner.ID, 1)
	_, spectator := read("watcher", 1)
	assert.Equal(t, 0, seated.Seat)
	assert.Equal(t, 0, seated.Perspective, "a player cannot request another player's private view")
	assert.Equal(t, -1, spectator.Seat)
	assert.Equal(t, 1, spectator.Perspective)
	assert.Empty(t, spectator.State.Legal)
	assert.NotContains(t, string(before.Data), `"serverTime"`)
	assert.NotContains(t, string(before.Data), `"chat"`)

	// Chat does not change rules revision but must invalidate shared views.
	require.NoError(t, host.Handle(t.Context(), code, owner, game.Input{ID: "chat-1", Chat: "durable message"}))
	after, withChat := read(owner.ID, 0)
	assert.Equal(t, seated.State.Revision, withChat.State.Revision)
	require.Len(t, after.Chat, 1)
	assert.Empty(t, before.Chat, "already published immutable data must not change")

	repository.mu.Lock()
	repository.fail = true
	repository.mu.Unlock()
	require.Error(t, host.Handle(t.Context(), code, owner, game.Input{ID: "chat-fails", Chat: "must not publish"}))
	failed, _ := read(owner.ID, 0)
	assert.Equal(t, after.Chat, failed.Chat)
	assert.JSONEq(t, string(after.Data), string(failed.Data))
	repository.mu.Lock()
	repository.fail = false
	repository.mu.Unlock()

	// CPU-controller changes and deployment notices do not advance game state.
	disconnect := host.ConnectPlayer(code, owner.ID, "controller")
	_, connected := read(owner.ID, 0)
	assert.Equal(t, 0, connected.CPUController)
	disconnect()
	_, disconnected := read(owner.ID, 0)
	assert.Equal(t, -1, disconnected.CPUController)
	host.SetUpdating(true)
	_, updating := read(owner.ID, 0)
	assert.True(t, updating.Updating)
	assert.Empty(t, updating.State.Legal)
	host.SetUpdating(false)
	_, resumed := read(owner.ID, 0)
	assert.False(t, resumed.Updating)
}
