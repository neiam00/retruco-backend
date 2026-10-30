package game

import (
	"encoding/json"
	"io"
	"log/slog"
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestImmutableHistoryPreservesOldPublicationsAndSnapshotOwnership(t *testing.T) {
	state, err := rules.New(rules.Config{Players: 4, Target: 24}, 0, rules.Deck())
	require.NoError(t, err)
	team := 1
	m := &Match{Code: "history", Config: state.Config, State: &state, Seats: make([]Seat, 4), Recent: map[string]uint64{},
		Chat: make([]Chat, 100, 150), Events: make([]GameEvent, 1, 20), CantoHistory: make([]GameEvent, 1, 30)}
	for i := range m.Chat {
		m.Chat[i] = Chat{ID: uint64(i + 1), Text: "retained"}
	}
	m.Events[0] = GameEvent{ID: 101, Event: rules.Event{Type: "privo", Team: &team}}
	m.CantoHistory[0] = m.Events[0]
	h := New(nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
	r := newRoom(m)
	h.rooms[m.Code] = r
	old, err := h.Publication(m.Code, "watcher", 0)
	require.NoError(t, err)
	before, err := json.Marshal(old)
	require.NoError(t, err)

	next := clone(m)
	next.Sequence = 102
	next.appendChat(Chat{ID: 102, Text: "new"})
	next.recordEvents([]rules.Event{{Type: "call", Kind: "truco"}, {Type: "play"}})
	write, err := encodeMatchWrite(next, m)
	require.NoError(t, err)
	assert.True(t, write.ChatChanged)
	assert.True(t, write.CantoHistoryChanged)
	assert.Equal(t, uint64(1), old.Chat[0].ID)
	assert.Len(t, old.CantoHistory, 1)
	assert.Len(t, old.Events, 1)
	after, err := json.Marshal(old)
	require.NoError(t, err)
	assert.Equal(t, string(before), string(after), "old readers survive history rollover and append")

	// Publish the successor, then mutate a caller-owned snapshot, including the
	// pointer nested in a rules event. Neither shared publication may change.
	r.match = next
	r.notify()
	current, err := h.Publication(m.Code, "watcher", 0)
	require.NoError(t, err)
	view, err := h.Snapshot(m.Code, "watcher", 0)
	require.NoError(t, err)
	view.Chat[0].Text = "caller mutation"
	view.CantoHistory[0].Event.Kind = "caller mutation"
	*view.CantoHistory[0].Event.Team = 0
	assert.Equal(t, "retained", current.Chat[0].Text)
	assert.Equal(t, 1, *current.CantoHistory[0].Event.Team)
	assert.Equal(t, 1, *old.CantoHistory[0].Event.Team)

	unchanged := clone(next)
	unchanged.Sequence++
	write, err = encodeMatchWrite(unchanged, next)
	require.NoError(t, err)
	assert.False(t, write.ChatChanged)
	assert.False(t, write.CantoHistoryChanged)
	assert.Nil(t, write.Chat)
	assert.Nil(t, write.CantoHistory)
}

func TestTrackedHistoryAfterRestoreKeepsDurableShape(t *testing.T) {
	for _, raw := range []string{`{"Code":"restored","Chat":null,"CantoHistory":null}`, `{"Code":"restored","Chat":[],"CantoHistory":[]}`} {
		var before Match
		require.NoError(t, json.Unmarshal([]byte(raw), &before))
		newRoom(&before)
		next := clone(&before)
		write, err := encodeMatchWrite(next, &before)
		require.NoError(t, err)
		assert.False(t, write.ChatChanged)
		assert.False(t, write.CantoHistoryChanged)
		full, err := encodeMatchWrite(next, nil)
		require.NoError(t, err)
		if before.Chat == nil {
			assert.Equal(t, "null", string(full.Chat))
		} else {
			assert.Equal(t, "[]", string(full.Chat))
		}
		next.appendChat(Chat{ID: 1, Text: "after restore"})
		write, err = encodeMatchWrite(next, &before)
		require.NoError(t, err)
		assert.True(t, write.ChatChanged)
		assert.False(t, write.CantoHistoryChanged)
		assert.Empty(t, before.Chat)
	}
}
