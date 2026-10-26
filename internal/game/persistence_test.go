package game

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMatchWritePreservesFullSnapshotAndOnlyEncodesChangedHistory(t *testing.T) {
	state, err := rules.New(rules.Config{Players: 4, Target: 24}, 0, rules.Deck())
	require.NoError(t, err)
	before := &Match{
		Code: "partitioned", CreatorID: "owner", CreatorName: "Ana", Private: true,
		Config: state.Config, State: &state, Deadline: time.Now().UTC(),
		Seats:        []Seat{{ID: "owner", Name: "Ana"}},
		Chat:         []Chat{{ID: 1, Name: "Ana", Text: "Retained chat"}},
		CantoHistory: []GameEvent{{ID: 2, Hand: 1, Event: rules.Event{Type: "call", Kind: "truco"}}},
		Recent:       map[string]uint64{"owner:request-1": 2}, Sequence: 2,
		LastActivity: time.Now().UTC(),
	}
	initial, err := encodeMatchWrite(before, nil)
	require.NoError(t, err)
	require.True(t, initial.ChatChanged)
	require.True(t, initial.CantoHistoryChanged)
	var restored map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(initial.Payload, &restored))
	assert.NotContains(t, restored, "Chat")
	assert.NotContains(t, restored, "CantoHistory")
	restored["Chat"], restored["CantoHistory"] = initial.Chat, initial.CantoHistory
	full, err := json.Marshal(before)
	require.NoError(t, err)
	combined, err := json.Marshal(restored)
	require.NoError(t, err)
	assert.JSONEq(t, string(full), string(combined), "the split must preserve all durable fields, including hidden state and request receipts")

	next := clone(before)
	next.Sequence++
	next.Recent["owner:request-2"] = next.Sequence
	write, err := encodeMatchWrite(next, before)
	require.NoError(t, err)
	assert.False(t, write.ChatChanged)
	assert.False(t, write.CantoHistoryChanged)
	assert.Nil(t, write.Chat)
	assert.Nil(t, write.CantoHistory)

	// Bounded history can change without growing: comparing only length or the
	// rules revision would silently discard new messages after the buffer fills.
	next.Chat[0] = Chat{ID: 3, Name: "Beto", Text: "Replacement chat"}
	next.CantoHistory[0].Event.Kind = "retruco"
	write, err = encodeMatchWrite(next, before)
	require.NoError(t, err)
	assert.True(t, write.ChatChanged)
	assert.True(t, write.CantoHistoryChanged)
	assert.JSONEq(t, `[{"id":3,"name":"Beto","text":"Replacement chat","spectator":false}]`, string(write.Chat))
}

func TestMatchWritePreservesNilAndEmptyHistory(t *testing.T) {
	for _, tc := range []struct {
		name     string
		previous *Match
		next     *Match
		want     string
	}{
		{"nil_to_empty", &Match{}, &Match{Chat: []Chat{}, CantoHistory: []GameEvent{}}, "[]"},
		{"empty_to_nil", &Match{Chat: []Chat{}, CantoHistory: []GameEvent{}}, &Match{}, "null"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			write, err := encodeMatchWrite(tc.next, tc.previous)
			require.NoError(t, err)
			assert.True(t, write.ChatChanged)
			assert.True(t, write.CantoHistoryChanged)
			assert.Equal(t, tc.want, string(write.Chat))
			assert.Equal(t, tc.want, string(write.CantoHistory))
		})
	}
}
