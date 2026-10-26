package api

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStreamReconstructsCoalescedAndTrimmedHistory(t *testing.T) {
	stream := gameStream{epoch: "connection-epoch"}
	now := time.Date(2026, 10, 26, 0, 0, 0, 0, time.UTC)
	publication := &game.Publication{Data: json.RawMessage(`{"sequence":7,"signals":[]}`),
		Chat:   []game.Chat{{ID: 2, Text: "first"}, {ID: 4, Text: "second"}},
		Events: []game.GameEvent{}, CantoHistory: []game.GameEvent{}}
	baseline, err := stream.message(publication, game.CPUWork{}, false, now)
	require.NoError(t, err)
	var decoded game.Snapshot
	require.NoError(t, json.Unmarshal(baseline.Data, &decoded))
	assert.Equal(t, publication.Chat, decoded.Chat)
	assert.Equal(t, now, decoded.ServerTime)
	assert.Equal(t, "snapshot", baseline.Type)
	assert.Equal(t, uint64(1), baseline.Stream.Sequence)
	assert.Zero(t, baseline.Stream.BaseSequence)

	// A one-slot notification channel can coalesce several accepted commands.
	// The retained suffix and all unseen entries must reconstruct the latest list.
	next := &game.Publication{Data: json.RawMessage(`{"sequence":11,"signals":[]}`),
		Chat:   []game.Chat{{ID: 4, Text: "second"}, {ID: 8, Text: "third"}, {ID: 11, Text: "fourth"}},
		Events: []game.GameEvent{}, CantoHistory: []game.GameEvent{}}
	update, err := stream.message(next, game.CPUWork{}, false, now.Add(time.Second))
	require.NoError(t, err)
	require.NotNil(t, update.History.Chat)
	assert.Equal(t, "game_update", update.Type)
	assert.Equal(t, uint64(1), update.Stream.BaseSequence)
	assert.Equal(t, uint64(2), update.Stream.Sequence)
	assert.False(t, update.History.Chat.Reset)
	rebuilt := append(decoded.Chat, update.History.Chat.Items...)
	rebuilt = rebuilt[len(rebuilt)-update.History.Chat.Keep:]
	assert.Equal(t, next.Chat, rebuilt)
	assert.NotContains(t, string(update.Data), `"chat"`)
	assert.Nil(t, update.History.CantoHistory)

	// A resync is a complete baseline even without another rules revision.
	resync, err := stream.message(next, game.CPUWork{}, true, now.Add(2*time.Second))
	require.NoError(t, err)
	assert.Equal(t, "snapshot", resync.Type)
	assert.Zero(t, resync.Stream.BaseSequence)
	require.NoError(t, json.Unmarshal(resync.Data, &decoded))
	assert.Equal(t, next.Chat, decoded.Chat)
	assert.Equal(t, now.Add(2*time.Second), decoded.ServerTime)
}

func TestHistoryChangesResetWhenRetainedWindowIsLost(t *testing.T) {
	for _, tc := range []struct {
		name    string
		current []game.Chat
	}{
		{"lost window", []game.Chat{{ID: 10}, {ID: 11}}},
		{"cleared", []game.Chat{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			delta := historyChange([]game.Chat{{ID: 1}, {ID: 2}}, tc.current, func(v game.Chat) uint64 { return v.ID })
			require.NotNil(t, delta)
			assert.True(t, delta.Reset)
			assert.Equal(t, tc.current, delta.Items)
			assert.Equal(t, len(tc.current), delta.Keep)
		})
	}
}
