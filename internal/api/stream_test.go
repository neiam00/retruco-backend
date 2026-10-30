package api

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type streamMessage struct {
	Type    string          `json:"type"`
	Data    json.RawMessage `json:"data"`
	CPU     game.CPUWork    `json:"cpu"`
	Stream  streamPosition  `json:"stream"`
	History *streamHistory  `json:"history,omitempty"`
}

// Decode the actual wire encoder for the existing reconstruction contracts.
func (s *gameStream) message(p *game.Publication, cpu game.CPUWork, baseline bool, now time.Time) (streamMessage, error) {
	payload, err := s.encode(p, cpu, baseline, now)
	if err != nil {
		return streamMessage{}, err
	}
	var result streamMessage
	err = json.Unmarshal(payload, &result)
	return result, err
}

func TestStreamEncodingMatchesPreviousWireBytes(t *testing.T) {
	data, err := json.Marshal(map[string]any{
		"sequence": 7, "signals": []game.Signal{}, "host": "<Ana> & Beto\u2028\u2029",
		"private": true, "perspective": 2, "state": map[string]any{"hand": []string{"private-card"}},
	})
	require.NoError(t, err)
	originalData := string(data)
	team := 1
	publication := &game.Publication{Data: data,
		Chat:         []game.Chat{{ID: 2, Text: "<first> & second\n\"quoted\"", Name: "é"}},
		Events:       []game.GameEvent{{ID: 3, Event: rules.Event{Type: "call", Team: &team}}},
		CantoHistory: []game.GameEvent{},
	}
	stamp := time.Date(2026, 10, 26, 1, 2, 3, 456000789, time.FixedZone("offset", -4*60*60))
	actual := gameStream{epoch: "connection<epoch>\""}
	previous := gameStream{epoch: actual.epoch}
	for _, step := range []struct {
		name        string
		publication *game.Publication
		baseline    bool
		cpu         game.CPUWork
	}{
		{"baseline", publication, false, game.CPUWork{}},
		{"unchanged histories", publication, false, game.CPUWork{Controlled: true, Seat: 2, View: &rules.View{Revision: 4}}},
		{"appended histories", &game.Publication{Data: data, Chat: append(append([]game.Chat{}, publication.Chat...), game.Chat{ID: 4, Text: "new"}), Events: publication.Events, CantoHistory: publication.CantoHistory}, false, game.CPUWork{}},
		{"window reset", &game.Publication{Data: data, Chat: []game.Chat{{ID: 10, Text: "after gap"}}, Events: []game.GameEvent{}, CantoHistory: nil}, false, game.CPUWork{}},
		{"explicit resync", publication, true, game.CPUWork{}},
	} {
		t.Run(step.name, func(t *testing.T) {
			wantMessage, err := referenceStreamMessage(&previous, step.publication, step.cpu, step.baseline, stamp)
			require.NoError(t, err)
			want, err := json.Marshal(wantMessage)
			require.NoError(t, err)
			got, err := actual.encode(step.publication, step.cpu, step.baseline, stamp)
			require.NoError(t, err)
			assert.Equal(t, string(want), string(got), "optimized encoding must preserve exact envelope, escaping and history bytes")
			assert.Equal(t, originalData, string(step.publication.Data), "shared publication bytes must remain immutable")
		})
		stamp = stamp.Add(time.Second)
	}
}

func TestStreamEncodingErrorDoesNotAdvanceCursor(t *testing.T) {
	s := gameStream{epoch: "epoch"}
	p := &game.Publication{Data: json.RawMessage(`{"sequence":1}`)}
	_, err := s.encode(p, game.CPUWork{}, false, time.Date(10000, 1, 1, 0, 0, 0, 0, time.UTC))
	require.Error(t, err)
	assert.Zero(t, s.sequence)
	assert.Nil(t, s.previous)
	message, err := s.message(p, game.CPUWork{}, false, time.Unix(0, 0))
	require.NoError(t, err)
	assert.Equal(t, uint64(1), message.Stream.Sequence)
	assert.Equal(t, "snapshot", message.Type)
}

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
