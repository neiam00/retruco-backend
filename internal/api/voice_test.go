package api

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVoiceSignalingCannotCrossRoomsOrSpoofIdentity(t *testing.T) {
	hub := newVoiceHub()
	newMember := func(id string) *voiceMember {
		_, cancel := context.WithCancel(t.Context())
		t.Cleanup(cancel)
		return &voiceMember{
			peer:   voicePeer{ID: id, Name: id, CanSpeak: true},
			userID: id, inbox: make(chan voiceMessage, 16), cancel: cancel,
		}
	}
	ana, beto, carla := newMember("ana"), newMember("beto"), newMember("carla")
	hub.join("ROOM-1234", ana, nil)
	hub.join("ROOM-1234", beto, nil)
	hub.join("ELSE-1234", carla, nil)
	require.Equal(t, "welcome", (<-ana.inbox).Type)
	require.Equal(t, "peers", (<-ana.inbox).Type)
	require.Equal(t, "welcome", (<-beto.inbox).Type)
	require.Equal(t, "welcome", (<-carla.inbox).Type)

	offer := json.RawMessage(`{"type":"offer","sdp":"test"}`)
	hub.relay("ROOM-1234", "ana", voiceMessage{
		Type: "peers", From: "carla", To: "beto", Description: offer,
		Peers: []voicePeer{{ID: "intruder", CanSpeak: true}},
	})
	received := <-beto.inbox
	assert.Equal(t, "signal", received.Type)
	assert.Equal(t, "ana", received.From)
	assert.Equal(t, offer, received.Description)
	assert.Empty(t, received.Peers)
	hub.relay("ROOM-1234", "ana", voiceMessage{To: "carla", Description: offer})
	hub.relay("ROOM-1234", "carla", voiceMessage{To: "beto", Description: offer})
	assert.Empty(t, carla.inbox)
	assert.Empty(t, beto.inbox)

	hub.leave("ROOM-1234", "ana")
	roster := <-beto.inbox
	require.Len(t, roster.Peers, 1)
	assert.Equal(t, "beto", roster.Peers[0].ID)
	hub.leave("ROOM-1234", "beto")
	assert.NotContains(t, hub.rooms, "ROOM-1234")
}

func TestVoiceNewTabReplacesOldConnectionWithoutRemovingNewPeer(t *testing.T) {
	hub := newVoiceHub()
	oldCtx, oldCancel := context.WithCancel(t.Context())
	defer oldCancel()
	previous := &voiceMember{peer: voicePeer{ID: "old"}, userID: "ana", inbox: make(chan voiceMessage, 4), cancel: oldCancel}
	nextCtx, nextCancel := context.WithCancel(t.Context())
	defer nextCancel()
	next := &voiceMember{peer: voicePeer{ID: "new"}, userID: "ana", inbox: make(chan voiceMessage, 4), cancel: nextCancel}
	hub.join("ROOM-1234", previous, nil)
	hub.join("ROOM-1234", next, nil)
	require.ErrorIs(t, oldCtx.Err(), context.Canceled)
	hub.leave("ROOM-1234", "old")
	assert.Same(t, next, hub.rooms["ROOM-1234"]["new"])
	require.NoError(t, nextCtx.Err())
	// Backpressure closes a stalled connection; it does not block other rooms.
	for range 5 {
		deliverVoice(next, voiceMessage{Type: "peers"})
	}
	assert.ErrorIs(t, nextCtx.Err(), context.Canceled)
}
