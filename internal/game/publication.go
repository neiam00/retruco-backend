package game

import (
	"encoding/json"
	"fmt"
)

// Publication is immutable after construction. Transport readers must not modify
// its byte or history slices. Controller work and server time are deliberately
// outside this shared view: both are specific to a connection or send instant.
type Publication struct {
	Data         json.RawMessage
	Chat         []Chat
	Events       []GameEvent
	CantoHistory []GameEvent
	updating     bool
}

// Publication shares rule projection and JSON encoding across identical
// audiences. Seated players and spectators never share a cache entry, even
// when their chosen perspective is the same. notify invalidates every view,
// including changes that do not advance the durable rules revision.
func (h *Host) Publication(code, userID string, perspective int) (*Publication, error) {
	r := h.lockRoom(code)
	if r == nil {
		return nil, fmt.Errorf("no encontramos esa mesa")
	}
	defer r.mu.Unlock()
	seat := seatOf(r.match, userID)
	if seat >= 0 {
		perspective = seat
	}
	if perspective < 0 || perspective >= len(r.match.Seats) {
		perspective = 0
	}
	key := perspective * 2
	if seat < 0 {
		key++
	}
	updating := h.Updating()
	if cached := r.publications[key]; cached != nil && cached.updating == updating {
		return cached, nil
	}
	value := snapshotViewLocked(r, code, userID, perspective, updating)
	// Shadow these embedded fields so histories are transmitted separately and
	// a fresh serverTime can be appended for every recipient.
	data, err := json.Marshal(struct {
		*Snapshot
		Chat         *struct{} `json:"chat,omitempty"`
		Events       *struct{} `json:"events,omitempty"`
		CantoHistory *struct{} `json:"cantoHistory,omitempty"`
		ServerTime   *struct{} `json:"serverTime,omitempty"`
	}{Snapshot: &value})
	if err != nil {
		return nil, err
	}
	p := &Publication{Data: data, Chat: wireHistory(r.match.Chat), Events: wireHistory(r.match.Events),
		CantoHistory: wireHistory(r.match.CantoHistory), updating: updating}
	r.publications[key] = p
	return p, nil
}
