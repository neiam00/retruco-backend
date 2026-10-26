package game

import (
	"sync"
	"time"

	"github.com/neiam00/retruco-backend/rules"
)

// room owns all mutable state for one game. Hold mu while reading or changing
// any field, including across persistence: only committed state is published.
// A nil match marks a removed room, so callers that already found it cannot
// revive it. Subscription/disconnect closures retain this exact room instance.
type room struct {
	mu              sync.Mutex
	match           *Match
	listeners       map[chan struct{}]bool
	connections     map[string]playerConnection
	controller      string
	connectionOrder uint64
	deadline        time.Time
	recovering      bool
}

func newRoom(match *Match) *room {
	r := &room{
		match:       match,
		listeners:   make(map[chan struct{}]bool),
		connections: make(map[string]playerConnection),
	}
	r.updateRoomDeadline(match)
	return r
}

// Never wait for a room while holding Host.mu. The only nested lock order is
// room.mu -> Host.mu, used to publish listing/capacity changes after persistence.
func (h *Host) lockRoom(code string) *room {
	h.mu.RLock()
	r := h.rooms[code]
	h.mu.RUnlock()
	if r == nil {
		return nil
	}
	r.mu.Lock()
	if r.match == nil {
		r.mu.Unlock()
		return nil
	}
	return r
}

func activeMatch(match *Match) bool {
	return match != nil && !match.Expired && (match.State == nil || match.State.Phase != rules.MatchOver)
}

// Caller holds Host.mu and, for an already published room, room.mu. Listings
// contain values only, allowing lobby reads without waiting on any room's I/O.
func (h *Host) updateListing(match *Match) {
	delete(h.listings, match.Code)
	if !activeMatch(match) || match.Private || occupiedSeats(match) == 0 {
		return
	}
	listing := Listing{
		Code: match.Code, Players: occupiedSeats(match),
		CPUPlayers: len(match.Options.CPUSeats), Capacity: match.Config.Players,
		Target: match.Config.Target, FlorRequired: match.Config.FlorRequired,
		Started: match.State != nil, Host: match.Seats[0].Name,
	}
	if listing.Host == "" {
		listing.Host = match.CreatorName
	}
	h.listings[match.Code] = listing
}
