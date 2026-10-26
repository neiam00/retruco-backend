package game

import (
	"context"
	"fmt"
	"time"

	"github.com/neiam00/retruco-backend/rules"
)

const roomReconnectGrace = 10 * time.Second
const restoreReconnectGrace = 2 * time.Minute

func (r *room) hasConnectedPlayer(match *Match) bool {
	for _, connection := range r.connections {
		seat := seatOf(match, connection.userID)
		if seat >= 0 && !match.Seats[seat].CPU {
			return true
		}
	}
	return false
}

// Caller holds r.mu. Only seated humans keep a room alive. A second
// browser tab counts as a live connection; spectators and CPU seats do not.
func (r *room) updateRoomDeadline(match *Match) {
	if r.hasConnectedPlayer(match) {
		r.deadline = time.Time{}
		return
	}
	if r.deadline.IsZero() {
		r.deadline = time.Now().Add(roomReconnectGrace)
	}
}

// Caller holds r.mu. Persist closure before removing access in memory so a
// successful removal cannot reappear when the server restarts. Started games
// remain in account history as abandoned, without awarding either team a win.
func (h *Host) closeVacantRoom(ctx context.Context, r *room, match *Match) error {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if match.State == nil {
		if err := h.repository.DeleteLobby(ctx, match.Code); err != nil {
			h.log.Error("delete lobby", "code", match.Code, "error", err)
			return fmt.Errorf("no se pudo cerrar la mesa. Inténtalo de nuevo")
		}
	} else if match.State.Phase != rules.MatchOver {
		next := clone(match)
		next.Expired = true
		next.Deadline = time.Time{}
		if err := h.save(ctx, next); err != nil {
			return err
		}
	}
	h.mu.Lock()
	delete(h.rooms, match.Code)
	delete(h.listings, match.Code)
	if activeMatch(r.match) {
		h.active--
	}
	h.notifyListings()
	h.mu.Unlock()
	r.match = nil
	r.deadline = time.Time{}
	r.controller = ""
	clear(r.connections)
	r.notify()
	return nil
}

func (h *Host) SubscribeListings() (<-chan struct{}, func()) {
	h.mu.Lock()
	defer h.mu.Unlock()
	updates := make(chan struct{}, 1)
	h.listingListeners[updates] = true
	return updates, func() {
		h.mu.Lock()
		defer h.mu.Unlock()
		delete(h.listingListeners, updates)
	}
}

// Caller holds h.mu. Coalescing keeps slow browsers from blocking game actions.
func (h *Host) notifyListings() {
	h.listingJSON = ""
	for updates := range h.listingListeners {
		select {
		case updates <- struct{}{}:
		default:
		}
	}
}
