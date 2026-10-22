package game

import (
	"context"
	"fmt"
	"time"

	"github.com/neiam00/retruco-backend/rules"
)

const roomReconnectGrace = 10 * time.Second

func (h *Host) hasConnectedPlayer(match *Match) bool {
	for _, connection := range h.connections[match.Code] {
		seat := seatOf(match, connection.userID)
		if seat >= 0 && !match.Seats[seat].CPU {
			return true
		}
	}
	return false
}

// Caller holds h.mu. Only seated humans keep a room alive. A second
// browser tab counts as a live connection; spectators and CPU seats do not.
func (h *Host) updateRoomDeadline(match *Match) {
	if h.hasConnectedPlayer(match) {
		delete(h.roomDeadlines, match.Code)
		return
	}
	if _, waiting := h.roomDeadlines[match.Code]; !waiting {
		h.roomDeadlines[match.Code] = time.Now().Add(roomReconnectGrace)
	}
}

// Caller holds h.mu. Persist closure before removing access in memory so a
// successful removal cannot reappear when the server restarts. Started games
// remain in account history as abandoned, without awarding either team a win.
func (h *Host) closeVacantRoom(ctx context.Context, match *Match) error {
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
	delete(h.matches, match.Code)
	delete(h.roomDeadlines, match.Code)
	delete(h.controllers, match.Code)
	delete(h.connections, match.Code)
	h.notify(match.Code)
	h.notifyListings()
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
	for updates := range h.listingListeners {
		select {
		case updates <- struct{}{}:
		default:
		}
	}
}
