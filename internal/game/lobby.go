package game

import (
	"context"
	"fmt"
	"time"
)

const lobbyReconnectGrace = 10 * time.Second

func (h *Host) hasConnectedPlayer(match *Match) bool {
	for _, connection := range h.connections[match.Code] {
		seat := seatOf(match, connection.userID)
		if seat >= 0 && !match.Seats[seat].CPU {
			return true
		}
	}
	return false
}

// Caller holds h.mu. Only seated humans keep a waiting room alive. A second
// browser tab counts as a live connection; spectators and CPU seats do not.
func (h *Host) updateLobbyDeadline(match *Match) {
	if match.State != nil || h.hasConnectedPlayer(match) {
		delete(h.lobbyDeadlines, match.Code)
		return
	}
	if _, waiting := h.lobbyDeadlines[match.Code]; !waiting {
		h.lobbyDeadlines[match.Code] = time.Now().Add(lobbyReconnectGrace)
	}
}

// Caller holds h.mu. Persist deletion before removing access in memory so a
// successful removal cannot reappear when the server restarts.
func (h *Host) deleteLobby(ctx context.Context, match *Match) error {
	if match.State != nil {
		return fmt.Errorf("a started match cannot be deleted as a vacant lobby")
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := h.repository.DeleteLobby(ctx, match.Code); err != nil {
		h.log.Error("delete lobby", "code", match.Code, "error", err)
		return fmt.Errorf("no se pudo cerrar la mesa. Inténtalo de nuevo")
	}
	delete(h.matches, match.Code)
	delete(h.lobbyDeadlines, match.Code)
	delete(h.controllers, match.Code)
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
