package game

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sort"
	"time"

	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
)

// A table keeps its invitation code while every game owns a separate record.
// Older snapshots used the table code as their record key.
func recordCode(match *Match) string {
	if match.RecordCode != "" {
		return match.RecordCode
	}
	return match.Code
}

// Caller holds r.mu. Consent belongs to the finished game, so an old request
// cannot vote in (or play a move in) the next game.
func (h *Host) rematch(ctx context.Context, r *room, user store.User, input Input, key string) error {
	m := r.match
	seat := seatOf(m, user.ID)
	if seat < 0 || m.Seats[seat].CPU || input.CPUSeat != nil {
		return fmt.Errorf("solo los jugadores de la mesa pueden pedir revancha")
	}
	if m.Expired || m.State == nil || m.State.Phase != rules.MatchOver || input.Revision != m.State.Revision {
		return fmt.Errorf("la revancha se pide al terminar la partida")
	}
	if slices.Contains(m.RematchVotes, seat) {
		return nil
	}
	next := clone(m)
	next.RematchVotes = append(next.RematchVotes, seat)
	next.Sequence++
	next.Recent[key] = next.Sequence
	for other, player := range m.Seats {
		if player.ID != "" && !player.CPU && !slices.Contains(next.RematchVotes, other) {
			return h.commit(ctx, r, next)
		}
	}

	// Reserve active-room capacity across the storage operation, just as Create
	// does. Failed persistence leaves both the old result and votes unchanged.
	for range 10 {
		code, err := matchCode()
		if err != nil {
			return err
		}
		h.mu.Lock()
		if h.active+len(h.reservations) >= 1000 {
			h.mu.Unlock()
			return fmt.Errorf("no hay espacio para más mesas en este momento")
		}
		if h.reservations[code] || h.rooms[code] != nil {
			h.mu.Unlock()
			continue
		}
		h.reservations[code] = true
		h.mu.Unlock()

		next.RecordCode = code
		next.GameNumber = max(1, m.GameNumber) + 1
		next.RematchVotes = nil
		next.State = nil
		next.Signals = nil
		next.Events = nil
		next.CantoHistory = nil
		next.cantoRevision = &historyRevision{}
		err = startWhenFull(next)
		if err == nil {
			// Revisions remain monotonic across games for stale moves and CPU work.
			next.State.Revision = m.State.Revision + 1
			err = h.commit(ctx, r, next)
		}
		h.mu.Lock()
		delete(h.reservations, code)
		h.mu.Unlock()
		if errors.Is(err, store.ErrMatchCodeInUse) {
			continue
		}
		return err
	}
	return fmt.Errorf("no pudimos preparar la revancha. Inténtalo de nuevo")
}

type ResumableMatch struct {
	Code              string       `json:"code"`
	Config            rules.Config `json:"config"`
	Names             []string     `json:"names"`
	ReconnectDeadline time.Time    `json:"reconnectDeadline"`
	LastActivity      time.Time    `json:"lastActivity"`
}

// Resumable returns only tables in which this identity owns a human seat.
// Copy the registry before taking room locks to preserve the lock ordering.
func (h *Host) Resumable(userID string) []ResumableMatch {
	h.mu.RLock()
	rooms := make([]*room, 0, len(h.rooms))
	for _, r := range h.rooms {
		rooms = append(rooms, r)
	}
	h.mu.RUnlock()
	result := []ResumableMatch{}
	for _, r := range rooms {
		r.mu.Lock()
		m := r.match
		if activeMatch(m) && seatOf(m, userID) >= 0 && (r.deadline.IsZero() || time.Now().Before(r.deadline)) {
			entry := ResumableMatch{Code: m.Code, Config: m.Config, LastActivity: m.LastActivity, ReconnectDeadline: r.deadline, Names: []string{}}
			for _, seat := range m.Seats {
				if seat.ID != "" && seat.ID != userID {
					entry.Names = append(entry.Names, seat.Name)
				}
			}
			result = append(result, entry)
		}
		r.mu.Unlock()
	}
	sort.Slice(result, func(i, j int) bool { return result[i].LastActivity.After(result[j].LastActivity) })
	return result
}
