package game

import (
	"time"

	"github.com/neiam00/retruco-backend/rules"
)

// Updating is process-local: a replacement API resumes persisted matches.
func (h *Host) Updating() bool {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.updating
}

// SetUpdating pauses both turn timers and vacant-room cleanup before the API
// stops. Snapshot notifications also reach players who join during the notice.
func (h *Host) SetUpdating(updating bool) {
	h.deploymentMu.Lock()
	defer h.deploymentMu.Unlock()
	h.mu.Lock()
	if h.updating == updating {
		h.mu.Unlock()
		return
	}
	if updating {
		h.updating = true
	}
	rooms := make([]*room, 0, len(h.rooms))
	for _, r := range h.rooms {
		rooms = append(rooms, r)
	}
	h.mu.Unlock()
	for _, r := range rooms {
		r.mu.Lock()
		if r.match != nil {
			if !updating {
				r.deadline = time.Time{}
				if r.recovering {
					r.deadline = time.Now().Add(restoreReconnectGrace)
				} else {
					renewDeadline(r.match)
					r.updateRoomDeadline(r.match)
				}
			}
			if updating {
				r.notify()
			}
		}
		r.mu.Unlock()
	}
	if !updating {
		h.mu.Lock()
		h.updating = false
		h.mu.Unlock()
		for _, r := range rooms {
			r.mu.Lock()
			r.notify()
			r.mu.Unlock()
		}
	}
}

func renewDeadline(match *Match) {
	if match.State == nil || !activeMatch(match) {
		return
	}
	duration := 30 * time.Second
	if match.State.Phase == rules.HandOver {
		duration = 5 * time.Second
	}
	match.Deadline = time.Now().Add(duration)
}
