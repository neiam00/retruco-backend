package game

import "fmt"

// VoiceAccess checks room existence and speaking rights without building a
// game view, copying chat/history, or calculating legal actions. Authentication
// remains the API's responsibility and must still run on each authorization.
func (h *Host) VoiceAccess(code, userID string) (bool, error) {
	r := h.lockRoom(code)
	if r == nil {
		return false, fmt.Errorf("no encontramos esa mesa")
	}
	defer r.mu.Unlock()
	return seatOf(r.match, userID) >= 0 || r.match.Options.SpectatorsCanSpeak, nil
}

// SubscribeVoiceAccess atomically checks permission and subscribes to changes.
// Only committed membership/permission changes and room removal wake listeners;
// ordinary moves and CPU-controller changes cannot change voice authorization.
func (h *Host) SubscribeVoiceAccess(code, userID string) (bool, <-chan struct{}, func(), error) {
	r := h.lockRoom(code)
	if r == nil {
		return false, nil, nil, fmt.Errorf("no encontramos esa mesa")
	}
	changes := make(chan struct{}, 1)
	r.voiceListeners[changes] = true
	canSpeak := seatOf(r.match, userID) >= 0 || r.match.Options.SpectatorsCanSpeak
	r.mu.Unlock()
	return canSpeak, changes, func() {
		r.mu.Lock()
		defer r.mu.Unlock()
		delete(r.voiceListeners, changes)
	}, nil
}

func voiceAccessChanged(previous, next *Match) bool {
	if previous.Options.SpectatorsCanSpeak != next.Options.SpectatorsCanSpeak || len(previous.Seats) != len(next.Seats) {
		return true
	}
	for seat := range previous.Seats {
		if previous.Seats[seat].ID != next.Seats[seat].ID {
			return true
		}
	}
	return false
}

// Caller holds r.mu. Permission notifications coalesce just like game updates.
func (r *room) notifyVoiceAccess() {
	for changes := range r.voiceListeners {
		select {
		case changes <- struct{}{}:
		default:
		}
	}
}
