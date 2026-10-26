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
