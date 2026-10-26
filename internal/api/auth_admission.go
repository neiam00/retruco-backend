package api

import "net/http"

// Expensive password work is bounded globally as well as by the existing
// per-client rate limit. A full slot rejects promptly instead of accumulating
// unbounded hashing goroutines or request queues on the single-CPU host.
func (s *Server) passwordAdmission(w http.ResponseWriter) (func(), bool) {
	select {
	case s.passwordSlots <- struct{}{}:
		return func() { <-s.passwordSlots }, true
	default:
		w.Header().Set("Retry-After", "2")
		fail(w, http.StatusServiceUnavailable, "Hay varios inicios de sesión en curso. Intenta de nuevo en unos segundos.")
		return nil, false
	}
}
