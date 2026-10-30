package api

import (
	"net/http"
	"strconv"
	"time"
)

// Expensive password work is bounded globally as well as by the existing
// per-client rate limit. A full slot rejects promptly instead of accumulating
// unbounded hashing goroutines or request queues on the single-CPU host.
func (s *Server) passwordAdmission(w http.ResponseWriter) (func(), bool) {
	select {
	case s.passwordSlots <- struct{}{}:
		s.passwordMu.Lock()
		remaining := time.Until(s.passwordNext)
		s.passwordMu.Unlock()
		if remaining > 0 {
			<-s.passwordSlots
			seconds := max(1, int((remaining+time.Second-1)/time.Second))
			s.passwordOverloaded(w, seconds)
			return nil, false
		}
		return func() { <-s.passwordSlots }, true
	default:
		s.passwordOverloaded(w, 2)
		return nil, false
	}
}

func (s *Server) passwordOverloaded(w http.ResponseWriter, seconds int) {
	w.Header().Set("Retry-After", strconv.Itoa(seconds))
	fail(w, http.StatusServiceUnavailable, "Hay varios inicios de sesión en curso. Intenta de nuevo en unos segundos.")
}

// Only hash work spends the duty budget. Repository and verification latency
// retain the admission lease but do not extend this cooldown.
func (s *Server) passwordWorkCompleted(started time.Time) {
	ended := time.Now()
	cooldown := ended.Sub(started) * time.Duration(100-s.config.PasswordWorkDutyPercent) / time.Duration(s.config.PasswordWorkDutyPercent)
	s.passwordMu.Lock()
	s.passwordNext = ended.Add(cooldown)
	s.passwordMu.Unlock()
}
