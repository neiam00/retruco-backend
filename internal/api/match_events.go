package api

import (
	"errors"
	"io"
	"net/http"
	"time"
)

// matchEvents streams only the same public room listings exposed by GET /matches.
func (s *Server) matchEvents(w http.ResponseWriter, r *http.Request) {
	updates, unsubscribe := s.host.SubscribeListings()
	defer unsubscribe()
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("X-Accel-Buffering", "no")
	controller := http.NewResponseController(w)
	write := func(parts ...string) error {
		if err := controller.SetWriteDeadline(time.Now().Add(5 * time.Second)); err != nil && !errors.Is(err, http.ErrNotSupported) {
			return err
		}
		// Limit a blocked write, not the idle time between events or heartbeats.
		defer func() { _ = controller.SetWriteDeadline(time.Time{}) }()
		for _, part := range parts {
			if _, err := io.WriteString(w, part); err != nil {
				return err
			}
		}
		return controller.Flush()
	}
	sendListings := func() error {
		payload, err := s.host.ListingJSON()
		if err != nil {
			return err
		}
		return write("data: ", payload, "\n\n")
	}
	if sendListings() != nil {
		return
	}
	heartbeat := time.NewTicker(15 * time.Second)
	defer heartbeat.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-s.ctx.Done():
			return
		case <-updates:
			if sendListings() != nil {
				return
			}
		case <-heartbeat.C:
			if write(": keepalive\n\n") != nil {
				return
			}
		}
	}
}
