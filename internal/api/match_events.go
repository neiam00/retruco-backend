package api

import (
	"errors"
	"io"
	"net/http"
	"time"
)

// matchEvents streams only the same public room listings exposed by GET /matches.
func (s *Server) matchEvents(w http.ResponseWriter, r *http.Request) {
	if r.URL.Query().Get("transport") == "2" {
		s.listingStream(w, r)
		return
	}
	updates, unsubscribe := s.host.SubscribeListings()
	defer unsubscribe()
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("X-Accel-Buffering", "no")
	w.Header().Set("Cache-Control", "no-cache")
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
		if err := write("data: ", payload, "\n\n"); err != nil {
			return err
		}
		return nil
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

// Each connection starts with a complete baseline. Coalesced notifications
// retrieve shared encoded frames, or a fresh baseline if the cursor fell out
// of the bounded publication ring. Last-Event-ID is intentionally not replayed.
func (s *Server) listingStream(w http.ResponseWriter, r *http.Request) {
	current, updates, unsubscribe, err := s.host.SubscribeListingStream()
	if err != nil {
		s.internalError(w, err)
		return
	}
	defer unsubscribe()
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("X-Accel-Buffering", "no")
	w.Header().Set("Cache-Control", "no-cache")
	controller := http.NewResponseController(w)
	write := func(payload string) error {
		if err := controller.SetWriteDeadline(time.Now().Add(5 * time.Second)); err != nil && !errors.Is(err, http.ErrNotSupported) {
			return err
		}
		defer func() { _ = controller.SetWriteDeadline(time.Time{}) }()
		if _, err := io.WriteString(w, payload); err != nil {
			return err
		}
		return controller.Flush()
	}
	if write("data: "+current.JSON+"\n\n") != nil {
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
			frames, err := s.host.ListingStreamFrames(current.Epoch, current.Revision)
			if err != nil {
				return
			}
			for _, frame := range frames {
				if write("data: "+frame.JSON+"\n\n") != nil {
					return
				}
				current = frame
			}
		case <-heartbeat.C:
			if write(": keepalive\n\n") != nil {
				return
			}
		}
	}
}
