package api

import (
	"net/http"
	"sync"
)

// A stream retains sockets, goroutines and subscriptions beyond the HTTP rate
// window. Share one budget across lobby transports, game sockets and voice.
// Leave headroom below the production file-descriptor limit and for browser tabs.
const maxLiveStreams = 1024
const maxLiveStreamsPerClient = 32

type streamAdmission struct {
	mu      sync.Mutex
	total   int
	clients map[string]int
}

func (a *streamAdmission) acquire(client string) (func(), bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.total >= maxLiveStreams || a.clients[client] >= maxLiveStreamsPerClient {
		return nil, false
	}
	if a.clients == nil {
		a.clients = make(map[string]int)
	}
	a.total++
	a.clients[client]++
	return sync.OnceFunc(func() {
		a.mu.Lock()
		defer a.mu.Unlock()
		a.total--
		a.clients[client]--
		if a.clients[client] == 0 {
			delete(a.clients, client)
		}
	}), true
}

// Route registration keeps aliases and both protocol versions behind the same
// guard. Preserve the original writer's flushing and WebSocket upgrade support.
func (s *Server) liveStream(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		release, ok := s.streams.acquire(s.clientIP(r))
		if !ok {
			w.Header().Set("Retry-After", "5")
			fail(w, http.StatusTooManyRequests, "Hay demasiadas conexiones abiertas. Cierra otra pestaña o inténtalo de nuevo.")
			return
		}
		defer release()
		next(w, r)
	}
}
