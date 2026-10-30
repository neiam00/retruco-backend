package api

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type streamReadyWriter struct {
	*httptest.ResponseRecorder
	ready chan struct{}
	once  sync.Once
}

func (w *streamReadyWriter) Flush() {
	w.ResponseRecorder.Flush()
	w.once.Do(func() { close(w.ready) })
}

// Exercise real subscriptions through the public router. A canceled probe makes
// an erroneously admitted SSE request return instead of hanging the regression.
func TestLiveStreamAdmissionSharesCapacityAndReleasesOnDisconnect(t *testing.T) {
	for _, proxied := range []bool{false, true} {
		t.Run(fmt.Sprint("proxy=", proxied), func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				log := slog.New(slog.NewTextHandler(io.Discard, nil))
				config := Config{Origins: []string{"https://retruco.test"}}
				if proxied {
					config.ProxySecret = "test-edge-secret"
				}
				server, err := New(t.Context(), config, nil, game.New(nil, log), log)
				require.NoError(t, err)
				handler := server.Handler()
				request := func(path, ip string) *http.Request {
					r := httptest.NewRequest(http.MethodGet, path, nil)
					r.RemoteAddr = ip + ":1234"
					if proxied {
						r.RemoteAddr = "127.0.0.1:1234"
						r.Header.Set("X-Retruco-Proxy-Secret", config.ProxySecret)
					}
					r.Header.Set("X-Retruco-Client-IP", ip)
					r.Header.Set("Origin", "https://retruco.test")
					return r
				}
				var stops []func()
				defer func() {
					for _, stop := range stops {
						stop()
					}
					synctest.Wait()
				}()
				open := func(path, ip string) func() {
					t.Helper()
					ctx, cancel := context.WithCancel(t.Context())
					w := &streamReadyWriter{ResponseRecorder: httptest.NewRecorder(), ready: make(chan struct{})}
					done := make(chan struct{})
					go func() {
						handler.ServeHTTP(w, request(path, ip).WithContext(ctx))
						close(done)
					}()
					stop := sync.OnceFunc(func() { cancel(); <-done })
					stops = append(stops, stop)
					select {
					case <-w.ready:
						require.Equal(t, http.StatusOK, w.Code)
					case <-done:
						t.Fatalf("stream rejected: %d %s", w.Code, w.Body.String())
					}
					return stop
				}
				probe := func(path, ip string) *httptest.ResponseRecorder {
					ctx, cancel := context.WithCancel(t.Context())
					cancel()
					r := request(path, ip).WithContext(ctx)
					if !proxied {
						// Direct callers cannot rotate a forged visitor header.
						r.Header.Set("X-Retruco-Client-IP", "203.0.113.99")
					}
					w := httptest.NewRecorder()
					handler.ServeHTTP(w, r)
					return w
				}
				var first func()
				// Failed upgrades must return their lease before another attempt.
				for range 40 {
					w := probe("/api/matches/ABCD-1234/ws?transport=2", "192.0.2.1")
					require.Equal(t, http.StatusUpgradeRequired, w.Code)
				}
				// The documented visitor limit leaves room for ordinary tabs/audio.
				for i := range 32 {
					path := "/api/matches/events"
					if i%2 == 0 {
						path += "?transport=2"
					}
					stop := open(path, "192.0.2.1")
					if i == 0 {
						first = stop
					}
				}
				for _, path := range []string{
					"/api/matches/events",
					"/api/matches/events?transport=2",
					"/api/matches/ABCD-1234/ws",
					"/api/matches/ABCD-1234/ws?transport=2",
					"/api/matches/ABCD-1234/voice",
				} {
					w := probe(path, "192.0.2.1")
					assert.Equal(t, http.StatusTooManyRequests, w.Code, path)
					assert.NotEmpty(t, w.Header().Get("Retry-After"), path)
				}
				assert.Equal(t, http.StatusOK, probe("/api/health", "192.0.2.1").Code)
				open("/api/matches/events", "192.0.2.2")
				time.Sleep(time.Minute + time.Second)
				assert.Equal(t, http.StatusTooManyRequests, probe("/api/matches/events?transport=2", "192.0.2.1").Code,
					"active leases must outlive the request-rate window")
				first()
				open("/api/matches/events?transport=2", "192.0.2.1")
			})
		})
	}
}

func TestLiveStreamAdmissionBoundsConcurrentVisitorsGlobally(t *testing.T) {
	server := &Server{}
	finished := make(chan struct{})
	var workers sync.WaitGroup
	defer workers.Wait()
	defer close(finished)
	results := make(chan bool, maxLiveStreams+64)
	handler := server.liveStream(func(http.ResponseWriter, *http.Request) {
		results <- true
		<-finished
	})
	for i := range maxLiveStreams + 64 {
		workers.Go(func() {
			r := httptest.NewRequest(http.MethodGet, "/api/matches/events", nil)
			r.RemoteAddr = fmt.Sprintf("192.0.%d.%d:1234", i/256, i%256)
			w := httptest.NewRecorder()
			handler(w, r)
			if w.Code == http.StatusTooManyRequests {
				results <- false
			}
		})
	}
	admitted := 0
	for range maxLiveStreams + 64 {
		if <-results {
			admitted++
		}
	}
	assert.Equal(t, maxLiveStreams, admitted, "parallel distinct visitors cannot exceed the process budget")
}
