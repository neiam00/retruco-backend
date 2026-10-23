package api_test

import (
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
	"testing/synctest"
	"time"

	"github.com/neiam00/retruco-backend/internal/api"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRequestPathChurnCannotExhaustOtherClients(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		server, err := api.New(t.Context(), api.Config{}, nil, nil,
			slog.New(slog.NewTextHandler(io.Discard, nil)))
		require.NoError(t, err)
		handler := server.Handler()
		request := func(method, path, address string) int {
			r := httptest.NewRequest(method, path, nil)
			r.RemoteAddr = address
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, r)
			return w.Code
		}
		limited := 0
		for index := range 10001 {
			path := fmt.Sprintf("/api/missing-%d", index)
			if index%2 == 0 {
				path = fmt.Sprintf("/api/matches/%d/missing", index)
			}
			if request(http.MethodGet, path, "192.0.2.1:1234") == http.StatusTooManyRequests {
				limited++
			}
		}
		assert.Equal(t, 10001-120, limited, "varying paths must share the same client budget")
		assert.Equal(t, http.StatusTooManyRequests, request(http.MethodHead, "/api/health", "192.0.2.1:1234"))
		assert.Equal(t, http.StatusOK, request(http.MethodGet, "/api/health", "192.0.2.2:1234"))
		time.Sleep(time.Minute)
		assert.Equal(t, http.StatusOK, request(http.MethodGet, "/api/health", "192.0.2.1:1234"))
	})
}

func TestAuthenticationRetainsItsTighterRateBudgets(t *testing.T) {
	server, err := api.New(t.Context(), api.Config{}, nil, nil,
		slog.New(slog.NewTextHandler(io.Discard, nil)))
	require.NoError(t, err)
	handler := server.Handler()
	for _, path := range []string{"/api/login", "/api/register", "/api/guest"} {
		t.Run(path, func(t *testing.T) {
			for index := range 11 {
				// GET reaches admission control but cannot perform an account mutation.
				r := httptest.NewRequest(http.MethodGet, path, nil)
				w := httptest.NewRecorder()
				handler.ServeHTTP(w, r)
				want := http.StatusMethodNotAllowed
				if index == 10 {
					want = http.StatusTooManyRequests
				}
				require.Equal(t, want, w.Code)
			}
		})
	}
}

func TestRateLimitCapacityRecoversWithoutResettingNewerClients(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		server, err := api.New(t.Context(), api.Config{}, nil, nil,
			slog.New(slog.NewTextHandler(io.Discard, nil)))
		require.NoError(t, err)
		handler := server.Handler()
		request := func(index int) int {
			r := httptest.NewRequest(http.MethodGet, "/api/health", nil)
			r.RemoteAddr = fmt.Sprintf("192.0.%d.%d:1234", index/256, index%256)
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, r)
			return w.Code
		}
		for index := range 9999 {
			require.Equal(t, http.StatusOK, request(index))
		}
		time.Sleep(30 * time.Second)
		for range 120 {
			require.Equal(t, http.StatusOK, request(9999))
		}
		assert.Equal(t, http.StatusTooManyRequests, request(10000), "new clients cannot exceed capacity")
		assert.Equal(t, http.StatusOK, request(0), "existing clients retain their budget at capacity")
		time.Sleep(30 * time.Second)
		for range 120 {
			require.Equal(t, http.StatusOK, request(10000), "expired clients free capacity immediately")
		}
		assert.Equal(t, http.StatusTooManyRequests, request(9999), "the newer client's exhausted budget must remain")
		time.Sleep(30 * time.Second)
		assert.Equal(t, http.StatusOK, request(9999))
		assert.Equal(t, http.StatusTooManyRequests, request(10000), "renewing an older client must not expire a newer one")
		time.Sleep(30 * time.Second)
		assert.Equal(t, http.StatusOK, request(10000))
	})
}

func TestAuthenticatedProxyKeepsVisitorRateLimitsSeparate(t *testing.T) {
	server, err := api.New(t.Context(), api.Config{ProxySecret: "test-edge-secret"}, nil, nil,
		slog.New(slog.NewTextHandler(io.Discard, nil)))
	require.NoError(t, err)
	handler := server.Handler()
	request := func(secret, clientIP string) int {
		r := httptest.NewRequest(http.MethodGet, "/api/health", nil)
		r.Header.Set("X-Retruco-Proxy-Secret", secret)
		r.Header.Set("X-Retruco-Client-IP", clientIP)
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, r)
		return w.Code
	}
	assert.Equal(t, http.StatusForbidden, request("", "192.0.2.1"))
	assert.Equal(t, http.StatusForbidden, request("wrong-secret", "192.0.2.1"))
	for range 120 {
		require.Equal(t, http.StatusOK, request("test-edge-secret", "192.0.2.1"))
	}
	assert.Equal(t, http.StatusTooManyRequests, request("test-edge-secret", "192.0.2.1"))
	assert.Equal(t, http.StatusOK, request("test-edge-secret", "192.0.2.2"))
}

func TestDirectRequestsCannotChooseTheirRateLimitAddress(t *testing.T) {
	server, err := api.New(t.Context(), api.Config{}, nil, nil,
		slog.New(slog.NewTextHandler(io.Discard, nil)))
	require.NoError(t, err)
	handler := server.Handler()
	for index := range 121 {
		r := httptest.NewRequest(http.MethodGet, "/api/health", nil)
		r.Header.Set("X-Retruco-Client-IP", "192.0.2.2")
		if index == 120 {
			r.Header.Set("X-Retruco-Client-IP", "192.0.2.3")
		}
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, r)
		expected := http.StatusOK
		if index == 120 {
			expected = http.StatusTooManyRequests
		}
		require.Equal(t, expected, w.Code)
	}
}
