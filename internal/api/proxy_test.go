package api_test

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/neiam00/retruco-backend/internal/api"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
