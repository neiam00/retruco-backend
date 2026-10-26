package api_test

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/neiam00/retruco-backend/internal/api"
	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDeploymentNoticeRequiresAuthenticatedLocalOperator(t *testing.T) {
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	host := game.New(nil, log)
	server, err := api.New(t.Context(), api.Config{ProxySecret: "operator-secret"}, nil, host, log)
	require.NoError(t, err)
	handler := server.Handler()
	request := func(method, address, secret, body string) *httptest.ResponseRecorder {
		r := httptest.NewRequest(method, "/internal/deployment", strings.NewReader(body))
		r.RemoteAddr = address
		r.Header.Set("X-Retruco-Proxy-Secret", secret)
		r.Header.Set("X-Forwarded-For", "127.0.0.1")
		r.Header.Set("X-Retruco-Client-IP", "127.0.0.1")
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, r)
		return w
	}
	for _, tc := range []struct {
		name, method, address, secret, body string
		status                              int
	}{
		{"missing secret", "POST", "127.0.0.1:9000", "", `{"updating":true}`, 403},
		{"wrong secret", "POST", "127.0.0.1:9000", "wrong", `{"updating":true}`, 403},
		{"forwarded remote address", "POST", "192.0.2.1:9000", "operator-secret", `{"updating":true}`, 403},
		{"read cannot mutate", "GET", "127.0.0.1:9000", "operator-secret", `{"updating":true}`, 405},
		{"missing state", "POST", "127.0.0.1:9000", "operator-secret", `{}`, 400},
		{"invalid state", "POST", "127.0.0.1:9000", "operator-secret", `{"updating":"true"}`, 400},
		{"unknown field", "POST", "127.0.0.1:9000", "operator-secret", `{"updating":true,"message":"anything"}`, 400},
		{"trailing input", "POST", "127.0.0.1:9000", "operator-secret", `{"updating":true}{}`, 400},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.status, request(tc.method, tc.address, tc.secret, tc.body).Code)
			assert.False(t, host.Updating())
		})
	}
	assert.Equal(t, http.StatusOK, request("POST", "127.0.0.1:9000", "operator-secret", `{"updating":true}`).Code)
	assert.True(t, host.Updating())
	assert.Equal(t, http.StatusOK, request("POST", "[::1]:9000", "operator-secret", `{"updating":false}`).Code)
	assert.False(t, host.Updating())
}

func TestDeploymentNoticeIsDisabledWithoutAnOperatorSecret(t *testing.T) {
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	host := game.New(nil, log)
	server, err := api.New(t.Context(), api.Config{}, nil, host, log)
	require.NoError(t, err)
	r := httptest.NewRequest("POST", "/internal/deployment", strings.NewReader(`{"updating":true}`))
	r.RemoteAddr = "127.0.0.1:9000"
	w := httptest.NewRecorder()
	server.Handler().ServeHTTP(w, r)
	assert.Equal(t, http.StatusForbidden, w.Code)
	assert.False(t, host.Updating())
}
