package api

import (
	"io"
	"log/slog"
	"net/http"
	"testing"
	"time"

	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStatisticsAndResumeAuthorizeTheCurrentIdentity(t *testing.T) {
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	repository := &sessionRepository{sessions: map[string]store.Session{
		digest("guest"):   {User: store.User{ID: "guest"}, ExpiresAt: time.Now().Add(time.Hour)},
		digest("account"): {User: store.User{ID: "account", Registered: true}, ExpiresAt: time.Now().Add(time.Hour)},
	}}
	server, err := New(t.Context(), Config{Origins: []string{"https://retruco.test"}}, repository, game.New(nil, log), log)
	require.NoError(t, err)
	handler := server.Handler()
	for _, scenario := range []struct {
		name, path, token string
		status            int
	}{
		{name: "anonymous statistics", path: "/api/statistics", status: http.StatusUnauthorized},
		{name: "guest statistics", path: "/api/statistics", token: "guest", status: http.StatusForbidden},
		{name: "account statistics", path: "/api/statistics", token: "account", status: http.StatusOK},
		{name: "anonymous resume", path: "/api/matches/resume", status: http.StatusUnauthorized},
		{name: "guest resume", path: "/api/matches/resume", token: "guest", status: http.StatusOK},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			response := sessionRequest(handler, http.MethodGet, scenario.path, scenario.token)
			assert.Equal(t, scenario.status, response.Code)
			if scenario.name == "guest resume" {
				assert.JSONEq(t, `[]`, response.Body.String())
			}
		})
	}
}
