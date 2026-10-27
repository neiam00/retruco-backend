//go:build integration

package api_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/neiam00/retruco-backend/internal/api"
	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVoiceCloudflareWelcomeRefreshAndFailure(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	user := store.User{ID: "owner", Name: "Owner"}
	hash := sha256.Sum256([]byte("session"))
	repository := &streamRepository{
		sessions: map[string]store.Session{hex.EncodeToString(hash[:]): {User: user, ExpiresAt: time.Now().Add(time.Hour)}},
		matches:  make(map[string][]byte),
	}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	host := game.New(repository, log)
	code, err := host.Create(ctx, user, rules.Config{Players: 2, Target: 12}, false, game.RoomOptions{})
	require.NoError(t, err)
	var calls atomic.Int32
	var unavailable atomic.Bool
	client := &http.Client{Transport: turnstileTransport(func(r *http.Request) (*http.Response, error) {
		n := calls.Add(1)
		status := http.StatusCreated
		body := fmt.Sprintf(`{"iceServers":[{"urls":["turns:turn.cloudflare.com:443?transport=tcp"],"username":"user-%d","credential":"temporary-password"}]}`, n)
		if unavailable.Load() {
			status, body = http.StatusServiceUnavailable, "private-token"
		}
		return &http.Response{StatusCode: status, Body: io.NopCloser(strings.NewReader(body))}, nil
	})}
	server, err := api.New(ctx, api.Config{
		Origins: []string{"https://retruco.test"},
		Voice:   api.VoiceConfig{TURNTokenID: "key-id", TURNAPIToken: "private-token", Client: client},
	}, repository, host, log)
	require.NoError(t, err)
	httpServer := httptest.NewServer(server.Handler())
	defer httpServer.Close()
	for _, tc := range []struct {
		name, cookie, origin, room string
		status                     int
	}{
		{name: "unauthenticated", origin: "https://retruco.test", room: code, status: 401},
		{name: "wrong origin", cookie: "session", origin: "https://example.com", room: code, status: 403},
		{name: "missing room", cookie: "session", origin: "https://retruco.test", room: "MISSING", status: 404},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodGet, "/api/matches/"+tc.room+"/voice", nil)
			request.Header.Set("Origin", tc.origin)
			request.AddCookie(&http.Cookie{Name: "retruco_session", Value: tc.cookie})
			response := httptest.NewRecorder()
			server.Handler().ServeHTTP(response, request)
			assert.Equal(t, tc.status, response.Code, response.Body.String())
			assert.Zero(t, calls.Load(), "rejected requests must not mint relay credentials")
		})
	}
	endpoint := "ws" + strings.TrimPrefix(httpServer.URL, "http") + "/api/matches/" + code + "/voice"
	headers := http.Header{"Origin": {"https://retruco.test"}, "Cookie": {"retruco_session=session"}}
	conn, _, err := websocket.Dial(ctx, endpoint, &websocket.DialOptions{HTTPHeader: headers})
	require.NoError(t, err)
	defer func() { _ = conn.CloseNow() }()
	var message struct {
		Type       string `json:"type"`
		ICEServers []struct {
			Username string `json:"username"`
		} `json:"iceServers"`
	}
	var raw json.RawMessage
	require.NoError(t, wsjson.Read(ctx, conn, &raw))
	assert.NotContains(t, string(raw), "private-token")
	require.NoError(t, json.Unmarshal(raw, &message))
	assert.Equal(t, "welcome", message.Type)
	require.Len(t, message.ICEServers, 1)
	assert.Equal(t, "user-1", message.ICEServers[0].Username)
	require.NoError(t, wsjson.Write(ctx, conn, map[string]string{"type": "refresh"}))
	require.NoError(t, wsjson.Read(ctx, conn, &message))
	assert.Equal(t, "ice", message.Type)
	assert.Equal(t, "user-2", message.ICEServers[0].Username)
	unavailable.Store(true)
	require.NoError(t, wsjson.Write(ctx, conn, map[string]string{"type": "refresh"}))
	require.Error(t, wsjson.Read(ctx, conn, &message), "refresh failure closes voice instead of installing empty credentials")
	assert.Equal(t, int32(3), calls.Load())
	_, response, err := websocket.Dial(ctx, endpoint, &websocket.DialOptions{HTTPHeader: headers})
	require.Error(t, err)
	require.NotNil(t, response)
	assert.Equal(t, http.StatusServiceUnavailable, response.StatusCode)
	assert.Equal(t, int32(4), calls.Load())
}
