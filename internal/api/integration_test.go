package api_test

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/cookiejar"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/neiam00/retruco-backend/internal/api"
	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Run against a disposable database with migrations already applied:
// DATABASE_TEST_URL=... go test ./internal/api -run TestAccountAndMatchPersistence
func TestAccountAndMatchPersistence(t *testing.T) {
	databaseURL := os.Getenv("DATABASE_TEST_URL")
	if databaseURL == "" {
		t.Skip("DATABASE_TEST_URL is required for the PostgreSQL integration test")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	pool, err := pgxpool.New(ctx, databaseURL)
	require.NoError(t, err)
	defer pool.Close()
	require.NoError(t, pool.Ping(ctx))

	repository := store.New(pool)
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	host := game.New(repository, log)
	const origin = "http://local.retruco.test"
	server, err := api.New(ctx, api.Config{Origins: []string{origin}}, repository, host, log)
	require.NoError(t, err)
	httpServer := httptest.NewServer(server.Handler())
	defer httpServer.Close()

	newClient := func() *http.Client {
		jar, jarErr := cookiejar.New(nil)
		require.NoError(t, jarErr)
		return &http.Client{Jar: jar, Timeout: 5 * time.Second}
	}
	request := func(client *http.Client, method, path string, body any, target any) int {
		payload, marshalErr := json.Marshal(body)
		require.NoError(t, marshalErr)
		// #nosec G704 -- The destination is this test's httptest server; all paths are test literals.
		req, requestErr := http.NewRequestWithContext(ctx, method, httpServer.URL+path, bytes.NewReader(payload))
		require.NoError(t, requestErr)
		req.Header.Set("Origin", origin)
		req.Header.Set("Content-Type", "application/json")
		// #nosec G704 -- Requests only target the local httptest server created above.
		response, doErr := client.Do(req)
		require.NoError(t, doErr)
		defer func() { _ = response.Body.Close() }()
		if target != nil {
			require.NoError(t, json.NewDecoder(response.Body).Decode(target))
		}
		return response.StatusCode
	}

	first := newClient()
	username := "test_" + strings.ToLower(rand.Text()[:12])
	credentials := map[string]string{"username": username, "password": "local-test-password-123"}
	var account store.User
	require.Equal(t, http.StatusOK, request(first, http.MethodPost, "/api/register", credentials, &account))
	assert.True(t, account.Registered)
	var storedHash string
	require.NoError(t, pool.QueryRow(ctx, `SELECT password_hash FROM users WHERE id=$1`, account.ID).Scan(&storedHash))
	assert.NotEqual(t, credentials["password"], storedHash)
	assert.True(t, strings.HasPrefix(storedHash, "$2a$") || strings.HasPrefix(storedHash, "$2b$"))

	var created struct {
		Code string `json:"code"`
	}
	require.Equal(t, http.StatusCreated, request(first, http.MethodPost, "/api/matches", map[string]any{
		"private": true,
		"config":  rules.Config{Players: 2, Target: 12, FlorRequired: true},
	}, &created))

	second := newClient()
	require.Equal(t, http.StatusOK, request(second, http.MethodPost, "/api/guest", map[string]string{"name": "Invitado de prueba"}, nil))
	require.Equal(t, http.StatusOK, request(second, http.MethodPost, "/api/matches/"+created.Code+"/join", struct{}{}, nil))

	// Voice authorizes speaking from room membership, not client-provided SDP.
	spectator := newClient()
	require.Equal(t, http.StatusOK, request(spectator, http.MethodPost, "/api/guest", map[string]string{"name": "Oyente"}, nil))
	joinVoice := func(client *http.Client, code string, canSpeak bool) *websocket.Conn {
		conn, _, dialErr := websocket.Dial(ctx, "ws"+strings.TrimPrefix(httpServer.URL, "http")+"/api/matches/"+code+"/voice", &websocket.DialOptions{
			HTTPClient: client, HTTPHeader: http.Header{"Origin": []string{origin}},
		})
		require.NoError(t, dialErr)
		t.Cleanup(func() { _ = conn.CloseNow() })
		var welcome struct {
			Type string `json:"type"`
			Self struct {
				ID       string `json:"id"`
				CanSpeak bool   `json:"canSpeak"`
			} `json:"self"`
		}
		require.NoError(t, wsjson.Read(ctx, conn, &welcome))
		require.Equal(t, "welcome", welcome.Type)
		require.NotEmpty(t, welcome.Self.ID)
		assert.Equal(t, canSpeak, welcome.Self.CanSpeak)
		return conn
	}
	playerVoice := joinVoice(first, created.Code, true)
	listenerVoice := joinVoice(spectator, created.Code, false)
	var openRoom struct {
		Code string `json:"code"`
	}
	require.Equal(t, http.StatusCreated, request(second, http.MethodPost, "/api/matches", map[string]any{
		"private": true,
		"config":  rules.Config{Players: 2, Target: 12},
		"options": game.RoomOptions{SpectatorsCanSpeak: true},
	}, &openRoom))
	speakerVoice := joinVoice(spectator, openRoom.Code, true)
	require.NoError(t, playerVoice.CloseNow())
	require.NoError(t, listenerVoice.CloseNow())
	require.NoError(t, speakerVoice.CloseNow())

	// CPU seats are persisted without inserting fake users into membership rows.
	var cpuRoom struct {
		Code string `json:"code"`
	}
	require.Equal(t, http.StatusCreated, request(second, http.MethodPost, "/api/matches", map[string]any{
		"private": true,
		"config":  rules.Config{Players: 2, Target: 12},
		"options": game.RoomOptions{CPUSeats: []int{1}},
	}, &cpuRoom))
	cpuConnection, _, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(httpServer.URL, "http")+"/api/matches/"+cpuRoom.Code+"/ws", &websocket.DialOptions{
		HTTPClient: second, HTTPHeader: http.Header{"Origin": []string{origin}},
	})
	require.NoError(t, err)
	defer func() { _ = cpuConnection.CloseNow() }()
	type cpuSnapshot struct {
		Type string        `json:"type"`
		Data game.Snapshot `json:"data"`
		CPU  game.CPUWork  `json:"cpu"`
	}
	var botMessage cpuSnapshot
	require.NoError(t, wsjson.Read(ctx, cpuConnection, &botMessage))
	require.True(t, botMessage.CPU.Controlled)
	require.Len(t, botMessage.Data.Seats, 2)
	assert.True(t, botMessage.Data.Seats[1].CPU)
	for move := 0; move < 5 && botMessage.CPU.View == nil; move++ {
		require.NotEmpty(t, botMessage.Data.State.Legal)
		revision := botMessage.Data.State.Revision
		require.NoError(t, wsjson.Write(ctx, cpuConnection, map[string]any{"input": game.Input{
			ID: fmt.Sprintf("human-before-cpu-%d", move), Revision: revision,
			Command: botMessage.Data.State.Legal[0],
		}}))
		for {
			require.NoError(t, wsjson.Read(ctx, cpuConnection, &botMessage))
			require.Equal(t, "snapshot", botMessage.Type)
			if botMessage.Data.State.Revision > revision {
				break
			}
		}
	}
	require.NotNil(t, botMessage.CPU.View)
	revision := botMessage.CPU.View.Revision
	require.NoError(t, wsjson.Write(ctx, cpuConnection, map[string]any{"input": game.Input{
		ID: "browser-cpu-move", Revision: revision, CPUSeat: &botMessage.CPU.Seat,
		Command: botMessage.CPU.View.Legal[0],
	}}))
	for {
		require.NoError(t, wsjson.Read(ctx, cpuConnection, &botMessage))
		require.Equal(t, "snapshot", botMessage.Type)
		if botMessage.Data.State.Revision > revision {
			break
		}
	}
	assert.Equal(t, revision+1, botMessage.Data.State.Revision)
	watcherConnection, _, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(httpServer.URL, "http")+"/api/matches/"+cpuRoom.Code+"/ws", &websocket.DialOptions{
		HTTPClient: spectator, HTTPHeader: http.Header{"Origin": []string{origin}},
	})
	require.NoError(t, err)
	defer func() { _ = watcherConnection.CloseNow() }()
	var watched cpuSnapshot
	require.NoError(t, wsjson.Read(ctx, watcherConnection, &watched))
	assert.False(t, watched.CPU.Controlled)
	assert.Nil(t, watched.CPU.View)

	connection, _, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(httpServer.URL, "http")+"/api/matches/"+created.Code+"/ws", &websocket.DialOptions{
		HTTPClient: first,
		HTTPHeader: http.Header{"Origin": []string{origin}},
	})
	require.NoError(t, err)
	defer func() { _ = connection.CloseNow() }()
	var initial struct {
		Type string        `json:"type"`
		Data game.Snapshot `json:"data"`
	}
	require.NoError(t, wsjson.Read(ctx, connection, &initial))
	require.NotNil(t, initial.Data.State)
	require.NoError(t, wsjson.Write(ctx, connection, map[string]any{"input": game.Input{
		ID: "leave-test", Revision: initial.Data.State.Revision, Command: rules.Command{Type: "leave"},
	}}))
	var finished struct {
		Data game.Snapshot `json:"data"`
	}
	require.NoError(t, wsjson.Read(ctx, connection, &finished))
	require.NotNil(t, finished.Data.State)
	assert.Equal(t, rules.MatchOver, finished.Data.State.Phase)
	assert.Equal(t, 1, finished.Data.State.Winner)

	// A different cookie jar simulates signing in on another device.
	third := newClient()
	var loggedIn store.User
	require.Equal(t, http.StatusOK, request(third, http.MethodPost, "/api/login", credentials, &loggedIn))
	assert.Equal(t, account.ID, loggedIn.ID)
	var history []store.HistoryEntry
	require.Equal(t, http.StatusOK, request(third, http.MethodGet, "/api/history", nil, &history))
	require.Len(t, history, 1)
	assert.Equal(t, created.Code, history[0].Code)
	assert.True(t, history[0].Finished)
	assert.Equal(t, 1, history[0].Winner)

	require.Equal(t, http.StatusOK, request(third, http.MethodPost, "/api/logout", struct{}{}, nil))
	assert.Equal(t, http.StatusUnauthorized, request(third, http.MethodGet, "/api/history", nil, nil))

	attack, err := http.NewRequestWithContext(ctx, http.MethodPost, httpServer.URL+"/api/guest", strings.NewReader(`{"name":"untrusted"}`))
	require.NoError(t, err)
	attack.Header.Set("Origin", "https://untrusted.example")
	attack.Header.Set("Content-Type", "application/json")
	response, err := first.Do(attack)
	require.NoError(t, err)
	defer func() { _ = response.Body.Close() }()
	assert.Equal(t, http.StatusForbidden, response.StatusCode)
}
