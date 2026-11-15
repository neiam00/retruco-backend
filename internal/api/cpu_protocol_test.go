package api_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
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

func TestCPUProtocolNegotiationIsIndependentOfStreamVersion(t *testing.T) {
	for _, query := range []string{"?cpuProtocol=2", "?transport=2&cpuProtocol=2", "?transport=2&cpuProtocol=2&cpuMemory=1", "?transport=2"} {
		t.Run(query, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()
			user := store.User{ID: "owner", Name: "Ana"}
			hash := sha256.Sum256([]byte(user.ID))
			repo := &streamRepository{sessions: map[string]store.Session{hex.EncodeToString(hash[:]): {User: user, ExpiresAt: time.Now().Add(time.Hour)}}, matches: map[string][]byte{}}
			log := slog.New(slog.NewTextHandler(io.Discard, nil))
			host := game.New(repo, log)
			config := rules.Config{Players: 4, Target: 12}
			code, err := host.Create(ctx, user, config, false, game.RoomOptions{CPUSeats: []int{1, 2, 3}})
			require.NoError(t, err)
			var match game.Match
			require.NoError(t, json.Unmarshal(repo.matches[code], &match))
			state, err := rules.New(config, 0, rules.Deck())
			require.NoError(t, err)
			state, _, err = rules.Apply(state, state.Turn, rules.Command{Type: "cut", Cut: 1})
			require.NoError(t, err)
			match.State = &state
			repo.matches[code], err = json.Marshal(match)
			require.NoError(t, err)
			host = game.New(repo, log)
			require.NoError(t, host.Restore(ctx))
			server, err := api.New(ctx, api.Config{Origins: []string{"https://retruco.test"}}, repo, host, log)
			require.NoError(t, err)
			httpServer := httptest.NewServer(server.Handler())
			defer httpServer.Close()
			url := "ws" + strings.TrimPrefix(httpServer.URL, "http") + "/api/matches/" + code + "/ws" + query
			conn, _, err := websocket.Dial(ctx, url, &websocket.DialOptions{HTTPClient: httpServer.Client(), HTTPHeader: http.Header{"Origin": []string{"https://retruco.test"}, "Cookie": []string{"retruco_session=owner"}}})
			require.NoError(t, err)
			defer conn.CloseNow()
			var message struct {
				Type  string       `json:"type"`
				CPU   game.CPUWork `json:"cpu"`
				Error string       `json:"error"`
			}
			require.NoError(t, wsjson.Read(ctx, conn, &message))
			require.Equal(t, "snapshot", message.Type)
			if !strings.Contains(query, "cpuProtocol=2") {
				assert.Equal(t, 1, message.CPU.Version)
				assert.Nil(t, message.CPU.Features)
				return
			}
			require.Equal(t, 2, message.CPU.Version)
			require.Equal(t, "signal", message.CPU.Kind)
			require.Equal(t, 1, message.CPU.Seat)
			if strings.Contains(query, "cpuMemory=1") {
				require.NotNil(t, message.CPU.Memory)
				assert.Equal(t, 1, message.CPU.Memory.Version)
				assert.Len(t, message.CPU.MemoryKey, 64)
			} else {
				assert.Nil(t, message.CPU.Memory)
			}
			work := message.CPU
			require.NoError(t, wsjson.Write(ctx, conn, map[string]any{"input": game.Input{ID: "none", CPUSeat: &work.Seat, CPUKind: work.Kind, CPUToken: work.Token, Signal: "none"}}))
			for attempt := 0; attempt < 10; attempt++ {
				require.NoError(t, wsjson.Read(ctx, conn, &message))
				require.Empty(t, message.Error)
				if message.CPU.Seat == 2 {
					assert.NotEqual(t, work.Token, message.CPU.Token)
					return
				}
			}
			t.Fatal("next CPU opportunity was not published")
		})
	}
}
