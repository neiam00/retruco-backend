package api_test

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Exercise the real 25-second heartbeat. A live application stream must not
// stall waiting for a redundant control pong; idle ping behavior is unchanged.
func TestGameStreamActiveTrafficDoesNotWaitForControlPong(t *testing.T) {
	ctx, server, code, users := newStreamServerWithTimeout(t, 40*time.Second)
	var pings atomic.Int64
	conn, _, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(server.URL, "http")+"/api/matches/"+code+"/ws?transport=2", &websocket.DialOptions{
		HTTPHeader:     http.Header{"Origin": {"https://retruco.test"}, "Cookie": {"retruco_session=" + users[0].ID}},
		OnPingReceived: func(context.Context, []byte) bool { pings.Add(1); return false },
	})
	require.NoError(t, err)
	defer conn.CloseNow()
	var baseline map[string]any
	require.NoError(t, wsjson.Read(ctx, conn, &baseline))
	deadline := time.Now().Add(27 * time.Second)
	for time.Now().Before(deadline) {
		requestCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
		err := wsjson.Write(requestCtx, conn, map[string]bool{"resync": true})
		if err == nil {
			err = wsjson.Read(requestCtx, conn, &baseline)
		}
		cancel()
		require.NoError(t, err, "active updates must continue across the heartbeat boundary")
		if pings.Load() > 0 {
			break
		}
		time.Sleep(250 * time.Millisecond)
	}
	assert.Zero(t, pings.Load(), "fresh application traffic makes a control ping redundant")
}

// Outbound publication traffic can succeed while a browser or proxy no longer
// answers heartbeats. Such a connection must not remain a seated human forever.
func TestGameStreamOutboundUpdatesDoNotKeepUnresponsivePlayerAlive(t *testing.T) {
	ctx, server, code, users := newStreamServerWithTimeout(t, 40*time.Second)
	var pings atomic.Int64
	conn, _, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(server.URL, "http")+"/api/matches/"+code+"/ws?transport=2", &websocket.DialOptions{
		HTTPHeader:     http.Header{"Origin": {"https://retruco.test"}, "Cookie": {"retruco_session=" + users[0].ID}},
		OnPingReceived: func(context.Context, []byte) bool { pings.Add(1); return false },
	})
	require.NoError(t, err)
	defer conn.CloseNow()
	var baseline map[string]any
	require.NoError(t, wsjson.Read(ctx, conn, &baseline))
	closed := make(chan error, 1)
	go func() {
		for {
			var update map[string]any
			if err := wsjson.Read(ctx, conn, &update); err != nil {
				closed <- err
				return
			}
		}
	}()
	other := dialStream(t, ctx, server, code, users[1].ID, true)
	other.read(t, ctx)
	tick := time.NewTicker(500 * time.Millisecond)
	defer tick.Stop()
	deadline := time.NewTimer(34 * time.Second)
	defer deadline.Stop()
	for id := 0; ; id++ {
		select {
		case <-closed:
			require.NoError(t, ctx.Err(), "the server must remove the connection before the test deadline")
			assert.Positive(t, pings.Load(), "outbound traffic cannot suppress checks of an unresponsive peer")
			return
		case <-deadline.C:
			t.Fatal("outbound-only updates kept an unresponsive player connected beyond the heartbeat timeout")
		case <-tick.C:
			require.NoError(t, wsjson.Write(ctx, other.conn, map[string]any{"input": game.Input{ID: fmt.Sprintf("outbound-%d", id), Chat: "Heartbeat fixture"}}))
			other.read(t, ctx)
		}
	}
}
