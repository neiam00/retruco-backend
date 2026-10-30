package api_test

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// net.Pipe keeps HTTP upgrades and WebSocket frames on real connections while
// making blocked reads visible to synctest. Production heartbeat and timeout
// durations then elapse on the virtual clock; other stream tests retain TCP.
type virtualListener struct {
	connections chan net.Conn
	done        chan struct{}
	once        sync.Once
}

func (l *virtualListener) Accept() (net.Conn, error) {
	select {
	case conn := <-l.connections:
		return conn, nil
	case <-l.done:
		return nil, net.ErrClosed
	}
}

func (l *virtualListener) Close() error {
	l.once.Do(func() { close(l.done) })
	return nil
}

func (*virtualListener) Addr() net.Addr {
	return &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 80}
}

func (l *virtualListener) dial(ctx context.Context, _, _ string) (net.Conn, error) {
	client, server := net.Pipe()
	select {
	case l.connections <- server:
		return client, nil
	case <-ctx.Done():
		_ = client.Close()
		_ = server.Close()
		return nil, ctx.Err()
	case <-l.done:
		_ = client.Close()
		_ = server.Close()
		return nil, net.ErrClosed
	}
}

func newVirtualHTTPServer(t *testing.T, handler http.Handler) *httptest.Server {
	t.Helper()
	server := httptest.NewUnstartedServer(handler)
	require.NoError(t, server.Listener.Close())
	listener := &virtualListener{connections: make(chan net.Conn), done: make(chan struct{})}
	server.Listener = listener
	server.Start()
	server.Client().Transport = &http.Transport{DialContext: listener.dial}
	t.Cleanup(server.Close)
	return server
}

func newHeartbeatServer(t *testing.T) (context.Context, *httptest.Server, string, []store.User) {
	t.Helper()
	ctx, handler, code, users := newStreamHandler(t, 40*time.Second)
	return ctx, newVirtualHTTPServer(t, handler), code, users
}

// Exercise the production 25-second heartbeat. A live application stream must not
// stall waiting for a redundant control pong; idle ping behavior is unchanged.
func TestGameStreamActiveTrafficDoesNotWaitForControlPong(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, server, code, users := newHeartbeatServer(t)
		var pings atomic.Int64
		conn, _, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(server.URL, "http")+"/api/matches/"+code+"/ws?transport=2", &websocket.DialOptions{
			HTTPHeader:     http.Header{"Origin": {"https://retruco.test"}, "Cookie": {"retruco_session=" + users[0].ID}},
			HTTPClient:     server.Client(),
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
	})
}

// Outbound publication traffic can succeed while a browser or proxy no longer
// answers heartbeats. Such a connection must not remain a seated human forever.
func TestGameStreamOutboundUpdatesDoNotKeepUnresponsivePlayerAlive(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, server, code, users := newHeartbeatServer(t)
		var pings atomic.Int64
		conn, _, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(server.URL, "http")+"/api/matches/"+code+"/ws?transport=2", &websocket.DialOptions{
			HTTPHeader:     http.Header{"Origin": {"https://retruco.test"}, "Cookie": {"retruco_session=" + users[0].ID}},
			HTTPClient:     server.Client(),
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
	})
}
