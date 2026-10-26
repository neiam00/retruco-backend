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
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/jackc/pgx/v5"
	"github.com/neiam00/retruco-backend/internal/api"
	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// This fixture implements storage only; authentication, room transitions,
// publication, WebSocket encoding, and protocol negotiation use real code.
type streamRepository struct {
	api.Repository
	mu       sync.Mutex
	sessions map[string]store.Session
	matches  map[string][]byte
}

func (r *streamRepository) Session(_ context.Context, hash string) (store.Session, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	value, ok := r.sessions[hash]
	if !ok {
		return store.Session{}, pgx.ErrNoRows
	}
	return value, nil
}

func (r *streamRepository) CreateMatch(_ context.Context, code string, payload []byte, _ []string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.matches[code]; exists {
		return store.ErrMatchCodeInUse
	}
	r.matches[code] = append([]byte(nil), payload...)
	return nil
}

func (r *streamRepository) SaveMatch(_ context.Context, code string, payload []byte, _ bool, _ []string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.matches[code] = append([]byte(nil), payload...)
	return nil
}

func (r *streamRepository) DeleteLobby(_ context.Context, code string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.matches, code)
	return nil
}

func (r *streamRepository) ActiveMatches(context.Context) ([][]byte, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	values := make([][]byte, 0, len(r.matches))
	for _, value := range r.matches {
		values = append(values, append([]byte(nil), value...))
	}
	return values, nil
}

type streamWireMessage struct {
	Type   string          `json:"type"`
	Status int             `json:"status"`
	Error  string          `json:"error"`
	Data   json.RawMessage `json:"data"`
	Stream struct {
		Epoch        string `json:"epoch"`
		Sequence     uint64 `json:"sequence"`
		BaseSequence uint64 `json:"baseSequence"`
	} `json:"stream"`
	History struct {
		Chat *struct {
			Reset bool        `json:"reset"`
			Keep  int         `json:"keep"`
			Items []game.Chat `json:"items"`
		} `json:"chat"`
	} `json:"history"`
}

type streamConnection struct {
	conn     *websocket.Conn
	sequence uint64
	epoch    string
}

func (c *streamConnection) read(t *testing.T, ctx context.Context) (streamWireMessage, game.Snapshot) {
	t.Helper()
	var message streamWireMessage
	require.NoError(t, wsjson.Read(ctx, c.conn, &message))
	var snapshot game.Snapshot
	if message.Type == "snapshot" || message.Type == "game_update" {
		require.NoError(t, json.Unmarshal(message.Data, &snapshot))
		if message.Stream.Epoch != "" {
			if c.epoch == "" {
				c.epoch = message.Stream.Epoch
			}
			assert.Equal(t, c.epoch, message.Stream.Epoch)
			assert.Equal(t, c.sequence+1, message.Stream.Sequence, "wire publications must be contiguous even when room notifications coalesce")
			if message.Type == "game_update" {
				assert.Equal(t, c.sequence, message.Stream.BaseSequence)
			} else {
				assert.Zero(t, message.Stream.BaseSequence)
			}
			c.sequence = message.Stream.Sequence
		}
	}
	return message, snapshot
}

func (c *streamConnection) until(t *testing.T, ctx context.Context, accept func(streamWireMessage, game.Snapshot) bool) (streamWireMessage, game.Snapshot) {
	t.Helper()
	for range 20 {
		message, snapshot := c.read(t, ctx)
		if accept(message, snapshot) {
			return message, snapshot
		}
	}
	t.Fatal("expected WebSocket publication did not arrive")
	return streamWireMessage{}, game.Snapshot{}
}

func newStreamServer(t *testing.T) (context.Context, *httptest.Server, string, []store.User) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	t.Cleanup(cancel)
	repository := &streamRepository{sessions: make(map[string]store.Session), matches: make(map[string][]byte)}
	users := []store.User{{ID: "owner", Name: "Owner"}, {ID: "guest", Name: "Guest"}, {ID: "watcher", Name: "Watcher"}}
	for _, user := range users {
		hash := sha256.Sum256([]byte(user.ID))
		repository.sessions[hex.EncodeToString(hash[:])] = store.Session{User: user, ExpiresAt: time.Now().Add(time.Hour)}
	}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	host := game.New(repository, log)
	code, err := host.Create(ctx, users[0], rules.Config{Players: 2, Target: 12}, false, game.RoomOptions{})
	require.NoError(t, err)
	require.NoError(t, host.Join(ctx, code, users[1], nil))
	server, err := api.New(ctx, api.Config{Origins: []string{"https://retruco.test"}}, repository, host, log)
	require.NoError(t, err)
	httpServer := httptest.NewServer(server.Handler())
	t.Cleanup(httpServer.Close)
	return ctx, httpServer, code, users
}

func dialStream(t *testing.T, ctx context.Context, server *httptest.Server, code, user string, v2 bool) *streamConnection {
	t.Helper()
	header := http.Header{"Origin": []string{"https://retruco.test"}}
	if user != "" {
		header.Set("Cookie", "retruco_session="+user)
	}
	url := "ws" + strings.TrimPrefix(server.URL, "http") + "/api/matches/" + code + "/ws"
	if v2 {
		url += "?transport=2"
	}
	conn, _, err := websocket.Dial(ctx, url, &websocket.DialOptions{HTTPHeader: header})
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.CloseNow() })
	return &streamConnection{conn: conn}
}

func TestGameStreamV2PlaysAlongsideLegacyAndResynchronizes(t *testing.T) {
	ctx, server, code, users := newStreamServer(t)
	legacy := dialStream(t, ctx, server, code, users[0].ID, false)
	legacyMessage, ownerBaseline := legacy.read(t, ctx)
	require.Equal(t, "snapshot", legacyMessage.Type)
	assert.Empty(t, legacyMessage.Stream.Epoch)
	assert.Equal(t, 0, ownerBaseline.Seat)
	guest := dialStream(t, ctx, server, code, users[1].ID, true)
	guestMessage, guestBaseline := guest.read(t, ctx)
	require.Equal(t, "snapshot", guestMessage.Type)
	require.NotEmpty(t, guestMessage.Stream.Epoch)
	assert.Equal(t, uint64(1), guestMessage.Stream.Sequence)
	assert.Equal(t, 1, guestBaseline.Seat)
	assert.NotNil(t, guestBaseline.Chat)
	watcher := dialStream(t, ctx, server, code, users[2].ID, true)
	watcherMessage, watcherBaseline := watcher.read(t, ctx)
	require.Equal(t, "snapshot", watcherMessage.Type)
	assert.Equal(t, -1, watcherBaseline.Seat)
	assert.Equal(t, 0, watcherBaseline.Perspective)
	assert.Empty(t, watcherBaseline.State.Legal)
	assert.NotEqual(t, guestMessage.Stream.Epoch, watcherMessage.Stream.Epoch, "connections cannot resume one another's stream")

	// Execute a real legal command through whichever player's transport owns
	// the cutting turn. Both protocols must publish the same game revision.
	actor := legacy
	if guestBaseline.State.Turn == 1 {
		actor = guest
	}
	require.NoError(t, wsjson.Write(ctx, actor.conn, map[string]any{"input": game.Input{
		ID: "cut-over-wire", Revision: guestBaseline.State.Revision, Command: rules.Command{Type: "cut", Cut: 1},
	}}))
	playing := func(message streamWireMessage, snapshot game.Snapshot) bool {
		return snapshot.State != nil && snapshot.State.Phase == rules.Playing
	}
	guestUpdate, guestPlaying := guest.until(t, ctx, playing)
	require.Equal(t, "game_update", guestUpdate.Type)
	assert.Equal(t, guestBaseline.State.Revision+1, guestPlaying.State.Revision)
	legacyUpdate, ownerPlaying := legacy.until(t, ctx, playing)
	assert.Equal(t, "snapshot", legacyUpdate.Type)
	assert.Equal(t, guestPlaying.State.Revision, ownerPlaying.State.Revision)
	_, watcherPlaying := watcher.until(t, ctx, playing)
	assert.Equal(t, ownerPlaying.State.Hand, watcherPlaying.State.Hand, "spectators retain the selected perspective's existing visibility")
	assert.Empty(t, watcherPlaying.State.Legal)
	assert.NotEqual(t, guestPlaying.State.Hand, watcherPlaying.State.Hand, "opposite perspectives must not share cached private hands")

	require.NoError(t, wsjson.Write(ctx, legacy.conn, map[string]any{"input": game.Input{ID: "chat-over-legacy", Chat: "Hello from the old client"}}))
	chatUpdate, _ := guest.until(t, ctx, func(message streamWireMessage, _ game.Snapshot) bool {
		return message.History.Chat != nil && len(message.History.Chat.Items) > 0
	})
	require.Equal(t, "game_update", chatUpdate.Type)
	require.NotNil(t, chatUpdate.History.Chat)
	assert.Equal(t, "Hello from the old client", chatUpdate.History.Chat.Items[0].Text)
	assert.NotContains(t, string(chatUpdate.Data), `"chat"`, "history must be delivered in its delta envelope")
	_, legacyChat := legacy.until(t, ctx, func(_ streamWireMessage, snapshot game.Snapshot) bool { return len(snapshot.Chat) == 1 })
	assert.Equal(t, "Hello from the old client", legacyChat.Chat[0].Text)

	require.NoError(t, wsjson.Write(ctx, guest.conn, map[string]bool{"resync": true}))
	resync, refreshed := guest.until(t, ctx, func(message streamWireMessage, _ game.Snapshot) bool { return message.Type == "snapshot" })
	assert.Equal(t, guestMessage.Stream.Epoch, resync.Stream.Epoch)
	require.Len(t, refreshed.Chat, 1)
	assert.Equal(t, legacyChat.Chat[0], refreshed.Chat[0])

	require.NoError(t, wsjson.Write(ctx, watcher.conn, map[string]int{"perspective": 1}))
	_, switched := watcher.until(t, ctx, func(message streamWireMessage, snapshot game.Snapshot) bool {
		return message.Type == "snapshot" && snapshot.Perspective == 1
	})
	assert.Equal(t, -1, switched.Seat)
	assert.Empty(t, switched.State.Legal)
	assert.Equal(t, guestPlaying.State.Hand, switched.State.Hand)
	require.Len(t, switched.Chat, 1, "a perspective switch includes a complete baseline even after missed history updates")
}

func TestGameStreamV2ReturnsTypedHandshakeErrors(t *testing.T) {
	ctx, server, code, users := newStreamServer(t)
	for _, tc := range []struct {
		name, code, user string
		status           int
	}{
		{"authentication", code, "", http.StatusUnauthorized},
		{"missing room", "MISSING-ROOM", users[0].ID, http.StatusNotFound},
	} {
		t.Run(tc.name, func(t *testing.T) {
			connection := dialStream(t, ctx, server, tc.code, tc.user, true)
			message, _ := connection.read(t, ctx)
			assert.Equal(t, "error", message.Type)
			assert.Equal(t, tc.status, message.Status)
			assert.NotEmpty(t, message.Error)
			var next streamWireMessage
			err := wsjson.Read(ctx, connection.conn, &next)
			assert.Equal(t, websocket.StatusPolicyViolation, websocket.CloseStatus(err))

			// Older clients retain HTTP upgrade failures instead of receiving
			// the new typed WebSocket error envelope.
			header := http.Header{"Origin": []string{"https://retruco.test"}}
			if tc.user != "" {
				header.Set("Cookie", "retruco_session="+tc.user)
			}
			conn, response, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(server.URL, "http")+"/api/matches/"+tc.code+"/ws", &websocket.DialOptions{HTTPHeader: header})
			if conn != nil {
				_ = conn.CloseNow()
			}
			require.Error(t, err)
			require.NotNil(t, response)
			assert.Equal(t, tc.status, response.StatusCode)
			_ = response.Body.Close()
		})
	}
}
