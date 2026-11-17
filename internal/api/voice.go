package api

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/go-chi/chi/v5"
	"github.com/jackc/pgx/v5"
)

// VoiceConfig holds Cloudflare Realtime TURN credentials. Only short-lived
// ICE credentials are sent to browsers; the API token stays on the backend.
type VoiceConfig struct {
	TURNTokenID  string
	TURNAPIToken string
	Client       *http.Client
}

type iceServer struct {
	URLs       []string `json:"urls"`
	Username   string   `json:"username,omitempty"`
	Credential string   `json:"credential,omitempty"`
}

type voicePeer struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	CanSpeak bool   `json:"canSpeak"`
}

type voiceMessage struct {
	Type        string          `json:"type"`
	To          string          `json:"to,omitempty"`
	From        string          `json:"from,omitempty"`
	Self        *voicePeer      `json:"self,omitempty"`
	Peers       []voicePeer     `json:"peers,omitempty"`
	ICEServers  []iceServer     `json:"iceServers,omitempty"`
	Description json.RawMessage `json:"description,omitempty"`
	Candidate   json.RawMessage `json:"candidate,omitempty"`
}

type voiceMember struct {
	peer     voicePeer
	userID   string
	inbox    chan voiceMessage
	cancel   context.CancelFunc
	replaced chan struct{}
}

// Signaling is transient and room-scoped. Audio flows directly between browsers
// (or through TURN); it never enters game snapshots or the database.
type voiceHub struct {
	mu    sync.Mutex
	rooms map[string]map[string]*voiceMember
}

func newVoiceHub() *voiceHub {
	return &voiceHub{rooms: make(map[string]map[string]*voiceMember)}
}

func deliverVoice(member *voiceMember, message voiceMessage) {
	select {
	case member.inbox <- message:
	default:
		// A stalled connection must reconnect rather than lose signaling silently.
		member.cancel()
	}
}

func (hub *voiceHub) join(code string, member *voiceMember, servers []iceServer) {
	hub.mu.Lock()
	defer hub.mu.Unlock()
	room := hub.rooms[code]
	if room == nil {
		room = make(map[string]*voiceMember)
		hub.rooms[code] = room
	}
	// A second tab replaces this user's old audio connection, preventing echo.
	for id, previous := range room {
		if previous.userID == member.userID {
			close(previous.replaced)
			delete(room, id)
		}
	}
	member.replaced = make(chan struct{})
	room[member.peer.ID] = member
	peers := make([]voicePeer, 0, len(room))
	for _, peer := range room {
		peers = append(peers, peer.peer)
	}
	deliverVoice(member, voiceMessage{Type: "welcome", Self: &member.peer, Peers: peers, ICEServers: servers})
	for _, peer := range room {
		if peer != member {
			deliverVoice(peer, voiceMessage{Type: "peers", Peers: peers})
		}
	}
}

func (hub *voiceHub) leave(code, id string) {
	hub.mu.Lock()
	defer hub.mu.Unlock()
	room := hub.rooms[code]
	if room[id] == nil {
		return
	}
	delete(room, id)
	if len(room) == 0 {
		delete(hub.rooms, code)
		return
	}
	peers := make([]voicePeer, 0, len(room))
	for _, member := range room {
		peers = append(peers, member.peer)
	}
	for _, member := range room {
		deliverVoice(member, voiceMessage{Type: "peers", Peers: peers})
	}
}

func (hub *voiceHub) relay(code, sender string, message voiceMessage) {
	hub.mu.Lock()
	defer hub.mu.Unlock()
	room := hub.rooms[code]
	if room[sender] == nil || message.To == sender {
		return
	}
	target := room[message.To]
	if target == nil {
		return
	}
	// Construct the envelope here: clients cannot impersonate a peer or inject
	// permission/roster messages, even if they know another connection's ID.
	deliverVoice(target, voiceMessage{
		Type: "signal", From: sender,
		Description: message.Description, Candidate: message.Candidate,
	})
}

func (s *Server) voiceSocket(w http.ResponseWriter, r *http.Request) {
	user, ok := s.requireUser(w, r)
	if !ok {
		return
	}
	if !s.allowedOrigin(r.Header.Get("Origin")) {
		fail(w, http.StatusForbidden, "Origen no permitido.")
		return
	}
	code := strings.ToUpper(chi.URLParam(r, "code"))
	canSpeak, err := s.host.VoiceAccess(code, user.ID)
	if err != nil {
		fail(w, http.StatusNotFound, "No encontramos esa mesa.")
		return
	}
	servers, err := s.voiceICEServers(r.Context(), user.ID)
	if err != nil {
		if errors.Is(err, errVoiceCredentialsLimited) {
			fail(w, http.StatusTooManyRequests, "Espera un momento antes de volver al audio.")
		} else {
			s.log.Error("voice credentials unavailable", "error", err)
			fail(w, http.StatusServiceUnavailable, "No pudimos conectar el audio. Inténtalo de nuevo.")
		}
		return
	}
	// Credential issuance can take several seconds; recheck room access before
	// registering a peer, in case the user left or changed seats meanwhile.
	canSpeak, updates, unsubscribe, err := s.host.SubscribeVoiceAccess(code, user.ID)
	if err != nil {
		fail(w, http.StatusNotFound, "No encontramos esa mesa.")
		return
	}
	defer unsubscribe()
	origins := make([]string, 0, len(s.config.Origins))
	for _, origin := range s.config.Origins {
		origins = append(origins, strings.TrimPrefix(strings.TrimPrefix(origin, "https://"), "http://"))
	}
	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{OriginPatterns: origins})
	if err != nil {
		return
	}
	defer func() { _ = conn.CloseNow() }()
	conn.SetReadLimit(32768)
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()
	member := &voiceMember{
		peer:   voicePeer{ID: rand.Text(), Name: user.Name, CanSpeak: canSpeak},
		userID: user.ID, inbox: make(chan voiceMessage, 128), cancel: cancel,
	}
	s.voice.join(code, member, servers)
	defer s.voice.leave(code, member.peer.ID)
	incoming := make(chan voiceMessage, 8)
	go func() {
		defer cancel()
		for {
			var message voiceMessage
			if wsjson.Read(ctx, conn, &message) != nil {
				return
			}
			select {
			case incoming <- message:
			case <-ctx.Done():
				return
			}
		}
	}()
	valid := func() bool {
		current, authErr := s.authenticate(r.WithContext(ctx))
		if authErr != nil && !errors.Is(authErr, pgx.ErrNoRows) && !errors.Is(authErr, http.ErrNoCookie) {
			_ = conn.Close(websocket.StatusTryAgainLater, "No pudimos comprobar la sesión.")
			return false
		}
		if authErr != nil || current.ID != user.ID {
			_ = conn.Close(websocket.StatusPolicyViolation, "Sesión de voz finalizada.")
			return false
		}
		canSpeak, accessErr := s.host.VoiceAccess(code, user.ID)
		// Rejoin after a seat change, so every peer gets fresh permissions.
		if accessErr != nil {
			_ = conn.Close(websocket.StatusPolicyViolation, "La mesa se cerró.")
			return false
		}
		if member.peer.CanSpeak != canSpeak {
			_ = conn.Close(websocket.StatusServiceRestart, "Cambió tu permiso de voz.")
			return false
		}
		return true
	}
	const heartbeatInterval = 25 * time.Second
	lastReceived := time.Now()
	heartbeat := time.NewTicker(heartbeatInterval)
	defer heartbeat.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-member.replaced:
			_ = conn.Close(4001, "El audio se abrió en otra pestaña.")
			return
		case <-updates:
			if !valid() {
				return
			}
		case <-heartbeat.C:
			if !valid() {
				return
			}
			// A proxy can terminate control pings on its own WebSocket leg.
			// Send application traffic all the way to an otherwise idle browser.
			if time.Since(lastReceived) >= 2*heartbeatInterval {
				pingCtx, done := context.WithTimeout(ctx, 5*time.Second)
				err = conn.Ping(pingCtx)
				done()
				if err != nil {
					return
				}
				lastReceived = time.Now()
			}
			deliverVoice(member, voiceMessage{Type: "heartbeat"})
		case message := <-member.inbox:
			payload, err := json.Marshal(message)
			if err != nil {
				return
			}
			writeCtx, done := context.WithTimeout(ctx, 5*time.Second)
			err = conn.Write(writeCtx, websocket.MessageText, payload)
			done()
			if err != nil {
				return
			}
		case message := <-incoming:
			lastReceived = time.Now()
			if !valid() {
				return
			}
			if !s.allow("voice:"+user.ID, 600) {
				_ = conn.Close(websocket.StatusPolicyViolation, "Demasiados mensajes de voz.")
				return
			}
			switch message.Type {
			case "refresh":
				servers, err := s.voiceICEServers(ctx, user.ID)
				if err != nil {
					s.log.Error("voice credential refresh unavailable", "error", err)
					deliverVoice(member, voiceMessage{Type: "refresh_retry"})
					continue
				}
				deliverVoice(member, voiceMessage{Type: "ice", ICEServers: servers})
			case "signal":
				s.voice.relay(code, member.peer.ID, message)
			}
		}
	}
}
