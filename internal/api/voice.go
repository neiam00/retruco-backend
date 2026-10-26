package api

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha1" // #nosec G505 -- TURN REST credentials require HMAC-SHA1.
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/go-chi/chi/v5"
)

// VoiceConfig uses coturn's time-limited REST authentication. The shared secret
// stays on the server; browsers receive credentials valid for one hour.
type VoiceConfig struct {
	STUNURLs   []string
	TURNURLs   []string
	TURNSecret string
}

type iceServer struct {
	URLs       []string `json:"urls"`
	Username   string   `json:"username,omitempty"`
	Credential string   `json:"credential,omitempty"`
}

func (config VoiceConfig) iceServers(userID string) []iceServer {
	servers := []iceServer{}
	if len(config.STUNURLs) > 0 {
		servers = append(servers, iceServer{URLs: config.STUNURLs})
	}
	if len(config.TURNURLs) > 0 && config.TURNSecret != "" {
		username := fmt.Sprintf("%d:%s", time.Now().Add(time.Hour).Unix(), userID)
		mac := hmac.New(sha1.New, []byte(config.TURNSecret))
		_, _ = mac.Write([]byte(username))
		servers = append(servers, iceServer{
			URLs: config.TURNURLs, Username: username,
			Credential: base64.StdEncoding.EncodeToString(mac.Sum(nil)),
		})
	}
	return servers
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
	peer   voicePeer
	userID string
	inbox  chan voiceMessage
	cancel context.CancelFunc
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
			previous.cancel()
			delete(room, id)
		}
	}
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
	s.voice.join(code, member, s.config.Voice.iceServers(user.ID))
	defer s.voice.leave(code, member.peer.ID)
	updates, unsubscribe := s.host.Subscribe(code)
	defer unsubscribe()
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
		current, authErr := s.authenticate(r)
		if authErr != nil || current.ID != user.ID {
			return false
		}
		canSpeak, accessErr := s.host.VoiceAccess(code, user.ID)
		// Rejoin after a seat change, so every peer gets fresh permissions.
		return accessErr == nil && member.peer.CanSpeak == canSpeak
	}
	heartbeat := time.NewTicker(25 * time.Second)
	defer heartbeat.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-updates:
			if !valid() {
				return
			}
		case <-heartbeat.C:
			if !valid() {
				return
			}
			pingCtx, done := context.WithTimeout(ctx, 5*time.Second)
			err = conn.Ping(pingCtx)
			done()
			if err != nil {
				return
			}
		case message := <-member.inbox:
			writeCtx, done := context.WithTimeout(ctx, 5*time.Second)
			err = wsjson.Write(writeCtx, conn, message)
			done()
			if err != nil {
				return
			}
		case message := <-incoming:
			if !valid() || !s.allow("voice:"+user.ID, 600) {
				return
			}
			switch message.Type {
			case "refresh":
				deliverVoice(member, voiceMessage{Type: "ice", ICEServers: s.config.Voice.iceServers(user.ID)})
			case "signal":
				s.voice.relay(code, member.peer.ID, message)
			}
		}
	}
}
