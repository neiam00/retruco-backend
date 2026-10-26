package api

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"golang.org/x/crypto/bcrypt"

	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
)

type Repository interface {
	CreateUser(context.Context, store.User, string, string) error
	Credentials(context.Context, string) (store.User, string, error)
	CreateSession(context.Context, string, string, time.Time) error
	Session(context.Context, string) (store.Session, error)
	DeleteSession(context.Context, string) error
	History(context.Context, string) ([]store.HistoryEntry, error)
}

type Config struct {
	Origins       []string
	SecureCookies bool
	ProxySecret   string
	Voice         VoiceConfig
	Turnstile     TurnstileConfig
}

type Server struct {
	config          Config
	store           Repository
	host            *game.Host
	log             *slog.Logger
	ctx             context.Context
	dummyHash       []byte
	limitsMu        sync.Mutex
	limits          map[string]window
	limitOrder      []string
	voice           *voiceHub
	turnstileClient *http.Client
	sessions        *sessionCache
	passwordSlots   chan struct{}
}

type window struct {
	start time.Time
	count int
}

func New(ctx context.Context, config Config, repository Repository, host *game.Host, log *slog.Logger) (*Server, error) {
	if (len(config.Voice.TURNURLs) > 0) != (config.Voice.TURNSecret != "") {
		return nil, fmt.Errorf("voice TURN URLs and shared secret must be configured together")
	}
	hash, err := bcrypt.GenerateFromPassword([]byte("unused-login-timing-placeholder"), 12)
	if err != nil {
		return nil, err
	}
	turnstileClient := config.Turnstile.Client
	if turnstileClient == nil {
		turnstileClient = &http.Client{
			Timeout: 5 * time.Second,
			CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
				return http.ErrUseLastResponse
			},
		}
	}
	return &Server{
		config:          config,
		store:           repository,
		host:            host,
		log:             log,
		ctx:             ctx,
		dummyHash:       hash,
		limits:          map[string]window{},
		voice:           newVoiceHub(),
		turnstileClient: turnstileClient,
		sessions:        newSessionCache(),
		passwordSlots:   make(chan struct{}, 1),
	}, nil
}

func (s *Server) Handler() http.Handler {
	r := chi.NewRouter()
	r.Use(middleware.RequestID, middleware.Recoverer, s.security)
	r.Get("/api/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 200, map[string]any{"status": "ok", "rulesVersion": rules.Version, "deploymentNotices": true})
	})
	r.Get("/api/session", s.session)
	r.Post("/api/guest", s.guest)
	r.Post("/api/register", s.register)
	r.Get("/api/turnstile", s.turnstileConfig)
	// Keep the existing endpoint for clients loaded before login verification shipped.
	r.Get("/api/registration", s.turnstileConfig)
	r.Post("/api/login", s.login)
	r.Post("/api/logout", s.logout)
	r.Get("/api/history", s.history)
	r.Get("/api/matches", func(w http.ResponseWriter, r *http.Request) {
		data, err := s.host.ListingJSON()
		if err != nil {
			s.internalError(w, err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, data)
	})
	r.Get("/api/matches/events", s.matchEvents)
	r.Post("/api/matches", s.createMatch)
	r.Get("/api/matches/{code}", s.matchSnapshot)
	r.Post("/api/matches/{code}/join", s.joinMatch)
	r.Get("/api/matches/{code}/ws", s.socket)
	r.Get("/api/matches/{code}/voice", s.voiceSocket)
	mux := http.NewServeMux()
	mux.HandleFunc("/internal/deployment", s.deployment)
	mux.Handle("/", r)
	return mux
}

func (s *Server) allowedOrigin(origin string) bool {
	for _, allowed := range s.config.Origins {
		if origin == allowed {
			return true
		}
	}
	return false
}

func (s *Server) security(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("Referrer-Policy", "same-origin")
		if s.config.ProxySecret != "" && subtle.ConstantTimeCompare(
			[]byte(r.Header.Get("X-Retruco-Proxy-Secret")), []byte(s.config.ProxySecret),
		) != 1 {
			fail(w, http.StatusForbidden, "Solicitud no permitida.")
			return
		}
		origin := r.Header.Get("Origin")
		if origin != "" {
			if !s.allowedOrigin(origin) {
				fail(w, 403, "Origen no permitido.")
				return
			}
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Access-Control-Allow-Credentials", "true")
			w.Header().Add("Vary", "Origin")
		}
		if r.Method == http.MethodOptions {
			if origin == "" {
				fail(w, 403, "Origen requerido.")
				return
			}
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
			w.WriteHeader(http.StatusNoContent)
			return
		}
		// Origin validation covers every mutation, including login/guest creation.
		if r.Method == http.MethodPost && (origin == "" || !strings.HasPrefix(r.Header.Get("Content-Type"), "application/json")) {
			fail(w, 403, "Solicitud no permitida.")
			return
		}
		ip, _, err := net.SplitHostPort(r.RemoteAddr)
		if err != nil {
			ip = r.RemoteAddr
		}
		if s.config.ProxySecret != "" {
			// Only the authenticated edge may supply a visitor address. Direct
			// development requests always use their actual connection address.
			if forwarded := net.ParseIP(r.Header.Get("X-Retruco-Client-IP")); forwarded != nil {
				ip = forwarded.String()
			}
		}
		limit := 120
		// Paths (including unknown routes and match codes) are attacker-controlled.
		// Keep a fixed number of HTTP buckets per client, separate from WS users.
		key := "http:" + ip
		if r.URL.Path == "/api/login" || r.URL.Path == "/api/register" || r.URL.Path == "/api/guest" {
			limit = 10
			key += ":" + r.URL.Path
		}
		if !s.allow(key, limit) {
			w.Header().Set("Retry-After", "60")
			fail(w, 429, "Demasiadas solicitudes. Intenta en un minuto.")
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, 8192)
		next.ServeHTTP(w, r)
	})
}

func (s *Server) allow(key string, maximum int) bool {
	s.limitsMu.Lock()
	defer s.limitsMu.Unlock()
	now := time.Now()
	// Windows have the same duration and are created under this lock. Expiry
	// order therefore matches insertion order; active clients need no full scan.
	for len(s.limitOrder) > 0 {
		id := s.limitOrder[0]
		if now.Sub(s.limits[id].start) < time.Minute {
			break
		}
		delete(s.limits, id)
		s.limitOrder[0] = ""
		s.limitOrder = s.limitOrder[1:]
	}
	entry := s.limits[key]
	if entry.start.IsZero() {
		if len(s.limits) >= 10000 {
			return false
		}
		entry.start = now
		s.limitOrder = append(s.limitOrder, key)
	}
	entry.count++
	s.limits[key] = entry
	return entry.count <= maximum
}

func decode(r *http.Request, target any) error {
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return fmt.Errorf("unexpected trailing body")
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func fail(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}

func token() string { return rand.Text() }
func digest(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func (s *Server) authenticate(r *http.Request) (store.User, error) {
	cookie, err := r.Cookie("retruco_session")
	if err != nil {
		return store.User{}, err
	}
	return s.sessions.lookup(r.Context(), digest(cookie.Value), s.store.Session)
}

func (s *Server) requireUser(w http.ResponseWriter, r *http.Request) (store.User, bool) {
	user, err := s.authenticate(r)
	if err != nil {
		fail(w, 401, "Entra con tu nombre para continuar.")
		return store.User{}, false
	}
	return user, true
}

func (s *Server) setSession(w http.ResponseWriter, r *http.Request, user store.User) {
	value := token()
	expires := time.Now().Add(30 * 24 * time.Hour)
	if err := s.store.CreateSession(r.Context(), digest(value), user.ID, expires); err != nil {
		s.internalError(w, err)
		return
	}
	// #nosec G124 -- Secure is deployment-configured to support local HTTP development; production enables SecureCookies.
	http.SetCookie(w, &http.Cookie{Name: "retruco_session", Value: value, Path: "/", Expires: expires, MaxAge: 30 * 24 * 3600, HttpOnly: true, Secure: s.config.SecureCookies, SameSite: http.SameSiteLaxMode})
	writeJSON(w, 200, user)
}

func (s *Server) session(w http.ResponseWriter, r *http.Request) {
	user, err := s.authenticate(r)
	if errors.Is(err, http.ErrNoCookie) || errors.Is(err, pgx.ErrNoRows) {
		writeJSON(w, 200, nil)
		return
	}
	if err != nil {
		s.internalError(w, err)
		return
	}
	writeJSON(w, 200, user)
}

func (s *Server) guest(w http.ResponseWriter, r *http.Request) {
	var input struct {
		Name string `json:"name"`
	}
	if decode(r, &input) != nil {
		fail(w, 400, "Nombre inválido.")
		return
	}
	input.Name = strings.TrimSpace(input.Name)
	if utf8.RuneCountInString(input.Name) < 2 || utf8.RuneCountInString(input.Name) > 24 {
		fail(w, 400, "Usa un nombre de 2 a 24 caracteres.")
		return
	}
	user := store.User{ID: token(), Name: input.Name}
	if err := s.store.CreateUser(r.Context(), user, "", ""); err != nil {
		s.internalError(w, err)
		return
	}
	s.setSession(w, r, user)
}

var usernamePattern = regexp.MustCompile(`^[a-z0-9_]{3,24}$`)

type credentials struct {
	Username       string `json:"username"`
	Password       string `json:"password"`
	TurnstileToken string `json:"turnstileToken"`
}

func (s *Server) register(w http.ResponseWriter, r *http.Request) {
	var input credentials
	if decode(r, &input) != nil {
		fail(w, 400, "Datos inválidos.")
		return
	}
	input.Username = strings.ToLower(strings.TrimSpace(input.Username))
	if !usernamePattern.MatchString(input.Username) || len(input.Password) < 10 || len(input.Password) > 72 {
		fail(w, 400, "Usa 3–24 letras, números o guiones bajos y una contraseña de 10–72 bytes.")
		return
	}
	if !s.requireTurnstile(w, r, input.TurnstileToken, "register") {
		return
	}
	release, ok := s.passwordAdmission(w)
	if !ok {
		return
	}
	defer release()
	hash, err := bcrypt.GenerateFromPassword([]byte(input.Password), 12)
	if err != nil {
		s.internalError(w, err)
		return
	}
	user := store.User{ID: token(), Name: input.Username, Registered: true}
	if err := s.store.CreateUser(r.Context(), user, input.Username, string(hash)); err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" {
			fail(w, 409, "Ese nombre de usuario ya está ocupado.")
			return
		}
		s.internalError(w, err)
		return
	}
	s.setSession(w, r, user)
}

func (s *Server) login(w http.ResponseWriter, r *http.Request) {
	var input credentials
	if decode(r, &input) != nil || len(input.Password) > 72 {
		fail(w, 400, "Datos inválidos.")
		return
	}
	if !s.requireTurnstile(w, r, input.TurnstileToken, "login") {
		return
	}
	release, ok := s.passwordAdmission(w)
	if !ok {
		return
	}
	defer release()
	user, hash, err := s.store.Credentials(r.Context(), strings.ToLower(strings.TrimSpace(input.Username)))
	if err != nil && !errors.Is(err, pgx.ErrNoRows) {
		s.internalError(w, err)
		return
	}
	if err != nil {
		hash = string(s.dummyHash)
	}
	if compareErr := bcrypt.CompareHashAndPassword([]byte(hash), []byte(input.Password)); compareErr != nil || err != nil {
		fail(w, 401, "Nombre o contraseña incorrectos.")
		return
	}
	s.setSession(w, r, user)
}

func (s *Server) logout(w http.ResponseWriter, r *http.Request) {
	if cookie, err := r.Cookie("retruco_session"); err == nil {
		hash := digest(cookie.Value)
		if err := s.store.DeleteSession(r.Context(), hash); err != nil {
			s.internalError(w, err)
			return
		}
		// Invalidate before acknowledging logout, including lookups that read
		// the session before deletion but have not yet returned from storage.
		s.sessions.revoke(hash)
	}
	// #nosec G124 -- Cookie deletion preserves the deployment-configured Secure setting used when issuing the cookie.
	http.SetCookie(w, &http.Cookie{Name: "retruco_session", Value: "", Path: "/", MaxAge: -1, HttpOnly: true, Secure: s.config.SecureCookies, SameSite: http.SameSiteLaxMode})
	writeJSON(w, 200, map[string]bool{"ok": true})
}

func (s *Server) history(w http.ResponseWriter, r *http.Request) {
	user, ok := s.requireUser(w, r)
	if !ok {
		return
	}
	if !user.Registered {
		fail(w, 403, "Crea una cuenta para guardar tu historial.")
		return
	}
	history, err := s.store.History(r.Context(), user.ID)
	if err != nil {
		s.internalError(w, err)
		return
	}
	writeJSON(w, 200, history)
}

func (s *Server) createMatch(w http.ResponseWriter, r *http.Request) {
	user, ok := s.requireUser(w, r)
	if !ok {
		return
	}
	var input struct {
		Private bool             `json:"private"`
		Config  rules.Config     `json:"config"`
		Options game.RoomOptions `json:"options"`
	}
	if decode(r, &input) != nil {
		fail(w, 400, "Configuración inválida.")
		return
	}
	code, err := s.host.Create(r.Context(), user, input.Config, input.Private, input.Options)
	if err != nil {
		fail(w, 400, "No se pudo crear la mesa. Revisa la configuración.")
		s.log.Error("create match", "error", err)
		return
	}
	writeJSON(w, 201, map[string]string{"code": code})
}

func (s *Server) joinMatch(w http.ResponseWriter, r *http.Request) {
	user, ok := s.requireUser(w, r)
	if !ok {
		return
	}
	var input struct {
		Seat *int `json:"seat"`
	}
	if err := decode(r, &input); err != nil && !errors.Is(err, io.EOF) {
		fail(w, http.StatusBadRequest, "Revisa el puesto elegido.")
		return
	}
	code := strings.ToUpper(chi.URLParam(r, "code"))
	if err := s.host.Join(r.Context(), code, user, input.Seat); err != nil {
		fail(w, 409, err.Error())
		return
	}
	writeJSON(w, 200, map[string]string{"code": code})
}

func (s *Server) matchSnapshot(w http.ResponseWriter, r *http.Request) {
	user, ok := s.requireUser(w, r)
	if !ok {
		return
	}
	code := strings.ToUpper(chi.URLParam(r, "code"))
	snapshot, err := s.host.Snapshot(code, user.ID, 0)
	if err != nil {
		fail(w, http.StatusNotFound, "No encontramos esa mesa. Revisa el código.")
		return
	}
	writeJSON(w, http.StatusOK, snapshot)
}

func (s *Server) internalError(w http.ResponseWriter, err error) {
	s.log.Error("request failed", "error", err)
	fail(w, 500, "Ocurrió un error. Inténtalo de nuevo.")
}

func (s *Server) socket(w http.ResponseWriter, r *http.Request) {
	v2 := r.URL.Query().Get("transport") == "2"
	user, authErr := s.authenticate(r)
	if authErr != nil && !v2 {
		fail(w, 401, "Inicia sesión para continuar.")
		return
	}
	if !s.allowedOrigin(r.Header.Get("Origin")) {
		fail(w, 403, "Origen no permitido.")
		return
	}
	code := strings.ToUpper(chi.URLParam(r, "code"))
	_, roomErr := s.host.VoiceAccess(code, user.ID)
	if roomErr != nil && !v2 {
		fail(w, 404, roomErr.Error())
		return
	}
	// Exact origins were checked above; the library's same-host check remains
	// useful, and configured cross-origin hosts are passed explicitly.
	origins := make([]string, 0, len(s.config.Origins))
	for _, origin := range s.config.Origins {
		origins = append(origins, strings.TrimPrefix(strings.TrimPrefix(origin, "https://"), "http://"))
	}
	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{OriginPatterns: origins})
	if err != nil {
		return
	}
	defer func() {
		// The connection may already have closed normally or due to a peer disconnect.
		_ = conn.CloseNow()
	}()
	conn.SetReadLimit(8192)
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()
	if authErr != nil || roomErr != nil {
		status, reason := 404, "No encontramos esa mesa. Revisa el código."
		if authErr != nil {
			status, reason = 401, "Inicia sesión para continuar."
		}
		writeCtx, done := context.WithTimeout(ctx, 5*time.Second)
		_ = wsjson.Write(writeCtx, conn, map[string]any{"type": "error", "status": status, "error": reason})
		done()
		_ = conn.Close(websocket.StatusPolicyViolation, reason)
		return
	}
	connectionID := rand.Text()
	disconnect := s.host.ConnectPlayer(code, user.ID, connectionID)
	defer disconnect()
	updates, unsubscribe := s.host.Subscribe(code)
	defer unsubscribe()
	type message struct {
		Input       game.Input `json:"input"`
		Perspective *int       `json:"perspective,omitempty"`
		Resync      bool       `json:"resync,omitempty"`
	}
	incoming := make(chan message, 8)
	go func() {
		defer cancel()
		for {
			var msg message
			if wsjson.Read(ctx, conn, &msg) != nil {
				return
			}
			select {
			case incoming <- msg:
			case <-ctx.Done():
				return
			}
		}
	}()
	perspective := 0
	send := func(value any) error {
		writeCtx, done := context.WithTimeout(ctx, 5*time.Second)
		defer done()
		return wsjson.Write(writeCtx, conn, value)
	}
	stream := gameStream{epoch: rand.Text()}
	snapshot := func(baseline bool) error {
		current, err := s.authenticate(r)
		if err != nil || current.ID != user.ID {
			return fmt.Errorf("session expired")
		}
		if v2 {
			value, err := s.host.Publication(code, user.ID, perspective)
			if err != nil {
				_ = send(map[string]string{"type": "room_closed"})
				return err
			}
			message, err := stream.message(value, s.host.CPUWork(code, user.ID, connectionID), baseline, time.Now())
			if err != nil {
				return err
			}
			return send(message)
		}
		value, err := s.host.Snapshot(code, user.ID, perspective)
		if err != nil {
			_ = send(map[string]string{"type": "room_closed"})
			return err
		}
		return send(map[string]any{
			"type": "snapshot", "data": value,
			"cpu": s.host.CPUWork(code, user.ID, connectionID),
		})
	}
	if snapshot(true) != nil {
		return
	}
	heartbeat := time.NewTicker(25 * time.Second)
	defer heartbeat.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-heartbeat.C:
			current, err := s.authenticate(r)
			if err != nil || current.ID != user.ID {
				return
			}
			pingContext, cancelPing := context.WithTimeout(ctx, 5*time.Second)
			err = conn.Ping(pingContext)
			cancelPing()
			if err != nil {
				return
			}
		case <-updates:
			if snapshot(false) != nil {
				return
			}
		case msg := <-incoming:
			// Recheck the session for each incoming action, so logout revokes it.
			current, err := s.authenticate(r)
			if err != nil || current.ID != user.ID {
				_ = conn.Close(websocket.StatusPolicyViolation, "Sesión finalizada")
				return
			}
			if !s.allow("ws:"+user.ID, 120) {
				if send(map[string]string{"type": "error", "error": "Espera un momento antes de volver a intentar."}) != nil {
					return
				}
				continue
			}
			if v2 && msg.Resync {
				if snapshot(true) != nil {
					return
				}
				continue
			}
			if msg.Perspective != nil {
				perspective = *msg.Perspective
				if snapshot(true) != nil {
					return
				}
				continue
			}
			if err := s.host.HandleFromConnection(ctx, code, user, msg.Input, connectionID); err != nil {
				if send(map[string]string{"type": "error", "error": err.Error()}) != nil {
					return
				}
				if snapshot(true) != nil {
					return
				}
			} else if msg.Input.Command.Type == "leave" && msg.Input.Chat == "" && msg.Input.Signal == "" {
				// Existing matches still acknowledge leaving with their final
				// snapshot; a deleted waiting room sends room_closed instead.
				_ = snapshot(true)
				return
			}
		}
	}
}
