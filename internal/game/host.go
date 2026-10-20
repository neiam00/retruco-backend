package game

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math/big"
	"strings"
	"sync"
	"time"

	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
)

type Repository interface {
	SaveMatch(context.Context, string, []byte, bool, []string) error
	ActiveMatches(context.Context) ([][]byte, error)
}

type Seat struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

type Chat struct {
	ID        uint64 `json:"id"`
	Name      string `json:"name"`
	Text      string `json:"text"`
	Spectator bool   `json:"spectator"`
}

type Signal struct {
	ID      uint64  `json:"id"`
	Seat    int     `json:"seat"`
	Kind    string  `json:"kind"`
	Visible [4]bool `json:"-"`
}

type Match struct {
	Code     string
	Private  bool
	Config   rules.Config
	Seats    []Seat
	State    *rules.State
	Deadline time.Time
	Chat     []Chat
	Sequence uint64
	Recent   map[string]uint64
	Signals  []Signal `json:"-"`
}

type Snapshot struct {
	Code        string       `json:"code"`
	Private     bool         `json:"private"`
	Config      rules.Config `json:"config"`
	Seats       []Seat       `json:"seats"`
	Seat        int          `json:"seat"`
	Perspective int          `json:"perspective"`
	State       *rules.View  `json:"state,omitempty"`
	Deadline    time.Time    `json:"deadline"`
	ServerTime  time.Time    `json:"serverTime"`
	Chat        []Chat       `json:"chat"`
	Signals     []Signal     `json:"signals"`
	Sequence    uint64       `json:"sequence"`
}

type Host struct {
	mu         sync.Mutex
	matches    map[string]*Match
	listeners  map[string]map[chan struct{}]bool
	repository Repository
	log        *slog.Logger
}

func New(repository Repository, log *slog.Logger) *Host {
	return &Host{matches: map[string]*Match{}, listeners: map[string]map[chan struct{}]bool{}, repository: repository, log: log}
}

func (h *Host) Restore(ctx context.Context) error {
	payloads, err := h.repository.ActiveMatches(ctx)
	if err != nil {
		return err
	}
	for _, payload := range payloads {
		var match Match
		if err := json.Unmarshal(payload, &match); err != nil {
			return err
		}
		if match.Recent == nil {
			match.Recent = map[string]uint64{}
		}
		h.matches[match.Code] = &match
	}
	return nil
}

func randomInt(n int) (int, error) {
	value, err := rand.Int(rand.Reader, big.NewInt(int64(n)))
	if err != nil {
		return 0, err
	}
	return int(value.Int64()), nil
}

func shuffledDeck() ([]rules.Card, error) {
	deck := rules.Deck()
	for i := len(deck) - 1; i > 0; i-- {
		j, err := randomInt(i + 1)
		if err != nil {
			return nil, err
		}
		deck[i], deck[j] = deck[j], deck[i]
	}
	return deck, nil
}

func matchCode() (string, error) {
	var b strings.Builder
	for i := 0; i < 8; i++ {
		alphabet := "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		if i >= 4 {
			alphabet = "0123456789"
		}
		if i == 4 {
			b.WriteByte('-')
		}
		index, err := randomInt(len(alphabet))
		if err != nil {
			return "", err
		}
		b.WriteByte(alphabet[index])
	}
	return b.String(), nil
}

func (h *Host) save(ctx context.Context, match *Match) error {
	payload, err := json.Marshal(match)
	if err != nil {
		return err
	}
	members := []string{}
	for _, seat := range match.Seats {
		members = append(members, seat.ID)
	}
	finished := match.State != nil && match.State.Phase == rules.MatchOver
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := h.repository.SaveMatch(ctx, match.Code, payload, finished, members); err != nil {
		h.log.Error("persist match", "code", match.Code, "error", err)
		return fmt.Errorf("no se pudo guardar la jugada. Inténtalo de nuevo")
	}
	return nil
}

func (h *Host) Create(ctx context.Context, user store.User, config rules.Config, private bool) (string, error) {
	if (config.Players != 2 && config.Players != 4) || (config.Target != 12 && config.Target != 24) {
		return "", fmt.Errorf("configuración inválida")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.matches) >= 1000 {
		return "", fmt.Errorf("no hay espacio para más mesas en este momento")
	}
	var code string
	for {
		generated, err := matchCode()
		if err != nil {
			return "", err
		}
		if h.matches[generated] == nil {
			code = generated
			break
		}
	}
	match := &Match{Code: code, Private: private, Config: config, Seats: []Seat{{ID: user.ID, Name: user.Name}}, Recent: map[string]uint64{}, Chat: []Chat{}, Sequence: 1}
	if err := h.save(ctx, match); err != nil {
		return "", err
	}
	h.matches[code] = match
	return code, nil
}

func (h *Host) Join(ctx context.Context, code string, user store.User) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	m := h.matches[code]
	if m == nil {
		return fmt.Errorf("no encontramos esa mesa")
	}
	for _, seat := range m.Seats {
		if seat.ID == user.ID {
			return nil
		}
	}
	if m.State != nil || len(m.Seats) >= m.Config.Players {
		return fmt.Errorf("la mesa está llena; puedes entrar a mirar")
	}
	next := clone(m)
	next.Seats = append(next.Seats, Seat{ID: user.ID, Name: user.Name})
	if len(next.Seats) == next.Config.Players {
		deck, err := shuffledDeck()
		if err != nil {
			return err
		}
		dealer, err := randomInt(next.Config.Players)
		if err != nil {
			return err
		}
		state, err := rules.New(next.Config, dealer, deck)
		if err != nil {
			return err
		}
		next.State = &state
		next.Deadline = time.Now().Add(30 * time.Second)
	}
	next.Sequence++
	return h.commit(ctx, next)
}

func clone(m *Match) *Match {
	next := *m
	next.Seats = append([]Seat{}, m.Seats...)
	next.Chat = append([]Chat{}, m.Chat...)
	next.Signals = append([]Signal{}, m.Signals...)
	next.Recent = make(map[string]uint64, len(m.Recent))
	for key, value := range m.Recent {
		next.Recent[key] = value
	}
	return &next
}

func (h *Host) commit(ctx context.Context, next *Match) error {
	if err := h.save(ctx, next); err != nil {
		return err
	}
	h.matches[next.Code] = next
	for channel := range h.listeners[next.Code] {
		select {
		case channel <- struct{}{}:
		default:
		}
	}
	return nil
}

func seatOf(m *Match, userID string) int {
	for seat, user := range m.Seats {
		if user.ID == userID {
			return seat
		}
	}
	return -1
}

func (h *Host) Snapshot(code, userID string, perspective int) (Snapshot, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	m := h.matches[code]
	if m == nil {
		return Snapshot{}, fmt.Errorf("no encontramos esa mesa")
	}
	seat := seatOf(m, userID)
	if seat >= 0 {
		perspective = seat
	}
	if perspective < 0 || perspective >= len(m.Seats) {
		perspective = 0
	}
	v := Snapshot{Code: code, Private: m.Private, Config: m.Config, Seats: append([]Seat{}, m.Seats...), Seat: seat, Perspective: perspective, Deadline: m.Deadline, ServerTime: time.Now(), Chat: append([]Chat{}, m.Chat...), Signals: []Signal{}, Sequence: m.Sequence}
	if m.State != nil {
		view := rules.ViewFor(*m.State, perspective, seat < 0)
		v.State = &view
	}
	for _, signal := range m.Signals {
		if signal.Visible[perspective] {
			v.Signals = append(v.Signals, signal)
		}
	}
	return v, nil
}

type Listing struct {
	Code         string `json:"code"`
	Players      int    `json:"players"`
	Capacity     int    `json:"capacity"`
	Target       int    `json:"target"`
	FlorRequired bool   `json:"florRequired"`
	Started      bool   `json:"started"`
	Host         string `json:"host"`
}

func (h *Host) List() []Listing {
	h.mu.Lock()
	defer h.mu.Unlock()
	list := []Listing{}
	for _, m := range h.matches {
		if m.Private || len(m.Seats) == 0 || (m.State != nil && m.State.Phase == rules.MatchOver) {
			continue
		}
		list = append(list, Listing{Code: m.Code, Players: len(m.Seats), Capacity: m.Config.Players, Target: m.Config.Target, FlorRequired: m.Config.FlorRequired, Started: m.State != nil, Host: m.Seats[0].Name})
	}
	return list
}

type Input struct {
	ID       string        `json:"id"`
	Revision uint64        `json:"revision"`
	Command  rules.Command `json:"command"`
	Chat     string        `json:"chat,omitempty"`
	Signal   string        `json:"signal,omitempty"`
}

func (h *Host) Handle(ctx context.Context, code string, user store.User, input Input) error {
	if input.ID == "" || len(input.ID) > 80 {
		return fmt.Errorf("identificador de acción inválido")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	m := h.matches[code]
	if m == nil {
		return fmt.Errorf("no encontramos esa mesa")
	}
	key := user.ID + ":" + input.ID
	if _, exists := m.Recent[key]; exists {
		return nil
	}
	if input.Chat == "" && input.Signal == "" && m.State != nil && !m.Deadline.IsZero() && !time.Now().Before(m.Deadline) {
		if err := h.expire(ctx, m); err != nil {
			return err
		}
		m = h.matches[code]
	}
	next := clone(m)
	seat := seatOf(m, user.ID)
	switch {
	case input.Chat != "":
		message := strings.TrimSpace(input.Chat)
		if message == "" || len([]rune(message)) > 500 {
			return fmt.Errorf("el mensaje debe tener entre 1 y 500 caracteres")
		}
		next.Chat = append(next.Chat, Chat{ID: m.Sequence + 1, Name: user.Name, Text: message, Spectator: seat < 0})
		if len(next.Chat) > 100 {
			next.Chat = next.Chat[len(next.Chat)-100:]
		}
	case input.Signal != "":
		if seat < 0 || m.State == nil || m.State.Phase != rules.Playing {
			return fmt.Errorf("no puedes hacer señas ahora")
		}
		if !validSignal(input.Signal) {
			return fmt.Errorf("seña inválida")
		}
		signal := Signal{ID: m.Sequence + 1, Seat: seat, Kind: input.Signal}
		for other := range m.Seats {
			if other%2 == seat%2 {
				signal.Visible[other] = true
			} else {
				roll, err := randomInt(10)
				if err != nil {
					return err
				}
				signal.Visible[other] = roll == 0
			}
		}
		next.Signals = append(next.Signals, signal)
		if len(next.Signals) > 20 {
			next.Signals = next.Signals[len(next.Signals)-20:]
		}
	default:
		if seat < 0 {
			return fmt.Errorf("los espectadores no pueden jugar cartas")
		}
		if m.State == nil {
			if input.Command.Type != "leave" {
				return fmt.Errorf("esperando a los demás jugadores")
			}
			next.Seats = append(next.Seats[:seat], next.Seats[seat+1:]...)
		} else {
			if input.Revision != m.State.Revision {
				return fmt.Errorf("la mesa cambió. Revisa el estado actualizado")
			}
			state, _, err := rules.Apply(*m.State, seat, input.Command)
			if err != nil {
				return fmt.Errorf("esa jugada no está disponible ahora")
			}
			next.State = &state
			next.Deadline = time.Now().Add(30 * time.Second)
			if state.Phase == rules.HandOver {
				next.Deadline = time.Now().Add(5 * time.Second)
			}
			if state.Phase == rules.MatchOver {
				next.Deadline = time.Time{}
			}
		}
	}
	next.Sequence++
	next.Recent[key] = next.Sequence
	for id, sequence := range next.Recent {
		if sequence+256 < next.Sequence {
			delete(next.Recent, id)
		}
	}
	return h.commit(ctx, next)
}

func validSignal(signal string) bool {
	return signal == "perico" || signal == "perica" || signal == "seven_spades" || signal == "seven_coins" || signal == "receta"
}

func (h *Host) Subscribe(code string) (<-chan struct{}, func()) {
	h.mu.Lock()
	defer h.mu.Unlock()
	channel := make(chan struct{}, 1)
	if h.listeners[code] == nil {
		h.listeners[code] = map[chan struct{}]bool{}
	}
	h.listeners[code][channel] = true
	return channel, func() {
		h.mu.Lock()
		defer h.mu.Unlock()
		delete(h.listeners[code], channel)
		if len(h.listeners[code]) == 0 {
			delete(h.listeners, code)
		}
	}
}

func (h *Host) Run(ctx context.Context) {
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case now := <-ticker.C:
			h.mu.Lock()
			for _, m := range h.matches {
				if m.State == nil || m.Deadline.IsZero() || now.Before(m.Deadline) {
					continue
				}
				if err := h.expire(ctx, m); err != nil && !errors.Is(err, context.Canceled) {
					h.log.Error("match timeout failed", "code", m.Code, "error", err)
				}
			}
			h.mu.Unlock()
		}
	}
}

func (h *Host) expire(ctx context.Context, m *Match) error {
	next := clone(m)
	var state rules.State
	var err error
	if m.State.Phase == rules.HandOver {
		deck, drawErr := shuffledDeck()
		if drawErr != nil {
			return drawErr
		}
		state, err = rules.StartHand(*m.State, deck)
		next.Signals = nil
	} else {
		choice := 0
		if m.State.Phase == rules.Cutting {
			choice, err = randomInt(39)
			choice++
		} else if m.State.Pending == nil {
			choice, err = randomInt(len(m.State.Remaining(m.State.Turn)))
		}
		if err != nil {
			return err
		}
		state, _, err = rules.Timeout(*m.State, choice)
	}
	if err != nil {
		return err
	}
	next.State = &state
	next.Sequence++
	next.Deadline = time.Now().Add(30 * time.Second)
	if state.Phase == rules.HandOver {
		next.Deadline = time.Now().Add(5 * time.Second)
	}
	if state.Phase == rules.MatchOver {
		next.Deadline = time.Time{}
	}
	return h.commit(ctx, next)
}
