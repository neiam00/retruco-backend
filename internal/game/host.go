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

// Repository implementations must support concurrent calls for different rooms.
type Repository interface {
	CreateMatch(context.Context, string, []byte, []string) error
	// SaveMatch adds supplied members; nil preserves all existing memberships.
	SaveMatch(context.Context, string, []byte, bool, []string) error
	DeleteLobby(context.Context, string) error
	ActiveMatches(context.Context) ([][]byte, error)
}

type Seat struct {
	ID   string `json:"id"`
	Name string `json:"name"`
	CPU  bool   `json:"cpu,omitempty"`
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

type GameEvent struct {
	ID    uint64      `json:"id"`
	Hand  int         `json:"hand"`
	Event rules.Event `json:"event"`
}

// RoomOptions contains social and hosting choices, separate from game rules.
type RoomOptions struct {
	SpectatorsCanSpeak bool  `json:"spectatorsCanSpeak"`
	CPUSeats           []int `json:"cpuSeats,omitempty"`
}

type Match struct {
	CreatorID    string
	CreatorName  string
	Options      RoomOptions
	Code         string
	Private      bool
	Config       rules.Config
	Seats        []Seat
	State        *rules.State
	Deadline     time.Time
	Chat         []Chat
	Sequence     uint64
	Recent       map[string]uint64
	Signals      []Signal `json:"-"`
	LastActivity time.Time
	Expired      bool
	Events       []GameEvent
	CantoHistory []GameEvent
}

type Snapshot struct {
	Updating      bool         `json:"updating,omitempty"`
	CPUController int          `json:"cpuController"`
	CPUPaused     bool         `json:"cpuPaused"`
	Options       RoomOptions  `json:"options"`
	Code          string       `json:"code"`
	Private       bool         `json:"private"`
	Config        rules.Config `json:"config"`
	Seats         []Seat       `json:"seats"`
	Seat          int          `json:"seat"`
	Perspective   int          `json:"perspective"`
	State         *rules.View  `json:"state,omitempty"`
	Deadline      time.Time    `json:"deadline"`
	ServerTime    time.Time    `json:"serverTime"`
	Chat          []Chat       `json:"chat"`
	Signals       []Signal     `json:"signals"`
	Sequence      uint64       `json:"sequence"`
	Events        []GameEvent  `json:"events"`
	CantoHistory  []GameEvent  `json:"cantoHistory"`
}

type Host struct {
	// mu protects only the registry, cached listings, capacity and listing
	// subscribers. Never hold it across database I/O or while locking a room.
	mu               sync.RWMutex
	rooms            map[string]*room
	listings         map[string]Listing
	reservations     map[string]bool
	active           int
	updating         bool
	deploymentMu     sync.Mutex
	listingListeners map[chan struct{}]bool
	repository       Repository
	log              *slog.Logger
}

func New(repository Repository, log *slog.Logger) *Host {
	return &Host{
		rooms:            make(map[string]*room),
		listings:         make(map[string]Listing),
		reservations:     make(map[string]bool),
		listingListeners: make(map[chan struct{}]bool),
		repository:       repository,
		log:              log,
	}
}

// Restore loads persisted rooms before the host starts serving requests.
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
		if match.LastActivity.IsZero() {
			// Older snapshots predate activity timestamps.
			match.LastActivity = time.Now()
		}
		if match.CantoHistory == nil {
			// Older snapshots only retain recent events from their current hand.
			for _, event := range match.Events {
				if isCanto(event.Event) {
					match.CantoHistory = append(match.CantoHistory, event)
				}
			}
		}
		match.restoreFlorDeclarations()
		r := newRoom(&match)
		if match.State != nil && activeMatch(&match) {
			// Wait for a seated human before restarting any card/canto clock.
			r.recovering = true
			match.Deadline = time.Time{}
			r.deadline = time.Now().Add(restoreReconnectGrace)
		}
		h.mu.Lock()
		h.rooms[match.Code] = r
		if activeMatch(&match) {
			h.active++
		}
		h.updateListing(&match)
		h.mu.Unlock()
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
	return h.persist(ctx, match, false, nil)
}

// Memberships are historical: leaving never removes one, and rejoining is an
// idempotent addition in the store. Ordinary moves have no membership changes.
func addedHumanMembers(previous, next *Match) []string {
	var members []string
	for _, seat := range next.Seats {
		if seat.ID != "" && !seat.CPU && (previous == nil || seatOf(previous, seat.ID) < 0) {
			members = append(members, seat.ID)
		}
	}
	return members
}

func (h *Host) persist(ctx context.Context, match *Match, create bool, members []string) error {
	payload, err := json.Marshal(match)
	if err != nil {
		return err
	}
	finished := match.Expired || (match.State != nil && match.State.Phase == rules.MatchOver)
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if create {
		err = h.repository.CreateMatch(ctx, match.Code, payload, members)
	} else {
		err = h.repository.SaveMatch(ctx, match.Code, payload, finished, members)
	}
	if errors.Is(err, store.ErrMatchCodeInUse) {
		return err
	}
	if err != nil {
		h.log.Error("persist match", "code", match.Code, "error", err)
		return fmt.Errorf("no se pudo guardar la jugada. Inténtalo de nuevo")
	}
	return nil
}

func (h *Host) Create(ctx context.Context, user store.User, config rules.Config, private bool, options RoomOptions) (string, error) {
	if (config.Players != 2 && config.Players != 4) || (config.Target != 12 && config.Target != 24) {
		return "", fmt.Errorf("configuración inválida")
	}
	if err := validateCPUSeats(config.Players, options.CPUSeats); err != nil {
		return "", err
	}
	options.CPUSeats = append([]int(nil), options.CPUSeats...)
	for range 10 {
		generated, err := matchCode()
		if err != nil {
			return "", err
		}
		match := &Match{
			CreatorID:    user.ID,
			CreatorName:  user.Name,
			Options:      options,
			Code:         generated,
			Private:      private,
			Config:       config,
			Seats:        make([]Seat, config.Players),
			Recent:       map[string]uint64{},
			Chat:         []Chat{},
			Sequence:     1,
			LastActivity: time.Now(),
		}
		match.Seats[0] = Seat{ID: user.ID, Name: user.Name}
		if len(options.CPUSeats) > 0 {
			for _, seat := range options.CPUSeats {
				match.Seats[seat] = Seat{
					ID:   fmt.Sprintf("cpu:%s:%d", generated, seat),
					Name: fmt.Sprintf("CPU %d", seat), CPU: true,
				}
			}
			if err := startWhenFull(match); err != nil {
				return "", err
			}
		}
		// Reserve capacity and the code before I/O, without exposing an
		// unpersisted room to joins, listings or timer processing.
		h.mu.Lock()
		if h.active+len(h.reservations) >= 1000 {
			h.mu.Unlock()
			return "", fmt.Errorf("no hay espacio para más mesas en este momento")
		}
		if h.rooms[generated] != nil || h.reservations[generated] {
			h.mu.Unlock()
			continue
		}
		h.reservations[generated] = true
		h.mu.Unlock()

		err = h.persist(ctx, match, true, addedHumanMembers(nil, match))
		h.mu.Lock()
		delete(h.reservations, generated)
		if err == nil {
			h.rooms[generated] = newRoom(match)
			h.active++
			h.updateListing(match)
			h.notifyListings()
		}
		h.mu.Unlock()
		if errors.Is(err, store.ErrMatchCodeInUse) {
			continue
		}
		if err != nil {
			return "", err
		}
		return generated, nil
	}
	return "", fmt.Errorf("no pudimos reservar un código de mesa. Inténtalo de nuevo")
}

// Join claims the requested seat atomically. A nil seat supports older clients
// by choosing the first vacancy; existing players retain their current seat.
func (h *Host) Join(ctx context.Context, code string, user store.User, requested *int) error {
	r := h.lockRoom(code)
	if r == nil {
		return fmt.Errorf("no encontramos esa mesa")
	}
	defer r.mu.Unlock()
	m := r.match
	for _, seat := range m.Seats {
		if seat.ID == user.ID {
			return nil
		}
	}
	if m.State != nil || occupiedSeats(m) >= m.Config.Players {
		return fmt.Errorf("la mesa está llena; puedes entrar a mirar")
	}
	next := clone(m)
	// Older persisted lobbies stored only occupied seats.
	for len(next.Seats) < next.Config.Players {
		next.Seats = append(next.Seats, Seat{})
	}
	chosen := -1
	if requested != nil {
		chosen = *requested
		if chosen < 0 || chosen >= next.Config.Players {
			return fmt.Errorf("ese puesto no existe")
		}
		if next.Seats[chosen].ID != "" {
			return fmt.Errorf("ese puesto ya está ocupado; elige otro")
		}
	} else {
		for seat, player := range next.Seats {
			if player.ID == "" {
				chosen = seat
				break
			}
		}
	}
	next.Seats[chosen] = Seat{ID: user.ID, Name: user.Name}
	if err := startWhenFull(next); err != nil {
		return err
	}
	next.Sequence++
	return h.commit(ctx, r, next)
}

func clone(m *Match) *Match {
	next := *m
	next.Options.CPUSeats = append([]int(nil), m.Options.CPUSeats...)
	next.Seats = append([]Seat{}, m.Seats...)
	next.Chat = append([]Chat{}, m.Chat...)
	next.Signals = append([]Signal{}, m.Signals...)
	next.Events = append([]GameEvent{}, m.Events...)
	next.CantoHistory = append([]GameEvent{}, m.CantoHistory...)
	next.Recent = make(map[string]uint64, len(m.Recent))
	for key, value := range m.Recent {
		next.Recent[key] = value
	}
	return &next
}

func (h *Host) commit(ctx context.Context, r *room, next *Match) error {
	previous := r.match
	next.LastActivity = time.Now()
	if err := h.persist(ctx, next, false, addedHumanMembers(previous, next)); err != nil {
		return err
	}
	r.match = next
	if previous.State != nil && next.State != nil && previous.State.Revision != next.State.Revision {
		r.recovering = false
	}
	r.updateRoomDeadline(next)
	r.selectCPUController(next)
	r.notify()
	if previous == nil || occupiedSeats(previous) != occupiedSeats(next) ||
		(previous.State == nil) != (next.State == nil) ||
		(next.State != nil && next.State.Phase == rules.MatchOver) {
		h.mu.Lock()
		if activeMatch(previous) && !activeMatch(next) {
			h.active--
		}
		h.updateListing(next)
		h.notifyListings()
		h.mu.Unlock()
	}
	return nil
}

// Caller holds r.mu. Notifications coalesce without blocking a game.
func (r *room) notify() {
	for channel := range r.listeners {
		select {
		case channel <- struct{}{}:
		default:
		}
	}
}

func seatOf(m *Match, userID string) int {
	if userID == "" {
		return -1
	}
	for seat, user := range m.Seats {
		if user.ID == userID {
			return seat
		}
	}
	return -1
}

func (h *Host) Snapshot(code, userID string, perspective int) (Snapshot, error) {
	r := h.lockRoom(code)
	if r == nil {
		return Snapshot{}, fmt.Errorf("no encontramos esa mesa")
	}
	defer r.mu.Unlock()
	m := r.match
	seat := seatOf(m, userID)
	if seat >= 0 {
		perspective = seat
	}
	if perspective < 0 || perspective >= len(m.Seats) {
		perspective = 0
	}
	v := Snapshot{
		Updating:      h.Updating(),
		CPUController: r.controllerSeat(m),
		CPUPaused:     len(m.Options.CPUSeats) > 0 && m.State != nil && m.State.Phase != rules.MatchOver && r.controller == "",
		Options:       m.Options,
		Code:          code,
		Private:       m.Private,
		Config:        m.Config,
		Seats:         append([]Seat{}, m.Seats...),
		Seat:          seat,
		Perspective:   perspective,
		Deadline:      m.Deadline,
		ServerTime:    time.Now(),
		Chat:          append([]Chat{}, m.Chat...),
		Signals:       []Signal{},
		Sequence:      m.Sequence,
		Events:        append([]GameEvent{}, m.Events...),
		CantoHistory:  append([]GameEvent{}, m.CantoHistory...),
	}
	v.Options.CPUSeats = append([]int(nil), m.Options.CPUSeats...)
	if m.State != nil {
		view := rules.ViewFor(*m.State, perspective, seat < 0)
		if v.Updating {
			view.Legal = []rules.Command{}
		}
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
	CPUPlayers   int    `json:"cpuPlayers"`
	Code         string `json:"code"`
	Players      int    `json:"players"`
	Capacity     int    `json:"capacity"`
	Target       int    `json:"target"`
	FlorRequired bool   `json:"florRequired"`
	Started      bool   `json:"started"`
	Host         string `json:"host"`
}

func (h *Host) List() []Listing {
	h.mu.RLock()
	defer h.mu.RUnlock()
	list := make([]Listing, 0, len(h.listings))
	for _, listing := range h.listings {
		list = append(list, listing)
	}
	return list
}

type Input struct {
	CPUSeat  *int          `json:"cpuSeat,omitempty"`
	ID       string        `json:"id"`
	Revision uint64        `json:"revision"`
	Command  rules.Command `json:"command"`
	Chat     string        `json:"chat,omitempty"`
	Signal   string        `json:"signal,omitempty"`
}

func (h *Host) Handle(ctx context.Context, code string, user store.User, input Input) error {
	return h.HandleFromConnection(ctx, code, user, input, "")
}

// HandleFromConnection binds delegated CPU actions to the server-issued socket
// identity. A browser cannot claim authority by sending another user's ID.
func (h *Host) HandleFromConnection(ctx context.Context, code string, user store.User, input Input, connectionID string) error {
	if input.ID == "" || len(input.ID) > 80 {
		return fmt.Errorf("identificador de acción inválido")
	}
	r := h.lockRoom(code)
	if r == nil {
		return fmt.Errorf("no encontramos esa mesa")
	}
	defer r.mu.Unlock()
	m := r.match
	seat := seatOf(m, user.ID)
	if h.Updating() && input.Chat == "" && input.Signal == "" && input.Command.Type != "leave" {
		return fmt.Errorf("la partida está en pausa mientras actualizamos el juego")
	}
	if input.CPUSeat != nil {
		cpuSeat := *input.CPUSeat
		connection := r.connections[connectionID]
		if connectionID == "" || r.controller != connectionID || connection.userID != user.ID ||
			cpuSeat < 0 || cpuSeat >= len(m.Seats) || !m.Seats[cpuSeat].CPU ||
			input.Chat != "" || input.Signal != "" || input.Command.Type == "leave" {
			return fmt.Errorf("no tienes el control de ese jugador CPU")
		}
		seat = cpuSeat
	}
	if len(m.Options.CPUSeats) > 0 && r.controller == "" && input.Chat == "" && input.Signal == "" && input.Command.Type != "leave" {
		return fmt.Errorf("esperando a un jugador para continuar con la CPU")
	}
	key := user.ID + ":" + input.ID
	if _, exists := m.Recent[key]; exists {
		return nil
	}
	if !h.Updating() && input.Chat == "" && input.Signal == "" && m.State != nil && !m.Deadline.IsZero() && !time.Now().Before(m.Deadline) {
		if err := h.expire(ctx, r, m); err != nil {
			return err
		}
		m = r.match
	}
	next := clone(m)
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
			next.Seats[seat] = Seat{}
			if !r.hasConnectedPlayer(next) {
				return h.closeVacantRoom(ctx, r, next)
			}
		} else {
			if input.Revision != m.State.Revision {
				return fmt.Errorf("la mesa cambió. Revisa el estado actualizado")
			}
			if input.CPUSeat != nil && cpuDefersCanto(m, seat) {
				return fmt.Errorf("el canto debe responderlo un jugador humano del equipo")
			}
			state, events, err := rules.Apply(*m.State, seat, input.Command)
			if err != nil {
				return fmt.Errorf("esa jugada no está disponible ahora")
			}
			next.State = &state
			next.recordEvents(events)
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
	return h.commit(ctx, r, next)
}

func validSignal(signal string) bool {
	return signal == "perico" || signal == "perica" || signal == "seven_spades" || signal == "seven_coins" || signal == "receta"
}

func (m *Match) recordEvents(events []rules.Event) {
	// One command can announce both teams' privo values. Give every event a
	// distinct stable ID, including across hands that clear the recent log.
	nextID := m.Sequence + 1
	for _, previous := range [][]GameEvent{m.Events, m.CantoHistory} {
		if len(previous) > 0 {
			nextID = max(nextID, previous[len(previous)-1].ID+1)
		}
	}
	for _, event := range events {
		record := GameEvent{
			ID:    nextID,
			Hand:  m.State.HandNumber,
			Event: event,
		}
		nextID++
		m.Events = append(m.Events, record)
		if isCanto(event) {
			m.CantoHistory = append(m.CantoHistory, record)
		}
	}
	if len(m.Events) > 24 {
		m.Events = m.Events[len(m.Events)-24:]
	}
}

func isCanto(event rules.Event) bool {
	return event.Type == "call" || event.Type == "respond" || event.Type == "privo" || event.Text != ""
}

func (h *Host) Subscribe(code string) (<-chan struct{}, func()) {
	channel := make(chan struct{}, 1)
	r := h.lockRoom(code)
	if r == nil {
		channel <- struct{}{}
		return channel, func() {}
	}
	r.listeners[channel] = true
	r.mu.Unlock()
	return channel, func() {
		r.mu.Lock()
		defer r.mu.Unlock()
		delete(r.listeners, channel)
	}
}

func (h *Host) Run(ctx context.Context) {
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	var workers sync.WaitGroup
	defer workers.Wait()
	for {
		select {
		case <-ctx.Done():
			return
		case now := <-ticker.C:
			if h.Updating() {
				continue
			}
			h.mu.RLock()
			rooms := make([]*room, 0, len(h.rooms))
			for _, r := range h.rooms {
				rooms = append(rooms, r)
			}
			h.mu.RUnlock()
			for _, r := range rooms {
				// Busy rooms are retried next tick. Due rooms retain their
				// lock in the worker, bounding timer work to one per room.
				if !r.mu.TryLock() {
					continue
				}
				m := r.match
				vacant := m != nil && !r.deadline.IsZero() && !now.Before(r.deadline)
				timedOut := m != nil && m.State != nil && !m.Deadline.IsZero() && !now.Before(m.Deadline) &&
					(len(m.Options.CPUSeats) == 0 || r.controller != "")
				if !vacant && !timedOut {
					r.mu.Unlock()
					continue
				}
				workers.Go(func() {
					defer r.mu.Unlock()
					if ctx.Err() != nil || h.Updating() {
						return
					}
					if vacant {
						if err := h.closeVacantRoom(ctx, r, m); err != nil {
							h.log.Error("close vacant room", "code", m.Code, "error", err)
						}
					} else if err := h.expire(ctx, r, m); err != nil && !errors.Is(err, context.Canceled) {
						h.log.Error("match timeout failed", "code", m.Code, "error", err)
					}
				})
			}
		}
	}
}

func (h *Host) expire(ctx context.Context, r *room, m *Match) error {
	next := clone(m)
	var state rules.State
	var events []rules.Event
	var err error
	if m.State.Phase == rules.HandOver {
		deck, drawErr := shuffledDeck()
		if drawErr != nil {
			return drawErr
		}
		state, err = rules.StartHand(*m.State, deck)
		next.Signals = nil
		next.Events = nil
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
		state, events, err = rules.Timeout(*m.State, choice)
	}
	if err != nil {
		return err
	}
	next.State = &state
	next.recordEvents(events)
	next.Sequence++
	next.Deadline = time.Now().Add(30 * time.Second)
	if state.Phase == rules.HandOver {
		next.Deadline = time.Now().Add(5 * time.Second)
	}
	if state.Phase == rules.MatchOver {
		next.Deadline = time.Time{}
	}
	return h.commit(ctx, r, next)
}
