package game

import (
	"fmt"
	"time"

	"github.com/neiam00/retruco-backend/cpu"
	"github.com/neiam00/retruco-backend/rules"
)

type playerConnection struct {
	userID string
	order  uint64
}

// CPUWork contains only the acting CPU's authorized view. It never contains
// other players' hidden cards or the deck, and is only sent to its controller.
type CPUWork struct {
	Version    int         `json:"version"`
	Controlled bool        `json:"controlled"`
	Seat       int         `json:"seat"`
	View       *rules.View `json:"view,omitempty"`
}

func validateCPUSeats(players int, seats []int) error {
	seen := make(map[int]bool, len(seats))
	for _, seat := range seats {
		if seat < 1 || seat >= players || seen[seat] {
			return fmt.Errorf("los puestos CPU deben ser distintos y dejar libre tu puesto")
		}
		seen[seat] = true
	}
	return nil
}

func occupiedSeats(match *Match) int {
	count := 0
	for _, seat := range match.Seats {
		if seat.ID != "" {
			count++
		}
	}
	return count
}

func startWhenFull(match *Match) error {
	if match.State != nil || occupiedSeats(match) != match.Config.Players {
		return nil
	}
	deck, err := shuffledDeck()
	if err != nil {
		return err
	}
	dealer, err := randomInt(match.Config.Players)
	if err != nil {
		return err
	}
	state, err := rules.New(match.Config, dealer, deck)
	if err != nil {
		return err
	}
	match.State = &state
	match.Deadline = time.Now().Add(30 * time.Second)
	return nil
}

// ConnectPlayer tracks game sockets independently of voice sockets and watchers.
// Only one connected human tab runs CPU decisions, even with several tabs open.
func (h *Host) ConnectPlayer(code, userID, connectionID string) func() {
	r := h.lockRoom(code)
	if r == nil {
		return func() {}
	}
	r.connectionOrder++
	r.connections[connectionID] = playerConnection{userID: userID, order: r.connectionOrder}
	r.updateRoomDeadline(r.match)
	r.selectCPUController(r.match)
	r.notify()
	r.mu.Unlock()
	return func() {
		r.mu.Lock()
		defer r.mu.Unlock()
		delete(r.connections, connectionID)
		if r.match != nil {
			r.updateRoomDeadline(r.match)
			r.selectCPUController(r.match)
			r.notify()
		}
	}
}

// Caller holds r.mu. Keep the current controller while connected; otherwise
// prefer the creator, then the oldest remaining human connection.
func (r *room) selectCPUController(match *Match) {
	if len(match.Options.CPUSeats) == 0 || (match.State != nil && match.State.Phase == rules.MatchOver) {
		r.controller = ""
		return
	}
	eligible := func(connection playerConnection) bool {
		seat := seatOf(match, connection.userID)
		return seat >= 0 && !match.Seats[seat].CPU
	}
	previous := r.controller
	room := r.connections
	if current, ok := room[previous]; ok && eligible(current) {
		return
	}
	chosenID := ""
	var chosen playerConnection
	for id, connection := range room {
		if !eligible(connection) {
			continue
		}
		creator := connection.userID == match.CreatorID
		chosenCreator := chosen.userID == match.CreatorID
		if chosenID == "" || (creator && !chosenCreator) || (creator == chosenCreator && connection.order < chosen.order) {
			chosenID, chosen = id, connection
		}
	}
	if chosenID == "" {
		r.controller = ""
	} else {
		r.controller = chosenID
		// A returning human gets a full turn after an unattended CPU room.
		if previous == "" && match.State != nil && match.State.Phase != rules.MatchOver {
			duration := 30 * time.Second
			if match.State.Phase == rules.HandOver {
				duration = 5 * time.Second
			}
			match.Deadline = time.Now().Add(duration)
		}
	}
}

func (r *room) controllerSeat(match *Match) int {
	id := r.controller
	if id == "" {
		return -1
	}
	return seatOf(match, r.connections[id].userID)
}

// A human teammate owns pending canto decisions, including counter-cantos.
// CPU-only teams must still answer so matches can continue.
func cpuDefersCanto(match *Match, seat int) bool {
	if match.State == nil || match.State.Pending == nil {
		return false
	}
	team := match.State.Team(seat)
	if match.State.Pending.Caller == team {
		return false
	}
	for other, player := range match.Seats {
		if player.ID != "" && !player.CPU && match.State.Team(other) == team {
			return true
		}
	}
	return false
}

func (h *Host) CPUWork(code, userID, connectionID string) CPUWork {
	r := h.lockRoom(code)
	if r == nil {
		return CPUWork{}
	}
	defer r.mu.Unlock()
	match := r.match
	connection := r.connections[connectionID]
	if connectionID == "" || r.controller != connectionID || connection.userID != userID {
		return CPUWork{}
	}
	work := CPUWork{Version: cpu.ProtocolVersion, Controlled: true, Seat: -1}
	if match.State == nil {
		return work
	}
	for seat, player := range match.Seats {
		if !player.CPU || cpuDefersCanto(match, seat) {
			continue
		}
		if match.State.Pending == nil && match.State.Turn != seat {
			continue
		}
		view := rules.ViewFor(*match.State, seat, false)
		if len(view.Legal) > 0 {
			work.Seat, work.View = seat, &view
			break
		}
	}
	return work
}
