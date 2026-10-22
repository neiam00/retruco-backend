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
	h.mu.Lock()
	match := h.matches[code]
	if match == nil {
		h.mu.Unlock()
		return func() {}
	}
	if h.connections[code] == nil {
		h.connections[code] = make(map[string]playerConnection)
	}
	h.connectionOrder++
	h.connections[code][connectionID] = playerConnection{userID: userID, order: h.connectionOrder}
	h.updateRoomDeadline(match)
	h.selectCPUController(match)
	h.notify(code)
	h.mu.Unlock()
	return func() {
		h.mu.Lock()
		defer h.mu.Unlock()
		delete(h.connections[code], connectionID)
		if len(h.connections[code]) == 0 {
			delete(h.connections, code)
		}
		if match := h.matches[code]; match != nil {
			h.updateRoomDeadline(match)
			h.selectCPUController(match)
			h.notify(code)
		}
	}
}

// Caller holds h.mu. Keep the current controller while connected; otherwise
// prefer the creator, then the oldest remaining human connection.
func (h *Host) selectCPUController(match *Match) {
	if len(match.Options.CPUSeats) == 0 || (match.State != nil && match.State.Phase == rules.MatchOver) {
		delete(h.controllers, match.Code)
		return
	}
	eligible := func(connection playerConnection) bool {
		seat := seatOf(match, connection.userID)
		return seat >= 0 && !match.Seats[seat].CPU
	}
	previous := h.controllers[match.Code]
	room := h.connections[match.Code]
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
		delete(h.controllers, match.Code)
	} else {
		h.controllers[match.Code] = chosenID
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

func (h *Host) controllerSeat(match *Match) int {
	id := h.controllers[match.Code]
	if id == "" {
		return -1
	}
	return seatOf(match, h.connections[match.Code][id].userID)
}

func (h *Host) CPUWork(code, userID, connectionID string) CPUWork {
	h.mu.Lock()
	defer h.mu.Unlock()
	match := h.matches[code]
	connection := h.connections[code][connectionID]
	if match == nil || connectionID == "" || h.controllers[code] != connectionID || connection.userID != userID {
		return CPUWork{}
	}
	work := CPUWork{Version: cpu.ProtocolVersion, Controlled: true, Seat: -1}
	if match.State == nil {
		return work
	}
	for seat, player := range match.Seats {
		if !player.CPU {
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
