// Package learning contains the transport-independent learning contract.
// Observation encoding accepts authorized views, never rules.State.
package learning

import "github.com/neiam00/retruco-backend/rules"

const Version = 2
const StateSize = 768
const ActionSize = 48

var SignalKinds = [...]string{"none", "perico", "perica", "seven_spades", "seven_coins", "receta"}

type Canto struct {
	Hand  int         `json:"hand"`
	Event rules.Event `json:"event"`
}

// Signal contains only a delivered claim. Visibility masks belong to the host.
type Signal struct {
	Seat int    `json:"seat"`
	Kind string `json:"kind"`
}

type Context struct {
	Cantos          []Canto      `json:"cantos"`
	Signals         []Signal     `json:"signals"`
	CPU             [4]bool      `json:"cpu"`
	Complete        bool         `json:"complete"`
	SignalCounts    [4][5]uint32 `json:"signalCounts"`
	SignalsComplete bool         `json:"signalsComplete"`
}

// Communication is private host state. Never send it to an actor.
type Communication struct {
	Hand     int
	Consumed [4][3]bool
	Signals  []DeliveredSignal
	// Counters are private per observer: unseen signals never affect its inputs.
	Counts   [4][4][5]uint32
	Complete bool
}

type DeliveredSignal struct {
	Signal  Signal
	Visible [4]bool
}

func (c *Communication) StartHand(hand int) {
	if c.Hand != hand {
		c.Hand = hand
		c.Consumed = [4][3]bool{}
		c.Signals = nil
	}
}

func (c *Communication) ResetMatch() { *c = Communication{Complete: true} }

func (c Communication) CountsFor(seat int) [4][5]uint32 {
	if seat < 0 || seat >= 4 {
		return [4][5]uint32{}
	}
	return c.Counts[seat]
}

// Record updates only observers who actually received this claim. The visibility
// draw is made once by the caller, and never inferred from whether a card shows.
func (c *Communication) Record(signal Signal, visible [4]bool) bool {
	if signal.Seat < 0 || signal.Seat >= 4 {
		return false
	}
	kind := -1
	for i, name := range SignalKinds[1:] {
		if name == signal.Kind {
			kind = i
		}
	}
	if kind < 0 {
		return false
	}
	c.Signals = append(c.Signals, DeliveredSignal{Signal: signal, Visible: visible})
	if len(c.Signals) > 20 {
		c.Signals = append([]DeliveredSignal(nil), c.Signals[len(c.Signals)-20:]...)
	}
	for observer, received := range visible {
		if received && c.Counts[observer][signal.Seat][kind] < ^uint32(0) {
			c.Counts[observer][signal.Seat][kind]++
		}
	}
	return true
}

func (c Communication) For(seat int) []Signal {
	result := []Signal{}
	if seat < 0 || seat >= 4 {
		return result
	}
	for _, s := range c.Signals {
		if s.Visible[seat] {
			result = append(result, s.Signal)
		}
	}
	return result
}

// Offer returns the next bounded signal opportunity, or -1. The caller must
// revalidate phase, hand, round and its work token before consuming it.
func (c Communication) Offer(v rules.View, cpu [4]bool) int {
	if v.Config.Players != 4 || v.Phase != rules.Playing || v.Pending != nil || v.Round < 0 || v.Round > 2 || c.Hand != v.HandNumber {
		return -1
	}
	for seat := 0; seat < v.Config.Players; seat++ {
		if cpu[seat] && !c.Consumed[seat][v.Round] {
			return seat
		}
	}
	return -1
}

// Consume validates before changing state. Chance draws are injected by host:
// secure in production, seeded in simulation. None needs no chance draws.
func (c *Communication) Consume(seat, round, kind, players int, noticed [4]bool) bool {
	if players != 4 || seat < 0 || seat >= players || round < 0 || round > 2 || kind < 0 || kind >= len(SignalKinds) || c.Consumed[seat][round] {
		return false
	}
	c.Consumed[seat][round] = true
	if kind == 0 {
		return true
	}
	s := DeliveredSignal{Signal: Signal{Seat: seat, Kind: SignalKinds[kind]}}
	for other := 0; other < players; other++ {
		s.Visible[other] = other%2 == seat%2 || noticed[other]
	}
	return c.Record(s.Signal, s.Visible)
}

// ActingSeat matches production's ascending-seat all-CPU canto convention.
// Human-proxy seats retain response authority when their teammate is a CPU.
func ActingSeat(s rules.State, cpu [4]bool) int {
	if s.Pending == nil {
		return s.Turn
	}
	team := 1 - s.Pending.Caller
	for seat := 0; seat < s.Config.Players; seat++ {
		if seat%2 == team && !cpu[seat] && len(rules.LegalActions(s, seat)) > 0 {
			return seat
		}
	}
	for seat := 0; seat < s.Config.Players; seat++ {
		if seat%2 == team && len(rules.LegalActions(s, seat)) > 0 {
			return seat
		}
	}
	return -1
}
