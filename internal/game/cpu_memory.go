package game

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/neiam00/retruco-backend/cpu/learning"
	"github.com/neiam00/retruco-backend/rules"
)

// Cursors are connection-local acknowledgements, never model state or durable
// authorization. A new controller or socket starts with a full authorized replay.
type cpuMemoryCursor struct{ Game, Hand, Cursor int }

func (m *Match) cpuContext(seat int) learning.Context {
	context := learning.Context{CPU: cpuRoles(m), Complete: m.CPUHistoryComplete,
		SignalCounts: m.CPUCommunication.CountsFor(seat), SignalsComplete: m.CPUCommunication.Complete}
	for _, c := range m.CantoHistory {
		context.Cantos = append(context.Cantos, learning.Canto{Hand: c.Hand, Event: c.Event})
	}
	if m.CPUMemoryVersion == learning.MemoryVersion {
		context.Signals = m.CPUVisibleSignals[seat]
	} else {
		for _, s := range m.Signals {
			if s.Visible[seat] {
				context.Signals = append(context.Signals, learning.Signal{Seat: s.Seat, Kind: s.Kind})
			}
		}
	}
	return context
}

func (m *Match) restoreCPUMemory() {
	if m.State == nil {
		return
	}
	if m.CPUMemoryVersion == learning.MemoryVersion && m.CPUMemory.Hand == m.State.HandNumber {
		for seat := 0; seat < m.Config.Players; seat++ {
			rows := m.CPUMemory.Seats[seat]
			if len(rows) == 0 || len(rows) > learning.MaxMemoryEvents {
				m.CPUMemory.Complete[seat] = false
			} else if rows[len(rows)-1] != learning.Encode(rules.ViewFor(*m.State, seat, true), seat, m.cpuContext(seat)) {
				// A partial/old snapshot must not silently become a complete
				// replay. The next hand can establish a fresh complete journal.
				m.CPUMemory.Complete[seat] = false
			}
		}
		return
	}
	// Old snapshots cannot recreate information as it was observed earlier in
	// this hand. Mark the gap explicitly instead of replaying today's view as history.
	m.CPUMemoryVersion = learning.MemoryVersion
	m.CPUMemory = learning.MemoryJournal{Hand: m.State.HandNumber}
	m.CPUVisibleSignals = [4][]learning.Signal{}
	for _, s := range m.Signals {
		m.recordCPUSignal(s)
	}
}

func (m *Match) captureCPUMemory(ownNone int) {
	if m.State == nil || len(m.Options.CPUSeats) == 0 {
		return
	}
	if m.CPUMemoryVersion != learning.MemoryVersion {
		m.restoreCPUMemory()
	}
	if m.CPUMemory.Hand != m.State.HandNumber {
		m.CPUMemory.StartHand(m.State.HandNumber)
		m.CPUVisibleSignals = [4][]learning.Signal{}
	}
	for seat := 0; seat < m.Config.Players; seat++ {
		// Matches the training journal: observation-time authorized features,
		// including off-turn changes, with no legal-action enumeration or critic data.
		view := rules.ViewFor(*m.State, seat, true)
		m.CPUMemory.Append(seat, learning.Encode(view, seat, m.cpuContext(seat)), seat == ownNone)
	}
}

func (m *Match) recordCPUSignal(s Signal) {
	for seat, visible := range s.Visible {
		if !visible {
			continue
		}
		rows := append(m.CPUVisibleSignals[seat], learning.Signal{Seat: s.Seat, Kind: s.Kind})
		if len(rows) > 20 {
			rows = append([]learning.Signal(nil), rows[len(rows)-20:]...)
		}
		m.CPUVisibleSignals[seat] = rows
	}
}

func (r *room) attachCPUMemory(m *Match, work *CPUWork) {
	connection := r.connections[r.controller]
	if connection.memoryVersion != learning.MemoryVersion || work.Seat < 0 {
		return
	}
	cursor := connection.memoryCursors[work.Seat]
	if cursor.Game != m.GameNumber || cursor.Hand != m.CPUMemory.Hand {
		cursor = cpuMemoryCursor{}
	}
	page, err := m.CPUMemory.Page(work.Seat, cursor.Hand, cursor.Cursor)
	if err != nil {
		page, err = m.CPUMemory.Page(work.Seat, 0, 0)
	}
	if err != nil {
		return
	}
	page.Complete = page.Complete && m.CPUHistoryComplete
	digest := sha256.Sum256([]byte(fmt.Sprintf("cpu-memory-v1/%s/%d/%d/%s/%d", m.Code, m.GameNumber, work.Seat, r.controller, connection.order)))
	work.MemoryKey = hex.EncodeToString(digest[:])
	work.Memory = &page
}

func (r *room) advanceCPUMemory(userID, connectionID string, input Input) error {
	connection, ok := r.connections[connectionID]
	if !ok || connectionID == "" || r.controller != connectionID || connection.userID != userID ||
		connection.cpuVersion != 2 || connection.memoryVersion != learning.MemoryVersion || input.CPUSeat == nil ||
		input.MemoryHand == nil || input.MemoryCursor == nil || input.Signal != "" || input.Chat != "" || input.Command != (rules.Command{}) {
		return fmt.Errorf("solicitud de memoria CPU no autorizada")
	}
	work := r.cpuWorkV2(r.match)
	if work.View == nil || work.Memory == nil || !work.Memory.Complete || work.Seat != *input.CPUSeat ||
		work.Token != input.CPUToken || *input.MemoryHand != work.Memory.Hand ||
		(*input.MemoryCursor != 0 && *input.MemoryCursor != work.Memory.End) {
		return fmt.Errorf("la memoria CPU quedó desactualizada")
	}
	connection.memoryCursors[work.Seat] = cpuMemoryCursor{Game: r.match.GameNumber, Hand: work.Memory.Hand, Cursor: *input.MemoryCursor}
	r.connections[connectionID] = connection
	// Memory paging neither consumes a decision nor renews the turn deadline.
	r.notify()
	return nil
}
