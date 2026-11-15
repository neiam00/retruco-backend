package learning

import "fmt"

const MemoryVersion = 1
const MemoryPageSize = 32
const MaxMemoryEvents = 2048

// MemoryJournal is private durable state, never a cross-seat actor payload.
// It stores observations at observation time, before any later revelation.
type MemoryJournal struct {
	Hand     int
	Seats    [4][][StateSize]float32
	Complete [4]bool
}

type MemoryPage struct {
	Version  int                  `json:"version"`
	Hand     int                  `json:"hand"`
	Start    int                  `json:"start"`
	End      int                  `json:"end"`
	Total    int                  `json:"total"`
	Complete bool                 `json:"complete"`
	Events   [][StateSize]float32 `json:"events"`
}

func (j *MemoryJournal) StartHand(hand int) {
	if j.Hand != hand {
		*j = MemoryJournal{Hand: hand, Complete: [4]bool{true, true, true, true}}
	}
}

// Append ignores unchanged authorized views. Hidden traffic must not advance a
// seat's cursor. force is only for that seat's own confirmed silent decision.
func (j *MemoryJournal) Append(seat int, features [StateSize]float32, force bool) {
	if seat < 0 || seat >= 4 || !j.Complete[seat] {
		return
	}
	rows := j.Seats[seat]
	if !force && len(rows) > 0 && rows[len(rows)-1] == features {
		return
	}
	if len(rows) >= MaxMemoryEvents {
		j.Complete[seat] = false
		return
	}
	j.Seats[seat] = append(rows, features)
}

func (j MemoryJournal) Page(seat, hand, cursor int) (MemoryPage, error) {
	if seat < 0 || seat >= 4 {
		return MemoryPage{}, fmt.Errorf("invalid memory seat")
	}
	if hand != j.Hand {
		cursor = 0
	}
	rows := j.Seats[seat]
	if cursor < 0 || cursor > len(rows) {
		return MemoryPage{}, fmt.Errorf("invalid memory cursor")
	}
	end := min(cursor+MemoryPageSize, len(rows))
	return MemoryPage{Version: MemoryVersion, Hand: j.Hand, Start: cursor, End: end, Total: len(rows), Complete: j.Complete[seat], Events: append([][StateSize]float32{}, rows[cursor:end]...)}, nil
}

// Clone permits copy-on-write host transactions without aliasing persisted rows.
func (j MemoryJournal) Clone() MemoryJournal {
	for seat := range j.Seats {
		j.Seats[seat] = append([][StateSize]float32(nil), j.Seats[seat]...)
	}
	return j
}
