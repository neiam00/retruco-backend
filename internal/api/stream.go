package api

import (
	"encoding/json"
	"time"

	"github.com/neiam00/retruco-backend/internal/game"
)

type streamPosition struct {
	Epoch        string `json:"epoch"`
	Sequence     uint64 `json:"sequence"`
	BaseSequence uint64 `json:"baseSequence,omitempty"`
}

type historyDelta[T any] struct {
	Reset bool `json:"reset"`
	Items []T  `json:"items"`
	Keep  int  `json:"keep"`
}

type streamHistory struct {
	Chat         *historyDelta[game.Chat]      `json:"chat,omitempty"`
	Events       *historyDelta[game.GameEvent] `json:"events,omitempty"`
	CantoHistory *historyDelta[game.GameEvent] `json:"cantoHistory,omitempty"`
}

// A stream's sequence counts sent publications, not game moves. Chat, control,
// perspective and deployment changes all have a position. Each socket starts
// a fresh epoch and baseline; it never resumes from an unverified old state.
type gameStream struct {
	epoch    string
	sequence uint64
	previous *game.Publication
}

// encode appends the already encoded, immutable publication without asking
// encoding/json to scan and reformat its RawMessage for every recipient. Only
// connection-specific fields and changed history need JSON encoding here.
// Publication.Data is a complete object produced by game.Host.Publication's
// json.Marshal; it never contains unvalidated client bytes.
func (s *gameStream) encode(p *game.Publication, cpu game.CPUWork, baseline bool, now time.Time) ([]byte, error) {
	baseline = baseline || s.previous == nil
	position := streamPosition{Epoch: s.epoch, Sequence: s.sequence + 1}
	fields := struct {
		ServerTime   time.Time         `json:"serverTime"`
		Chat         *[]game.Chat      `json:"chat,omitempty"`
		Events       *[]game.GameEvent `json:"events,omitempty"`
		CantoHistory *[]game.GameEvent `json:"cantoHistory,omitempty"`
	}{ServerTime: now}
	var history *streamHistory
	prefix := `{"type":"snapshot","data":`
	if baseline {
		fields.Chat, fields.Events, fields.CantoHistory = &p.Chat, &p.Events, &p.CantoHistory
	} else {
		prefix = `{"type":"game_update","data":`
		position.BaseSequence = s.sequence
		history = &streamHistory{
			Chat:         historyChange(s.previous.Chat, p.Chat, func(v game.Chat) uint64 { return v.ID }),
			Events:       historyChange(s.previous.Events, p.Events, func(v game.GameEvent) uint64 { return v.ID }),
			CantoHistory: historyChange(s.previous.CantoHistory, p.CantoHistory, func(v game.GameEvent) uint64 { return v.ID }),
		}
	}
	dataFields, err := json.Marshal(fields)
	if err != nil {
		return nil, err
	}
	tail, err := json.Marshal(struct {
		CPU     game.CPUWork   `json:"cpu"`
		Stream  streamPosition `json:"stream"`
		History *streamHistory `json:"history,omitempty"`
	}{CPU: cpu, Stream: position, History: history})
	if err != nil {
		return nil, err
	}
	payload := make([]byte, 0, len(prefix)+len(p.Data)+len(dataFields)+len(tail))
	payload = append(payload, prefix...)
	payload = append(payload, p.Data[:len(p.Data)-1]...)
	if len(p.Data) > 2 {
		payload = append(payload, ',')
	}
	payload = append(payload, dataFields[1:]...)
	payload = append(payload, ',')
	payload = append(payload, tail[1:]...)
	s.sequence++
	s.previous = p
	return payload, nil
}

// History records are immutable and have stable IDs. Compare endpoints to skip
// unchanged histories in constant time; locate retained overlap only on append
// or truncation. Coalesced room notifications may include many new records.
func historyChange[T any](old, current []T, id func(T) uint64) *historyDelta[T] {
	if len(old) == len(current) && (len(old) == 0 || (id(old[0]) == id(current[0]) && id(old[len(old)-1]) == id(current[len(current)-1]))) {
		return nil
	}
	if len(old) > 0 && len(current) > 0 {
		last := id(old[len(old)-1])
		for i, item := range current {
			if id(item) == last && i < len(old) && id(old[len(old)-i-1]) == id(current[0]) {
				return &historyDelta[T]{Items: current[i+1:], Keep: len(current)}
			}
		}
	}
	return &historyDelta[T]{Reset: true, Items: current, Keep: len(current)}
}
