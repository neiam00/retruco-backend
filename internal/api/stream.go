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

type streamMessage struct {
	Type    string          `json:"type"`
	Data    json.RawMessage `json:"data"`
	CPU     game.CPUWork    `json:"cpu"`
	Stream  streamPosition  `json:"stream"`
	History *streamHistory  `json:"history,omitempty"`
}

// A stream's sequence counts sent publications, not game moves. Chat, control,
// perspective and deployment changes all have a position. Each socket starts
// a fresh epoch and baseline; it never resumes from an unverified old state.
type gameStream struct {
	epoch    string
	sequence uint64
	previous *game.Publication
}

func (s *gameStream) message(p *game.Publication, cpu game.CPUWork, baseline bool, now time.Time) (streamMessage, error) {
	baseline = baseline || s.previous == nil
	position := streamPosition{Epoch: s.epoch, Sequence: s.sequence + 1}
	message := streamMessage{Type: "snapshot", CPU: cpu, Stream: position}
	data := append([]byte(nil), p.Data[:len(p.Data)-1]...)
	appendField := func(name string, value any) error {
		encoded, err := json.Marshal(value)
		if err != nil {
			return err
		}
		data = append(data, ',')
		data = append(data, '"')
		data = append(data, name...)
		data = append(data, '"', ':')
		data = append(data, encoded...)
		return nil
	}
	if err := appendField("serverTime", now); err != nil {
		return streamMessage{}, err
	}
	if baseline {
		for _, field := range []struct {
			name  string
			value any
		}{{"chat", p.Chat}, {"events", p.Events}, {"cantoHistory", p.CantoHistory}} {
			if err := appendField(field.name, field.value); err != nil {
				return streamMessage{}, err
			}
		}
	} else {
		message.Type = "game_update"
		message.Stream.BaseSequence = s.sequence
		message.History = &streamHistory{
			Chat:         historyChange(s.previous.Chat, p.Chat, func(v game.Chat) uint64 { return v.ID }),
			Events:       historyChange(s.previous.Events, p.Events, func(v game.GameEvent) uint64 { return v.ID }),
			CantoHistory: historyChange(s.previous.CantoHistory, p.CantoHistory, func(v game.GameEvent) uint64 { return v.ID }),
		}
	}
	message.Data = append(data, '}')
	s.sequence++
	s.previous = p
	return message, nil
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
