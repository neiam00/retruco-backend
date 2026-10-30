package api

import (
	"encoding/json"
	"github.com/neiam00/retruco-backend/internal/game"
	"time"
)

// Frozen pre-optimization encoder provides an independent wire-compatibility
// oracle. Only tests use this path; production never reparses cached JSON.
func referenceStreamMessage(s *gameStream, p *game.Publication, cpu game.CPUWork, baseline bool, now time.Time) (streamMessage, error) {
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
