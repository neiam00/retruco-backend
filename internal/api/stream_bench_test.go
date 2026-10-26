package api

import (
	"encoding/json"
	"fmt"
	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/rules"
	"testing"
	"time"
)

// Compare repeated full snapshots with unchanged-history updates using the same
// real four-player rules view. This measures encoding and bytes, not network I/O.
func BenchmarkStreamHistory(b *testing.B) {
	state, err := rules.New(rules.Config{Players: 4, Target: 24}, 0, rules.Deck())
	if err != nil {
		b.Fatal(err)
	}
	state, _, err = rules.Apply(state, state.Turn, rules.Command{Type: "cut", Cut: 1})
	if err != nil {
		b.Fatal(err)
	}
	view := rules.ViewFor(state, state.Turn, false)
	data, err := json.Marshal(map[string]any{
		"code": "benchmark", "sequence": 100, "signals": []game.Signal{}, "state": view,
		"config": state.Config, "seats": []game.Seat{{ID: "a", Name: "Ana"}, {ID: "b", Name: "Beto"}, {ID: "c", Name: "Carlos"}, {ID: "d", Name: "Diana"}},
		"seat": state.Turn, "perspective": state.Turn, "private": false, "cpuController": -1, "cpuPaused": false,
		"options": game.RoomOptions{}, "deadline": time.Unix(100, 0),
	})
	if err != nil {
		b.Fatal(err)
	}
	p := &game.Publication{Data: data,
		Chat: []game.Chat{}, Events: []game.GameEvent{}, CantoHistory: []game.GameEvent{}}
	for i := range 100 {
		p.Chat = append(p.Chat, game.Chat{ID: uint64(i + 1), Name: "Player", Text: fmt.Sprintf("Retained chat message %d that must not be retransmitted with every card", i)})
		p.CantoHistory = append(p.CantoHistory, game.GameEvent{ID: uint64(i + 1), Hand: i / 10})
	}
	for _, baseline := range []bool{true, false} {
		b.Run(fmt.Sprintf("full_snapshot_%t", baseline), func(b *testing.B) {
			stream := gameStream{epoch: "bench", previous: p}
			b.ReportAllocs()
			var size int
			for b.Loop() {
				message, err := stream.message(p, game.CPUWork{}, baseline, time.Unix(1, 0))
				if err != nil {
					b.Fatal(err)
				}
				encoded, err := json.Marshal(message)
				if err != nil {
					b.Fatal(err)
				}
				size = len(encoded)
			}
			b.ReportMetric(float64(size), "wire-B/op")
		})
	}
}
