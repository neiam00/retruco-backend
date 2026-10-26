package game

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/neiam00/retruco-backend/rules"
)

// Compare the actual pre-change JSON path against the partitioned encoder with
// a full retained chat window and a long match's canto history. Database latency
// is benchmarked separately in internal/store.
func BenchmarkPersistEncoding(b *testing.B) {
	state, err := rules.New(rules.Config{Players: 4, Target: 24}, 0, rules.Deck())
	if err != nil {
		b.Fatal(err)
	}
	match := &Match{Code: "benchmark", Config: state.Config, State: &state, Recent: make(map[string]uint64), Chat: []Chat{}, CantoHistory: []GameEvent{}}
	for i := range 100 {
		match.Chat = append(match.Chat, Chat{ID: uint64(i + 1), Name: "Player", Text: strings.Repeat("message ", 50)})
	}
	for i := range 300 {
		match.CantoHistory = append(match.CantoHistory, GameEvent{ID: uint64(i + 101), Hand: i / 6, Event: rules.Event{Type: "call", Kind: "truco", Text: "Player called truco"}})
	}
	for i := range 257 {
		match.Recent[fmt.Sprintf("player:request-%d", i)] = uint64(i + 1)
	}
	next := clone(match)
	next.Sequence++
	for _, variant := range []string{"full_snapshot", "compact_unchanged_history"} {
		b.Run(variant, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if variant == "full_snapshot" {
					payload, err := json.Marshal(next)
					if err != nil {
						b.Fatal(err)
					}
					b.ReportMetric(float64(len(payload)), "payload-B/op")
				} else {
					write, err := encodeMatchWrite(next, match)
					if err != nil {
						b.Fatal(err)
					}
					b.ReportMetric(float64(len(write.Payload)), "payload-B/op")
				}
			}
		})
	}
}
