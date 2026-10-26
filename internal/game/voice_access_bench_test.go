package game

import (
	"fmt"
	"io"
	"log/slog"
	"strings"
	"testing"

	"github.com/neiam00/retruco-backend/rules"
)

// Compare the previous voice permission path with the metadata-only check,
// using a real playing state and retained histories that voice never needs.
func BenchmarkVoiceAccess(b *testing.B) {
	config := rules.Config{Players: 4, Target: 24}
	state, err := rules.New(config, 0, rules.Deck())
	if err != nil {
		b.Fatal(err)
	}
	state, _, err = rules.Apply(state, state.Turn, rules.Command{Type: "cut", Cut: 1})
	if err != nil {
		b.Fatal(err)
	}
	match := &Match{Code: "BENCH-VOICE", Config: config, State: &state}
	for index := range 4 {
		match.Seats = append(match.Seats, Seat{ID: fmt.Sprintf("player-%d", index), Name: "Jugador"})
	}
	for index := range 100 {
		match.Chat = append(match.Chat, Chat{ID: uint64(index + 1), Name: "Jugador", Text: strings.Repeat("a", 100)})
		match.CantoHistory = append(match.CantoHistory, GameEvent{ID: uint64(index + 1), Hand: index / 8, Event: rules.Event{Type: "call", Kind: "truco"}})
	}
	host := New(nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
	host.rooms[match.Code] = newRoom(match)
	userID := match.Seats[state.Turn].ID
	b.Run("full_snapshot", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			view, err := host.Snapshot(match.Code, userID, 0)
			if err != nil || view.Seat < 0 {
				b.Fatalf("voice authorization failed: %v", err)
			}
		}
	})
	b.Run("permission_only", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			canSpeak, err := host.VoiceAccess(match.Code, userID)
			if err != nil || !canSpeak {
				b.Fatalf("voice authorization failed: %v", err)
			}
		}
	})
}
