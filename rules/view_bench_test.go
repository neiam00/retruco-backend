package rules_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/rules"
)

func BenchmarkLegalActions(b *testing.B) {
	state, err := rules.New(rules.Config{Players: 4, Target: 24}, 0, rules.Deck())
	if err != nil {
		b.Fatal(err)
	}
	state, _, err = rules.Apply(state, state.Turn, rules.Command{Type: "cut", Cut: 1})
	if err != nil {
		b.Fatal(err)
	}
	truco, _, err := rules.Apply(state, state.Turn, rules.Command{Type: "call", Kind: "truco"})
	if err != nil {
		b.Fatal(err)
	}
	for _, tc := range []struct {
		name  string
		state rules.State
		seat  int
	}{
		{"turn", state, state.Turn},
		{"waiting", state, (state.Turn + 1) % 4},
		{"truco_response", truco, (state.Turn + 1) % 4},
		{"truco_caller", truco, state.Turn},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				rules.LegalActions(tc.state, tc.seat)
			}
		})
	}
}
