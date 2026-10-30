package rules

import (
	"encoding/json"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/require"
)

// Freeze the pre-optimization catalogue: every candidate is tested through
// Apply. Unlike the optimized path it deliberately tries impossible raises,
// burns and stacks. This oracle protects the exact UI command list and order.
func referenceLegalActions(s State, seat int) []Command {
	legal := []Command{}
	if seat < 0 || seat >= s.Config.Players || s.Phase == MatchOver {
		return legal
	}
	if s.Phase == Cutting {
		if seat == s.Turn {
			for cut := 1; cut < 40; cut++ {
				legal = append(legal, Command{Type: "cut", Cut: cut})
			}
		}
		return legal
	}
	if s.Phase != Playing {
		return legal
	}
	// Skip whole families that cannot pass their turn/team guards. Apply remains
	// the authority for individual commands, without copying State and allocating
	// rejection errors for every impossible candidate in each player's snapshot.
	cardTurn := s.Pending == nil && s.Turn == seat
	responding := s.Pending != nil && s.Pending.Caller != s.Team(seat)
	canCall := cardTurn || responding && s.Pending.Kind == "truco"
	if !cardTurn && !responding {
		return legal
	}
	add := func(c Command) {
		if _, _, err := Apply(s, seat, c); err == nil {
			legal = append(legal, c)
		}
	}
	if s.Pending != nil && s.Pending.Kind == "privo" {
		for _, kind := range []string{"accept", "decline", "present_flor"} {
			add(Command{Type: "respond", Kind: kind})
		}
		return legal
	}
	if cardTurn {
		remaining := s.Remaining(seat)
		for _, declaration := range []string{"", "flor", "a_ley", "presento_flor", "flor_flor", "con_mi_flor"} {
			if s.MandatoryStack {
				add(Command{Type: "burn", Declaration: declaration})
			}
			for _, card := range remaining {
				add(Command{Type: "play", Card: card, Declaration: declaration})
				if !s.MandatoryStack {
					add(Command{Type: "burn", Card: card, Declaration: declaration})
				}
				for _, top := range remaining {
					if top != card {
						add(Command{Type: "stack", Card: card, Top: top, Declaration: declaration})
					}
				}
			}
		}
	}
	for _, declaration := range []string{"", "flor", "presento_flor", "flor_flor", "con_mi_flor"} {
		if canCall {
			for _, kind := range []string{"envido", "ley_envido", "flor"} {
				add(Command{Type: "call", Kind: kind, Declaration: declaration})
			}
			for _, mode := range []string{"con_ley", "sin_ley"} {
				add(Command{Type: "call", Kind: "ley_envido", Mode: mode, Declaration: declaration})
			}
			for _, amount := range []int{3, 6, 9, 100} {
				add(Command{Type: "call", Kind: "truco", Amount: amount, Declaration: declaration})
			}
		}
		if !responding {
			continue
		}
		for _, mode := range []string{"", "con_ley", "sin_ley"} {
			if mode != "" && (s.Pending.Kind != "ley_envido" || s.Pending.Mode != "") {
				continue
			}
			for _, kind := range []string{"accept", "decline", "flowers_no", "falta"} {
				if kind == "decline" && mode != "" {
					continue
				}
				add(Command{Type: "respond", Kind: kind, Mode: mode, Declaration: declaration})
			}
			for amount := 1; amount <= s.Config.Target; amount++ {
				add(Command{Type: "respond", Kind: "raise", Amount: amount, Mode: mode, Declaration: declaration})
			}
			add(Command{Type: "respond", Kind: "raise", Amount: 100, Mode: mode, Declaration: declaration})
		}
		add(Command{Type: "respond", Kind: "decline", Mode: "sin_ley", Declaration: declaration})
	}
	return legal
}

func TestLegalActionsMatchReferenceAcrossReachableStates(t *testing.T) {
	// #nosec G404 -- Seeded test inputs only; live deals use crypto/rand.
	random := rand.New(rand.NewPCG(2901, 1000))
	checked := 0
	for trial := range 8 {
		config := Config{Players: 2 + 2*(trial%2), Target: 12 + 12*((trial/2)%2), FlorRequired: trial >= 4}
		deck := Deck()
		var state State
		reset := func() {
			random.Shuffle(len(deck), func(i, j int) { deck[i], deck[j] = deck[j], deck[i] })
			var err error
			state, err = New(config, random.IntN(config.Players), deck)
			require.NoError(t, err)
			if trial%3 == 0 {
				state.Scores = [2]int{config.Target - 1, config.Target - 1}
			}
		}
		reset()
		for range 120 {
			before, err := json.Marshal(state)
			require.NoError(t, err)
			for seat := range config.Players {
				expected := referenceLegalActions(state, seat)
				require.Equal(t, expected, LegalActions(state, seat), "trial=%d revision=%d seat=%d phase=%s round=%d", trial, state.Revision, seat, state.Phase, state.Round)
				checked++
			}
			after, err := json.Marshal(state)
			require.NoError(t, err)
			require.Equal(t, string(before), string(after), "legal-action enumeration cannot mutate authoritative state")
			if state.Phase == MatchOver {
				reset()
				continue
			}
			if state.Phase == HandOver {
				random.Shuffle(len(deck), func(i, j int) { deck[i], deck[j] = deck[j], deck[i] })
				state, err = StartHand(state, deck)
				require.NoError(t, err)
				continue
			}
			seat := state.Turn
			if state.Pending != nil {
				seat = 1 - state.Pending.Caller
			}
			actions := referenceLegalActions(state, seat)
			require.NotEmpty(t, actions)
			preferred := make([]Command, 0)
			for _, action := range actions {
				if action.Type == "cut" || action.Type == "play" || action.Type == "stack" || action.Type == "burn" || action.Kind == "accept" {
					preferred = append(preferred, action)
				}
			}
			if len(preferred) > 0 && random.IntN(4) != 0 {
				actions = preferred
			}
			state, _, err = Apply(state, seat, actions[random.IntN(len(actions))])
			require.NoError(t, err)
		}
	}
	require.Equal(t, 2880, checked)
}
