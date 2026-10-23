package rules_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLegalActionsKeepTeammateResponsesAndTrucoInterruptions(t *testing.T) {
	state, err := rules.New(rules.Config{Players: 4, Target: 24}, 0, rules.Deck())
	require.NoError(t, err)
	state = apply(t, state, state.Turn, rules.Command{Type: "cut", Cut: 1})
	caller := state.Turn
	for seat := range 4 {
		if seat == caller {
			assert.Contains(t, rules.LegalActions(state, seat), rules.Command{Type: "play", Card: 0})
		} else {
			assert.Empty(t, rules.LegalActions(state, seat))
		}
	}
	state = apply(t, state, caller, rules.Command{Type: "call", Kind: "truco"})
	for seat := range 4 {
		actions := rules.LegalActions(state, seat)
		if state.Team(seat) == state.Team(caller) {
			assert.Empty(t, actions)
			continue
		}
		// Either opponent can answer or interrupt, even though neither owns the
		// current card turn. The pending truco must still block playing a card.
		assert.Contains(t, actions, rules.Command{Type: "respond", Kind: "accept"})
		assert.Contains(t, actions, rules.Command{Type: "respond", Kind: "raise", Amount: 100})
		assert.Contains(t, actions, rules.Command{Type: "call", Kind: "envido"})
		assert.Contains(t, actions, rules.Command{Type: "call", Kind: "ley_envido"})
		for _, command := range actions {
			assert.Contains(t, []string{"call", "respond"}, command.Type)
		}
	}
	interrupter := (caller + 1) % 4
	state = apply(t, state, interrupter, rules.Command{Type: "call", Kind: "envido"})
	for seat := range 4 {
		actions := rules.LegalActions(state, seat)
		if state.Team(seat) == state.Team(interrupter) {
			assert.Empty(t, actions)
			continue
		}
		require.NotEmpty(t, actions)
		assert.Contains(t, actions, rules.Command{Type: "respond", Kind: "decline"})
		for _, command := range actions {
			assert.Equal(t, "respond", command.Type)
		}
	}
}
