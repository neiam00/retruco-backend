package rules_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestALeyIsOnlyAvailableInTheFirstRound(t *testing.T) {
	state := fixture(t, false,
		[3]rules.Card{{rules.Spades, 1}, {rules.Clubs, 4}, {rules.Cups, 6}},
		[3]rules.Card{{rules.Coins, 4}, {rules.Cups, 3}, {rules.Coins, 5}},
	)
	state = apply(t, state, 1, rules.Command{Type: "play", Card: 0, Declaration: "a_ley"})
	assert.True(t, state.Ley)
	state = apply(t, state, 0, rules.Command{Type: "play", Card: 0})
	require.Equal(t, 1, state.Round)

	for state.Round <= 2 && state.Phase == rules.Playing {
		seat := state.Turn
		commands := []rules.Command{
			{Type: "play", Card: state.Round, Declaration: "a_ley"},
			{Type: "call", Kind: "truco", Declaration: "a_ley"},
		}
		if state.Round == 1 {
			commands = append(commands, rules.Command{Type: "stack", Card: 1, Top: 2, Declaration: "a_ley"})
		}
		for _, command := range commands {
			next, events, err := rules.Apply(state, seat, command)
			require.Error(t, err, "round %d: %s must reject A ley", state.Round+1, command.Type)
			assert.Equal(t, state, next)
			assert.Empty(t, events)
		}
		for _, action := range rules.LegalActions(state, seat) {
			assert.NotEqual(t, "a_ley", action.Declaration)
		}
		pending := apply(t, state, seat, rules.Command{Type: "call", Kind: "truco"})
		_, _, err := rules.Apply(pending, 1-seat, rules.Command{Type: "respond", Kind: "accept", Declaration: "a_ley"})
		require.Error(t, err)

		round := state.Round
		state = apply(t, state, seat, rules.Command{Type: "play", Card: round})
		state = apply(t, state, state.Turn, rules.Command{Type: "play", Card: round})
	}
	assert.Equal(t, rules.HandOver, state.Phase)
}

func TestLeyEnvidoRejectsFlorPrefix(t *testing.T) {
	state := simple(t)
	for _, declaration := range []string{"con_mi_flor", "con_flor"} {
		assertRejectedCanto(t, state, state.Turn, rules.Command{Type: "call", Kind: "ley_envido", Declaration: declaration})
	}
}
