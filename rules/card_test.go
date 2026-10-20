package rules_test

import (
	"strconv"
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestViraSubstitutions(t *testing.T) {
	for _, tc := range []struct{ vira, perico, perica int }{
		{4, 11, 10}, {10, 11, 12}, {11, 12, 10}, {12, 11, 10},
	} {
		t.Run(strconv.Itoa(tc.vira), func(t *testing.T) {
			vira := rules.Card{Suit: rules.Spades, Number: tc.vira}
			perico, perica := rules.Pieces(vira)
			assert.Equal(t, tc.perico, perico.Number)
			assert.Equal(t, tc.perica, perica.Number)
			assert.Greater(t, rules.Strength(perico, vira), rules.Strength(perica, vira))
			assert.Greater(t, rules.Strength(perica, vira), rules.Strength(rules.Card{Suit: rules.Spades, Number: 1}, vira))
			assert.Equal(t, 10, rules.EnvidoCardValue(perico, vira))
			assert.Equal(t, 9, rules.EnvidoCardValue(perica, vira))
		})
	}
}

func TestEvaluateHolding(t *testing.T) {
	vira := rules.Card{Suit: rules.Spades, Number: 4}
	for _, tc := range []struct {
		name  string
		cards [3]rules.Card
		want  rules.Holding
	}{
		{"highest pair rather than three-card sum", [3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}}, rules.Holding{Envido: 33, Flor: 38}},
		{"no pair is legal", [3]rules.Card{{rules.Clubs, 7}, {rules.Cups, 6}, {rules.Coins, 5}}, rules.Holding{Envido: 7}},
		{"ordinary faces count zero", [3]rules.Card{{rules.Clubs, 12}, {rules.Clubs, 11}, {rules.Cups, 10}}, rules.Holding{Envido: 20}},
		{"perico is wildcard", [3]rules.Card{{rules.Spades, 11}, {rules.Clubs, 7}, {rules.Clubs, 6}}, rules.Holding{Envido: 37, Flor: 43}},
		{"wildcard chooses strongest pair", [3]rules.Card{{rules.Spades, 10}, {rules.Clubs, 4}, {rules.Cups, 7}}, rules.Holding{Envido: 36}},
		{"receta", [3]rules.Card{{rules.Spades, 11}, {rules.Spades, 10}, {rules.Coins, 7}}, rules.Holding{Envido: 39, Flor: 46, Receta: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := rules.Evaluate(tc.cards, vira)
			require.NoError(t, err)
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestHoldingRejectsImpossibleCards(t *testing.T) {
	vira := rules.Card{Suit: rules.Spades, Number: 4}
	for _, cards := range [][3]rules.Card{
		{{rules.Coins, 8}, {rules.Clubs, 1}, {rules.Cups, 1}},
		{{rules.Coins, 1}, {rules.Coins, 1}, {rules.Cups, 1}},
		{{rules.Spades, 4}, {rules.Clubs, 1}, {rules.Cups, 1}},
		{{"unknown", 1}, {rules.Clubs, 1}, {rules.Cups, 1}},
	} {
		_, err := rules.Evaluate(cards, vira)
		assert.Error(t, err)
	}
}

func TestDeckHasFortyUniqueLegalCards(t *testing.T) {
	seen := make(map[rules.Card]bool)
	for _, card := range rules.Deck() {
		assert.True(t, card.Valid())
		assert.False(t, seen[card], "duplicate %v", card)
		seen[card] = true
	}
	assert.Len(t, seen, 40)
}
