package cpu_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/cpu"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEnvidoBeforeInterruptibleTrucoWithTwentySeven(t *testing.T) {
	// Diagnostic deal 120634, second hand. Both aces make truco attractive.
	// The ordinary envido can be offered now and truco still called afterward.
	v := tableWithVira(t, 4, 0, [3]rules.Card{
		{Suit: rules.Spades, Number: 1}, {Suit: rules.Clubs, Number: 1}, {Suit: rules.Spades, Number: 6},
	}, rules.Card{Suit: rules.Coins, Number: 11})
	v.Config.Target, v.Scores = 12, [2]int{4, 0}
	envido := rules.Command{Type: "call", Kind: "envido"}
	truco := rules.Command{Type: "call", Kind: "truco", Amount: 3}
	v.Legal = append(v.Legal, envido, truco)
	require.Equal(t, 27, v.Holding.Envido)
	assert.Equal(t, []rules.Command{envido}, cpu.PlayingChoices(v, 0))
	assert.Contains(t, v.Legal, truco)
}

func TestWeakEnvidoDelaysTrucoUntilInterruptionCloses(t *testing.T) {
	v := table(t, 4, 0, [3]rules.Card{
		{Suit: rules.Spades, Number: 1}, {Suit: rules.Clubs, Number: 1}, {Suit: rules.Cups, Number: 2},
	})
	truco := rules.Command{Type: "call", Kind: "truco", Amount: 3}
	envido := rules.Command{Type: "call", Kind: "envido"}
	v.Legal = append(v.Legal, truco, envido)
	require.Equal(t, 2, v.Holding.Envido)
	choices := cpu.PlayingChoices(v, 0)
	assert.NotContains(t, choices, envido)
	assert.NotContains(t, choices, truco)
	require.NotEmpty(t, choices)
	for _, choice := range choices {
		assert.Equal(t, "play", choice.Type)
	}
	// If envido is already settled, neither rival can interrupt.
	v.Legal = v.Legal[:len(v.Legal)-1]
	assert.Contains(t, cpu.PlayingChoices(v, 0), truco)
}

func TestPartnerPreviewCanEstablishTeamEnvido(t *testing.T) {
	v := table(t, 4, 0, [3]rules.Card{
		{Suit: rules.Cups, Number: 2}, {Suit: rules.Spades, Number: 12}, {Suit: rules.Clubs, Number: 10},
	})
	v.Mano = 2
	envido := rules.Command{Type: "call", Kind: "envido"}
	v.Legal = append(v.Legal, envido)
	assert.NotContains(t, cpu.PlayingChoices(v, 0), envido)
	// A known perico gives the teammate at least 30 with any second card.
	v.Preview = &rules.Card{Suit: rules.Coins, Number: 11}
	assert.Equal(t, []rules.Command{envido}, cpu.PlayingChoices(v, 0))
}

func TestLeyOpeningRespectsAnOpponentsDeclaredFlor(t *testing.T) {
	v := table(t, 4, 0, [3]rules.Card{
		{Suit: rules.Cups, Number: 7}, {Suit: rules.Cups, Number: 6}, {Suit: rules.Spades, Number: 3},
	})
	v.Mano = 3
	v.Seats[3].Plays[0] = played(rules.Clubs, 7)
	v.Seats[3].FlorDeclared = true
	ley := rules.Command{Type: "call", Kind: "ley_envido"}
	v.Legal = append(v.Legal, ley)
	assert.NotContains(t, cpu.PlayingChoices(v, 0), ley)
	assert.Contains(t, v.Legal, ley)
}
