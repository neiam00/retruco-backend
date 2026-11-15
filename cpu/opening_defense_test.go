package cpu_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/cpu"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHeadsUpOpeningBluffIsAnOrdinaryAffordableTruco(t *testing.T) {
	hand := [3]rules.Card{{Suit: rules.Coins, Number: 4}, {Suit: rules.Cups, Number: 5}, {Suit: rules.Clubs, Number: 6}}
	v := tableWithVira(t, 2, 0, hand, rules.Card{Suit: rules.Spades, Number: 4})
	bid := rules.Command{Type: "call", Kind: "truco", Amount: 3}
	envido := rules.Command{Type: "call", Kind: "envido"}
	allIn := rules.Command{Type: "call", Kind: "truco", Amount: 100}
	v.Legal = append(v.Legal, bid, envido, allIn)
	assert.NotContains(t, cpu.ChoicesForProfile(v, 0, "value-and-team-v7"), bid)
	choices := cpu.PlayingChoices(v, 0)
	assert.Contains(t, choices, bid)
	assert.NotContains(t, choices, envido)
	assert.NotContains(t, choices, allIn)
	v.Scores[0] = 2
	assert.NotContains(t, cpu.PlayingChoices(v, 0), bid, "protect a lead on a weak opening")
	assert.Contains(t, cpu.ChoicesForProfile(v, 0, cpu.RetrucoLadderProfile), bid)
	v.Scores[1] = v.Config.Target - 3
	assert.NotContains(t, cpu.PlayingChoices(v, 0), bid)
}

func TestHeadsUpAgainstVoluntaryPegadoPricesTheHiddenBottom(t *testing.T) {
	v := table(t, 2, 0, [3]rules.Card{{Suit: rules.Cups, Number: 4}, {Suit: rules.Spades, Number: 5}, {Suit: rules.Clubs, Number: 6}})
	v.Mano, v.Round = 1, 1
	v.Hand[0].Used = true
	v.Seats[0].Plays[0], v.Seats[1].Plays[0] = played(rules.Cups, 4), played(rules.Clubs, 3)
	v.Seats[1].Plays[1] = rules.PublicPlay{Present: true, Concealed: true}
	v.Seats[1].Plays[2] = played(rules.Cups, 2)
	low, high := rules.Command{Type: "play", Card: 1}, rules.Command{Type: "play", Card: 2}
	bid := rules.Command{Type: "call", Kind: "truco", Amount: 3}
	v.Legal = []rules.Command{low, high, bid}
	require.Contains(t, cpu.ChoicesForProfile(v, 0, "value-and-team-v7"), bid)
	assert.Equal(t, []rules.Command{high}, cpu.PlayingChoices(v, 0), "winning the bottom is the only way to keep this hand alive")
	v.Pending = &rules.Wager{Kind: "truco", Value: 100, Previous: 3, Caller: 1}
	accept, decline := rules.Command{Type: "respond", Kind: "accept"}, rules.Command{Type: "respond", Kind: "decline"}
	v.Legal = []rules.Command{accept, decline}
	assert.Equal(t, []rules.Command{decline}, cpu.PlayingChoices(v, 0))
	v.Scores[1] = v.Config.Target - 3
	assert.Equal(t, []rules.Command{accept}, cpu.PlayingChoices(v, 0), "declining would concede the match")
	v.Scores[1] = 0
	v.Hand[2].Card = rules.Card{Suit: rules.Coins, Number: 11}
	assert.Equal(t, []rules.Command{accept}, cpu.PlayingChoices(v, 0), "the perico beats every possible hidden bottom")
}

func TestHeadsUpEnvidoAtMatchPoint(t *testing.T) {
	v := table(t, 2, 0, [3]rules.Card{{Suit: rules.Cups, Number: 7}, {Suit: rules.Cups, Number: 6}, {Suit: rules.Clubs, Number: 4}})
	v.Scores[0] = v.Config.Target - 2
	v.Pending = &rules.Wager{Kind: "envido", Caller: 1, Value: 2, Previous: 1}
	accept := rules.Command{Type: "respond", Kind: "accept"}
	decline := rules.Command{Type: "respond", Kind: "decline"}
	v.Legal = []rules.Command{decline, accept, {Type: "respond", Kind: "raise", Amount: 2}, {Type: "respond", Kind: "falta"}}
	require.Equal(t, []rules.Command{accept}, cpu.PlayingChoices(v, 0), "take the two points on offer with a strong pair")
}
