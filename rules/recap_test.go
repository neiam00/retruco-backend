package rules_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHandRecapRecordsPaymentsAndRetainsOnlyPublicCardsAcrossHands(t *testing.T) {
	state := simple(t)
	state.Scores = [2]int{3, 2}
	state = apply(t, state, 1, rules.Command{Type: "call", Kind: "envido"})
	state = apply(t, state, 0, rules.Command{Type: "respond", Kind: "accept"})
	state = apply(t, state, 1, rules.Command{Type: "call", Kind: "truco"})
	state = apply(t, state, 0, rules.Command{Type: "respond", Kind: "accept"})
	for _, card := range []int{0, 1} {
		for range 2 {
			state = apply(t, state, state.Turn, rules.Command{Type: "play", Card: card})
		}
	}
	require.Equal(t, rules.HandOver, state.Phase)
	view := rules.ViewFor(state, 0, false)
	recap := view.LastHand
	require.NotNil(t, recap)
	assert.Equal(t, "two_tricks", recap.Reason)
	assert.Equal(t, 1, recap.Winner)
	assert.Equal(t, -1, recap.MatchWinner)
	assert.Equal(t, [2]int{3, 2}, recap.ScoresBefore)
	assert.Equal(t, [2]int{3, 7}, recap.ScoresAfter)
	assert.Equal(t, []rules.Payout{{Kind: "envido", Team: 1, Points: 2}, {Kind: "truco", Team: 1, Points: 3}}, recap.Payouts)
	require.Len(t, recap.Tricks, 3)
	assert.Equal(t, "won", recap.Tricks[0].Status)
	assert.Equal(t, 1, recap.Tricks[0].WinnerSeat)
	assert.Equal(t, "unplayed", recap.Tricks[2].Status)
	for _, play := range recap.Tricks[2].Plays {
		assert.Nil(t, play.Card, "unplayed hands are never disclosed by review")
	}
	assert.Len(t, recap.EnvidoResults, 2)

	next, err := rules.StartHand(state, rules.Deck())
	require.NoError(t, err)
	assert.Equal(t, recap, rules.ViewFor(next, 1, false).LastHand, "the next deal preserves the last completed hand")
	assert.Nil(t, rules.ViewFor(next, 1, false).Vira)
	assert.Equal(t, rules.Card{Suit: rules.Spades, Number: 4}, *recap.Vira)
	// Returned views own their arrays and pointers, including the retained recap.
	recap.Vira.Number = 12
	recap.Tricks[0].Plays[0].Card.Number = 12
	recap.Payouts[0].Points = 99
	recap.EnvidoResults[0].Value = 99
	unchanged := rules.ViewFor(next, 1, false).LastHand
	assert.Equal(t, 4, unchanged.Vira.Number)
	assert.Equal(t, 1, unchanged.Tricks[0].Plays[0].Card.Number)
	assert.Equal(t, 2, unchanged.Payouts[0].Points)
	assert.NotEqual(t, 99, unchanged.EnvidoResults[0].Value)
}

func TestHandRecapDistinguishesMandatoryTopsBottomsAndManoTies(t *testing.T) {
	for _, scenario := range []struct {
		name, reason  string
		first, second [3]rules.Card
		winner        int
		hidden        bool
	}{
		{name: "tops decide without revealing bottoms", reason: "mandatory_top", winner: 0, hidden: true,
			first: [3]rules.Card{{rules.Coins, 1}, {rules.Clubs, 1}, {rules.Coins, 3}}, second: [3]rules.Card{{rules.Cups, 1}, {rules.Spades, 1}, {rules.Cups, 2}}},
		{name: "tied tops reveal bottoms", reason: "mandatory_bottom", winner: 1,
			first: [3]rules.Card{{rules.Coins, 1}, {rules.Clubs, 1}, {rules.Coins, 2}}, second: [3]rules.Card{{rules.Cups, 1}, {rules.Spades, 1}, {rules.Cups, 2}}},
		{name: "all tied goes to mano", reason: "mano_tie", winner: 1,
			first: [3]rules.Card{{rules.Coins, 1}, {rules.Coins, 10}, {rules.Coins, 2}}, second: [3]rules.Card{{rules.Cups, 1}, {rules.Cups, 10}, {rules.Cups, 2}}},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			state := fixture(t, false, scenario.first, scenario.second)
			for range 2 {
				state = apply(t, state, state.Turn, rules.Command{Type: "play", Card: 0})
			}
			require.True(t, state.MandatoryStack)
			for range 2 {
				state = apply(t, state, state.Turn, rules.Command{Type: "stack", Card: 1, Top: 2})
			}
			require.Equal(t, rules.HandOver, state.Phase)
			recap := rules.ViewFor(state, -1, true).LastHand
			require.NotNil(t, recap)
			assert.Equal(t, scenario.reason, recap.Reason)
			assert.Equal(t, scenario.winner, recap.Winner)
			assert.Equal(t, "tie", recap.Tricks[0].Status)
			if scenario.hidden {
				assert.Equal(t, "hidden", recap.Tricks[1].Status)
				assert.Equal(t, -1, recap.Tricks[1].WinnerSeat)
				for _, play := range recap.Tricks[1].Plays {
					assert.Nil(t, play.Card)
					assert.True(t, play.Concealed)
				}
			} else {
				for _, play := range recap.Tricks[1].Plays {
					assert.NotNil(t, play.Card)
					assert.False(t, play.Concealed)
				}
			}
		})
	}
}

func TestDepartureRecapDoesNotInventStonesOrRevealCards(t *testing.T) {
	state, err := rules.New(rules.Config{Players: 2, Target: 12}, 0, rules.Deck())
	require.NoError(t, err)
	state.Scores = [2]int{6, 4}
	finished := apply(t, state, 0, rules.Command{Type: "leave"})
	recap := rules.ViewFor(finished, 1, false).LastHand
	require.NotNil(t, recap)
	assert.Equal(t, "forfeit", recap.Reason)
	assert.Equal(t, -1, recap.Winner)
	assert.Equal(t, 1, recap.MatchWinner)
	assert.Equal(t, [2]int{6, 4}, recap.ScoresBefore)
	assert.Equal(t, recap.ScoresBefore, recap.ScoresAfter)
	assert.Empty(t, recap.Payouts)
	assert.Nil(t, recap.Vira)
	for _, trick := range recap.Tricks {
		for _, play := range trick.Plays {
			assert.Nil(t, play.Card)
		}
	}
}

func TestHandRecapKeepsVoluntarilyBurnedCardsPrivate(t *testing.T) {
	state := simple(t)
	for range 2 {
		state = apply(t, state, state.Turn, rules.Command{Type: "play", Card: 0})
	}
	for range 2 {
		state = apply(t, state, state.Turn, rules.Command{Type: "burn", Card: 1})
	}
	require.Equal(t, rules.HandOver, state.Phase)
	recap := rules.ViewFor(state, -1, true).LastHand
	require.NotNil(t, recap)
	assert.Equal(t, "first_and_tie", recap.Reason)
	assert.Equal(t, "tie", recap.Tricks[1].Status)
	for _, play := range recap.Tricks[1].Plays {
		assert.True(t, play.Burned)
		assert.Nil(t, play.Card, "ending a hand must not turn burned cards into revealed cards")
	}
}
