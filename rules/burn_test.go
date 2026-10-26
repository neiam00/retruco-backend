package rules_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVoluntaryBurnConsumesChosenCardWithoutRevealingOrWinning(t *testing.T) {
	s := simple(t)
	s = apply(t, s, 1, rules.Command{Type: "play", Card: 0})
	s = apply(t, s, 0, rules.Command{Type: "play", Card: 0})
	require.Contains(t, rules.LegalActions(s, 1), rules.Command{Type: "burn", Card: 1})

	next, events, err := rules.Apply(s, 1, rules.Command{Type: "burn", Card: 1})
	require.NoError(t, err)
	assert.Equal(t, []rules.Event{{Type: "burn", Seat: 1}}, events)
	assert.Equal(t, s.Revision+1, next.Revision)
	assert.Equal(t, 0, next.Turn)
	assert.Equal(t, []int{2}, next.Remaining(1))
	assert.True(t, next.Players[1].Plays[1].Burned)
	for _, perspective := range []int{-1, 0, 1} {
		play := rules.ViewFor(next, perspective, perspective < 0).Seats[1].Plays[1]
		assert.True(t, play.Present)
		assert.True(t, play.Burned)
		assert.Nil(t, play.Card, "a burned card must stay face down on the table")
	}

	// The burned seven would beat this six if it were still a valid card.
	next = apply(t, next, 0, rules.Command{Type: "play", Card: 1})
	assert.Equal(t, 2, next.Round)
	assert.Equal(t, 0, next.Turn)
	next = apply(t, next, 0, rules.Command{Type: "burn", Card: 2})
	next = apply(t, next, 1, rules.Command{Type: "play", Card: 2})
	assert.Equal(t, rules.HandOver, next.Phase)
	assert.Equal(t, 1, next.HandWinner)
}

func TestVoluntaryBurnRejectsInvalidTurnsAndCardsWithoutChangingState(t *testing.T) {
	firstRound := simple(t)
	secondRound := apply(t, firstRound, 1, rules.Command{Type: "play", Card: 0})
	secondRound = apply(t, secondRound, 0, rules.Command{Type: "play", Card: 0})
	pending := apply(t, secondRound, 1, rules.Command{Type: "call", Kind: "truco"})
	for _, test := range []struct {
		name  string
		state rules.State
		seat  int
		card  int
	}{
		{"first round", firstRound, 1, 1},
		{"other player's turn", secondRound, 0, 1},
		{"pending canto", pending, 1, 1},
		{"used card", secondRound, 1, 0},
		{"negative index", secondRound, 1, -1},
		{"out of range", secondRound, 1, 3},
	} {
		t.Run(test.name, func(t *testing.T) {
			command := rules.Command{Type: "burn", Card: test.card}
			next, events, err := rules.Apply(test.state, test.seat, command)
			require.Error(t, err)
			assert.Equal(t, test.state, next)
			assert.Empty(t, events)
			assert.NotContains(t, rules.LegalActions(test.state, test.seat), command)
		})
	}
}

func TestVoluntaryBurnRequiresNormalFlorDeclarations(t *testing.T) {
	for _, test := range []struct {
		name   string
		second string
		third  string
		valid  bool
	}{
		{"both declared", "flor", "presento_flor", true},
		{"second omitted", "", "presento_flor", false},
		{"third omitted", "flor", "", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			s := fixture(t, true,
				[3]rules.Card{{rules.Clubs, 4}, {rules.Cups, 5}, {rules.Spades, 6}},
				[3]rules.Card{{rules.Coins, 3}, {rules.Coins, 7}, {rules.Coins, 6}})
			s = apply(t, s, 1, rules.Command{Type: "play", Card: 0, Declaration: "flor"})
			s = apply(t, s, 0, rules.Command{Type: "play", Card: 0})
			s = apply(t, s, 1, rules.Command{Type: "burn", Card: 1, Declaration: test.second})
			s = apply(t, s, 0, rules.Command{Type: "play", Card: 1})
			s = apply(t, s, 0, rules.Command{Type: "burn", Card: 2})
			s = apply(t, s, 1, rules.Command{Type: "burn", Card: 2, Declaration: test.third})
			assert.Equal(t, test.valid, s.Players[1].FlorValid)
		})
	}
}

func TestVoluntaryBurnDuringMandatoryPegadoBurnsBothCards(t *testing.T) {
	for _, declaration := range []string{"", "flor", "flor_flor"} {
		t.Run("declaration="+declaration, func(t *testing.T) {
			s := fixture(t, true,
				[3]rules.Card{{rules.Clubs, 3}, {rules.Cups, 4}, {rules.Spades, 5}},
				[3]rules.Card{{rules.Coins, 3}, {rules.Coins, 6}, {rules.Coins, 7}})
			s = apply(t, s, 1, rules.Command{Type: "play", Card: 0, Declaration: "flor"})
			s = apply(t, s, 0, rules.Command{Type: "play", Card: 0})
			require.True(t, s.MandatoryStack)
			command := rules.Command{Type: "burn", Declaration: declaration}
			require.Contains(t, rules.LegalActions(s, 1), command)
			s = apply(t, s, 1, command)
			assert.Empty(t, s.Remaining(1))
			assert.Equal(t, 0, s.Turn)
			assert.Equal(t, declaration != "", s.Players[1].FlorValid)
			s = apply(t, s, 0, rules.Command{Type: "burn"})
			assert.Equal(t, rules.HandOver, s.Phase)
			assert.Equal(t, 1, s.HandWinner, "all rounds tied: mano wins")
			for seat := range 2 {
				for _, round := range []int{1, 2} {
					play := rules.ViewFor(s, -1, true).Seats[seat].Plays[round]
					assert.True(t, play.Burned)
					assert.True(t, play.Present)
					assert.Nil(t, play.Card)
				}
			}
		})
	}
}

func TestVoluntaryBurnPreservesEnvidoAndProtectedFlores(t *testing.T) {
	for _, test := range []struct {
		name     string
		required bool
		hand     [3]rules.Card
	}{
		{"free flor", false, [3]rules.Card{{rules.Coins, 3}, {rules.Coins, 6}, {rules.Coins, 7}}},
		{"receta", true, [3]rules.Card{{rules.Spades, 1}, {rules.Spades, 10}, {rules.Spades, 11}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			s := fixture(t, test.required,
				[3]rules.Card{{rules.Clubs, 4}, {rules.Cups, 5}, {rules.Spades, 6}}, test.hand)
			holding := s.Players[1].Holding
			s = apply(t, s, 1, rules.Command{Type: "play", Card: 0})
			s = apply(t, s, 0, rules.Command{Type: "play", Card: 0})
			s = apply(t, s, 1, rules.Command{Type: "burn", Card: 1})
			assert.True(t, s.Players[1].FlorValid)
			assert.Equal(t, holding, s.Players[1].Holding, "burning changes the play, not the original envido value")
		})
	}
}
