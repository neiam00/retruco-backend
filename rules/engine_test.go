package rules_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLeavingBeforeTheDealDoesNotRevealUninitializedCards(t *testing.T) {
	state, err := rules.New(rules.Config{Players: 2, Target: 12}, 0, rules.Deck())
	require.NoError(t, err)
	finished, _, err := rules.Apply(state, 0, rules.Command{Type: "leave"})
	require.NoError(t, err)
	require.Equal(t, rules.MatchOver, finished.Phase)
	view := rules.ViewFor(finished, 1, false)
	assert.Nil(t, view.Vira)
	assert.Empty(t, view.Hand)
	assert.Nil(t, view.Holding)
}

// fixture deals supplied hands through the public cut/deal command. It builds
// the input deck independently using the specification's draw order.
func fixture(t *testing.T, required bool, hands ...[3]rules.Card) rules.State {
	t.Helper()
	n := len(hands)
	counts := make([]int, n)
	draws := []rules.Card{}
	if n == 4 {
		for _, viewer := range []int{3, 0} {
			receiver := (viewer + 2) % 4
			draws = append(draws, hands[receiver][0])
			counts[receiver]++
		}
	}
	for dealt, seat := len(draws), n-1; dealt < n*3; seat = (seat + 1) % n {
		if counts[seat] < 3 {
			draws = append(draws, hands[seat][counts[seat]])
			counts[seat]++
			dealt++
		}
	}
	vira := rules.Card{Suit: rules.Spades, Number: 4}
	draws = append(draws, vira)
	used := map[rules.Card]bool{}
	for _, c := range draws {
		require.False(t, used[c], "duplicate fixture card %v", c)
		used[c] = true
	}
	for _, c := range rules.Deck() {
		if !used[c] {
			draws = append(draws, c)
		}
	}
	cutDeck := make([]rules.Card, 40)
	for i, c := range draws {
		cutDeck[39-i] = c
	}
	deck := append([]rules.Card{cutDeck[39]}, cutDeck[:39]...)
	s, err := rules.New(rules.Config{Players: n, Target: 12, FlorRequired: required}, 0, deck)
	require.NoError(t, err)
	s = apply(t, s, n-1, rules.Command{Type: "cut", Cut: 1})
	for seat, hand := range hands {
		require.Equal(t, hand, s.Players[seat].Cards)
	}
	return s
}

func apply(t *testing.T, s rules.State, seat int, c rules.Command) rules.State {
	t.Helper()
	next, _, err := rules.Apply(s, seat, c)
	require.NoError(t, err, "seat %d command %+v", seat, c)
	return next
}

func simple(t *testing.T) rules.State {
	return fixture(t, false,
		[3]rules.Card{{rules.Clubs, 1}, {rules.Coins, 6}, {rules.Cups, 5}},
		[3]rules.Card{{rules.Spades, 1}, {rules.Coins, 7}, {rules.Cups, 6}})
}

func TestTrucoRaisesAndInterruption(t *testing.T) {
	s := simple(t)
	s = apply(t, s, 1, rules.Command{Type: "call", Kind: "truco"})
	s = apply(t, s, 0, rules.Command{Type: "call", Kind: "envido"})
	require.NotNil(t, s.Suspended)
	before := s
	next, _, err := rules.Apply(s, 1, rules.Command{Type: "play", Card: 0})
	require.Error(t, err)
	assert.Equal(t, before, next)
	s = apply(t, s, 1, rules.Command{Type: "respond", Kind: "raise", Amount: 2})
	s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 5})
	s = apply(t, s, 1, rules.Command{Type: "respond", Kind: "decline"})
	assert.Equal(t, 4, s.Envido.Value)
	assert.Equal(t, [2]int{}, s.Scores, "envido settles only at hand end")
	require.NotNil(t, s.Pending)
	assert.Equal(t, "truco", s.Pending.Kind)
	s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 9})
	s = apply(t, s, 1, rules.Command{Type: "respond", Kind: "decline"})
	assert.Equal(t, [2]int{7, 0}, s.Scores, "4 envido plus previous 3-point truco")
}

func TestEnvidoCapAndTerminalFalta(t *testing.T) {
	s := simple(t)
	s.Scores = [2]int{8, 7}
	s = apply(t, s, 1, rules.Command{Type: "call", Kind: "envido"})
	s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 8})
	for _, c := range []rules.Command{{Type: "respond", Kind: "raise", Amount: 5}, {Type: "respond", Kind: "falta"}} {
		next, _, err := rules.Apply(s, 1, c)
		require.Error(t, err)
		assert.Equal(t, s, next)
	}
	s = simple(t)
	s.Scores = [2]int{8, 7}
	s = apply(t, s, 1, rules.Command{Type: "call", Kind: "envido"})
	s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 2})
	s = apply(t, s, 1, rules.Command{Type: "respond", Kind: "falta"})
	assert.Equal(t, 4, s.Pending.Value, "equal-value falta is legal")
	_, _, err := rules.Apply(s, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 2})
	assert.Error(t, err, "falta cannot be raised")
}

func TestLeyScoresBeforeFlorAndEndsMatch(t *testing.T) {
	s := fixture(t, false,
		[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}},
		[3]rules.Card{{rules.Cups, 1}, {rules.Coins, 4}, {rules.Spades, 5}})
	s.Scores = [2]int{10, 11}
	s = apply(t, s, 1, rules.Command{Type: "call", Kind: "ley_envido"})
	s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "accept", Mode: "con_ley"})
	s = apply(t, s, 1, rules.Command{Type: "call", Kind: "truco"})
	s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "decline"})
	assert.Equal(t, 0, s.Winner)
	assert.Equal(t, [2]int{12, 11}, s.Scores)
	assert.Equal(t, []rules.Payout{{Kind: "envido", Team: 0, Points: 2}}, s.Payouts)
}

func TestOrdinaryEnvidoWithFlorWordingPreservesCallerAndOpposingFlorBlocks(t *testing.T) {
	s := fixture(t, false,
		[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}},
		[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}})
	s = apply(t, s, 1, rules.Command{Type: "call", Kind: "envido", Declaration: "con_mi_flor"})
	assert.True(t, s.Players[1].FlorValid)
	s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "flowers_no"})
	assert.True(t, s.EnvidoCanceled)
	assert.True(t, s.Players[0].FlorValid)
	_, _, err := rules.Apply(s, 1, rules.Command{Type: "call", Kind: "envido"})
	assert.Error(t, err)
}

func TestMandatoryPegadoUsesTopFirstAndKeepsBottomHidden(t *testing.T) {
	s := fixture(t, false,
		[3]rules.Card{{rules.Clubs, 3}, {rules.Spades, 11}, {rules.Cups, 4}},
		[3]rules.Card{{rules.Coins, 3}, {rules.Coins, 4}, {rules.Spades, 1}})
	s = apply(t, s, 1, rules.Command{Type: "call", Kind: "envido"})
	s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "accept"})
	assert.Empty(t, rules.ViewFor(s, 1, true).EnvidoResults, "accepted values stay private until settlement")
	s = apply(t, s, 1, rules.Command{Type: "play", Card: 0})
	s = apply(t, s, 0, rules.Command{Type: "play", Card: 0})
	assert.True(t, s.MandatoryStack)
	s = apply(t, s, 1, rules.Command{Type: "stack", Card: 1, Top: 2})
	s = apply(t, s, 0, rules.Command{Type: "stack", Card: 1, Top: 2})
	assert.Equal(t, 1, s.HandWinner, "top espadilla beats top four despite opposing bottom perico")
	assert.True(t, s.Players[0].Plays[1].Concealed)
	v := rules.ViewFor(s, 1, true)
	assert.Nil(t, v.Seats[0].Plays[1].Card)
	assert.Empty(t, v.Legal)
	assert.Len(t, v.Hand, 3)
	assert.Empty(t, rules.ViewFor(s, -1, true).Hand)
	assert.Equal(t, []rules.EnvidoResult{
		{Seat: 0, Value: 34, Eligible: true},
		{Seat: 1, Value: 27, Eligible: true},
	}, v.EnvidoResults, "only the final comparison values are revealed")
}

func TestTimeoutsBurnBothMandatoryCardsButNeverFirstCard(t *testing.T) {
	s := fixture(t, false,
		[3]rules.Card{{rules.Clubs, 3}, {rules.Spades, 11}, {rules.Cups, 4}},
		[3]rules.Card{{rules.Coins, 3}, {rules.Coins, 4}, {rules.Spades, 1}})
	var err error
	s, _, err = rules.Timeout(s, 0)
	require.NoError(t, err)
	assert.False(t, s.Players[1].Plays[0].Burned)
	s = apply(t, s, 0, rules.Command{Type: "play", Card: 0})
	s, _, err = rules.Timeout(s, 0)
	require.NoError(t, err)
	s, _, err = rules.Timeout(s, 0)
	require.NoError(t, err)
	assert.Equal(t, 1, s.HandWinner, "all rounds tied: mano wins")
	for seat := 0; seat < 2; seat++ {
		assert.True(t, s.Players[seat].Plays[1].Burned)
		assert.True(t, s.Players[seat].Plays[2].Burned)
	}
}

func TestVoluntaryPegadoBurnsLosingTopCard(t *testing.T) {
	s := fixture(t, false,
		[3]rules.Card{{rules.Spades, 1}, {rules.Clubs, 1}, {rules.Cups, 4}},
		[3]rules.Card{{rules.Coins, 4}, {rules.Cups, 5}, {rules.Spades, 11}})
	s = apply(t, s, 1, rules.Command{Type: "play", Card: 0})
	s = apply(t, s, 0, rules.Command{Type: "play", Card: 0})
	s = apply(t, s, 0, rules.Command{Type: "play", Card: 1})
	s = apply(t, s, 1, rules.Command{Type: "stack", Card: 1, Top: 2})
	assert.True(t, s.Players[1].Plays[1].Concealed)
	assert.Equal(t, rules.Playing, s.Phase)
	s = apply(t, s, 0, rules.Command{Type: "play", Card: 2})
	assert.True(t, s.Players[1].Plays[2].Burned)
	assert.Equal(t, 0, s.HandWinner)
}

func TestFlorFlorPreservesStackedFlor(t *testing.T) {
	for _, declaration := range []string{"flor_flor", "presento_flor", "flor", "con_mi_flor", ""} {
		s := fixture(t, true,
			[3]rules.Card{{rules.Clubs, 3}, {rules.Cups, 4}, {rules.Spades, 1}},
			[3]rules.Card{{rules.Coins, 3}, {rules.Coins, 7}, {rules.Coins, 6}})
		s = apply(t, s, 1, rules.Command{Type: "play", Card: 0, Declaration: "flor"})
		s = apply(t, s, 0, rules.Command{Type: "play", Card: 0})
		s = apply(t, s, 1, rules.Command{Type: "stack", Card: 1, Top: 2, Declaration: declaration})
		assert.Equal(t, declaration != "", s.Players[1].FlorValid)
	}
}

func TestVoluntaryBottomStaysHiddenWhenVisibleCardsAlreadyDecideTheHand(t *testing.T) {
	for _, bottom := range []rules.Card{{Suit: rules.Spades, Number: 11}, {Suit: rules.Cups, Number: 4}} {
		s := fixture(t, false,
			[3]rules.Card{{rules.Spades, 1}, {rules.Coins, 7}, {rules.Clubs, 1}},
			[3]rules.Card{{rules.Coins, 4}, bottom, {rules.Cups, 5}},
		)
		s = apply(t, s, 1, rules.Command{Type: "play", Card: 0})
		s = apply(t, s, 0, rules.Command{Type: "play", Card: 0})
		s = apply(t, s, 0, rules.Command{Type: "play", Card: 1})
		s = apply(t, s, 1, rules.Command{Type: "stack", Card: 1, Top: 2})
		s = apply(t, s, 0, rules.Command{Type: "play", Card: 2})
		assert.Equal(t, 0, s.HandWinner)
		assert.True(t, s.Players[1].Plays[1].Concealed)
		assert.Nil(t, rules.ViewFor(s, 0, true).Seats[1].Plays[1].Card)
		assert.Equal(t, bottom.Suit == rules.Cups, s.Players[1].Plays[2].Burned,
			"the top-card penalty still applies even when the bottom need not be shown")
	}
}

func TestVisibleThirdRoundTieDoesNotExposeUnneededVoluntaryBottom(t *testing.T) {
	s := fixture(t, false,
		[3]rules.Card{{rules.Spades, 1}, {rules.Coins, 7}, {rules.Coins, 3}},
		[3]rules.Card{{rules.Coins, 4}, {rules.Cups, 7}, {rules.Coins, 5}},
		[3]rules.Card{{rules.Cups, 4}, {rules.Clubs, 4}, {rules.Clubs, 3}},
		[3]rules.Card{{rules.Spades, 5}, {rules.Clubs, 5}, {rules.Spades, 3}},
	)
	for range 4 {
		s = apply(t, s, s.Turn, rules.Command{Type: "play", Card: 0})
	}
	s = apply(t, s, 0, rules.Command{Type: "play", Card: 1})
	s = apply(t, s, 1, rules.Command{Type: "stack", Card: 1, Top: 2})
	for _, seat := range []int{2, 3} {
		s = apply(t, s, seat, rules.Command{Type: "play", Card: 1})
	}
	for _, seat := range []int{2, 3, 0} {
		s = apply(t, s, seat, rules.Command{Type: "play", Card: 2})
	}
	assert.Equal(t, 0, s.HandWinner)
	assert.True(t, s.Players[1].Plays[1].Concealed)
	assert.True(t, s.Players[1].Plays[2].Burned)
}

func TestRecetaWagerReplacesAutomaticPointAndDoesNotGuaranteeTruco(t *testing.T) {
	s := fixture(t, false,
		[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}},
		[3]rules.Card{{rules.Spades, 11}, {rules.Spades, 10}, {rules.Cups, 1}})
	s = apply(t, s, 1, rules.Command{Type: "call", Kind: "ley_envido"})
	s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "accept", Mode: "con_ley"})
	s = apply(t, s, 1, rules.Command{Type: "call", Kind: "truco"})
	s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 9})
	s = apply(t, s, 1, rules.Command{Type: "respond", Kind: "decline"})
	assert.Equal(t, [2]int{6, 5}, s.Scores, "receta receives 3 flor + 2 envido; opponent gets 3 flor + 3 truco")
	assert.Equal(t, 0, s.HandWinner)
}
