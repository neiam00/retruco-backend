package rules_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
)

func opposingFlores(t *testing.T) rules.State {
	t.Helper()
	s := fixture(t, true,
		[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}},
		[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}},
		[3]rules.Card{{rules.Coins, 7}, {rules.Coins, 6}, {rules.Coins, 5}},
		[3]rules.Card{{rules.Spades, 7}, {rules.Spades, 6}, {rules.Spades, 5}},
	)
	s = apply(t, s, 1, rules.Command{Type: "play", Card: 0, Declaration: "a_ley"})
	return apply(t, s, 2, rules.Command{Type: "call", Kind: "flor"})
}

func TestFlorPrefixDeclaresFlorForContestEligibility(t *testing.T) {
	for _, declarationPoint := range []string{"call", "response"} {
		t.Run(declarationPoint, func(t *testing.T) {
			s := fixture(t, false,
				[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}},
				[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}},
			)
			if declarationPoint == "call" {
				s = apply(t, s, 1, rules.Command{Type: "call", Kind: "truco", Declaration: "con_mi_flor"})
				s = apply(t, s, 0, rules.Command{Type: "call", Kind: "flor"})
				assert.NotNil(t, s.Suspended, "the pending truco remains suspended during the flor contest")
			} else {
				s = apply(t, s, 1, rules.Command{Type: "call", Kind: "truco"})
				s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "accept", Declaration: "con_mi_flor"})
				s = apply(t, s, 1, rules.Command{Type: "call", Kind: "flor"})
			}
			assert.Equal(t, "flor", s.Pending.Kind)
		})
	}
}

func TestFlorContestRefusalInvalidatesBothTeammates(t *testing.T) {
	s := opposingFlores(t)
	s = apply(t, s, 3, rules.Command{Type: "respond", Kind: "decline", Declaration: "con_mi_flor"})
	assert.False(t, s.Players[1].FlorValid)
	assert.False(t, s.Players[3].FlorValid)
	assert.Equal(t, [2]int{}, s.Scores, "refusal still waits for hand settlement")
	s = apply(t, s, 2, rules.Command{Type: "call", Kind: "truco", Declaration: "con_mi_flor"})
	s = apply(t, s, 1, rules.Command{Type: "respond", Kind: "decline"})
	assert.Equal(t, [2]int{8, 0}, s.Scores)
	assert.Equal(t, []rules.Payout{
		{Kind: "flor", Team: 0, Points: 6},
		{Kind: "envido", Team: 0, Points: 1},
		{Kind: "truco", Team: 0, Points: 1},
	}, s.Payouts)
}

func TestFlorContestTieUsesManoAndInvalidatesAllLosingOrdinaryFlores(t *testing.T) {
	s := opposingFlores(t)
	s = apply(t, s, 3, rules.Command{Type: "respond", Kind: "accept", Declaration: "con_mi_flor"})
	s = apply(t, s, 2, rules.Command{Type: "call", Kind: "truco", Declaration: "con_mi_flor"})
	s = apply(t, s, 1, rules.Command{Type: "respond", Kind: "decline", Declaration: "con_mi_flor"})
	assert.False(t, s.Players[0].FlorValid)
	assert.False(t, s.Players[2].FlorValid)
	assert.True(t, s.Players[1].FlorValid)
	assert.True(t, s.Players[3].FlorValid)
	assert.Equal(t, [2]int{1, 8}, s.Scores, "mano wins tied flores despite losing truco")
}

func TestFlorContestPaysNeitherTeamWhenEveryFlorBurns(t *testing.T) {
	s := opposingFlores(t)
	s = apply(t, s, 3, rules.Command{Type: "respond", Kind: "accept", Declaration: "con_mi_flor"})
	// Each player speaks without the required flor prefix. The final surviving
	// flor must not keep an earlier provisional entitlement to the wager.
	s = apply(t, s, 2, rules.Command{Type: "call", Kind: "truco"})
	s = apply(t, s, 1, rules.Command{Type: "respond", Kind: "raise", Amount: 6})
	s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 9})
	s = apply(t, s, 3, rules.Command{Type: "respond", Kind: "decline"})
	assert.Equal(t, [2]int{6, 0}, s.Scores)
	assert.Equal(t, []rules.Payout{{Kind: "truco", Team: 0, Points: 6}}, s.Payouts)
	for _, result := range s.EnvidoResults {
		assert.False(t, result.Eligible)
	}
}

func TestSimultaneousUncontestedFloresEndMatchForManoBeforeValePartida(t *testing.T) {
	s := fixture(t, false,
		[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}},
		[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}},
	)
	s.Scores = [2]int{10, 10}
	s = apply(t, s, 1, rules.Command{Type: "call", Kind: "truco"})
	s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 100})
	s = apply(t, s, 1, rules.Command{Type: "respond", Kind: "accept"})
	for range 4 {
		s = apply(t, s, s.Turn, rules.Command{Type: "play", Card: s.Round})
		if s.MandatoryStack {
			break
		}
	}
	// The first-round sevens have distinct ranks: clubs beats cups. A tied six
	// in round two ends the hand for team 0, but flor settles before vale partida.
	assert.Equal(t, rules.MatchOver, s.Phase)
	assert.Equal(t, 0, s.HandWinner)
	assert.Equal(t, 1, s.Winner)
	assert.Equal(t, [2]int{13, 13}, s.Scores)
}
