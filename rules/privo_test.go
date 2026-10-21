package rules_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func privoHand(t *testing.T, target int, scores [2]int, required bool, hands ...[3]rules.Card) rules.State {
	t.Helper()
	dealt := fixture(t, required, hands...)
	state, err := rules.New(rules.Config{Players: len(hands), Target: target, FlorRequired: required}, 0, dealt.Deck[:])
	require.NoError(t, err)
	state.Scores = scores
	return apply(t, state, state.Turn, rules.Command{Type: "cut", Cut: 1})
}

func TestPrivoAnnouncesOnlyTeamMaximumAndLocksBettingBeforePlay(t *testing.T) {
	for _, target := range []int{12, 24} {
		state := privoHand(t, target, [2]int{target - 1, 0}, true,
			[3]rules.Card{{rules.Clubs, 1}, {rules.Clubs, 6}, {rules.Coins, 2}},
			[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 4}, {rules.Coins, 3}},
			[3]rules.Card{{rules.Coins, 7}, {rules.Coins, 6}, {rules.Spades, 3}},
			[3]rules.Card{{rules.Spades, 7}, {rules.Spades, 5}, {rules.Cups, 1}},
		)
		require.NotNil(t, state.Pending)
		assert.Equal(t, rules.Wager{Kind: "privo", Caller: 0, Value: 2, Previous: 1, Terminal: true}, *state.Pending)
		assert.True(t, state.EnvidoClosed)
		for _, perspective := range []int{-1, 0, 1, 2, 3} {
			view := rules.ViewFor(state, perspective, perspective < 0)
			assert.Equal(t, []rules.PrivoEnvido{{Team: 0, Value: 33}}, view.PrivoEnvido)
			assert.Empty(t, view.EnvidoResults, "individual values stay private before settlement")
			if perspective < 0 {
				assert.Empty(t, view.Hand)
				assert.Nil(t, view.Holding)
			}
		}
		assert.Empty(t, rules.LegalActions(state, 0))
		assert.Equal(t, []rules.Command{{Type: "respond", Kind: "accept"}, {Type: "respond", Kind: "decline"}}, rules.LegalActions(state, 3))
		for _, command := range []rules.Command{
			{Type: "play", Card: 0},
			{Type: "call", Kind: "truco"},
			{Type: "respond", Kind: "raise", Amount: 2},
			{Type: "respond", Kind: "falta"},
			{Type: "respond", Kind: "accept", Amount: 9},
			{Type: "respond", Kind: "accept", Mode: "con_ley"},
			{Type: "respond", Kind: "present_flor"},
		} {
			rejected, _, err := rules.Apply(state, 1, command)
			require.Error(t, err)
			assert.Equal(t, state, rejected)
		}
		accepted := apply(t, state, 3, rules.Command{Type: "respond", Kind: "accept"})
		assert.Nil(t, accepted.Pending)
		assert.Equal(t, 1, accepted.TrucoValue, "Privo y Truco does not itself raise truco")
		assert.True(t, accepted.EnvidoAccepted)
		for _, kind := range []string{"envido", "ley_envido", "flor"} {
			_, _, err := rules.Apply(accepted, accepted.Turn, rules.Command{Type: "call", Kind: kind})
			assert.Error(t, err)
		}
	}
}

func TestPrivoRefusalAndTimeoutEndMatchImmediately(t *testing.T) {
	for _, target := range []int{12, 24} {
		state := privoHand(t, target, [2]int{0, target - 1}, false,
			[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Coins, 2}},
			[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 4}, {rules.Coins, 3}},
		)
		for _, timeout := range []bool{false, true} {
			var next rules.State
			if timeout {
				var events []rules.Event
				var err error
				next, events, err = rules.Timeout(state, 0)
				require.NoError(t, err)
				require.Len(t, events, 1)
				assert.True(t, events[0].Automatic)
			} else {
				next = apply(t, state, 0, rules.Command{Type: "respond", Kind: "decline"})
			}
			assert.Equal(t, rules.MatchOver, next.Phase)
			assert.Equal(t, 1, next.Winner)
			assert.Equal(t, target, next.Scores[1])
			assert.Equal(t, []rules.Payout{{Kind: "envido", Team: 1, Points: 1}}, next.Payouts)
		}
	}
}

func TestPrivoEnvidoWinsBeforeOpponentsThreePointTruco(t *testing.T) {
	state := privoHand(t, 24, [2]int{23, 20}, false,
		[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Coins, 2}},
		[3]rules.Card{{rules.Spades, 1}, {rules.Cups, 7}, {rules.Cups, 3}},
	)
	state = apply(t, state, 1, rules.Command{Type: "respond", Kind: "accept"})
	state = apply(t, state, 1, rules.Command{Type: "call", Kind: "truco"})
	state = apply(t, state, 0, rules.Command{Type: "respond", Kind: "accept"})
	state = apply(t, state, 1, rules.Command{Type: "play", Card: 0})
	state = apply(t, state, 0, rules.Command{Type: "play", Card: 0})
	state = apply(t, state, 1, rules.Command{Type: "play", Card: 2})
	state = apply(t, state, 0, rules.Command{Type: "play", Card: 1})
	assert.Equal(t, 1, state.HandWinner)
	assert.Equal(t, 0, state.Winner)
	assert.Equal(t, 20, state.Scores[1], "truco is not awarded after envido ends the match")
	assert.Equal(t, []rules.Payout{{Kind: "envido", Team: 0, Points: 2}}, state.Payouts)
}

func TestPrivoFlorOutcomesAndRetainedFlor(t *testing.T) {
	plain := [3]rules.Card{{rules.Clubs, 1}, {rules.Clubs, 6}, {rules.Coins, 2}}
	flor := [3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}}
	for _, required := range []bool{false, true} {
		state := privoHand(t, 24, [2]int{23, 20}, required, plain, flor)
		presented := apply(t, state, 1, rules.Command{Type: "respond", Kind: "present_flor"})
		assert.Equal(t, rules.HandOver, presented.Phase)
		assert.Equal(t, [2]int{23, 23}, presented.Scores)
		assert.Equal(t, []rules.Payout{{Kind: "flor", Team: 1, Points: 3}}, presented.Payouts)
		assert.Empty(t, presented.Envido.Kind)
		assert.Nil(t, presented.Pending)

		state.Scores[1] = 0
		state = apply(t, state, 1, rules.Command{Type: "respond", Kind: "accept"})
		assert.True(t, state.Players[1].FlorValid, "playing instead preserves flor")
		state = apply(t, state, 1, rules.Command{Type: "call", Kind: "truco", Declaration: "con_mi_flor"})
		state = apply(t, state, 0, rules.Command{Type: "respond", Kind: "decline"})
		assert.Equal(t, []rules.Payout{
			{Kind: "envido", Team: 1, Points: 2},
			{Kind: "flor", Team: 1, Points: 3},
			{Kind: "truco", Team: 1, Points: 1},
		}, state.Payouts)
		assert.Equal(t, rules.HandOver, state.Phase)
	}
	for _, target := range []int{12, 24} {
		state := privoHand(t, target, [2]int{0, target - 1}, true, plain, flor)
		assert.Equal(t, rules.MatchOver, state.Phase)
		assert.Equal(t, 1, state.Winner)
		assert.GreaterOrEqual(t, state.Scores[1], target)
		assert.Nil(t, state.Pending)
	}
}

func TestBothTeamsPrivandoUseManoForTiesAndFlorPriority(t *testing.T) {
	state := privoHand(t, 12, [2]int{11, 11}, false,
		[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Coins, 2}},
		[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Coins, 3}},
	)
	assert.Equal(t, []rules.PrivoEnvido{{Team: 0, Value: 33}, {Team: 1, Value: 33}}, rules.ViewFor(state, -1, true).PrivoEnvido)
	assert.True(t, state.EnvidoAccepted)
	assert.Nil(t, state.Pending)
	state = apply(t, state, 1, rules.Command{Type: "call", Kind: "truco"})
	state = apply(t, state, 0, rules.Command{Type: "respond", Kind: "decline"})
	assert.Equal(t, 1, state.Winner)
	assert.Equal(t, []rules.Payout{{Kind: "envido", Team: 1, Points: 2}}, state.Payouts)

	state = privoHand(t, 12, [2]int{11, 11}, true,
		[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}},
		[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}},
	)
	assert.Equal(t, 1, state.Winner)
	assert.Equal(t, rules.MatchOver, state.Phase)
	assert.Equal(t, []rules.Payout{{Kind: "flor", Team: 1, Points: 3}}, state.Payouts)
}

func TestPrivoEnvidoCanEndMatchBeforeRetainedOpposingFlor(t *testing.T) {
	state := privoHand(t, 12, [2]int{11, 10}, true,
		[3]rules.Card{{rules.Spades, 11}, {rules.Coins, 7}, {rules.Cups, 2}},
		[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}},
	)
	state = apply(t, state, 1, rules.Command{Type: "respond", Kind: "accept"})
	assert.True(t, state.Players[1].FlorValid)
	state = apply(t, state, 1, rules.Command{Type: "call", Kind: "truco", Declaration: "con_mi_flor"})
	state = apply(t, state, 0, rules.Command{Type: "respond", Kind: "decline"})
	assert.Equal(t, 0, state.Winner)
	assert.Equal(t, 10, state.Scores[1])
	assert.Equal(t, []rules.Payout{{Kind: "envido", Team: 0, Points: 2}}, state.Payouts)
}

func TestPrivoRecetaCountsAsFlorAndPresentationAwardsOnlyThree(t *testing.T) {
	plain := [3]rules.Card{{rules.Clubs, 1}, {rules.Clubs, 6}, {rules.Coins, 2}}
	receta := [3]rules.Card{{rules.Spades, 11}, {rules.Spades, 10}, {rules.Coins, 7}}
	state := privoHand(t, 12, [2]int{0, 11}, true, plain, receta)
	assert.Equal(t, 1, state.Winner)
	assert.Equal(t, rules.MatchOver, state.Phase)
	state = privoHand(t, 12, [2]int{11, 0}, true, plain, receta)
	state = apply(t, state, 1, rules.Command{Type: "respond", Kind: "present_flor"})
	assert.Equal(t, [2]int{11, 3}, state.Scores)
	assert.Equal(t, []rules.Payout{{Kind: "flor", Team: 1, Points: 3}}, state.Payouts)

	state = privoHand(t, 12, [2]int{11, 0}, true,
		[3]rules.Card{{rules.Clubs, 1}, {rules.Clubs, 6}, {rules.Spades, 3}},
		[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}},
		[3]rules.Card{{rules.Spades, 7}, {rules.Spades, 5}, {rules.Clubs, 2}},
		[3]rules.Card{{rules.Coins, 7}, {rules.Coins, 6}, {rules.Coins, 5}},
	)
	state = apply(t, state, 3, rules.Command{Type: "respond", Kind: "present_flor"})
	assert.Equal(t, [2]int{11, 3}, state.Scores, "presentation pays three for the hand, not six for two flores")
	assert.Equal(t, rules.HandOver, state.Phase)
}

func TestPrivoActivatesOnDealOnlyAndNextHandResetsNegotiation(t *testing.T) {
	state := privoHand(t, 12, [2]int{10, 0}, false,
		[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Coins, 2}},
		[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 4}, {rules.Coins, 3}},
	)
	assert.Empty(t, rules.ViewFor(state, -1, true).PrivoEnvido)
	state = apply(t, state, 1, rules.Command{Type: "play", Card: 0})
	state = apply(t, state, 0, rules.Command{Type: "call", Kind: "truco"})
	state = apply(t, state, 1, rules.Command{Type: "respond", Kind: "decline"})
	assert.Equal(t, 11, state.Scores[0])
	assert.Equal(t, [2]bool{}, state.Privando, "scoring does not reopen the completed hand")
	next, err := rules.StartHand(state, rules.Deck())
	require.NoError(t, err)
	assert.False(t, next.EnvidoAccepted)
	assert.False(t, next.EnvidoClosed)
	assert.Empty(t, next.Envido.Kind)
	next = apply(t, next, next.Turn, rules.Command{Type: "cut", Cut: 1})
	assert.True(t, next.Privando[0])
}
