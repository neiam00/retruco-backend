package rules_test

import (
	"math/rand/v2"
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLegalPlayAlwaysReachesAMatchResult(t *testing.T) {
	// #nosec G404 -- Reproducible test inputs only; the online host uses crypto/rand.
	random := rand.New(rand.NewPCG(42, 19))
	for trial := range 24 {
		players := 2
		if trial%2 == 0 {
			players = 4
		}
		deck := rules.Deck()
		random.Shuffle(len(deck), func(i, j int) { deck[i], deck[j] = deck[j], deck[i] })
		state, err := rules.New(rules.Config{
			Players:      players,
			Target:       12,
			FlorRequired: trial%3 == 0,
		}, random.IntN(players), deck)
		require.NoError(t, err)

		for steps := 0; state.Phase != rules.MatchOver && steps < 2000; steps++ {
			if state.Phase == rules.HandOver {
				random.Shuffle(len(deck), func(i, j int) { deck[i], deck[j] = deck[j], deck[i] })
				state, err = rules.StartHand(state, deck)
				require.NoError(t, err)
				continue
			}

			seat := state.Turn
			if state.Pending != nil {
				seat = 1 - state.Pending.Caller
			}
			actions := rules.LegalActions(state, seat)
			require.NotEmpty(t, actions, "trial %d phase %s round %d has a deadlock", trial, state.Phase, state.Round)
			beforeRevision := state.Revision
			state, _, err = rules.Apply(state, seat, actions[random.IntN(len(actions))])
			require.NoError(t, err)
			assert.Equal(t, beforeRevision+1, state.Revision)
		}
		require.Equal(t, rules.MatchOver, state.Phase, "trial %d never finished", trial)
		assert.GreaterOrEqual(t, state.Scores[state.Winner], state.Config.Target)
	}
}

func TestBothTeammatesNeedALeyToKeepTheirOrdinaryFlores(t *testing.T) {
	for _, firstDeclaration := range []string{"a_ley", "flor"} {
		state := fixture(t, true,
			[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}},
			[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}},
			[3]rules.Card{{rules.Coins, 7}, {rules.Coins, 6}, {rules.Coins, 5}},
			[3]rules.Card{{rules.Spades, 1}, {rules.Coins, 1}, {rules.Cups, 1}},
		)
		state = apply(t, state, 1, rules.Command{Type: "play", Card: 0, Declaration: "flor"})
		state = apply(t, state, 2, rules.Command{Type: "play", Card: 0, Declaration: firstDeclaration})
		state = apply(t, state, 3, rules.Command{Type: "play", Card: 0})
		state = apply(t, state, 0, rules.Command{Type: "play", Card: 0, Declaration: "flor"})
		assert.True(t, state.Players[2].FlorValid)
		assert.Equal(t, firstDeclaration == "a_ley", state.Players[0].FlorValid)
	}
}

func TestOrdinaryCantoTimeoutPreservesFlores(t *testing.T) {
	state := fixture(t, true,
		[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}},
		[3]rules.Card{{rules.Cups, 1}, {rules.Coins, 4}, {rules.Spades, 5}},
	)
	state = apply(t, state, 1, rules.Command{Type: "call", Kind: "truco"})
	state, _, err := rules.Timeout(state, 0)
	require.NoError(t, err)
	assert.True(t, state.Players[0].FlorValid)
	assert.Equal(t, [2]int{3, 1}, state.Scores)
}
