package learning_test

import (
	"fmt"
	"github.com/neiam00/retruco-backend/cpu/learning"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestTruthfulSignalsUseRemainingCardsAndViraSubstitutions(t *testing.T) {
	for _, n := range []int{4, 10, 11} {
		t.Run(fmt.Sprintf("vira_%d", n), func(t *testing.T) {
			vira := rules.Card{Suit: rules.Spades, Number: n}
			perico, perica := rules.Pieces(vira)
			view := rules.View{Config: rules.Config{Players: 4}, Vira: &vira, Hand: []rules.HandCard{{Card: perico}, {Card: perica}, {Card: rules.Card{Suit: rules.Coins, Number: 7}}}}
			assert.Equal(t, [6]bool{true, true, true, false, true, true}, learning.TruthfulSignals(view))
			view.Hand[0].Used = true
			assert.Equal(t, [6]bool{true, false, true, false, true, false}, learning.TruthfulSignals(view))
		})
	}
	assert.Equal(t, [6]bool{true}, learning.TruthfulSignals(rules.View{}))
}

func TestStrictSimulatorRejectsFalseCPUClaimsAtomically(t *testing.T) {
	e, err := learning.NewEnvironment(rules.Config{Players: 4, Target: 12}, 451, [4]bool{true, true, true, true})
	require.NoError(t, err)
	e.TruthfulCPU = true
	require.NoError(t, e.Step(0))
	o := e.Observe()
	require.Equal(t, "signal", o.Kind)
	require.NotNil(t, o.SignalMask)
	illegal := -1
	for i := 1; i < 6; i++ {
		if !o.SignalMask[i] {
			illegal = i
			break
		}
	}
	require.Positive(t, illegal)
	before := e.Communication.Consumed
	state := e.State
	require.ErrorContains(t, e.Step(illegal), "truthful")
	assert.Equal(t, before, e.Communication.Consumed)
	assert.Equal(t, state, e.State)
	e.CPU[o.Seat] = false
	assert.Nil(t, e.Observe().SignalMask, "human proxies may make false claims")
	require.NoError(t, e.Step(illegal))
}
