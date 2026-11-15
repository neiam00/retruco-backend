package learning_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/cpu/learning"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHeadsUpHasNoTeammateSignalOpportunity(t *testing.T) {
	vira := rules.Card{Suit: rules.Clubs, Number: 5}
	perico, perica := rules.Pieces(vira)
	view := rules.View{Config: rules.Config{Players: 2, Target: 24}, Phase: rules.Playing,
		HandNumber: 1, Vira: &vira, Hand: []rules.HandCard{{Card: perico}, {Card: perica}}}
	var communication learning.Communication
	communication.StartHand(1)
	assert.Equal(t, -1, communication.Offer(view, [4]bool{true, true}))
	assert.Equal(t, [6]bool{true}, learning.TruthfulSignals(view))
	for kind := range learning.SignalKinds {
		before := communication
		assert.False(t, communication.Consume(1, 0, kind, 2, [4]bool{true, true}))
		assert.Equal(t, before, communication)
	}
	view.Config.Players = 4
	assert.Equal(t, 1, communication.Offer(view, [4]bool{false, true}))
	assert.True(t, learning.TruthfulSignal(view, "receta"), "2v2 communication must remain available")
}

func TestHeadsUpTrainingHasNoSignalDecisions(t *testing.T) {
	for _, roles := range [][4]bool{{true, true}, {false, false}} {
		env, err := learning.NewEnvironment(rules.Config{Players: 2, Target: 12}, 100120, roles)
		require.NoError(t, err)
		env.TruthfulCPU = true
		for steps := 0; steps < 10000; steps++ {
			observation := env.Observe()
			if observation.Done {
				break
			}
			require.Equal(t, "game", observation.Kind, "neither learner nor human proxy has a teammate in 1v1")
			require.NoError(t, env.Step(observation.Baseline))
		}
		require.True(t, env.Observe().Done)
		assert.Empty(t, env.Communication.Signals)
	}
}
