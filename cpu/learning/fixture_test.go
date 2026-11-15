package learning_test

import (
	"github.com/neiam00/retruco-backend/cpu/learning"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestWeakOpeningChangesHiddenDealsButNotTheActorsInformation(t *testing.T) {
	config := rules.Config{Players: 2, Target: 12}
	first, err := learning.NewWeakOpening(config, 1)
	require.NoError(t, err)
	observation := first.Observe()
	assert.Equal(t, 0, observation.Seat)
	assert.Equal(t, "game", observation.Kind)
	for _, card := range first.State.Players[0].Cards {
		assert.Less(t, rules.Strength(card, first.State.Vira), 12)
	}
	hidden := map[[3]rules.Card]bool{}
	for seed := uint64(1); seed <= 64; seed++ {
		environment, err := learning.NewWeakOpening(config, seed)
		require.NoError(t, err)
		current := environment.Observe()
		assert.Equal(t, observation.Features, current.Features)
		assert.Equal(t, observation.Actions, current.Actions)
		assert.Equal(t, observation.Legal, current.Legal)
		assert.Equal(t, observation.View, current.View)
		hidden[environment.State.Players[1].Cards] = true
	}
	assert.Greater(t, len(hidden), 50, "the fixture must actually vary unknown deals")
}
