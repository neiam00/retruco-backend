package learning_test

import (
	"github.com/neiam00/retruco-backend/cpu/learning"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestReplayAndSignalProgress(t *testing.T) {
	for _, players := range []int{2, 4} {
		for _, target := range []int{12, 24} {
			for _, flor := range []bool{false, true} {
				config := rules.Config{Players: players, Target: target, FlorRequired: flor}
				roles := [4]bool{true, true, true, true}
				a, err := learning.NewEnvironment(config, 42, roles)
				require.NoError(t, err)
				b, err := learning.NewEnvironment(config, 42, roles)
				require.NoError(t, err)
				for step := 0; step < 10000; step++ {
					o := a.Observe()
					require.Equal(t, o, b.Observe())
					if o.Done {
						assert.GreaterOrEqual(t, a.State.Scores[o.Winner], target)
						require.NotNil(t, o.FinalHand)
						assert.Equal(t, o.Scores, o.FinalHand.ScoresAfter)
						before := o.FinalHand.ScoresBefore
						for _, payout := range o.FinalHand.Payouts {
							before[payout.Team] += payout.Points
						}
						assert.Equal(t, o.Scores, before, "include the final hand in the payout audit")
						assert.Nil(t, o.View)
						break
					}
					index := o.Baseline
					if o.Kind == "signal" {
						index = step % 6
					}
					require.NoError(t, a.Step(index))
					require.NoError(t, b.Step(index))
				}
				require.True(t, a.Observe().Done)
			}
		}
	}
}
func TestVisibilityAndConsumption(t *testing.T) {
	var c learning.Communication
	c.StartHand(1)
	require.True(t, c.Consume(0, 0, 1, 4, [4]bool{}))
	assert.Len(t, c.For(0), 1)
	assert.Len(t, c.For(2), 1)
	assert.Empty(t, c.For(1))
	assert.Empty(t, c.For(3))
	assert.False(t, c.Consume(0, 0, 2, 4, [4]bool{}))
	require.True(t, c.Consume(1, 0, 0, 4, [4]bool{}))
	assert.Len(t, c.Signals, 1)
	c.StartHand(2)
	assert.Empty(t, c.Signals)
	assert.True(t, c.Consume(0, 0, 5, 4, [4]bool{}))
}
func TestHiddenHandsCannotChangeEncoding(t *testing.T) {
	e, err := learning.NewEnvironment(rules.Config{Players: 4, Target: 12}, 7, [4]bool{true, true, true, true})
	require.NoError(t, err)
	require.NoError(t, e.Step(0))
	before := rules.ViewFor(e.State, 0, false)
	copyState := e.State
	copyState.Players[1].Cards, copyState.Players[3].Cards = copyState.Players[3].Cards, copyState.Players[1].Cards
	context := learning.Context{Complete: true}
	assert.Equal(t, learning.Encode(before, 0, context), learning.Encode(rules.ViewFor(copyState, 0, false), 0, context))
	before.Revision += 100
	before.HandNumber += 20
	assert.Equal(t, learning.Encode(rules.ViewFor(e.State, 0, false), 0, context), learning.Encode(before, 0, context), "transport counters must not be features")
}
