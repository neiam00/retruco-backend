package learning

import (
	"fmt"
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTrainingHistoryMatchesAuthorizedActionView(t *testing.T) {
	for _, players := range []int{2, 4} {
		t.Run(fmt.Sprintf("players=%d", players), func(t *testing.T) {
			e, err := NewEnvironment(rules.Config{Players: players, Target: 24, FlorRequired: true}, 317, [4]bool{true, true, true, true})
			require.NoError(t, err)
			e.TruthfulCPU = true
			e.EnableTraining()
			decisions, claims := 0, 0
			for steps := 0; steps < 200; steps++ {
				o := e.Observe()
				if o.Done {
					break
				}
				for seat := 0; seat < players; seat++ {
					view := rules.ViewFor(e.State, seat, false)
					context := Context{Cantos: e.Cantos, Signals: e.training.visible[seat], CPU: e.CPU, Complete: true,
						SignalCounts: e.Communication.CountsFor(seat), SignalsComplete: true}
					assert.Equal(t, Encode(view, seat, context), e.TrainingFeatures(seat))
				}
				action := o.Baseline
				if o.Kind == "signal" {
					action = 0
					for index, legal := range o.SignalMask {
						if index > 0 && legal {
							action = index
							claims++
							break
						}
					}
				} else {
					decisions++
				}
				require.NoError(t, e.Step(action))
			}
			require.Positive(t, decisions)
			if players == 4 {
				require.Positive(t, claims)
			} else {
				require.Zero(t, claims, "heads-up training must not reward or record teammate signals")
			}
		})
	}
}
