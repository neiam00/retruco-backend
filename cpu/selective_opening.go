package cpu

import (
	"slices"

	"github.com/neiam00/retruco-backend/rules"
)

// profitableOpening estimates the difference in match equity after calling or
// leading quietly. It runs only for already bounded v11 weak openings. The
// ridge coefficients were fit to paired development continuations of 3,255
// public positions against the native CPU. The old 4-5-6 veto runs first.
func profitableOpening(v rules.View, seat int) bool {
	var ranks [3]int
	for i, card := range v.Hand {
		ranks[i] = rules.Strength(card.Card, *v.Vira)
	}
	slices.Sort(ranks[:])
	low, middle, high := float64(ranks[0])/17, float64(ranks[1])/17, float64(ranks[2])/17
	gain := 0.007895895 + 0.000396729*high - 0.026976859*middle - 0.122255335*low
	if ranks[2] >= 12 {
		gain -= 0.005962188
	}
	if ranks[2] >= 15 {
		gain += 0.010087118
	}
	envido := v.Holding.Envido
	gain -= 0.005231454 * float64(envido) / 40
	if envido >= 28 {
		gain += 0.004020095
	}
	if envido >= 32 {
		gain += 0.005797818
	}
	theirs := v.Scores[1-seat]
	target := float64(v.Config.Target)
	gain += 0.033664923*float64(theirs-v.Scores[seat])/target + 0.005754543*float64(v.Config.Target-theirs)/target
	if v.Config.Target == 12 {
		gain += 0.006238466
	}
	if v.Config.FlorRequired {
		gain += 0.003041910
	}
	return gain >= 0.01
}
