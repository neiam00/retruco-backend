package cpu

import "github.com/neiam00/retruco-backend/rules"

// The score transitions approximate how a heads-up match continues after the
// opening hand. Weights are rounded frequencies from 3,200 ordinary development
// matches against the native CPU, excluding hands starting within three of target.
// Payouts of six or more share a bucket. No release deals enter the estimate.
var headsUpMatchEquity = func() [25][25]float64 {
	payouts := [...]struct{ us, them, weight int }{
		{0, 1, 13}, {0, 2, 3}, {0, 3, 6}, {0, 4, 3}, {0, 5, 1}, {0, 6, 2},
		{1, 0, 10}, {1, 1, 12}, {1, 2, 3}, {1, 3, 3}, {2, 0, 10}, {2, 1, 1},
		{3, 0, 12}, {3, 1, 2}, {3, 2, 1}, {3, 3, 2}, {4, 0, 10}, {5, 0, 1}, {6, 0, 3},
	}
	var equity [25][25]float64
	// A coarse simultaneous payout loses the ordering at double match point.
	equity[0][0] = 0.5
	for j := 1; j < len(equity); j++ {
		equity[0][j] = 1
	}
	total := 0
	for _, p := range payouts {
		total += p.weight
	}
	for i := 1; i < len(equity); i++ {
		for j := 1; j < len(equity); j++ {
			for _, p := range payouts {
				equity[i][j] += float64(p.weight) * equity[max(0, i-p.us)][max(0, j-p.them)]
			}
			equity[i][j] /= float64(total)
		}
	}
	return equity
}()

// Hands with zero, one, or multiple high cards are increasingly likely to
// decline, accept, or counterraise a weak opening. Their card strength also
// affects the outcome of playing quietly. Price both lines on the same deals.
type openingOutcomes struct {
	weight      [3]float64
	deals, wins [3]int
}

func (o *openingOutcomes) setWeights(d publicDeal, vira rules.Card) {
	high := 0
	for _, card := range d.pool[:d.available] {
		if rules.Strength(card, vira) >= 12 {
			high++
		}
	}
	n := d.available
	if n < 3 {
		return
	}
	low := n - high
	den := float64(n * (n - 1) * (n - 2))
	o.weight[0] = float64(low*(low-1)*(low-2)) / den
	o.weight[1] = float64(3*high*low*(low-1)) / den
	o.weight[2] = 1 - o.weight[0] - o.weight[1]
}

func (o *openingOutcomes) record(ranks [3]int, won bool) {
	n := 0
	for _, rank := range ranks {
		if rank >= 12 {
			n++
		}
	}
	k := min(n, 2)
	o.deals[k]++
	if won {
		o.wins[k]++
	}
}

func (o *openingOutcomes) worthwhile(us, them int) bool {
	if us <= 3 || them <= 3 || us >= len(headsUpMatchEquity) || them >= len(headsUpMatchEquity) {
		return false
	}
	e := &headsUpMatchEquity
	winOne, loseOne := e[us-1][them], e[us][them-1]
	winThree, loseThree := e[us-3][them], e[us][them-3]
	var gain float64
	for k, weight := range o.weight {
		if weight <= 0 {
			continue
		}
		if o.deals[k] == 0 {
			return false
		}
		p := float64(o.wins[k]) / float64(o.deals[k])
		quiet := p*winOne + (1-p)*loseOne
		accepted := p*winThree + (1-p)*loseThree
		after := accepted
		switch k {
		case 0:
			// Allow for a defender calling light.
			after = .95*winOne + .05*accepted
		case 2:
			// Protect against a six-point counterraise. Folding costs three.
			after = loseThree
		}
		gain += weight * (after - quiet)
	}
	return gain > 0.0025
}
