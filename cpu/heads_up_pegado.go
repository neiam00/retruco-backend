package cpu

import "github.com/neiam00/retruco-backend/rules"

// Once the rival has stacked, their first card and top are public. Exactly one
// opposing card remains unknown. Compare both legal orders against every
// possible bottom before choosing either card or pricing another bet.
type stackOutcomes struct {
	total, best int
	byCard      [3]int
	plan        [3]bool
}

func (o stackOutcomes) dominated(c rules.Command) bool {
	if o.total == 0 {
		return false
	}
	switch c.Type {
	case "burn":
		return o.best > 0
	case "play", "stack":
		return c.Card >= 0 && c.Card < 3 && o.plan[c.Card] && o.byCard[c.Card] < o.best
	}
	return false
}

func publicRank(p rules.PublicPlay, vira rules.Card) (int, bool) {
	if !p.Present || p.Concealed {
		return 0, false
	}
	if p.Burned {
		return 0, true
	}
	if p.Card == nil || !p.Card.Valid() {
		return 0, false
	}
	return rules.Strength(*p.Card, vira), true
}

func againstShownStack(v rules.View, seat int) stackOutcomes {
	if v.Round < 1 || v.Config.Players != 2 {
		return stackOutcomes{}
	}
	var plays [2][3]rules.PublicPlay
	for _, p := range v.Seats {
		if p.Seat < 0 || p.Seat > 1 {
			return stackOutcomes{}
		}
		plays[p.Seat] = p.Plays
	}
	rival, own := plays[1-seat], plays[seat]
	top, ok := publicRank(rival[2], *v.Vira)
	if !ok || !rival[1].Concealed || !rival[1].Present {
		return stackOutcomes{}
	}
	oursFirst, oursKnown := publicRank(own[0], *v.Vira)
	theirsFirst, theirsKnown := publicRank(rival[0], *v.Vira)
	if !oursKnown || !theirsKnown || (oursFirst == theirsFirst) != v.MandatoryStack {
		return stackOutcomes{}
	}
	deal, ok := dealFromView(v, seat)
	if !ok || deal.available == 0 {
		return stackOutcomes{}
	}
	type remaining struct{ index, rank int }
	var cards []remaining
	for _, h := range v.Hand {
		if h.Index < 0 || h.Index >= 3 || !h.Card.Valid() {
			return stackOutcomes{}
		}
		if !h.Used {
			cards = append(cards, remaining{h.Index, rules.Strength(h.Card, *v.Vira)})
		}
	}
	committed, secondKnown := publicRank(own[1], *v.Vira)
	if !(v.Round == 1 && len(cards) == 2 && !own[1].Present ||
		v.Round == 2 && len(cards) == 1 && secondKnown && !v.MandatoryStack) {
		return stackOutcomes{}
	}
	result := stackOutcomes{total: deal.available}
	for i, c := range cards {
		ourSecond, ourThird := committed, c.rank
		if len(cards) == 2 {
			ourSecond, ourThird = c.rank, cards[1-i].rank
		}
		result.plan[c.index] = true
		for _, hidden := range deal.pool[:deal.available] {
			bottom := rules.Strength(hidden, *v.Vira)
			won := false
			switch {
			case v.MandatoryStack:
				won = ourThird > top || ourThird == top && (ourSecond > bottom || ourSecond == bottom && seat == v.Mano)
			case ourSecond == bottom:
				won = oursFirst > theirsFirst
			case ourSecond > bottom:
				// A beaten voluntary bottom burns its visible top.
				won = oursFirst > theirsFirst || ourThird > 0
			default:
				won = oursFirst > theirsFirst && ourThird >= top
			}
			if won {
				result.byCard[c.index]++
			}
		}
		result.best = max(result.best, result.byCard[c.index])
	}
	return result
}
