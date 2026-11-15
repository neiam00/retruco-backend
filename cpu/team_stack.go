package cpu

import "github.com/neiam00/retruco-backend/rules"

// In a compulsory pegado the top cards decide first. The perico is unique,
// so showing it wins even when every other card is still hidden.
func compulsoryPerico(v rules.View) int {
	if !v.MandatoryStack || v.Round != 1 {
		return -1
	}
	perico, _ := rules.Pieces(*v.Vira)
	for _, h := range v.Hand {
		if !h.Used && h.Card == perico {
			return h.Index
		}
	}
	return -1
}

// In a voluntary pegado the top burns if another seat beats its bottom,
// even a teammate. An equal teammate bottom can save the top, but an equal
// opponent bottom cannot. A concealed card gives no evidence either way.
func teamStackBurnsTop(v rules.View, seat int, c rules.Command) bool {
	if c.Type != "stack" || v.MandatoryStack || v.Round != 1 {
		return false
	}
	for _, h := range v.Hand {
		if h.Index != c.Card || h.Used {
			continue
		}
		bottom := rules.Strength(h.Card, *v.Vira)
		for _, player := range v.Seats {
			p := player.Plays[1]
			if player.Seat == seat || !p.Present || p.Burned || p.Concealed || p.Card == nil {
				continue
			}
			rank := rules.Strength(*p.Card, *v.Vira)
			if rank > bottom || (rank == bottom && player.Team != seat%2) {
				return true
			}
		}
		break
	}
	return false
}
