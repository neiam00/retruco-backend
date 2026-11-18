package cpu

import (
	"math/bits"
	"math/rand/v2"

	"github.com/neiam00/retruco-backend/rules"
)

// Each physical choice is tried on the same hypothetical deals. Every follow-up
// player uses just its own remaining cards and the exposed table. A concealed
// bottom is sampled for settlement, never shown to the players choosing cards.
type trickForecast struct {
	table             [3][4]int
	present           [3][4]bool
	hidden            [4]bool
	stacked           [4]bool
	left              [4][3]int
	round, turn, mano int
	mandatory         bool
}

func forecastTable(v rules.View) trickForecast {
	f := trickForecast{round: v.Round, turn: v.Turn, mano: v.Mano, mandatory: v.MandatoryStack}
	for _, s := range v.Seats {
		for r, p := range s.Plays {
			f.present[r][s.Seat] = p.Present
			if p.Present && p.Card != nil && !p.Burned && !p.Concealed {
				f.table[r][s.Seat] = rules.Strength(*p.Card, *v.Vira)
			}
		}
		f.hidden[s.Seat] = s.Plays[1].Concealed
		f.stacked[s.Seat] = s.Plays[1].Concealed || v.MandatoryStack && s.Plays[2].Present
	}
	return f
}

func removeRank(hand *[3]int, rank int) bool {
	for i, value := range hand {
		if value == rank && value != 0 {
			hand[i] = 0
			return true
		}
	}
	return false
}

// Sample physical cards, matching public plays to their exact slots before assigning
// concealed or burned cards. A teammate preview is already fixed by dealFromView.
func (f trickForecast) withDeal(d publicDeal, v rules.View, seat int, rng *rand.Rand) (trickForecast, bool) {
	hands, pool, left := d.hands, d.pool, d.available
	for _, s := range v.Seats {
		row := &hands[s.Seat]
		for slot := d.known[s.Seat]; slot < 3; slot++ {
			i := rng.IntN(left)
			left--
			row[slot], pool[i] = pool[i], pool[left]
		}
		var used [3]bool
		for _, play := range s.Plays {
			if !play.Present || play.Burned || play.Concealed || play.Card == nil {
				continue
			}
			for i, c := range row {
				if c == *play.Card {
					used[i] = true
				}
			}
		}
		for r, p := range s.Plays {
			if !p.Present || !p.Burned && !p.Concealed {
				continue
			}
			var unknown [3]int
			n := 0
			for slot, c := range row {
				if used[slot] {
					continue
				}
				if s.Seat == seat {
					ownUsed := false
					for _, h := range v.Hand {
						if h.Card == c {
							ownUsed = h.Used
							break
						}
					}
					if !ownUsed {
						continue
					}
				}
				unknown[n] = slot
				n++
			}
			if n == 0 {
				return f, false
			}
			slot := unknown[rng.IntN(n)]
			used[slot] = true
			if p.Concealed && !p.Burned {
				f.table[r][s.Seat] = rules.Strength(row[slot], *v.Vira)
			}
		}
		for slot, c := range row {
			if !used[slot] {
				f.left[s.Seat][slot] = rules.Strength(c, *v.Vira)
			}
		}
	}
	return f, true
}

type trickMove struct {
	kind        string
	bottom, top int
}

func (f *trickForecast) play(s int, m trickMove, r int) {
	f.present[r][s] = true
	if m.kind == "burn" {
		if m.bottom > 0 {
			removeRank(&f.left[s], m.bottom)
		}
		if f.mandatory {
			f.present[2][s] = true
			f.stacked[s] = true
			f.left[s] = [3]int{}
		}
		return
	}
	f.table[r][s] = m.bottom
	removeRank(&f.left[s], m.bottom)
	if m.kind == "stack" {
		f.stacked[s], f.hidden[s], f.present[2][s] = true, true, true
		f.table[2][s] = m.top
		removeRank(&f.left[s], m.top)
	}
}

func (f *trickForecast) finishTrick(round, lead, firstTeam int) {
	for offset := range 4 {
		s := (lead + offset) % 4
		if f.present[round][s] {
			continue
		}
		if f.mandatory {
			m := trickMove{kind: "stack"}
			for _, rank := range f.left[s] {
				if rank > m.top {
					m.bottom, m.top = m.top, rank
				} else if rank > 0 {
					m.bottom = rank
				}
			}
			f.play(s, m, round)
			continue
		}
		visible := f.table[round]
		if round == 1 {
			for i, hidden := range f.hidden {
				if hidden {
					visible[i] = 0
				}
			}
		}
		f.play(s, trickMove{kind: "play", bottom: chooseRank(f.left[s], visible, 4, s, firstTeam)}, round)
	}
}

// Voluntary pegados commit the third card before its hidden bottom is compared.
// The first-trick team wins immediately if no possible top-card burn can overturn
// its visible third. Otherwise only a winning bottom or an allied high tie saves its top.
func (f *trickForecast) resolveStackTops(firstTeam, second int, stacks uint) bool {
	for mask := stacks; ; mask = (mask - 1) & stacks {
		top := f.table[2]
		for s := range 4 {
			if mask&(1<<s) != 0 {
				top[s] = 0
			}
		}
		w := rules.TrickWinner(top, 4, f.mano)
		if w >= 0 && w%2 != firstTeam {
			break
		}
		if mask == 0 {
			return true
		}
	}
	for s, stacked := range f.stacked {
		if stacked && (second < 0 || f.table[1][s] != f.table[1][second]) {
			f.table[2][s] = 0
		}
	}
	return false
}

func (f trickForecast) winnerAfter(own trickMove) int {
	f.play(f.turn, own, f.round)
	next := (f.turn + 1) % 4
	if f.round == 0 {
		f.finishTrick(0, next, -1)
	}
	first := rules.TrickWinner(f.table[0], 4, f.mano)
	f.mandatory = first < 0
	if f.round <= 1 {
		lead := next
		if f.round == 0 {
			lead = first
			if f.mandatory {
				lead = f.mano
			}
		}
		f.finishTrick(1, lead, first%2)
	}
	secondLead := first
	if f.mandatory {
		secondLead = f.mano
	}
	second := rules.TrickWinner(f.table[1], 4, secondLead)
	if f.mandatory {
		if w := rules.TrickWinner(f.table[2], 4, f.mano); w >= 0 {
			return w % 2
		}
		if second >= 0 {
			return second % 2
		}
		return f.mano % 2
	}
	var stacks uint
	for s, b := range f.stacked {
		if b {
			stacks |= 1 << s
		}
	}
	if stacks != 0 {
		lead := next
		if f.round != 2 {
			lead = f.mano
			if bits.OnesCount(stacks) == 1 {
				lead = (bits.TrailingZeros(stacks) + 1) % 4
			}
		}
		f.finishTrick(2, lead, first%2)
		if f.resolveStackTops(first%2, second, stacks) {
			return first % 2
		}
	}
	if second < 0 || second%2 == first%2 {
		return first % 2
	}
	if stacks == 0 {
		lead := second
		if f.round == 2 {
			lead = next
		}
		f.finishTrick(2, lead, first%2)
	}
	third := rules.TrickWinner(f.table[2], 4, f.mano)
	if third < 0 {
		return first % 2
	}
	return third % 2
}

type forecastRow struct {
	move     trickMove
	wins     int
	outcomes [(dealSamples + 63) / 64]uint64
}

// The last opposing seat has three unplayed cards. Count hands whose highest
// card could tie or beat the partner but could still be covered by our card.
func (d *publicDeal) likelyCover(vira rules.Card, partner, cover int) bool {
	belowPartner, upToCover := 0, 0
	for _, card := range d.pool[:d.available] {
		rank := rules.Strength(card, vira)
		if rank < partner {
			belowPartner++
		}
		if rank <= cover {
			upToCover++
		}
	}
	hands := func(n int) int { return n * (n - 1) * (n - 2) / 6 }
	return 3*(hands(upToCover)-hands(belowPartner)) >= hands(d.available)
}

func (a *wagerAssessment) inferiorTeamCard(card, best forecastRow, coverRank int, deal *publicDeal) bool {
	gain := best.wins - card.wins
	if gain <= 0 {
		return false
	}
	cover := coverRank > 0 && card.move.kind == "play" && card.move.bottom > coverRank
	minGainPercent, disagreementFactor := 4, 4
	if a.planTeamPlay {
		minGainPercent, disagreementFactor = 1, 1
		if cover {
			// Preserve insurance against a live last seat. For an unlikely
			// cover, save the card only with a substantial whole-hand advantage.
			if deal.likelyCover(*a.view.Vira, coverRank, card.move.bottom) {
				return false
			}
			minGainPercent = 10
		}
	} else if cover {
		return false
	}
	if gain*100 < minGainPercent*dealSamples {
		return false
	}
	var disagreed int
	for i := range card.outcomes {
		disagreed += bits.OnesCount64(card.outcomes[i] ^ best.outcomes[i])
	}
	return gain*gain > disagreementFactor*disagreed
}

func (a *wagerAssessment) partnershipCards(choices []rules.Command) ([]rules.Command, int, bool) {
	v, seat := a.view, a.seat
	var ranks [3]int
	for _, h := range v.Hand {
		if h.Index < 0 || h.Index >= 3 {
			return choices, 0, false
		}
		if !h.Used {
			ranks[h.Index] = rules.Strength(h.Card, *v.Vira)
		}
	}
	moveFor := func(c rules.Command) trickMove {
		if c.Type != "play" && c.Type != "stack" && c.Type != "burn" || c.Card < 0 || c.Card >= 3 || c.Top < 0 || c.Top >= 3 {
			return trickMove{}
		}
		m := trickMove{kind: c.Type, bottom: ranks[c.Card]}
		if c.Type == "stack" {
			m.top = ranks[c.Top]
		}
		return m
	}
	indices := map[trickMove]int{}
	rows := []forecastRow{}
	for _, c := range choices {
		m := moveFor(c)
		if m.kind == "" {
			continue
		}
		if _, ok := indices[m]; !ok {
			indices[m] = len(rows)
			rows = append(rows, forecastRow{move: m})
		}
	}
	if len(rows) == 0 {
		return choices, 0, false
	}
	d, ok := dealFromView(v, seat)
	if !ok {
		return choices, 0, false
	}
	rng := rand.New(rand.NewPCG(39827, 87401)) // #nosec G404 -- reproducible hypothetical deals.
	base := forecastTable(v)
	if v.Round > 0 {
		for _, present := range base.present[0] {
			if !present {
				return choices, 0, false
			}
		}
		if (rules.TrickWinner(base.table[0], 4, v.Mano) < 0) != v.MandatoryStack {
			return choices, 0, false
		}
	}
	for i := range dealSamples {
		f, ok := base.withDeal(d, v, seat, rng)
		if !ok {
			return choices, 0, false
		}
		for j := range rows {
			if f.winnerAfter(rows[j].move) == seat%2 {
				rows[j].wins++
				rows[j].outcomes[i/64] |= 1 << (i % 64)
			}
		}
	}
	best := 0
	for i := 1; i < len(rows); i++ {
		if rows[i].wins > rows[best].wins {
			best = i
		}
	}
	// A cover over a leading teammate can matter against the last opponent.
	// Give those covers more room for uncertainty in the hand forecast.
	partner := (seat + 2) % 4
	coverRank := 0
	if v.Round == 0 && (!base.present[0][(seat+1)%4] || !base.present[0][(seat+3)%4]) &&
		base.table[0][partner] > max(base.table[0][(seat+1)%4], base.table[0][(seat+3)%4]) {
		coverRank = base.table[0][partner]
	}
	groups := make(map[string]int)
	for _, c := range choices {
		m := moveFor(c)
		if m.kind == "" {
			continue
		}
		i := indices[m]
		old, ok := groups[c.Declaration]
		if !ok || rows[i].wins > rows[old].wins {
			groups[c.Declaration] = i
		}
	}
	kept := choices[:0]
	for _, c := range choices {
		m := moveFor(c)
		if m.kind != "" {
			o := rows[indices[m]]
			top := rows[groups[c.Declaration]]
			if a.inferiorTeamCard(o, top, coverRank, &d) {
				continue
			}
		}
		kept = append(kept, c)
	}
	return kept, rows[best].wins, true
}
