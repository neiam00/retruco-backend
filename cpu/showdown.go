package cpu

import (
	"math/rand/v2"

	"github.com/neiam00/retruco-backend/rules"
)

// Deterministic shared-deal estimates keep the candidate order and CPU work
// token stable on reconnect. These are hypothetical deals, never the live deck.
const dealSamples = 192

type showdownOdds struct {
	opening                                                      openingOutcomes
	envidoDeals, ordinaryDeals, contestedDeals                   int
	ordinaryWins, leyWins, leyOpeningWins, florWins, allFlorWins int
	trucoWins, openingValue                                      int
	envidoKnown, trucoKnown                                      bool
	cardWins                                                     [3]int
	cardDisagreed                                                [3][3]int
	cardKnown                                                    bool
}

type publicDeal struct {
	hands          [4][3]rules.Card
	known          [4]int
	pool           [40]rules.Card
	available      int
	plays          [3][4]int // zero means the seat has not played
	ordinaryTricks bool
}

func dealFromView(v rules.View, seat int) (publicDeal, bool) {
	d := publicDeal{ordinaryTricks: !v.MandatoryStack}
	if !v.Vira.Valid() || v.Mano < 0 || v.Mano >= v.Config.Players {
		return d, false
	}
	owner := map[rules.Card]int{*v.Vira: -1}
	add := func(s int, c rules.Card) bool {
		if other, exists := owner[c]; exists {
			return other == s
		}
		if !c.Valid() || d.known[s] == 3 {
			return false
		}
		d.hands[s][d.known[s]] = c
		d.known[s]++
		owner[c] = s
		return true
	}
	for _, c := range v.Hand {
		if !add(seat, c.Card) {
			return d, false
		}
	}
	if d.known[seat] != 3 {
		return d, false
	}
	var seen [4]bool
	for _, s := range v.Seats {
		if s.Seat < 0 || s.Seat >= v.Config.Players || seen[s.Seat] || s.Team != s.Seat%2 {
			return d, false
		}
		seen[s.Seat] = true
		for round, p := range s.Plays {
			if !p.Present {
				continue
			}
			if round > v.Round || p.Burned || p.Concealed || p.Card == nil {
				d.ordinaryTricks = false
			}
			if p.Burned || p.Concealed || p.Card == nil {
				continue
			}
			if !add(s.Seat, *p.Card) {
				return d, false
			}
			d.plays[round][s.Seat] = rules.Strength(*p.Card, *v.Vira)
		}
	}
	if v.Config.Players == 4 && v.Preview != nil && !add((seat+2)%4, *v.Preview) {
		return d, false
	}
	for _, c := range rules.Deck() {
		if _, used := owner[c]; !used {
			d.pool[d.available] = c
			d.available++
		}
	}
	return d, true
}

func headsUpLead(v rules.View, seat int) bool {
	if v.Config.Players != 2 || v.Round != 0 || v.Pending != nil || v.Turn != seat {
		return false
	}
	for _, s := range v.Seats {
		if s.Plays[0].Present {
			return false
		}
	}
	return true
}

func sampleShowdowns(v rules.View, seat int, revised, priceOpening, priceHeadsUp, priceMatch bool) showdownOdds {
	d, ok := dealFromView(v, seat)
	if !ok {
		return showdownOdds{}
	}
	n := v.Config.Players
	team := seat % 2
	needTricks := false
	needEnvido := false
	for _, c := range v.Legal {
		kind := c.Kind
		if c.Type == "respond" && v.Pending != nil {
			kind = v.Pending.Kind
		}
		switch kind {
		case "truco":
			needTricks = true
		case "envido", "ley_envido", "flor":
			needEnvido = true
		}
	}
	// A disputed hidden pegado or burned card needs a different play model.
	// Abstain from speculative all-ins in such positions.
	var cardRanks [3]int
	planCards := d.ordinaryTricks && (canPlanLead(v, seat) || revised && headsUpLead(v, seat))
	if planCards {
		for _, card := range v.Hand {
			if card.Index < 0 || card.Index >= 3 {
				return showdownOdds{}
			}
			if !card.Used {
				cardRanks[card.Index] = rules.Strength(card.Card, *v.Vira)
			}
		}
	}
	odds := showdownOdds{envidoKnown: needEnvido, trucoKnown: d.ordinaryTricks && needTricks, cardKnown: planCards}
	measureOpening := priceMatch && n == 2 && v.TrucoValue == 1 && v.Pending == nil && v.Mano == seat && headsUpLead(v, seat) && d.known[1-seat] == 0
	if measureOpening {
		odds.opening.setWeights(d, *v.Vira)
	}
	var declared [4]bool
	if priceOpening && needEnvido {
		for _, player := range v.Seats {
			declared[player.Seat] = player.FlorDeclared
		}
	}
	rng := rand.New(rand.NewPCG(39827, 87401)) // #nosec G404 -- reproducible public-information analysis.
	for range dealSamples {
		hands, pool, left := d.hands, d.pool, d.available
		best := [3]int{-1, -1, -1}
		winners := [3]int{-1, -1, -1}
		var ranks [4][3]int
		var blocksOrdinary [2]bool
		rivalValue, acceptance := 0, 0
		consistent := true
		for i := 0; i < n; i++ {
			s := (v.Mano + i) % n
			for slot := d.known[s]; slot < 3; slot++ {
				j := rng.IntN(left)
				left--
				hands[s][slot], pool[j] = pool[j], pool[left]
			}
			if needEnvido {
				holding := *v.Holding
				if s != seat {
					holding, _ = rules.Evaluate(hands[s], *v.Vira)
				}
				// A publicly declared flor cannot have been dealt a florless hand.
				if declared[s] && holding.Flor == 0 {
					consistent = false
				}
				if s%2 != team {
					rivalValue = max(rivalValue, holding.Envido)
					if priceOpening {
						// Either opposing seat can answer. Stronger pairs are more
						// likely to accept; even a very weak hand can bluff.
						acceptance += min(9, max(1, holding.Envido-22))
					}
				}
				hasFlor := holding.Flor > 0 && (s != seat || v.FlorValid)
				values := [3]int{holding.Envido, holding.Envido, -1}
				if hasFlor {
					values[1], values[2] = holding.Flor, holding.Flor
					if !holding.Receta {
						blocksOrdinary[s%2] = true
					}
				}
				for k, val := range values {
					// Strict comparison in mano order implements envido tie priority.
					if val > best[k] {
						best[k], winners[k] = val, s%2
					}
				}
			}
			if odds.trucoKnown || planCards {
				for k, c := range hands[s] {
					ranks[s][k] = rules.Strength(c, *v.Vira)
				}
			}
		}
		weight := 1
		if needEnvido && v.Pending != nil && v.Pending.Kind != "truco" {
			// A voluntary opposing bid is evidence of a pair. Still give
			// weak opposing hands nonzero weight to allow for bluffing.
			switch {
			case v.Pending.Kind == "envido" && n == 2 && blocksOrdinary[1-team]:
				weight = 0 // A single opponent with ordinary flor cannot open plain envido.
			case priceHeadsUp && v.Pending.Kind == "envido" && v.Pending.Previous >= 2 && rivalValue >= 30:
				// An expensive counterraise after our bid conveys much more
				// strength than an unsolicited two-point opening.
				weight = 4 * min(6, max(1, v.Pending.Value-v.Pending.Previous))
			case priceHeadsUp && v.Pending.Kind == "envido" && v.Pending.Value == 2 && v.Pending.Previous == 1:
				weight = 1 + min(2, max(0, rivalValue-22)/4)
			case rivalValue >= 30:
				weight = 4
			case rivalValue >= 25:
				weight = 2
			}
		}
		if !consistent {
			weight = 0
		}
		odds.envidoDeals += weight
		if !blocksOrdinary[1-team] {
			// An opposing ordinary flor cancels plain envido. A cancellation
			// contributes neither an accepted win nor an accepted loss.
			odds.ordinaryDeals += weight
			if priceOpening && consistent {
				// Two possible responders, each with a probability in tenths.
				// An opposing ordinary flor can void the bid and earns no credit.
				stake := -min(2, v.Config.Target-v.Scores[1-team])
				if winners[0] == team {
					stake = min(2, v.Config.Target-v.Scores[team])
				}
				odds.openingValue += (20 - acceptance) + acceptance*stake
			}
			if winners[0] == team {
				odds.ordinaryWins += weight
			}
		}
		if winners[1] == team {
			odds.leyWins += weight
		}
		if winners[2] == team {
			odds.allFlorWins += weight
		}
		leyWinner := winners[1]
		if blocksOrdinary[1-team] {
			// A rival with flor can choose Con ley. A dedicated flor offer
			// earns no envido from Sin ley, no quiero; assess the contest itself.
			leyWinner = winners[2]
			odds.contestedDeals += weight
			if winners[2] == team {
				odds.florWins += weight
			}
		}
		if leyWinner == team {
			odds.leyOpeningWins += weight
		}
		if planCards {
			var won [3]bool
			for slot, rank := range cardRanks {
				if rank == 0 {
					continue
				}
				plays := d.plays
				plays[v.Round][seat] = rank
				won[slot] = playShowdown(plays, ranks, n, v.Mano) == team
				if won[slot] {
					odds.cardWins[slot]++
				}
			}
			for i := range won {
				for j := i + 1; j < len(won); j++ {
					if won[i] != won[j] {
						odds.cardDisagreed[i][j]++
					}
				}
			}
		}
		if odds.trucoKnown {
			won := playShowdown(d.plays, ranks, n, v.Mano) == team
			if measureOpening {
				odds.opening.record(ranks[1-seat], won)
			}
			if won {
				odds.trucoWins++
			}
		}
	}
	return odds
}

// Every sampled seat follows the same simple card policy in an ordinary hand.
// Take the lead with strength, cover with the cheapest sufficient card, and
// discard low under a winning teammate. The forecast includes first empardes.
func playShowdown(plays [3][4]int, hands [4][3]int, n, mano int) int {
	lead, firstTeam := mano, -1
	for round := range 3 {
		for i := 0; i < n; i++ {
			s := (lead + i) % n
			card := plays[round][s]
			if card == 0 {
				card = chooseRank(hands[s], plays[round], n, s, firstTeam)
				plays[round][s] = card
			}
			for slot, rank := range hands[s] {
				if rank == card {
					hands[s][slot] = 0
					break
				}
			}
		}
		winner := rules.TrickWinner(plays[round], n, lead)
		if round == 0 {
			if winner < 0 {
				var top, bottom [4]int
				for s := 0; s < n; s++ {
					for _, rank := range hands[s] {
						if rank > top[s] {
							bottom[s], top[s] = top[s], rank
						} else if rank > bottom[s] {
							bottom[s] = rank
						}
					}
				}
				for _, trick := range [2][4]int{top, bottom} {
					if w := rules.TrickWinner(trick, n, mano); w >= 0 {
						return w % 2
					}
				}
				return mano % 2
			}
			firstTeam = winner % 2
		} else if winner < 0 || winner%2 == firstTeam {
			return firstTeam
		} else if round == 2 {
			return winner % 2
		}
		lead = winner
	}
	return firstTeam
}

func chooseRank(hand [3]int, trick [4]int, n, seat, firstTeam int) int {
	ally, rival := 0, 0
	for s := 0; s < n; s++ {
		if s%2 == seat%2 {
			ally = max(ally, trick[s])
		} else {
			rival = max(rival, trick[s])
		}
	}
	low, high, cover := 18, 0, 18
	for _, rank := range hand {
		if rank == 0 {
			continue
		}
		low = min(low, rank)
		high = max(high, rank)
		if rank > rival || rank == rival && firstTeam == seat%2 {
			cover = min(cover, rank)
		}
	}
	if ally == 0 && rival == 0 {
		return high
	}
	if ally > rival || cover == 18 {
		return low
	}
	return cover
}
