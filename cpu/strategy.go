package cpu

import (
	"slices"

	"github.com/neiam00/retruco-backend/rules"
)

// PlayingProfile identifies the CPU choices used in live games and paired evaluations.
const PlayingProfile = "value-and-team-v11"
const HeadsUpEnvidoProfile = "value-and-team-v10"
const RetrucoLadderProfile = "value-and-team-v9"
const OpeningDefenseProfile = "value-and-team-v8"
const ReferenceEnvidoProfile = "value-and-team-v7"
const PartnershipProfile = "value-and-team-v6"
const CompulsoryPericoProfile = "value-and-team-v5"
const TeamStackProfile = "value-and-team-v4"
const HeadsUpWagerProfile = "value-and-team-v3"
const HeadsUpCardProfile = "value-and-team-v2"
const OriginalPlayingProfile = "value-and-team-v1"

// PlayingChoices narrows a CPU's game decisions using only its authorized view.
// rules.LegalActions remains the full action list for a human at the same seat.
func PlayingChoices(v rules.View, seat int) []rules.Command {
	return ChoicesForProfile(v, seat, PlayingProfile)
}

// ChoicesForProfile retains a reproducible baseline for old checkpoints and paired evaluations.
func ChoicesForProfile(v rules.View, seat int, profile string) []rules.Command {
	choices, _ := PolicyChoices(v, seat, profile)
	return choices
}

// PolicyChoices also identifies an ordinarily filtered opening that the recurrent
// policy may play at a bounded frequency. Human legal actions are independent.
func PolicyChoices(v rules.View, seat int, profile string) ([]rules.Command, bool) {
	n := v.Config.Players
	if v.Phase != rules.Playing || (n != 2 && n != 4) || seat < 0 || seat >= n || v.Round < 0 || v.Round > 2 ||
		v.Vira == nil || !v.Vira.Valid() || v.Holding == nil || len(v.Hand) != 3 || len(v.Seats) != n || len(v.Legal) < 2 {
		return v.Legal, false
	}
	openingDefense := profile == PlayingProfile || profile == HeadsUpEnvidoProfile || profile == RetrucoLadderProfile || profile == OpeningDefenseProfile
	referenceEnvido := openingDefense || profile == ReferenceEnvidoProfile
	partnershipProfile := profile == PartnershipProfile || referenceEnvido
	pericoProfile := profile == CompulsoryPericoProfile || partnershipProfile
	teamProfile := profile == TeamStackProfile || pericoProfile
	capRaiseAtTarget := profile == HeadsUpWagerProfile || teamProfile
	e := wagerAssessment{view: v, seat: seat, priceOpening: n == 4 && referenceEnvido, revised: profile == HeadsUpCardProfile || capRaiseAtTarget,
		capRaiseAtTarget: capRaiseAtTarget, capTruco: teamProfile && (n == 4 || pericoProfile)}
	e.priceMatch = n == 2 && profile == PlayingProfile
	e.priceHeadsUp = n == 2 && (profile == PlayingProfile || profile == HeadsUpEnvidoProfile)
	e.headsUpDefense = n == 2 && openingDefense
	if e.headsUpDefense {
		e.pegado = againstShownStack(v, seat)
	}
	pericoTop, lastEmptyLey := -1, false
	if pericoProfile {
		pericoTop = compulsoryPerico(v)
		lastEmptyLey = n == 4 && v.Round == 0 && (!v.FlorValid || v.Holding.Receta) &&
			(seat+1)%n == v.Mano
	}
	for _, c := range v.Legal {
		if c.Type == "call" && c.Kind == "envido" && c.Mode == "" {
			e.plainEnvido = true
			break
		}
	}
	discards := wastefulCards(v, seat)
	choices := make([]rules.Command, 0, len(v.Legal))
	for _, c := range v.Legal {
		if e.pegado.dominated(c) {
			continue
		}
		if e.revised && (!headsUpStack(v, seat, c) || emptyLey(v, c)) {
			continue
		}
		if n == 4 && teamProfile && teamStackBurnsTop(v, seat, c) {
			continue
		}
		if pericoTop >= 0 && (c.Type == "burn" || c.Type == "stack" && c.Top != pericoTop) {
			continue
		}
		if lastEmptyLey && c.Type == "play" && c.Declaration == "a_ley" {
			continue
		}
		if !discards[c] && e.allows(c) {
			choices = append(choices, c)
		}
	}
	if canPlanLead(v, seat) || e.revised && headsUpLead(v, seat) {
		choices = e.leadCardChoices(choices)
	}
	if n == 4 && partnershipProfile && len(choices) > 1 {
		choices = e.partnershipChoices(choices)
	}
	if len(choices) == 0 || len(choices) == len(v.Legal) {
		return v.Legal, e.openingBluff && len(choices) != 0
	}
	return choices, e.openingBluff
}

type wagerAssessment struct {
	priceMatch       bool
	priceHeadsUp     bool
	headsUpDefense   bool
	openingBluff     bool
	pegado           stackOutcomes
	capTruco         bool
	capRaiseAtTarget bool
	revised          bool
	view             rules.View
	seat             int
	odds             *showdownOdds
	plainEnvido      bool
	priceOpening     bool
}

func (a *wagerAssessment) estimate() *showdownOdds {
	if a.odds == nil {
		odds := sampleShowdowns(a.view, a.seat, a.revised, a.priceOpening && a.view.Pending == nil, a.priceHeadsUp, a.priceMatch)
		a.odds = &odds
	}
	return a.odds
}

func (a *wagerAssessment) allows(c rules.Command) bool {
	v := a.view
	kind := c.Kind
	if c.Type == "respond" && v.Pending != nil {
		kind = v.Pending.Kind
	}
	if a.revised && kind == "envido" && c.Type == "respond" && securedEnvido(v, a.seat) {
		switch c.Kind {
		case "accept", "decline":
			return false
		case "raise":
			// A larger stake cannot improve the result of winning this match.
			if a.capRaiseAtTarget && c.Amount > v.Config.Target-v.Scores[a.seat%2]-v.Pending.Value {
				return false
			}
		}
	}
	if kind == "truco" {
		if a.priceHeadsUp && v.Pending != nil && (c.Kind == "accept" || c.Kind == "decline") {
			if winner, known := publicTrucoWinner(v); known {
				return (c.Kind == "accept") == (winner == a.seat%2)
			}
			if c.Kind == "decline" && v.Pending.Previous >= v.Config.Target-v.Scores[1-a.seat%2] {
				return false
			}
		}
		if a.exposesWeakEnvido(c) {
			return false
		}
		if a.capTruco && (c.Type == "call" || c.Kind == "raise") &&
			v.TrucoValue >= v.Config.Target-v.Scores[a.seat%2] {
			return false
		}
		if (c.Type == "call" || c.Kind == "raise") && c.Amount == 100 {
			return a.matchRaise()
		}
		if c.Kind == "accept" && v.Pending != nil && v.Pending.Value == 100 {
			// Folding for the previous stake would concede the match.
			if v.Pending.Previous >= v.Config.Target-v.Scores[1-a.seat%2] {
				return true
			}
			if winner, certain := publicTrucoWinner(v); certain {
				return winner == a.seat%2
			}
			if a.headsUpDefense {
				if a.pegado.total > 0 {
					return 100*a.pegado.best >= 75*a.pegado.total
				}
				odds := a.estimate()
				return odds.trucoKnown && odds.trucoWins*100 >= 70*dealSamples
			}
			odds := a.estimate()
			return !odds.trucoKnown || odds.trucoWins*100 >= 60*dealSamples
		}
		if a.headsUpDefense && c.Kind == "decline" && v.Pending != nil && v.Pending.Value == 100 {
			if v.Pending.Previous >= v.Config.Target-v.Scores[1-a.seat%2] ||
				(a.pegado.total > 0 && a.pegado.best == a.pegado.total) {
				return false
			}
		}
		if c.Type == "call" || c.Kind == "raise" {
			if a.pegado.total > 0 {
				// A stack commits both rival cards. Calling again can invite an
				// immediate match raise while its hidden bottom still decides the hand.
				minimum := 65
				if c.Amount >= 9 {
					minimum = 80
				}
				return 100*a.pegado.best >= minimum*a.pegado.total
			}
			odds := a.estimate()
			if !odds.trucoKnown {
				return !a.headsUpDefense
			}
			// Ordinary truco can still be a bluff. More expensive calls
			// need progressively stronger cards at the table.
			minimum := 40
			if c.Amount == 6 {
				minimum = 60
			}
			if c.Amount == 9 {
				minimum = 75
			}
			if odds.trucoWins*100 >= minimum*dealSamples {
				return true
			}
			if a.headsUpDefense && c == (rules.Command{Type: "call", Kind: "truco", Amount: 3}) &&
				v.TrucoValue == 1 && v.Mano == a.seat && !v.FlorValid && headsUpLead(v, a.seat) &&
				v.Config.Target-max(v.Scores[0], v.Scores[1]) > 3 && (!a.priceHeadsUp || v.Scores[a.seat%2] <= v.Scores[1-a.seat%2]) {
				if a.priceMatch && !odds.opening.worthwhile(v.Config.Target-v.Scores[a.seat%2], v.Config.Target-v.Scores[1-a.seat%2]) {
					return false
				}
				a.openingBluff = true
				return true
			}
			return false
		}
		if c.Kind == "accept" && v.Pending != nil {
			if a.pegado.total > 0 {
				return a.worthAccepting(a.pegado.best, a.pegado.total, v.Pending.Value, v.Pending.Previous)
			}
			odds := a.estimate()
			return !odds.trucoKnown || a.worthAccepting(odds.trucoWins, dealSamples, v.Pending.Value, v.Pending.Previous)
		}
		return true
	}
	if kind != "envido" && kind != "ley_envido" && kind != "flor" {
		return true
	}
	// A cheap plain envido can be a steal. Keep the call available with
	// sufficient showdown equity even when the actor learned to fold.
	if a.priceHeadsUp && kind == "envido" && c.Type == "respond" && c.Kind == "decline" && c.Mode == "" {
		odds := a.estimate()
		if odds.ordinaryDeals > 0 && v.Pending != nil && (v.Pending.Previous >= v.Config.Target-v.Scores[1-a.seat%2] ||
			v.Pending.Value == 2 && v.Pending.Previous == 1 && odds.ordinaryWins*100 >= 40*odds.ordinaryDeals) {
			return false
		}
	}
	if c.Type != "call" && c.Kind != "raise" && c.Kind != "falta" && c.Kind != "accept" {
		return true
	}
	// Without flor, plain envido keeps the normal refusal point. Avoid a
	// gratuitous A ley opening that gives the rival a free Sin ley refusal.
	if c.Type == "call" && kind == "ley_envido" && c.Mode == "" && !v.FlorValid && a.plainEnvido {
		return false
	}
	odds := a.estimate()
	if !odds.envidoKnown {
		return true
	}
	wins, deals := odds.ordinaryWins, odds.ordinaryDeals
	if kind != "envido" {
		wins, deals = odds.leyWins, odds.envidoDeals
		if c.Type == "call" {
			wins = odds.leyOpeningWins
		}
		mode := c.Mode
		if v.Pending != nil && v.Pending.Mode != "" {
			mode = v.Pending.Mode
		}
		if kind == "flor" || mode == "con_ley" {
			wins, deals = odds.florWins, odds.contestedDeals
		}
	}
	if c.Kind == "accept" && v.Pending != nil {
		price, fold := v.Pending.Value, v.Pending.Previous
		if v.Pending.Mode == "" && c.Mode == "sin_ley" {
			fold = 0
		}
		if kind == "flor" || c.Mode == "con_ley" || v.Pending.Mode == "con_ley" {
			price += 3
			fold += 3
			// A caller may have said A ley without having a flor. If we can
			// accept con ley, we still win those uncontested showdowns.
			wins, deals = odds.allFlorWins, odds.envidoDeals
			if v.Pending.FlorOffered || kind == "flor" {
				wins, deals = odds.florWins, odds.contestedDeals
			}
		}
		return deals == 0 || a.worthAccepting(wins, deals, price, fold)
	}
	// A voluntary opening needs a favored showdown. Require a larger edge
	// when a rival has already bid or the proposed raise can decide the match.
	if c.Type == "call" {
		if a.priceOpening && v.Pending == nil && kind == "envido" {
			return a.pricedOrdinaryOpening()
		}
		return wins*2 > deals
	}
	opponentNeeds := v.Config.Target - v.Scores[1-a.seat%2]
	if a.priceHeadsUp && kind == "envido" && v.Pending != nil {
		ours := v.Config.Target - v.Scores[a.seat%2]
		// Once accepting already pays enough to finish, offering more only
		// lets a weaker rival decline or a stronger rival charge us more.
		if v.Pending.Value >= ours || c.Kind == "raise" && c.Amount > ours-v.Pending.Value {
			return false
		}
	}
	if deals == 0 {
		return false
	}
	if c.Kind == "falta" || v.Pending != nil && 2*(v.Pending.Value+c.Amount) >= opponentNeeds {
		return wins*100 >= 80*deals
	}
	return wins*100 >= 65*deals
}

func canPlanLead(v rules.View, seat int) bool {
	if v.Config.Players != 4 || v.Pending != nil || v.Turn != seat || v.MandatoryStack || v.Round >= 2 {
		return false
	}
	for _, p := range v.Seats {
		if p.Seat == (seat+2)%4 {
			return !p.Plays[v.Round].Present
		}
	}
	return false
}

// Compare the available plays on identical unseen deals. Defer close calls
// to the actor, including plays that differ only because of sampling noise.
func (a *wagerAssessment) leadCardChoices(choices []rules.Command) []rules.Command {
	var slots [3]bool
	for _, c := range choices {
		if c.Type == "play" && c.Card >= 0 && c.Card < 3 {
			slots[c.Card] = true
		}
	}
	count := 0
	for _, present := range slots {
		if present {
			count++
		}
	}
	if count < 2 {
		return choices
	}
	odds := a.estimate()
	if !odds.cardKnown {
		return choices
	}
	keep := choices[:0]
	// Snapshot the alternatives before compacting the slice.
	best := make(map[rules.Command]int)
	for _, c := range choices {
		if c.Type != "play" || c.Card < 0 || c.Card >= 3 {
			continue
		}
		key := c
		key.Card = 0
		if current, ok := best[key]; !ok || odds.cardWins[c.Card] > odds.cardWins[current] {
			best[key] = c.Card
		}
	}
	for _, c := range choices {
		if c.Type == "play" && c.Card >= 0 && c.Card < 3 {
			key := c
			key.Card = 0
			other := best[key]
			gain := odds.cardWins[other] - odds.cardWins[c.Card]
			diff := odds.cardDisagreed[min(c.Card, other)][max(c.Card, other)]
			if 10*gain >= dealSamples && gain*gain > 4*diff {
				continue
			}
		}
		keep = append(keep, c)
	}
	return keep
}

// Compare accepting the offered stake with the points conceded by declining.
func (a *wagerAssessment) worthAccepting(wins, deals, stake, fold int) bool {
	v := a.view
	gain := min(stake, v.Config.Target-v.Scores[a.seat%2])
	loss := min(stake, v.Config.Target-v.Scores[1-a.seat%2])
	return wins*(gain+loss) >= (loss-fold)*deals
}

func (a *wagerAssessment) matchRaise() bool {
	v := a.view
	previous := v.TrucoValue
	if v.Pending != nil {
		previous = v.Pending.Value
	}
	ours, theirs := v.Config.Target-v.Scores[a.seat%2], v.Config.Target-v.Scores[1-a.seat%2]
	// If the current stake already wins for us, offering more only helps the rival.
	if previous >= ours {
		return false
	}
	atRisk := previous >= theirs
	if previous < 9 && !atRisk {
		return false
	}
	if winner, certain := publicTrucoWinner(v); certain {
		return winner == a.seat%2
	}
	odds := a.estimate()
	if !odds.trucoKnown {
		return false
	}
	if atRisk {
		return odds.trucoWins*100 >= 70*dealSamples
	}
	return odds.trucoWins*100 >= 90*dealSamples
}

// With a partner leading, compare a costly card to the cheapest legal card
// with the same declaration. Keep the costly card selectable whenever it can
// improve the trick against even one possible remaining opposing rank.
func wastefulCards(v rules.View, seat int) map[rules.Command]bool {
	if v.Pending != nil || v.MandatoryStack {
		return nil
	}
	var ranks [4]int
	waiting, others := 0, 0
	for _, player := range v.Seats {
		if player.Seat < 0 || player.Seat >= v.Config.Players || player.Plays[1].Concealed {
			return nil
		}
		p := player.Plays[v.Round]
		if player.Seat != seat && p.Present {
			others++
		}
		if p.Concealed || p.Present && !p.Burned && p.Card == nil {
			return nil
		}
		if p.Present && !p.Burned {
			ranks[player.Seat] = rules.Strength(*p.Card, *v.Vira)
		} else if !p.Present && player.Seat%2 != seat%2 {
			waiting++
		}
	}
	var strength [3]int
	for _, card := range v.Hand {
		if card.Index < 0 || card.Index >= len(strength) {
			return nil
		}
		if !card.Used {
			strength[card.Index] = rules.Strength(card.Card, *v.Vira)
		}
	}
	if others == v.Config.Players-1 {
		return closingDiscards(v, seat, ranks, strength)
	}
	if v.Config.Players != 4 {
		return nil
	}
	partner := ranks[(seat+2)%4]
	rival := max(ranks[(seat+1)%4], ranks[(seat+3)%4])
	if partner == 0 || partner <= rival || waiting > 1 {
		return nil
	}
	cheapest := map[rules.Command]int{}
	for _, c := range v.Legal {
		if (c.Type != "play" && c.Type != "burn") || c.Card < 0 || c.Card >= 3 {
			continue
		}
		key := c
		key.Card = 0
		rank := strength[c.Card]
		if earlier, ok := cheapest[key]; !ok || rank < earlier {
			cheapest[key] = rank
		}
	}
	var avoided map[rules.Command]bool
	for _, c := range v.Legal {
		key := c
		key.Card = 0
		cheap, ok := cheapest[key]
		if !ok || c.Card < 0 || c.Card >= 3 || strength[c.Card] <= cheap {
			continue
		}
		expensive := strength[c.Card]
		helps := false
		if c.Type == "play" {
			highest := 0
			if waiting != 0 {
				highest = 17
			}
			for last := 0; last <= highest; last++ {
				// Ranks not actually available to the rival are included conservatively.
				// A unique card already on the table or in our hand cannot be duplicated.
				if last >= 12 && (last == partner || last == rival || slices.Contains(strength[:], last)) {
					continue
				}
				opponent := max(rival, last)
				if partnerOutcome(v.Round, partner, expensive, opponent) > partnerOutcome(v.Round, partner, cheap, opponent) {
					helps = true
					break
				}
			}
		}
		if !helps {
			if avoided == nil {
				avoided = make(map[rules.Command]bool)
			}
			avoided[c] = true
		}
	}
	return avoided
}

func partnerOutcome(round, partner, ours, rival int) int {
	best := max(partner, ours)
	if best < rival {
		return -1
	}
	if best == rival || (round > 0 && partner == ours) {
		return 0
	}
	return 1
}

// Once everyone else has played, the trick result is known for every choice.
// Keep the cheapest card for each outcome. In later tricks take a certain hand
// win, or keep the hand alive when another play would concede it immediately.
func closingDiscards(v rules.View, seat int, table [4]int, strength [3]int) map[rules.Command]bool {
	n := v.Config.Players
	first := -1
	if v.Round > 0 {
		var initial [4]int
		for _, player := range v.Seats {
			p := player.Plays[0]
			if !p.Present || p.Concealed || p.Card == nil || p.Burned {
				return nil
			}
			initial[player.Seat] = rules.Strength(*p.Card, *v.Vira)
		}
		first = trickWinner(initial, n, v.Mano, true)
		if first < 0 {
			return nil
		}
	}
	type option struct {
		key           rules.Command
		outcome, rank int
	}
	options := make(map[rules.Command]option)
	type price struct {
		key     rules.Command
		outcome int
	}
	cheapest := make(map[price]int)
	best := make(map[rules.Command]int)
	for _, c := range v.Legal {
		if (c.Type != "play" && c.Type != "burn") || c.Card < 0 || c.Card >= 3 {
			continue
		}
		table[seat] = strength[c.Card]
		if c.Type == "burn" {
			table[seat] = 0
		}
		winner := trickWinner(table, n, v.Mano, v.Round == 0)
		result := 0
		if winner < 0 && v.Round > 0 {
			winner = first
		}
		if winner >= 0 {
			result = -1
			if winner%2 == seat%2 {
				result = 1
			}
		}
		key := c
		key.Card = 0
		// Burning the same physical card gives up its visibility. Preserve that
		// choice unless a play can win the hand or avoid immediate defeat.
		cost := price{key, result}
		rank := strength[c.Card]
		if old, ok := cheapest[cost]; !ok || rank < old {
			cheapest[cost] = rank
		}
		options[c] = option{key, result, rank}
		key.Type = "play"
		if old, ok := best[key]; !ok || result > old {
			best[key] = result
		}
	}
	avoid := make(map[rules.Command]bool)
	for c, opt := range options {
		key := opt.key
		key.Type = "play"
		if opt.rank > cheapest[price{opt.key, opt.outcome}] || v.Round > 0 && opt.outcome < best[key] {
			avoid[c] = true
		}
	}
	return avoid
}
