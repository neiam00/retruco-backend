package cpu

import "github.com/neiam00/retruco-backend/rules"

// In heads-up the stacked top cannot survive a losing bottom, and both cards
// are committed before the rival replies. With the first trick already won,
// a bottom that ties or wins already suffices when played alone.
// Prefer the single to concealing an uncertain bottom under a strong top.
func headsUpStack(v rules.View, seat int, c rules.Command) bool {
	if c.Type != "stack" || v.Config.Players != 2 || v.MandatoryStack || v.Round != 1 {
		return true
	}
	var ranks [3]int
	for _, h := range v.Hand {
		if h.Index < 0 || h.Index >= 3 {
			return true
		}
		if !h.Used {
			ranks[h.Index] = rules.Strength(h.Card, *v.Vira)
		}
	}
	if c.Card < 0 || c.Card >= 3 || c.Top < 0 || c.Top >= 3 || ranks[c.Card] == 0 || ranks[c.Top] == 0 {
		return true
	}
	// A single card keeps the normal reply to a played second card or stack.
	for _, p := range v.Seats {
		if p.Seat != seat && p.Plays[1].Present {
			return false
		}
	}
	// When the CPU leads second in an ordinary 1v1 it won first. Conceal the
	// bottom only if no unseen opposing card can beat or tie it.
	bottom := ranks[c.Card]
	for _, card := range rules.Deck() {
		if card == *v.Vira {
			continue
		}
		known := false
		for _, h := range v.Hand {
			if card == h.Card {
				known = true
				break
			}
		}
		for _, s := range v.Seats {
			p := s.Plays[0]
			if p.Present && !p.Burned && !p.Concealed && p.Card != nil && card == *p.Card {
				known = true
			}
		}
		if !known && rules.Strength(card, *v.Vira) >= bottom {
			return false
		}
	}
	return true
}

// CPU remarks and wagers remain legal for humans. An empty A ley with no
// ordinary flor in heads-up cannot preserve a teammate's holding.
func emptyLey(v rules.View, c rules.Command) bool {
	return v.Config.Players == 2 && c.Type == "play" && c.Declaration == "a_ley" &&
		(!v.FlorValid || v.Holding.Receta)
}

// In a negotiated ordinary heads-up envido, the opponent's raise confirms
// the contest. Conditioning on the CPU holding the perico leaves at most a
// perica + seven (36) for the opponent to show, with ties decided by mano.
func securedEnvido(v rules.View, seat int) bool {
	if v.Config.Players != 2 || v.Pending == nil || v.Pending.Kind != "envido" || v.Pending.Previous < 2 ||
		v.Pending.Terminal || v.Pending.Value >= v.Config.Target-v.Scores[seat%2] {
		return false
	}
	perico, _ := rules.Pieces(*v.Vira)
	for _, h := range v.Hand {
		if h.Card == perico {
			return v.Holding.Envido > 36 || (v.Holding.Envido == 36 && v.Mano == seat)
		}
	}
	return false
}
