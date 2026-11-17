package cpu

import (
	"slices"

	"github.com/neiam00/retruco-backend/rules"
)

const minimumOpeningPair = 24

func (a *wagerAssessment) partnershipChoices(choices []rules.Command) []rules.Command {
	if a.view.Pending != nil || a.view.Turn != a.seat {
		return choices
	}
	choices, wins, known := a.partnershipCards(choices)
	return a.partnershipWagers(choices, wins, known)
}

// Compare voluntary openings only. The partner keeps its own response to a rival.
func (a *wagerAssessment) partnershipWagers(choices []rules.Command, trickWins int, trickKnown bool) []rules.Command {
	v := a.view
	var envido, truco []rules.Command
	for _, c := range choices {
		if c.Type != "call" {
			continue
		}
		switch c.Kind {
		case "envido", "ley_envido", "flor":
			if a.priceOpening && c.Kind == "envido" && a.pricedOrdinaryOpening() {
				envido = append(envido, c)
				continue
			}
			if v.Holding.Envido < 27 && !v.FlorValid {
				continue
			}
			odds := a.estimate()
			wins, deals := odds.ordinaryWins, odds.ordinaryDeals
			if c.Kind != "envido" {
				wins, deals = odds.leyOpeningWins, odds.envidoDeals
			}
			if c.Kind == "flor" || c.Mode == "con_ley" {
				wins, deals = odds.florWins, odds.contestedDeals
			}
			// Opening con ley can put an otherwise payable flor at risk.
			if a.priceTeamStakes && c.Kind == "ley_envido" && v.FlorValid && !v.Holding.Receta &&
				odds.contestedDeals > 0 && 100*odds.florWins < 55*odds.contestedDeals {
				continue
			}
			if odds.envidoKnown && deals > 0 && 100*wins >= 58*deals {
				envido = append(envido, c)
			}
		case "truco":
			// Do not bypass the existing restrictions on match-point and all-in offers.
			var required int
			switch c.Amount {
			case 3:
				required = 65
			case 6:
				required = 80
			case 9:
				required = 90
			default:
				continue
			}
			if trickKnown && 100*trickWins >= required*dealSamples {
				truco = append(truco, c)
			}
		}
	}
	// The envido window closes on our first card. A truco may still be
	// offered afterward. Every retained command came from the seat's legal list.
	if len(envido) > 0 {
		return envido
	}
	if len(truco) > 0 {
		return truco
	}
	if a.priceTeamStakes {
		// Announcing the first compulsory flor immediately can burn the
		// partner's possible second flor. Keep the choice to say A ley then.
		partnerHasPlayed := v.Seats[(a.seat+2)%4].Plays[0].Present
		if v.FlorValid && !v.Holding.Receta && !v.Seats[a.seat].FlorDeclared &&
			(!v.Config.FlorRequired || partnerHasPlayed) {
			var declared []rules.Command
			for _, c := range choices {
				if c.Type == "play" && c.Declaration == "flor" {
					declared = append(declared, c)
				}
			}
			if len(declared) > 0 {
				return declared
			}
		}
		// A raise that was not justified above should not re-enter through
		// the actor's ordinary menu. A forced choice still belongs to the rules.
		if trickKnown && 100*trickWins < 80*dealSamples {
			if affordable := slices.DeleteFunc(slices.Clone(choices), func(c rules.Command) bool {
				return c.Type == "call" && c.Kind == "truco" && c.Amount >= 6
			}); len(affordable) > 0 {
				return affordable
			}
		}
	}
	return choices
}

// The first ordinary envido pays one on refusal and two at showdown. A middling
// pair can be worth offering before our first card, even when a strong rival
// is likelier to accept. Require both a credible holding and a positive margin.
func (a *wagerAssessment) pricedOrdinaryOpening() bool {
	odds := a.estimate()
	// Allow for refusal equity, but require at least a 36% showdown chance
	// and more than one tenth of a point after accounting for cancellations.
	return shownTeamEnvido(a.view, a.seat) >= minimumOpeningPair && odds.ordinaryDeals > 0 &&
		odds.ordinaryWins*100 >= 36*odds.ordinaryDeals && odds.openingValue > odds.envidoDeals*2
}

// A preview and an exposed play can establish a partner's pair. A known pieza
// pairs with any second card. Seeing the same physical card twice adds nothing.
func shownTeamEnvido(v rules.View, seat int) int {
	best := v.Holding.Envido
	var exposed *rules.Card
	for _, player := range v.Seats {
		if player.Seat == (seat+2)%4 {
			play := player.Plays[0]
			if play.Present && !play.Burned && !play.Concealed {
				exposed = play.Card
			}
		}
	}
	perico, perica := rules.Pieces(*v.Vira)
	for _, card := range []*rules.Card{v.Preview, exposed} {
		if card != nil && (*card == perico || *card == perica) {
			best = max(best, 20+rules.EnvidoCardValue(*card, *v.Vira))
		}
	}
	if v.Preview != nil && exposed != nil && *v.Preview != *exposed &&
		(v.Preview.Suit == exposed.Suit || *v.Preview == perico || *v.Preview == perica || *exposed == perico || *exposed == perica) {
		best = max(best, 20+rules.EnvidoCardValue(*v.Preview, *v.Vira)+rules.EnvidoCardValue(*exposed, *v.Vira))
	}
	return best
}

// An opening truco lets a rival who has yet to play interrupt with envido.
// With a weak team holding, play the card first and keep the truco voice.
func (a *wagerAssessment) exposesWeakEnvido(c rules.Command) bool {
	v := a.view
	if !a.priceOpening || !a.plainEnvido || c.Type != "call" || c.Amount != 3 ||
		v.Pending != nil || v.Round != 0 || shownTeamEnvido(v, a.seat) >= minimumOpeningPair {
		return false
	}
	for _, player := range v.Seats {
		if player.Team != a.seat%2 && !player.Plays[0].Present {
			return true
		}
	}
	return false
}
