package rules

import "fmt"

func (s *State) play(seat int, command Command, timeout bool) error {
	if s.Phase != Playing || s.Pending != nil || s.Turn != seat {
		return fmt.Errorf("not your card turn")
	}
	voluntaryBurn := command.Type == "burn"
	if voluntaryBurn && s.Round == 0 {
		return fmt.Errorf("first-round cards cannot be burned")
	}
	if voluntaryBurn && s.MandatoryStack {
		remaining := s.Remaining(seat)
		if len(remaining) != 2 {
			return fmt.Errorf("must burn both remaining cards")
		}
		command.Card, command.Top = remaining[0], remaining[1]
	}
	p := &s.Players[seat]
	if command.Card < 0 || command.Card >= 3 || p.Used[command.Card] {
		return fmt.Errorf("card unavailable")
	}
	stack := command.Type == "stack" || voluntaryBurn && s.MandatoryStack
	if s.MandatoryStack && !stack {
		return fmt.Errorf("must play pegado")
	}
	if stack && (s.Round != 1 || command.Top < 0 || command.Top >= 3 || command.Top == command.Card || p.Used[command.Top]) {
		return fmt.Errorf("invalid pegado")
	}
	if !timeout || s.Round == 0 {
		s.declareCards(seat, command.Declaration)
	}
	burned := voluntaryBurn || timeout && s.Round > 0
	p.Used[command.Card] = true
	p.Plays[s.Round] = Play{Card: p.Cards[command.Card], Present: true, Burned: burned}
	if s.Round == 0 {
		p.FirstPlayed = true
	}
	if stack {
		p.Used[command.Top] = true
		p.Stacked = true
		p.Plays[1].Concealed = !burned
		p.Plays[2] = Play{Card: p.Cards[command.Top], Present: true, Burned: burned}
	}
	s.advanceCards()
	return nil
}

func (s *State) declareCards(seat int, declaration string) {
	p := &s.Players[seat]
	if declaration == "a_ley" && !p.FirstPlayed {
		s.Ley = true
	}
	if !p.FlorValid || p.Holding.Receta {
		return
	}
	if p.FlorEnvidada || !s.Config.FlorRequired {
		if declaresFlor(declaration) {
			p.FlorDeclared = true
		}
		return
	}
	valid := declaresFlor(declaration)
	if !p.FirstPlayed && declaration == "a_ley" {
		valid = true
		s.DoubleFlor[s.Team(seat)] = true
	}
	if !valid {
		p.FlorValid = false
		return
	}
	p.FlorDeclared = p.FlorDeclared || declaresFlor(declaration)
	team := s.Team(seat)
	if !p.FirstPlayed && declaration != "a_ley" && !s.DoubleFlor[team] {
		if s.FlorClaimant[team] >= 0 && s.FlorClaimant[team] != seat {
			p.FlorValid = false
		} else {
			s.FlorClaimant[team] = seat
			if s.Config.Players == 4 {
				partner := (seat + 2) % 4
				if !s.Players[partner].FlorEnvidada {
					s.burnFlor(partner)
				}
			}
		}
	}
}

func (s *State) advanceCards() {
	for step := 1; step <= s.Config.Players; step++ {
		seat := (s.Turn + step) % s.Config.Players
		if !s.Players[seat].Plays[s.Round].Present {
			s.Turn = seat
			return
		}
	}
	if s.Round == 0 {
		winner := s.roundWinner(0)
		s.Round = 1
		s.MandatoryStack = winner < 0
		if winner < 0 {
			s.Turn = s.Mano
		} else {
			s.Turn = winner
		}
		return
	}
	if s.MandatoryStack {
		winner := s.roundWinner(2)
		reason := "mandatory_top"
		if winner < 0 {
			s.revealBottoms()
			winner = s.roundWinner(1)
			reason = "mandatory_bottom"
		}
		if winner < 0 {
			winner = s.Mano
			reason = "mano_tie"
		}
		s.settle(s.Team(winner), reason)
		return
	}
	stackLead := s.voluntaryThirdLead()
	if s.Round == 1 && stackLead >= 0 {
		s.Round = 2
		for step := 0; step < s.Config.Players; step++ {
			seat := (stackLead + step) % s.Config.Players
			if !s.Players[seat].Plays[2].Present {
				s.Turn = seat
				return
			}
		}
	}
	if s.Round == 2 && stackLead >= 0 {
		first := s.roundWinner(0)
		visibleCardsDecide := s.visibleCardsDecideHand(s.Team(first))
		// Compare in a private copy to apply pegado's top-card penalty without
		// revealing bottom cards that cannot change the visible hand outcome.
		comparison := *s
		comparison.revealBottoms()
		winner := comparison.roundWinner(1)
		winningRank := 0
		if winner >= 0 {
			winningRank = Strength(comparison.Players[winner].Plays[1].Card, s.Vira)
		}
		for seat := 0; seat < s.Config.Players; seat++ {
			if !s.Players[seat].Stacked {
				continue
			}
			bottom := comparison.Players[seat].Plays[1]
			if bottom.Burned || Strength(bottom.Card, s.Vira) != winningRank {
				s.Players[seat].Plays[2].Burned = true
			}
		}
		if visibleCardsDecide {
			s.settle(s.Team(first), "visible_first_and_third")
			return
		}
		s.revealBottoms()
	}
	first, second := s.roundWinner(0), s.roundWinner(1)
	if second < 0 || s.Team(first) == s.Team(second) {
		reason := "two_tricks"
		if second < 0 {
			reason = "first_and_tie"
		}
		s.settle(s.Team(first), reason)
		return
	}
	if s.Round == 1 {
		s.Round, s.Turn = 2, second
		return
	}
	third := s.roundWinner(2)
	if third < 0 {
		s.settle(s.Team(first), "third_tie")
	} else {
		s.settle(s.Team(third), "third_trick")
	}
}

// A lone voluntary stack passes play to its right. With multiple stacks,
// unstacked players take their third turns in mano order. Returns -1 without stacks.
func (s *State) voluntaryThirdLead() int {
	stack := -1
	for seat := 0; seat < s.Config.Players; seat++ {
		if s.Players[seat].Stacked {
			if stack >= 0 {
				return s.Mano
			}
			stack = seat
		}
	}
	if stack < 0 {
		return -1
	}
	return s.right(stack)
}

func (s *State) roundWinner(round int) int {
	var ranks [4]int
	for seat := 0; seat < s.Config.Players; seat++ {
		play := s.Players[seat].Plays[round]
		if play.Present && !play.Burned && !play.Concealed {
			ranks[seat] = Strength(play.Card, s.Vira)
		}
	}
	winner := TrickWinner(ranks, s.Config.Players, s.Mano)
	if winner < 0 || round == 0 || s.MandatoryStack || s.Config.Players != 4 {
		return winner
	}
	partner := (winner + 2) % 4
	if ranks[partner] != ranks[winner] {
		return winner
	}
	// Only teammates sharing the high card need their actual play order.
	if round == 1 || s.Players[winner].Stacked && s.Players[partner].Stacked {
		return TrickWinner(ranks, s.Config.Players, s.roundWinner(0))
	}
	// A stacked top was played before any loose third card.
	if s.Players[winner].Stacked {
		return winner
	}
	if s.Players[partner].Stacked {
		return partner
	}
	lead := s.voluntaryThirdLead()
	if lead < 0 {
		lead = s.roundWinner(1)
	}
	return TrickWinner(ranks, s.Config.Players, lead)
}

// TrickWinner returns the first seat in play order holding the highest valid
// rank, or -1 if opponents share that rank or all ranks are zero.
func TrickWinner(ranks [4]int, players, lead int) int {
	best, winner := 0, -1
	for offset := 0; offset < players; offset++ {
		seat := (lead + offset) % players
		if rank := ranks[seat]; rank > best {
			best, winner = rank, seat
		} else if rank == best && winner >= 0 && seat%2 != winner%2 {
			winner = -1
		}
	}
	return winner
}

func (s *State) revealBottoms() {
	for seat := 0; seat < s.Config.Players; seat++ {
		s.Players[seat].Plays[1].Concealed = false
	}
}

// A voluntary top card may or may not survive its hidden bottom's comparison.
// If every possible combination still gives the first-round team the hand,
// revealing those bottoms would disclose cards without changing the outcome.
func (s State) visibleCardsDecideHand(firstTeam int) bool {
	var stacked []int
	for seat := 0; seat < s.Config.Players; seat++ {
		if s.Players[seat].Stacked {
			stacked = append(stacked, seat)
		}
	}
	for burned := 0; burned < 1<<len(stacked); burned++ {
		possible := s
		for index, seat := range stacked {
			possible.Players[seat].Plays[2].Burned = s.Players[seat].Plays[2].Burned || burned&(1<<index) != 0
		}
		third := possible.roundWinner(2)
		if third >= 0 && possible.Team(third) != firstTeam {
			return false
		}
	}
	return true
}
