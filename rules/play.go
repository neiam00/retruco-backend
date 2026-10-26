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
	if !s.Config.FlorRequired {
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
				s.burnFlor((seat + 2) % 4)
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
		if winner < 0 {
			s.revealBottoms()
			winner = s.roundWinner(1)
		}
		if winner < 0 {
			winner = s.Mano
		}
		s.settle(s.Team(winner))
		return
	}
	stacked, firstStack := 0, -1
	for seat := 0; seat < s.Config.Players; seat++ {
		if s.Players[seat].Stacked {
			stacked++
			firstStack = seat
		}
	}
	if s.Round == 1 && stacked > 0 {
		s.Round = 2
		start := s.Mano
		if stacked == 1 {
			start = s.right(firstStack)
		}
		for step := 0; step < s.Config.Players; step++ {
			seat := (start + step) % s.Config.Players
			if !s.Players[seat].Plays[2].Present {
				s.Turn = seat
				return
			}
		}
	}
	if s.Round == 2 && stacked > 0 {
		first := s.roundWinner(0)
		visibleCardsDecide := s.visibleCardsDecideHand(s.Team(first))
		// Compare in a private copy to apply pegado's top-card penalty without
		// revealing bottom cards that cannot change the visible hand outcome.
		comparison := *s
		comparison.revealBottoms()
		winner := comparison.roundWinner(1)
		for seat := 0; seat < s.Config.Players; seat++ {
			if !s.Players[seat].Stacked || winner == seat {
				continue
			}
			if !comparison.teammateStackTie(seat) {
				s.Players[seat].Plays[2].Burned = true
			}
		}
		if visibleCardsDecide {
			s.settle(s.Team(first))
			return
		}
		s.revealBottoms()
	}
	first, second := s.roundWinner(0), s.roundWinner(1)
	if second < 0 || s.Team(first) == s.Team(second) {
		s.settle(s.Team(first))
		return
	}
	if s.Round == 1 {
		s.Round, s.Turn = 2, second
		return
	}
	third := s.roundWinner(2)
	if third < 0 {
		s.settle(s.Team(first))
	} else {
		s.settle(s.Team(third))
	}
}

func (s State) roundWinner(round int) int {
	best, winner, ties := 0, -1, 0
	for seat := 0; seat < s.Config.Players; seat++ {
		play := s.Players[seat].Plays[round]
		if !play.Present || play.Burned || play.Concealed {
			continue
		}
		strength := Strength(play.Card, s.Vira)
		if strength > best {
			best, winner, ties = strength, seat, 1
		} else if strength == best {
			ties++
		}
	}
	if ties != 1 {
		return -1
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

func (s State) teammateStackTie(seat int) bool {
	if s.Config.Players != 4 {
		return false
	}
	card := s.Players[seat].Plays[1]
	other := s.Players[(seat+2)%4].Plays[1]
	if card.Burned || other.Burned || Strength(card.Card, s.Vira) != Strength(other.Card, s.Vira) {
		return false
	}
	for opponent := 0; opponent < 4; opponent++ {
		if s.Team(opponent) == s.Team(seat) {
			continue
		}
		play := s.Players[opponent].Plays[1]
		if !play.Burned && Strength(play.Card, s.Vira) >= Strength(card.Card, s.Vira) {
			return false
		}
	}
	return true
}
