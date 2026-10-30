package rules

import "fmt"

func (s *State) call(seat int, c Command) error {
	if s.Phase != Playing {
		return fmt.Errorf("hand not active")
	}
	team := s.Team(seat)
	interrupt := s.Pending != nil && s.Pending.Kind == "truco" && s.Pending.Caller != team && c.Kind != "truco"
	if s.Pending != nil && !interrupt {
		return fmt.Errorf("answer the pending canto")
	}
	if !interrupt && s.Turn != seat {
		return fmt.Errorf("canto requires your turn")
	}
	if c.Kind == "truco" {
		if s.TrucoVoice >= 0 && s.TrucoVoice != team {
			return fmt.Errorf("team lacks truco voice")
		}
		amount := c.Amount
		if amount == 0 {
			amount = 3
		}
		if !trucoLevel(amount) || amount <= s.TrucoValue || (s.TrucoValue == 1 && amount != 3) {
			return fmt.Errorf("invalid truco level")
		}
		s.declareCanto(seat, c.Declaration)
		s.Pending = &Wager{Kind: "truco", Caller: team, Value: amount, Previous: s.TrucoValue}
		return nil
	}
	if s.Round != 0 || s.Players[seat].FirstPlayed {
		return fmt.Errorf("envido window closed for player")
	}
	if s.EnvidoClosed && (!s.EnvidoCanceled || c.Kind != "flor") {
		return fmt.Errorf("envido betting closed")
	}
	// Check earlier declarations before this canto can declare the caller's flor.
	// Once anyone has announced flor, envido must use the flor contest instead.
	if (c.Kind == "envido" || c.Kind == "ley_envido") && s.hasDeclaredFlor() {
		return fmt.Errorf("a flor was declared: use flor tengo y la mia envida")
	}
	wager := Wager{Kind: c.Kind, Caller: team, Value: 2, Previous: 1}
	switch c.Kind {
	case "envido":
		if c.Mode != "" && c.Mode != "sin_ley" {
			return fmt.Errorf("invalid ordinary envido mode")
		}
		if c.Mode == "sin_ley" && !s.Ley {
			return fmt.Errorf("no a ley declaration")
		}
		wager.Mode = c.Mode
		s.declareCanto(seat, c.Declaration)
		if !declaresFlor(c.Declaration) || c.Mode == "sin_ley" {
			s.burnFlor(seat)
		}
	case "ley_envido":
		if c.Mode == "" && declaresFlor(c.Declaration) {
			return fmt.Errorf("a ley envido cannot announce flor")
		}
		if c.Mode != "" && c.Mode != "con_ley" && c.Mode != "sin_ley" {
			return fmt.Errorf("invalid ley mode")
		}
		if c.Mode != "" && !s.Ley {
			return fmt.Errorf("no a ley declaration")
		}
		wager.Mode = c.Mode
		s.Ley = true
		// The call itself says A ley, before either teammate shows a card.
		s.DoubleFlor[team] = true
		if declaresFlor(c.Declaration) {
			s.declareCanto(seat, c.Declaration)
		}
		if c.Mode == "sin_ley" {
			s.burnFlor(seat)
		}
	case "flor":
		if !s.Players[seat].FlorValid || s.Players[seat].Holding.Receta || !s.teamHasFlor(1-team, false) {
			return fmt.Errorf("flor contest requires opposing flores")
		}
		if !s.hasDeclaredFlor() {
			return fmt.Errorf("a flor must be declared first")
		}
		s.Players[seat].FlorDeclared = true
	default:
		return fmt.Errorf("unknown canto")
	}
	if interrupt {
		previous := *s.Pending
		s.Suspended = &previous
	}
	s.Pending = &wager
	return nil
}

func (s *State) respond(seat int, c Command) error {
	if s.Phase != Playing || s.Pending == nil || s.Pending.Caller == s.Team(seat) {
		return fmt.Errorf("not the responding team")
	}
	wager := *s.Pending
	team := s.Team(seat)
	if wager.Kind == "privo" {
		return s.respondPrivo(seat, c)
	}
	choosesLey := wager.Kind == "ley_envido" && wager.Mode == "" &&
		(c.Kind == "accept" || c.Kind == "raise" || c.Kind == "falta")
	declinesSinLey := (wager.Kind == "envido" || wager.Kind == "ley_envido") &&
		c.Kind == "decline" && c.Mode == "sin_ley" && (s.Ley || wager.Kind == "ley_envido")
	// LegalActions uses this validation too, so irrelevant modes must be
	// rejected rather than producing duplicate replies with different labels.
	if c.Mode != "" && !choosesLey && !declinesSinLey {
		return fmt.Errorf("this response does not choose a ley mode")
	}
	if c.Kind == "flowers_no" {
		if wager.Kind != "envido" || !s.Players[seat].FlorValid || s.Players[seat].Holding.Receta {
			return fmt.Errorf("no ordinary flor to block envido")
		}
		s.Players[seat].FlorDeclared = true
		s.EnvidoCanceled, s.EnvidoClosed = true, true
		s.resumeTruco()
		return nil
	}
	if c.Kind != "accept" && c.Kind != "decline" && c.Kind != "raise" && c.Kind != "falta" {
		return fmt.Errorf("invalid response")
	}
	s.declareCanto(seat, c.Declaration)
	if c.Mode == "sin_ley" {
		s.burnFlor(seat)
	}
	if wager.Kind == "envido" && c.Kind != "decline" && s.teamHasFlor(team, true) {
		return fmt.Errorf("ordinary flor blocks envido: a las flores no")
	}
	if wager.Kind == "ley_envido" && wager.Mode == "" && c.Kind != "decline" {
		if c.Mode != "con_ley" && c.Mode != "sin_ley" {
			return fmt.Errorf("choose con ley or sin ley")
		}
		wager.Mode = c.Mode
	}
	if c.Kind == "decline" {
		if wager.Kind == "truco" {
			s.TrucoValue = wager.Previous
			s.Pending = nil
			s.settle(wager.Caller, "truco_declined")
		} else {
			if wager.Kind == "flor" {
				// Con mi flor me achico forfeits only the speaker's flor.
				s.burnFlor(seat)
			}
			s.Envido = wager
			s.Envido.Value = wager.Previous
			s.EnvidoWinner = wager.Caller
			s.EnvidoClosed, s.EnvidoAccepted, s.EnvidoCanceled = true, false, false
			s.resumeTruco()
		}
		return nil
	}
	if c.Kind == "accept" {
		if wager.Kind == "truco" {
			s.TrucoValue, s.TrucoVoice = wager.Value, team
			s.Pending = nil
		} else {
			s.Envido = wager
			s.EnvidoClosed, s.EnvidoAccepted, s.EnvidoCanceled = true, true, false
			s.resumeTruco()
		}
		return nil
	}
	if wager.Terminal {
		return fmt.Errorf("la falta is terminal")
	}
	var next int
	terminal := false
	if wager.Kind == "truco" {
		next = c.Amount
		if c.Kind != "raise" || !trucoLevel(next) || next <= wager.Value {
			return fmt.Errorf("invalid truco raise")
		}
	} else if c.Kind == "falta" {
		next = s.Config.Target - max(s.Scores[0], s.Scores[1])
		if next < wager.Value {
			return fmt.Errorf("la falta cannot lower the wager")
		}
		terminal = true
	} else {
		// Bound the operands before addition: a positive int can otherwise wrap
		// into a negative wager and pass an upper-bound check on the result.
		if wager.Value < 0 || wager.Value > s.Config.Target || c.Amount <= 0 || c.Amount > s.Config.Target-wager.Value {
			return fmt.Errorf("envido raise exceeds limit or is not positive")
		}
		next = wager.Value + c.Amount
	}
	wager.Previous, wager.Value, wager.Caller, wager.Terminal = wager.Value, next, team, terminal
	s.Pending = &wager
	return nil
}

func trucoLevel(value int) bool { return value == 3 || value == 6 || value == 9 || value == 100 }

func (s *State) resumeTruco() { s.Pending, s.Suspended = s.Suspended, nil }

func (s *State) burnFlor(seat int) {
	if !s.Players[seat].Holding.Receta {
		s.Players[seat].FlorValid = false
	}
}

func (s *State) burnTeamFlor(team int) {
	for seat := 0; seat < s.Config.Players; seat++ {
		if s.Team(seat) == team {
			s.burnFlor(seat)
		}
	}
}

func (s *State) declareCanto(seat int, declaration string) {
	hasFlorPrefix := declaresFlor(declaration)
	player := &s.Players[seat]
	if hasFlorPrefix && player.FlorValid && !player.Holding.Receta {
		player.FlorDeclared = true
	}
	if s.Config.FlorRequired && !hasFlorPrefix {
		s.burnFlor(seat)
	}
}

func declaresFlor(declaration string) bool {
	switch declaration {
	case "flor", "presento_flor", "flor_flor", "con_mi_flor", "con_flor":
		return true
	default:
		return false
	}
}

func (s State) hasDeclaredFlor() bool {
	for _, player := range s.Players {
		if player.FlorDeclared {
			return true
		}
	}
	return false
}

func (s State) teamHasFlor(team int, ordinaryOnly bool) bool {
	for seat := 0; seat < s.Config.Players; seat++ {
		p := s.Players[seat]
		if s.Team(seat) == team && p.FlorValid && (!ordinaryOnly || !p.Holding.Receta) {
			return true
		}
	}
	return false
}
