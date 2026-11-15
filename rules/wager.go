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
		if s.Envido.Kind == "privo" {
			return fmt.Errorf("truco is fixed at three points during privo")
		}
		if c.Mode != "" {
			return fmt.Errorf("truco has no ley mode")
		}
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
	if s.EnvidoClosed && !(s.EnvidoCanceled && (c.Kind == "ley_envido" || c.Kind == "flor")) {
		return fmt.Errorf("envido betting closed")
	}
	if c.Declaration != "" || c.Amount != 0 {
		return fmt.Errorf("invalid envido opening")
	}
	declared := s.otherDeclaredFlor(-1)
	holdingFlor := s.ordinaryFlor(seat)
	wager := Wager{Kind: c.Kind, Caller: team, Value: 2, Previous: 1}
	switch c.Kind {
	case "envido":
		if holdingFlor || declared {
			return fmt.Errorf("ordinary envido cannot be called with or after a flor")
		}
		if c.Mode != "" && !(c.Mode == "sin_ley" && s.Ley) {
			return fmt.Errorf("invalid ordinary envido mode")
		}
		wager.Mode = c.Mode
	case "ley_envido":
		if s.Ley && !declared {
			if c.Mode != s.leyReplyMode(seat) {
				return fmt.Errorf("qualify the envido after A ley according to your holding")
			}
			if holdingFlor {
				// The opening states the caller's flor; the rival must still answer.
				wager.FlorOffered = true
				s.Players[seat].FlorDeclared = true
			} else {
				wager.Mode = c.Mode
			}
		} else if c.Mode != "" {
			return fmt.Errorf("a qualified opening requires a prior A ley without a confirmed flor")
		}
		s.Ley = true
		s.DoubleFlor[team] = true
	case "flor":
		if c.Mode != "" || !holdingFlor || !s.otherDeclaredFlor(seat) {
			return fmt.Errorf("flor tengo y la mia envida requires your flor and another player's declared flor")
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

func (s *State) respond(seat int, c Command, automatic bool) error {
	if s.Phase != Playing || s.Pending == nil || s.Pending.Caller == s.Team(seat) {
		return fmt.Errorf("not the responding team")
	}
	if c.Kind != "raise" && c.Amount != 0 {
		return fmt.Errorf("only a raise may specify additional points")
	}
	wager := *s.Pending
	team := s.Team(seat)
	if wager.Kind == "privo" {
		return s.respondPrivo(seat, c)
	}
	ley := wager.Kind == "ley_envido" || wager.Kind == "flor"
	if ley && wager.Mode == "" {
		mode := s.leyReplyMode(seat)
		if !automatic && c.Mode != mode {
			return fmt.Errorf("ley reply must match the speaker's holding")
		}
		if (wager.Kind == "flor" || wager.FlorOffered) && mode == "sin_ley" && c.Kind != "decline" {
			return fmt.Errorf("answer a flor envidada without flor with Sin ley, no quiero")
		}
		wager.Mode = mode
		if mode == "con_ley" && !automatic {
			s.Players[seat].FlorDeclared = true
		}
	} else if c.Mode != "" && !(wager.Kind == "envido" && c.Kind == "decline" &&
		c.Mode == "sin_ley" && s.Ley && !s.ordinaryFlor(seat)) {
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
	if wager.Kind == "envido" && c.Kind != "decline" && s.teamHasFlor(team, true) {
		return fmt.Errorf("ordinary flor blocks envido: a las flores no")
	}
	if !ley && !automatic {
		s.declareCanto(seat, c.Declaration)
	}
	if ley && wager.Mode == "con_ley" && c.Kind != "decline" {
		s.protectContestedFlores()
	}
	if c.Kind == "decline" {
		if wager.Kind == "truco" {
			s.TrucoValue = wager.Previous
			s.Pending = nil
			s.settle(wager.Caller, "truco_declined")
		} else {
			if automatic && (wager.Kind == "flor" || wager.FlorOffered) {
				s.burnTeamFlor(team)
			} else if ley && wager.Mode == "con_ley" {
				s.burnFlor(seat)
			}
			s.Envido = wager
			s.Envido.Value = wager.Previous
			if ley && s.Pending.Mode == "" && wager.Mode == "sin_ley" {
				s.Envido.Value = 0 // Sin ley, no quiero voids the opening envido.
			}
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

// Opposing envidada flores survive later omissions on cards and cantos.
func (s *State) protectContestedFlores() bool {
	if !s.teamHasFlor(0, false) || !s.teamHasFlor(1, false) {
		return false
	}
	changed := false
	for seat := 0; seat < s.Config.Players; seat++ {
		player := &s.Players[seat]
		if player.FlorValid && !player.FlorEnvidada {
			player.FlorEnvidada = true
			changed = true
		}
	}
	return changed
}

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
	if s.Config.FlorRequired && !hasFlorPrefix && !player.FlorEnvidada {
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

func (s State) otherDeclaredFlor(except int) bool {
	for seat, player := range s.Players {
		if seat != except && player.FlorDeclared {
			return true
		}
	}
	return false
}

func (s State) leyReplyMode(seat int) string {
	if s.ordinaryFlor(seat) {
		return "con_ley"
	}
	return "sin_ley"
}

func (s State) ordinaryFlor(seat int) bool {
	p := s.Players[seat]
	return p.FlorValid && !p.Holding.Receta
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
