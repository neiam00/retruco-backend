package rules

func (s *State) settle(handWinner int) {
	s.HandWinner = handWinner
	s.Phase = HandOver
	s.Pending, s.Suspended = nil, nil
	s.Payouts = nil
	envidoWinner := s.EnvidoWinner
	if s.EnvidoAccepted {
		s.EnvidoResults = make([]EnvidoResult, 0, s.Config.Players)
		for seat := 0; seat < s.Config.Players; seat++ {
			value, eligible := s.envidoValue(seat)
			s.EnvidoResults = append(s.EnvidoResults, EnvidoResult{Seat: seat, Value: value, Eligible: eligible})
		}
		envidoWinner = s.compareEnvido()
		if s.Envido.Kind == "flor" && envidoWinner >= 0 {
			s.burnTeamFlor(1 - envidoWinner)
		}
	}
	flor := [2]int{}
	for seat := 0; seat < s.Config.Players; seat++ {
		if s.Players[seat].FlorValid {
			flor[s.Team(seat)] += 3
		}
	}
	envido := [2]int{}
	if s.Envido.Kind != "" {
		if envidoWinner >= 0 {
			envido[envidoWinner] = s.Envido.Value
		}
	} else {
		for seat := 0; seat < s.Config.Players; seat++ {
			if s.Players[seat].Holding.Receta {
				envido[s.Team(seat)] = 1
			}
		}
	}
	// A ley is the user's explicit exception to the usual scoring order.
	if s.Envido.Kind == "ley_envido" {
		if s.award("envido", envido) || s.award("flor", flor) {
			return
		}
	} else {
		if s.award("flor", flor) || s.award("envido", envido) {
			return
		}
	}
	truco := [2]int{}
	truco[handWinner] = s.TrucoValue
	if s.TrucoValue == 100 {
		truco[handWinner] = s.Config.Target - s.Scores[handWinner]
	}
	s.award("truco", truco)
}

func (s *State) award(kind string, points [2]int) bool {
	for team, amount := range points {
		if amount > 0 {
			s.Scores[team] += amount
			s.Payouts = append(s.Payouts, Payout{Kind: kind, Team: team, Points: amount})
		}
	}
	// For simultaneous uncontested flores, mano's team has priority.
	for _, team := range []int{s.Team(s.Mano), 1 - s.Team(s.Mano)} {
		if s.Scores[team] >= s.Config.Target {
			s.Winner, s.Phase = team, MatchOver
			return true
		}
	}
	return false
}

func (s State) compareEnvido() int {
	best, winner := -1, -1
	for offset := 0; offset < s.Config.Players; offset++ {
		seat := (s.Mano + offset) % s.Config.Players
		value, eligible := s.envidoValue(seat)
		if !eligible {
			continue
		}
		if value > best {
			best, winner = value, s.Team(seat)
		}
	}
	return winner
}

func (s State) envidoValue(seat int) (int, bool) {
	player := s.Players[seat]
	if s.Envido.Kind == "flor" {
		return player.Holding.Flor, player.FlorValid
	}
	if s.Envido.Mode == "con_ley" && player.FlorValid {
		return player.Holding.Flor, true
	}
	return player.Holding.Envido, true
}
