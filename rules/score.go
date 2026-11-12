package rules

func (s *State) settle(handWinner int, reason string) {
	defer s.recordHand(s.Scores, reason)
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
		winner := s.bestEnvidoSeat()
		envidoWinner = -1
		if winner >= 0 {
			envidoWinner = s.Team(winner)
			if s.Envido.Kind == "flor" || s.Envido.Mode == "con_ley" {
				// Only the highest flor earns the contested three points. Receta stays protected.
				for seat := 0; seat < s.Config.Players; seat++ {
					if seat != winner {
						s.burnFlor(seat)
					}
				}
			}
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
	// A ley and privo settle envido before flor, stopping as soon as it wins.
	if s.Envido.Kind == "ley_envido" || s.Envido.Kind == "flor" || s.Envido.Kind == "privo" {
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

func (s State) bestEnvidoSeat() int {
	best, winner := -1, -1
	for offset := 0; offset < s.Config.Players; offset++ {
		seat := (s.Mano + offset) % s.Config.Players
		value, eligible := s.envidoValue(seat)
		if !eligible {
			continue
		}
		if value > best {
			best, winner = value, seat
		}
	}
	return winner
}

func (s State) envidoValue(seat int) (int, bool) {
	player := s.Players[seat]
	if s.Envido.Kind == "flor" || s.Envido.Mode == "con_ley" {
		return player.Holding.Flor, player.FlorValid
	}
	// With Sin ley the player without flor uses ordinary envido. A valid
	// opposing flor still competes with all three cards.
	if s.Envido.Kind == "ley_envido" && player.FlorValid {
		return player.Holding.Flor, true
	}
	return player.Holding.Envido, true
}
