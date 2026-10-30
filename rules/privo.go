package rules

import "fmt"

// PrivoEnvido is a team's truthful announcement, without identifying which
// teammate holds it or disclosing either player's cards.
type PrivoEnvido struct {
	Team  int `json:"team"`
	Value int `json:"value"`
}

func (s State) highestEnvido(team int) int {
	value := 0
	for seat := team; seat < s.Config.Players; seat += 2 {
		value = max(value, s.Players[seat].Holding.Envido)
	}
	return value
}

func (s State) privoAnnouncements() []PrivoEnvido {
	announcements := []PrivoEnvido{}
	for team, privando := range s.Privando {
		if privando {
			announcements = append(announcements, PrivoEnvido{Team: team, Value: s.highestEnvido(team)})
		}
	}
	return announcements
}

// beginPrivo runs only after dealing. Points earned later in this hand cannot
// retroactively introduce another wager or announcement.
func (s *State) beginPrivo() {
	for team, score := range s.Scores {
		s.Privando[team] = score == s.Config.Target-1
	}
	if !s.Privando[0] && !s.Privando[1] {
		return
	}
	// A privando team's flor wins immediately. If both qualify, mano wins.
	for _, team := range []int{s.Team(s.Mano), 1 - s.Team(s.Mano)} {
		if s.Privando[team] && s.teamHasFlor(team, false) {
			s.finishPrivo("flor", team, 3, "privo_flor")
			return
		}
	}
	caller := 0
	if !s.Privando[0] {
		caller = 1
	}
	s.Envido = Wager{Kind: "privo", Caller: caller, Value: 2, Previous: 1, Terminal: true}
	s.EnvidoClosed = true
	if s.Privando[0] && s.Privando[1] {
		s.EnvidoAccepted = true
		return
	}
	wager := s.Envido
	s.Pending = &wager
}

func (s *State) respondPrivo(seat int, command Command) error {
	if command.Declaration != "" || command.Mode != "" || command.Amount != 0 {
		return fmt.Errorf("privo has a fixed envido and no declaration modifiers")
	}
	team := s.Team(seat)
	switch command.Kind {
	case "accept":
		// Accepting privo does not burn an ordinary flor. Its later card
		// declarations still follow the room's normal flor rules.
		s.EnvidoAccepted = true
		s.Pending = nil
	case "decline":
		s.EnvidoWinner = s.Pending.Caller
		s.Envido.Value = 1
		s.finishPrivo("envido", s.Pending.Caller, 1, "privo_declined")
	case "present_flor":
		if !s.Players[seat].FlorValid {
			return fmt.Errorf("presenting flor requires your own valid flor")
		}
		s.Players[seat].FlorDeclared = true
		s.Envido = Wager{}
		s.EnvidoCanceled = true
		s.finishPrivo("flor", team, 3, "flor_presented")
	default:
		return fmt.Errorf("accept privo, decline, or present flor")
	}
	return nil
}

// These outcomes end the hand without awarding its cancelled truco or envido.
func (s *State) finishPrivo(kind string, team, points int, reason string) {
	defer s.recordHand(s.Scores, reason)
	s.Phase, s.HandWinner = HandOver, team
	s.Pending, s.Suspended = nil, nil
	s.Payouts = nil
	awards := [2]int{}
	awards[team] = points
	s.award(kind, awards)
}
