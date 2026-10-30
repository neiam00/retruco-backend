package rules

// HandRecap is a completed hand's public result. It never includes a private
// holding, unplayed card, or a pegado bottom that stayed concealed.
type HandRecap struct {
	Vira          *Card          `json:"vira,omitempty"`
	HandNumber    int            `json:"handNumber"`
	Mano          int            `json:"mano"`
	Winner        int            `json:"winner"`
	MatchWinner   int            `json:"matchWinner"`
	Target        int            `json:"target"`
	Reason        string         `json:"reason"`
	ScoresBefore  [2]int         `json:"scoresBefore"`
	ScoresAfter   [2]int         `json:"scoresAfter"`
	Tricks        []TrickRecap   `json:"tricks"`
	Payouts       []Payout       `json:"payouts"`
	EnvidoResults []EnvidoResult `json:"envidoResults"`
}

type TrickRecap struct {
	Round      int         `json:"round"`
	Status     string      `json:"status"`
	WinnerSeat int         `json:"winnerSeat"`
	WinnerTeam int         `json:"winnerTeam"`
	Plays      []RecapPlay `json:"plays"`
}

type RecapPlay struct {
	Seat int `json:"seat"`
	PublicPlay
}

// Called only after the hand ends. Compare only completed, fully public tricks;
// concealed bottoms cannot be reconstructed or labelled from private state.
func (s *State) recordHand(before [2]int, reason string) {
	recap := &HandRecap{
		HandNumber: s.HandNumber, Mano: s.Mano, Winner: s.HandWinner,
		MatchWinner: s.Winner, Target: s.Config.Target, Reason: reason,
		ScoresBefore: before, ScoresAfter: s.Scores,
		Tricks: []TrickRecap{}, Payouts: append([]Payout{}, s.Payouts...),
		EnvidoResults: append([]EnvidoResult{}, s.EnvidoResults...),
	}
	if s.Vira.Valid() {
		vira := s.Vira
		recap.Vira = &vira
	}
	// A departure concedes the match without deciding or paying its pending
	// hand. Payouts/envido values may still belong to a previous HandOver state.
	if reason == "forfeit" {
		recap.Winner = -1
		recap.Payouts = []Payout{}
		recap.EnvidoResults = []EnvidoResult{}
	}
	for round := range 3 {
		trick := TrickRecap{Round: round, Status: "unplayed", WinnerSeat: -1, WinnerTeam: -1, Plays: []RecapPlay{}}
		present, concealed := 0, false
		for seat := 0; seat < s.Config.Players; seat++ {
			play := s.Players[seat].Plays[round]
			public := PublicPlay{Present: play.Present, Burned: play.Burned, Concealed: play.Concealed}
			if play.Present {
				present++
				concealed = concealed || play.Concealed
				if !play.Burned && !play.Concealed {
					card := play.Card
					public.Card = &card
				}
			}
			trick.Plays = append(trick.Plays, RecapPlay{Seat: seat, PublicPlay: public})
		}
		switch {
		case present == 0:
		case present < s.Config.Players:
			trick.Status = "incomplete"
		case concealed:
			trick.Status = "hidden"
		default:
			trick.WinnerSeat = s.roundWinner(round)
			if trick.WinnerSeat >= 0 {
				trick.Status, trick.WinnerTeam = "won", s.Team(trick.WinnerSeat)
			} else {
				trick.Status = "tie"
			}
		}
		recap.Tricks = append(recap.Tricks, trick)
	}
	s.LastHand = recap
}

func copyHandRecap(source *HandRecap) *HandRecap {
	if source == nil {
		return nil
	}
	result := *source
	if source.Vira != nil {
		vira := *source.Vira
		result.Vira = &vira
	}
	result.Payouts = append([]Payout{}, source.Payouts...)
	result.EnvidoResults = append([]EnvidoResult{}, source.EnvidoResults...)
	result.Tricks = make([]TrickRecap, len(source.Tricks))
	for index, trick := range source.Tricks {
		result.Tricks[index] = trick
		result.Tricks[index].Plays = append([]RecapPlay{}, trick.Plays...)
		for playIndex, play := range trick.Plays {
			if play.Card != nil {
				card := *play.Card
				result.Tricks[index].Plays[playIndex].Card = &card
			}
		}
	}
	return &result
}
