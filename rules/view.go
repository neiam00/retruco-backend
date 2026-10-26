package rules

type PublicPlay struct {
	Card      *Card `json:"card,omitempty"`
	Present   bool  `json:"present"`
	Burned    bool  `json:"burned"`
	Concealed bool  `json:"concealed"`
}

type SeatView struct {
	Seat      int           `json:"seat"`
	Team      int           `json:"team"`
	Remaining int           `json:"remaining"`
	Plays     [3]PublicPlay `json:"plays"`
}

type HandCard struct {
	Index int  `json:"index"`
	Card  Card `json:"card"`
	Used  bool `json:"used"`
}

type View struct {
	Version        string         `json:"version"`
	Revision       uint64         `json:"revision"`
	Config         Config         `json:"config"`
	Phase          Phase          `json:"phase"`
	HandNumber     int            `json:"handNumber"`
	Dealer         int            `json:"dealer"`
	Mano           int            `json:"mano"`
	Turn           int            `json:"turn"`
	Round          int            `json:"round"`
	Vira           *Card          `json:"vira,omitempty"`
	Seats          []SeatView     `json:"seats"`
	Hand           []HandCard     `json:"hand"`
	Preview        *Card          `json:"preview,omitempty"`
	Holding        *Holding       `json:"holding,omitempty"`
	FlorValid      bool           `json:"florValid"`
	Scores         [2]int         `json:"scores"`
	Winner         int            `json:"winner"`
	HandWinner     int            `json:"handWinner"`
	Pending        *Wager         `json:"pending,omitempty"`
	Suspended      *Wager         `json:"suspended,omitempty"`
	TrucoValue     int            `json:"trucoValue"`
	MandatoryStack bool           `json:"mandatoryStack"`
	Legal          []Command      `json:"legal"`
	Payouts        []Payout       `json:"payouts"`
	EnvidoResults  []EnvidoResult `json:"envidoResults"`
	Privando       [2]bool        `json:"privando"`
	PrivoEnvido    []PrivoEnvido  `json:"privoEnvido"`
}

// ViewFor authorizes one player's perspective. Spectators receive the same card
// visibility but never legal actions. A negative perspective is public-only.
func ViewFor(s State, perspective int, spectator bool) View {
	v := View{Version: Version, Revision: s.Revision, Config: s.Config, Phase: s.Phase,
		HandNumber: s.HandNumber, Dealer: s.Dealer, Mano: s.Mano, Turn: s.Turn,
		Round: s.Round, Scores: s.Scores, Winner: s.Winner, HandWinner: s.HandWinner,
		TrucoValue: s.TrucoValue, MandatoryStack: s.MandatoryStack,
		Payouts: append([]Payout{}, s.Payouts...), Seats: []SeatView{}, Hand: []HandCard{}, Legal: []Command{},
		EnvidoResults: append([]EnvidoResult{}, s.EnvidoResults...),
		Privando:      s.Privando, PrivoEnvido: s.privoAnnouncements(),
	}
	if s.Pending != nil {
		pending := *s.Pending
		v.Pending = &pending
	}
	if s.Suspended != nil {
		suspended := *s.Suspended
		v.Suspended = &suspended
	}
	if s.Vira.Valid() {
		vira := s.Vira
		v.Vira = &vira
	}
	for seat := 0; seat < s.Config.Players; seat++ {
		player := s.Players[seat]
		sv := SeatView{Seat: seat, Team: s.Team(seat), Remaining: len(s.Remaining(seat))}
		for round, play := range player.Plays {
			pv := PublicPlay{Present: play.Present, Burned: play.Burned, Concealed: play.Concealed}
			if play.Present && !play.Burned && !play.Concealed {
				card := play.Card
				pv.Card = &card
			}
			sv.Plays[round] = pv
		}
		v.Seats = append(v.Seats, sv)
	}
	if perspective >= 0 && perspective < s.Config.Players {
		if s.Vira.Valid() {
			player := s.Players[perspective]
			for i, card := range player.Cards {
				v.Hand = append(v.Hand, HandCard{Index: i, Card: card, Used: player.Used[i]})
			}
			holding := player.Holding
			v.Holding, v.FlorValid = &holding, player.FlorValid
			if s.Preview[perspective] != nil {
				card := *s.Preview[perspective]
				v.Preview = &card
			}
		}
		if !spectator {
			v.Legal = LegalActions(s, perspective)
		}
	}
	return v
}

// LegalActions uses the same validation path as commands. Card indices refer to
// the caller's own hand; no opponent hand is included in the returned actions.
func LegalActions(s State, seat int) []Command {
	legal := []Command{}
	if seat < 0 || seat >= s.Config.Players || s.Phase == MatchOver {
		return legal
	}
	if s.Phase == Cutting {
		if seat == s.Turn {
			for cut := 1; cut < 40; cut++ {
				legal = append(legal, Command{Type: "cut", Cut: cut})
			}
		}
		return legal
	}
	if s.Phase != Playing {
		return legal
	}
	// Skip whole families that cannot pass their turn/team guards. Apply remains
	// the authority for individual commands, without copying State and allocating
	// rejection errors for every impossible candidate in each player's snapshot.
	cardTurn := s.Pending == nil && s.Turn == seat
	responding := s.Pending != nil && s.Pending.Caller != s.Team(seat)
	canCall := cardTurn || responding && s.Pending.Kind == "truco"
	if !cardTurn && !responding {
		return legal
	}
	add := func(c Command) {
		if _, _, err := Apply(s, seat, c); err == nil {
			legal = append(legal, c)
		}
	}
	if s.Pending != nil && s.Pending.Kind == "privo" {
		for _, kind := range []string{"accept", "decline", "present_flor"} {
			add(Command{Type: "respond", Kind: kind})
		}
		return legal
	}
	if cardTurn {
		remaining := s.Remaining(seat)
		for _, declaration := range []string{"", "flor", "a_ley", "presento_flor", "flor_flor", "con_mi_flor"} {
			if s.MandatoryStack {
				add(Command{Type: "burn", Declaration: declaration})
			}
			for _, card := range remaining {
				add(Command{Type: "play", Card: card, Declaration: declaration})
				if !s.MandatoryStack {
					add(Command{Type: "burn", Card: card, Declaration: declaration})
				}
				for _, top := range remaining {
					if top != card {
						add(Command{Type: "stack", Card: card, Top: top, Declaration: declaration})
					}
				}
			}
		}
	}
	for _, declaration := range []string{"", "flor", "presento_flor", "flor_flor", "con_mi_flor"} {
		if canCall {
			for _, kind := range []string{"envido", "ley_envido", "flor"} {
				add(Command{Type: "call", Kind: kind, Declaration: declaration})
			}
			if !s.Ley {
				add(Command{Type: "call", Kind: "envido", Mode: "sin_ley", Declaration: declaration})
			}
			for _, mode := range []string{"con_ley", "sin_ley"} {
				add(Command{Type: "call", Kind: "ley_envido", Mode: mode, Declaration: declaration})
			}
			for _, amount := range []int{3, 6, 9, 100} {
				add(Command{Type: "call", Kind: "truco", Amount: amount, Declaration: declaration})
			}
		}
		if !responding {
			continue
		}
		for _, mode := range []string{"", "con_ley", "sin_ley"} {
			if mode != "" && (s.Pending.Kind != "ley_envido" || s.Pending.Mode != "") {
				continue
			}
			for _, kind := range []string{"accept", "decline", "flowers_no", "falta"} {
				if kind == "decline" && mode != "" {
					continue
				}
				add(Command{Type: "respond", Kind: kind, Mode: mode, Declaration: declaration})
			}
			for amount := 1; amount <= s.Config.Target; amount++ {
				add(Command{Type: "respond", Kind: "raise", Amount: amount, Mode: mode, Declaration: declaration})
			}
			add(Command{Type: "respond", Kind: "raise", Amount: 100, Mode: mode, Declaration: declaration})
		}
		add(Command{Type: "respond", Kind: "decline", Mode: "sin_ley", Declaration: declaration})
	}
	return legal
}
