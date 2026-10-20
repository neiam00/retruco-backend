package rules

import "fmt"

const Version = "1"

type Phase string

const (
	Cutting   Phase = "cutting"
	Playing   Phase = "playing"
	HandOver  Phase = "hand_over"
	MatchOver Phase = "match_over"
)

type Config struct {
	Players      int  `json:"players"`
	Target       int  `json:"target"`
	FlorRequired bool `json:"florRequired"`
}

type Play struct {
	Card      Card `json:"card"`
	Present   bool `json:"present"`
	Burned    bool `json:"burned"`
	Concealed bool `json:"concealed"`
}

type Player struct {
	Cards        [3]Card
	Used         [3]bool
	Holding      Holding
	FlorValid    bool
	FlorDeclared bool
	FirstPlayed  bool
	Stacked      bool
	Plays        [3]Play
}

type Wager struct {
	Kind     string `json:"kind"`
	Caller   int    `json:"caller"`
	Value    int    `json:"value"`
	Previous int    `json:"previous"`
	Terminal bool   `json:"terminal"`
	Mode     string `json:"mode,omitempty"`
}

// State contains secrets and must never be serialized to a client. View is the
// only client-facing representation. Seats increase counter-clockwise; teammates
// occupy alternating seats in four-player matches.
type State struct {
	Config         Config
	Phase          Phase
	Revision       uint64
	HandNumber     int
	Dealer         int
	Mano           int
	Turn           int
	Round          int
	Deck           [40]Card
	Vira           Card
	Players        [4]Player
	Preview        [4]*Card
	Scores         [2]int
	Winner         int
	HandWinner     int
	Pending        *Wager
	Suspended      *Wager
	TrucoValue     int
	TrucoVoice     int
	Envido         Wager
	EnvidoClosed   bool
	EnvidoAccepted bool
	EnvidoCanceled bool
	EnvidoWinner   int
	Ley            bool
	DoubleFlor     [2]bool
	FlorClaimant   [2]int
	MandatoryStack bool
	Payouts        []Payout
}

type Payout struct {
	Kind   string `json:"kind"`
	Team   int    `json:"team"`
	Points int    `json:"points"`
}

type Command struct {
	Type        string `json:"type"`
	Card        int    `json:"card,omitempty"`
	Top         int    `json:"top,omitempty"`
	Cut         int    `json:"cut,omitempty"`
	Kind        string `json:"kind,omitempty"`
	Amount      int    `json:"amount,omitempty"`
	Declaration string `json:"declaration,omitempty"`
	Mode        string `json:"mode,omitempty"`
}

type Event struct {
	Type string `json:"type"`
	Seat int    `json:"seat"`
	Text string `json:"text,omitempty"`
}

func (s State) Team(seat int) int  { return seat % 2 }
func (s State) right(seat int) int { return (seat + 1) % s.Config.Players }
func (s State) left(seat int) int  { return (seat + s.Config.Players - 1) % s.Config.Players }

func New(config Config, dealer int, deck []Card) (State, error) {
	if (config.Players != 2 && config.Players != 4) || (config.Target != 12 && config.Target != 24) {
		return State{}, fmt.Errorf("invalid match settings")
	}
	if dealer < 0 || dealer >= config.Players {
		return State{}, fmt.Errorf("invalid dealer")
	}
	s := State{Config: config, Dealer: dealer, Phase: HandOver, Winner: -1}
	return StartHand(s, deck)
}

// StartHand accepts a host-shuffled deck. No randomness occurs in this package.
func StartHand(s State, deck []Card) (State, error) {
	if s.Phase != HandOver || len(deck) != 40 {
		return s, fmt.Errorf("cannot start hand")
	}
	seen := make(map[Card]bool, 40)
	for _, card := range deck {
		if !card.Valid() || seen[card] {
			return s, fmt.Errorf("invalid deck")
		}
		seen[card] = true
	}
	dealer := s.Dealer
	if s.HandNumber > 0 {
		dealer = s.right(dealer)
	}
	next := State{
		Config: s.Config, Phase: Cutting, Revision: s.Revision + 1,
		HandNumber: s.HandNumber + 1, Dealer: dealer, Scores: s.Scores,
		Winner: -1, HandWinner: -1, TrucoValue: 1, TrucoVoice: -1,
		EnvidoWinner: -1, FlorClaimant: [2]int{-1, -1},
	}
	next.Mano = next.right(dealer)
	next.Turn = next.left(dealer)
	copy(next.Deck[:], deck)
	return next, nil
}

// Apply validates a command against a value copy. Rejected commands leave the
// input state unchanged, including its revision and active response window.
func Apply(s State, seat int, command Command) (State, []Event, error) {
	if seat < 0 || seat >= s.Config.Players || s.Phase == MatchOver {
		return s, nil, fmt.Errorf("invalid player or finished match")
	}
	next := s
	if s.Pending != nil {
		pending := *s.Pending
		next.Pending = &pending
	}
	if s.Suspended != nil {
		suspended := *s.Suspended
		next.Suspended = &suspended
	}
	next.Payouts = append([]Payout(nil), s.Payouts...)
	var err error
	switch command.Type {
	case "leave":
		next.Winner = 1 - next.Team(seat)
		next.Phase = MatchOver
	case "cut":
		if next.Phase != Cutting || seat != next.Turn || command.Cut < 1 || command.Cut >= 40 {
			err = fmt.Errorf("invalid cut")
		} else {
			err = next.deal(command.Cut)
		}
	case "play", "stack":
		err = next.play(seat, command, false)
	case "call":
		err = next.call(seat, command)
	case "respond":
		err = next.respond(seat, command)
	default:
		err = fmt.Errorf("unknown command")
	}
	if err != nil {
		return s, nil, err
	}
	next.Revision++
	return next, []Event{{Type: command.Type, Seat: seat, Text: command.Declaration}}, nil
}

// Timeout is a host-only entry point. choice supplies secure random selection:
// a cut in [1,39], or an index into the timed-out player's remaining cards.
func Timeout(s State, choice int) (State, []Event, error) {
	if s.Phase == Cutting {
		return Apply(s, s.Turn, Command{Type: "cut", Cut: choice})
	}
	if s.Phase != Playing {
		return s, nil, fmt.Errorf("no active timeout")
	}
	if s.Pending != nil {
		seat := (s.Pending.Caller + 1) % 2
		// A team timeout has no individual speaker; ordinary flores survive.
		// Flor-contest refusal still invalidates the team's ordinary flores.
		return Apply(s, seat, Command{Type: "respond", Kind: "decline", Declaration: "con_mi_flor"})
	}
	remaining := s.Remaining(s.Turn)
	if choice < 0 || choice >= len(remaining) {
		return s, nil, fmt.Errorf("invalid timeout choice")
	}
	command := Command{Type: "play", Card: remaining[choice]}
	if s.MandatoryStack {
		command.Type = "stack"
		command.Top = remaining[1]
		command.Card = remaining[0]
	}
	next := s
	if err := next.play(s.Turn, command, true); err != nil {
		return s, nil, err
	}
	next.Revision++
	return next, []Event{{Type: "timeout", Seat: s.Turn}}, nil
}

func (s State) Remaining(seat int) []int {
	var indices []int
	for i, used := range s.Players[seat].Used {
		if !used {
			indices = append(indices, i)
		}
	}
	return indices
}

func (s *State) deal(cut int) error {
	deck := append(append(make([]Card, 0, 40), s.Deck[cut:]...), s.Deck[:cut]...)
	draw := func() Card { card := deck[len(deck)-1]; deck = deck[:len(deck)-1]; return card }
	counts := [4]int{}
	if s.Config.Players == 4 {
		for _, viewer := range []int{s.left(s.Dealer), s.Dealer} {
			card := draw()
			receiver := (viewer + 2) % 4
			s.Preview[viewer] = &card
			s.Players[receiver].Cards[0] = card
			counts[receiver]++
		}
	}
	for dealt, seat := 0, s.left(s.Dealer); dealt < s.Config.Players*3; seat = s.right(seat) {
		if counts[seat] < 3 {
			s.Players[seat].Cards[counts[seat]] = draw()
			counts[seat]++
		}
		dealt = 0
		for _, count := range counts {
			dealt += count
		}
	}
	s.Vira = draw()
	for seat := 0; seat < s.Config.Players; seat++ {
		holding, err := Evaluate(s.Players[seat].Cards, s.Vira)
		if err != nil {
			return err
		}
		s.Players[seat].Holding = holding
		s.Players[seat].FlorValid = holding.Flor > 0
	}
	s.Phase, s.Turn = Playing, s.Mano
	return nil
}
