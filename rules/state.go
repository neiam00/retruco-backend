package rules

import "fmt"

const Version = "2"

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
	FlorEnvidada bool
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
	LastHand       *HandRecap `json:",omitempty"`
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
	BareLey        bool // The preceding card was played with A ley; consumed by the next action.
	DoubleFlor     [2]bool
	FlorClaimant   [2]int
	MandatoryStack bool
	Payouts        []Payout
	EnvidoResults  []EnvidoResult
	Privando       [2]bool
}

// EnvidoResult reveals only the value used at settlement, never hidden cards.
type EnvidoResult struct {
	Seat     int  `json:"seat"`
	Value    int  `json:"value"`
	Eligible bool `json:"eligible"`
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
	Type      string `json:"type"`
	Seat      int    `json:"seat"`
	Text      string `json:"text,omitempty"`
	Kind      string `json:"kind,omitempty"`
	Canto     string `json:"canto,omitempty"`
	Amount    int    `json:"amount,omitempty"`
	Mode      string `json:"mode,omitempty"`
	Automatic bool   `json:"automatic,omitempty"`
	Team      *int   `json:"team,omitempty"`
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
		LastHand:     s.LastHand,
		Config:       s.Config,
		Phase:        Cutting,
		Revision:     s.Revision + 1,
		HandNumber:   s.HandNumber + 1,
		Dealer:       dealer,
		Scores:       s.Scores,
		Winner:       -1,
		HandWinner:   -1,
		TrucoValue:   1,
		TrucoVoice:   -1,
		EnvidoWinner: -1,
		FlorClaimant: [2]int{-1, -1},
	}
	next.Mano = next.right(dealer)
	next.Turn = next.left(dealer)
	copy(next.Deck[:], deck)
	return next, nil
}

// RestoreActiveHand reconciles decisions still pending in an older saved hand.
// Settled hands and cards already played keep their recorded outcomes.
func (s *State) RestoreActiveHand(invalidOrdinaryOpening bool) {
	if s.Phase != Playing {
		return
	}
	changed := false
	if invalidOrdinaryOpening && s.Pending != nil && s.Pending.Kind == "envido" {
		s.resumeTruco()
		changed = true
	}
	if s.Round == 1 && s.MandatoryStack {
		if winner := s.roundWinner(0); winner >= 0 {
			s.MandatoryStack = false
			for offset := 0; offset < s.Config.Players; offset++ {
				seat := (winner + offset) % s.Config.Players
				if !s.Players[seat].Plays[1].Present {
					s.Turn = seat
					break
				}
			}
			changed = true
		}
	}
	contest := s.EnvidoAccepted && (s.Envido.Kind == "flor" || s.Envido.Mode == "con_ley") ||
		s.Pending != nil && (s.Pending.Mode == "con_ley" || s.Pending.Kind == "flor" && s.Pending.Previous >= 2)
	if contest && s.protectContestedFlores() {
		changed = true
	}
	if changed {
		s.Revision++
	}
}

// Apply validates a command against a value copy. Rejected commands leave the
// input state unchanged, including its revision and active response window.
func Apply(s State, seat int, command Command) (State, []Event, error) {
	return applyCommand(s, seat, command, false)
}

// automatic is internal to the response clock: it never manufactures spoken
// declarations, and cannot be requested through a client/model command.
func applyCommand(s State, seat int, command Command, automatic bool) (State, []Event, error) {
	if seat < 0 || seat >= s.Config.Players || s.Phase == MatchOver {
		return s, nil, fmt.Errorf("invalid player or finished match")
	}
	// The spoken flor reply to an explicit wager is the same decision as Con ley, quiero.
	if s.Pending != nil && s.Pending.Kind == "ley_envido" && s.Pending.Mode == "" &&
		command.Type == "call" && command.Kind == "flor" && command.Declaration == "" && command.Mode == "" && command.Amount == 0 {
		command = Command{Type: "respond", Kind: "accept", Mode: "con_ley"}
	}
	if err := s.validateDeclaration(seat, command); err != nil {
		return s, nil, err
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
		next.recordHand(s.Scores, "forfeit")
	case "cut":
		if next.Phase != Cutting || seat != next.Turn || command.Cut < 1 || command.Cut >= 40 {
			err = fmt.Errorf("invalid cut")
		} else {
			err = next.deal(command.Cut)
		}
	case "play", "stack", "burn":
		err = next.play(seat, command, false)
	case "call":
		err = next.call(seat, command)
	case "respond":
		err = next.respond(seat, command, automatic)
	default:
		err = fmt.Errorf("unknown command")
	}
	if err != nil {
		return s, nil, err
	}
	if command.Type != "play" {
		next.BareLey = false
	}
	next.Revision++
	events := []Event{publicEvent(seat, command, s.Pending)}
	if command.Type == "cut" {
		for _, announcement := range next.privoAnnouncements() {
			team := announcement.Team
			events = append(events, Event{Type: "privo", Seat: -1, Team: &team,
				Kind: "envido", Amount: announcement.Value, Automatic: true})
		}
		if next.Phase == MatchOver {
			team := next.Winner
			events = append(events, Event{Type: "privo", Seat: -1, Team: &team,
				Kind: "flor", Automatic: true})
		}
	}
	return next, events, nil
}

// A declaration asserts an eligible holding; it is not an unrestricted label.
// Apply owns this check so clients, CPUs, and training LegalActions share it.
func (s State) validateDeclaration(seat int, command Command) error {
	if command.Declaration == "" {
		return nil // Omitting a required declaration still carries its normal penalty.
	}
	if command.Declaration == "a_ley" {
		if command.Type != "play" || s.Round != 0 || s.Players[seat].FirstPlayed {
			return fmt.Errorf("a ley is only a first-card declaration")
		}
		return nil // A ley means the player may or may not hold flor.
	}
	if !declaresFlor(command.Declaration) {
		return fmt.Errorf("unknown declaration")
	}
	switch command.Type {
	case "play", "stack", "burn", "call", "respond":
	default:
		return fmt.Errorf("this action cannot declare flor")
	}
	player := s.Players[seat]
	if !player.FlorValid || player.Holding.Flor == 0 || player.Holding.Receta {
		return fmt.Errorf("flor declaration requires a valid ordinary flor")
	}
	expected := ""
	switch command.Type {
	case "play", "burn":
		expected = "flor"
		if command.Type == "burn" && s.MandatoryStack {
			expected = "flor_flor"
		} else if len(s.Remaining(seat)) == 1 {
			expected = "presento_flor"
		}
	case "stack":
		expected = "flor_flor"
	case "call":
		if command.Kind == "truco" {
			expected = "con_mi_flor"
		}
	case "respond":
		if (command.Kind == "accept" || command.Kind == "decline" || command.Kind == "raise" || command.Kind == "falta") &&
			s.Pending != nil && (s.Pending.Kind == "truco" || s.Pending.Kind == "envido") {
			expected = "con_mi_flor"
		}
	}
	if command.Declaration != expected {
		return fmt.Errorf("flor declaration does not match this action")
	}
	return nil
}

// Public events describe declarations and negotiations. They never contain a
// hand, card index, timeout choice, or arbitrary client-supplied text.
func publicEvent(seat int, command Command, pending *Wager) Event {
	event := Event{Type: command.Type, Seat: seat}
	switch command.Declaration {
	case "flor", "a_ley", "presento_flor", "flor_flor", "con_mi_flor", "con_flor":
		event.Text = command.Declaration
	}
	if command.Type == "call" || command.Type == "respond" {
		event.Kind = command.Kind
		event.Amount = command.Amount
		if command.Mode == "con_ley" || command.Mode == "sin_ley" {
			event.Mode = command.Mode
		}
		if pending != nil {
			event.Canto = pending.Kind
		}
	}
	return event
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
		next, events, err := applyCommand(s, seat, Command{Type: "respond", Kind: "decline"}, true)
		if len(events) > 0 {
			events[0].Automatic = true
			events[0].Text = ""
		}
		return next, events, err
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
	return next, []Event{{Type: "timeout", Seat: s.Turn, Automatic: true}}, nil
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
	draw := func() Card {
		card := deck[len(deck)-1]
		deck = deck[:len(deck)-1]
		return card
	}
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
	s.beginPrivo()
	return nil
}
