package cpu_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/cpu"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func table(t testing.TB, players, seat int, hand [3]rules.Card) rules.View {
	t.Helper()
	return tableWithVira(t, players, seat, hand, rules.Card{Suit: rules.Coins, Number: 4})
}

func tableWithVira(t testing.TB, players, seat int, hand [3]rules.Card, vira rules.Card) rules.View {
	t.Helper()
	h, err := rules.Evaluate(hand, vira)
	require.NoError(t, err)
	v := rules.View{Phase: rules.Playing, Config: rules.Config{Players: players, Target: 24}, Vira: &vira, Holding: &h, FlorValid: h.Flor > 0, Turn: seat, Mano: seat, TrucoValue: 1}
	for i, c := range hand {
		v.Hand = append(v.Hand, rules.HandCard{Index: i, Card: c})
		v.Legal = append(v.Legal, rules.Command{Type: "play", Card: i})
	}
	for i := 0; i < players; i++ {
		v.Seats = append(v.Seats, rules.SeatView{Seat: i, Team: i % 2, Remaining: 3})
	}
	return v
}

func played(s rules.Suit, n int) rules.PublicPlay {
	return rules.PublicPlay{Present: true, Card: &rules.Card{Suit: s, Number: n}}
}

func TestEnvidoUsesShowdownValueBeforeOpeningOrRaising(t *testing.T) {
	weak := [3]rules.Card{{Suit: rules.Cups, Number: 2}, {Suit: rules.Spades, Number: 12}, {Suit: rules.Clubs, Number: 10}}
	strong := [3]rules.Card{{Suit: rules.Cups, Number: 7}, {Suit: rules.Cups, Number: 6}, {Suit: rules.Spades, Number: 2}}
	for _, n := range []int{2, 4} {
		for _, tc := range []struct {
			name  string
			cards [3]rules.Card
			bet   bool
		}{{"two while mano", weak, false}, {"thirty-three", strong, true}} {
			t.Run(tc.name, func(t *testing.T) {
				v := table(t, n, 1, tc.cards)
				envido := rules.Command{Type: "call", Kind: "envido"}
				ley := rules.Command{Type: "call", Kind: "ley_envido"}
				v.Legal = append(v.Legal, envido, ley)
				choices := cpu.PlayingChoices(v, 1)
				for _, bid := range []rules.Command{envido, ley} {
					if tc.bet && bid.Kind == "envido" {
						assert.Contains(t, choices, bid)
					} else {
						assert.NotContains(t, choices, bid)
					}
					assert.Contains(t, v.Legal, bid, "the game must still allow human bluffs")
				}
				if !tc.bet {
					v.Pending = &rules.Wager{Kind: "envido", Caller: 0, Value: 2, Previous: 1}
					accept := rules.Command{Type: "respond", Kind: "accept"}
					decline := rules.Command{Type: "respond", Kind: "decline"}
					v.Legal = []rules.Command{accept, decline, {Type: "respond", Kind: "raise", Amount: 2}, {Type: "respond", Kind: "falta"}}
					if n == 2 {
						assert.Equal(t, []rules.Command{decline}, cpu.PlayingChoices(v, 1))
					} else {
						// A teammate can still win an ordinary two-point wager.
						assert.NotContains(t, cpu.PlayingChoices(v, 1), rules.Command{Type: "respond", Kind: "falta"})
					}
				}
			})
		}
	}
}

func TestThirdSeatKeepsAceUnderPartnersPerica(t *testing.T) {
	// Seat zero still has to play.
	v := table(t, 4, 3, [3]rules.Card{{Suit: rules.Spades, Number: 1}, {Suit: rules.Coins, Number: 5}, {Suit: rules.Cups, Number: 10}})
	v.Mano = 1
	v.Seats[1].Plays[0] = played(rules.Coins, 10)
	v.Seats[2].Plays[0] = played(rules.Coins, 12)
	for i := 0; i < 3; i++ {
		v.Legal = append(v.Legal, rules.Command{Type: "play", Card: i, Declaration: "a_ley"})
	}
	want := []rules.Command{{Type: "play", Card: 1}, {Type: "play", Card: 1, Declaration: "a_ley"}}
	require.Equal(t, want, cpu.PlayingChoices(v, 3))
	// The ace can cover a remaining opponent's 3 if the partner led a 2.
	v.Seats[1].Plays[0] = played(rules.Cups, 2)
	assert.Contains(t, cpu.PlayingChoices(v, 3), rules.Command{Type: "play", Card: 0})
	// With an opponent already above the partner, keep the winning card available.
	v.Seats[2].Plays[0] = played(rules.Clubs, 3)
	assert.Contains(t, cpu.PlayingChoices(v, 3), rules.Command{Type: "play", Card: 0})
}

func TestValePartidaNeedsBothMatchContextAndCards(t *testing.T) {
	strong := [3]rules.Card{{Suit: rules.Coins, Number: 11}, {Suit: rules.Coins, Number: 10}, {Suit: rules.Spades, Number: 1}}
	weak := [3]rules.Card{{Suit: rules.Cups, Number: 4}, {Suit: rules.Spades, Number: 5}, {Suit: rules.Clubs, Number: 6}}
	for _, tc := range []struct {
		name   string
		hand   [3]rules.Card
		stake  int
		scores [2]int
		allow  bool
	}{
		{"no opening jump even with the masters", strong, 3, [2]int{}, false},
		{"nine with poor cards", weak, 9, [2]int{}, false},
		{"nine and the two masters", strong, 9, [2]int{}, true},
		{"current stake already wins the match", strong, 9, [2]int{17, 10}, false},
		{"rival already playing for the match", strong, 6, [2]int{8, 19}, true},
		{"being behind alone is insufficient", weak, 6, [2]int{8, 19}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			v := table(t, 2, 0, tc.hand)
			v.Scores = tc.scores
			v.TrucoValue = tc.stake
			raise := rules.Command{Type: "call", Kind: "truco", Amount: 100}
			v.Legal = append(v.Legal, raise)
			check := func(choices []rules.Command, c rules.Command) {
				if tc.allow {
					assert.Contains(t, choices, c)
				} else {
					assert.NotContains(t, choices, c)
				}
			}
			check(cpu.PlayingChoices(v, 0), raise)
			v.Pending = &rules.Wager{Kind: "truco", Caller: 1, Value: tc.stake, Previous: 1}
			raise = rules.Command{Type: "respond", Kind: "raise", Amount: 100}
			v.Legal = []rules.Command{{Type: "respond", Kind: "decline"}, {Type: "respond", Kind: "accept"}, raise}
			check(cpu.PlayingChoices(v, 0), raise)
		})
	}
	v := table(t, 2, 0, weak)
	v.Pending = &rules.Wager{Kind: "truco", Caller: 1, Value: 100, Previous: 9}
	accept := rules.Command{Type: "respond", Kind: "accept"}
	v.Legal = []rules.Command{accept, {Type: "respond", Kind: "decline"}}
	assert.NotContains(t, cpu.PlayingChoices(v, 0), accept)
	v.Scores[1] = 16
	assert.Contains(t, cpu.PlayingChoices(v, 0), accept, "folding would itself lose the match")
}

func TestClosingChoiceKeepsAnAceWhenLosingAndUsesItWhenItCanDecideTheHand(t *testing.T) {
	v := table(t, 2, 0, [3]rules.Card{{Suit: rules.Spades, Number: 1}, {Suit: rules.Cups, Number: 5}, {Suit: rules.Clubs, Number: 4}})
	v.Mano = 1
	v.Seats[1].Plays[0] = played(rules.Coins, 10)
	choices := cpu.PlayingChoices(v, 0)
	require.Equal(t, []rules.Command{{Type: "play", Card: 2}}, choices)
	// We won the first. Either card can be played, but only the ace beats the 3 and wins now.
	v.Round = 1
	v.Seats[1].Plays[0] = played(rules.Cups, 4)
	v.Seats[0].Plays[0] = played(rules.Cups, 5)
	v.Hand[1].Used = true
	v.Seats[1].Plays[1] = played(rules.Cups, 3)
	v.Legal = []rules.Command{{Type: "play", Card: 0}, {Type: "play", Card: 2}, {Type: "burn", Card: 2}}
	assert.Equal(t, []rules.Command{{Type: "play", Card: 0}}, cpu.PlayingChoices(v, 0))
}

func BenchmarkPlayingChoicesWithBids(b *testing.B) {
	v := table(b, 4, 1, [3]rules.Card{{Suit: rules.Cups, Number: 7}, {Suit: rules.Cups, Number: 6}, {Suit: rules.Spades, Number: 2}})
	v.Legal = append(v.Legal, rules.Command{Type: "call", Kind: "truco", Amount: 3}, rules.Command{Type: "call", Kind: "envido"})
	for b.Loop() {
		cpu.PlayingChoices(v, 1)
	}
}

func TestTeamPegadoKeepsTheTopOnlyWhenTheBottomCanSurvive(t *testing.T) {
	// CPU 1 closes after its partner's king of oros.
	v := table(t, 4, 1, [3]rules.Card{{Suit: rules.Clubs, Number: 2}, {Suit: rules.Cups, Number: 6}, {Suit: rules.Clubs, Number: 3}})
	vira := rules.Card{Suit: rules.Cups, Number: 1}
	v.Vira, v.Round = &vira, 1
	v.Hand[0].Used = true
	v.Seats[0].Plays[1] = played(rules.Coins, 5)
	v.Seats[2].Plays[1] = played(rules.Clubs, 4)
	v.Seats[3].Plays[1] = played(rules.Coins, 12)
	single := rules.Command{Type: "play", Card: 1}
	lowBottom := rules.Command{Type: "stack", Card: 1, Top: 2}
	highBottom := rules.Command{Type: "stack", Card: 2, Top: 1}
	v.Legal = []rules.Command{single, lowBottom, highBottom}
	assert.NotContains(t, cpu.PlayingChoices(v, 1), lowBottom)
	assert.Contains(t, cpu.PlayingChoices(v, 1), single)
	assert.Contains(t, cpu.PlayingChoices(v, 1), highBottom)
	// A tie with only a teammate preserves both tops. A rival matching it burns them.
	v.Seats[3].Plays[1] = played(rules.Spades, 6)
	assert.Contains(t, cpu.PlayingChoices(v, 1), lowBottom)
	v.Seats[0].Plays[1] = played(rules.Coins, 6)
	assert.NotContains(t, cpu.PlayingChoices(v, 1), lowBottom)
	v.Seats[0].Plays[1] = rules.PublicPlay{Present: true, Concealed: true}
	assert.Contains(t, cpu.PlayingChoices(v, 1), lowBottom)
	v.MandatoryStack = true
	v.Seats[0].Plays[1] = played(rules.Spades, 1)
	v.Legal = []rules.Command{lowBottom, highBottom}
	assert.Equal(t, v.Legal, cpu.PlayingChoices(v, 1))
}

func TestTeamTrucoPlaysForTheAgreedMatchPoint(t *testing.T) {
	// Ellos led 23–14 and CPU 3 offered three.
	v := table(t, 4, 3, [3]rules.Card{{Suit: rules.Coins, Number: 11}, {Suit: rules.Coins, Number: 10}, {Suit: rules.Spades, Number: 1}})
	v.Scores = [2]int{14, 23}
	call := rules.Command{Type: "call", Kind: "truco", Amount: 3}
	v.Legal = append(v.Legal, call)
	assert.NotContains(t, cpu.PlayingChoices(v, 3), call)
	v.Scores[1] = 20
	assert.Contains(t, cpu.ChoicesForProfile(v, 3, cpu.FirstTrickValueProfile), call)
	assert.NotContains(t, cpu.PlayingChoices(v, 3), call)
	// The ace won the first trick, leaving both pieces for a genuine value call.
	v.Seats[0].Plays[0] = played(rules.Cups, 4)
	v.Seats[1].Plays[0] = played(rules.Cups, 5)
	v.Seats[2].Plays[0] = played(rules.Clubs, 4)
	v.Seats[3].Plays[0] = played(rules.Spades, 1)
	v.Hand[2].Used = true
	v.Round = 1
	assert.Contains(t, cpu.PlayingChoices(v, 3), call)
	// An opposing raise remains answerable, even when our own stake already suffices.
	v.Scores[1] = 21
	v.TrucoValue = 3
	v.Pending = &rules.Wager{Kind: "truco", Caller: 0, Value: 6, Previous: 3}
	accept := rules.Command{Type: "respond", Kind: "accept"}
	raise := rules.Command{Type: "respond", Kind: "raise", Amount: 9}
	v.Legal = []rules.Command{accept, {Type: "respond", Kind: "decline"}, raise}
	options := cpu.PlayingChoices(v, 3)
	assert.Contains(t, options, accept)
	assert.NotContains(t, options, raise)
}

func TestBTNI5121DoNotBurnPericoUnderShownPerica(t *testing.T) {
	v := table(t, 2, 1, [3]rules.Card{{Suit: rules.Coins, Number: 12}, {Suit: rules.Clubs, Number: 7}, {Suit: rules.Spades, Number: 11}})
	vira := rules.Card{Suit: rules.Spades, Number: 5}
	v.Vira = &vira
	v.Round = 1
	v.Scores = [2]int{21, 13}
	v.TrucoValue = 3
	v.Hand[0].Used = true
	v.Seats[1].Plays[0] = played(rules.Coins, 12)
	v.Seats[0].Plays[0] = played(rules.Spades, 1)
	v.Seats[0].Plays[1] = played(rules.Spades, 10)
	v.Legal = []rules.Command{{Type: "play", Card: 1}, {Type: "play", Card: 2}, {Type: "burn", Card: 1}, {Type: "burn", Card: 2},
		{Type: "stack", Card: 1, Top: 2}, {Type: "stack", Card: 2, Top: 1}}
	assert.Equal(t, []rules.Command{{Type: "play", Card: 2}}, cpu.PlayingChoices(v, 1))
	assert.Len(t, v.Legal, 6, "the CPU policy must not change the human player's legal choices")
}

func TestVoluntaryPegadoOnlyConcealsASecureBottom(t *testing.T) {
	v := table(t, 2, 1, [3]rules.Card{{Suit: rules.Coins, Number: 11}, {Suit: rules.Cups, Number: 7}, {Suit: rules.Spades, Number: 1}})
	v.Round = 1
	v.Hand[2].Used = true
	v.Seats[1].Plays[0] = played(rules.Spades, 1)
	v.Seats[0].Plays[0] = played(rules.Clubs, 6)
	safe := rules.Command{Type: "stack", Card: 0, Top: 1}
	risky := rules.Command{Type: "stack", Card: 1, Top: 0}
	v.Legal = []rules.Command{{Type: "play", Card: 0}, {Type: "play", Card: 1}, safe, risky}
	assert.Contains(t, cpu.PlayingChoices(v, 1), safe)
	assert.NotContains(t, cpu.PlayingChoices(v, 1), risky)
	v.MandatoryStack = true
	v.Seats[0].Plays[0] = played(rules.Cups, 1)
	v.Seats[1].Plays[0] = played(rules.Coins, 1)
	v.Legal = []rules.Command{safe, risky, {Type: "burn"}}
	assert.Equal(t, v.Legal, cpu.ChoicesForProfile(v, 1, cpu.TeamStackProfile), "replay the earlier policy")
	assert.Equal(t, []rules.Command{risky}, cpu.PlayingChoices(v, 1), "the perico now belongs on top")
}

func TestValueRaiseRequiresAProvenHandAndRoomForMorePoints(t *testing.T) {
	v := table(t, 2, 1, [3]rules.Card{{Suit: rules.Coins, Number: 11}, {Suit: rules.Clubs, Number: 7}, {Suit: rules.Spades, Number: 12}})
	v.Scores = [2]int{21, 13}
	v.Pending = &rules.Wager{Kind: "envido", Caller: 0, Value: 4, Previous: 2}
	accept := rules.Command{Type: "respond", Kind: "accept"}
	raise := rules.Command{Type: "respond", Kind: "raise", Amount: 2}
	v.Legal = []rules.Command{accept, {Type: "respond", Kind: "decline"}, raise}
	require.Equal(t, 37, v.Holding.Envido)
	assert.Equal(t, []rules.Command{raise}, cpu.PlayingChoices(v, 1))
	winMatch := rules.Command{Type: "respond", Kind: "raise", Amount: 7}
	excess := rules.Command{Type: "respond", Kind: "raise", Amount: 8}
	v.Legal = append(v.Legal, winMatch, excess)
	assert.Equal(t, []rules.Command{raise, winMatch}, cpu.PlayingChoices(v, 1))
	assert.Contains(t, cpu.ChoicesForProfile(v, 1, cpu.HeadsUpCardProfile), excess)
	v.Scores[1] = 20
	assert.Contains(t, cpu.PlayingChoices(v, 1), accept)
	v.Scores[1] = 13
	v.Pending.Terminal = true
	v.Legal = v.Legal[:2]
	assert.Contains(t, cpu.PlayingChoices(v, 1), accept)
	v.Pending.Terminal = false
	v.Pending.Kind = "ley_envido"
	v.Pending.Mode = "con_ley"
	v.Legal = []rules.Command{accept, {Type: "respond", Kind: "decline"}, raise}
	assert.Contains(t, cpu.PlayingChoices(v, 1), rules.Command{Type: "respond", Kind: "decline"})
}

func TestCompulsoryPegadoShowsThePerico(t *testing.T) {
	// The public top is compared before either bottom.
	for _, players := range []int{2, 4} {
		seat := players - 2
		vira := rules.Card{Suit: rules.Spades, Number: 2}
		v := tableWithVira(t, players, seat, [3]rules.Card{{Suit: rules.Spades, Number: 6}, {Suit: rules.Spades, Number: 11}, {Suit: rules.Coins, Number: 4}}, vira)
		v.Round, v.MandatoryStack = 1, true
		v.Hand[0].Used = true
		top := rules.Command{Type: "stack", Card: 2, Top: 1}
		bottom := rules.Command{Type: "stack", Card: 1, Top: 2}
		v.Legal = []rules.Command{top, bottom, {Type: "burn"}}
		assert.Equal(t, []rules.Command{top}, cpu.PlayingChoices(v, seat))
		assert.Len(t, v.Legal, 3)
		// The king replaces the perico when the eleven is the vira.
		v.Vira = &rules.Card{Suit: rules.Spades, Number: 11}
		v.Hand[1].Card.Number = 12
		assert.Equal(t, []rules.Command{top}, cpu.PlayingChoices(v, seat))
		v.Hand[1].Card.Number = 10
		assert.Contains(t, cpu.PlayingChoices(v, seat), bottom, "the perica can still lose")
	}
}

func TestAgreedHeadsUpTrucoAlreadyWinsMatch(t *testing.T) {
	// CPU 1 offered nine after accepting three at 21–18.
	v := table(t, 2, 1, [3]rules.Card{{Suit: rules.Spades, Number: 3}, {Suit: rules.Cups, Number: 11}, {Suit: rules.Cups, Number: 6}})
	v.Vira = &rules.Card{Suit: rules.Cups, Number: 5}
	v.Mano, v.TrucoValue, v.Scores = 0, 3, [2]int{18, 21}
	v.Seats[0].Plays[0] = played(rules.Spades, 2)
	raise := rules.Command{Type: "call", Kind: "truco", Amount: 9}
	v.Legal = append(v.Legal, raise)
	assert.NotContains(t, cpu.PlayingChoices(v, 1), raise)
	v.Scores[1] = 20
	assert.Contains(t, cpu.PlayingChoices(v, 1), raise)
}

func TestEmptyLeyOnLastFirstCard(t *testing.T) {
	v := table(t, 4, 3, [3]rules.Card{{Suit: rules.Spades, Number: 4}, {Suit: rules.Coins, Number: 10}, {Suit: rules.Clubs, Number: 2}})
	v.Mano = 0
	v.Seats[0].Plays[0] = played(rules.Coins, 11)
	v.Seats[1].Plays[0] = played(rules.Clubs, 10)
	v.Seats[2].Plays[0] = played(rules.Coins, 7)
	plain := rules.Command{Type: "play"}
	ley := rules.Command{Type: "play", Declaration: "a_ley"}
	v.Legal = []rules.Command{plain, ley}
	assert.Equal(t, []rules.Command{plain}, cpu.PlayingChoices(v, 3))
	v.Mano = 1
	v.Seats[0].Plays[0] = rules.PublicPlay{}
	assert.Contains(t, cpu.PlayingChoices(v, 3), ley, "a later player can still open a qualified envido")
	v.Mano = 0
	v.Seats[0].Plays[0] = played(rules.Coins, 11)
	v.FlorValid, v.Holding.Flor = true, 31
	assert.Contains(t, cpu.PlayingChoices(v, 3), ley, "the speaker can preserve an ordinary flor")
}

func TestRecoverSecondTrickWithExposedThree(t *testing.T) {
	// Deal 183245, hand 5. The opponents won first and led second with a two.
	// Putting the three under a four gives up the choice of how to play third.
	vira := rules.Card{Suit: rules.Spades, Number: 12}
	v := tableWithVira(t, 4, 1, [3]rules.Card{{Suit: rules.Coins, Number: 2}, {Suit: rules.Clubs, Number: 4}, {Suit: rules.Clubs, Number: 3}}, vira)
	v.Round, v.Mano, v.Config.Target, v.Scores = 1, 1, 12, [2]int{4, 8}
	v.Hand[0].Used = true
	v.Seats[0].Plays[0], v.Seats[0].Plays[1] = played(rules.Coins, 3), played(rules.Cups, 2)
	v.Seats[1].Plays[0] = played(rules.Coins, 2)
	v.Seats[2].Plays[0] = played(rules.Coins, 4)
	v.Seats[3].Plays[0] = played(rules.Coins, 11)
	single := rules.Command{Type: "play", Card: 2}
	stack := rules.Command{Type: "stack", Card: 2, Top: 1}
	v.Legal = []rules.Command{{Type: "play", Card: 1}, {Type: "burn", Card: 1}, {Type: "stack", Card: 1, Top: 2}, single, {Type: "burn", Card: 2}, stack, {Type: "call", Kind: "truco", Amount: 3}}
	assert.Contains(t, cpu.ChoicesForProfile(v, 1, cpu.CompulsoryPericoProfile), stack)
	assert.Equal(t, []rules.Command{single}, cpu.PlayingChoices(v, 1))
	assert.Contains(t, v.Legal, stack, "the same pegado remains legal at the table")
}
