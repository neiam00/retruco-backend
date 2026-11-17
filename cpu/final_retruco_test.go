package cpu_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/cpu"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/require"
)

func TestRetrucoFacingAnExposedLastCard(t *testing.T) {
	// CPU 1 won second and has exposed its 11 of cups. The opposing third
	// card (later revealed as the 7 of spades) is still private.
	vira := rules.Card{Suit: rules.Clubs, Number: 11}
	hand := [3]rules.Card{{Suit: rules.Spades, Number: 5}, {Suit: rules.Coins, Number: 7}, {Suit: rules.Cups, Number: 11}}
	v := tableWithVira(t, 2, 1, hand, vira)
	v.Round, v.Turn, v.Mano, v.TrucoValue = 2, 0, 1, 3
	v.Seats[0].Plays[0], v.Seats[0].Plays[1] = played(rules.Clubs, 6), played(rules.Coins, 11)
	for i, c := range hand {
		v.Hand[i].Used = true
		v.Seats[1].Plays[i] = played(c.Suit, c.Number)
	}
	v.Seats[1].Remaining, v.Seats[0].Remaining = 0, 1
	v.Pending = &rules.Wager{Kind: "truco", Caller: 0, Value: 6, Previous: 3}
	accept, fold := rules.Command{Type: "respond", Kind: "accept"}, rules.Command{Type: "respond", Kind: "decline"}
	raise := rules.Command{Type: "respond", Kind: "raise", Amount: 9}
	v.Legal = []rules.Command{accept, fold, raise}
	require.Equal(t, []rules.Command{fold}, cpu.PlayingChoices(v, 1))
	// Refusing three would give the opponent the match.
	v.Scores[0] = v.Config.Target - 3
	require.Contains(t, cpu.PlayingChoices(v, 1), accept)
	require.NotContains(t, cpu.PlayingChoices(v, 1), fold)
	// An exposed perico really does win the deciding trick.
	v.Scores[0] = 0
	perico, _ := rules.Pieces(vira)
	v.Hand[2].Card = perico
	v.Seats[1].Plays[2] = played(perico.Suit, perico.Number)
	require.Contains(t, cpu.PlayingChoices(v, 1), accept)
	require.NotContains(t, cpu.PlayingChoices(v, 1), fold)
}

func TestLastCardRetrucoIncludesTeammateAndFirstTrickTieBreak(t *testing.T) {
	vira := rules.Card{Suit: rules.Clubs, Number: 11}
	hand := [3]rules.Card{{Suit: rules.Spades, Number: 5}, {Suit: rules.Coins, Number: 7}, {Suit: rules.Cups, Number: 11}}
	v := tableWithVira(t, 4, 1, hand, vira)
	v.Round, v.Mano, v.Turn, v.TrucoValue = 2, 1, 0, 3
	for i, c := range hand {
		v.Hand[i].Used = true
		v.Seats[1].Plays[i] = played(c.Suit, c.Number)
	}
	v.Seats[0].Plays[0], v.Seats[0].Plays[1] = played(rules.Clubs, 6), played(rules.Coins, 11)
	v.Seats[2].Plays[0], v.Seats[2].Plays[1], v.Seats[2].Plays[2] = played(rules.Coins, 4), played(rules.Cups, 5), played(rules.Cups, 12)
	v.Seats[3].Plays[0], v.Seats[3].Plays[1], v.Seats[3].Plays[2] = played(rules.Cups, 4), played(rules.Spades, 6), played(rules.Spades, 11)
	v.Pending = &rules.Wager{Kind: "truco", Caller: 0, Value: 6, Previous: 3}
	yes, no := rules.Command{Type: "respond", Kind: "accept"}, rules.Command{Type: "respond", Kind: "decline"}
	v.Legal = []rules.Command{yes, no}
	// An opponent already covered both CPUs; the last rival cannot undo it.
	require.Equal(t, []rules.Command{no}, cpu.PlayingChoices(v, 1))
	// The two teams tie on top. The opponents won first, so the CPU still loses.
	v.Seats[2].Plays[2] = played(rules.Coins, 10)
	v.Seats[3].Plays[2] = played(rules.Cups, 10)
	v.Hand[2].Card = rules.Card{Suit: rules.Spades, Number: 10}
	v.Seats[1].Plays[2] = played(rules.Spades, 10)
	require.Equal(t, []rules.Command{no}, cpu.PlayingChoices(v, 1))
	// A partner's visible perico wins the deciding trick regardless of our own card.
	v.Seats[3].Plays[2] = played(rules.Clubs, 12)
	require.Equal(t, []rules.Command{yes}, cpu.PlayingChoices(v, 1))
}
