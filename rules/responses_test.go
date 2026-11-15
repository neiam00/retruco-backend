package rules_test

import (
	"slices"
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
)

func TestTrucoOffersEachReplyOnceWithoutLeyVariants(t *testing.T) {
	state := apply(t, simple(t), 1, rules.Command{Type: "call", Kind: "truco"})
	var replies []rules.Command
	for _, command := range rules.LegalActions(state, 0) {
		if command.Type == "respond" && command.Declaration == "" {
			replies = append(replies, command)
		}
	}
	assert.ElementsMatch(t, []rules.Command{
		{Type: "respond", Kind: "accept"},
		{Type: "respond", Kind: "decline"},
		{Type: "respond", Kind: "raise", Amount: 6},
		{Type: "respond", Kind: "raise", Amount: 100},
	}, replies)
	assertRejectedCanto(t, state, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 9})
	retruco := apply(t, state, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 6})
	nine := apply(t, retruco, 1, rules.Command{Type: "respond", Kind: "raise", Amount: 9})
	assert.Equal(t, 6, nine.Pending.Previous)
	settled := apply(t, nine, 0, rules.Command{Type: "respond", Kind: "decline"})
	assert.Equal(t, 6, settled.Scores[1], "declining Vale 9 pays the previously offered Retruco")
}

func TestLeyResponseChoicesExistOnlyUntilTheModeIsChosen(t *testing.T) {
	ordinary := apply(t, simple(t), 1, rules.Command{Type: "call", Kind: "envido"})
	previousALey := apply(t, simple(t), 1, rules.Command{Type: "play", Card: 0, Declaration: "a_ley"})
	ordinaryAfterALey := apply(t, previousALey, 0, rules.Command{Type: "call", Kind: "envido"})
	four := fixture(t, false,
		[3]rules.Card{{rules.Clubs, 1}, {rules.Coins, 6}, {rules.Cups, 5}},
		[3]rules.Card{{rules.Spades, 1}, {rules.Coins, 7}, {rules.Cups, 6}},
		[3]rules.Card{{rules.Coins, 1}, {rules.Clubs, 6}, {rules.Spades, 5}},
		[3]rules.Card{{rules.Cups, 1}, {rules.Spades, 6}, {rules.Coins, 5}},
	)
	four = apply(t, four, 1, rules.Command{Type: "play", Card: 0, Declaration: "a_ley"})
	respondersTeammate := apply(t, four, 2, rules.Command{Type: "call", Kind: "envido"})
	four = apply(t, four, 2, rules.Command{Type: "play", Card: 0})
	callersTeammate := apply(t, four, 3, rules.Command{Type: "call", Kind: "envido"})
	ley := apply(t, simple(t), 1, rules.Command{Type: "call", Kind: "ley_envido"})
	flor := opposingFlores(t)
	conLey := apply(t, flor, 1, rules.Command{Type: "respond", Kind: "raise", Amount: 2, Mode: "con_ley"})
	sinLey := apply(t, ley, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 2, Mode: "sin_ley"})
	for _, tc := range []struct {
		name              string
		state             rules.State
		seat              int
		accepts, declines []string
	}{
		{"ordinary", ordinary, 0, []string{""}, []string{""}},
		{"ordinary after A ley", ordinaryAfterALey, 1, []string{""}, []string{"", "sin_ley"}},
		{"responder's teammate said A ley", respondersTeammate, 3, []string{""}, []string{"", "sin_ley"}},
		{"caller's teammate said A ley", callersTeammate, 0, []string{""}, []string{"", "sin_ley"}},
		{"holding flor", flor, 1, []string{"con_ley"}, []string{"con_ley"}},
		{"without flor", ley, 0, []string{"sin_ley"}, []string{"sin_ley"}},
		{"raised con ley", conLey, 0, []string{""}, []string{""}},
		{"raised sin ley", sinLey, 1, []string{""}, []string{""}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var accepts, declines []string
			for _, cmd := range rules.LegalActions(tc.state, tc.seat) {
				if cmd.Type != "respond" || cmd.Declaration != "" {
					continue
				}
				switch cmd.Kind {
				case "accept":
					accepts = append(accepts, cmd.Mode)
				case "decline":
					declines = append(declines, cmd.Mode)
				}
			}
			assert.ElementsMatch(t, tc.accepts, accepts)
			assert.ElementsMatch(t, tc.declines, declines)
			for _, mode := range []string{"", "sin_ley", "con_ley"} {
				for _, kind := range []string{"accept", "decline"} {
					cmd := rules.Command{Type: "respond", Kind: kind, Mode: mode}
					_, _, err := rules.Apply(tc.state, tc.seat, cmd)
					assert.Equal(t, slices.Contains(rules.LegalActions(tc.state, tc.seat), cmd), err == nil)
				}
			}
		})
	}
}
