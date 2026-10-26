package rules_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
		{Type: "respond", Kind: "raise", Amount: 9},
		{Type: "respond", Kind: "raise", Amount: 100},
	}, replies)
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
	conLey := apply(t, ley, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 2, Mode: "con_ley"})
	sinLey := apply(t, ley, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 2, Mode: "sin_ley"})
	for _, scenario := range []struct {
		name          string
		state         rules.State
		seat          int
		chooseLey     bool
		sinLeyDecline bool
	}{
		{name: "ordinary envido", state: ordinary, seat: 0},
		{name: "ordinary envido after a ley card", state: ordinaryAfterALey, seat: 1, sinLeyDecline: true},
		{name: "responders teammate called a ley", state: respondersTeammate, seat: 3, sinLeyDecline: true},
		{name: "callers teammate called a ley", state: callersTeammate, seat: 0, sinLeyDecline: true},
		{name: "flor contest", state: opposingFlores(t), seat: 1},
		{name: "a ley awaiting choice", state: ley, seat: 0, chooseLey: true, sinLeyDecline: true},
		{name: "con ley already chosen", state: conLey, seat: 1, sinLeyDecline: true},
		{name: "sin ley already chosen", state: sinLey, seat: 1, sinLeyDecline: true},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			var accepts, declines []string
			for _, command := range rules.LegalActions(scenario.state, scenario.seat) {
				if command.Type != "respond" || command.Declaration != "" {
					continue
				}
				if command.Kind == "accept" {
					accepts = append(accepts, command.Mode)
				}
				if command.Kind == "decline" {
					declines = append(declines, command.Mode)
				}
				if command.Kind == "decline" && (scenario.state.Pending.Kind == "envido" || scenario.state.Pending.Kind == "ley_envido") {
					continue
				}
				if !scenario.chooseLey || command.Kind == "decline" || command.Kind == "flowers_no" {
					assert.Empty(t, command.Mode, "reply %+v must not offer an irrelevant ley choice", command)
				}
			}
			if scenario.sinLeyDecline {
				assert.ElementsMatch(t, []string{"", "sin_ley"}, declines, "a previous A ley permits Sin ley, no quiero")
				apply(t, scenario.state, scenario.seat, rules.Command{Type: "respond", Kind: "decline", Mode: "sin_ley"})
			} else {
				assert.Equal(t, []string{""}, declines)
				for _, declaration := range []string{"", "flor", "presento_flor", "flor_flor", "con_mi_flor", "con_flor"} {
					assertRejectedCanto(t, scenario.state, scenario.seat, rules.Command{Type: "respond", Kind: "decline", Mode: "sin_ley", Declaration: declaration})
				}
			}
			if scenario.chooseLey {
				assert.ElementsMatch(t, []string{"con_ley", "sin_ley"}, accepts)
			} else {
				assert.Equal(t, []string{""}, accepts)
				before := scenario.state
				after, _, err := rules.Apply(before, scenario.seat, rules.Command{Type: "respond", Kind: "accept", Mode: "con_ley"})
				require.Error(t, err, "the engine must reject a mode that is not being chosen")
				assert.Equal(t, before, after)
			}
		})
	}
}
