package rules_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Exercise the command boundary and the exact candidate list used by the model.
// A real flor does not make every flor phrase legal in every action context.
func TestFlorDeclarationMatchesActionContext(t *testing.T) {
	first := fixture(t, true,
		[3]rules.Card{{rules.Clubs, 1}, {rules.Clubs, 5}, {rules.Cups, 4}},
		[3]rules.Card{{rules.Coins, 3}, {rules.Coins, 7}, {rules.Coins, 6}},
	)
	second := apply(t, first, 1, rules.Command{Type: "play", Card: 0, Declaration: "flor"})
	second = apply(t, second, 0, rules.Command{Type: "play", Card: 0})
	second = apply(t, second, 0, rules.Command{Type: "play", Card: 1})
	third := apply(t, second, 1, rules.Command{Type: "play", Card: 1, Declaration: "flor"})
	responding := fixture(t, true,
		[3]rules.Card{{rules.Coins, 3}, {rules.Coins, 7}, {rules.Coins, 6}},
		[3]rules.Card{{rules.Clubs, 1}, {rules.Clubs, 5}, {rules.Cups, 4}},
	)
	truco := apply(t, responding, 1, rules.Command{Type: "call", Kind: "truco"})
	envido := apply(t, responding, 1, rules.Command{Type: "call", Kind: "envido"})
	contest := opposingFlores(t)
	for _, tc := range []struct {
		name        string
		state       rules.State
		seat        int
		command     rules.Command
		declaration string
	}{
		{"first card", first, 1, rules.Command{Type: "play", Card: 0}, "flor"},
		{"second card", second, 1, rules.Command{Type: "play", Card: 1}, "flor"},
		{"last card", third, 1, rules.Command{Type: "play", Card: 2}, "presento_flor"},
		{"burn one", second, 1, rules.Command{Type: "burn", Card: 1}, "flor"},
		{"burn last", third, 1, rules.Command{Type: "burn", Card: 2}, "presento_flor"},
		{"pegado", second, 1, rules.Command{Type: "stack", Card: 1, Top: 2}, "flor_flor"},
		{"truco", first, 1, rules.Command{Type: "call", Kind: "truco", Amount: 3}, "con_mi_flor"},
		{"ordinary accept", truco, 0, rules.Command{Type: "respond", Kind: "accept"}, "con_mi_flor"},
		{"ordinary refusal", truco, 0, rules.Command{Type: "respond", Kind: "decline"}, "con_mi_flor"},
		{"vale partida", truco, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 100}, "con_mi_flor"},
		{"intrinsic flowers no", envido, 0, rules.Command{Type: "respond", Kind: "flowers_no"}, ""},
		{"flor accept", contest, 1, rules.Command{Type: "respond", Kind: "accept", Mode: "con_ley"}, ""},
		{"flor raise", contest, 1, rules.Command{Type: "respond", Kind: "raise", Amount: 2, Mode: "con_ley"}, ""},
		{"flor falta", contest, 1, rules.Command{Type: "respond", Kind: "falta", Mode: "con_ley"}, ""},
		{"intrinsic flor refusal", contest, 1, rules.Command{Type: "respond", Kind: "decline", Mode: "con_ley"}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, declaration := range []string{"", "flor", "presento_flor", "flor_flor", "con_mi_flor", "con_flor"} {
				command := tc.command
				command.Declaration = declaration
				if declaration != "" && declaration != tc.declaration {
					assertRejectedCanto(t, tc.state, tc.seat, command)
					continue
				}
				require.Contains(t, rules.LegalActions(tc.state, tc.seat), command)
				next, events, err := rules.Apply(tc.state, tc.seat, command)
				require.NoError(t, err)
				require.NotEmpty(t, events)
				assert.Equal(t, declaration, events[0].Text)
				if declaration != "" {
					assert.True(t, next.Players[tc.seat].FlorValid)
				}
			}
		})
	}
}
