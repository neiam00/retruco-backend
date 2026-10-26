package rules_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDeclaredFlorBlocksNewEnvidoForTheWholeTable(t *testing.T) {
	for _, required := range []bool{false, true} {
		for _, relation := range []string{"self", "opponent", "teammate", "burned"} {
			if relation == "burned" && !required {
				continue // Omitting flor only burns it with flor por derecho.
			}
			t.Run(map[bool]string{false: "free", true: "required"}[required]+"/"+relation, func(t *testing.T) {
				s := fixture(t, required,
					[3]rules.Card{{rules.Clubs, 1}, {rules.Coins, 6}, {rules.Cups, 5}},
					[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 4}},
					[3]rules.Card{{rules.Spades, 1}, {rules.Coins, 7}, {rules.Cups, 3}},
					[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 4}},
				)
				s = apply(t, s, 1, rules.Command{Type: "call", Kind: "truco", Declaration: "con_mi_flor"})
				s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "accept"})
				if relation != "self" {
					declaration := "a_ley"
					if relation == "burned" {
						declaration = ""
					}
					s = apply(t, s, 1, rules.Command{Type: "play", Card: 0, Declaration: declaration})
					if relation == "burned" {
						require.False(t, s.Players[1].FlorValid)
					}
				}
				if relation == "teammate" {
					s = apply(t, s, 2, rules.Command{Type: "play", Card: 0})
					require.True(t, s.Players[3].FlorValid, "even a teammate with flor must use the flor contest")
				}
				for _, command := range []rules.Command{
					{Type: "call", Kind: "envido"},
					{Type: "call", Kind: "envido", Declaration: "con_mi_flor"},
					{Type: "call", Kind: "envido", Mode: "sin_ley"},
					{Type: "call", Kind: "ley_envido"},
					{Type: "call", Kind: "ley_envido", Mode: "con_ley", Declaration: "con_mi_flor"},
					{Type: "call", Kind: "ley_envido", Mode: "sin_ley"},
				} {
					assertRejectedCanto(t, s, s.Turn, command)
				}
				for _, action := range rules.LegalActions(s, s.Turn) {
					if action.Type == "call" {
						assert.NotContains(t, []string{"envido", "ley_envido"}, action.Kind)
					}
				}
			})
		}
	}
}

func TestDeclaredFlorPreventsEnvidoFromInterruptingTruco(t *testing.T) {
	s := fixture(t, true,
		[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}},
		[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}},
	)
	s = apply(t, s, 1, rules.Command{Type: "call", Kind: "truco", Declaration: "con_mi_flor"})
	assertRejectedCanto(t, s, 0, rules.Command{Type: "call", Kind: "envido"})
	assertRejectedCanto(t, s, 0, rules.Command{Type: "call", Kind: "ley_envido"})
	s = apply(t, s, 0, rules.Command{Type: "call", Kind: "flor"})
	require.NotNil(t, s.Suspended)
	assert.Equal(t, "truco", s.Suspended.Kind)
}

func TestALeyDoesNotDeclareFlorOrPreventEnvido(t *testing.T) {
	for _, required := range []bool{false, true} {
		t.Run(map[bool]string{false: "free", true: "required"}[required], func(t *testing.T) {
			s := fixture(t, required,
				[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}},
				[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}},
			)
			s = apply(t, s, 1, rules.Command{Type: "play", Card: 0, Declaration: "a_ley"})
			assert.False(t, s.Players[1].FlorDeclared, "A ley only says the player might hold flor")
			assert.True(t, s.Players[1].FlorValid)
			assertRejectedCanto(t, s, 0, rules.Command{Type: "call", Kind: "flor"})
			for _, command := range []rules.Command{
				{Type: "call", Kind: "envido", Declaration: "con_mi_flor"},
				{Type: "call", Kind: "ley_envido"},
				{Type: "call", Kind: "ley_envido", Mode: "con_ley"},
				{Type: "call", Kind: "ley_envido", Mode: "sin_ley"},
			} {
				require.Contains(t, rules.LegalActions(s, 0), command)
				next := apply(t, s, 0, command)
				assert.Equal(t, command.Kind, next.Pending.Kind)
			}
		})
	}
}

func TestDeclaredFlorStillAllowsEligibleFlorContest(t *testing.T) {
	for _, declaration := range []string{"flor", "presento_flor", "con_mi_flor"} {
		t.Run(declaration, func(t *testing.T) {
			s := fixture(t, true,
				[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}},
				[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}},
			)
			s = apply(t, s, 1, rules.Command{Type: "play", Card: 0, Declaration: declaration})
			assertRejectedCanto(t, s, 0, rules.Command{Type: "call", Kind: "envido", Declaration: "con_mi_flor"})
			command := rules.Command{Type: "call", Kind: "flor"}
			require.Contains(t, rules.LegalActions(s, 0), command)
			s = apply(t, s, 0, command)
			assert.Equal(t, "flor", s.Pending.Kind)
			assert.True(t, s.Players[0].FlorValid)
		})
	}
}

func TestDeclaredFlorDoesNotAllowFlorContestWithoutValidOrdinaryFlor(t *testing.T) {
	for _, holding := range []string{"no flor", "receta", "burned flor"} {
		t.Run(holding, func(t *testing.T) {
			hand := [3]rules.Card{{rules.Clubs, 1}, {rules.Coins, 6}, {rules.Cups, 4}}
			if holding == "receta" {
				hand = [3]rules.Card{{rules.Spades, 11}, {rules.Spades, 10}, {rules.Coins, 7}}
			} else if holding == "burned flor" {
				hand = [3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}}
			}
			s := fixture(t, true, hand,
				[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}},
			)
			s = apply(t, s, 1, rules.Command{Type: "call", Kind: "truco", Declaration: "flor"})
			s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "accept"})
			s = apply(t, s, 1, rules.Command{Type: "play", Card: 0, Declaration: "flor"})
			assertRejectedCanto(t, s, 0, rules.Command{Type: "call", Kind: "envido"})
			assertRejectedCanto(t, s, 0, rules.Command{Type: "call", Kind: "flor"})
		})
	}
}

func TestTeammatesFlorAllowsOnlyTheFlorContestForAnotherFlorHolder(t *testing.T) {
	s := fixture(t, false,
		[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}},
		[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}},
		[3]rules.Card{{rules.Coins, 7}, {rules.Coins, 6}, {rules.Coins, 5}},
		[3]rules.Card{{rules.Spades, 7}, {rules.Spades, 6}, {rules.Spades, 5}},
	)
	s = apply(t, s, 1, rules.Command{Type: "call", Kind: "truco", Declaration: "flor"})
	s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "accept"})
	s = apply(t, s, 1, rules.Command{Type: "play", Card: 0, Declaration: "a_ley"})
	s = apply(t, s, 2, rules.Command{Type: "play", Card: 0})
	require.True(t, s.Players[1].FlorDeclared)
	for _, seat := range []int{0, 2, 3} {
		require.False(t, s.Players[seat].FlorDeclared, "only the caller's teammate declared flor")
	}
	assertRejectedCanto(t, s, 3, rules.Command{Type: "call", Kind: "envido", Declaration: "con_mi_flor"})
	command := rules.Command{Type: "call", Kind: "flor"}
	require.Contains(t, rules.LegalActions(s, 3), command)
	s = apply(t, s, 3, command)
	assert.Equal(t, "flor", s.Pending.Kind)
	assert.True(t, s.Players[3].FlorValid)
}
