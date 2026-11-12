package rules_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDeclaredFlorRequiresExplicitLeyEnvidoForTheWholeTable(t *testing.T) {
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
					require.True(t, s.Players[3].FlorValid)
				}
				for _, command := range []rules.Command{
					{Type: "call", Kind: "envido"},
					{Type: "call", Kind: "envido", Declaration: "con_mi_flor"},
					{Type: "call", Kind: "envido", Mode: "sin_ley"},
					{Type: "call", Kind: "ley_envido", Mode: "con_ley"},
					{Type: "call", Kind: "ley_envido", Mode: "con_ley", Declaration: "con_mi_flor"},
					{Type: "call", Kind: "ley_envido", Mode: "sin_ley"},
				} {
					assertRejectedCanto(t, s, s.Turn, command)
				}
				opening := rules.Command{Type: "call", Kind: "ley_envido"}
				legal := rules.LegalActions(s, s.Turn)
				require.Contains(t, legal, opening)
				for _, action := range legal {
					if action.Type == "call" {
						assert.NotEqual(t, "envido", action.Kind)
						if action.Kind == "ley_envido" {
							assert.Equal(t, opening, action)
						}
					}
				}
			})
		}
	}
}

func TestLeyEnvidoAfterFlorRequiresAnExplicitResponseAndResumesTruco(t *testing.T) {
	s := fixture(t, true,
		[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}},
		[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}},
	)
	s = apply(t, s, 1, rules.Command{Type: "call", Kind: "truco", Declaration: "con_mi_flor"})
	assertRejectedCanto(t, s, 0, rules.Command{Type: "call", Kind: "envido"})
	s = apply(t, s, 0, rules.Command{Type: "call", Kind: "ley_envido"})
	require.NotNil(t, s.Suspended)
	assert.Equal(t, "truco", s.Suspended.Kind)
	for _, reply := range []rules.Command{{Type: "respond", Kind: "accept"}, {Type: "respond", Kind: "raise", Amount: 2}, {Type: "respond", Kind: "falta"}} {
		assertRejectedCanto(t, s, 1, reply)
	}
	assertRejectedCanto(t, s, 1, rules.Command{Type: "respond", Kind: "accept", Mode: "sin_ley"})
	reply := rules.Command{Type: "respond", Kind: "accept", Mode: "con_ley"}
	require.Contains(t, rules.LegalActions(s, 1), reply)
	finished := apply(t, s, 1, reply)
	assert.Equal(t, "con_ley", finished.Envido.Mode)
	assert.True(t, finished.EnvidoAccepted)
	require.NotNil(t, finished.Pending)
	assert.Equal(t, "truco", finished.Pending.Kind)
	alias := apply(t, s, 1, rules.Command{Type: "call", Kind: "flor"})
	assert.Equal(t, finished, alias)
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
			assertRejectedCanto(t, s, 0, rules.Command{Type: "call", Kind: "envido"})
			for _, command := range []rules.Command{
				{Type: "call", Kind: "ley_envido"},
				{Type: "call", Kind: "flor"},
			} {
				require.Contains(t, rules.LegalActions(s, 0), command)
				next := apply(t, s, 0, command)
				assert.Equal(t, command.Kind, next.Pending.Kind)
			}
		})
	}
}

func TestDeclaredFlorStillAllowsEligibleFlorContest(t *testing.T) {
	s := fixture(t, true, [3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}}, [3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}})
	s = apply(t, s, 1, rules.Command{Type: "play", Card: 0, Declaration: "flor"})
	command := rules.Command{Type: "call", Kind: "flor"}
	require.Contains(t, rules.LegalActions(s, 0), command)
	s = apply(t, s, 0, command)
	assert.Equal(t, "flor", s.Pending.Kind)
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
			s = apply(t, s, 1, rules.Command{Type: "call", Kind: "truco", Declaration: "con_mi_flor"})
			s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "accept"})
			s = apply(t, s, 1, rules.Command{Type: "play", Card: 0, Declaration: "flor"})
			assertRejectedCanto(t, s, 0, rules.Command{Type: "call", Kind: "envido"})
			assertRejectedCanto(t, s, 0, rules.Command{Type: "call", Kind: "flor"})
		})
	}
}

func TestTeammatesFlorAllowsAnotherHolderToChallenge(t *testing.T) {
	s := fixture(t, false,
		[3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Clubs, 5}},
		[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}},
		[3]rules.Card{{rules.Coins, 7}, {rules.Coins, 6}, {rules.Coins, 5}},
		[3]rules.Card{{rules.Spades, 7}, {rules.Spades, 6}, {rules.Spades, 5}},
	)
	s = apply(t, s, 1, rules.Command{Type: "call", Kind: "truco", Declaration: "con_mi_flor"})
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

func TestFlorLeyPayoutsAtTheTable(t *testing.T) {
	// The caller's flor is 36, but its best pair is only 32. The reply has a
	// 33-point pair, so scoring both as ordinary envido would reverse the result.
	caller := [3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 5}, {rules.Clubs, 4}}
	pair := [3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Coins, 5}}
	strongFlor := [3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}}
	for _, tc := range []struct {
		name        string
		hand        [3]rules.Card
		reply, mode string
		points      [2]int
		envido      int
	}{
		{"sin ley quiero uses 33 against 36", pair, "accept", "sin_ley", [2]int{0, 6}, 2},
		{"sin ley no quiero voids envido", pair, "decline", "sin_ley", [2]int{0, 4}, 0},
		{"con ley quiero burns the lower flor", strongFlor, "accept", "con_ley", [2]int{5, 1}, 2},
		{"con ley no quiero achica the responder", strongFlor, "decline", "con_ley", [2]int{0, 5}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := fixture(t, true, tc.hand, caller)
			s = apply(t, s, 1, rules.Command{Type: "call", Kind: "ley_envido"})
			other := "sin_ley"
			if tc.mode == "sin_ley" {
				other = "con_ley"
			}
			for _, wrong := range []string{"", other} {
				assertRejectedCanto(t, s, 0, rules.Command{Type: "respond", Kind: tc.reply, Mode: wrong})
			}
			s = apply(t, s, 0, rules.Command{Type: "respond", Kind: tc.reply, Mode: tc.mode})
			require.Equal(t, tc.envido, s.Envido.Value)
			s = apply(t, s, 1, rules.Command{Type: "call", Kind: "truco", Declaration: "con_mi_flor"})
			s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "decline"})
			assert.Equal(t, tc.points, s.Scores, "includes one point for the caller's truco")
			if tc.reply == "accept" {
				value := 33
				if tc.mode == "con_ley" {
					value = 38
				}
				assert.Equal(t, []rules.EnvidoResult{{Seat: 0, Value: value, Eligible: true}, {Seat: 1, Value: 36, Eligible: true}}, s.EnvidoResults)
			}
		})
	}
}

func TestBareLeyFlorChallengeDoesNotRevealWhetherOpponentHasFlor(t *testing.T) {
	holder := [3]rules.Card{{rules.Coins, 7}, {rules.Coins, 6}, {rules.Coins, 5}}
	noFlor := [3]rules.Card{{rules.Clubs, 7}, {rules.Clubs, 6}, {rules.Cups, 5}}
	s := fixture(t, true, holder, noFlor)
	s = apply(t, s, 1, rules.Command{Type: "play", Card: 0, Declaration: "a_ley"})
	challenge := rules.Command{Type: "call", Kind: "flor"}
	require.Contains(t, rules.LegalActions(s, 0), challenge)
	pending := apply(t, s, 0, challenge)
	assert.Equal(t, []rules.Command{{Type: "respond", Kind: "decline", Mode: "sin_ley"}}, rules.LegalActions(pending, 1))
	assertRejectedCanto(t, pending, 1, rules.Command{Type: "respond", Kind: "accept", Mode: "sin_ley"})
	done := apply(t, pending, 1, rules.Command{Type: "respond", Kind: "decline", Mode: "sin_ley"})
	assert.Zero(t, done.Envido.Value)
	assert.True(t, done.Players[0].FlorValid)
}

func TestLeyReplyUsesTheSpeakersOwnFlor(t *testing.T) {
	s := fixture(t, false, [3]rules.Card{{rules.Cups, 1}, {rules.Cups, 3}, {rules.Cups, 4}}, [3]rules.Card{{rules.Spades, 7}, {rules.Clubs, 1}, {rules.Clubs, 6}}, [3]rules.Card{{rules.Coins, 1}, {rules.Clubs, 7}, {rules.Coins, 6}}, [3]rules.Card{{rules.Clubs, 3}, {rules.Coins, 3}, {rules.Spades, 5}})
	s = apply(t, s, 1, rules.Command{Type: "call", Kind: "ley_envido"})
	for _, tc := range []struct {
		seat int
		mode string
	}{{0, "con_ley"}, {2, "sin_ley"}} {
		require.Contains(t, rules.LegalActions(s, tc.seat), rules.Command{Type: "respond", Kind: "accept", Mode: tc.mode})
	}
	s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "accept", Mode: "con_ley"})
	s = apply(t, s, 1, rules.Command{Type: "call", Kind: "truco"})
	s = apply(t, s, 0, rules.Command{Type: "respond", Kind: "decline"})
	assert.False(t, s.EnvidoResults[1].Eligible, "a bluffer's ordinary pair cannot win a con-ley comparison")
	assert.Contains(t, s.Payouts, rules.Payout{Kind: "envido", Team: 0, Points: 2})
}
