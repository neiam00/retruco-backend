package rules_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFlorAnnouncementsRequireAValidOrdinaryFlor(t *testing.T) {
	for _, holding := range []string{"no flor", "burned flor", "receta"} {
		for _, required := range []bool{false, true} {
			t.Run(holding+map[bool]string{false: "/free", true: "/required"}[required], func(t *testing.T) {
				hand := [3]rules.Card{{rules.Clubs, 3}, {rules.Cups, 6}, {rules.Coins, 5}}
				if holding == "burned flor" {
					hand = [3]rules.Card{{rules.Clubs, 3}, {rules.Clubs, 6}, {rules.Clubs, 5}}
				} else if holding == "receta" {
					hand = [3]rules.Card{{rules.Spades, 11}, {rules.Spades, 10}, {rules.Coins, 5}}
				}
				s := fixture(t, required, [3]rules.Card{{rules.Clubs, 1}, {rules.Coins, 6}, {rules.Cups, 5}}, hand)
				if holding == "burned flor" {
					s.Players[1].FlorValid = false // Already forfeited in this hand.
				}
				truco := apply(t, s, 1, rules.Command{Type: "call", Kind: "truco"})
				truco = apply(t, truco, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 6})
				for _, scenario := range []struct {
					name    string
					state   rules.State
					command rules.Command
				}{
					{"card", s, rules.Command{Type: "play", Card: 0}},
					{"truco", s, rules.Command{Type: "call", Kind: "truco"}},
					{"accept", truco, rules.Command{Type: "respond", Kind: "accept"}},
					{"refuse", truco, rules.Command{Type: "respond", Kind: "decline"}},
					{"vale partida", truco, rules.Command{Type: "respond", Kind: "raise", Amount: 100}},
				} {
					t.Run(scenario.name, func(t *testing.T) {
						for _, declaration := range []string{"flor", "presento_flor", "flor_flor", "con_mi_flor", "con_flor"} {
							command := scenario.command
							command.Declaration = declaration
							assertRejectedCanto(t, scenario.state, 1, command)
							assert.NotContains(t, rules.LegalActions(scenario.state, 1), command)
						}
						_, _, err := rules.Apply(scenario.state, 1, scenario.command)
						require.NoError(t, err, "removing the false declaration keeps the underlying action legal")
					})
				}
			})
		}
	}
}

func TestDeclarationPayloadMustMatchTheAction(t *testing.T) {
	s := simple(t)
	for _, c := range []rules.Command{
		{Type: "play", Card: 0, Declaration: "made_up"},
		{Type: "call", Kind: "truco", Declaration: "a_ley"},
		{Type: "leave", Declaration: "flor"},
	} {
		assertRejectedCanto(t, s, 1, c)
	}
	_, _, err := rules.Apply(s, 1, rules.Command{Type: "play", Card: 0, Declaration: "a_ley"})
	require.NoError(t, err, "A ley may truthfully mean the player may or may not have flor")
}

func TestAutomaticCantoRefusalDoesNotInventAFlorDeclaration(t *testing.T) {
	for _, hand := range [][3]rules.Card{
		{{rules.Clubs, 1}, {rules.Coins, 6}, {rules.Cups, 5}},
		{{rules.Coins, 1}, {rules.Coins, 6}, {rules.Coins, 5}},
		{{rules.Spades, 11}, {rules.Spades, 10}, {rules.Coins, 5}},
	} {
		for _, kind := range []string{"truco", "envido"} {
			s := fixture(t, true, hand, [3]rules.Card{{rules.Clubs, 3}, {rules.Cups, 6}, {rules.Spades, 5}})
			s = apply(t, s, 1, rules.Command{Type: "call", Kind: kind})
			next, events, err := rules.Timeout(s, 0)
			require.NoError(t, err)
			require.Len(t, events, 1)
			assert.True(t, events[0].Automatic)
			assert.Empty(t, events[0].Text)
			assert.False(t, next.Players[0].FlorDeclared, "the clock cannot announce a player's holding")
			assert.Equal(t, s.Players[0].FlorValid, next.Players[0].FlorValid, "ordinary response timeout does not forfeit flor")
		}
	}
}

func TestALeyIsOnlyAvailableInTheFirstRound(t *testing.T) {
	state := fixture(t, false,
		[3]rules.Card{{rules.Spades, 1}, {rules.Clubs, 4}, {rules.Cups, 6}},
		[3]rules.Card{{rules.Coins, 4}, {rules.Cups, 3}, {rules.Coins, 5}},
	)
	state = apply(t, state, 1, rules.Command{Type: "play", Card: 0, Declaration: "a_ley"})
	assert.True(t, state.Ley)
	state = apply(t, state, 0, rules.Command{Type: "play", Card: 0})
	require.Equal(t, 1, state.Round)

	for state.Round <= 2 && state.Phase == rules.Playing {
		seat := state.Turn
		commands := []rules.Command{
			{Type: "play", Card: state.Round, Declaration: "a_ley"},
			{Type: "call", Kind: "truco", Declaration: "a_ley"},
		}
		if state.Round == 1 {
			commands = append(commands, rules.Command{Type: "stack", Card: 1, Top: 2, Declaration: "a_ley"})
		}
		for _, command := range commands {
			next, events, err := rules.Apply(state, seat, command)
			require.Error(t, err, "round %d: %s must reject A ley", state.Round+1, command.Type)
			assert.Equal(t, state, next)
			assert.Empty(t, events)
		}
		for _, action := range rules.LegalActions(state, seat) {
			assert.NotEqual(t, "a_ley", action.Declaration)
		}
		pending := apply(t, state, seat, rules.Command{Type: "call", Kind: "truco"})
		_, _, err := rules.Apply(pending, 1-seat, rules.Command{Type: "respond", Kind: "accept", Declaration: "a_ley"})
		require.Error(t, err)

		round := state.Round
		state = apply(t, state, seat, rules.Command{Type: "play", Card: round})
		state = apply(t, state, state.Turn, rules.Command{Type: "play", Card: round})
	}
	assert.Equal(t, rules.HandOver, state.Phase)
}

func TestContextualFlorWordingPreservesFlorOnEveryCard(t *testing.T) {
	for _, declaration := range []string{"flor", ""} {
		t.Run("declaration="+declaration, func(t *testing.T) {
			s := fixture(t, true,
				[3]rules.Card{{rules.Clubs, 1}, {rules.Clubs, 5}, {rules.Cups, 4}},
				[3]rules.Card{{rules.Coins, 3}, {rules.Coins, 7}, {rules.Coins, 6}},
			)
			for round := range 3 {
				for range 2 {
					seat := s.Turn
					wording := ""
					if seat == 1 {
						wording = declaration
						if wording != "" && round == 2 {
							wording = "presento_flor"
						}
					}
					s = apply(t, s, seat, rules.Command{Type: "play", Card: round, Declaration: wording})
				}
			}
			assert.Equal(t, rules.HandOver, s.Phase)
			assert.Equal(t, declaration != "", s.Players[1].FlorValid)
			want := 4
			if declaration == "" {
				want = 1
			}
			assert.Equal(t, want, s.Scores[1], "the final Flor must still pay three flor points")
		})
	}
}

func TestFlorPrefixPreservesOrdinaryFlorOnTruco(t *testing.T) {
	s := fixture(t, true, [3]rules.Card{{rules.Clubs, 3}, {rules.Clubs, 6}, {rules.Clubs, 5}}, [3]rules.Card{{rules.Coins, 3}, {rules.Coins, 7}, {rules.Coins, 6}})
	s = apply(t, s, 1, rules.Command{Type: "call", Kind: "truco", Amount: 3, Declaration: "con_mi_flor"})
	for _, reply := range []rules.Command{{Type: "respond", Kind: "accept"}, {Type: "respond", Kind: "raise", Amount: 6}, {Type: "respond", Kind: "decline"}} {
		reply.Declaration = "con_mi_flor"
		require.Contains(t, rules.LegalActions(s, 0), reply)
		next := apply(t, s, 0, reply)
		assert.True(t, next.Players[0].FlorValid)
		assert.True(t, next.Players[0].FlorDeclared)
	}
}

func TestALeyEnvidoCannotConfirmFlor(t *testing.T) {
	s := simple(t)
	for _, declaration := range []string{"flor", "presento_flor", "flor_flor", "con_mi_flor", "con_flor"} {
		t.Run(declaration, func(t *testing.T) {
			assertRejectedCanto(t, s, s.Turn, rules.Command{Type: "call", Kind: "ley_envido", Declaration: declaration})
		})
	}
	_, _, err := rules.Apply(s, s.Turn, rules.Command{Type: "call", Kind: "ley_envido"})
	require.NoError(t, err, "A ley remains legal without announcing flor")
}

func TestHolderCannotOpenSinLey(t *testing.T) {
	for _, required := range []bool{false, true} {
		t.Run(map[bool]string{false: "free flor", true: "required flor"}[required], func(t *testing.T) {
			s := fixture(t, required,
				[3]rules.Card{{rules.Clubs, 3}, {rules.Clubs, 6}, {rules.Clubs, 5}},
				[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}},
				[3]rules.Card{{rules.Coins, 7}, {rules.Coins, 6}, {rules.Coins, 5}},
				[3]rules.Card{{rules.Spades, 7}, {rules.Spades, 6}, {rules.Spades, 5}},
			)
			s = apply(t, s, 1, rules.Command{Type: "play", Card: 0, Declaration: "a_ley"})
			assertRejectedCanto(t, s, 2, rules.Command{Type: "call", Kind: "ley_envido", Mode: "sin_ley"})
			for _, seat := range []int{0, 1, 2, 3} {
				assert.True(t, s.Players[seat].FlorValid)
			}
		})
	}
}

func TestSinLeyDoesNotInvalidateReceta(t *testing.T) {
	for _, action := range []string{"call", "respond"} {
		t.Run(action, func(t *testing.T) {
			hands := [][3]rules.Card{
				{{rules.Clubs, 1}, {rules.Clubs, 5}, {rules.Cups, 4}},
				{{rules.Spades, 11}, {rules.Spades, 10}, {rules.Coins, 7}},
			}
			seat := 1
			if action == "call" {
				seat = 0
				hands[0], hands[1] = hands[1], hands[0]
			}
			s := fixture(t, true, hands...)
			s = apply(t, s, 1, rules.Command{Type: "play", Card: 0, Declaration: "a_ley"})
			command := rules.Command{Type: "call", Kind: "ley_envido", Mode: "sin_ley"}
			if action == "respond" {
				s = apply(t, s, 0, rules.Command{Type: "call", Kind: "ley_envido"})
				command.Type, command.Kind = "respond", "decline"
			}
			s = apply(t, s, seat, command)
			assert.True(t, s.Players[seat].FlorValid)
		})
	}
}
