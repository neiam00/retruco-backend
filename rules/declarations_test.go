package rules_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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

func TestFlorWordingPreservesFlorOnEveryCard(t *testing.T) {
	for _, declaration := range []string{"flor", "presento_flor", "flor_flor", "con_mi_flor", "con_flor", ""} {
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

func TestFlorCantosPreserveFlorExceptExplicitForfeitures(t *testing.T) {
	for _, tc := range []struct {
		name      string
		pending   string
		command   rules.Command
		valid     bool
		seat      int
		priorALey bool
	}{
		{name: "truco call", command: rules.Command{Type: "call", Kind: "truco", Amount: 3}, valid: true, seat: 1},
		{name: "ordinary envido call", command: rules.Command{Type: "call", Kind: "envido"}, valid: true, seat: 1},
		{name: "sin ley call", command: rules.Command{Type: "call", Kind: "ley_envido", Mode: "sin_ley"}, priorALey: true},
		{name: "accept truco", pending: "truco", command: rules.Command{Type: "respond", Kind: "accept"}, valid: true},
		{name: "raise truco", pending: "truco", command: rules.Command{Type: "respond", Kind: "raise", Amount: 6}, valid: true},
		{name: "decline truco", pending: "truco", command: rules.Command{Type: "respond", Kind: "decline"}, valid: true},
		{name: "decline envido", pending: "envido", command: rules.Command{Type: "respond", Kind: "decline"}, valid: true},
		{name: "decline ley envido", pending: "ley_envido", command: rules.Command{Type: "respond", Kind: "decline"}, valid: true},
		{name: "accept con ley", pending: "ley_envido", command: rules.Command{Type: "respond", Kind: "accept", Mode: "con_ley"}, valid: true},
		{name: "raise con ley", pending: "ley_envido", command: rules.Command{Type: "respond", Kind: "raise", Amount: 2, Mode: "con_ley"}, valid: true},
		{name: "falta con ley", pending: "ley_envido", command: rules.Command{Type: "respond", Kind: "falta", Mode: "con_ley"}, valid: true},
		{name: "accept sin ley", pending: "ley_envido", command: rules.Command{Type: "respond", Kind: "accept", Mode: "sin_ley"}},
		{name: "raise sin ley", pending: "ley_envido", command: rules.Command{Type: "respond", Kind: "raise", Amount: 2, Mode: "sin_ley"}},
		{name: "decline envido sin ley", pending: "envido", command: rules.Command{Type: "respond", Kind: "decline", Mode: "sin_ley"}, seat: 1, priorALey: true},
		{name: "decline ley sin ley", pending: "ley_envido", command: rules.Command{Type: "respond", Kind: "decline", Mode: "sin_ley"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, declaration := range []string{"flor", "presento_flor", "flor_flor", "con_mi_flor", "con_flor"} {
				t.Run(declaration, func(t *testing.T) {
					hands := [][3]rules.Card{
						{{rules.Clubs, 1}, {rules.Clubs, 5}, {rules.Cups, 4}},
						{{rules.Coins, 3}, {rules.Coins, 7}, {rules.Coins, 6}},
					}
					seat := tc.seat
					if seat == 0 {
						hands[0], hands[1] = hands[1], hands[0]
					}
					s := fixture(t, true, hands...)
					if tc.priorALey {
						s = apply(t, s, 1, rules.Command{Type: "play", Card: 0, Declaration: "a_ley"})
					}
					if tc.pending != "" {
						// Start the wager before either player confirms flor.
						s = apply(t, s, 1-seat, rules.Command{Type: "call", Kind: tc.pending})
					}
					command := tc.command
					command.Declaration = declaration
					if declaration != "con_flor" {
						require.Contains(t, rules.LegalActions(s, seat), command, "the declaration must also be available to the UI")
					}
					s = apply(t, s, seat, command)
					assert.Equal(t, tc.valid, s.Players[seat].FlorValid)
					assert.True(t, s.Players[seat].FlorDeclared)
				})
			}
		})
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

func TestSinLeyCallBurnsOnlyTheSpeakersOrdinaryFlor(t *testing.T) {
	for _, required := range []bool{false, true} {
		t.Run(map[bool]string{false: "free flor", true: "required flor"}[required], func(t *testing.T) {
			s := fixture(t, required,
				[3]rules.Card{{rules.Clubs, 3}, {rules.Clubs, 6}, {rules.Clubs, 5}},
				[3]rules.Card{{rules.Cups, 7}, {rules.Cups, 6}, {rules.Cups, 5}},
				[3]rules.Card{{rules.Coins, 7}, {rules.Coins, 6}, {rules.Coins, 5}},
				[3]rules.Card{{rules.Spades, 7}, {rules.Spades, 6}, {rules.Spades, 5}},
			)
			s = apply(t, s, 1, rules.Command{Type: "play", Card: 0, Declaration: "a_ley"})
			s = apply(t, s, 2, rules.Command{Type: "call", Kind: "ley_envido", Mode: "sin_ley", Declaration: "con_mi_flor"})
			assert.False(t, s.Players[2].FlorValid)
			for _, seat := range []int{0, 1, 3} {
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
			command := rules.Command{Type: "call", Kind: "ley_envido", Mode: "sin_ley", Declaration: "con_mi_flor"}
			if action == "respond" {
				s = apply(t, s, 0, rules.Command{Type: "call", Kind: "ley_envido"})
				command.Type, command.Kind = "respond", "decline"
			}
			s = apply(t, s, seat, command)
			assert.True(t, s.Players[seat].FlorValid)
		})
	}
}
