package rules_test

import (
	"encoding/json"
	"math"
	"slices"
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAdditiveRaisesRejectOverflowWithoutChangingTheHand(t *testing.T) {
	ley := apply(t, simple(t), 1, rules.Command{Type: "call", Kind: "ley_envido"})
	for _, scenario := range []struct {
		name  string
		state rules.State
		seat  int
		mode  string
	}{
		{name: "envido", state: apply(t, simple(t), 1, rules.Command{Type: "call", Kind: "envido"}), seat: 0},
		{name: "flor", state: opposingFlores(t), seat: 1},
		{name: "ley awaiting choice", state: ley, seat: 0, mode: "con_ley"},
		{name: "chosen con ley", state: apply(t, ley, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 2, Mode: "con_ley"}), seat: 1},
		{name: "chosen sin ley", state: apply(t, ley, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 2, Mode: "sin_ley"}), seat: 1},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			remaining := scenario.state.Config.Target - scenario.state.Pending.Value
			for _, amount := range []struct {
				name  string
				value int
			}{
				{name: "maximum integer", value: math.MaxInt},
				{name: "over target", value: remaining + 1},
				{name: "zero", value: 0},
				{name: "negative", value: -1},
			} {
				t.Run(amount.name, func(t *testing.T) {
					command := rules.Command{Type: "respond", Kind: "raise", Amount: amount.value, Mode: scenario.mode}
					next, events, err := rules.Apply(scenario.state, scenario.seat, command)
					require.Error(t, err)
					assert.Equal(t, scenario.state, next)
					assert.Empty(t, events)
				})
			}
			for _, amount := range []int{1, remaining} {
				next, _, err := rules.Apply(scenario.state, scenario.seat,
					rules.Command{Type: "respond", Kind: "raise", Amount: amount, Mode: scenario.mode})
				require.NoError(t, err)
				assert.Equal(t, scenario.state.Pending.Value+amount, next.Pending.Value)
			}
		})
	}
}

func TestTrucoVoiceAlternatesAcrossTurnsAndRounds(t *testing.T) {
	state := apply(t, simple(t), 1, rules.Command{Type: "call", Kind: "truco"})
	state = apply(t, state, 0, rules.Command{Type: "respond", Kind: "accept"})
	assertTrucoCalls(t, state, 1, nil)

	state = apply(t, state, 1, rules.Command{Type: "play", Card: 0})
	assertTrucoCalls(t, state, 0, []int{6, 9, 100})
	state = apply(t, state, 0, rules.Command{Type: "call", Kind: "truco", Amount: 6})
	state = apply(t, state, 1, rules.Command{Type: "respond", Kind: "accept"})
	assertTrucoCalls(t, state, 0, nil)

	state = apply(t, state, 0, rules.Command{Type: "play", Card: 0})
	require.Equal(t, 1, state.Round, "the first-round winner opens the second round")
	assertTrucoCalls(t, state, 1, []int{9, 100})
	state = apply(t, state, 1, rules.Command{Type: "call", Kind: "truco", Amount: 9})
	state = apply(t, state, 0, rules.Command{Type: "respond", Kind: "accept"})

	// Reconnecting or restarting the host must preserve which team may raise.
	snapshot, err := json.Marshal(state)
	require.NoError(t, err)
	var restored rules.State
	require.NoError(t, json.Unmarshal(snapshot, &restored))
	assertTrucoCalls(t, restored, 1, nil)
	state = apply(t, restored, 1, rules.Command{Type: "play", Card: 1})
	assertTrucoCalls(t, state, 0, []int{100})
}

func TestTrucoVoiceBelongsToTheTeam(t *testing.T) {
	state := fixture(t, false,
		[3]rules.Card{{rules.Clubs, 1}, {rules.Coins, 6}, {rules.Cups, 5}},
		[3]rules.Card{{rules.Coins, 4}, {rules.Clubs, 5}, {rules.Cups, 6}},
		[3]rules.Card{{rules.Cups, 4}, {rules.Coins, 5}, {rules.Clubs, 6}},
		[3]rules.Card{{rules.Spades, 1}, {rules.Coins, 7}, {rules.Spades, 6}})
	state = apply(t, state, 1, rules.Command{Type: "call", Kind: "truco"})
	for _, seat := range []int{1, 3} {
		assertRejectedCanto(t, state, seat, rules.Command{Type: "respond", Kind: "raise", Amount: 6})
	}
	// Either rival may counter, even though neither has the card turn.
	for _, seat := range []int{0, 2} {
		_, _, err := rules.Apply(state, seat, rules.Command{Type: "respond", Kind: "raise", Amount: 6})
		require.NoError(t, err)
	}
	state = apply(t, state, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 6})
	for _, seat := range []int{0, 2} {
		assertRejectedCanto(t, state, seat, rules.Command{Type: "respond", Kind: "raise", Amount: 9})
	}
	state = apply(t, state, 3, rules.Command{Type: "respond", Kind: "accept"})

	// The caller's teammate is also blocked on their own turn. Both members
	// of the accepting team can raise, including the original truco caller.
	for _, turn := range []struct {
		seat   int
		raises []int
	}{
		{seat: 1, raises: []int{9, 100}},
		{seat: 2},
		{seat: 3, raises: []int{9, 100}},
		{seat: 0},
	} {
		assertTrucoCalls(t, state, turn.seat, turn.raises)
		state = apply(t, state, turn.seat, rules.Command{Type: "play", Card: 0})
	}
}

func TestEnvidoDoesNotTransferTrucoVoice(t *testing.T) {
	state := apply(t, simple(t), 1, rules.Command{Type: "call", Kind: "truco"})
	state = apply(t, state, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 6})
	state = apply(t, state, 1, rules.Command{Type: "call", Kind: "envido"})
	state = apply(t, state, 0, rules.Command{Type: "respond", Kind: "accept"})
	assertRejectedCanto(t, state, 0, rules.Command{Type: "respond", Kind: "raise", Amount: 9})
	state = apply(t, state, 1, rules.Command{Type: "respond", Kind: "accept"})
	assertTrucoCalls(t, state, 1, []int{9, 100})
	state = apply(t, state, 1, rules.Command{Type: "play", Card: 0})
	assertTrucoCalls(t, state, 0, nil)

	// Having made the latest truco call does not prohibit calling envido.
	state = apply(t, simple(t), 1, rules.Command{Type: "call", Kind: "truco"})
	state = apply(t, state, 0, rules.Command{Type: "respond", Kind: "accept"})
	state = apply(t, state, 1, rules.Command{Type: "call", Kind: "envido"})
	state = apply(t, state, 0, rules.Command{Type: "respond", Kind: "accept"})
	assertTrucoCalls(t, state, 1, nil)
}

func assertTrucoCalls(t *testing.T, state rules.State, seat int, amounts []int) {
	t.Helper()
	require.Equal(t, seat, state.Turn, "check voice on the player's actual card turn")
	var offered []int
	for _, command := range rules.LegalActions(state, seat) {
		if command.Type == "call" && command.Kind == "truco" && command.Declaration == "" {
			offered = append(offered, command.Amount)
		}
	}
	assert.ElementsMatch(t, amounts, offered, "only authorized raises should be offered")
	for _, amount := range []int{6, 9, 100} {
		command := rules.Command{Type: "call", Kind: "truco", Amount: amount}
		if slices.Contains(amounts, amount) {
			_, _, err := rules.Apply(state, seat, command)
			require.NoError(t, err)
		} else {
			assertRejectedCanto(t, state, seat, command)
		}
	}
}

func assertRejectedCanto(t *testing.T, state rules.State, seat int, command rules.Command) {
	t.Helper()
	next, _, err := rules.Apply(state, seat, command)
	require.Error(t, err, "seat %d must not be allowed to send %+v", seat, command)
	assert.Equal(t, state, next, "rejected cantos must not change the hand")
	assert.NotContains(t, rules.LegalActions(state, seat), command)
}
