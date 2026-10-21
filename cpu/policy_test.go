package cpu_test

import (
	"math/rand"
	"testing"

	"github.com/neiam00/retruco-backend/cpu"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPolicyCompletesMatchesUsingOnlyAuthorizedViews(t *testing.T) {
	for _, players := range []int{2, 4} {
		for _, required := range []bool{false, true} {
			for seed := int64(0); seed < 8; seed++ {
				// #nosec G404 -- Reproducible test deals, never production shuffling.
				random := rand.New(rand.NewSource(seed))
				deal := func() []rules.Card {
					deck := rules.Deck()
					random.Shuffle(len(deck), func(i, j int) { deck[i], deck[j] = deck[j], deck[i] })
					return deck
				}
				config := rules.Config{Players: players, Target: 12, FlorRequired: required}
				state, err := rules.New(config, int(seed)%players, deal())
				require.NoError(t, err)
				for step := 0; step < 2000 && state.Phase != rules.MatchOver; step++ {
					if state.Phase == rules.HandOver {
						state, err = rules.StartHand(state, deal())
						require.NoError(t, err)
						continue
					}
					seat := state.Turn
					if state.Pending != nil {
						seat = (state.Pending.Caller + 1) % players
					}
					view := rules.ViewFor(state, seat, false)
					decision := cpu.Decide(cpu.Request{Version: cpu.ProtocolVersion, Seat: seat, Entropy: random.Uint32(), View: view})
					require.Empty(t, decision.Error)
					require.NotNil(t, decision.Command)
					require.Contains(t, view.Legal, *decision.Command)
					state, _, err = rules.Apply(state, seat, *decision.Command)
					require.NoError(t, err, "players=%d flor=%t seed=%d step=%d", players, required, seed, step)
				}
				require.Equal(t, rules.MatchOver, state.Phase, "CPU policy must not stall a match")
			}
		}
	}
}

func TestPolicyUsesVisibleLeaderAndPreservesStrongCardsForTeammates(t *testing.T) {
	vira := rules.Card{Suit: rules.Spades, Number: 4}
	leader := rules.Card{Suit: rules.Coins, Number: 2}
	view := rules.View{
		Version: rules.Version, Config: rules.Config{Players: 4, Target: 12},
		Phase: rules.Playing, Vira: &vira, Holding: &rules.Holding{Envido: 27},
		Hand: []rules.HandCard{
			{Index: 0, Card: rules.Card{Suit: rules.Coins, Number: 4}},
			{Index: 1, Card: rules.Card{Suit: rules.Spades, Number: 11}},
			{Index: 2, Card: rules.Card{Suit: rules.Clubs, Number: 7}},
		},
		Legal: []rules.Command{{Type: "play", Card: 0}, {Type: "play", Card: 1}, {Type: "play", Card: 2}},
		Seats: []rules.SeatView{{Seat: 1, Team: 1, Plays: [3]rules.PublicPlay{{Present: true, Card: &leader}}}},
	}
	response := cpu.Decide(cpu.Request{Version: cpu.ProtocolVersion, Seat: 0, View: view})
	require.NotNil(t, response.Command)
	assert.Equal(t, 1, response.Command.Card, "perico beats the visible opposing two")
	view.Seats[0].Seat, view.Seats[0].Team = 2, 0
	response = cpu.Decide(cpu.Request{Version: cpu.ProtocolVersion, Seat: 0, View: view})
	require.NotNil(t, response.Command)
	assert.Equal(t, 0, response.Command.Card, "save the perico while the teammate leads")
}

func TestBrowserProtocolRejectsMismatchesAndSupportsFullUint32Entropy(t *testing.T) {
	state, err := rules.New(rules.Config{Players: 2, Target: 12}, 0, rules.Deck())
	require.NoError(t, err)
	view := rules.ViewFor(state, state.Turn, false)
	request := cpu.Request{Version: cpu.ProtocolVersion, Seat: state.Turn, View: view, Entropy: ^uint32(0)}
	response := cpu.Decide(request)
	require.Empty(t, response.Error)
	require.NotNil(t, response.Command)
	assert.Contains(t, view.Legal, *response.Command)
	request.View.Version = "1"
	assert.Empty(t, cpu.Decide(request).Error, "new browser engine supports hosts during rollout")
	request.Version++
	assert.NotEmpty(t, cpu.Decide(request).Error)
	request.Version = cpu.ProtocolVersion
	request.View.Version = "future-rules"
	assert.NotEmpty(t, cpu.Decide(request).Error)
}

func TestCPURespondsToPrivoUsingOnlyItsHandAndTheAnnouncement(t *testing.T) {
	for _, test := range []struct {
		name   string
		envido int
		flor   bool
		want   string
	}{
		{name: "higher envido", envido: 34, want: "accept"},
		{name: "lower envido and weak cards", envido: 25, want: "decline"},
		{name: "present flor", envido: 30, flor: true, want: "present_flor"},
	} {
		t.Run(test.name, func(t *testing.T) {
			vira := rules.Card{Suit: rules.Spades, Number: 4}
			view := rules.View{
				Version: rules.Version, Config: rules.Config{Players: 2, Target: 12},
				Phase: rules.Playing, Vira: &vira, Holding: &rules.Holding{Envido: test.envido},
				PrivoEnvido: []rules.PrivoEnvido{{Team: 0, Value: 33}},
				Pending:     &rules.Wager{Kind: "privo", Caller: 0, Value: 2, Terminal: true},
				Hand:        []rules.HandCard{{Index: 0, Card: rules.Card{Suit: rules.Coins, Number: 4}}},
				Legal:       []rules.Command{{Type: "respond", Kind: "accept"}, {Type: "respond", Kind: "decline"}},
			}
			if test.flor {
				view.Legal = append(view.Legal, rules.Command{Type: "respond", Kind: "present_flor"})
			}
			response := cpu.Decide(cpu.Request{Version: cpu.ProtocolVersion, Seat: 1, View: view})
			require.Empty(t, response.Error)
			require.NotNil(t, response.Command)
			assert.Equal(t, test.want, response.Command.Kind)
		})
	}
}
