package cpu

import (
	"math/rand/v2"
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/require"
)

// Compare the rank forecast with the game's settlement on complete, ordinary
// 2v2 deals. The oracle plays with seat views. Its exposed ranks exclude
// concealed bottoms, even though both implementations resolve the same deal.
func TestForecastSettlementMatchesGameWithPegadosAndBurns(t *testing.T) {
	rng := rand.New(rand.NewPCG(2731, 91607))
	outcomes := map[string]bool{}
	checked := 0
	for deal := range 256 {
		deck := rules.Deck()
		rng.Shuffle(len(deck), func(i, j int) { deck[i], deck[j] = deck[j], deck[i] })
		state, err := rules.New(rules.Config{Players: 4, Target: 24}, rng.IntN(4), deck)
		require.NoError(t, err)
		state, _, err = rules.Apply(state, state.Turn, rules.Command{Type: "cut", Cut: 20})
		require.NoError(t, err)
		firstTeam := -1
		for state.Phase == rules.Playing {
			seat := state.Turn
			view := rules.ViewFor(state, seat, false)
			if view.Round == 1 && firstTeam < 0 && !view.MandatoryStack {
				w := trickWinner(forecastTable(view).table[0], 4, view.Mano, true)
				require.NotEqual(t, -1, w)
				firstTeam = w % 2
			}
			var moves []rules.Command
			for _, cmd := range view.Legal {
				if cmd.Declaration == "" && (cmd.Type == "play" || cmd.Type == "burn" || cmd.Type == "stack") {
					moves = append(moves, cmd)
				}
			}
			cmd := moves[rng.IntN(len(moves))]
			f := forecastTable(view)
			for s := range 4 {
				p := state.Players[s]
				if p.Plays[1].Concealed {
					f.table[1][s] = rules.Strength(p.Plays[1].Card, state.Vira)
				}
				for i, c := range p.Cards {
					if !p.Used[i] {
						f.left[s][i] = rules.Strength(c, state.Vira)
					}
				}
			}
			m := trickMove{kind: cmd.Type, bottom: f.left[seat][cmd.Card]}
			if cmd.Type == "stack" {
				m.top = f.left[seat][cmd.Top]
			}
			want := f.winnerAfter(m)
			follow, _, err := rules.Apply(state, seat, cmd)
			require.NoError(t, err)
			state = follow
			team := firstTeam
			for follow.Phase == rules.Playing {
				s := follow.Turn
				v := rules.ViewFor(follow, s, false)
				var hand [3]int
				for _, h := range v.Hand {
					if !h.Used {
						hand[h.Index] = rules.Strength(h.Card, follow.Vira)
					}
				}
				if v.Round == 1 && team < 0 && !v.MandatoryStack {
					team = trickWinner(forecastTable(v).table[0], 4, v.Mano, true) % 2
				}
				rank := chooseRank(hand, forecastTable(v).table[v.Round], 4, s, team)
				var next rules.Command
				for _, legal := range v.Legal {
					if legal.Declaration != "" {
						continue
					}
					if v.MandatoryStack && legal.Type == "stack" && hand[legal.Top] >= hand[legal.Card] || !v.MandatoryStack && legal.Type == "play" && hand[legal.Card] == rank {
						next = legal
						break
					}
				}
				require.NotEmpty(t, next.Type)
				follow, _, err = rules.Apply(follow, s, next)
				require.NoError(t, err)
			}
			require.Equal(t, follow.HandWinner, want, "deal %d; seat %d; round %d; command %+v", deal, seat, view.Round, cmd)
			outcomes[follow.LastHand.Reason] = true
			checked++
		}
	}
	require.Greater(t, checked, 1500)
	for _, reason := range []string{"mandatory_top", "visible_first_and_third", "two_tricks", "third_trick", "first_and_tie"} {
		require.True(t, outcomes[reason], reason)
	}
}
