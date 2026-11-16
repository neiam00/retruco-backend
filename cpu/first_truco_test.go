package cpu_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/cpu"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/require"
)

func TestFirstTrickTrucoCanWaitAfterAnOpponentPlays(t *testing.T) {
	for _, tc := range []struct {
		name                   string
		players, first, second int
		allow                  bool
	}{
		{"heads-up marginal cover", 2, 1, 2, false},
		{"heads-up strong cover", 2, 1, 3, true},
		{"partners marginal cover", 4, 1, 2, false},
		{"partners strong cover", 4, 3, 7, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			v := tableWithVira(t, tc.players, 1, [3]rules.Card{{Suit: rules.Cups, Number: tc.first}, {Suit: rules.Spades, Number: tc.second}, {Suit: rules.Clubs, Number: 5}}, rules.Card{Suit: rules.Coins, Number: 4})
			v.Mano = 0
			v.Seats[0].Plays[0] = played(rules.Coins, 5)
			v.Seats[0].Remaining = 2
			bid := rules.Command{Type: "call", Kind: "truco", Amount: 3}
			v.Legal = append(v.Legal, bid)
			previous, bounded := cpu.PolicyChoices(v, 1, cpu.AdaptiveSelectionProfile)
			require.False(t, bounded)
			require.Contains(t, previous, bid)
			current, bounded := cpu.PolicyChoices(v, 1, cpu.FirstTrickValueProfile)
			require.False(t, bounded, "a deferred value wager must not become an opening bluff")
			if tc.allow {
				require.Contains(t, current, bid)
			} else {
				require.NotContains(t, current, bid)
			}
			deferred, marked := cpu.PolicyChoices(v, 1, cpu.PlayingProfile)
			require.NotContains(t, deferred, bid, "even a strong ordinary value call waits for the first trick to finish")
			require.False(t, marked)
			require.Contains(t, v.Legal, bid, "human legal wagers remain available")
		})
	}
}
