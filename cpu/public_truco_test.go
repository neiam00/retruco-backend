package cpu_test

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/neiam00/retruco-backend/cpu"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNativeOpponentAcceptsThePubliclyWonAllIn(t *testing.T) {
	data, err := os.ReadFile("testdata/public-winning-truco.json")
	require.NoError(t, err)
	var view rules.View
	require.NoError(t, json.Unmarshal(data, &view))
	// Our teammate won the first trick, then visibly
	// played the unique perico on the second. The human has no strong cards left.
	decision := cpu.Decide(cpu.Request{Version: 1, Seat: 0, View: view})
	require.NotNil(t, decision.Command)
	assert.Equal(t, "accept", decision.Command.Kind)

	t.Run("teammates tied highest in the first trick", func(t *testing.T) {
		var allied rules.View
		require.NoError(t, json.Unmarshal(data, &allied))
		allied.Seats[0].Plays[0].Card = &rules.Card{Suit: rules.Cups, Number: 1}
		result := cpu.Decide(cpu.Request{Version: 1, Seat: 0, View: allied})
		require.NotNil(t, result.Command)
		assert.Equal(t, "accept", result.Command.Kind)
	})

	for _, change := range []string{"concealed", "missing first card", "perica can be beaten", "opponent won first", "first tied"} {
		t.Run(change, func(t *testing.T) {
			var uncertain rules.View
			require.NoError(t, json.Unmarshal(data, &uncertain))
			switch change {
			case "concealed":
				uncertain.Seats[2].Plays[1].Concealed = true
				uncertain.Seats[2].Plays[1].Card = nil
			case "missing first card":
				uncertain.Seats[1].Plays[0] = rules.PublicPlay{}
			case "perica can be beaten":
				uncertain.Seats[2].Plays[1].Card = &rules.Card{Suit: rules.Clubs, Number: 10}
			case "opponent won first":
				uncertain.Seats[1].Plays[0].Card = &rules.Card{Suit: rules.Spades, Number: 1}
			case "first tied":
				uncertain.Seats[1].Plays[0].Card = &rules.Card{Suit: rules.Cups, Number: 1}
			}
			decision := cpu.Decide(cpu.Request{Version: 1, Seat: 0, View: uncertain})
			require.NotNil(t, decision.Command)
			assert.Equal(t, "decline", decision.Command.Kind, "current trick leadership alone must not force high-stake acceptance")
		})
	}
}
