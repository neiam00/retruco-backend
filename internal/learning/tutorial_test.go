package learning_test

import (
	"testing"

	"github.com/neiam00/retruco-backend/internal/learning"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPreparedLessonUsesLegalMovesAndSettlesTheExplainedHand(t *testing.T) {
	lesson, err := learning.NewLesson()
	require.NoError(t, err)
	require.Equal(t, 1, lesson.Version)
	require.Len(t, lesson.Steps, 10)
	for _, step := range lesson.Steps {
		t.Run(step.ID, func(t *testing.T) {
			assert.Equal(t, rules.Card{Suit: rules.Coins, Number: 1}, *step.View.Vira)
			require.Len(t, step.View.Hand, 3)
			assert.Equal(t, rules.Card{Suit: rules.Coins, Number: 11}, step.View.Hand[0].Card)
			assert.Equal(t, 37, step.View.Holding.Envido)
			assert.Zero(t, step.View.Holding.Flor, "the teaching hand must not unexpectedly require flor declarations")
			if step.Command != nil {
				assert.Contains(t, step.View.Legal, *step.Command, "the learner follows an engine-legal action")
			}
			if step.ID != "recap" {
				assert.Equal(t, [2]int{}, step.View.Scores, "accepted envido is not paid early")
				assert.Empty(t, step.View.EnvidoResults, "the opponent's envido is revealed only at settlement")
			}
			for _, play := range step.View.Seats[1].Plays {
				if play.Card != nil {
					assert.Contains(t, []rules.Card{{Suit: rules.Cups, Number: 4}, {Suit: rules.Coins, Number: 10}}, *play.Card)
				}
			}
		})
	}
	final := lesson.Steps[len(lesson.Steps)-1]
	assert.Equal(t, "recap", final.ID)
	assert.Equal(t, rules.HandOver, final.View.Phase)
	assert.Equal(t, [2]int{5, 0}, final.View.Scores)
	assert.Equal(t, []rules.Payout{{Kind: "envido", Team: 0, Points: 2}, {Kind: "truco", Team: 0, Points: 3}}, final.View.Payouts)
	require.NotNil(t, final.View.LastHand)
	assert.Equal(t, "two_tricks", final.View.LastHand.Reason)
	assert.Equal(t, []rules.EnvidoResult{{Seat: 0, Value: 37, Eligible: true}, {Seat: 1, Value: 34, Eligible: true}}, final.View.EnvidoResults)
	assert.Nil(t, final.View.Seats[1].Plays[2].Card, "the opponent's unplayed third card remains private")
	assert.Len(t, final.Events, 9, "cut, two calls, two accepts, and four played cards are captured")
}
