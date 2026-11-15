package learning_test

import (
	"github.com/neiam00/retruco-backend/cpu/learning"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestSignalStatisticsArePerObserverAndSurviveHands(t *testing.T) {
	var c learning.Communication
	c.ResetMatch()
	c.StartHand(1)
	require.True(t, c.Record(learning.Signal{Seat: 1, Kind: "perico"}, [4]bool{false, true, false, true}))
	assert.Zero(t, c.CountsFor(0))
	assert.Equal(t, uint32(1), c.CountsFor(3)[1][0])
	c.StartHand(2)
	assert.Empty(t, c.For(3))
	assert.Equal(t, uint32(1), c.CountsFor(3)[1][0])
	c.ResetMatch()
	assert.Zero(t, c.CountsFor(3))
}

func TestResponseRatesExcludeTimeoutsAndSeparateFamilies(t *testing.T) {
	v := rules.View{Config: rules.Config{Players: 4, Target: 12}, HandNumber: 2}
	context := learning.Context{Complete: true, Cantos: []learning.Canto{
		{Hand: 1, Event: rules.Event{Type: "respond", Seat: 1, Canto: "truco", Kind: "accept"}},
		{Hand: 2, Event: rules.Event{Type: "respond", Seat: 1, Canto: "truco", Kind: "decline"}},
		{Hand: 2, Event: rules.Event{Type: "respond", Seat: 1, Canto: "truco", Kind: "decline", Automatic: true}},
		{Hand: 2, Event: rules.Event{Type: "respond", Seat: 1, Canto: "envido", Kind: "raise"}},
	}}
	x := learning.Encode(v, 0, context)
	assert.InDelta(t, .5, x[512+40+6], 1e-6, "truco accept rate")
	assert.InDelta(t, .5, x[512+40+7], 1e-6, "truco decline rate, ignoring timeout")
	assert.Equal(t, float32(1), x[512+40+10+8], "envido raise rate")
	assert.Equal(t, float32(0), x[512+40+10+6], "envido accept rate")
}

func TestEncoderCannotSeeAnUnreceivedSignal(t *testing.T) {
	v := rules.View{Config: rules.Config{Players: 4, Target: 12}, HandNumber: 1}
	var c learning.Communication
	c.ResetMatch()
	c.StartHand(1)
	before := learning.Encode(v, 0, learning.Context{SignalCounts: c.CountsFor(0), Signals: c.For(0), SignalsComplete: c.Complete})
	require.True(t, c.Record(learning.Signal{Seat: 1, Kind: "receta"}, [4]bool{false, true, false, true}))
	after := learning.Encode(v, 0, learning.Context{SignalCounts: c.CountsFor(0), Signals: c.For(0), SignalsComplete: c.Complete})
	assert.Equal(t, before, after)
}
