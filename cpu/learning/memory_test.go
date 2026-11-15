package learning

import (
	"encoding/json"
	"testing"

	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMemoryJournalReplayAndIsolation(t *testing.T) {
	j := MemoryJournal{}
	j.StartHand(1)
	var f [StateSize]float32
	j.Append(0, f, false)
	j.Append(0, f, false)
	j.Append(1, f, true)
	p, err := j.Page(0, 1, 0)
	require.NoError(t, err)
	assert.Len(t, p.Events, 1)
	j.Append(0, f, true)
	for i := 0; i < 40; i++ {
		f[0] = float32(i + 1)
		j.Append(0, f, false)
	}
	p, err = j.Page(0, 1, 0)
	require.NoError(t, err)
	assert.Len(t, p.Events, MemoryPageSize)
	copy := j.Clone()
	j.Seats[0][0][0] = 99
	b, err := json.Marshal(copy)
	require.NoError(t, err)
	var restored MemoryJournal
	require.NoError(t, json.Unmarshal(b, &restored))
	p, err = restored.Page(0, 1, 32)
	require.NoError(t, err)
	assert.Equal(t, 42, p.Total)
	assert.Equal(t, 42, p.End)
	assert.Equal(t, float32(31), p.Events[0][0])
	assert.Zero(t, restored.Seats[0][0][0])
	_, err = restored.Page(0, 1, 43)
	assert.Error(t, err)
	restored.StartHand(2)
	p, err = restored.Page(0, 1, 42)
	require.NoError(t, err)
	assert.Empty(t, p.Events)
	assert.Zero(t, p.Start)
}

func TestTrainingChannelSeparatesSecretsAndAuthorizedHistory(t *testing.T) {
	e, err := NewEnvironment(rules.Config{Players: 4, Target: 12}, 42, [4]bool{true, true, true, true})
	require.NoError(t, err)
	assert.Nil(t, e.Observe().Training)
	e.EnableTraining()
	require.NoError(t, e.Step(0)) // independent uniform cut
	before, err := e.TrainingData(0, 0, 0)
	require.NoError(t, err)
	features := e.TrainingFeatures(0)
	e.State.Players[1].Cards[0], e.State.Players[1].Cards[1] = e.State.Players[1].Cards[1], e.State.Players[1].Cards[0]
	e.captureTraining(-1)
	after, err := e.TrainingData(0, 0, 0)
	require.NoError(t, err)
	assert.Equal(t, features, e.TrainingFeatures(0))
	assert.Equal(t, before.Memory.Total, after.Memory.Total)
	assert.NotEqual(t, before.Privileged, after.Privileged)
	private := DeliveredSignal{Signal: Signal{Seat: 1, Kind: "perico"}, Visible: [4]bool{false, true, false, true}}
	e.Communication.Record(private.Signal, private.Visible)
	e.recordTrainingSignal(private)
	e.captureTraining(-1)
	after, err = e.TrainingData(0, 0, 0)
	require.NoError(t, err)
	assert.Equal(t, before.Memory.Total, after.Memory.Total)
	assert.Equal(t, features, e.TrainingFeatures(0))
	visible := private
	visible.Visible[0] = true
	e.Communication.Record(visible.Signal, visible.Visible)
	e.recordTrainingSignal(visible)
	e.captureTraining(-1)
	after, err = e.TrainingData(0, 0, 0)
	require.NoError(t, err)
	assert.Greater(t, after.Memory.Total, before.Memory.Total)
	assert.NotEqual(t, features, e.TrainingFeatures(0))
}

func TestMemoryOverflowFailsClosedWithoutDroppingHistory(t *testing.T) {
	j := MemoryJournal{}
	j.StartHand(1)
	var f [StateSize]float32
	for i := 0; i <= MaxMemoryEvents; i++ {
		j.Append(0, f, true)
	}
	p, err := j.Page(0, 1, 0)
	require.NoError(t, err)
	assert.False(t, p.Complete)
	assert.Equal(t, MaxMemoryEvents, p.Total)
}

func TestPartnerSignalAblationRemovesClaimsAndCountsOnlyForObserver(t *testing.T) {
	e, err := NewEnvironment(rules.Config{Players: 4, Target: 12}, 91, [4]bool{true, true, true, true})
	require.NoError(t, err)
	observer := 0
	e.IgnorePartnerSignalsFor = &observer
	e.EnableTraining()
	require.NoError(t, e.Step(0))
	before := e.TrainingFeatures(0)
	senderBefore := e.TrainingFeatures(2)
	s := DeliveredSignal{Signal: Signal{Seat: 2, Kind: "perico"}, Visible: [4]bool{true, false, true, false}}
	e.Communication.Record(s.Signal, s.Visible)
	e.recordTrainingSignal(s)
	e.captureTraining(-1)
	assert.Equal(t, before, e.TrainingFeatures(0))
	assert.NotEqual(t, senderBefore, e.TrainingFeatures(2))
	s = DeliveredSignal{Signal: Signal{Seat: 1, Kind: "perica"}, Visible: [4]bool{true, true, false, true}}
	e.Communication.Record(s.Signal, s.Visible)
	e.recordTrainingSignal(s)
	e.captureTraining(-1)
	assert.NotEqual(t, before, e.TrainingFeatures(0), "opponent interceptions remain available")
}

func TestSignalDeliveryInterventionKeepsOtherObserversAndSenderUnchanged(t *testing.T) {
	for _, tc := range []struct {
		name     string
		observer int
	}{
		{name: "teammate", observer: 0},
		{name: "intercepting_opponent", observer: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			makeEnvironment := func() *Environment {
				e, err := NewEnvironment(rules.Config{Players: 4, Target: 12}, 91, [4]bool{true, true, true, true})
				require.NoError(t, err)
				e.EnableTraining()
				require.NoError(t, e.Step(0))
				// This fixture isolates delivery, not interception probability.
				e.Communication.Consumed = [4][3]bool{{true}, {true}, {true}, {true}}
				e.State.Turn = tc.observer
				e.captureTraining(-1)
				return e
			}
			control, blocked := makeEnvironment(), makeEnvironment()
			blocked.SignalBlocks[tc.observer][2] = true
			before := blocked.TrainingFeatures(tc.observer)
			for _, e := range []*Environment{control, blocked} {
				signal := DeliveredSignal{Signal: Signal{Seat: 2, Kind: "perico"}, Visible: [4]bool{true, true, true, true}}
				require.True(t, e.Communication.Record(signal.Signal, signal.Visible))
				e.recordTrainingSignal(signal)
				e.captureTraining(-1)
			}
			assert.Equal(t, before, blocked.TrainingFeatures(tc.observer))
			assert.NotEqual(t, control.TrainingFeatures(tc.observer), blocked.TrainingFeatures(tc.observer))
			assert.Empty(t, blocked.Observe().Signals)
			assert.Len(t, control.Observe().Signals, 1)
			assert.Equal(t, blocked.TrainingFeatures(tc.observer), blocked.Observe().Features)
			for other := 0; other < 4; other++ {
				if other != tc.observer {
					assert.Equal(t, control.TrainingFeatures(other), blocked.TrainingFeatures(other))
				}
			}
		})
	}
}
