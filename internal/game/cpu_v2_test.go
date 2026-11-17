package game_test

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"testing"

	"github.com/neiam00/retruco-backend/cpu/learning"
	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func cpuV2Fixture(t *testing.T) (*game.Host, *memoryRepository, store.User, string) {
	t.Helper()
	repo := &memoryRepository{payloads: map[string][]byte{}}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	host := game.New(repo, log)
	user := store.User{ID: "owner", Name: "Ana"}
	config := rules.Config{Players: 4, Target: 12}
	code, err := host.Create(t.Context(), user, config, false, game.RoomOptions{CPUSeats: []int{1, 2, 3}})
	require.NoError(t, err)
	var match game.Match
	require.NoError(t, json.Unmarshal(repo.payloads[code], &match))
	deck := rules.Deck()
	// At cut 1 these positions deal both spade pieces to CPU seat 1 while
	// retaining spade 11 as vira. Every card remains unique.
	deck[0], deck[29] = deck[29], deck[0]
	deck[36], deck[27] = deck[27], deck[36]
	state, err := rules.New(config, 0, deck)
	require.NoError(t, err)
	state, _, err = rules.Apply(state, state.Turn, rules.Command{Type: "cut", Cut: 1})
	require.NoError(t, err)
	match.State = &state
	repo.payloads[code], err = json.Marshal(match)
	require.NoError(t, err)
	host = game.New(repo, log)
	require.NoError(t, host.Restore(t.Context()))
	t.Cleanup(host.ConnectPlayer(code, user.ID, "controller", 2))
	return host, repo, user, code
}

func signalInput(work game.CPUWork, id, signal string) game.Input {
	return game.Input{ID: id, CPUSeat: &work.Seat, CPUKind: work.Kind, CPUToken: work.Token, Revision: work.View.Revision, Signal: signal}
}

func TestCPUv2ConsumesSignalOpportunitiesWithoutChangingDeadline(t *testing.T) {
	host, _, user, code := cpuV2Fixture(t)
	before, err := host.Snapshot(code, user.ID, 0)
	require.NoError(t, err)
	for seat := 1; seat < 4; seat++ {
		work := host.CPUWork(code, user.ID, "controller")
		require.Equal(t, 2, work.Version)
		require.Equal(t, "signal", work.Kind)
		require.Equal(t, seat, work.Seat)
		require.NotNil(t, work.Features)
		require.Len(t, work.Token, 64)
		input := signalInput(work, fmt.Sprint(seat), "none")
		require.NoError(t, host.HandleFromConnection(t.Context(), code, user, input, "controller"))
		// Identical request ID is idempotent; a new request cannot reuse the token.
		require.NoError(t, host.HandleFromConnection(t.Context(), code, user, input, "controller"))
		input.ID += "-replay"
		require.Error(t, host.HandleFromConnection(t.Context(), code, user, input, "controller"))
	}
	after, err := host.Snapshot(code, user.ID, 0)
	require.NoError(t, err)
	assert.Empty(t, after.Signals)
	assert.Equal(t, before.State.Revision, after.State.Revision)
	assert.Equal(t, before.Deadline, after.Deadline)
	work := host.CPUWork(code, user.ID, "controller")
	require.Equal(t, "game", work.Kind)
	require.Equal(t, "value-and-team-v17", work.Features.ChoicePolicy)
	require.Len(t, work.Features.Actions, len(work.View.Legal))
	for i, c := range work.View.Legal {
		assert.Equal(t, learning.EncodeAction(c), work.Features.Actions[i], "browser indices must address the same approved CPU candidate")
	}
	input := game.Input{ID: "play", CPUSeat: &work.Seat, CPUKind: "game", CPUToken: work.Token, Revision: work.View.Revision, Command: work.View.Legal[0]}
	require.NoError(t, host.HandleFromConnection(t.Context(), code, user, input, "controller"))
}

func TestCPUv2SignalsAndPrivateVisibilitySurviveRestore(t *testing.T) {
	host, repo, user, code := cpuV2Fixture(t)
	work := host.CPUWork(code, user.ID, "controller")
	require.NoError(t, host.HandleFromConnection(t.Context(), code, user, signalInput(work, "signal", "perico"), "controller"))
	before := make([]game.Snapshot, 4)
	for seat := 0; seat < 4; seat++ {
		var err error
		before[seat], err = host.Snapshot(code, "spectator", seat)
		require.NoError(t, err)
		encoded, err := json.Marshal(before[seat])
		require.NoError(t, err)
		assert.NotContains(t, string(encoded), "Visibility")
		assert.NotContains(t, string(encoded), "Visible")
		assert.NotContains(t, string(encoded), "StoredSignals")
	}
	assert.Len(t, before[3].Signals, 1, "the signaling CPU's teammate always sees its claim")
	var stored game.Match
	require.NoError(t, json.Unmarshal(repo.payloads[code], &stored))
	for seat := 0; seat < 4; seat++ {
		assert.Equal(t, uint32(len(before[seat].Signals)), stored.CPUCommunication.CountsFor(seat)[1][0], "only delivered signals contribute to persistent statistics")
	}

	restored := game.New(repo, slog.New(slog.NewTextHandler(io.Discard, nil)))
	require.NoError(t, restored.Restore(t.Context()))
	t.Cleanup(restored.ConnectPlayer(code, user.ID, "new-controller", 2))
	for seat := 0; seat < 4; seat++ {
		after, err := restored.Snapshot(code, "spectator", seat)
		require.NoError(t, err)
		assert.Equal(t, before[seat].Signals, after.Signals, "visibility must not be redrawn")
	}
	newWork := restored.CPUWork(code, user.ID, "new-controller")
	assert.Equal(t, 2, newWork.Seat, "consumed seat 1 does not get another opportunity")
	assert.NotEqual(t, work.Token, newWork.Token)
}

func TestCPUv2HumanSignalInvalidatesWorkAndAuthorityIsRequired(t *testing.T) {
	host, _, user, code := cpuV2Fixture(t)
	work := host.CPUWork(code, user.ID, "controller")
	input := signalInput(work, "stale", "receta")
	require.Error(t, host.HandleFromConnection(t.Context(), code, user, input, "other-controller"))
	spoof := input
	seat := 0
	spoof.CPUSeat = &seat
	require.Error(t, host.HandleFromConnection(t.Context(), code, user, spoof, "controller"))
	require.NoError(t, host.Handle(t.Context(), code, user, game.Input{ID: "human-signal", Signal: "perica"}))
	updated := host.CPUWork(code, user.ID, "controller")
	assert.Equal(t, work.View.Revision, updated.View.Revision)
	assert.NotEqual(t, work.Token, updated.Token)
	require.Error(t, host.HandleFromConnection(t.Context(), code, user, input, "controller"))
	require.NoError(t, host.HandleFromConnection(t.Context(), code, user, signalInput(updated, "fresh", "receta"), "controller"))
}

func TestCPUv2RejectsFalseClaimButHumanCanBluff(t *testing.T) {
	host, _, user, code := cpuV2Fixture(t)
	work := host.CPUWork(code, user.ID, "controller")
	require.NotNil(t, work.SignalMask)
	require.False(t, work.SignalMask[3])
	require.ErrorContains(t, host.HandleFromConnection(t.Context(), code, user, signalInput(work, "false-cpu", "seven_spades"), "controller"), "verdaderas")
	unchanged := host.CPUWork(code, user.ID, "controller")
	assert.Equal(t, work.Token, unchanged.Token)
	require.NoError(t, host.Handle(t.Context(), code, user, game.Input{ID: "human-bluff", Signal: "seven_spades"}))
}

func TestCPUv1DoesNotReceiveOrSendSignalWork(t *testing.T) {
	host, repo, user, code := cpuV2Fixture(t)
	_ = host
	legacy := game.New(repo, slog.New(slog.NewTextHandler(io.Discard, nil)))
	require.NoError(t, legacy.Restore(t.Context()))
	t.Cleanup(legacy.ConnectPlayer(code, user.ID, "legacy"))
	work := legacy.CPUWork(code, user.ID, "legacy")
	assert.Equal(t, 1, work.Version)
	assert.Empty(t, work.Kind)
	assert.Empty(t, work.Token)
	assert.Nil(t, work.Features)
	seat := 1
	require.Error(t, legacy.HandleFromConnection(t.Context(), code, user, game.Input{ID: "legacy-signal", CPUSeat: &seat, Signal: "receta"}, "legacy"))
}
