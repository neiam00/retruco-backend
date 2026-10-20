package game_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type memoryRepository struct {
	payloads map[string][]byte
	fail     bool
}

func (repository *memoryRepository) CreateMatch(ctx context.Context, code string, payload []byte, members []string) error {
	if _, exists := repository.payloads[code]; exists {
		return store.ErrMatchCodeInUse
	}
	return repository.SaveMatch(ctx, code, payload, false, members)
}

func (repository *memoryRepository) SaveMatch(_ context.Context, code string, payload []byte, _ bool, _ []string) error {
	if repository.fail {
		return errors.New("storage unavailable")
	}
	repository.payloads[code] = append([]byte(nil), payload...)
	return nil
}

func (repository *memoryRepository) ActiveMatches(context.Context) ([][]byte, error) {
	var payloads [][]byte
	for _, payload := range repository.payloads {
		var match game.Match
		if err := json.Unmarshal(payload, &match); err != nil {
			return nil, err
		}
		if match.Expired || (match.State != nil && match.State.Phase == rules.MatchOver) {
			continue
		}
		payloads = append(payloads, append([]byte(nil), payload...))
	}
	return payloads, nil
}

func TestInactiveLobbyExpiresAndStaysExpiredAfterRestart(t *testing.T) {
	repository := &memoryRepository{payloads: make(map[string][]byte)}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	old := game.Match{
		Code:         "OLDM-1234",
		Config:       rules.Config{Players: 2, Target: 12},
		Seats:        []game.Seat{{ID: "ana", Name: "Ana"}},
		LastActivity: time.Now().Add(-25 * time.Hour),
	}
	payload, err := json.Marshal(old)
	require.NoError(t, err)
	repository.payloads[old.Code] = payload
	host := game.New(repository, log)
	require.NoError(t, host.Restore(t.Context()))
	ctx, cancel := context.WithCancel(t.Context())
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		host.Run(ctx)
	}()
	t.Cleanup(func() { cancel(); <-stopped })
	require.Eventually(t, func() bool { return len(host.List()) == 0 }, time.Second, 10*time.Millisecond)
	cancel()
	<-stopped
	_, err = host.Snapshot(old.Code, "ana", 0)
	require.Error(t, err)
	restored := game.New(repository, log)
	require.NoError(t, restored.Restore(t.Context()))
	assert.Empty(t, restored.List())
}

func newHost(t *testing.T, private bool) (*game.Host, *memoryRepository, string, []store.User) {
	t.Helper()
	repository := &memoryRepository{payloads: make(map[string][]byte)}
	host := game.New(repository, slog.New(slog.NewTextHandler(io.Discard, nil)))
	players := []store.User{
		{ID: "ana", Name: "Ana"},
		{ID: "beto", Name: "Beto"},
		{ID: "carla", Name: "Carla"},
		{ID: "diego", Name: "Diego"},
	}
	code, err := host.Create(t.Context(), players[0], rules.Config{Players: 4, Target: 12}, private)
	require.NoError(t, err)
	for _, player := range players[1:] {
		require.NoError(t, host.Join(t.Context(), code, player))
	}
	return host, repository, code, players
}

func cut(t *testing.T, host *game.Host, code string, players []store.User) game.Snapshot {
	t.Helper()
	view, err := host.Snapshot(code, players[0].ID, 0)
	require.NoError(t, err)
	input := game.Input{ID: "cut", Revision: view.State.Revision, Command: rules.Command{Type: "cut", Cut: 19}}
	require.NoError(t, host.Handle(t.Context(), code, players[view.State.Turn], input))
	view, err = host.Snapshot(code, players[0].ID, 0)
	require.NoError(t, err)
	return view
}

func TestSpectatorAndSignalViewsRespectTheSelectedPlayer(t *testing.T) {
	host, _, code, players := newHost(t, true)
	assert.Empty(t, host.List(), "private matches are absent from the public browser")
	cut(t, host, code, players)
	require.NoError(t, host.Handle(t.Context(), code, players[0], game.Input{ID: "signal", Signal: "receta"}))
	for seat, player := range players {
		playerView, err := host.Snapshot(code, player.ID, (seat+1)%4)
		require.NoError(t, err)
		spectatorView, err := host.Snapshot(code, "spectator", seat)
		require.NoError(t, err)
		assert.Equal(t, seat, playerView.Perspective, "seated users cannot select an opponent's hand")
		assert.Equal(t, playerView.State.Hand, spectatorView.State.Hand)
		assert.Equal(t, playerView.Signals, spectatorView.Signals, "no new visibility rolls for spectators")
		assert.Empty(t, spectatorView.State.Legal)
		if seat%2 == 0 {
			require.Len(t, playerView.Signals, 1, "both teammates always see a signal, including bluffs")
		}
		payload, err := json.Marshal(spectatorView)
		require.NoError(t, err)
		assert.NotContains(t, string(payload), "Visible", "recipient decisions are never sent to clients")
	}
	err := host.Handle(t.Context(), code, store.User{ID: "spectator"}, game.Input{ID: "illegal", Command: rules.Command{Type: "play"}})
	require.Error(t, err)
}

func TestDuplicateCommandsAndReconnectDoNotResetTheClock(t *testing.T) {
	host, repository, code, players := newHost(t, false)
	view := cut(t, host, code, players)
	turn := view.State.Turn
	input := game.Input{ID: "first-card", Revision: view.State.Revision, Command: rules.Command{Type: "play", Card: 0, Declaration: "a_ley"}}
	require.NoError(t, host.Handle(t.Context(), code, players[turn], input))
	after, err := host.Snapshot(code, players[turn].ID, turn)
	require.NoError(t, err)
	require.NoError(t, host.Handle(t.Context(), code, players[turn], input))
	reconnected, err := host.Snapshot(code, players[turn].ID, turn)
	require.NoError(t, err)
	assert.Equal(t, after.State.Revision, reconnected.State.Revision)
	assert.Equal(t, after.Deadline, reconnected.Deadline)
	require.NotEmpty(t, after.Events)
	assert.Equal(t, after.Events, reconnected.Events, "retrying a move must not duplicate its public announcement")
	assert.Equal(t, "a_ley", after.Events[len(after.Events)-1].Event.Text)
	spectator, err := host.Snapshot(code, "spectator", (turn+1)%4)
	require.NoError(t, err)
	assert.Equal(t, after.Events, spectator.Events, "declarations are public across perspectives")

	restored := game.New(repository, slog.New(slog.NewTextHandler(io.Discard, nil)))
	require.NoError(t, restored.Restore(t.Context()))
	restarted, err := restored.Snapshot(code, players[turn].ID, turn)
	require.NoError(t, err)
	assert.Equal(t, after.State, restarted.State)
	assert.True(t, after.Deadline.Equal(restarted.Deadline))
	assert.Equal(t, after.Events, restarted.Events, "reconnect and restore retain the same public announcements")

	input.ID = "stale-command"
	require.Error(t, host.Handle(t.Context(), code, players[turn], input))
}

func TestPersistenceFailureDoesNotPublishAnUncommittedMove(t *testing.T) {
	host, repository, code, players := newHost(t, false)
	before := cut(t, host, code, players)
	repository.fail = true
	err := host.Handle(t.Context(), code, players[before.State.Turn], game.Input{
		ID:       "failed-card",
		Revision: before.State.Revision,
		Command:  rules.Command{Type: "play", Card: 0},
	})
	require.Error(t, err)
	after, snapshotErr := host.Snapshot(code, players[0].ID, 0)
	require.NoError(t, snapshotErr)
	assert.Equal(t, before.State, after.State)
	assert.Equal(t, before.Deadline, after.Deadline)
	assert.Equal(t, before.Sequence, after.Sequence)
}

func TestCantoInterruptionAndResumptionEachReceiveThirtySeconds(t *testing.T) {
	host, repository, code, players := newHost(t, false)
	view := cut(t, host, code, players)
	caller := view.State.Turn
	responder := (caller + 1) % len(players)
	log := slog.New(slog.NewTextHandler(io.Discard, nil))

	// Resume from a valid saved match with only five seconds left. This checks
	// that transitions create a fresh window rather than inheriting that clock.
	shortenSavedWindow := func() {
		var saved game.Match
		require.NoError(t, json.Unmarshal(repository.payloads[code], &saved))
		saved.Deadline = time.Now().Add(5 * time.Second)
		payload, err := json.Marshal(saved)
		require.NoError(t, err)
		repository.payloads[code] = payload
		host = game.New(repository, log)
		require.NoError(t, host.Restore(t.Context()))
	}
	act := func(id string, seat int, command rules.Command) game.Snapshot {
		before, err := host.Snapshot(code, players[seat].ID, seat)
		require.NoError(t, err)
		require.NoError(t, host.Handle(t.Context(), code, players[seat], game.Input{
			ID: id, Revision: before.State.Revision, Command: command,
		}))
		after, err := host.Snapshot(code, players[seat].ID, seat)
		require.NoError(t, err)
		assert.InDelta(t, 30, after.Deadline.Sub(after.ServerTime).Seconds(), 1)
		return after
	}
	shortenSavedWindow()
	act("truco", caller, rules.Command{Type: "call", Kind: "truco", Declaration: "con_mi_flor"})
	shortenSavedWindow()
	interrupted := act("interrupt", responder, rules.Command{Type: "call", Kind: "ley_envido"})
	require.NotNil(t, interrupted.State.Suspended)
	shortenSavedWindow()
	resumed := act("accept-envido", caller, rules.Command{Type: "respond", Kind: "accept", Mode: "sin_ley", Declaration: "con_mi_flor"})
	require.NotNil(t, resumed.State.Pending)
	assert.Equal(t, "truco", resumed.State.Pending.Kind)
	assert.Nil(t, resumed.State.Suspended)
	shortenSavedWindow()
	playing := act("accept-truco", responder, rules.Command{Type: "respond", Kind: "accept", Declaration: "con_mi_flor"})
	assert.Nil(t, playing.State.Pending)
	assert.Equal(t, caller, playing.State.Turn)
}
