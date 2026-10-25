package game_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The gate stalls one actual repository operation, without blocking storage for
// other rooms. Later calls pass through so missing serialization is observable.
type gatedRepository struct {
	*memoryRepository
	operation string
	code      string
	entered   chan struct{}
	release   chan struct{}
	blocked   atomic.Bool
	err       error
}

func newGatedRepository(t *testing.T) *gatedRepository {
	t.Helper()
	r := &gatedRepository{
		memoryRepository: &memoryRepository{payloads: make(map[string][]byte)},
		entered:          make(chan struct{}),
		release:          make(chan struct{}),
	}
	t.Cleanup(func() {
		select {
		case <-r.release:
		default:
			close(r.release)
		}
	})
	return r
}

func (r *gatedRepository) wait(ctx context.Context, operation, code string) error {
	if r.operation != operation || (r.code != "" && r.code != code) || !r.blocked.CompareAndSwap(false, true) {
		return nil
	}
	close(r.entered)
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-r.release:
		return r.err
	}
}

func (r *gatedRepository) CreateMatch(ctx context.Context, code string, payload []byte, members []string) error {
	if err := r.wait(ctx, "create", code); err != nil {
		return err
	}
	return r.memoryRepository.CreateMatch(ctx, code, payload, members)
}

func (r *gatedRepository) SaveMatch(ctx context.Context, code string, payload []byte, finished bool, members []string) error {
	if err := r.wait(ctx, "save", code); err != nil {
		return err
	}
	return r.memoryRepository.SaveMatch(ctx, code, payload, finished, members)
}

func (r *gatedRepository) DeleteLobby(ctx context.Context, code string) error {
	if err := r.wait(ctx, "delete", code); err != nil {
		return err
	}
	return r.memoryRepository.DeleteLobby(ctx, code)
}

func startCall[T any](call func() T) <-chan T {
	result := make(chan T, 1)
	go func() { result <- call() }()
	return result
}

func finishCall[T any](t *testing.T, result <-chan T) T {
	t.Helper()
	select {
	case value := <-result:
		return value
	case <-time.After(2 * time.Second):
		t.Fatal("operation blocked behind another room's storage")
		var zero T
		return zero
	}
}

func TestBlockedRoomWriteDoesNotBlockOtherRooms(t *testing.T) {
	for _, operation := range []string{"save", "delete", "create"} {
		t.Run(operation, func(t *testing.T) {
			r := newGatedRepository(t)
			host := game.New(r, slog.New(slog.NewTextHandler(io.Discard, nil)))
			owner := store.User{ID: "owner", Name: "Owner"}
			guest := store.User{ID: "guest", Name: "Guest"}
			config := rules.Config{Players: 4, Target: 12}
			slow, err := host.Create(t.Context(), owner, config, false, game.RoomOptions{})
			require.NoError(t, err)
			fast, err := host.Create(t.Context(), owner, config, false, game.RoomOptions{CPUSeats: []int{3}})
			require.NoError(t, err)
			updates, unsubscribe := host.Subscribe(slow)
			t.Cleanup(unsubscribe)
			r.operation, r.code = operation, slow
			if operation == "create" {
				r.code = ""
			}
			seat := 1
			pending := startCall(func() error {
				switch operation {
				case "save":
					return host.Join(t.Context(), slow, guest, &seat)
				case "delete":
					return host.Handle(t.Context(), slow, owner, game.Input{ID: "leave", Command: rules.Command{Type: "leave"}})
				default:
					_, err := host.Create(t.Context(), owner, config, false, game.RoomOptions{})
					return err
				}
			})
			finishCall(t, r.entered)
			listings := finishCall(t, startCall(host.List))
			assert.Len(t, listings, 2, "only committed room membership is visible")
			for _, listing := range listings {
				if listing.Code == slow {
					assert.Equal(t, 1, listing.Players)
				}
			}
			assert.Empty(t, updates, "a blocked write must not publish an update")

			// A competing join must observe the first join/closure after it
			// commits, rather than claiming the same seat or reviving the room.
			var competing <-chan error
			if operation != "create" {
				competing = startCall(func() error {
					return host.Join(t.Context(), slow, store.User{ID: "late"}, &seat)
				})
			}
			require.NoError(t, finishCall(t, startCall(func() error {
				return host.Join(t.Context(), fast, guest, &seat)
			})))
			disconnect := finishCall(t, startCall(func() func() {
				return host.ConnectPlayer(fast, owner.ID, "owner-tab")
			}))
			defer disconnect()
			view, err := host.Snapshot(fast, owner.ID, 0)
			require.NoError(t, err)
			assert.Equal(t, guest.ID, view.Seats[seat].ID)
			assert.True(t, host.CPUWork(fast, owner.ID, "owner-tab").Controlled)
			fastUpdates, unwatch := host.Subscribe(fast)
			require.NoError(t, host.Handle(t.Context(), fast, owner, game.Input{ID: "chat", Chat: "hello"}))
			assert.Len(t, fastUpdates, 1)
			unwatch()

			close(r.release)
			require.NoError(t, finishCall(t, pending))
			if competing != nil {
				require.Error(t, finishCall(t, competing))
				assert.Len(t, updates, 1)
			}
			if operation == "delete" {
				_, err := host.Snapshot(slow, owner.ID, 0)
				assert.Error(t, err)
			}
		})
	}
}

func TestConcurrentRoomActionsPreserveEveryUniqueCommand(t *testing.T) {
	r := &memoryRepository{payloads: make(map[string][]byte)}
	host := game.New(r, slog.New(slog.NewTextHandler(io.Discard, nil)))
	owner := store.User{ID: "owner", Name: "Owner"}
	codes := make([]string, 4)
	for i := range codes {
		code, err := host.Create(t.Context(), owner, rules.Config{Players: 2, Target: 12}, false, game.RoomOptions{})
		require.NoError(t, err)
		codes[i] = code
	}
	var workers sync.WaitGroup
	results := make(chan error, len(codes)*40)
	for _, code := range codes {
		for i := range 40 {
			workers.Go(func() {
				// Each command arrives twice, as on a reconnect/retry.
				results <- host.Handle(t.Context(), code, owner, game.Input{ID: fmt.Sprint(i / 2), Chat: fmt.Sprint(i / 2)})
			})
		}
	}
	workers.Wait()
	close(results)
	for err := range results {
		require.NoError(t, err)
	}
	restored := game.New(r, slog.New(slog.NewTextHandler(io.Discard, nil)))
	require.NoError(t, restored.Restore(t.Context()))
	for _, code := range codes {
		view, err := restored.Snapshot(code, owner.ID, 0)
		require.NoError(t, err)
		assert.Equal(t, uint64(21), view.Sequence)
		require.Len(t, view.Chat, 20)
		messages := make(map[string]bool)
		for _, message := range view.Chat {
			messages[message.Text] = true
		}
		assert.Len(t, messages, 20, "concurrent writes must not lose distinct commands")
	}
}

func TestRoomTimersProgressIndependentlyAndStopOnCancellation(t *testing.T) {
	r := newGatedRepository(t)
	config := rules.Config{Players: 2, Target: 12}
	for i, code := range []string{"SLOW-0000", "FAST-0000"} {
		state, err := rules.New(config, 0, rules.Deck())
		require.NoError(t, err)
		match := game.Match{
			Code: code, Config: config, State: &state, Sequence: 1,
			Seats: []game.Seat{{ID: "owner"}, {ID: "guest"}},
			// The second deadline occurs after the first write is blocked.
			Deadline: time.Now().Add(time.Duration(i) * time.Second),
		}
		payload, err := json.Marshal(match)
		require.NoError(t, err)
		r.payloads[code] = payload
	}
	host := game.New(r, slog.New(slog.NewTextHandler(io.Discard, nil)))
	require.NoError(t, host.Restore(t.Context()))
	updates, unsubscribe := host.Subscribe("FAST-0000")
	defer unsubscribe()
	r.operation, r.code = "save", "SLOW-0000"
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	stopped := startCall(func() struct{} { host.Run(ctx); return struct{}{} })
	finishCall(t, r.entered)
	finishCall(t, updates)
	fast, err := host.Snapshot("FAST-0000", "owner", 0)
	require.NoError(t, err)
	assert.Equal(t, uint64(2), fast.Sequence)
	assert.NotEqual(t, rules.Cutting, fast.State.Phase)

	// Cancellation must reach the blocked write and Run must join its workers.
	cancel()
	finishCall(t, stopped)
	slow, err := host.Snapshot("SLOW-0000", "owner", 0)
	require.NoError(t, err)
	assert.Equal(t, uint64(1), slow.Sequence, "canceled persistence must not publish a timeout")
}

func TestConcurrentCreationReservesCapacityAndReleasesFailedReservations(t *testing.T) {
	r := newGatedRepository(t)
	owner := store.User{ID: "owner", Name: "Owner"}
	config := rules.Config{Players: 2, Target: 12}
	for i := range 999 {
		match := game.Match{Code: fmt.Sprintf("ROOM-%04d", i), Config: config, Seats: []game.Seat{{ID: owner.ID}, {}}}
		payload, err := json.Marshal(match)
		require.NoError(t, err)
		r.payloads[match.Code] = payload
	}
	host := game.New(r, slog.New(slog.NewTextHandler(io.Discard, nil)))
	require.NoError(t, host.Restore(t.Context()))
	r.operation, r.err = "create", errors.New("storage unavailable")
	create := func() error {
		_, err := host.Create(t.Context(), owner, config, false, game.RoomOptions{})
		return err
	}
	pending := startCall(create)
	finishCall(t, r.entered)
	err := finishCall(t, startCall(create))
	require.ErrorContains(t, err, "no hay espacio")
	assert.Len(t, finishCall(t, startCall(host.List)), 999)
	close(r.release)
	require.Error(t, finishCall(t, pending))

	code, err := host.Create(t.Context(), owner, config, false, game.RoomOptions{})
	require.NoError(t, err, "failed creation must release its capacity reservation")
	require.ErrorContains(t, create(), "no hay espacio")
	require.NoError(t, host.Join(t.Context(), code, store.User{ID: "guest"}, nil))
	view, err := host.Snapshot(code, owner.ID, 0)
	require.NoError(t, err)
	require.NoError(t, host.Handle(t.Context(), code, owner, game.Input{
		ID: "forfeit", Revision: view.State.Revision, Command: rules.Command{Type: "leave"},
	}))
	require.NoError(t, create(), "completed matches must release active capacity")
	require.ErrorContains(t, create(), "no hay espacio")
	require.NoError(t, host.Handle(t.Context(), "ROOM-0000", owner, game.Input{ID: "close", Command: rules.Command{Type: "leave"}}))
	require.NoError(t, create(), "deleted lobbies must release active capacity")
}
