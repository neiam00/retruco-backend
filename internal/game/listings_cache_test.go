package game_test

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestListingStreamPublishesCommittedChangesDuringDeploymentPause(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		repository := &memoryRepository{payloads: make(map[string][]byte)}
		host := game.New(repository, slog.New(slog.NewTextHandler(io.Discard, nil)))
		baseline, updates, unsubscribe, err := host.SubscribeListingStream()
		require.NoError(t, err)
		defer unsubscribe()
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		go host.Run(ctx)
		synctest.Wait()
		host.SetUpdating(true)
		owner := store.User{ID: "owner", Name: "Owner"}
		guest := store.User{ID: "guest", Name: "Guest"}
		code, err := host.Create(t.Context(), owner, rules.Config{Players: 4, Target: 12}, false, game.RoomOptions{})
		require.NoError(t, err)
		_, err = host.Create(t.Context(), owner, rules.Config{Players: 4, Target: 12}, true, game.RoomOptions{})
		require.NoError(t, err)
		repository.fail = true
		require.Error(t, host.Join(t.Context(), code, guest, nil))
		repository.fail = false
		select {
		case <-updates:
			t.Fatal("batch published early")
		default:
		}
		time.Sleep(250 * time.Millisecond)
		synctest.Wait()
		select {
		case <-updates:
		default:
			t.Fatal("deployment pause stopped lobby publication")
		}
		frames, err := host.ListingStreamFrames(baseline.Epoch, baseline.Revision)
		require.NoError(t, err)
		require.Len(t, frames, 1)
		var delta struct {
			Upserts []game.Listing `json:"upserts"`
		}
		require.NoError(t, json.Unmarshal([]byte(frames[0].JSON), &delta))
		require.Len(t, delta.Upserts, 1, "private rooms are not published")
		assert.Equal(t, 1, delta.Upserts[0].Players, "failed membership write cannot reach lobby")
		require.NoError(t, host.Join(t.Context(), code, guest, nil))
		time.Sleep(250 * time.Millisecond)
		synctest.Wait()
		frames, err = host.ListingStreamFrames(frames[0].Epoch, frames[0].Revision)
		require.NoError(t, err)
		require.Len(t, frames, 1)
		require.NoError(t, json.Unmarshal([]byte(frames[0].JSON), &delta))
		require.Len(t, delta.Upserts, 1)
		assert.Equal(t, 2, delta.Upserts[0].Players)
		cancel()
		synctest.Wait()
	})
}

func TestListingJSONPublishesOnlyCommittedPublicRooms(t *testing.T) {
	repository := &memoryRepository{payloads: make(map[string][]byte)}
	host := game.New(repository, slog.New(slog.NewTextHandler(io.Discard, nil)))
	owner := store.User{ID: "owner", Name: "Owner"}
	guest := store.User{ID: "guest", Name: "Guest"}
	empty, err := host.ListingJSON()
	require.NoError(t, err)
	assert.JSONEq(t, `[]`, empty)
	_, err = host.Create(t.Context(), owner, rules.Config{Players: 2, Target: 12}, true, game.RoomOptions{})
	require.NoError(t, err)
	private, err := host.ListingJSON()
	require.NoError(t, err)
	assert.JSONEq(t, `[]`, private)
	code, err := host.Create(t.Context(), owner, rules.Config{Players: 2, Target: 12}, false, game.RoomOptions{})
	require.NoError(t, err)
	before, err := host.ListingJSON()
	require.NoError(t, err)
	var listings []game.Listing
	require.NoError(t, json.Unmarshal([]byte(before), &listings))
	require.Len(t, listings, 1)
	assert.Equal(t, game.Listing{Code: code, Players: 1, Capacity: 2, Target: 12, Host: "Owner"}, listings[0])
	repository.fail = true
	require.Error(t, host.Join(t.Context(), code, guest, nil))
	failed, err := host.ListingJSON()
	require.NoError(t, err)
	assert.Equal(t, before, failed)
	repository.fail = false
	require.NoError(t, host.Join(t.Context(), code, guest, nil))
	after, err := host.ListingJSON()
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal([]byte(after), &listings))
	require.Len(t, listings, 1)
	assert.Equal(t, 2, listings[0].Players)
	assert.True(t, listings[0].Started)
	assert.NotEqual(t, before, after)

	// Started games with no connected seated player close when one leaves.
	view, err := host.Snapshot(code, owner.ID, 0)
	require.NoError(t, err)
	require.NoError(t, host.Handle(t.Context(), code, owner, game.Input{ID: "leave", Revision: view.State.Revision, Command: rules.Command{Type: "leave"}}))
	closed, err := host.ListingJSON()
	require.NoError(t, err)
	assert.JSONEq(t, `[]`, closed)
}

func TestListingJSONConcurrentReadersSeeWholePublications(t *testing.T) {
	repository := &memoryRepository{payloads: make(map[string][]byte)}
	host := game.New(repository, slog.New(slog.NewTextHandler(io.Discard, nil)))
	owner := store.User{ID: "owner", Name: "Owner"}
	code, err := host.Create(t.Context(), owner, rules.Config{Players: 4, Target: 12}, false, game.RoomOptions{})
	require.NoError(t, err)
	start := make(chan struct{})
	var readers sync.WaitGroup
	for range 16 {
		readers.Go(func() {
			<-start
			for range 25 {
				payload, readErr := host.ListingJSON()
				if !assert.NoError(t, readErr) {
					return
				}
				var listings []game.Listing
				if !assert.NoError(t, json.Unmarshal([]byte(payload), &listings)) || !assert.Len(t, listings, 1) {
					return
				}
				assert.Equal(t, code, listings[0].Code)
				assert.Contains(t, []int{1, 2}, listings[0].Players)
			}
		})
	}
	close(start)
	require.NoError(t, host.Join(t.Context(), code, store.User{ID: "guest", Name: "Guest"}, nil))
	readers.Wait()
	payload, err := host.ListingJSON()
	require.NoError(t, err)
	var listings []game.Listing
	require.NoError(t, json.Unmarshal([]byte(payload), &listings))
	require.Len(t, listings, 1)
	assert.Equal(t, 2, listings[0].Players)
}
