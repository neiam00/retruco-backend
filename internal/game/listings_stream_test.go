package game

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestListingStreamCoalescesAndReplaysContiguousRevisions(t *testing.T) {
	h := New(nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
	baseline, changes, unsubscribe, err := h.SubscribeListingStream()
	require.NoError(t, err)
	defer unsubscribe()
	assert.True(t, baseline.Snapshot)
	var snapshot listingSnapshot
	require.NoError(t, json.Unmarshal([]byte(baseline.JSON), &snapshot))
	assert.Empty(t, snapshot.Listings)
	assert.Equal(t, "listings_snapshot", snapshot.Type)

	// Multiple committed changes in one window produce one final upsert. A
	// transient room that disappears in that window never reaches subscribers.
	h.listings["one"] = Listing{Code: "one", Players: 1}
	h.notifyListings()
	h.listings["one"] = Listing{Code: "one", Players: 3}
	h.listings["transient"] = Listing{Code: "transient"}
	h.notifyListings()
	delete(h.listings, "transient")
	h.notifyListings()
	select {
	case <-changes:
		t.Fatal("published before batch")
	default:
	}
	require.NoError(t, h.publishListings())
	select {
	case <-changes:
	default:
		t.Fatal("missing batched update")
	}
	frames, err := h.ListingStreamFrames(baseline.Epoch, baseline.Revision)
	require.NoError(t, err)
	require.Len(t, frames, 1)
	var first listingDelta
	require.NoError(t, json.Unmarshal([]byte(frames[0].JSON), &first))
	assert.Equal(t, []Listing{{Code: "one", Players: 3}}, first.Upserts)
	assert.Empty(t, first.Removed)
	assert.Equal(t, baseline.Revision, first.BaseRevision)
	assert.Equal(t, baseline.Revision+1, first.Revision)
	assert.False(t, frames[0].Snapshot)

	delete(h.listings, "one")
	h.notifyListings()
	require.NoError(t, h.publishListings())
	frames, err = h.ListingStreamFrames(baseline.Epoch, baseline.Revision)
	require.NoError(t, err)
	require.Len(t, frames, 2, "slow readers receive every retained revision")
	var second listingDelta
	require.NoError(t, json.Unmarshal([]byte(frames[1].JSON), &second))
	assert.Equal(t, first.Revision, second.BaseRevision)
	assert.Equal(t, []string{"one"}, second.Removed)
	assert.Empty(t, second.Upserts)
	current, err := h.ListingSnapshotJSON()
	require.NoError(t, err)
	assert.Equal(t, second.Revision, current.Revision)
	frames, err = h.ListingStreamFrames(current.Epoch, current.Revision)
	require.NoError(t, err)
	assert.Empty(t, frames)
	frames, err = h.ListingStreamFrames("previous-process", current.Revision)
	require.NoError(t, err)
	require.Len(t, frames, 1)
	assert.Equal(t, current, frames[0])

	// Waking after a non-visible/private change must not create empty revisions.
	h.notifyListings()
	require.NoError(t, h.publishListings())
	unchanged, err := h.ListingSnapshotJSON()
	require.NoError(t, err)
	assert.Equal(t, current, unchanged)
}

func TestListingStreamBoundsReplayAndResetsOversizedChanges(t *testing.T) {
	h := New(nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
	initial, err := h.ListingSnapshotJSON()
	require.NoError(t, err)
	for i := range listingRingFrames + 1 {
		h.listings["room"] = Listing{Code: "room", Host: fmt.Sprint(i)}
		h.notifyListings()
		require.NoError(t, h.publishListings())
	}
	frames, err := h.ListingStreamFrames(initial.Epoch, initial.Revision)
	require.NoError(t, err)
	require.Len(t, frames, 1)
	assert.True(t, frames[0].Snapshot, "an evicted revision must reset instead of skipping deltas")
	require.LessOrEqual(t, len(h.listingStream.ring), listingRingFrames)
	require.LessOrEqual(t, h.listingStream.ringBytes, listingRingBytes)

	last := frames[0]
	h.listings["room"] = Listing{Code: "room", Host: strings.Repeat("x", listingRingBytes)}
	h.notifyListings()
	require.NoError(t, h.publishListings())
	frames, err = h.ListingStreamFrames(last.Epoch, last.Revision)
	require.NoError(t, err)
	require.Len(t, frames, 1)
	assert.True(t, frames[0].Snapshot)
	assert.Empty(t, h.listingStream.ring)
	assert.Zero(t, h.listingStream.ringBytes)

	last = frames[0]
	delete(h.listings, "room")
	h.notifyListings()
	require.NoError(t, h.publishListings())
	frames, err = h.ListingStreamFrames(last.Epoch, last.Revision)
	require.NoError(t, err)
	require.Len(t, frames, 1)
	assert.False(t, frames[0].Snapshot, "replay resumes after an oversized revision")
}

func BenchmarkListingDeltaPublication(b *testing.B) {
	for _, rooms := range []int{250, 1000} {
		b.Run(fmt.Sprintf("rooms_%d", rooms), func(b *testing.B) {
			h := New(nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
			for i := range rooms {
				code := fmt.Sprintf("room-%d", i)
				h.listings[code] = Listing{Code: code, Players: 3}
			}
			if _, err := h.ListingSnapshotJSON(); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			players := 3
			for b.Loop() {
				players = 5 - players
				h.listings["room-0"] = Listing{Code: "room-0", Players: players}
				h.notifyListings()
				if err := h.publishListings(); err != nil {
					b.Fatal(err)
				}
				frame := h.listingStream.ring[len(h.listingStream.ring)-1]
				b.ReportMetric(float64(len(frame.JSON)), "wire-B/op")
			}
		})
	}
}
