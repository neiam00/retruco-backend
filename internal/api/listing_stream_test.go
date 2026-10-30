package api_test

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/neiam00/retruco-backend/internal/api"
	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type listingMessage struct {
	Type         string         `json:"type"`
	Epoch        string         `json:"epoch"`
	Revision     uint64         `json:"revision"`
	BaseRevision uint64         `json:"baseRevision"`
	Listings     []game.Listing `json:"listings"`
	Upserts      []game.Listing `json:"upserts"`
	Removed      []string       `json:"removed"`
}

func TestListingStreamNegotiatesDeltasAndReconnectBaselines(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
		defer cancel()
		repository := &streamRepository{matches: map[string][]byte{}}
		log := slog.New(slog.NewTextHandler(io.Discard, nil))
		host := game.New(repository, log)
		owner := store.User{ID: "owner", Name: "Owner"}
		privateCode, err := host.Create(ctx, owner, rules.Config{Players: 4, Target: 24}, true, game.RoomOptions{})
		require.NoError(t, err)
		server, err := api.New(ctx, api.Config{}, repository, host, log)
		require.NoError(t, err)
		httpServer := newVirtualHTTPServer(t, server.Handler())
		t.Cleanup(func() { cancel(); httpServer.Close() })
		go host.Run(ctx)
		get := func(path string) *http.Response {
			t.Helper()
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, httpServer.URL+path, nil)
			require.NoError(t, err)
			response, err := httpServer.Client().Do(req)
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, response.StatusCode)
			t.Cleanup(func() { _ = response.Body.Close() })
			return response
		}
		read := func(scanner *bufio.Scanner) listingMessage {
			t.Helper()
			for scanner.Scan() {
				line := scanner.Text()
				if !strings.HasPrefix(line, "data: ") {
					continue
				}
				assert.NotContains(t, line, privateCode)
				var value listingMessage
				require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &value))
				return value
			}
			require.NoError(t, scanner.Err())
			t.Fatal("stream ended before expected listing publication")
			return listingMessage{}
		}
		response := get("/api/matches/events?transport=2")
		assert.Equal(t, "text/event-stream", response.Header.Get("Content-Type"))
		assert.Equal(t, "no-cache", response.Header.Get("Cache-Control"))
		scanner := bufio.NewScanner(response.Body)
		baseline := read(scanner)
		assert.Equal(t, "listings_snapshot", baseline.Type)
		assert.NotEmpty(t, baseline.Epoch)
		assert.Empty(t, baseline.Listings)
		publicCode, err := host.Create(ctx, owner, rules.Config{Players: 4, Target: 24}, false, game.RoomOptions{})
		require.NoError(t, err)
		legacy := get("/api/matches")
		var rows []game.Listing
		require.NoError(t, json.NewDecoder(legacy.Body).Decode(&rows))
		require.Len(t, rows, 1)
		assert.Equal(t, publicCode, rows[0].Code)
		change := read(scanner)
		assert.Equal(t, "listings_delta", change.Type)
		assert.Equal(t, baseline.Epoch, change.Epoch)
		assert.Equal(t, baseline.Revision, change.BaseRevision)
		assert.Equal(t, baseline.Revision+1, change.Revision)
		require.Len(t, change.Upserts, 1)
		assert.Equal(t, publicCode, change.Upserts[0].Code)
		current := get("/api/matches?transport=2")
		var httpSnapshot listingMessage
		require.NoError(t, json.NewDecoder(current.Body).Decode(&httpSnapshot))
		assert.Equal(t, change.Revision, httpSnapshot.Revision)
		assert.Equal(t, rows, httpSnapshot.Listings)
		_ = response.Body.Close()
		reconnected := get("/api/matches/events?transport=2")
		restored := read(bufio.NewScanner(reconnected.Body))
		assert.Equal(t, "listings_snapshot", restored.Type)
		assert.Equal(t, httpSnapshot, restored)
		legacyStream := get("/api/matches/events")
		legacyScanner := bufio.NewScanner(legacyStream.Body)
		require.True(t, legacyScanner.Scan())
		require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(legacyScanner.Text(), "data: ")), &rows))
		require.Len(t, rows, 1)
		assert.Equal(t, publicCode, rows[0].Code)
	})
}
