package store_test

import (
	"crypto/rand"
	"encoding/json"
	"os"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMatchCodeReservationCannotReplaceHistory(t *testing.T) {
	databaseURL := os.Getenv("DATABASE_TEST_URL")
	if databaseURL == "" {
		t.Skip("DATABASE_TEST_URL must identify a disposable migrated PostgreSQL database")
	}
	pool, err := pgxpool.New(t.Context(), databaseURL)
	require.NoError(t, err)
	defer pool.Close()
	repository := store.New(pool)
	owner := store.User{ID: rand.Text(), Name: "Historical owner"}
	challenger := store.User{ID: rand.Text(), Name: "New owner"}
	require.NoError(t, repository.CreateUser(t.Context(), owner, "", ""))
	require.NoError(t, repository.CreateUser(t.Context(), challenger, "", ""))
	code := "store-test-" + rand.Text()
	finished := []byte(`{"State":{"Scores":[12,6],"Winner":0}}`)
	require.NoError(t, repository.CreateMatch(t.Context(), code, finished, []string{owner.ID}))
	require.NoError(t, repository.SaveMatch(t.Context(), code, finished, true, []string{owner.ID}))

	err = repository.CreateMatch(t.Context(), code, []byte(`{"State":null}`), []string{challenger.ID})
	require.ErrorIs(t, err, store.ErrMatchCodeInUse)
	ownerHistory, err := repository.History(t.Context(), owner.ID)
	require.NoError(t, err)
	require.Len(t, ownerHistory, 1)
	assert.Equal(t, [2]int{12, 6}, ownerHistory[0].Scores)
	assert.True(t, ownerHistory[0].Finished)
	assert.Equal(t, -1, ownerHistory[0].Team, "legacy snapshots must not invent the user's team")
	assert.Empty(t, ownerHistory[0].Players)
	challengerHistory, err := repository.History(t.Context(), challenger.ID)
	require.NoError(t, err)
	assert.Empty(t, challengerHistory, "a failed reservation must not add the challenger to the old match")

	var stored json.RawMessage
	require.NoError(t, pool.QueryRow(t.Context(), `SELECT payload FROM matches WHERE code=$1`, code).Scan(&stored))
	assert.JSONEq(t, string(finished), string(stored))
	err = repository.SaveMatch(t.Context(), code+"-missing", finished, true, []string{owner.ID})
	require.ErrorIs(t, err, store.ErrMatchNotFound, "an update must not silently recreate a missing match")
}

func TestHistoryIncludesTeamsAndHumanAndCPUPlayers(t *testing.T) {
	databaseURL := os.Getenv("DATABASE_TEST_URL")
	if databaseURL == "" {
		t.Skip("DATABASE_TEST_URL must identify a disposable migrated PostgreSQL database")
	}
	pool, err := pgxpool.New(t.Context(), databaseURL)
	require.NoError(t, err)
	defer pool.Close()
	repository := store.New(pool)
	ana := store.User{ID: rand.Text(), Name: "Ana"}
	beto := store.User{ID: rand.Text(), Name: "Beto"}
	for _, user := range []store.User{ana, beto} {
		require.NoError(t, repository.CreateUser(t.Context(), user, "", ""))
	}
	// Use the persisted JSON contract, including lower-case seat field names.
	payload, err := json.Marshal(map[string]any{
		"State": map[string]any{"Scores": [2]int{6, 12}, "Winner": 1},
		"Seats": []map[string]any{
			{"id": ana.ID, "name": "Ana"},
			{"id": "cpu:1", "name": "CPU 1", "cpu": true},
			{"id": "cpu:2", "name": "CPU 2", "cpu": true},
			{"id": beto.ID, "name": "Beto"},
		},
	})
	require.NoError(t, err)
	code := "history-" + rand.Text()
	members := []string{ana.ID, beto.ID}
	require.NoError(t, repository.CreateMatch(t.Context(), code, payload, members))
	require.NoError(t, repository.SaveMatch(t.Context(), code, payload, true, members))
	for team, user := range []store.User{ana, beto} {
		history, err := repository.History(t.Context(), user.ID)
		require.NoError(t, err)
		require.Len(t, history, 1)
		entry := history[0]
		assert.Equal(t, team, entry.Team)
		assert.Equal(t, 1, entry.Winner)
		assert.Equal(t, [2]int{6, 12}, entry.Scores)
		assert.Equal(t, []store.HistoryPlayer{
			{Name: "Ana", Seat: 0, Team: 0},
			{Name: "CPU 1", Seat: 1, Team: 1, CPU: true},
			{Name: "CPU 2", Seat: 2, Team: 0, CPU: true},
			{Name: "Beto", Seat: 3, Team: 1},
		}, entry.Players)
	}
}

func TestDeleteLobbyRemovesMembershipButKeepsStartedMatches(t *testing.T) {
	databaseURL := os.Getenv("DATABASE_TEST_URL")
	if databaseURL == "" {
		t.Skip("DATABASE_TEST_URL must identify a disposable migrated PostgreSQL database")
	}
	pool, err := pgxpool.New(t.Context(), databaseURL)
	require.NoError(t, err)
	defer pool.Close()
	repository := store.New(pool)
	owner := store.User{ID: rand.Text(), Name: "Lobby owner"}
	require.NoError(t, repository.CreateUser(t.Context(), owner, "", ""))
	waiting, started := "waiting-"+rand.Text(), "started-"+rand.Text()
	require.NoError(t, repository.CreateMatch(t.Context(), waiting, []byte(`{"State":null}`), []string{owner.ID}))
	require.NoError(t, repository.CreateMatch(t.Context(), started, []byte(`{"State":{"Phase":"playing"}}`), []string{owner.ID}))
	require.NoError(t, repository.DeleteLobby(t.Context(), waiting))
	require.NoError(t, repository.DeleteLobby(t.Context(), waiting), "deletion retries are idempotent")
	require.NoError(t, repository.DeleteLobby(t.Context(), started))
	var exists bool
	require.NoError(t, pool.QueryRow(t.Context(), `SELECT EXISTS(SELECT 1 FROM matches WHERE code=$1)`, waiting).Scan(&exists))
	assert.False(t, exists)
	history, err := repository.History(t.Context(), owner.ID)
	require.NoError(t, err)
	require.Len(t, history, 1, "lobby memberships cascade away; started matches remain")
	assert.Equal(t, started, history[0].Code)
}

func TestMatchMembershipUpdatesAreAtomicAndIdempotent(t *testing.T) {
	databaseURL := os.Getenv("DATABASE_TEST_URL")
	if databaseURL == "" {
		t.Skip("DATABASE_TEST_URL must identify a disposable migrated PostgreSQL database")
	}
	pool, err := pgxpool.New(t.Context(), databaseURL)
	require.NoError(t, err)
	defer pool.Close()
	repository := store.New(pool)
	owner := store.User{ID: rand.Text(), Name: "Owner"}
	guest := store.User{ID: rand.Text(), Name: "Guest"}
	for _, user := range []store.User{owner, guest} {
		require.NoError(t, repository.CreateUser(t.Context(), user, "", ""))
	}
	code := "members-" + rand.Text()
	original := []byte(`{"State":{"Scores":[3,0],"Winner":-1}}`)
	updated := []byte(`{"State":{"Scores":[12,0],"Winner":0}}`)
	require.NoError(t, repository.CreateMatch(t.Context(), code, original, []string{owner.ID, owner.ID}))
	require.Error(t, repository.SaveMatch(t.Context(), code, updated, true, []string{guest.ID, "missing-" + rand.Text()}))
	ownerHistory, err := repository.History(t.Context(), owner.ID)
	require.NoError(t, err)
	require.Len(t, ownerHistory, 1)
	assert.Equal(t, [2]int{3, 0}, ownerHistory[0].Scores, "a failed membership must roll back the new game state")
	assert.False(t, ownerHistory[0].Finished)
	guestHistory, err := repository.History(t.Context(), guest.ID)
	require.NoError(t, err)
	assert.Empty(t, guestHistory, "a failed membership must roll back the other new members")
	for range 2 {
		require.NoError(t, repository.SaveMatch(t.Context(), code, updated, true, []string{owner.ID, guest.ID, guest.ID}))
	}
	require.NoError(t, repository.SaveMatch(t.Context(), code, updated, true, nil), "an empty membership list preserves existing members")
	for _, user := range []store.User{owner, guest} {
		history, err := repository.History(t.Context(), user.ID)
		require.NoError(t, err)
		require.Len(t, history, 1)
		assert.Equal(t, [2]int{12, 0}, history[0].Scores)
		assert.True(t, history[0].Finished)
	}
}
