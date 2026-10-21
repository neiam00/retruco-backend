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
	challengerHistory, err := repository.History(t.Context(), challenger.ID)
	require.NoError(t, err)
	assert.Empty(t, challengerHistory, "a failed reservation must not add the challenger to the old match")

	var stored json.RawMessage
	require.NoError(t, pool.QueryRow(t.Context(), `SELECT payload FROM matches WHERE code=$1`, code).Scan(&stored))
	assert.JSONEq(t, string(finished), string(stored))
	err = repository.SaveMatch(t.Context(), code+"-missing", finished, true, []string{owner.ID})
	require.ErrorIs(t, err, store.ErrMatchNotFound, "an update must not silently recreate a missing match")
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
