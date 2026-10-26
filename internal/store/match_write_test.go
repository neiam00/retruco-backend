package store_test

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jackc/pgx/v5/stdlib"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/pressly/goose/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A private schema permits migration Up/Down testing without touching other
// tests' tables. DATABASE_TEST_URL must still identify a disposable database.
func migrationDatabase(t testing.TB, version int64) (*pgxpool.Pool, *goose.Provider) {
	t.Helper()
	databaseURL := os.Getenv("DATABASE_TEST_URL")
	if databaseURL == "" {
		t.Skip("DATABASE_TEST_URL must identify a disposable PostgreSQL database")
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	require.NoError(t, err)
	admin, err := pgxpool.NewWithConfig(t.Context(), config.Copy())
	require.NoError(t, err)
	schema := "migration_" + strings.ToLower(rand.Text())
	_, err = admin.Exec(t.Context(), "CREATE SCHEMA "+pgx.Identifier{schema}.Sanitize())
	require.NoError(t, err)
	config.ConnConfig.RuntimeParams["search_path"] = schema
	pool, err := pgxpool.NewWithConfig(t.Context(), config)
	require.NoError(t, err)
	db := stdlib.OpenDB(*config.ConnConfig)
	provider, err := goose.NewProvider(goose.DialectPostgres, db, os.DirFS("../../migrations"), goose.WithDisableGlobalRegistry(true))
	require.NoError(t, err)
	t.Cleanup(func() {
		pool.Close()
		assert.NoError(t, db.Close())
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_, err := admin.Exec(ctx, "DROP SCHEMA "+pgx.Identifier{schema}.Sanitize()+" CASCADE")
		assert.NoError(t, err)
		admin.Close()
	})
	_, err = provider.UpTo(t.Context(), version)
	require.NoError(t, err)
	return pool, provider
}

func TestHistoryMigrationPreservesLegacyRowsAndMaterializesLatestWritesOnRollback(t *testing.T) {
	pool, provider := migrationDatabase(t, 1)
	fixtures := map[string]string{
		"missing":  `{"Code":"missing","Sequence":17,"Recent":{"ana:request":17},"State":null}`,
		"null":     `{"Code":"null","Chat":null,"CantoHistory":null,"State":null}`,
		"empty":    `{"Code":"empty","Chat":[],"CantoHistory":[],"State":null}`,
		"retained": `{"Code":"retained","CreatorID":"ana","Private":true,"Sequence":29,"Recent":{"ana:request":29},"Chat":[{"id":27,"name":"Ana","text":"Persist me","spectator":false}],"CantoHistory":[{"id":28,"hand":2,"event":{"type":"call","kind":"truco"}}],"State":{"Scores":[6,3],"Winner":-1}}`,
	}
	stamp := time.Now().Add(-time.Hour).Truncate(time.Microsecond)
	for code, payload := range fixtures {
		_, err := pool.Exec(t.Context(), `INSERT INTO matches(code,payload,updated_at) VALUES($1,$2,$3)`, code, payload, stamp)
		require.NoError(t, err)
	}
	_, err := provider.UpTo(t.Context(), 2)
	require.NoError(t, err)
	repository := store.New(pool)
	assertRestoredDocuments(t, repository, fixtures)
	var updatedAt time.Time
	require.NoError(t, pool.QueryRow(t.Context(), `SELECT updated_at FROM matches WHERE code='retained'`).Scan(&updatedAt))
	assert.True(t, updatedAt.Equal(stamp), "backfill must not rewrite the user's match history ordering")
	var containsHistory bool
	require.NoError(t, pool.QueryRow(t.Context(), `SELECT payload ? 'Chat' OR payload ? 'CantoHistory' FROM matches WHERE code='retained'`).Scan(&containsHistory))
	assert.False(t, containsHistory)

	// This write takes place after the migration, so rollback must merge its
	// latest cold values rather than recover a stale pre-migration backup.
	latest := `{"Code":"retained","Sequence":30,"Recent":{"ana:request":29,"ana:new":30},"Chat":[{"id":30,"name":"Ana","text":"New durable chat"}],"CantoHistory":[],"State":{"Scores":[9,3],"Winner":-1}}`
	require.NoError(t, repository.SaveMatch(t.Context(), "retained", []byte(latest), false, nil))
	fixtures["retained"] = latest
	_, err = provider.DownTo(t.Context(), 1)
	require.NoError(t, err)
	for code, expected := range fixtures {
		var actual []byte
		require.NoError(t, pool.QueryRow(t.Context(), `SELECT payload FROM matches WHERE code=$1`, code).Scan(&actual))
		assert.JSONEq(t, expected, string(actual), code)
	}
	_, err = provider.UpTo(t.Context(), 2)
	require.NoError(t, err)
	assertRestoredDocuments(t, repository, fixtures)
}

func TestPartitionedWritesKeepHistoryAndReceiptsAtomicAcrossRestart(t *testing.T) {
	pool, _ := migrationDatabase(t, 2)
	repository := store.New(pool)
	owner := store.User{ID: "ana", Name: "Ana"}
	require.NoError(t, repository.CreateUser(t.Context(), owner, "", ""))
	initial := `{"Code":"durable","Sequence":1,"Recent":{"ana:first":1},"Chat":[{"id":1,"name":"Ana","text":"Keep me"}],"CantoHistory":[{"id":1,"hand":1,"event":{"type":"call","kind":"truco"}}],"State":{"Scores":[3,0],"Winner":-1}}`
	require.NoError(t, repository.CreateMatch(t.Context(), "durable", []byte(initial), []string{owner.ID}))
	write := store.MatchWrite{Payload: []byte(`{"Code":"durable","Sequence":2,"Recent":{"ana:first":1,"ana:second":2},"State":{"Scores":[6,0],"Winner":-1}}`)}
	require.NoError(t, repository.SaveMatchParts(t.Context(), "durable", write, false, nil))
	beforeFailure := `{"Code":"durable","Sequence":2,"Recent":{"ana:first":1,"ana:second":2},"Chat":[{"id":1,"name":"Ana","text":"Keep me"}],"CantoHistory":[{"id":1,"hand":1,"event":{"type":"call","kind":"truco"}}],"State":{"Scores":[6,0],"Winner":-1}}`
	assertRestoredDocuments(t, repository, map[string]string{"durable": beforeFailure})
	write.Payload = []byte(`{"Code":"durable","Sequence":3,"Recent":{"ana:third":3},"State":{"Scores":[12,0],"Winner":0}}`)
	write.ChatChanged, write.Chat = true, []byte(`[{"id":3,"name":"Ana","text":"Must roll back"}]`)
	require.Error(t, repository.SaveMatchParts(t.Context(), "durable", write, true, []string{"nonexistent-member"}))
	assertRestoredDocuments(t, repository, map[string]string{"durable": beforeFailure})

	require.NoError(t, repository.SaveMatchParts(t.Context(), "durable", write, false, nil))
	fresh, err := pgxpool.NewWithConfig(t.Context(), pool.Config())
	require.NoError(t, err)
	defer fresh.Close()
	assertRestoredDocuments(t, store.New(fresh), map[string]string{"durable": `{"Code":"durable","Sequence":3,"Recent":{"ana:third":3},"Chat":[{"id":3,"name":"Ana","text":"Must roll back"}],"CantoHistory":[{"id":1,"hand":1,"event":{"type":"call","kind":"truco"}}],"State":{"Scores":[12,0],"Winner":0}}`})
	// Legacy full replacement can remove keys, rather than leaving stale cold
	// columns to reappear when the snapshot is reconstructed after restart.
	require.NoError(t, repository.SaveMatch(t.Context(), "durable", []byte(`{"Code":"durable","State":null}`), false, nil))
	assertRestoredDocuments(t, repository, map[string]string{"durable": `{"Code":"durable","State":null}`})
	require.ErrorIs(t, repository.SaveMatchParts(t.Context(), "absent", write, false, nil), store.ErrMatchNotFound)
}

func TestHistoryMigrationRollsBackUnexpectedDocumentShape(t *testing.T) {
	pool, provider := migrationDatabase(t, 1)
	_, err := pool.Exec(t.Context(), `INSERT INTO matches(code,payload) VALUES('invalid','["not a match object"]')`)
	require.NoError(t, err)
	_, err = provider.UpTo(t.Context(), 2)
	require.ErrorContains(t, err, "roundtrip failed")
	var payload []byte
	require.NoError(t, pool.QueryRow(t.Context(), `SELECT payload FROM matches WHERE code='invalid'`).Scan(&payload))
	assert.JSONEq(t, `["not a match object"]`, string(payload))
	var exists bool
	require.NoError(t, pool.QueryRow(t.Context(), `SELECT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='matches' AND column_name='chat')`).Scan(&exists))
	assert.False(t, exists, "failed data verification must also roll back the schema alteration")
}

func TestExpiredSessionCleanupIsBoundedAndPreservesLiveSessionsAndUsers(t *testing.T) {
	pool, _ := migrationDatabase(t, 2)
	repository := store.New(pool)
	user := store.User{ID: "session-owner", Name: "Owner"}
	require.NoError(t, repository.CreateUser(t.Context(), user, "", ""))
	for i, offset := range []time.Duration{-2 * time.Hour, -time.Hour, time.Hour, 2 * time.Hour} {
		require.NoError(t, repository.CreateSession(t.Context(), string(rune('a'+i)), user.ID, time.Now().Add(offset)))
	}
	deleted, err := repository.CleanupExpiredSessions(t.Context(), 1)
	require.NoError(t, err)
	assert.EqualValues(t, 1, deleted)
	var remaining int
	require.NoError(t, pool.QueryRow(t.Context(), `SELECT count(*) FROM sessions`).Scan(&remaining))
	assert.Equal(t, 3, remaining)
	deleted, err = repository.CleanupExpiredSessions(t.Context(), 100)
	require.NoError(t, err)
	assert.EqualValues(t, 1, deleted)
	deleted, err = repository.CleanupExpiredSessions(t.Context(), 100)
	require.NoError(t, err)
	assert.Zero(t, deleted)
	for _, token := range []string{"c", "d"} {
		session, err := repository.Session(t.Context(), token)
		require.NoError(t, err)
		assert.Equal(t, user, session.User)
	}
	require.NoError(t, pool.QueryRow(t.Context(), `SELECT count(*) FROM users`).Scan(&remaining))
	assert.Equal(t, 1, remaining)
	_, err = repository.CleanupExpiredSessions(t.Context(), 0)
	require.Error(t, err)
	_, err = repository.CleanupExpiredSessions(t.Context(), 1001)
	require.Error(t, err)
}

func assertRestoredDocuments(t *testing.T, repository *store.Postgres, expected map[string]string) {
	t.Helper()
	rows, err := repository.ActiveMatches(t.Context())
	require.NoError(t, err)
	require.Len(t, rows, len(expected))
	for _, row := range rows {
		var metadata struct{ Code string }
		require.NoError(t, json.Unmarshal(row, &metadata))
		require.Contains(t, expected, metadata.Code)
		assert.JSONEq(t, expected[metadata.Code], string(row), metadata.Code)
	}
}
