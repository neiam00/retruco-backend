package main

import (
	"context"
	"crypto/rand"
	"database/sql"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Exercise the shipped executable, not goose directly: rollback version
// arguments must survive the CLI boundary before an old API can be restarted.
func TestMigrationCLIForwardsTargetVersionAndPreservesHistory(t *testing.T) {
	databaseURL := os.Getenv("DATABASE_TEST_URL")
	if databaseURL == "" {
		t.Skip("DATABASE_TEST_URL must point to a disposable PostgreSQL database")
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	db, err := sql.Open("pgx", databaseURL)
	require.NoError(t, err)
	t.Cleanup(func() { _ = db.Close() })
	schema := "cli_" + strings.ToLower(rand.Text())
	quoted := pgx.Identifier{schema}.Sanitize()
	_, err = db.ExecContext(ctx, "CREATE SCHEMA "+quoted)
	require.NoError(t, err)
	t.Cleanup(func() {
		cleanup, done := context.WithTimeout(context.Background(), 5*time.Second)
		defer done()
		_, _ = db.ExecContext(cleanup, "DROP SCHEMA "+quoted+" CASCADE")
	})
	parsed, err := url.Parse(databaseURL)
	require.NoError(t, err)
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	bin := filepath.Join(t.TempDir(), "migrate")
	build := exec.CommandContext(ctx, "go", "build", "-o", bin, ".")
	output, err := build.CombinedOutput()
	require.NoError(t, err, "%s", output)
	run := func(args ...string) {
		t.Helper()
		command := exec.CommandContext(ctx, bin, args...)
		command.Dir = filepath.Join("..", "..")
		command.Env = append(os.Environ(), "APP_ENV=", "DATABASE_URL="+parsed.String())
		output, err := command.CombinedOutput()
		require.NoError(t, err, "%s", output)
	}
	run("up")
	_, err = db.ExecContext(ctx, "INSERT INTO "+quoted+`.matches(code,payload,chat,canto_history) VALUES('rollback', '{"Code":"rollback","Sequence":8}', '[{"id":8,"text":"latest durable chat"}]', '[]')`)
	require.NoError(t, err)
	run("down-to", "1")
	var payload string
	require.NoError(t, db.QueryRowContext(ctx, "SELECT payload::text FROM "+quoted+".matches WHERE code='rollback'").Scan(&payload))
	assert.JSONEq(t, `{"Code":"rollback","Sequence":8,"Chat":[{"id":8,"text":"latest durable chat"}],"CantoHistory":[]}`, payload)
	run("up-to", "2")
	var chat string
	require.NoError(t, db.QueryRowContext(ctx, "SELECT chat::text FROM "+quoted+".matches WHERE code='rollback'").Scan(&chat))
	assert.JSONEq(t, `[{"id":8,"text":"latest durable chat"}]`, chat)
}
