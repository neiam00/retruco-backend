package store_test

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/require"
)

// Run only against a disposable migrated database, like the integration tests.
func BenchmarkSaveMatch(b *testing.B) {
	databaseURL := os.Getenv("DATABASE_TEST_URL")
	if databaseURL == "" {
		b.Skip("DATABASE_TEST_URL must identify a disposable migrated PostgreSQL database")
	}
	for _, players := range []int{1, 2, 6} {
		b.Run(fmt.Sprintf("players_%d", players), func(b *testing.B) {
			pool, err := pgxpool.New(b.Context(), databaseURL)
			require.NoError(b, err)
			defer pool.Close()
			repository := store.New(pool)
			members := make([]string, players)
			for index := range members {
				members[index] = rand.Text()
				require.NoError(b, repository.CreateUser(b.Context(), store.User{ID: members[index], Name: "Benchmark"}, "", ""))
			}
			code := "benchmark-" + rand.Text()
			payload := []byte(`{"State":{"Scores":[6,3],"Winner":-1}}`)
			require.NoError(b, repository.CreateMatch(b.Context(), code, payload, members))
			b.ReportAllocs()
			for b.Loop() {
				if err := repository.SaveMatch(b.Context(), code, payload, false, members); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkSaveMatchOrdinary isolates the database cost of a real ordinary move:
// one durable UPDATE, no membership writes, and retained chat/canto history.
// Run alone on a disposable instance: WAL-B/op includes other database traffic
// if anything else writes to the same PostgreSQL server during measurement.
func BenchmarkSaveMatchOrdinary(b *testing.B) {
	state, err := rules.New(rules.Config{Players: 4, Target: 24}, 0, rules.Deck())
	require.NoError(b, err)
	state, _, err = rules.Apply(state, state.Turn, rules.Command{Type: "cut", Cut: 1})
	require.NoError(b, err)
	chat := make([]map[string]any, 100)
	for i := range chat {
		chat[i] = map[string]any{"id": i + 1, "name": "Player", "text": rand.Text() + rand.Text() + rand.Text() + rand.Text(), "spectator": false}
	}
	cantos := make([]map[string]any, 300)
	for i := range cantos {
		cantos[i] = map[string]any{"id": i + 101, "hand": i / 6, "event": map[string]any{"type": "call", "kind": "truco", "text": "Player called truco"}}
	}
	recent := make(map[string]uint64, 257)
	for i := range 257 {
		recent[fmt.Sprintf("player:request-%d", i)] = uint64(i + 1)
	}
	var full, hot [2][]byte
	for i := range 2 {
		state.Revision++
		payload := map[string]any{
			"Code": "ordinary", "Sequence": i + 1, "Recent": recent, "State": state,
			"Config": state.Config, "CreatorID": "ana", "CreatorName": "Ana",
			"Options": map[string]any{"spectatorsCanSpeak": false}, "Private": false,
			"Seats":    []map[string]string{{"id": "ana", "name": "Ana"}, {"id": "beto", "name": "Beto"}, {"id": "carlos", "name": "Carlos"}, {"id": "diana", "name": "Diana"}},
			"Deadline": time.Date(2026, 10, 26, 10, 0, 0, 0, time.UTC), "LastActivity": time.Date(2026, 10, 26, 9, 59, 30, 0, time.UTC),
		}
		var err error
		hot[i], err = json.Marshal(payload)
		require.NoError(b, err)
		payload["Chat"], payload["CantoHistory"] = chat, cantos
		full[i], err = json.Marshal(payload)
		require.NoError(b, err)
	}
	for _, variant := range []string{"legacy_full_snapshot", "compact_unchanged_history"} {
		b.Run(variant, func(b *testing.B) {
			pool, _ := migrationDatabase(b, 2)
			repository := store.New(pool)
			require.NoError(b, repository.CreateMatch(b.Context(), "ordinary", full[0], nil))
			_, err := pool.Exec(b.Context(), `CREATE TABLE legacy_matches (code text PRIMARY KEY, payload jsonb NOT NULL, finished boolean NOT NULL DEFAULT false, updated_at timestamptz NOT NULL DEFAULT now())`)
			require.NoError(b, err)
			_, err = pool.Exec(b.Context(), `INSERT INTO legacy_matches(code,payload) VALUES('ordinary',$1)`, full[0])
			require.NoError(b, err)
			var startLSN string
			require.NoError(b, pool.QueryRow(b.Context(), `SELECT pg_current_wal_insert_lsn()::text`).Scan(&startLSN))
			b.ReportAllocs()
			iteration := 0
			for b.Loop() {
				iteration++
				if variant == "legacy_full_snapshot" {
					_, err = pool.Exec(b.Context(), `UPDATE legacy_matches SET payload=$2,finished=$3,updated_at=now() WHERE code=$1`, "ordinary", full[iteration%2], false)
				} else {
					err = repository.SaveMatchParts(b.Context(), "ordinary", store.MatchWrite{Payload: hot[iteration%2]}, false, nil)
				}
				if err != nil {
					b.Fatal(err)
				}
			}
			var walBytes float64
			require.NoError(b, pool.QueryRow(b.Context(), `SELECT pg_wal_lsn_diff(pg_current_wal_insert_lsn(),$1::pg_lsn)::float8`, startLSN).Scan(&walBytes))
			b.ReportMetric(walBytes/float64(b.N), "WAL-B/op")
			payloadBytes := len(hot[0])
			if variant == "legacy_full_snapshot" {
				payloadBytes = len(full[0])
			}
			b.ReportMetric(float64(payloadBytes), "payload-B/op")
		})
	}
}
