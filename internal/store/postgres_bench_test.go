package store_test

import (
	"crypto/rand"
	"fmt"
	"os"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/neiam00/retruco-backend/internal/store"
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
