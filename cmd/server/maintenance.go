package main

import (
	"context"
	"log/slog"
	"time"
)

type sessionCleaner interface {
	CleanupExpiredSessions(context.Context, int) (int64, error)
}

// Bound each sweep so maintenance cannot compete with gameplay for a long
// transaction. Expired authentication tokens are the only records removed;
// users, memberships and completed matches remain intact.
func maintainSessions(ctx context.Context, repository sessionCleaner, log *slog.Logger) {
	ticker := time.NewTicker(time.Hour)
	defer ticker.Stop()
	for {
		work, cancel := context.WithTimeout(ctx, 5*time.Second)
		removed, err := repository.CleanupExpiredSessions(work, 500)
		cancel()
		if err != nil && ctx.Err() == nil {
			log.Warn("expired session cleanup failed", "error", err)
		} else if removed > 0 {
			log.Info("expired sessions removed", "count", removed)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}
