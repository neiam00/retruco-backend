package api

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSessionCacheRefreshesWithoutExtendingDatabaseExpiration(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		cache := newSessionCache()
		expires := time.Now().Add(sessionCacheTTL + time.Second)
		calls := 0
		load := func(context.Context, string) (store.Session, error) {
			calls++
			return store.Session{User: store.User{ID: "user", Name: fmt.Sprint(calls)}, ExpiresAt: expires}, nil
		}
		for range 3 {
			user, err := cache.lookup(t.Context(), "hash", load)
			require.NoError(t, err)
			assert.Equal(t, "1", user.Name)
		}
		assert.Equal(t, 1, calls)
		time.Sleep(sessionCacheTTL)
		user, err := cache.lookup(t.Context(), "hash", load)
		require.NoError(t, err)
		assert.Equal(t, "2", user.Name, "the TTL refreshes user information from storage")
		time.Sleep(time.Second)
		_, err = cache.lookup(t.Context(), "hash", load)
		require.ErrorIs(t, err, pgx.ErrNoRows, "even a stale database result cannot extend expiration")
		assert.Equal(t, 3, calls)
	})
}

func TestSessionCacheBoundsEntriesAndEvictsLeastRecentlyUsed(t *testing.T) {
	cache := newSessionCache()
	calls := 0
	load := func(_ context.Context, hash string) (store.Session, error) {
		calls++
		return store.Session{User: store.User{ID: hash}, ExpiresAt: time.Now().Add(time.Hour)}, nil
	}
	for i := range sessionCacheCapacity {
		_, err := cache.lookup(t.Context(), fmt.Sprint(i), load)
		require.NoError(t, err)
	}
	for _, hash := range []string{"0", "extra", "0"} {
		_, err := cache.lookup(t.Context(), hash, load)
		require.NoError(t, err)
	}
	assert.Equal(t, sessionCacheCapacity+1, calls, "the recently used session stays cached")
	_, err := cache.lookup(t.Context(), "1", load)
	require.NoError(t, err)
	assert.Equal(t, sessionCacheCapacity+2, calls, "overflow must evict an older entry")
}

func TestSessionCacheDoesNotCacheMissingSessionsOrStorageFailures(t *testing.T) {
	for _, failure := range []error{pgx.ErrNoRows, errors.New("storage unavailable")} {
		t.Run(failure.Error(), func(t *testing.T) {
			cache := newSessionCache()
			calls := 0
			load := func(context.Context, string) (store.Session, error) {
				calls++
				if calls == 1 {
					return store.Session{}, failure
				}
				return store.Session{User: store.User{ID: "user"}, ExpiresAt: time.Now().Add(time.Hour)}, nil
			}
			_, err := cache.lookup(t.Context(), "hash", load)
			require.ErrorIs(t, err, failure)
			user, err := cache.lookup(t.Context(), "hash", load)
			require.NoError(t, err)
			assert.Equal(t, "user", user.ID)
			assert.Equal(t, 2, calls)
		})
	}
}

func TestSessionCacheCoalescesLookupsAndAllowsWaiterCancellation(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		cache := newSessionCache()
		release := make(chan struct{})
		calls := 0
		load := func(context.Context, string) (store.Session, error) {
			calls++
			<-release
			return store.Session{User: store.User{ID: "user"}, ExpiresAt: time.Now().Add(time.Hour)}, nil
		}
		results := make(chan error, 10)
		for range 10 {
			go func() { _, err := cache.lookup(t.Context(), "hash", load); results <- err }()
		}
		synctest.Wait()
		assert.Equal(t, 1, calls)
		ctx, cancel := context.WithCancel(t.Context())
		canceled := make(chan error, 1)
		go func() { _, err := cache.lookup(ctx, "hash", load); canceled <- err }()
		synctest.Wait()
		cancel()
		require.ErrorIs(t, <-canceled, context.Canceled)
		close(release)
		for range 10 {
			require.NoError(t, <-results)
		}
		assert.Equal(t, 1, calls)
	})
}

type sessionRepository struct {
	Repository
	mu          sync.Mutex
	sessions    map[string]store.Session
	lookups     int
	deleteError error
	entered     chan struct{}
	release     chan struct{}
}

func (r *sessionRepository) Session(ctx context.Context, hash string) (store.Session, error) {
	r.mu.Lock()
	r.lookups++
	session, ok := r.sessions[hash]
	entered, release := r.entered, r.release
	r.entered, r.release = nil, nil // Stall only the first lookup.
	r.mu.Unlock()
	if entered != nil {
		close(entered)
		select {
		case <-ctx.Done():
			return store.Session{}, ctx.Err()
		case <-release:
		}
	}
	if !ok {
		return store.Session{}, pgx.ErrNoRows
	}
	return session, nil
}

func (r *sessionRepository) DeleteSession(_ context.Context, hash string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.deleteError != nil {
		return r.deleteError
	}
	delete(r.sessions, hash)
	return nil
}

func (r *sessionRepository) History(context.Context, string) ([]store.HistoryEntry, error) {
	return []store.HistoryEntry{}, nil
}

func (r *sessionRepository) Statistics(context.Context, string) (store.Statistics, error) {
	return store.Statistics{Partners: []store.PartnerStatistics{}}, nil
}

func sessionRequest(handler http.Handler, method, path, token string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, path, strings.NewReader(`{}`))
	r.Header.Set("Origin", "https://retruco.test")
	r.Header.Set("Content-Type", "application/json")
	r.AddCookie(&http.Cookie{Name: "retruco_session", Value: token})
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, r)
	return w
}

func TestLogoutInvalidatesCachedSessionButPreservesOtherSessions(t *testing.T) {
	for _, fails := range []bool{false, true} {
		t.Run(fmt.Sprintf("delete_fails=%t", fails), func(t *testing.T) {
			session := store.Session{User: store.User{ID: "user", Registered: true}, ExpiresAt: time.Now().Add(time.Hour)}
			repository := &sessionRepository{sessions: map[string]store.Session{digest("first"): session, digest("second"): session}}
			if fails {
				repository.deleteError = errors.New("storage unavailable")
			}
			server, err := New(t.Context(), Config{Origins: []string{"https://retruco.test"}}, repository, nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
			require.NoError(t, err)
			handler := server.Handler()
			for _, token := range []string{"first", "second", "first", "second"} {
				assert.Equal(t, http.StatusOK, sessionRequest(handler, http.MethodGet, "/api/history", token).Code)
			}
			assert.Equal(t, 2, repository.lookups, "repeated authorization uses cached sessions")
			logout := sessionRequest(handler, http.MethodPost, "/api/logout", "first")
			if fails {
				assert.Equal(t, http.StatusInternalServerError, logout.Code)
				assert.Empty(t, logout.Result().Cookies(), "failed deletion must not acknowledge logout")
				assert.Equal(t, http.StatusOK, sessionRequest(handler, http.MethodGet, "/api/history", "first").Code)
			} else {
				assert.Equal(t, http.StatusOK, logout.Code)
				// Replay the old cookie, rather than merely omitting a cleared cookie.
				assert.Equal(t, http.StatusUnauthorized, sessionRequest(handler, http.MethodGet, "/api/history", "first").Code)
			}
			lookups := repository.lookups
			assert.Equal(t, http.StatusOK, sessionRequest(handler, http.MethodGet, "/api/history", "second").Code)
			assert.Equal(t, lookups, repository.lookups, "logout must not flush other sessions")
		})
	}
}

func TestLogoutRejectsInFlightSessionLookupAndItsWaiters(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		entered, release := make(chan struct{}), make(chan struct{})
		repository := &sessionRepository{
			sessions: map[string]store.Session{digest("token"): {
				User: store.User{ID: "user", Registered: true}, ExpiresAt: time.Now().Add(time.Hour),
			}},
			entered: entered, release: release,
		}
		server, err := New(t.Context(), Config{Origins: []string{"https://retruco.test"}}, repository, nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
		require.NoError(t, err)
		handler := server.Handler()
		leader := make(chan int, 1)
		go func() { leader <- sessionRequest(handler, http.MethodGet, "/api/history", "token").Code }()
		<-entered
		waiter := make(chan int, 1)
		go func() { waiter <- sessionRequest(handler, http.MethodGet, "/api/history", "token").Code }()
		synctest.Wait()
		assert.Equal(t, 1, repository.lookups)
		assert.Equal(t, http.StatusOK, sessionRequest(handler, http.MethodPost, "/api/logout", "token").Code)
		replay := make(chan int, 1)
		go func() { replay <- sessionRequest(handler, http.MethodGet, "/api/history", "token").Code }()
		synctest.Wait()
		for _, result := range []<-chan int{waiter, replay} {
			select {
			case status := <-result:
				assert.Equal(t, http.StatusUnauthorized, status)
			default:
				t.Error("logout must reject waiters and new requests without waiting for the old database read")
			}
		}
		close(release)
		assert.Equal(t, http.StatusUnauthorized, <-leader, "a pre-logout result must not authorize a request or repopulate the cache")
		assert.Equal(t, http.StatusUnauthorized, sessionRequest(handler, http.MethodGet, "/api/history", "token").Code)
	})
}
