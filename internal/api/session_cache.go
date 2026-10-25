package api

import (
	"container/list"
	"context"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/neiam00/retruco-backend/internal/store"
)

const (
	sessionCacheCapacity = 4096
	sessionCacheTTL      = 30 * time.Second
	sessionLookupTimeout = 5 * time.Second
)

type cachedSession struct {
	hash    string
	session store.Session
	until   time.Time
}

type sessionLookup struct {
	done    chan struct{}
	revoked bool
}

// sessionCache is local to the single API instance. Only successful lookups
// are cached, keyed by token hash. LRU eviction bounds retained entries; active
// lookups are removed on completion and never outlive their database timeout.
// mu never covers database I/O. All logout paths must revoke after deletion.
type sessionCache struct {
	mu      sync.Mutex
	entries map[string]*list.Element
	order   list.List
	loading map[string]*sessionLookup
}

func newSessionCache() *sessionCache {
	return &sessionCache{
		entries: make(map[string]*list.Element),
		loading: make(map[string]*sessionLookup),
	}
}

func (c *sessionCache) lookup(ctx context.Context, hash string, load func(context.Context, string) (store.Session, error)) (store.User, error) {
	for {
		if err := ctx.Err(); err != nil {
			return store.User{}, err
		}
		c.mu.Lock()
		if element := c.entries[hash]; element != nil {
			entry := element.Value.(cachedSession)
			if time.Now().Before(entry.until) {
				c.order.MoveToFront(element)
				c.mu.Unlock()
				return entry.session.User, nil
			}
			c.remove(element)
		}
		if pending := c.loading[hash]; pending != nil {
			revoked := pending.revoked
			c.mu.Unlock()
			if revoked {
				return store.User{}, pgx.ErrNoRows
			}
			select {
			case <-ctx.Done():
				return store.User{}, ctx.Err()
			case <-pending.done:
				// Recheck under mu, including revocation. A canceled leader
				// does not cancel unrelated callers waiting for that session.
				continue
			}
		}
		pending := &sessionLookup{done: make(chan struct{})}
		c.loading[hash] = pending
		c.mu.Unlock()

		lookupCtx, cancel := context.WithTimeout(ctx, sessionLookupTimeout)
		session, err := load(lookupCtx, hash)
		if err == nil {
			err = lookupCtx.Err()
		}
		cancel()

		c.mu.Lock()
		delete(c.loading, hash)
		if pending.revoked || (err == nil && !time.Now().Before(session.ExpiresAt)) {
			err = pgx.ErrNoRows
		}
		if err == nil {
			until := time.Now().Add(sessionCacheTTL)
			if session.ExpiresAt.Before(until) {
				until = session.ExpiresAt
			}
			if len(c.entries) >= sessionCacheCapacity {
				c.remove(c.order.Back())
			}
			c.entries[hash] = c.order.PushFront(cachedSession{hash: hash, session: session, until: until})
		}
		if !pending.revoked {
			close(pending.done)
		}
		c.mu.Unlock()
		if err != nil {
			return store.User{}, err
		}
		return session.User, nil
	}
}

// revoke also rejects already-running lookups and wakes their waiters. Keeping
// the marker until that lookup completes prevents stale results from returning
// or repopulating the cache, without accumulating permanent token tombstones.
func (c *sessionCache) revoke(hash string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if element := c.entries[hash]; element != nil {
		c.remove(element)
	}
	if pending := c.loading[hash]; pending != nil && !pending.revoked {
		pending.revoked = true
		close(pending.done)
	}
}

// Caller holds c.mu.
func (c *sessionCache) remove(element *list.Element) {
	delete(c.entries, element.Value.(cachedSession).hash)
	c.order.Remove(element)
}
