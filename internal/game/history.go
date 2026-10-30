package game

// A revision identifies an immutable history value. It deliberately has a
// nonzero size: distinct zero-sized allocations may compare equal in Go.
// It never points to an earlier match or history, so old matches are collectible.
type historyRevision struct{ marker byte }

func (m *Match) trackHistories() {
	if m.chatRevision == nil {
		m.chatRevision = &historyRevision{}
	}
	if m.cantoRevision == nil {
		m.cantoRevision = &historyRevision{}
	}
}

// appendRetained always allocates before appending. A previous publication can
// still reference the old buffer, including when it has spare capacity.
func appendRetained[T any](previous, added []T, limit int) []T {
	if len(added) == 0 {
		return previous
	}
	if limit > 0 && len(added) >= limit {
		added = added[len(added)-limit:]
		previous = nil
	} else if limit > 0 && len(previous)+len(added) > limit {
		previous = previous[len(previous)+len(added)-limit:]
	}
	next := make([]T, len(previous)+len(added))
	copy(next, previous)
	copy(next[len(previous):], added)
	return next
}

func (m *Match) appendChat(message Chat) {
	m.Chat = appendRetained(m.Chat, []Chat{message}, 100)
	m.chatRevision = &historyRevision{}
}

// Public snapshots own the pointed-to Team value as well as their slice.
func copyGameEvents(events []GameEvent) []GameEvent {
	copy := append([]GameEvent{}, events...)
	for i := range copy {
		if copy[i].Event.Team != nil {
			team := *copy[i].Event.Team
			copy[i].Event.Team = &team
		}
	}
	return copy
}

func wireHistory[T any](history []T) []T {
	if history == nil {
		return []T{}
	}
	return history
}
