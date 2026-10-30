package game

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
)

func (h *Host) persist(ctx context.Context, match *Match, create bool, members []string, previous ...*Match) error {
	finished := match.Expired || (match.State != nil && match.State.Phase == rules.MatchOver)
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	var err error
	if repository, ok := h.repository.(store.PartitionedMatchRepository); ok {
		var before *Match
		if len(previous) != 0 {
			before = previous[0]
		}
		var write store.MatchWrite
		write, err = encodeMatchWrite(match, before)
		if err == nil {
			if create {
				err = repository.CreateMatchParts(ctx, match.Code, write, members)
			} else {
				err = repository.SaveMatchParts(ctx, match.Code, write, finished, members)
			}
		}
	} else {
		var payload []byte
		payload, err = json.Marshal(match)
		if err == nil {
			if create {
				err = h.repository.CreateMatch(ctx, match.Code, payload, members)
			} else {
				err = h.repository.SaveMatch(ctx, match.Code, payload, finished, members)
			}
		}
	}
	if errors.Is(err, store.ErrMatchCodeInUse) {
		return err
	}
	if err != nil {
		h.log.Error("persist match", "code", match.Code, "error", err)
		return fmt.Errorf("no se pudo guardar la jugada. Inténtalo de nuevo")
	}
	return nil
}

func encodeMatchWrite(match, previous *Match) (store.MatchWrite, error) {
	// Explicit outer fields shadow the histories in the embedded match. Their
	// omission avoids traversing, allocating or encoding retained chat/cantos on
	// ordinary moves, while future Match fields continue to persist by default.
	write := store.MatchWrite{}
	var err error
	write.Payload, err = json.Marshal(struct {
		*Match
		Chat         *struct{} `json:"Chat,omitempty"`
		CantoHistory *struct{} `json:"CantoHistory,omitempty"`
	}{Match: match})
	if err != nil {
		return write, err
	}
	write.ChatChanged = previous == nil
	write.CantoHistoryChanged = previous == nil
	if previous != nil {
		if match.chatRevision != nil && previous.chatRevision != nil {
			write.ChatChanged = match.chatRevision != previous.chatRevision
		} else {
			write.ChatChanged = !sameChat(match.Chat, previous.Chat)
		}
		if match.cantoRevision != nil && previous.cantoRevision != nil {
			write.CantoHistoryChanged = match.cantoRevision != previous.cantoRevision
		} else {
			write.CantoHistoryChanged = !sameCantos(match.CantoHistory, previous.CantoHistory)
		}
	}
	if write.ChatChanged {
		write.Chat, err = json.Marshal(match.Chat)
		if err != nil {
			return write, err
		}
	}
	if write.CantoHistoryChanged {
		write.CantoHistory, err = json.Marshal(match.CantoHistory)
	}
	return write, err
}

func sameChat(a, b []Chat) bool {
	return (a == nil) == (b == nil) && slices.Equal(a, b)
}

func sameCantos(a, b []GameEvent) bool {
	return (a == nil) == (b == nil) && slices.Equal(a, b)
}
