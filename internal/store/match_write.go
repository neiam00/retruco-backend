package store

import (
	"context"
	"encoding/json"
	"fmt"
)

// MatchWrite separates frequently changing state from retained histories.
// A false Changed flag leaves the existing column untouched. A true flag with
// nil bytes removes the field; []byte("null") preserves an explicit JSON null.
type MatchWrite struct {
	Payload             []byte
	Chat                []byte
	CantoHistory        []byte
	ChatChanged         bool
	CantoHistoryChanged bool
}

// PartitionedMatchRepository lets the game avoid encoding unchanged history.
// The ordinary Repository interface remains usable by other implementations.
type PartitionedMatchRepository interface {
	CreateMatchParts(context.Context, string, MatchWrite, []string) error
	SaveMatchParts(context.Context, string, MatchWrite, bool, []string) error
}

var _ PartitionedMatchRepository = (*Postgres)(nil)

// splitMatch preserves the full-document semantics of CreateMatch/SaveMatch,
// including removing a history key omitted from a replacement document.
func splitMatch(payload []byte) (MatchWrite, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(payload, &fields); err != nil {
		return MatchWrite{}, fmt.Errorf("decode match: %w", err)
	}
	if fields == nil {
		return MatchWrite{}, fmt.Errorf("match must be a JSON object")
	}
	write := MatchWrite{
		Chat: fields["Chat"], CantoHistory: fields["CantoHistory"],
		ChatChanged: true, CantoHistoryChanged: true,
	}
	delete(fields, "Chat")
	delete(fields, "CantoHistory")
	var err error
	write.Payload, err = json.Marshal(fields)
	return write, err
}

func (w MatchWrite) update(code string, finished bool) (string, []any) {
	statement := `UPDATE matches SET payload=$2, finished=$3, updated_at=now()`
	args := []any{code, w.Payload, finished}
	// Only select among constant column names. All values remain parameters.
	if w.ChatChanged {
		args = append(args, nullableJSON(w.Chat))
		statement += fmt.Sprintf(", chat=$%d", len(args))
	}
	if w.CantoHistoryChanged {
		args = append(args, nullableJSON(w.CantoHistory))
		statement += fmt.Sprintf(", canto_history=$%d", len(args))
	}
	return statement + ` WHERE code=$1`, args
}

func nullableJSON(value []byte) any {
	if value == nil {
		return nil
	}
	return value
}
