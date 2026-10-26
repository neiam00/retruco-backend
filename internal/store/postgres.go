package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

type User struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Registered bool   `json:"registered"`
}

// Session carries the database expiration so cached authentication cannot
// extend a session beyond its original lifetime.
type Session struct {
	User      User
	ExpiresAt time.Time
}

type Postgres struct{ pool *pgxpool.Pool }

var (
	ErrMatchCodeInUse = errors.New("match code already exists")
	ErrMatchNotFound  = errors.New("match does not exist")
)

func New(pool *pgxpool.Pool) *Postgres { return &Postgres{pool: pool} }

func (p *Postgres) CreateUser(ctx context.Context, user User, username, hash string) error {
	_, err := p.pool.Exec(ctx, `INSERT INTO users(id,name,username,password_hash) VALUES($1,$2,NULLIF($3,''),NULLIF($4,''))`, user.ID, user.Name, username, hash)
	return err
}

func (p *Postgres) Credentials(ctx context.Context, username string) (User, string, error) {
	var user User
	var hash string
	err := p.pool.QueryRow(ctx, `SELECT id,name,password_hash FROM users WHERE username=$1`, username).Scan(&user.ID, &user.Name, &hash)
	user.Registered = true
	return user, hash, err
}

func (p *Postgres) CreateSession(ctx context.Context, hash, userID string, expires time.Time) error {
	_, err := p.pool.Exec(ctx, `INSERT INTO sessions(token_hash,user_id,expires_at) VALUES($1,$2,$3)`, hash, userID, expires)
	return err
}

func (p *Postgres) Session(ctx context.Context, hash string) (Session, error) {
	var session Session
	err := p.pool.QueryRow(ctx, `SELECT u.id,u.name,u.username IS NOT NULL,s.expires_at FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=$1 AND s.expires_at>now()`, hash).Scan(&session.User.ID, &session.User.Name, &session.User.Registered, &session.ExpiresAt)
	return session, err
}

func (p *Postgres) DeleteSession(ctx context.Context, hash string) error {
	_, err := p.pool.Exec(ctx, `DELETE FROM sessions WHERE token_hash=$1`, hash)
	return err
}

// CleanupExpiredSessions removes a bounded batch using the existing expiry
// index. Active sessions and all user/match history remain unchanged. Lock
// skipping lets maintenance coexist with logout without waiting on its rows.
func (p *Postgres) CleanupExpiredSessions(ctx context.Context, limit int) (int64, error) {
	if limit < 1 || limit > 1000 {
		return 0, fmt.Errorf("session cleanup limit must be between 1 and 1000")
	}
	result, err := p.pool.Exec(ctx, `WITH expired AS (
		SELECT token_hash FROM sessions WHERE expires_at<=now()
		ORDER BY expires_at LIMIT $1 FOR UPDATE SKIP LOCKED
	) DELETE FROM sessions s USING expired e WHERE s.token_hash=e.token_hash`, limit)
	if err != nil {
		return 0, err
	}
	return result.RowsAffected(), nil
}

// SaveMatch adds only the supplied memberships, retaining historical members.
// It replaces the complete legacy JSON document, including its histories.
func (p *Postgres) SaveMatch(ctx context.Context, code string, payload []byte, finished bool, members []string) error {
	write, err := splitMatch(payload)
	if err != nil {
		return err
	}
	return p.SaveMatchParts(ctx, code, write, finished, members)
}

// SaveMatchParts atomically persists an ordinary move with one UPDATE while
// leaving unmodified history columns untouched, including their TOAST storage.
func (p *Postgres) SaveMatchParts(ctx context.Context, code string, write MatchWrite, finished bool, members []string) error {
	if len(members) == 0 {
		statement, args := write.update(code, finished)
		result, err := p.pool.Exec(ctx, statement, args...)
		if err != nil {
			return err
		}
		if result.RowsAffected() == 0 {
			return ErrMatchNotFound
		}
		return nil
	}
	return p.persistMatch(ctx, code, write, finished, members, false)
}

// CreateMatch reserves a code atomically. A finished match keeps its code so a
// new lobby can never overwrite another account's historical match.
func (p *Postgres) CreateMatch(ctx context.Context, code string, payload []byte, members []string) error {
	write, err := splitMatch(payload)
	if err != nil {
		return err
	}
	return p.CreateMatchParts(ctx, code, write, members)
}

func (p *Postgres) CreateMatchParts(ctx context.Context, code string, write MatchWrite, members []string) error {
	return p.persistMatch(ctx, code, write, false, members, true)
}

func (p *Postgres) persistMatch(ctx context.Context, code string, write MatchWrite, finished bool, members []string, create bool) error {
	tx, err := p.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer func() {
		// Rollback is best-effort cleanup and is also called after a successful commit.
		_ = tx.Rollback(ctx)
	}()
	statement, args := write.update(code, finished)
	if create {
		statement = `INSERT INTO matches(code,payload,finished,chat,canto_history) VALUES($1,$2,$3,$4,$5) ON CONFLICT(code) DO NOTHING`
		args = []any{code, write.Payload, finished, nullableJSON(write.Chat), nullableJSON(write.CantoHistory)}
	}
	result, err := tx.Exec(ctx, statement, args...)
	if err != nil {
		return err
	}
	if result.RowsAffected() == 0 {
		if create {
			return ErrMatchCodeInUse
		}
		return ErrMatchNotFound
	}
	if len(members) > 0 {
		// New memberships and the snapshot must commit together. Existing
		// memberships remain intact when a player leaves or later rejoins.
		if _, err = tx.Exec(ctx, `INSERT INTO match_members(match_code,user_id) SELECT $1, unnest($2::text[]) ON CONFLICT DO NOTHING`, code, members); err != nil {
			return err
		}
	}
	return tx.Commit(ctx)
}

// DeleteLobby is idempotent and never removes a started match or its history.
func (p *Postgres) DeleteLobby(ctx context.Context, code string) error {
	_, err := p.pool.Exec(ctx, `DELETE FROM matches WHERE code=$1 AND COALESCE(payload->'State', 'null'::jsonb)='null'::jsonb`, code)
	return err
}

func (p *Postgres) ActiveMatches(ctx context.Context) ([][]byte, error) {
	rows, err := p.pool.Query(ctx, `SELECT payload
		|| CASE WHEN chat IS NULL THEN '{}'::jsonb ELSE jsonb_build_object('Chat', chat) END
		|| CASE WHEN canto_history IS NULL THEN '{}'::jsonb ELSE jsonb_build_object('CantoHistory', canto_history) END
		FROM matches WHERE NOT finished ORDER BY updated_at LIMIT 1000`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var result [][]byte
	for rows.Next() {
		var payload []byte
		if err := rows.Scan(&payload); err != nil {
			return nil, err
		}
		result = append(result, payload)
	}
	return result, rows.Err()
}

type HistoryEntry struct {
	Code      string          `json:"code"`
	Scores    [2]int          `json:"scores"`
	Winner    int             `json:"winner"`
	Finished  bool            `json:"finished"`
	Expired   bool            `json:"expired"`
	UpdatedAt time.Time       `json:"updatedAt"`
	Team      int             `json:"team"`
	Players   []HistoryPlayer `json:"players"`
}

type HistoryPlayer struct {
	Name string `json:"name"`
	Seat int    `json:"seat"`
	Team int    `json:"team"`
	CPU  bool   `json:"cpu"`
}

func (p *Postgres) History(ctx context.Context, userID string) ([]HistoryEntry, error) {
	rows, err := p.pool.Query(ctx, `SELECT m.code,m.payload,m.finished,m.updated_at FROM matches m JOIN match_members mm ON mm.match_code=m.code WHERE mm.user_id=$1 ORDER BY m.updated_at DESC LIMIT 50`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := []HistoryEntry{}
	for rows.Next() {
		var entry HistoryEntry
		var payload []byte
		if err := rows.Scan(&entry.Code, &payload, &entry.Finished, &entry.UpdatedAt); err != nil {
			return nil, err
		}
		var match struct {
			Expired bool
			Seats   []struct {
				ID   string
				Name string
				CPU  bool
			}
			State *struct {
				Scores [2]int
				Winner int
			}
		}
		if err := json.Unmarshal(payload, &match); err != nil {
			return nil, fmt.Errorf("decode match history: %w", err)
		}
		entry.Winner = -1
		entry.Team = -1
		entry.Players = []HistoryPlayer{}
		for seat, player := range match.Seats {
			if player.ID == "" {
				continue
			}
			entry.Players = append(entry.Players, HistoryPlayer{
				Name: player.Name, Seat: seat, Team: seat % 2, CPU: player.CPU,
			})
			if player.ID == userID {
				entry.Team = seat % 2
			}
		}
		entry.Expired = match.Expired
		if match.State != nil {
			entry.Scores, entry.Winner = match.State.Scores, match.State.Winner
		}
		result = append(result, entry)
	}
	return result, rows.Err()
}
