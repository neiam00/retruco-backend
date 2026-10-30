package store

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/neiam00/retruco-backend/rules"
)

type historySeat struct {
	ID   string
	Name string
	CPU  bool
}

func decodeHistory(payload []byte, userID string, entry *HistoryEntry) ([]historySeat, error) {
	var match struct {
		Code       string
		GameNumber int
		Config     rules.Config
		Expired    bool
		Seats      []historySeat
		State      *struct {
			Scores     [2]int
			Winner     *int
			HandNumber int
		}
	}
	if err := json.Unmarshal(payload, &match); err != nil {
		return nil, fmt.Errorf("decode match history: %w", err)
	}
	entry.RoomCode = match.Code
	if entry.RoomCode == "" {
		entry.RoomCode = entry.Code
	}
	entry.GameNumber = max(1, match.GameNumber)
	entry.Config = match.Config
	entry.Winner, entry.Team = -1, -1
	entry.Players = []HistoryPlayer{}
	for seat, player := range match.Seats {
		if player.ID == "" {
			continue
		}
		entry.Players = append(entry.Players, HistoryPlayer{Name: player.Name, Seat: seat, Team: seat % 2, CPU: player.CPU})
		entry.HasCPU = entry.HasCPU || player.CPU
		if player.ID == userID {
			entry.Team = seat % 2
		}
	}
	entry.Expired = match.Expired
	if match.State != nil {
		entry.Scores, entry.Hands = match.State.Scores, match.State.HandNumber
		if match.State.Winner != nil {
			entry.Winner = *match.State.Winner
		}
	}
	return match.Seats, nil
}

type Performance struct {
	Games   int `json:"games"`
	Wins    int `json:"wins"`
	Losses  int `json:"losses"`
	WinRate int `json:"winRate"`
}

func (p *Performance) add(won bool) {
	p.Games++
	if won {
		p.Wins++
	} else {
		p.Losses++
	}
	p.WinRate = (p.Wins*100 + p.Games/2) / p.Games
}

type PartnerStatistics struct {
	ID    string      `json:"id"`
	Name  string      `json:"name"`
	Human Performance `json:"human"`
	CPU   Performance `json:"cpu"`
}

type Statistics struct {
	Overall  Performance         `json:"overall"`
	Human    Performance         `json:"human"`
	CPU      Performance         `json:"cpu"`
	Partners []PartnerStatistics `json:"partners"`
}

// Statistics covers all completed results, independently of History's 50-row
// display limit. Only public result fields cross the database boundary here.
func (p *Postgres) Statistics(ctx context.Context, userID string) (Statistics, error) {
	result := Statistics{Partners: []PartnerStatistics{}}
	rows, err := p.pool.Query(ctx, `SELECT jsonb_build_object(
		'Seats', m.payload->'Seats', 'Expired', m.payload->'Expired',
		'State', jsonb_build_object('Scores', m.payload#>'{State,Scores}', 'Winner', m.payload#>'{State,Winner}'))
		FROM matches m JOIN match_members mm ON mm.match_code=m.code
		WHERE mm.user_id=$1 AND m.finished
		ORDER BY m.updated_at DESC`, userID)
	if err != nil {
		return result, err
	}
	defer rows.Close()
	partners := make(map[string]*PartnerStatistics)
	for rows.Next() {
		var payload []byte
		if err := rows.Scan(&payload); err != nil {
			return result, err
		}
		entry := HistoryEntry{}
		seats, err := decodeHistory(payload, userID, &entry)
		if err != nil {
			return result, err
		}
		if entry.Expired || entry.Team < 0 || (entry.Winner != 0 && entry.Winner != 1) {
			continue
		}
		won := entry.Winner == entry.Team
		result.Overall.add(won)
		if entry.HasCPU {
			result.CPU.add(won)
		} else {
			result.Human.add(won)
		}
		for seat, player := range seats {
			if player.ID == "" || player.ID == userID || player.CPU || seat%2 != entry.Team {
				continue
			}
			partner := partners[player.ID]
			if partner == nil {
				partner = &PartnerStatistics{ID: player.ID, Name: player.Name}
				partners[player.ID] = partner
			}
			if entry.HasCPU {
				partner.CPU.add(won)
			} else {
				partner.Human.add(won)
			}
		}
	}
	if err := rows.Err(); err != nil {
		return result, err
	}
	for _, partner := range partners {
		result.Partners = append(result.Partners, *partner)
	}
	sort.Slice(result.Partners, func(i, j int) bool {
		a, b := result.Partners[i], result.Partners[j]
		if a.Human.Games+a.CPU.Games != b.Human.Games+b.CPU.Games {
			return a.Human.Games+a.CPU.Games > b.Human.Games+b.CPU.Games
		}
		if a.Name != b.Name {
			return a.Name < b.Name
		}
		return a.ID < b.ID
	})
	return result, nil
}
