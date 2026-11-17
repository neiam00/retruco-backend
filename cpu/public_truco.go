package cpu

import "github.com/neiam00/retruco-backend/rules"

// publicTrucoWinner proves only a narrow public certainty: the unique perico
// has already been played in a deciding trick. It never infers hidden cards,
// interprets a seña as a promise, or treats ordinary trick leadership as a win.
func publicTrucoWinner(v rules.View) (int, bool) {
	if v.Phase != rules.Playing || v.Vira == nil || !v.Vira.Valid() ||
		(v.Config.Players != 2 && v.Config.Players != 4) || len(v.Seats) != v.Config.Players || v.Round < 1 || v.Round > 2 {
		return -1, false
	}
	seen := [4]bool{}
	for _, seat := range v.Seats {
		if seat.Seat < 0 || seat.Seat >= v.Config.Players || seen[seat.Seat] || seat.Team != seat.Seat%2 {
			return -1, false
		}
		seen[seat.Seat] = true
	}
	completed := func(round int) (int, bool) {
		var ranks [4]int
		for _, seat := range v.Seats {
			play := seat.Plays[round]
			if !play.Present || play.Concealed || (!play.Burned && (play.Card == nil || !play.Card.Valid())) {
				return -1, false
			}
			if play.Burned {
				continue
			}
			ranks[seat.Seat] = rules.Strength(*play.Card, *v.Vira)
		}
		return rules.TrickWinner(ranks, v.Config.Players, 0) % 2, true
	}
	first, complete := completed(0)
	if !complete || first < 0 {
		return -1, false
	}
	if v.Round == 2 {
		second, complete := completed(1)
		if !complete || second < 0 || second == first {
			return -1, false
		}
	}
	for _, seat := range v.Seats {
		play := seat.Plays[v.Round]
		if play.Present && !play.Burned && !play.Concealed && play.Card != nil && rules.Strength(*play.Card, *v.Vira) == 17 {
			if v.Round == 2 || seat.Team == first {
				return seat.Team, true
			}
		}
	}
	return -1, false
}
