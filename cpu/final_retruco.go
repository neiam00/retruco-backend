package cpu

import "github.com/neiam00/retruco-backend/rules"

// When only one rival card remains hidden, the raiser has seen the entire
// opposing trick. Count each possible last card using the normal parda rule.
func exposedRetruco(v rules.View, seat int) (wins, losses int) {
	if v.Round != 2 || v.Pending == nil || v.Pending.Kind != "truco" || v.Pending.Value != 6 ||
		v.Pending.Caller == seat%2 || !v.Seats[seat].Plays[2].Present {
		return 0, 0
	}
	d, ok := dealFromView(v, seat)
	if !ok || !d.ordinaryTricks {
		return 0, 0
	}
	n := v.Config.Players
	remaining := -1
	for s := range n {
		if d.plays[0][s] == 0 || d.plays[1][s] == 0 {
			return 0, 0
		}
		if d.plays[2][s] == 0 {
			if remaining >= 0 || s%2 == seat%2 || d.known[s] != 2 {
				return 0, 0
			}
			remaining = s
		}
	}
	if remaining < 0 {
		return 0, 0
	}
	first := rules.TrickWinner(d.plays[0], n, v.Mano)
	if first < 0 {
		return 0, 0
	}
	second := rules.TrickWinner(d.plays[1], n, first)
	if second < 0 || second%2 == first%2 {
		return 0, 0
	}
	for _, card := range d.pool[:d.available] {
		d.plays[2][remaining] = rules.Strength(card, *v.Vira)
		winner := rules.TrickWinner(d.plays[2], n, second)
		if winner < 0 {
			winner = first
		}
		if winner%2 == seat%2 {
			wins++
		} else {
			losses++
		}
	}
	return wins, losses
}
