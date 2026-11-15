package learning

import "github.com/neiam00/retruco-backend/rules"

// TruthfulSignals allows only claims supported by the sender's unplayed cards.
// It uses the same authorized view as inference. Team signals require 2v2;
// human claims in that mode remain unrestricted in truthfulness.
func TruthfulSignals(v rules.View) (mask [6]bool) {
	mask[0] = true
	if v.Config.Players != 4 || v.Vira == nil || !v.Vira.Valid() {
		return
	}
	perico, perica := rules.Pieces(*v.Vira)
	for _, held := range v.Hand {
		if held.Used {
			continue
		}
		switch held.Card {
		case perico:
			mask[1] = true
		case perica:
			mask[2] = true
		case rules.Card{Suit: rules.Spades, Number: 7}:
			mask[3] = true
		case rules.Card{Suit: rules.Coins, Number: 7}:
			mask[4] = true
		}
	}
	mask[5] = mask[1] && mask[2]
	return
}

func TruthfulSignal(v rules.View, kind string) bool {
	mask := TruthfulSignals(v)
	for index, name := range SignalKinds {
		if kind == name {
			return mask[index]
		}
	}
	return false
}
