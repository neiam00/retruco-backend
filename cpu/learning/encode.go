package learning

import "github.com/neiam00/retruco-backend/rules"

func flag(b bool) float32 {
	if b {
		return 1
	}
	return 0
}
func enum(s string, values ...string) float32 {
	for i, v := range values {
		if s == v {
			return float32(i+1) / float32(len(values))
		}
	}
	return 0
}
func relative(seat, actor, players int) float32 {
	if seat < 0 {
		return -1
	}
	return float32((seat-actor+players)%players) / 3
}
func card(c *rules.Card, vira *rules.Card) []float32 {
	if c == nil {
		return []float32{0, 0, 0, 0}
	}
	strength := float32(0)
	if vira != nil {
		strength = float32(rules.Strength(*c, *vira)) / 17
	}
	return []float32{1, enum(string(c.Suit), "coins", "cups", "spades", "clubs"), float32(c.Number) / 12, strength}
}

// Encode v2 layout: view [0,192), recent cantos [192,352), match voluntary
// counters [352,384), hand counters [384,416), delivered signals [416,476),
// roles/completeness [476,482), family statistics [512,672), observed signal
// counts/rates [672,716), signal completeness [716]. Rest is zero padding.
// Categorical order and normalizers are part of the versioned model ABI.
func Encode(v rules.View, seat int, context Context) [StateSize]float32 {
	var out [StateSize]float32
	if seat < 0 || seat >= v.Config.Players || (v.Config.Players != 2 && v.Config.Players != 4) {
		return out
	}
	i := 0
	add := func(xs ...float32) { copy(out[i:], xs); i += len(xs) }
	add(float32(v.Config.Players)/4, float32(v.Config.Target)/24, flag(v.Config.FlorRequired), enum(string(v.Phase), "cutting", "playing", "hand_over", "match_over"), float32(v.Round)/2)
	add(relative(v.Dealer, seat, v.Config.Players), relative(v.Mano, seat, v.Config.Players), relative(v.Turn, seat, v.Config.Players))
	add(float32(v.Scores[seat%2])/float32(v.Config.Target), float32(v.Scores[1-seat%2])/float32(v.Config.Target), float32(v.TrucoValue)/100, flag(v.MandatoryStack), flag(v.FlorValid))
	add(card(v.Vira, v.Vira)...)
	add(card(v.Preview, v.Vira)...)
	for index := 0; index < 3; index++ {
		var found *rules.HandCard
		for j := range v.Hand {
			if v.Hand[j].Index == index {
				found = &v.Hand[j]
			}
		}
		if found == nil {
			add(0, 0, 0, 0, 0)
		} else {
			add(card(&found.Card, v.Vira)...)
			add(flag(found.Used))
		}
	}
	if v.Holding == nil {
		add(0, 0, 0)
	} else {
		add(float32(v.Holding.Envido)/40, float32(v.Holding.Flor)/50, flag(v.Holding.Receta))
	}
	for offset := 0; offset < 4; offset++ {
		var found *rules.SeatView
		if offset < v.Config.Players {
			for j := range v.Seats {
				if v.Seats[j].Seat == (seat+offset)%v.Config.Players {
					found = &v.Seats[j]
				}
			}
		}
		if found == nil {
			add(make([]float32, 22)...)
			continue
		}
		add(float32(found.Remaining) / 3)
		for _, p := range found.Plays {
			add(card(p.Card, v.Vira)...)
			add(flag(p.Present), flag(p.Burned), flag(p.Concealed))
		}
	}
	for _, w := range []*rules.Wager{v.Pending, v.Suspended} {
		if w == nil {
			add(0, 0, 0, 0, 0, 0, 0)
		} else {
			add(1, enum(w.Kind, "truco", "envido", "ley_envido", "flor", "privo"), flag(w.Caller == seat%2), float32(w.Value)/100, float32(w.Previous)/100, flag(w.Terminal), enum(w.Mode, "con_ley", "sin_ley"))
		}
	}
	for offset := 0; offset < 2; offset++ {
		add(flag(v.Privando[(seat+offset)%2]))
		value := float32(0)
		for _, p := range v.PrivoEnvido {
			if p.Team == (seat+offset)%2 {
				value = float32(p.Value) / 40
			}
		}
		add(value)
	}
	recent := []Canto{}
	var families [4][4][5]float32 // relative seat, canto family, call/accept/decline/raise/other
	for _, c := range context.Cantos {
		if c.Hand == v.HandNumber {
			recent = append(recent, c)
		}
		e := c.Event
		if e.Automatic || e.Seat < 0 || e.Seat >= v.Config.Players {
			continue
		}
		bucket := -1
		switch {
		case e.Type == "call":
			bucket = 0
		case e.Kind == "raise":
			bucket = 1
		case e.Kind == "accept":
			bucket = 2
		case e.Kind == "decline":
			bucket = 3
		case e.Text != "":
			bucket = 4
		}
		if bucket >= 0 {
			offset := (e.Seat-seat+v.Config.Players)%v.Config.Players*8 + bucket
			out[352+offset] += 1
			if c.Hand == v.HandNumber {
				out[384+offset] += 1
			}
		}
		// A declaration can accompany a call/response and must be counted too.
		if e.Text != "" && bucket != 4 {
			offset := (e.Seat-seat+v.Config.Players)%v.Config.Players*8 + 4
			out[352+offset]++
			if c.Hand == v.HandNumber {
				out[384+offset]++
			}
		}
		family := e.Canto
		if e.Type == "call" {
			family = e.Kind
		}
		f := -1
		switch family {
		case "truco":
			f = 0
		case "envido", "ley_envido":
			f = 1
		case "flor":
			f = 2
		case "privo":
			f = 3
		}
		if f >= 0 {
			b := 4
			switch {
			case e.Type == "call":
				b = 0
			case e.Kind == "accept":
				b = 1
			case e.Kind == "decline":
				b = 2
			case e.Kind == "raise":
				b = 3
			}
			families[(e.Seat-seat+v.Config.Players)%v.Config.Players][f][b]++
		}
	}
	if len(recent) > 16 {
		recent = recent[len(recent)-16:]
	}
	i = 192
	for _, c := range recent {
		e := c.Event
		add(1, relative(e.Seat, seat, v.Config.Players), enum(e.Type, "call", "respond", "privo", "play", "stack", "burn"), enum(e.Kind, "truco", "envido", "ley_envido", "flor", "privo", "accept", "decline", "raise", "flowers_no", "falta", "present_flor"), enum(e.Canto, "truco", "envido", "ley_envido", "flor", "privo"), float32(e.Amount)/100, enum(e.Text, "flor", "a_ley", "presento_flor", "flor_flor", "con_mi_flor", "con_flor"), enum(e.Mode, "con_ley", "sin_ley"), flag(e.Automatic), 0)
	}
	// Bounded count transform retains sample size without unbounded inputs.
	for j := 352; j < 416; j++ {
		out[j] = out[j] / (out[j] + 10)
	}
	signals := context.Signals
	if len(signals) > 20 {
		signals = signals[len(signals)-20:]
	}
	i = 416
	for _, s := range signals {
		add(1, relative(s.Seat, seat, v.Config.Players), enum(s.Kind, SignalKinds[1:]...))
	}
	i = 476
	for offset := 0; offset < 4; offset++ {
		add(flag(offset < v.Config.Players && context.CPU[(seat+offset)%v.Config.Players]))
	}
	add(flag(context.Complete))
	defers := false
	if v.Pending != nil && v.Pending.Caller != seat%2 {
		for other := 0; other < v.Config.Players; other++ {
			if other%2 == seat%2 && !context.CPU[other] {
				defers = true
			}
		}
	}
	add(flag(defers))
	i = 512
	for _, seatFamilies := range families {
		for _, counts := range seatFamilies {
			for _, count := range counts {
				add(count / (count + 10))
			}
			total := counts[1] + counts[2] + counts[3] + counts[4]
			add(total / (total + 10))
			for _, count := range counts[1:] {
				rate := float32(0)
				if total > 0 {
					rate = count / total
				}
				add(rate)
			}
		}
	}
	i = 672
	for offset := 0; offset < 4; offset++ {
		if offset >= v.Config.Players {
			add(make([]float32, 11)...)
			continue
		}
		counts := context.SignalCounts[(seat+offset)%v.Config.Players]
		total := float32(0)
		for _, count := range counts {
			x := float32(count)
			total += x
			add(x / (x + 10))
		}
		for _, count := range counts {
			rate := float32(0)
			if total > 0 {
				rate = float32(count) / total
			}
			add(rate)
		}
		add(total / (total + 10))
	}
	add(flag(context.SignalsComplete))
	return out
}

func EncodeAction(c rules.Command) [ActionSize]float32 {
	var x [ActionSize]float32
	groups := [][]string{{"cut", "play", "stack", "burn", "call", "respond"}, {"truco", "envido", "ley_envido", "flor", "accept", "decline", "raise", "flowers_no", "falta", "present_flor"}, {"flor", "a_ley", "presento_flor", "flor_flor", "con_mi_flor", "con_flor"}, {"con_ley", "sin_ley"}}
	values := []string{c.Type, c.Kind, c.Declaration, c.Mode}
	offset := 0
	for j, group := range groups {
		for k, v := range group {
			if values[j] == v {
				x[offset+k] = 1
			}
		}
		offset += len(group) + 1
		if values[j] == "" {
			x[offset-1] = 1
		}
	}
	x[32] = float32(c.Card) / 2
	x[33] = float32(c.Top) / 2
	x[34] = float32(c.Cut) / 39
	x[35] = float32(c.Amount) / 100
	return x
}

// Slot 47 is a decision constraint on the newly admitted ordinary bluff.
// It is removed before neural scoring. Other commands keep their v2 encoding.
func PolicyAction(c rules.Command, openingBluff bool) [ActionSize]float32 {
	x := EncodeAction(c)
	if openingBluff && c == (rules.Command{Type: "call", Kind: "truco", Amount: 3}) {
		x[47] = -1
	}
	return x
}
