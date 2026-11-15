// cpu-fixtures emits native policy golden cases for browser fallback parity.
package main

import (
	"encoding/json"
	"github.com/neiam00/retruco-backend/cpu"
	"github.com/neiam00/retruco-backend/cpu/learning"
	"github.com/neiam00/retruco-backend/rules"
	"math/rand/v2"
	"os"
)

type fixture struct {
	Request cpu.Request  `json:"request"`
	Want    cpu.Response `json:"want"`
}

func main() {
	// #nosec G404 -- Seeded fixture generation only.
	random := rand.New(rand.NewPCG(492, 781))
	var cases []fixture
	for trial := 0; trial < 16; trial++ {
		players := 2
		if trial%2 == 1 {
			players = 4
		}
		target := 12
		if trial%4 >= 2 {
			target = 24
		}
		config := rules.Config{Players: players, Target: target, FlorRequired: trial%8 >= 4}
		deal := func() []rules.Card {
			d := rules.Deck()
			random.Shuffle(len(d), func(i, j int) { d[i], d[j] = d[j], d[i] })
			return d
		}
		state, err := rules.New(config, trial%players, deal())
		if err != nil {
			panic(err)
		}
		var reservoir []fixture
		seen := 0
		for steps := 0; steps < 2000 && state.Phase != rules.MatchOver; steps++ {
			if state.Phase == rules.HandOver {
				state, err = rules.StartHand(state, deal())
				if err != nil {
					panic(err)
				}
				continue
			}
			seat := learning.ActingSeat(state, [4]bool{true, true, true, true})
			request := cpu.Request{Version: cpu.ProtocolVersion, Seat: seat, Entropy: random.Uint32(), View: rules.ViewFor(state, seat, false)}
			decision := cpu.Decide(request)
			if decision.Error != "" || decision.Command == nil {
				panic("invalid native decision")
			}
			f := fixture{request, decision}
			seen++
			if len(reservoir) < 16 {
				reservoir = append(reservoir, f)
			} else if index := random.IntN(seen); index < 16 {
				reservoir[index] = f
			}
			command := *decision.Command
			if trial >= 8 {
				command = request.View.Legal[random.IntN(len(request.View.Legal))]
			}
			state, _, err = rules.Apply(state, seat, command)
			if err != nil {
				panic(err)
			}
		}
		if state.Phase != rules.MatchOver {
			panic("fixture match stalled")
		}
		cases = append(cases, reservoir...)
	}
	if err := json.NewEncoder(os.Stdout).Encode(cases); err != nil {
		panic(err)
	}
}
