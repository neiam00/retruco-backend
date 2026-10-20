// Package cpu chooses moves from an authorized rules view. It deliberately
// accepts no rules.State: the policy cannot inspect the deck or hidden hands.
package cpu

import "github.com/neiam00/retruco-backend/rules"

const ProtocolVersion = 1

type Request struct {
	Version int        `json:"version"`
	Seat    int        `json:"seat"`
	Entropy uint32     `json:"entropy"`
	View    rules.View `json:"view"`
}

type Response struct {
	Version  int            `json:"version"`
	Revision uint64         `json:"revision"`
	Seat     int            `json:"seat"`
	Command  *rules.Command `json:"command,omitempty"`
	Error    string         `json:"error,omitempty"`
}

// Decide runs one bounded pass over the legal action list supplied by the
// authoritative engine. Native and WebAssembly builds use this exact policy.
func Decide(request Request) Response {
	response := Response{Version: ProtocolVersion, Revision: request.View.Revision, Seat: request.Seat}
	if request.Version != ProtocolVersion || request.View.Version != rules.Version {
		response.Error = "La versión del motor CPU no coincide. Recarga la página."
		return response
	}
	if request.Seat < 0 || request.Seat >= request.View.Config.Players || len(request.View.Legal) > 1024 {
		response.Error = "La vista del jugador CPU no es válida."
		return response
	}
	if len(request.View.Legal) == 0 {
		return response
	}
	if request.View.Phase == rules.Cutting {
		// #nosec G115 -- The nonempty action list is bounded to 1024 entries above.
		command := request.View.Legal[int(request.Entropy%uint32(len(request.View.Legal)))]
		response.Command = &command
		return response
	}
	if request.View.Vira == nil || request.View.Holding == nil || request.View.Round < 0 || request.View.Round > 2 {
		response.Error = "Faltan las cartas del jugador CPU."
		return response
	}
	position := evaluate(request.View, request.Seat)
	bestScore := -100000
	for _, command := range request.View.Legal {
		score := position.score(command)
		if score > bestScore {
			selected := command
			response.Command = &selected
			bestScore = score
		}
	}
	return response
}

type position struct {
	view            rules.View
	strengths       map[int]int
	remaining       int
	strongCards     int
	highest         int
	visibleBest     int
	teammateWinning bool
	ordinaryFlor    bool
}

func evaluate(view rules.View, seat int) position {
	p := position{view: view, strengths: make(map[int]int)}
	p.ordinaryFlor = view.FlorValid && view.Holding.Flor > 0 && !view.Holding.Receta
	for _, card := range view.Hand {
		if card.Used {
			continue
		}
		strength := rules.Strength(card.Card, *view.Vira)
		p.strengths[card.Index] = strength
		p.remaining++
		p.highest = max(p.highest, strength)
		if strength >= 12 {
			p.strongCards++
		}
	}
	for _, other := range view.Seats {
		play := other.Plays[view.Round]
		if play.Card == nil || play.Burned || play.Concealed {
			continue
		}
		strength := rules.Strength(*play.Card, *view.Vira)
		if strength > p.visibleBest {
			p.visibleBest = strength
			p.teammateWinning = other.Team == seat%2
		} else if strength == p.visibleBest && other.Team != seat%2 {
			p.teammateWinning = false
		}
	}
	return p
}

func (p position) score(command rules.Command) int {
	score := -1000
	switch command.Type {
	case "play":
		strength := p.strengths[command.Card]
		switch {
		case p.teammateWinning:
			score = 60 - strength
		case p.visibleBest == 0:
			// Take the initiative with a strong card when leading.
			score = 40 + strength
		case strength > p.visibleBest:
			// Spend the cheapest card that beats the visible leader.
			score = 80 - strength
		default:
			score = 30 - strength
		}
	case "stack":
		if p.view.MandatoryStack {
			// In an emparde the top (third-round) card decides first.
			score = 60 + 2*p.strengths[command.Top] + p.strengths[command.Card]
		} else {
			score = 10
		}
	case "call":
		switch command.Kind {
		case "envido", "ley_envido":
			if p.view.Holding.Envido >= 30 && (!p.ordinaryFlor || command.Kind == "ley_envido") {
				score = 110
			}
		case "flor":
			if p.ordinaryFlor && p.view.Holding.Envido >= 30 {
				score = 115
			}
		case "truco":
			if command.Amount == 3 && p.strongCards >= 2 {
				score = 105
			}
		}
	case "respond":
		score = p.responseScore(command)
	}
	return score + p.declarationScore(command)
}

func (p position) responseScore(command rules.Command) int {
	if command.Kind == "flowers_no" {
		return 200
	}
	willing := false
	if p.view.Pending != nil {
		if p.view.Pending.Kind == "truco" {
			willing = p.strongCards > 0 || p.teammateWinning || (p.remaining == 1 && p.highest >= p.visibleBest)
			if p.view.Pending.Value >= 9 {
				willing = p.strongCards >= 2
			}
		} else {
			willing = p.view.Holding.Envido >= 28
			if p.view.Pending.Terminal {
				willing = p.view.Holding.Envido >= 32
			}
		}
	}
	switch command.Kind {
	case "accept":
		if willing {
			return 100
		}
		return 20
	case "decline":
		if !willing {
			return 100
		}
		return 20
	case "raise":
		if p.view.Pending != nil && p.view.Pending.Kind == "truco" && command.Amount == 6 && p.strongCards >= 2 {
			return 115
		}
	}
	return -1000
}

func (p position) declarationScore(command rules.Command) int {
	expected := ""
	if p.ordinaryFlor {
		switch command.Type {
		case "play":
			expected = "flor"
			if p.remaining == 1 {
				expected = "presento_flor"
			} else if p.view.Round == 0 && p.view.Config.Players == 4 {
				// A ley preserves the teammate's possible ordinary flor too.
				expected = "a_ley"
			}
		case "stack":
			expected = "flor_flor"
		case "call", "respond":
			expected = "con_mi_flor"
		}
	}
	score := 0
	if command.Declaration == expected {
		score += 5
	}
	if command.Mode == "con_ley" && p.ordinaryFlor || command.Mode == "sin_ley" && !p.ordinaryFlor {
		score++
	}
	return score
}
