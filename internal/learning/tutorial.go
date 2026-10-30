// Package learning prepares a no-clock practice hand using the same rules and
// authorized views as real matches. The opponent follows a fixed teaching path.
package learning

import (
	"encoding/json"
	"fmt"
	"sync"

	"github.com/neiam00/retruco-backend/rules"
)

type Event struct {
	ID    uint64      `json:"id"`
	Hand  int         `json:"hand"`
	Event rules.Event `json:"event"`
}

type Step struct {
	ID      string         `json:"id"`
	View    rules.View     `json:"view"`
	Command *rules.Command `json:"command,omitempty"`
	Events  []Event        `json:"events"`
}

type Lesson struct {
	Version int    `json:"version"`
	Steps   []Step `json:"steps"`
}

var preparedJSON = sync.OnceValues(func() ([]byte, error) {
	lesson, err := NewLesson()
	if err != nil {
		return nil, err
	}
	return json.Marshal(lesson)
})

// JSON returns a cached immutable lesson for the anonymous read-only endpoint.
// Callers may write these bytes to a response but must not modify them.
func JSON() ([]byte, error) { return preparedJSON() }

func NewLesson() (Lesson, error) {
	deck := rules.Deck()
	draws := []rules.Card{
		{Suit: rules.Coins, Number: 11}, {Suit: rules.Coins, Number: 10},
		{Suit: rules.Cups, Number: 7}, {Suit: rules.Cups, Number: 4},
		{Suit: rules.Spades, Number: 6}, {Suit: rules.Spades, Number: 5},
		{Suit: rules.Coins, Number: 1},
	}
	// With a cut at 20, the engine draws the original positions 19,18,... .
	// Swapping retains a complete valid 40-card deck with no duplicates.
	for index, card := range draws {
		for position, candidate := range deck {
			if card == candidate {
				target := 19 - index
				deck[target], deck[position] = deck[position], deck[target]
				break
			}
		}
	}
	state, err := rules.New(rules.Config{Players: 2, Target: 12, FlorRequired: true}, 1, deck)
	if err != nil {
		return Lesson{}, fmt.Errorf("prepare teaching deck: %w", err)
	}
	lesson := Lesson{Version: 1, Steps: []Step{}}
	events := []Event{}
	apply := func(seat int, command rules.Command) error {
		next, added, err := rules.Apply(state, seat, command)
		if err != nil {
			return fmt.Errorf("prepare teaching action %s: %w", command.Type, err)
		}
		state = next
		for _, event := range added {
			events = append(events, Event{ID: uint64(len(events) + 1), Hand: state.HandNumber, Event: event})
		}
		return nil
	}
	add := func(id string, command *rules.Command) {
		lesson.Steps = append(lesson.Steps, Step{ID: id, View: rules.ViewFor(state, 0, false), Command: command, Events: append([]Event{}, events...)})
	}
	if err := apply(0, rules.Command{Type: "cut", Cut: 20}); err != nil {
		return Lesson{}, err
	}
	add("vira", nil)
	add("pieces", nil)
	envido := rules.Command{Type: "call", Kind: "envido"}
	add("envido", &envido)
	if err := apply(0, envido); err != nil {
		return Lesson{}, err
	}
	if err := apply(1, rules.Command{Type: "respond", Kind: "accept"}); err != nil {
		return Lesson{}, err
	}
	add("envido-accepted", nil)
	first := rules.Command{Type: "play", Card: 1}
	add("first-card", &first)
	if err := apply(0, first); err != nil {
		return Lesson{}, err
	}
	if err := apply(1, rules.Command{Type: "play", Card: 1}); err != nil {
		return Lesson{}, err
	}
	add("first-trick", nil)
	truco := rules.Command{Type: "call", Kind: "truco", Amount: 3}
	add("truco", &truco)
	if err := apply(0, truco); err != nil {
		return Lesson{}, err
	}
	if err := apply(1, rules.Command{Type: "respond", Kind: "accept"}); err != nil {
		return Lesson{}, err
	}
	add("truco-accepted", nil)
	perico := rules.Command{Type: "play", Card: 0}
	add("perico", &perico)
	if err := apply(0, perico); err != nil {
		return Lesson{}, err
	}
	if err := apply(1, rules.Command{Type: "play", Card: 0}); err != nil {
		return Lesson{}, err
	}
	add("recap", nil)
	return lesson, nil
}
