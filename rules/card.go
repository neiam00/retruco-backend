// Package rules implements deterministic Venezuelan truco rules. It has no
// dependencies on transport, storage, wall clocks, or sources of randomness.
package rules

import "fmt"

type Suit string

const (
	Coins  Suit = "coins"
	Cups   Suit = "cups"
	Spades Suit = "spades"
	Clubs  Suit = "clubs"
)

type Card struct {
	Suit   Suit `json:"suit"`
	Number int  `json:"number"`
}

func (c Card) Valid() bool {
	switch c.Suit {
	case Coins, Cups, Spades, Clubs:
		return c.Number >= 1 && c.Number <= 12 && c.Number != 8 && c.Number != 9
	default:
		return false
	}
}

// Deck returns an ordered Spanish deck. Shuffling belongs to the match host.
func Deck() []Card {
	deck := make([]Card, 0, 40)
	for _, suit := range []Suit{Coins, Cups, Spades, Clubs} {
		for _, number := range []int{1, 2, 3, 4, 5, 6, 7, 10, 11, 12} {
			deck = append(deck, Card{Suit: suit, Number: number})
		}
	}
	return deck
}

func Pieces(vira Card) (perico, perica Card) {
	perico = Card{Suit: vira.Suit, Number: 11}
	perica = Card{Suit: vira.Suit, Number: 10}
	if vira.Number == 10 {
		perica.Number = 12
	}
	if vira.Number == 11 {
		perico.Number = 12
	}
	return perico, perica
}

// Strength increases with rank. Equal strengths are a tie, even for teammates.
// The spec ranks the seven of clubs above the seven of cups.
func Strength(card, vira Card) int {
	perico, perica := Pieces(vira)
	switch card {
	case perico:
		return 17
	case perica:
		return 16
	case Card{Spades, 1}:
		return 15
	case Card{Clubs, 1}:
		return 14
	case Card{Spades, 7}:
		return 13
	case Card{Coins, 7}:
		return 12
	case Card{Clubs, 7}:
		return 5
	case Card{Cups, 7}:
		return 4
	}
	switch card.Number {
	case 3:
		return 11
	case 2:
		return 10
	case 1:
		return 9
	case 12:
		return 8
	case 11:
		return 7
	case 10:
		return 6
	case 6:
		return 3
	case 5:
		return 2
	case 4:
		return 1
	default:
		return 0
	}
}

func EnvidoCardValue(card, vira Card) int {
	perico, perica := Pieces(vira)
	switch card {
	case perico:
		return 10
	case perica:
		return 9
	}
	if card.Number >= 10 {
		return 0
	}
	return card.Number
}

type Holding struct {
	Envido int  `json:"envido"`
	Flor   int  `json:"flor"`
	Receta bool `json:"receta"`
}

// Evaluate calculates a complete three-card holding, before declarations or
// team-wide flor penalties. Flor is zero when there is no flor.
func Evaluate(cards [3]Card, vira Card) (Holding, error) {
	if !vira.Valid() {
		return Holding{}, fmt.Errorf("invalid vira")
	}
	seen := map[Card]bool{vira: true}
	for _, card := range cards {
		if !card.Valid() || seen[card] {
			return Holding{}, fmt.Errorf("invalid or duplicate card: %v", card)
		}
		seen[card] = true
	}
	perico, perica := Pieces(vira)
	wild := func(card Card) bool { return card == perico || card == perica }
	result := Holding{Receta: seen[perico] && seen[perica]}
	sum := 20
	var naturalSuit Suit
	flor := true
	for i, card := range cards {
		value := EnvidoCardValue(card, vira)
		sum += value
		result.Envido = max(result.Envido, value)
		if !wild(card) {
			if naturalSuit != "" && naturalSuit != card.Suit {
				flor = false
			}
			naturalSuit = card.Suit
		}
		for _, other := range cards[:i] {
			if card.Suit == other.Suit || wild(card) || wild(other) {
				result.Envido = max(result.Envido, 20+value+EnvidoCardValue(other, vira))
			}
		}
	}
	if flor {
		result.Flor = sum
	}
	return result, nil
}
