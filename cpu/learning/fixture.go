package learning

import (
	"fmt"
	"github.com/neiam00/retruco-backend/rules"
)

// NewWeakOpening samples hidden deals conditional on a fixed, weak mano hand
// and vira. It is an offline information-set diagnostic, not a training reset.
// There are no earlier plays, declarations, previews or received signals to
// constrain the posterior. Sequential swaps preserve uniform unseen-card deals.
func NewWeakOpening(config rules.Config, seed uint64) (*Environment, error) {
	if config.Players != 2 {
		return nil, fmt.Errorf("weak-opening fixture requires two players")
	}
	e, err := NewEnvironment(config, seed, [4]bool{true, true, true, true})
	if err != nil {
		return nil, err
	}
	deck := append([]rules.Card(nil), e.State.Deck[:]...)
	deal := func() (rules.State, error) {
		s, err := rules.New(config, 1, deck)
		if err != nil {
			return s, err
		}
		s, _, err = rules.Apply(s, s.Turn, rules.Command{Type: "cut", Cut: 1})
		return s, err
	}
	desired := []rules.Card{{Suit: rules.Coins, Number: 4}, {Suit: rules.Cups, Number: 5}, {Suit: rules.Clubs, Number: 6}, {Suit: rules.Spades, Number: 4}}
	for index, wanted := range desired {
		s, err := deal()
		if err != nil {
			return nil, err
		}
		actual := s.Vira
		if index < 3 {
			actual = s.Players[0].Cards[index]
		}
		for i, card := range deck {
			switch card {
			case actual:
				deck[i] = wanted
			case wanted:
				deck[i] = actual
			}
		}
	}
	e.State, err = deal()
	if err != nil {
		return nil, err
	}
	// Condition on both players choosing no signal before the initial decision.
	e.Communication.Consumed[0][0] = true
	e.Communication.Consumed[1][0] = true
	return e, nil
}
