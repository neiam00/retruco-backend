package learning

import "github.com/neiam00/retruco-backend/rules"

const CriticSize = 128

// TrainingObservation is exclusively the headless simulator's opt-in channel.
// No production CPU protocol embeds this type. Never pass Privileged to actors.
type TrainingObservation struct {
	Version    int                 `json:"version"`
	Memory     MemoryPage          `json:"memory"`
	Privileged [CriticSize]float32 `json:"privileged"`
}

type trainingState struct {
	journal MemoryJournal
	visible [4][]Signal
}

func (e *Environment) EnableTraining() {
	e.training = &trainingState{}
	e.captureTraining(-1)
}

func (e *Environment) TrainingFeatures(seat int) [StateSize]float32 {
	rows := e.training.journal.Seats[seat]
	return rows[len(rows)-1]
}

func (e *Environment) captureTraining(ownNone int) {
	if e.training == nil {
		return
	}
	t := e.training
	if t.journal.Hand != e.State.HandNumber {
		t.visible = [4][]Signal{}
		t.journal.StartHand(e.State.HandNumber)
	}
	for seat := 0; seat < e.State.Config.Players; seat++ {
		// History encoding needs the seat's authorized cards/public state, not
		// legal commands. Spectator=true suppresses only action enumeration for
		// this perspective; it does not reveal or remove that seat's own hand.
		v := rules.ViewFor(e.State, seat, true)
		signals, counts := e.signalContext(seat, t.visible[seat], e.Communication.CountsFor(seat))
		features := Encode(v, seat, Context{Cantos: e.Cantos, Signals: signals, CPU: e.CPU, Complete: true, SignalCounts: counts, SignalsComplete: true})
		t.journal.Append(seat, features, seat == ownNone)
	}
}

func (e *Environment) recordTrainingSignal(s DeliveredSignal) {
	if e.training == nil {
		return
	}
	for seat, visible := range s.Visible {
		if visible {
			rows := append(e.training.visible[seat], s.Signal)
			if len(rows) > 20 {
				rows = append([]Signal(nil), rows[len(rows)-20:]...)
			}
			e.training.visible[seat] = rows
		}
	}
}

// TrainingData includes current hands but deliberately not State.Deck, future
// deals, RNG, controller identities, interception masks or retrospective labels.
func (e *Environment) TrainingData(seat, hand, cursor int) (*TrainingObservation, error) {
	if e.training == nil {
		return nil, nil
	}
	page, err := e.training.journal.Page(seat, hand, cursor)
	if err != nil {
		return nil, err
	}
	t := &TrainingObservation{Version: 1, Memory: page}
	i := 0
	for relativeSeat := 0; relativeSeat < 4; relativeSeat++ {
		s := (seat + relativeSeat) % e.State.Config.Players
		present := relativeSeat < e.State.Config.Players
		t.Privileged[i] = flag(present)
		i++
		for k, c := range e.State.Players[s].Cards {
			var p *rules.Card
			if present && c.Valid() {
				c := c
				p = &c
			}
			values := card(p, &e.State.Vira)
			copy(t.Privileged[i:], values)
			i += 4
			t.Privileged[i] = flag(present && e.State.Players[s].Used[k])
			i++
		}
	}
	return t, nil
}
