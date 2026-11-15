package learning

import (
	"fmt"
	"github.com/neiam00/retruco-backend/cpu"
	"github.com/neiam00/retruco-backend/rules"
	"math/rand/v2"
)

type Observation struct {
	// On completion there is no acting seat or view. Keep the public hand
	// settlement so evaluation can account for the points that ended the match.
	FinalHand    *rules.HandRecap      `json:"finalHand,omitempty"`
	ChoicePolicy string                `json:"choicePolicy,omitempty"`
	View         *rules.View           `json:"view,omitempty"`
	Signals      []Signal              `json:"signals,omitempty"`
	Version      int                   `json:"version"`
	Seat         int                   `json:"seat"`
	Kind         string                `json:"kind"`
	Features     [StateSize]float32    `json:"features"`
	Actions      [][ActionSize]float32 `json:"actions"`
	Legal        []rules.Command       `json:"legal"`
	Baseline     int                   `json:"baseline"`
	Done         bool                  `json:"done"`
	Winner       int                   `json:"winner"`
	Scores       [2]int                `json:"scores"`
	Training     *TrainingObservation  `json:"training,omitempty"`
	SignalMask   *[6]bool              `json:"signalMask,omitempty"`
	TruthMask    *[6]bool              `json:"truthMask,omitempty"`
}

// Environment runs the authoritative rules without transport or wall clocks.
type Environment struct {
	// PlayingSeats opt evaluation participants into the same decision choices as the local browser CPU.
	ChoicePolicy            string
	PlayingSeats            [4]bool
	State                   rules.State
	CPU                     [4]bool
	Communication           Communication
	Cantos                  []Canto
	deals                   *rand.Rand
	visibility              *rand.Rand
	policy                  *rand.Rand
	steps                   int
	training                *trainingState
	TruthfulCPU             bool
	IgnorePartnerSignalsFor *int
	// SignalBlocks[observer][sender] is an offline evaluation intervention.
	// It never enters actor inputs or changes the sender's legal signal support.
	SignalBlocks [4][4]bool
}

func (e *Environment) blockedSignal(observer, sender int) bool {
	return e.SignalBlocks[observer][sender] || (e.IgnorePartnerSignalsFor != nil &&
		*e.IgnorePartnerSignalsFor == observer && e.State.Config.Players == 4 && sender == (observer+2)%4)
}

func (e *Environment) signalContext(observer int, signals []Signal, counts [4][5]uint32) ([]Signal, [4][5]uint32) {
	blocked := false
	for sender := 0; sender < e.State.Config.Players; sender++ {
		if e.blockedSignal(observer, sender) {
			counts[sender] = [5]uint32{}
			blocked = true
		}
	}
	if !blocked {
		return signals, counts
	}
	visible := make([]Signal, 0, len(signals))
	for _, signal := range signals {
		if !e.blockedSignal(observer, signal.Seat) {
			visible = append(visible, signal)
		}
	}
	return visible, counts
}

func NewEnvironment(config rules.Config, seed uint64, roles [4]bool) (*Environment, error) {
	if (config.Players != 2 && config.Players != 4) || (config.Target != 12 && config.Target != 24) {
		return nil, fmt.Errorf("invalid simulation settings")
	}
	// #nosec G404 -- Independent deterministic simulation streams, never production.
	e := &Environment{ChoicePolicy: cpu.PlayingProfile, CPU: roles, deals: rand.New(rand.NewPCG(seed, 1)), visibility: rand.New(rand.NewPCG(seed, 2)), policy: rand.New(rand.NewPCG(seed, 3))}
	s, err := rules.New(config, e.deals.IntN(config.Players), e.deck())
	if err != nil {
		return nil, err
	}
	e.State = s
	e.Communication.ResetMatch()
	e.Communication.StartHand(s.HandNumber)
	return e, nil
}
func (e *Environment) deck() []rules.Card {
	d := rules.Deck()
	e.deals.Shuffle(len(d), func(i, j int) { d[i], d[j] = d[j], d[i] })
	return d
}

func (e *Environment) Observe() Observation {
	if e.State.Phase == rules.MatchOver {
		return Observation{Version: Version, Done: true, Winner: e.State.Winner, Scores: e.State.Scores, FinalHand: e.State.LastHand}
	}
	seat := ActingSeat(e.State, e.CPU)
	kind := "game"
	// Human proxies may signal off-turn too. Bound proxy communication to the
	// same three opportunities while retaining human canto-response authority.
	if s := e.Communication.Offer(rules.View{Config: e.State.Config, Phase: e.State.Phase, Pending: e.State.Pending, Round: e.State.Round, HandNumber: e.State.HandNumber}, [4]bool{true, true, true, true}); s >= 0 {
		seat = s
		kind = "signal"
	}
	v := rules.ViewFor(e.State, seat, false)
	if kind == "game" && e.PlayingSeats[seat] {
		v.Legal = cpu.ChoicesForProfile(v, seat, e.ChoicePolicy)
	}
	signals, counts := e.signalContext(seat, e.Communication.For(seat), e.Communication.CountsFor(seat))
	o := Observation{Version: Version, Seat: seat, Kind: kind, Winner: -1, Features: Encode(v, seat, Context{Cantos: e.Cantos, Signals: signals, CPU: e.CPU, Complete: true, SignalCounts: counts, SignalsComplete: e.Communication.Complete}), Legal: v.Legal, Actions: [][ActionSize]float32{}}
	if kind == "game" && e.PlayingSeats[seat] {
		o.ChoicePolicy = e.ChoicePolicy
	}
	o.View = &v
	o.Scores = e.State.Scores
	o.Signals = signals
	if kind == "signal" {
		o.Legal = nil
		if e.TruthfulCPU {
			mask := TruthfulSignals(v)
			o.TruthMask = &mask
			if e.CPU[seat] {
				o.SignalMask = &mask
			}
		}
		return o
	}
	for _, c := range v.Legal {
		o.Actions = append(o.Actions, EncodeAction(c))
	}
	decision := cpu.Decide(cpu.Request{Version: cpu.ProtocolVersion, Seat: seat, Entropy: 0, View: v})
	if decision.Command != nil {
		for i, c := range v.Legal {
			if c == *decision.Command {
				o.Baseline = i
				break
			}
		}
	}
	return o
}

func (e *Environment) Step(index int) error {
	if e.steps >= 10000 {
		return fmt.Errorf("simulation exceeded step budget")
	}
	o := e.Observe()
	if o.Done {
		return fmt.Errorf("match finished")
	}
	if o.Kind == "signal" {
		if index < 0 || index >= 6 {
			return fmt.Errorf("invalid signal index")
		}
		if o.SignalMask != nil && !o.SignalMask[index] {
			return fmt.Errorf("CPU signal must be truthful")
		}
		var noticed [4]bool
		if index != 0 {
			for seat := 0; seat < e.State.Config.Players; seat++ {
				if seat%2 != o.Seat%2 {
					noticed[seat] = e.visibility.IntN(10) == 0
				}
			}
		}
		if !e.Communication.Consume(o.Seat, e.State.Round, index, e.State.Config.Players, noticed) {
			return fmt.Errorf("signal already consumed")
		}
		if index != 0 && e.training != nil {
			e.recordTrainingSignal(e.Communication.Signals[len(e.Communication.Signals)-1])
		}
	} else {
		if index < 0 || index >= len(o.Legal) {
			return fmt.Errorf("invalid action index")
		}
		command := o.Legal[index]
		if e.State.Phase == rules.Cutting {
			command = o.Legal[e.policy.IntN(len(o.Legal))]
		}
		s, events, err := rules.Apply(e.State, o.Seat, command)
		if err != nil {
			return err
		}
		e.State = s
		for _, event := range events {
			if event.Type == "call" || event.Type == "respond" || event.Type == "privo" || event.Text != "" {
				e.Cantos = append(e.Cantos, Canto{Hand: s.HandNumber, Event: event})
			}
		}
		if s.Phase == rules.HandOver {
			e.State, err = rules.StartHand(s, e.deck())
			if err != nil {
				return err
			}
			e.Communication.StartHand(e.State.HandNumber)
		}
	}
	e.steps++
	ownNone := -1
	if o.Kind == "signal" && index == 0 {
		ownNone = o.Seat
	}
	e.captureTraining(ownNone)
	return nil
}
