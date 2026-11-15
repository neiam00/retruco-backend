// selfplay serves persistent JSON-lines batches over stdin/stdout. Logs belong
// on stderr; no network, database or production credentials are used.
package main

import (
	"fmt"
	"github.com/neiam00/retruco-backend/cpu"
	"github.com/neiam00/retruco-backend/cpu/learning"
	"github.com/neiam00/retruco-backend/rules"
	"os"
)

type request struct {
	ChoicePolicy            string            `json:"choicePolicy,omitempty"`
	PlayingSeats            [4]bool           `json:"playingSeats,omitempty"`
	Protocol                string            `json:"protocol,omitempty"`
	Op                      string            `json:"op"`
	ID                      int               `json:"id"`
	Seed                    uint64            `json:"seed"`
	Config                  rules.Config      `json:"config"`
	Roles                   [4]bool           `json:"roles"`
	Action                  int               `json:"action"`
	Fixture                 string            `json:"fixture,omitempty"`
	View                    *rules.View       `json:"view,omitempty"`
	Seat                    int               `json:"seat,omitempty"`
	Context                 *learning.Context `json:"context,omitempty"`
	Kind                    string            `json:"kind,omitempty"`
	Training                bool              `json:"training,omitempty"`
	TruthfulCPU             bool              `json:"truthfulCPU,omitempty"`
	IgnorePartnerSignalsFor *int              `json:"ignorePartnerSignalsFor,omitempty"`
	SuppressSignalDelivery  [][2]int          `json:"suppressSignalDelivery,omitempty"` // sender, observer
	MemoryHand              int               `json:"memoryHand,omitempty"`
	MemoryCursor            int               `json:"memoryCursor,omitempty"`
	MemoryCursors           [4]int            `json:"memoryCursors,omitempty"`
}
type response struct {
	ID          int                   `json:"id"`
	Observation *learning.Observation `json:"observation,omitempty"`
	Error       string                `json:"error,omitempty"`
}

// Absence of a profile with playingSeats is the original v1 wire contract.
func requestedProfile(name string) (string, error) {
	switch name {
	case "":
		return cpu.OriginalPlayingProfile, nil
	case cpu.OriginalPlayingProfile, cpu.HeadsUpCardProfile, cpu.HeadsUpWagerProfile, cpu.TeamStackProfile, cpu.CompulsoryPericoProfile, cpu.PartnershipProfile, cpu.ReferenceEnvidoProfile, cpu.PlayingProfile:
		return name, nil
	default:
		return "", fmt.Errorf("unknown CPU choice policy %q", name)
	}
}

// encodeView is an offline probe of the shared encoder. It accepts only an
// already-authorized view and delivered context, and never changes a simulator.
func encodeView(r request) (learning.Observation, error) {
	if r.View == nil || r.Context == nil || (r.View.Config.Players != 2 && r.View.Config.Players != 4) || r.Seat < 0 || r.Seat >= r.View.Config.Players || len(r.View.Legal) > 1024 || (r.Kind != "game" && r.Kind != "signal") {
		return learning.Observation{}, fmt.Errorf("invalid authorized encoder probe")
	}
	profile, err := requestedProfile(r.ChoicePolicy)
	if err != nil {
		return learning.Observation{}, err
	}
	view := *r.View
	openingBluff := false
	if r.Kind == "game" && r.PlayingSeats[r.Seat] {
		view.Legal, openingBluff = cpu.PolicyChoices(view, r.Seat, profile)
	}
	o := learning.Observation{Version: learning.Version, Seat: r.Seat, Kind: r.Kind, View: &view,
		Features: learning.Encode(view, r.Seat, *r.Context), Legal: view.Legal,
		Actions: [][learning.ActionSize]float32{}, Signals: r.Context.Signals, Scores: r.View.Scores}
	if r.Kind == "game" {
		if r.PlayingSeats[r.Seat] {
			o.ChoicePolicy = profile
		}
		for _, command := range view.Legal {
			o.Actions = append(o.Actions, learning.PolicyAction(command, openingBluff))
		}
	} else {
		o.Legal = nil
	}
	return o, nil
}

func process(requests []request, envs map[int]*learning.Environment) []response {
	var results []response
	for _, r := range requests {
		result := response{ID: r.ID}
		if r.Op == "encode" {
			observation, err := encodeView(r)
			if err != nil {
				result.Error = err.Error()
			} else {
				result.Observation = &observation
			}
			results = append(results, result)
			continue
		}
		var err error
		switch r.Op {
		case "reset":
			if r.ID < 0 || r.ID >= 64 || (r.Config.Players != 2 && r.Config.Players != 4) {
				err = fmt.Errorf("invalid environment or players")
			} else {
				var e *learning.Environment
				switch r.Fixture {
				case "":
					e, err = learning.NewEnvironment(r.Config, r.Seed, r.Roles)
				case "weak-opening-v1":
					e, err = learning.NewWeakOpening(r.Config, r.Seed)
				default:
					err = fmt.Errorf("unknown diagnostic fixture")
				}
				if err == nil {
					profile, profileErr := requestedProfile(r.ChoicePolicy)
					if profileErr != nil {
						result.Error = profileErr.Error()
						results = append(results, result)
						continue
					}
					e.ChoicePolicy = profile
					e.PlayingSeats = r.PlayingSeats
					e.TruthfulCPU = r.TruthfulCPU
					if r.IgnorePartnerSignalsFor != nil && (*r.IgnorePartnerSignalsFor < 0 || *r.IgnorePartnerSignalsFor >= r.Config.Players) {
						result.Error = "invalid signal observer"
						results = append(results, result)
						continue
					}
					e.IgnorePartnerSignalsFor = r.IgnorePartnerSignalsFor
					for _, pair := range r.SuppressSignalDelivery {
						sender, observer := pair[0], pair[1]
						if !r.Training || sender < 0 || sender >= r.Config.Players || observer < 0 || observer >= r.Config.Players || sender == observer {
							err = fmt.Errorf("invalid offline signal intervention")
							break
						}
						e.SignalBlocks[observer][sender] = true
					}
					if err != nil {
						result.Error = err.Error()
						results = append(results, result)
						continue
					}
					if r.Training {
						e.EnableTraining()
					}
					envs[r.ID] = e
				}
			}
		case "step":
			if envs[r.ID] == nil {
				err = fmt.Errorf("unknown environment")
			} else {
				err = envs[r.ID].Step(r.Action)
			}
		case "observe":
			if envs[r.ID] == nil {
				err = fmt.Errorf("unknown environment")
			}
		default:
			err = fmt.Errorf("unknown operation")
		}
		if err != nil {
			result.Error = err.Error()
		} else {
			o := envs[r.ID].Observe()
			if !o.Done {
				cursor := r.MemoryCursors[o.Seat]
				if r.MemoryCursor != 0 {
					cursor = r.MemoryCursor
				}
				o.Training, err = envs[r.ID].TrainingData(o.Seat, r.MemoryHand, cursor)
				if err != nil {
					result.Error = err.Error()
					results = append(results, result)
					continue
				}
				if o.Training != nil {
					rows := envs[r.ID].TrainingFeatures(o.Seat)
					o.Features = rows
				}
			}
			result.Observation = &o
		}
		results = append(results, result)
	}
	return results
}

func main() {
	if err := serve(os.Stdin, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
