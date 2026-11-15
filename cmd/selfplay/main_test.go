package main

import (
	"reflect"
	"testing"

	"github.com/neiam00/retruco-backend/cpu/learning"
	"github.com/neiam00/retruco-backend/rules"
)

func TestEncodeProbePreservesSimulatorObservation(t *testing.T) {
	roles := [4]bool{true, true, true, true}
	e, err := learning.NewEnvironment(rules.Config{Players: 4, Target: 12}, 78, roles)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 100; i++ {
		original := e.Observe()
		if original.Done {
			break
		}
		context := learning.Context{Cantos: e.Cantos, Signals: e.Communication.For(original.Seat), CPU: roles, Complete: true, SignalCounts: e.Communication.CountsFor(original.Seat), SignalsComplete: e.Communication.Complete}
		probe, err := encodeView(request{View: original.View, Context: &context, Seat: original.Seat, Kind: original.Kind})
		if err != nil {
			t.Fatal(err)
		}
		if probe.Features != original.Features || !reflect.DeepEqual(probe.Actions, original.Actions) || !reflect.DeepEqual(probe.Legal, original.Legal) {
			t.Fatal("probe diverged from shared observation")
		}
		if !reflect.DeepEqual(original, e.Observe()) {
			t.Fatal("probe mutated simulator")
		}
		action := original.Baseline
		if original.Kind == "signal" {
			action = 1
		}
		if err := e.Step(action); err != nil {
			t.Fatal(err)
		}
	}
}

func TestEncodeProbeBoundsAndCandidateIdentity(t *testing.T) {
	view := rules.View{Config: rules.Config{Players: 2}, Legal: make([]rules.Command, 1024)}
	context := learning.Context{}
	valid := request{View: &view, Context: &context, Kind: "game"}
	for i := range view.Legal {
		view.Legal[i] = rules.Command{Type: "call", Kind: "envido", Amount: i}
	}
	encoded, err := encodeView(valid)
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded.Actions) != 1024 {
		t.Fatal("capacity truncated")
	}
	for i, command := range view.Legal {
		if encoded.Actions[i] != learning.EncodeAction(command) {
			t.Fatal("command identity lost")
		}
	}
	for _, invalid := range []request{
		{View: &view, Context: &context, Kind: "game", Seat: -1},
		{View: &view, Context: &context, Kind: "game", Seat: 2},
		{View: &view, Context: &context, Kind: "other"},
		{View: &view, Kind: "game"},
		{Context: &context, Kind: "game"},
	} {
		if _, err := encodeView(invalid); err == nil {
			t.Fatal("invalid probe accepted")
		}
	}
	view.Legal = append(view.Legal, rules.Command{})
	if _, err := encodeView(valid); err == nil {
		t.Fatal("over-capacity probe accepted")
	}
}
