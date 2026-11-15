package learning

import (
	"fmt"
	"testing"

	"github.com/neiam00/retruco-backend/rules"
)

func BenchmarkTrainingCapture(b *testing.B) {
	for _, players := range []int{2, 4} {
		b.Run(fmt.Sprintf("players=%d", players), func(b *testing.B) {
			e, err := NewEnvironment(rules.Config{Players: players, Target: 24, FlorRequired: true}, 317, [4]bool{true, true, true, true})
			if err != nil {
				b.Fatal(err)
			}
			e.EnableTraining()
			if err := e.Step(0); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			for b.Loop() {
				e.captureTraining(-1)
			}
		})
	}
}
