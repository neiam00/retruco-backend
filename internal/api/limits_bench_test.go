package api

import (
	"fmt"
	"testing"
)

// Exercise the shared admission-control path with quiet clients still occupying
// their one-minute buckets. Active gameplay must not scan all of those clients.
func BenchmarkAllow(b *testing.B) {
	for _, clients := range []int{1, 1000, 10000} {
		b.Run(fmt.Sprintf("clients_%d", clients), func(b *testing.B) {
			server := &Server{limits: make(map[string]window)}
			for index := range clients {
				server.allow(fmt.Sprintf("http:client-%d", index), 120)
			}
			b.ReportAllocs()
			for b.Loop() {
				server.allow("http:client-0", 1<<60)
			}
		})
	}
}
