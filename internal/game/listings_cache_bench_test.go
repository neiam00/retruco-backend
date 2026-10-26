package game

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"testing"
)

// One operation prepares a changed lobby publication for 100 subscribers.
// Network writes are intentionally excluded: caching saves preparation work,
// not the bytes that must still reach each browser.
func BenchmarkListingPublication(b *testing.B) {
	const subscribers = 100
	for _, rooms := range []int{100, 1000} {
		b.Run(fmt.Sprintf("rooms_%d", rooms), func(b *testing.B) {
			host := New(nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
			for index := range rooms {
				code := fmt.Sprintf("ROOM-%04d", index)
				host.listings[code] = Listing{Code: code, Players: 3, Capacity: 4, Target: 24, Host: "Jugador", Started: true}
			}
			b.Run("per_subscriber_encoding", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					for range subscribers {
						if _, err := json.Marshal(host.List()); err != nil {
							b.Fatal(err)
						}
					}
				}
			})
			b.Run("shared_encoding", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					host.mu.Lock()
					host.notifyListings()
					host.mu.Unlock()
					for range subscribers {
						if _, err := host.ListingJSON(); err != nil {
							b.Fatal(err)
						}
					}
				}
				b.ReportMetric(subscribers, "subscribers/op")
			})
		})
	}
}
