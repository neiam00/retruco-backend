package game

import (
	"crypto/rand"
	"encoding/json"
	"maps"
	"slices"
)

const listingRingFrames = 64
const listingRingBytes = 1 << 20

// ListingFrame is an immutable encoded publication with its stream position.
type ListingFrame struct {
	JSON     string
	Epoch    string
	Revision uint64
	Snapshot bool
}

type listingStreamState struct {
	epoch       string
	revision    uint64
	published   map[string]Listing
	dirty       bool
	snapshot    ListingFrame
	ring        []ListingFrame
	ringBytes   int
	subscribers map[chan struct{}]bool
}

type listingSnapshot struct {
	Type     string    `json:"type"`
	Epoch    string    `json:"epoch"`
	Revision uint64    `json:"revision"`
	Listings []Listing `json:"listings"`
}

type listingDelta struct {
	Type         string    `json:"type"`
	Epoch        string    `json:"epoch"`
	BaseRevision uint64    `json:"baseRevision"`
	Revision     uint64    `json:"revision"`
	Upserts      []Listing `json:"upserts"`
	Removed      []string  `json:"removed"`
}

func sortListings(listings []Listing) {
	slices.SortFunc(listings, func(a, b Listing) int {
		if a.Code < b.Code {
			return -1
		}
		if a.Code > b.Code {
			return 1
		}
		return 0
	})
}

// Caller holds h.mu. The first baseline starts from committed listings; later
// baselines use the same batched publication revision as all stream readers.
func (h *Host) listingSnapshotLocked() (ListingFrame, error) {
	s := &h.listingStream
	if s.epoch == "" {
		s.epoch = rand.Text()
		s.published = maps.Clone(h.listings)
		s.dirty = false
		s.subscribers = make(map[chan struct{}]bool)
	}
	if s.snapshot.JSON != "" {
		return s.snapshot, nil
	}
	listings := make([]Listing, 0, len(s.published))
	for _, listing := range s.published {
		listings = append(listings, listing)
	}
	sortListings(listings)
	encoded, err := json.Marshal(listingSnapshot{"listings_snapshot", s.epoch, s.revision, listings})
	if err != nil {
		return ListingFrame{}, err
	}
	s.snapshot = ListingFrame{JSON: string(encoded), Epoch: s.epoch, Revision: s.revision, Snapshot: true}
	return s.snapshot, nil
}

func (h *Host) ListingSnapshotJSON() (ListingFrame, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.listingSnapshotLocked()
}

// SubscribeListingStream cannot lose a publication between baseline and
// subscription. Every connection starts from this complete baseline.
func (h *Host) SubscribeListingStream() (ListingFrame, <-chan struct{}, func(), error) {
	h.mu.Lock()
	baseline, err := h.listingSnapshotLocked()
	if err != nil {
		h.mu.Unlock()
		return ListingFrame{}, nil, nil, err
	}
	updates := make(chan struct{}, 1)
	h.listingStream.subscribers[updates] = true
	h.mu.Unlock()
	return baseline, updates, func() {
		h.mu.Lock()
		defer h.mu.Unlock()
		delete(h.listingStream.subscribers, updates)
	}, nil
}

// ListingStreamFrames returns a contiguous delta chain, or a complete baseline
// when an epoch changed, a reader fell behind, or a frame exceeded ring limits.
func (h *Host) ListingStreamFrames(epoch string, revision uint64) ([]ListingFrame, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	s := &h.listingStream
	if epoch == s.epoch && epoch != "" && revision == s.revision {
		return nil, nil
	}
	if epoch == s.epoch && revision < s.revision && len(s.ring) > 0 && revision >= s.ring[0].Revision-1 {
		frames := make([]ListingFrame, 0, len(s.ring))
		for _, frame := range s.ring {
			if frame.Revision > revision {
				frames = append(frames, frame)
			}
		}
		return frames, nil
	}
	baseline, err := h.listingSnapshotLocked()
	if err != nil {
		return nil, err
	}
	return []ListingFrame{baseline}, nil
}

// publishListings runs on the existing 250ms host tick, including deployment
// pauses. Legacy subscribers still observe every committed change immediately.
func (h *Host) publishListings() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	s := &h.listingStream
	if s.epoch == "" || !s.dirty {
		return nil
	}
	upserts := make([]Listing, 0)
	removed := make([]string, 0)
	for code, listing := range h.listings {
		if old, exists := s.published[code]; !exists || old != listing {
			upserts = append(upserts, listing)
		}
	}
	for code := range s.published {
		if _, exists := h.listings[code]; !exists {
			removed = append(removed, code)
		}
	}
	if len(upserts) == 0 && len(removed) == 0 {
		s.dirty = false
		return nil
	}
	sortListings(upserts)
	slices.Sort(removed)
	encoded, err := json.Marshal(listingDelta{"listings_delta", s.epoch, s.revision, s.revision + 1, upserts, removed})
	if err != nil {
		return err
	}
	for _, listing := range upserts {
		s.published[listing.Code] = listing
	}
	for _, code := range removed {
		delete(s.published, code)
	}
	s.revision++
	s.dirty = false
	s.snapshot = ListingFrame{}
	frame := ListingFrame{JSON: string(encoded), Epoch: s.epoch, Revision: s.revision}
	if len(encoded) > listingRingBytes {
		// No reader can replay across this revision. The next read resets it.
		s.ring = nil
		s.ringBytes = 0
	} else {
		s.ring = append(s.ring, frame)
		s.ringBytes += len(encoded)
		for len(s.ring) > listingRingFrames || s.ringBytes > listingRingBytes {
			s.ringBytes -= len(s.ring[0].JSON)
			s.ring[0] = ListingFrame{}
			s.ring = s.ring[1:]
		}
	}
	for updates := range s.subscribers {
		select {
		case updates <- struct{}{}:
		default:
		}
	}
	return nil
}
