package game

import "encoding/json"

// ListingJSON returns an immutable encoding of the current public listings.
// All subscribers share the same encoding until a committed listing change.
// Unlike game snapshots, listings contain no viewer-specific information.
func (h *Host) ListingJSON() (string, error) {
	h.mu.RLock()
	payload := h.listingJSON
	h.mu.RUnlock()
	if payload != "" {
		return payload, nil
	}

	h.mu.Lock()
	defer h.mu.Unlock()
	if h.listingJSON != "" {
		return h.listingJSON, nil
	}
	list := make([]Listing, 0, len(h.listings))
	for _, listing := range h.listings {
		list = append(list, listing)
	}
	encoded, err := json.Marshal(list)
	if err != nil {
		return "", err
	}
	h.listingJSON = string(encoded)
	return h.listingJSON, nil
}
