package game

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/neiam00/retruco-backend/cpu"
	"github.com/neiam00/retruco-backend/cpu/learning"
	"github.com/neiam00/retruco-backend/rules"
)

// persistedSignal is exclusively a storage representation. Public Signal keeps
// its visibility mask excluded from JSON even when forwarded in a Snapshot.
type persistedSignal struct {
	Value      Signal
	Visibility [4]bool
}

type CPUFeatures struct {
	ChoicePolicy string                         `json:"choicePolicy,omitempty"`
	Version      int                            `json:"version"`
	State        [learning.StateSize]float32    `json:"state"`
	Actions      [][learning.ActionSize]float32 `json:"actions"`
}

func cpuRoles(m *Match) (roles [4]bool) {
	for seat, player := range m.Seats {
		roles[seat] = player.CPU
	}
	return
}

func (r *room) cpuWorkV2(m *Match) CPUWork {
	w := CPUWork{Version: 2, Controlled: true, Seat: -1}
	if m.State == nil || m.State.Phase == rules.MatchOver || m.State.Phase == rules.HandOver {
		return w
	}
	state := *m.State
	roles := cpuRoles(m)
	communication := m.CPUCommunication
	communication.StartHand(state.HandNumber)
	w.Seat = communication.Offer(rules.View{Config: state.Config, Phase: state.Phase, Pending: state.Pending, Round: state.Round, HandNumber: state.HandNumber}, roles)
	w.Kind = "signal"
	if w.Seat < 0 {
		w.Seat = learning.ActingSeat(state, roles)
		w.Kind = "game"
		if w.Seat < 0 || !roles[w.Seat] {
			w.Seat = -1
			return w
		}
	}
	view := rules.ViewFor(state, w.Seat, false)
	openingBluff := false
	if w.Kind == "game" {
		view.Legal, openingBluff = cpu.PolicyChoices(view, w.Seat, cpu.PlayingProfile)
	}
	if w.Kind == "game" && len(view.Legal) == 0 {
		w.Seat = -1
		return w
	}
	context := m.cpuContext(w.Seat)
	w.View = &view
	if w.Kind == "signal" {
		mask := learning.TruthfulSignals(view)
		w.SignalMask = &mask
	}
	w.Features = &CPUFeatures{Version: learning.Version, State: learning.Encode(view, w.Seat, context), Actions: [][learning.ActionSize]float32{}}
	if w.Kind == "game" {
		w.Features.ChoicePolicy = cpu.PlayingProfile
		for _, c := range view.Legal {
			w.Features.Actions = append(w.Features.Actions, learning.PolicyAction(c, openingBluff))
		}
	}
	r.attachCPUMemory(m, &w)
	// A private transport token is never a model input. Include the controller
	// generation and match sequence so signals, human actions and handoffs revoke
	// previous work even when rules.Revision has not changed.
	encoded, err := json.Marshal(w)
	if err != nil {
		return CPUWork{Version: 2}
	}
	digest := sha256.Sum256(append(encoded, []byte(fmt.Sprintf("/%s/%s/%d/%d/%d", m.Code, r.controller, r.connections[r.controller].order, m.GameNumber, m.Sequence))...))
	w.Token = hex.EncodeToString(digest[:])
	return w
}

// CPUControllerV2 is also used to separate bounded delegated CPU traffic from
// the human action limiter. It does not authorize an action: Handle rechecks it.
func (h *Host) CPUControllerV2(code, userID, connectionID string) bool {
	r := h.lockRoom(code)
	if r == nil {
		return false
	}
	defer r.mu.Unlock()
	c := r.connections[connectionID]
	return connectionID != "" && r.controller == connectionID && c.userID == userID && c.cpuVersion == 2
}
