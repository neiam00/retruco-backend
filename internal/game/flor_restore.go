package game

import "github.com/neiam00/retruco-backend/rules"

// Recover the opening seat and the order of public flor declarations from the
// hand history. Older saves could also mark a bare A ley as a confirmed flor.
func (m *Match) restoreFlorDeclarations() {
	if m.State == nil {
		return
	}
	var aLey, confirmed [4]bool
	publicFlor, invalidEnvido := false, false
	opener := -1
	for _, record := range m.CantoHistory {
		event := record.Event
		if record.Hand != m.State.HandNumber || event.Seat < 0 || event.Seat >= m.State.Config.Players {
			continue
		}
		if event.Type == "play" && event.Text == "a_ley" {
			aLey[event.Seat] = true
		}
		declaresFlor := event.Text != "" && event.Text != "a_ley"
		if event.Type == "call" && (event.Kind == "envido" || event.Kind == "ley_envido" || event.Kind == "flor") {
			opener = event.Seat
			invalidEnvido = event.Kind == "envido" && (publicFlor || declaresFlor)
		} else if opener >= 0 && event.Seat != opener && declaresFlor {
			invalidEnvido = true
		}
		// Public declaration text contains only A ley or recognized flor wording.
		if declaresFlor ||
			event.Type == "call" && (event.Kind == "flor" || event.Mode == "con_ley") ||
			event.Type == "respond" && (event.Kind == "flowers_no" || event.Kind == "present_flor" || event.Mode == "con_ley") {
			confirmed[event.Seat] = true
			publicFlor = true
		}
	}
	for seat := range m.State.Players {
		if aLey[seat] && !confirmed[seat] {
			m.State.Players[seat].FlorDeclared = false
		}
	}
	if m.State.Phase == rules.Playing && opener >= 0 {
		m.State.EnvidoOpener = opener
	}
	m.State.RestoreActiveHand(invalidEnvido)
}
