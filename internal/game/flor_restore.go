package game

// Older rules recorded A ley card plays as confirmed flor. Repair only flags
// with that evidence, preserving actual announcements and the rest of the hand.
func (m *Match) restoreFlorDeclarations() {
	if m.State == nil {
		return
	}
	var aLey, confirmed [4]bool
	for _, record := range m.CantoHistory {
		event := record.Event
		if record.Hand != m.State.HandNumber || event.Seat < 0 || event.Seat >= m.State.Config.Players {
			continue
		}
		if event.Type == "play" && event.Text == "a_ley" {
			aLey[event.Seat] = true
		}
		// Public declaration text contains only A ley or recognized flor wording.
		if event.Text != "" && event.Text != "a_ley" ||
			event.Type == "call" && event.Kind == "flor" ||
			event.Type == "respond" && (event.Kind == "flowers_no" || event.Kind == "present_flor") {
			confirmed[event.Seat] = true
		}
	}
	for seat := range m.State.Players {
		if aLey[seat] && !confirmed[seat] {
			m.State.Players[seat].FlorDeclared = false
		}
	}
}
