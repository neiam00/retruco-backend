package game

// Older rules recorded A ley card plays as confirmed flor. Repair only flags
// with that evidence, preserving actual announcements and the rest of the hand.
func (m *Match) restoreFlorDeclarations() {
	if m.State == nil {
		return
	}
	// Older snapshots have no BareLey field. Only the most recent action can
	// establish the narrow reply window; canto history alone omits silent plays.
	if len(m.Events) > 0 {
		last := m.Events[len(m.Events)-1]
		m.State.BareLey = last.Hand == m.State.HandNumber && last.Event.Type == "play" &&
			last.Event.Text == "a_ley" && m.State.Round == 0 && m.State.Pending == nil
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
			event.Type == "respond" && (event.Kind == "flowers_no" || event.Kind == "present_flor" || event.Mode == "con_ley") {
			confirmed[event.Seat] = true
		}
	}
	for seat := range m.State.Players {
		if aLey[seat] && !confirmed[seat] {
			m.State.Players[seat].FlorDeclared = false
		}
	}
}
