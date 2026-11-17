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
	var invalidLeyOpening, earlierLey bool
	var earlierDouble [2]bool
	opener := -1
	for _, record := range m.CantoHistory {
		event := record.Event
		if record.Hand != m.State.HandNumber || event.Seat < 0 || event.Seat >= m.State.Config.Players {
			continue
		}
		if event.Type == "play" && event.Text == "a_ley" {
			aLey[event.Seat] = true
			earlierLey = true
			player := m.State.Players[event.Seat]
			if m.State.Config.FlorRequired && player.FlorValid && !player.Holding.Receta {
				earlierDouble[m.State.Team(event.Seat)] = true
			}
		}
		declaresFlor := event.Text != "" && event.Text != "a_ley"
		if event.Type == "call" && (event.Kind == "envido" || event.Kind == "ley_envido" || event.Kind == "flor") {
			opener = event.Seat
			invalidLeyOpening = event.Kind == "ley_envido" && event.Mode == "" && confirmed[opener]
			if event.Kind == "ley_envido" && !invalidLeyOpening {
				earlierLey = true
				earlierDouble[m.State.Team(opener)] = true
			}
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
		if confirmed[seat] || aLey[seat] {
			m.State.Players[seat].FlorDeclared = confirmed[seat]
		}
	}
	if m.State.Phase == rules.Playing && opener >= 0 {
		m.State.EnvidoOpener = opener
		// Withdraw an unanswered opening from an old server. Only the
		// negotiation is rewound; the suspended Truco keeps its price and voice.
		if w := m.State.Pending; invalidLeyOpening && w != nil && w.Kind == "ley_envido" && w.Previous == 1 && w.Mode == "" {
			m.State.Pending, m.State.Suspended = m.State.Suspended, nil
			m.State.Ley = earlierLey
			m.State.DoubleFlor[m.State.Team(opener)] = earlierDouble[m.State.Team(opener)]
			m.State.Revision++
		}
	}
	m.State.RestoreActiveHand(invalidEnvido)
}
