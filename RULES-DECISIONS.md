# Implementation decisions

The source of game requirements is `../truco-spec.md`. The following direct user
clarifications override that document where they differ (2026-10-19):

- Use manual constructor injection for service wiring.
- Ley envido wagers bypass the opposing unannounced ordinary-flor block, subject
  to the table-wide declared-flor restriction below.
- Flor wording is manually selected, including when playing pegado.
- Ordinary envido without flor wording cancels only its caller's ordinary flor.
  Receta remains protected.
- An opposing valid ordinary flor still blocks ordinary envido with **A las flores no**.
- **A ley, envido** preserves flor and scores envido before flor. If envido reaches
  the winning score, the match ends before flor or truco is scored.
- Accepting envido closes all envido/flor-contest betting for that hand.
- An accepted envido wager replaces receta's automatic envido point whether its
  team wins or loses the wager.
- Shared ordinary-envido/truco response timeouts preserve ordinary flores.
  Flor-contest timeouts invalidate only the speaker's ordinary flor.
- Chat may remain in saved match snapshots (direct user override on 2026-10-19).
  Do not write chat text to application logs.
- The public API remains under `https://retru.co.ve/api`; the frontend edge
  forwards it to the configured NGINX backend origin.
- Accepted envido reveals each player's comparison value at settlement, without
  revealing otherwise concealed cards.
- Unstarted tables expire after 24 hours without activity.
- Saying "Con mi flor" while calling or answering a canto counts as declaring
  that valid ordinary flor for later flor-contest eligibility.
- Voluntary pegado bottoms stay concealed when the first-round winning team
  also has the uniquely winning non-stacked third-round card. Top-card burning
  still follows the actual second-round comparison.

Direct user clarifications on 2026-10-25 supersede earlier declaration rules:

- **Flor**, **Presento flor**, **Flor flor**, **Con mi flor**, and **Con flor**
  preserve ordinary flor on any card play or canto, including the final card,
  pegado, calls, acceptances, raises, and refusals. Missing declarations still
  burn flor when flor por derecho is enabled; the first-card A ley team rule
  still applies.
- Calling ordinary envido with flor wording preserves the caller's flor.
  **A ley, envido** cannot include wording that confirms flor, such as
  **Con mi flor**: A ley only says the player may hold flor. Explicit
  **Con ley, envido** and **Sin ley, envido** retain their separate mode rules.
- Explicitly saying **Sin ley** when calling or answering envido, including
  refusal, burns only the speaker's ordinary flor, even with flor wording and
  even without flor por derecho. It does not burn another player's flor.
- **Con mi flor me achico** declines the flor-contest envido and burns only the
  speaker's flor. Their teammate's flor remains valid. Losing an accepted flor
  contest still invalidates the losing team's ordinary flores at settlement.
- Receta retains its existing protection against invalidation.

Direct user clarification on 2026-10-26 supersedes earlier envido initiation rules:

- After any player has declared flor, including the caller or their teammate,
  nobody may start ordinary envido or a ley/con ley/sin ley envido. Adding
  **Con mi flor** does not bypass this restriction. It also applies when
  interrupting truco and remains in force if the declared flor later burns.
- A player with a valid ordinary flor may instead call **Flor tengo y la mía
  envida**, subject to the existing opposing-flor and first-card window rules.
- **A ley** does not confirm flor and must not record a flor declaration. Merely
  holding an unannounced flor does not trigger the table-wide restriction.
- Check prior declarations before processing a new canto's prefix. If no flor
  has been declared yet, the existing **Con mi flor, envido** rule still applies.
- On restore, use the current hand's saved canto history (or recent events for
  older snapshots) to clear legacy A ley-only flor flags without clearing actual
  flor announcements or changing the cards, turn, or pending wager.
- **Con ley, envido**, **Sin ley, envido**, and **Sin ley, no quiero** require
  an earlier **A ley** in that hand, including one spoken by a CPU. An initial
  **A ley, envido** also establishes ley for the responding team's choices.
  Sin ley refusal applies only to envido wagers, never truco or flor contests.
  Clients receive only the canonical ley-envido calls; legacy ordinary-envido
  Sin ley payloads must satisfy the same prior-A-ley requirement.

Seats increase counter-clockwise. The dealer's right neighbor is mano; the left
neighbor cuts and receives the first dealt card. Four-player teammates sit opposite.
The rules package receives shuffled decks and timeout selections from the host;
it never reads a clock or generates randomness.

This repository is under implementation. These decisions do not imply that P0
has been completed or that all game transitions have been validated.
