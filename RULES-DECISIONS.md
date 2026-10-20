# Implementation decisions

The source of game requirements is `../truco-spec.md`. The following direct user
clarifications override that document where they differ (2026-10-19):

- Use manual constructor injection for service wiring.
- Ley envido wagers bypass the opposing ordinary-flor block.
- A player with ordinary flor must manually say **Flor flor** when playing pegado.
- Ordinary envido cancels only its caller's ordinary flor. Receta remains protected.
- An opposing valid ordinary flor still blocks ordinary envido with **A las flores no**.
- **A ley, envido** preserves flor and scores envido before flor. If envido reaches
  the winning score, the match ends before flor or truco is scored.
- Accepting envido closes all envido/flor-contest betting for that hand.
- An accepted envido wager replaces receta's automatic envido point whether its
  team wins or loses the wager.
- Shared ordinary-envido/truco response timeouts preserve ordinary flores.
  Flor-contest timeouts still invalidate the whole team's ordinary flores.
- Chat may remain in saved match snapshots (direct user override on 2026-10-19).
  Do not write chat text to application logs.
- The public API remains under `https://retru.co.ve/api`; the frontend edge
  forwards it to the configured NGINX backend origin.
- Accepted envido reveals each player's comparison value at settlement, without
  revealing otherwise concealed cards.
- Unstarted tables expire after 24 hours without activity.
- Saying "Con mi flor" while calling or answering a canto counts as declaring
  that valid ordinary flor for later flor-contest eligibility.

Seats increase counter-clockwise. The dealer's right neighbor is mano; the left
neighbor cuts and receives the first dealt card. Four-player teammates sit opposite.
The rules package receives shuffled decks and timeout selections from the host;
it never reads a clock or generates randomness.

This repository is under implementation. These decisions do not imply that P0
has been completed or that all game transitions have been validated.
