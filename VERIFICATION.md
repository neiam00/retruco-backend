# P0 verification map

Scope: `truco-spec.md` P0, with direct user clarifications recorded in
`RULES-DECISIONS.md`. Voice chat and local CPU opponents remain P1 and are not
part of this implementation. Deployment credentials are operator-configurable;
the verification below is local, not a claim of a public production rollout.

## Rules and state

| Requirement | Implementation and evidence |
| --- | --- |
| Forty-card deck, vira, perico/perica substitutions, ranking and envido values | `rules/card.go`; `card_test.go` covers deck uniqueness, substitutions, best pairs, wildcard cards, receta and invalid cards. |
| Random dealer/shuffle, chosen cut, bottom dealing, preview/pass and rotation | `internal/game/host.go` uses `crypto/rand`; `rules/state.go` handles deterministic cut/deal/rotation. The independent fixture in `engine_test.go` constructs the specified draw order for two/four players and verifies dealt hands. |
| Turn/voice ownership, paused play, truco levels and envido interruption | `wager.go` and `play.go`; `TestTrucoRaisesAndInterruption`, legal-state simulations, and host deadline tests. |
| Cumulative envido, caps, falta value/terminal state and refusal payouts | `wager.go`; cap/falta tests and multi-raise/refusal transition tests. |
| Ordinary flor declarations, double flor/a ley, receta protection and explicit omissions | `declareCards`/`declareCanto`; declaration, pegado, double-flor, receta and shared-timeout tests. User clarifications are reflected in the Spanish rulebook. |
| Flor contests, team-wide invalidation, ties and no-valid-flor settlement | `settlement_test.go` exercises refusal, tied contest mano priority, all flores burned and prefixed-canto eligibility. |
| Flor/envido/truco priority, a ley exception and vale partida | `score.go`; ley-before-flor, receta and simultaneous-flor-before-vale-partida tests. |
| Empardes, mandatory/voluntary pegado, concealed bottoms and top penalties | `play.go`; mandatory top-first resolution, timeout burning, voluntary loss, unnecessary-bottom concealment and visible-third-tie tests. The teammate tie exception is implemented in `teammateStackTie`. |
| Deterministic shared engine/native-WASM equivalence | The same `rules` fixture/test suite passes natively and through Go's JS/WASM test runner. No browser/network/database/clock/random imports in `rules`. |

## Online host and storage

| Requirement | Implementation and evidence |
| --- | --- |
| One authoritative command owner, command IDs and revisions | Host mutex, `Recent` IDs, revision validation and commit-before-publish; duplicate/stale-action and failed-persistence tests. Deployment is explicitly one API instance. |
| Private hands and spectator perspectives | `rules.ViewFor` filters hands and concealed cards before JSON; host tests compare authorized views and the browser suite switches spectator perspectives. |
| Signals with independent opponent rolls, bluffing and spectator reuse | Host accepts any listed signal, uses a fresh `crypto/rand` draw per opponent, and reuses that visibility mask for spectators. Signal/view tests verify authorized recipients and absence of recipient masks on the wire. |
| Shared 30-second deadlines and fresh clocks after interruptions | Host deadline scheduling and explicit rules timeouts; first-card/mandatory-card timeout tests, duplicate/reconnect clock tests, and restored short-window interruption/resumption tests. |
| Disconnect, reconnect and explicit leave | Server snapshots retain the authoritative deadline; rules leave forfeits the team. Live browser tests cover reconnect and forfeit. |
| Public declarations/cantos | Safe, bounded public events, persisted with snapshots; host retry/restore tests and browser tests verify another player sees Flor. |
| Accounts and history | bcrypt passwords, hashed session tokens and PostgreSQL snapshots; real PostgreSQL integration tests plus UI registration/login/history in a fresh browser. |
| Historical-code integrity and inactive lobbies | Atomic code reservation and update-only saves; PostgreSQL regression test fails with old overwrite behavior. Host expiry/restart test covers the 24-hour inactive-lobby policy. |
| Chat | WebSocket chat, bounded history, Svelte text escaping and no chat-body logging. Chat persistence is explicitly permitted by the user. Browser tests verify delivery/reconnect. |

## Frontend and deployment

| Requirement | Implementation and evidence |
| --- | --- |
| Requested frontend stack and readable code | Pinned manifests, shadcn controls, DiceBear, Lucide, Tailwind, Svelte components/SVG; ESLint, Prettier and Svelte check pass. |
| Responsive Yuno-inspired gameplay, motion, keyboard/touch and rulebook | `DESIGN.md`, fullscreen felt, perimeter badges, fanned hand, transitions/reduced motion and 44px actions. Desktop/mobile screenshots and live tests cover controls, keyboard play, hierarchy and rulebook. |
| Public/private creation, listing, links and ABCD-1234 joining | Browser acceptance tests cover public browse/join, private shared links and case-normalized code joining. |
| Spanish text, static-page SEO and security headers | Spanish pages/error page, prerendered lobby/rules/credits, metadata/robots/sitemap, CSP, `_headers` and runtime security headers. Built Pages runtime verified static headers and Spanish 404. |
| Same-origin `/api`, CORS/CSRF and secure cookies | Exact-origin checks, HttpOnly/Secure cookies, authenticated edge forwarding and visitor rate limits. Proxy tests and real local Pages session/WebSocket/reconnect test pass. |
| Go/PostgreSQL/goose/Docker/NGINX | Docker image builds and runs as non-root; image migrations and health check passed against a separate disposable PostgreSQL database. Production Compose validates and NGINX template passes `nginx -t`. |
| B2 and Analytics configuration | SDK-based explicit asset uploader; private key configuration; optional public card CDN and analytics IDs. No credentials embedded. Live B2 publishing and Analytics collection await deployment configuration. |
| Separate repositories and meaningful commits/pushes | Backend and frontend are separate Git repositories; completed batches are committed and pushed to their configured GitHub remotes. |

See each repository's README for exact check and deployment commands. The
PostgreSQL and built-edge tests require their documented local services; a skipped
optional test is not counted as proof that those services were exercised.
