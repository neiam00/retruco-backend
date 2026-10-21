# Retruco backend

Authoritative Venezuelan Truco rules and multiplayer API. The frontend lives in
[retruco-webapp](https://github.com/neiam00/retruco-webapp). P0 supports two or four
players, accounts and guests, public/private tables, spectators, chat, señas,
server deadlines, and persisted matches. Voice and CPU games are deferred to P1.

## Local development

Requirements: Go 1.27.2+, Docker with Compose, and PostgreSQL 18.

1. Copy `.env.example` to `.env` and choose a local URL-safe database password.
   Keep the password in `DATABASE_URL` consistent with `POSTGRES_PASSWORD`.
2. Run `docker compose up -d db`.
3. Run `APP_ENV=development go run ./cmd/migrate up`.
4. Run `make dev` (equivalent to `APP_ENV=development go run ./cmd/server`).
5. Start the frontend at `http://127.0.0.1:5173`. Its Vite proxy forwards `/api`
   and WebSockets to `127.0.0.1:8080`.

The server and migration command load only `.env` from the current working
directory when `APP_ENV=development` is set in the process environment.
Existing environment variables take precedence, including empty values.
A missing `.env` is allowed; an unreadable or malformed file stops startup.
`.env.example` is never loaded. Without `APP_ENV=development`, configuration
comes entirely from the process environment.

Alternatively, `docker compose up --build` runs the database, migrations, and API
together. PostgreSQL and the API bind only to loopback on the host. Never use
`docker compose down -v` on a database whose contents you need to keep.

## Checks

```sh
go test -race ./...
go vet ./...
golangci-lint run ./...
GOOS=js GOARCH=wasm go test -exec="$(go env GOROOT)/lib/wasm/go_js_wasm_exec" ./rules
```

The PostgreSQL integration test is opt-in and creates real test users and
matches. Point `DATABASE_TEST_URL` at a **disposable**, migrated database, then
run `go test -race ./internal/api -run TestAccountAndMatchPersistence -count=1`.
Without that variable the integration test reports a skip, not a database pass.
The shared rules tests run under both native Go and the Go JavaScript/WASM test
runner; Node must be installed for the latter.

## Architecture and privacy

Players may send `{"type":"burn","card":1}` on their card turn in the second
or third round. The selected card is consumed, stays face down in every public
table view, and cannot win the round. During mandatory pegado, `{"type":"burn"}`
burns both remaining cards. Normal flor declarations still apply through the
`declaration` field, including `flor_flor` for the two-card burn. Timeouts keep
their existing declaration behavior.

CPU rooms accept `options.cpuSeats` when creating a match: `[1]` for 1v1, or
any distinct subset of seats 1–3 for 2v2. Seat 0 belongs to the creator; all
unselected seats remain available to humans. Teams are alternating seats.
Listings include `cpuPlayers`; snapshots expose `cpuController` (a seat, or
`-1`) and `cpuPaused`. These choices survive restart with the match snapshot.

Each game WebSocket receives a separate `cpu` envelope. Only the designated
human connection receives `controlled: true` and the next CPU's authorized
`view`, `seat`, and protocol `version`. Submit its chosen command as an ordinary
`input` with `cpuSeat`, `revision`, and a unique `id`. The server binds authority
to the connection and validates the move against the authoritative engine.
Spectators and other tabs never receive that work. On disconnect, control moves
to another connected human; rooms without connected humans wait without
advancing their deadline. Voice connections do not count as CPU controllers.

The `cpu` package is a bounded heuristic policy over a single authorized view.
It cannot accept the secret `rules.State`. `make browser-engine` builds
`bin/browser-engine/cpu.wasm` and its matching `wasm_exec.js` runtime. They must
be shipped together by the frontend and loaded in one dedicated Worker only
when that connection controls CPU players. The Worker calls the exported
`retrucoCPUDecide` with a JSON `cpu.Request` and receives a JSON `cpu.Response`.
Both protocol and rules versions are checked before choosing a move.
`make test-wasm` runs the shared rules and CPU simulations with Go's Node/WASM
runner; the same policy tests run natively under `go test`.
The initial Go 1.27.2 artifact is 4,419,352 bytes (1,231,593 bytes with gzip);
its matching JavaScript runtime is 16,992 bytes (4,356 bytes with gzip).
Browser startup, memory, and decision latency still need measurement when the
frontend Worker is integrated; native/WASM test times are not mobile benchmarks.

- `rules` is deterministic Go without network, clock, database, or random APIs.
  Hosts supply shuffled decks and random timeout choices. See
  [RULES-DECISIONS.md](RULES-DECISIONS.md) for clarified game rules.
- `internal/game` owns matches, serializes commands, validates revisions and
  command IDs, persists accepted moves, and sends perspective-filtered snapshots.
  Hidden hands and concealed pegado cards are filtered before JSON transmission.
- `internal/api` handles username/password login, HttpOnly cookies, exact-origin
  checks, rate limits, and WebSocket connections. Passwords use bcrypt; only
  session-token hashes are stored.
- `internal/store` persists users, sessions and match snapshots in PostgreSQL.
  Chat remains in snapshots by the user's explicit override; application logs
  must not include chat text, passwords, cookies, or proxy secrets.

Run **one API instance**. Match ownership and WebSocket subscriptions are local
to that process; PostgreSQL supplies durability across restarts. Redis would not
solve cross-instance ownership by itself and is unnecessary for this P0 topology.
Do not scale replicas until match routing and distributed ownership are designed.

Unstarted tables are deleted when their last connected human leaves explicitly. When all
seated humans disconnect, there is a 10-second grace period for a page refresh
or brief connection loss. Another connected tab keeps the room alive;
spectators and CPU seats do not. Newly created and restored waiting rooms also
have 10 seconds for a player to connect. Deletion removes the stored waiting
room and its memberships, so it cannot be joined or restored after a restart.
`GET /api/matches/events` streams public room-list updates to open browsers.
Started matches retain their normal reconnect and forfeit behavior.
Completed matches do not consume the active-table capacity.
Accepted envido reveals comparison values at settlement; concealed cards stay
concealed. These behaviors follow the direct user clarifications.

Public cantos, declarations, and replies are retained for the entire match,
including their hand and calling seat. Reconnecting clients receive this history
grouped by hand in the interface. Older saved matches can recover only the recent
events they already retained; discarded events from earlier hands cannot be rebuilt.

## Production: Cloudflare Pages → NGINX → Go

For automatic deployment on pushes to `master`, see
[GitHub Actions VPS deployment](deploy/GITHUB-ACTIONS.md).

For voice signaling and the self-hosted TURN relay, see [voice deployment](deploy/VOICE.md).

The public frontend and API use `https://retru.co.ve` and
`https://retru.co.ve/api`. The frontend's server hook forwards `/api/*` to a
separate backend origin. That origin needs DNS and a valid TLS certificate.
It is an operator-selected hostname, not the public browser API URL.

1. Copy `.env.production.example` to `.env.production` on the backend host.
   Set a strong URL-safe database password, the backend hostname, certificate
   directory, and a shared random `API_PROXY_SECRET` of at least 32 bytes.
2. Put `fullchain.pem` and `privkey.pem` in `TLS_CERT_DIR`. Arrange certificate
   renewal outside this application and reload NGINX after renewal.
3. Run:

   ```sh
   docker compose --env-file .env.production -f compose.yaml -f compose.production.yaml config --quiet
   docker compose --env-file .env.production -f compose.yaml -f compose.production.yaml up -d --build
   ```

4. In Cloudflare Pages, set server-only runtime `BACKEND_ORIGIN` to
   `https://<your-backend-host>` and `API_PROXY_SECRET` to the identical value.
   Keep public `API_ORIGIN` empty. Production allows only `https://retru.co.ve`
   as a browser origin; preview domains need an explicitly configured allowance.
5. Verify `https://retru.co.ve/api/health`, then register/login, create a private
   table, join from a second browser, play, reconnect, and check match history.
   A direct backend request without the shared secret must return 403.

NGINX forwards WebSocket upgrades and disables buffering and access logs. Go
sends heartbeats more frequently than the proxy's idle timeout. The edge replaces
visitor-IP forwarding headers, and Go trusts them only after authenticating the
shared secret. Set `COOKIE_SECURE=true` in every HTTPS deployment. Keep port 8080
and the database private; expose only NGINX's TLS endpoint.

Back up PostgreSQL before upgrades. Apply migrations before starting the new API;
restore the backup if a schema rollback would lose data. Match snapshots include
hidden hands, so treat database backups as private. No hosting, DNS, certificate,
or secret is provisioned by these files.

## B2 card publishing

Use an existing bucket and an application key restricted to writing its card
prefix. Export `B2_APPLICATION_KEY_ID`, `B2_APPLICATION_KEY`, and `B2_BUCKET_NAME`
only in the operator's shell, then run:

```sh
go run ./cmd/upload-assets ../frontend/static/cards
```

Run `pnpm cards:build` in the frontend after editing its `artwork/cards/*.svg` sources.
The SDK uploader expects the 40 generated WebP faces and one back, writes
`cards/*`, and sets `image/webp`. Upload these files before deploying a frontend
that references their content-hashed filenames. Older object versions can remain
available for clients still using the previous deployment.
Keep the upstream attribution accessible alongside the deployed app. Configure
the frontend's public `CARD_ASSET_ORIGIN` with the bucket/CDN origin and rebuild.
Leave it empty to serve the bundled assets. Never place B2 keys in the frontend.
