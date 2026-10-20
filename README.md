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
3. Export the values from `.env` into your shell. The Go commands do not read
   dotenv files automatically.
4. Run `go run ./cmd/migrate up`, then `go run ./cmd/server`.
5. Start the frontend at `http://127.0.0.1:5173`. Its Vite proxy forwards `/api`
   and WebSockets to `127.0.0.1:8080`.

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

## Production: Cloudflare Pages → NGINX → Go

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

The SDK uploader expects the 40-card SVG deck and one back, writes `cards/*`,
and preserves SVG content types. It may replace an existing object version.
Keep the upstream attribution accessible alongside the deployed app. Configure
the frontend's public `CARD_ASSET_ORIGIN` with the bucket/CDN origin and rebuild.
Leave it empty to serve the bundled assets. Never place B2 keys in the frontend.
