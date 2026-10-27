# Retruco backend | [retru.co.ve](https://retru.co.ve)

The server and rules engine for Venezuelan Truco. It supports two- and
four-player matches, guests and accounts, public and private tables,
spectators, chat, voice signaling, and CPU opponents. The
[web client](https://github.com/neiam00/retruco-webapp) provides the interface.

## Getting started

Requirements: Go 1.27.3 or later, Docker with Compose, and PostgreSQL 18.

1. Copy `.env.example` to `.env`. Choose a local URL-safe password and use it in
   both `POSTGRES_PASSWORD` and `DATABASE_URL`.
2. Start PostgreSQL: `docker compose up -d db`.
3. Apply migrations: `APP_ENV=development go run ./cmd/migrate up`.
4. Start the API: `make dev`.
5. Start the frontend at `http://127.0.0.1:5173`; its development proxy forwards
   `/api` and WebSocket connections to `127.0.0.1:8080`.

The server and migration command load `.env` from the working directory only
when `APP_ENV=development` is set. Process environment variables take
precedence. Alternatively, `docker compose up --build` starts the database,
migrations, and API together. Keep database volumes when you need their data.

Login and registration require backend Turnstile keys; guests can play without
them. Voice can use direct browser connections locally; configure the optional
TURN credentials for production relay. See `.env.example` and the
[voice guide](deploy/VOICE.md).

## Checks

```sh
go test -race ./...
go vet ./...
golangci-lint run ./...
make test-wasm
```

The WASM tests require Node. PostgreSQL integration tests require a **disposable**
migrated database supplied in `DATABASE_TEST_URL`; otherwise they are skipped.

## Architecture

- `rules` contains the deterministic game engine; the host supplies shuffled
  decks and timeout choices.
- `internal/game` manages matches, persistence, deadlines, and views filtered
  for each player. CPU moves are validated by the server.
- `internal/api` provides HTTP and WebSocket endpoints, sessions, origin checks,
  and voice signaling.
- `internal/store` persists users, sessions, and match snapshots in PostgreSQL.
- `cpu` contains the policy compiled to WASM for CPU-controlled seats. Run
  `make browser-engine` to build the module and its matching Go runtime together.

Run a single API instance: match ownership and WebSocket subscriptions live in
that process, while PostgreSQL provides durability across restarts. Scaling to
multiple instances requires distributed ownership and routing.

## Deployment

See the [VPS deployment guide](DEPLOYMENT-VPS.md),
[automated deployment notes](deploy/GITHUB-ACTIONS.md), and
[capacity and migration notes](deploy/CAPACITY.md). Review deployment-specific
paths and configuration before using the supplied scripts. Production Compose
reads frontend origins from `ALLOWED_ORIGINS` in the private environment file;
the backend hostname is configured with `BACKEND_HOST`.

## License

The project code is available under the [MIT License](LICENSE). Dependencies
are subject to their respective licenses.
