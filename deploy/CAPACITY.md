# Capacity changes and verification

The backend remains one Go API with PostgreSQL durability. Rule processing,
hidden-card permissions, CPU-controller ownership, accepted-request deduplication
and commit-before-publication ordering are unchanged. No extra coordination
service or nondurable write-behind cache is introduced.

## Persistence and migration

Migration 00002 moves `Chat` and `CantoHistory` out of the frequently updated
`matches.payload` JSONB into independent JSONB columns. Ordinary moves omit
unchanged columns from their single atomic UPDATE. Membership additions and
state/history changes still commit in one transaction. Request receipts stay
durable in the compact payload. All users, hashes, live sessions, memberships,
scores and retained history are preserved. Expired sessions alone are removed
by bounded hourly maintenance (at most 500 per sweep).

The migration reconstructs and compares every original JSON document inside the
same transaction. A mismatch aborts both schema and data changes. Missing keys,
JSON null and empty arrays remain distinguishable. Recovery reconstructs the
original document shape before the existing room-restoration code runs.

Production rollout uses the existing deployment sequence: notify/pause players,
stop the old API, take a server-side pg_dump backup, migrate, then start the new
API. Do not run old and new writers concurrently across this migration.

Rollback requires stopping the API and taking a fresh backup, using the **new**
migration binary to run `down-to 1`, then starting the old image. Down merges
the latest columns back into payload, so post-upgrade chat/history survives.
Do not start an old image against stale compact payloads. Existing snapshots
remain sufficient; there is no action journal or invented historical replay.

## Runtime and transport

Voice authorization now reads room permission metadata instead of constructing
game snapshots. Its existing session checks still run at the same boundaries.
The lobby prepares immutable JSON once per committed listing change. Game
transport v2 prepares authorized player/spectator views once per publication,
with connection-specific CPU work and fresh server time outside the cache.

Clients opt into `?transport=2`. Every connection starts with a complete baseline
and a new stream epoch. Subsequent updates carry complete compact game views and
incremental chat/events/canto history. Coalesced notifications send every retained
unseen entry; a lost history window resets that list. A sequence gap requests a
full resync. Perspective changes force a fresh baseline. The legacy full-snapshot
protocol remains supported for old clients and rolling frontend deployments.

Every visible room notification invalidates cached views, including controller
changes and deployment notices that do not advance rules revisions. Player and
spectator perspectives have separate cache entries. Slow writers retain the
existing bounded notification/input queues and five-second write deadline.

Global password verification concurrency is one, retaining bcrypt cost 12 and
existing per-client limits. Excess work returns 503 and Retry-After rather than
accumulating CPU work. This bounds simultaneous work; it does not reserve CPU
against a sustained stream of otherwise admitted authentication requests.

## Production resource settings

The production overlay mounts `nginx-main.conf`, sets 8192 NGINX worker
connections and 16384 open-file limits for NGINX and API containers. Proxied
WebSockets/SSE each use a client and upstream socket; voice adds another socket.
These limits remove an artificial ceiling, not establish a tested player count.

`GOMEMLIMIT` defaults to 512MiB as a soft Go runtime memory target. It is not a
total process-RSS cap; leave RAM for PostgreSQL, kernel/network buffers, Docker,
NGINX and TURN. Override it after measuring heap/GC pressure. The PostgreSQL pool
remains 10 connections, with durable commits enabled. TURN relay capacity and
bandwidth still need independent measurement.

## Reproduce verification

Use a disposable PostgreSQL database, never production. Apply all migrations
with `DATABASE_URL` and set `DATABASE_TEST_URL` to the same disposable database:

```sh
go run ./cmd/migrate up
go test -race ./...
go vet ./...
python3 -m unittest discover -s deploy/tests -p 'test_*.py' -v

GOMAXPROCS=1 go test ./internal/store -run '^$' -bench BenchmarkSaveMatchOrdinary -benchmem -count=6
GOMAXPROCS=1 go test ./internal/game -run '^$' -bench 'Benchmark(PersistEncoding|VoiceAccess|ListingPublication)$' -benchmem -count=6
GOMAXPROCS=1 go test ./internal/api -run '^$' -bench BenchmarkStreamHistory -benchmem -count=6
```

Run benchmarks serially without other traffic. WAL measurement covers the whole
PostgreSQL instance, so other writes contaminate WAL-B/op. Database benchmarks
compare the legacy durable UPDATE with the compact durable UPDATE using actual
rules state, 100 retained chat messages, 300 canto records and 257 request receipts.
They exclude network fanout and should not be multiplied into player estimates.

Local measurements on 2026-10-26 (Apple M5, Go 1.27.3, GOMAXPROCS=1;
six persistence samples and three stream samples) gave these medians:

| Operation | Previous path | Optimized path |
| --- | ---: | ---: |
| Durable history-heavy database UPDATE | 815 microseconds | 372 microseconds |
| WAL bytes per ordinary action | about 19,094 | about 4,280 |
| Persistence JSON encoding | 194.6 microseconds | 54.8 microseconds |
| Encoder allocated bytes | 92,201 | 12,369 |
| Four-player wire message with unchanged retained history | 20,657 bytes | 3,384 bytes |

The stream fixture uses an actual four-player rules view with 100 chat and 100
canto records. An initial snapshot still carries full history. Separate local
benchmarks showed zero allocations for metadata-only voice authorization and
roughly one encoding per lobby publication instead of one per subscriber.
These are local component measurements, not measurements of the VPS or an
assertion that all traffic enjoys the same reduction.

The tests cover migration Up/Down and verification failure rollback, atomic
membership writes, restart restoration, unchanged data after failed persistence,
cache audience isolation, metadata invalidation, real old/new WebSocket gameplay,
resynchronization, permission changes and authentication overload recovery.

Load-test sustained human games separately at 100, 200, 500 and 1000 players,
including long histories, spectators, bot rooms, voice and reconnect bursts.
Measure action-to-update percentiles, process/host CPU and memory, DB pool waits,
WAL and network bytes. No concurrency guarantee follows from microbenchmarks.
