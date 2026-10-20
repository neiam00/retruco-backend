# Automatic VPS deployment

The workflow in `.github/workflows/deploy-vps.yml` deploys this **backend repository**
on pushes to **`master`**, and supports **Actions → Deploy backend to VPS → Run workflow**.
Manual runs also require `master`. The frontend keeps its separate Cloudflare Pages deployment.

## One-time prerequisites

Complete the VPS deployment guide first: Docker/Compose, working DNS and TLS for
`origin.retru.co.ve`, the database, API, NGINX, and Pages must already be configured.
The workflow updates an installed application; it does not provision the server,
issue the initial certificate, or write application secrets.

The expected server paths are:

- `/opt/retruco/backend/.env.production`
- `/opt/retruco/backend/compose.vps.yaml`
- `/opt/retruco/backend/deploy/nginx.vps.conf.template`
- `/usr/local/sbin/retruco-compose`

The helper must use the `retruco` Compose project and the production files from
that directory, as in the VPS guide. If voice is enabled, retain `compose.voice.yaml`
in the helper; deployment detects and updates its `turn` service.

The SSH user must be able to run Docker, execute the helper, and write
`/opt/retruco/backend` and `/opt/retruco/backups`. The fresh-VPS guide uses `root`,
which is the default here. The server needs `bash`, `rsync`, `tar`, `curl`, and
`flock` (provided by `util-linux` on Ubuntu). Keep inbound SSH available to the
runner, as well as the existing web and optional voice firewall rules.

## GitHub secrets

In the **backend GitHub repository**, create an environment named **`production`**
under **Settings → Environments**. Add these environment secrets there
(repository Actions secrets also work):

| Secret | Required? | Value |
| --- | --- | --- |
| `VPS_HOST` | Yes | Droplet IPv4 address or `origin.retru.co.ve`, without a scheme or path |
| `VPS_SSH_PRIVATE_KEY` | Yes | Complete private SSH key, including its BEGIN/END lines; dedicated unencrypted key for CI |
| `VPS_KNOWN_HOSTS` | Yes | Verified SSH known-hosts entry matching `VPS_HOST` and the SSH port |
| `VPS_USER` | No | SSH username; defaults to `root` |
| `VPS_PORT` | No | SSH port; defaults to `22` |

The workflow requires verified host keys. It does not disable host-key checking or
trust whatever key is returned during a deployment.

To create a dedicated deploy key, run on your Mac:

```sh
ssh-keygen -t ed25519 -f ~/.ssh/retruco_actions -C retruco-github-actions -N ''
```

Append the contents of `~/.ssh/retruco_actions.pub` to the deploy user's
`~/.ssh/authorized_keys` on the VPS, preserving existing entries.
Use directory mode 700 and file mode 600 there.
Copy the **private** file `~/.ssh/retruco_actions` into `VPS_SSH_PRIVATE_KEY`.
Do not commit either the private key or application secrets.

Collect the host key using the same host/IP you will put in `VPS_HOST`:

```sh
ssh-keyscan -t ed25519 origin.retru.co.ve > /tmp/retruco_known_hosts
ssh-keygen -lf /tmp/retruco_known_hosts
```

Compare that fingerprint with this command run through the trusted DigitalOcean
console on the VPS:

```sh
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

Only when they match, copy `/tmp/retruco_known_hosts` into `VPS_KNOWN_HOSTS`.
For a custom SSH port, add `-p PORT` to `ssh-keyscan`; its output should contain
`[host]:PORT`. A server reset changes its host key, so verify and update the secret
again after a future reset.

`POSTGRES_PASSWORD`, `API_PROXY_SECRET`, and optional TURN secrets stay in the
VPS's `.env.production`. No B2, Cloudflare API, or container-registry token is
needed. Pages must already have `BACKEND_ORIGIN=https://origin.retru.co.ve` and
the matching `API_PROXY_SECRET`.

## What a deployment does

1. Checks the deployment script, runs its failure-path tests, builds the Docker
   builder stage, and runs `go test ./...` and `go vet ./...` with that toolchain.
   PostgreSQL integration tests remain opt-in; this job does not supply
   `DATABASE_TEST_URL`.
2. Archives the exact pushed commit and uploads it over verified SSH.
3. Takes a server-side lock and synchronizes application source, removing obsolete
   source files. Preserves `.env*`, `.git`, `.agents`, `.agent`, `compose.vps.yaml`,
   `deploy/nginx.vps.conf.template`, `.deployed-revision`, `bin`, and `coverage.out`.
4. Builds the API and migration images while the old API keeps running.
5. Stops the API, saves a PostgreSQL backup under `/opt/retruco/backups`, and
   applies migrations. This creates a maintenance window and disconnects games.
6. Starts the new API, checks it locally using the existing proxy secret,
   updates TURN if configured, and recreates NGINX.
7. Checks `https://retru.co.ve/api/health` and records the deployed SHA in
   `/opt/retruco/backend/.deployed-revision`.

Treat the backend directory as managed application source. Put other server-only
files outside it unless they are explicitly protected above. The synchronization
does not delete database volumes, certificates, the helper, or the backup directory.
Copy backups off the VPS and manage their retention; deployment does not prune them.

Deployments are serialized and an active deployment is not cancelled by a new push.
GitHub may replace an older queued run with a newer push. A server-side lock also
rejects overlapping calls to the deployment script.

## Failures and recovery

- Build failure: the old API continues running; correct the source and push again.
- Backup failure: the script restarts the existing API container without changing
  the schema.
- Migration failure: the workflow fails and leaves the API stopped. Review the
  migration and backup, then fix forward or restore deliberately.
- Startup/public-health failure: the workflow fails. Inspect API, proxy, and Pages
  configuration before declaring the deployment healthy.

There is **no automatic database or image rollback after migrations**. An old
image may be incompatible with the new schema. Do not retry with an old commit
without reviewing migration compatibility.

On the VPS, use `retruco-compose ps -a` and
`retruco-compose logs --tail=100 api proxy` to investigate. Health checks do not
replace a browser check of login, gameplay, and WebSockets.

For a manually customized installation, `deploy/deploy-vps.sh` accepts environment
overrides `RETRUCO_DEPLOY_DIR`, `RETRUCO_COMPOSE`, and `RETRUCO_HEALTH_URL`.
The workflow uses the defaults from the VPS guide.

After adding secrets, commit and push the workflow, script, and tests to `master`.
Watch the first run under **Actions**. No secrets have been configured, workflow
pushed, or live deployment executed as part of creating these files.

Local validation passed: Actions syntax (`actionlint`), shell syntax, six deployment
tests, `go test ./...`, and `go vet ./...`. The deployment tests use real file
synchronization with simulated external services. The Docker-based CI step was
not run locally because the local Docker socket was inaccessible; the first
Actions run will exercise that container build and test gate.

References: [GitHub secrets](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets),
[deployment concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).
