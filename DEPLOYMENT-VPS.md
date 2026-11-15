# VPS deployment

This repository includes a production Compose configuration for a single API
instance, PostgreSQL, and NGINX. The frontend can run on Cloudflare Pages,
forwarding `/api` and WebSocket traffic to the backend over HTTPS. Adapt the
configuration to your own domain and hosting environment before deployment.

1. Provision a server with Docker and Compose. Set up DNS and TLS for a backend
   hostname, and restrict access to the origin according to your proxy and
   administrative access requirements.
2. Copy `.env.production.example` to a private environment file and configure
   the database password, backend hostname, certificate directory, proxy secret,
   and any voice relay credentials. Configure Turnstile keys for accounts.
   Set `ALLOWED_ORIGINS` for the frontend origin and enable secure cookies for
   HTTPS. Production Compose requires `ALLOWED_ORIGINS` from the environment
   file; use comma-separated exact origins with no paths or trailing slashes.
   Consult the NGINX templates when adapting the configuration.
3. On a first installation, build the backend for the VPS platform with
   `docker build -t retruco-backend:production .`.
   Production Compose removes the development build configuration and only uses
   this local image, with registry pulls disabled. Use a current Docker Compose
   release supporting `!reset` overrides.
   Start PostgreSQL, apply the migrations, and start the API and proxy. Keep
   database volumes and maintain backups outside the server.
4. Configure the frontend's `BACKEND_ORIGIN` and `API_PROXY_SECRET`, using the
   same secret as the backend. Verify the public `/api/health` endpoint, login,
   gameplay, WebSocket connections, and voice from separate devices.
5. Set up certificate renewal and verify that NGINX reloads after renewal. The
   `deploy/renew-certificate.sh` hook accepts the certificate lineage as its
   first argument and the Compose helper as its second. With no lineage
   argument, it reads `BACKEND_HOST` from the running proxy and handles only
   `/etc/letsencrypt/live/<BACKEND_HOST>`.

## Upgrading an existing server

Use the bundled upgrade script with an already running installation. Its defaults are
`/opt/retruco/backend` for the deployment directory and `/usr/local/sbin/retruco-compose`
for the host's Compose helper. The helper should select the `retruco` project, its private
`.env.production` and the base, production and host-specific Compose files. The script
preserves `compose.vps.yaml` and `deploy/nginx.vps.conf.template` in the deployment directory.
Configure the host-specific proxy and mount paths before the first upgrade.

Build from a clean checkout of the revision you tested, on a Docker builder targeting
the VPS operating system and architecture. Create the archives outside the checkout:

```sh
revision=$(git rev-parse HEAD)
git archive --format=tar.gz -o "../retruco-source-$revision.tar.gz" "$revision"
docker build --label "org.opencontainers.image.revision=$revision" \
  -t "retruco-backend:$revision" .
docker image save "retruco-backend:$revision" | gzip > "../retruco-image-$revision.tar.gz"
sha256sum "../retruco-image-$revision.tar.gz"
```

Transfer the two archives and the matching `deploy/deploy-vps.sh` to the VPS.
Run the script during a maintenance window, supplying the full commit ID and the
image digest from the build machine:

```sh
bash deploy-vps.sh "$source_archive" "$revision" "$image_archive" "$image_sha256"
```

The script verifies the image checksum, revision label and target platform. It
notifies players, stops the API, backs up PostgreSQL and runs migrations before
restarting the service. It records `.deployed-revision` after the internal and
public health checks succeed. Set `RETRUCO_HEALTH_URL` when the public health
endpoint is outside the first configured frontend origin. Voice relay settings
are documented in the [voice guide](deploy/VOICE.md).

A failed migration can leave the API stopped. Inspect the migration output and
the backup in the sibling `backups/` directory before attempting recovery.
Match ownership and WebSockets live in one API process. During a successful
restart, the client reconnects and the server restores active hands from the
last committed database snapshot.
