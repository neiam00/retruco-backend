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
3. Start PostgreSQL, apply the migrations, and start the API and proxy. Keep
   database volumes and maintain backups outside the server.
4. Configure the frontend's `BACKEND_ORIGIN` and `API_PROXY_SECRET`, using the
   same secret as the backend. Verify the public `/api/health` endpoint, login,
   gameplay, WebSocket connections, and voice from separate devices.
5. Set up certificate renewal and verify that NGINX reloads after renewal. The
   `deploy/renew-certificate.sh` hook accepts the certificate lineage as its
   first argument and the Compose helper as its second. With no lineage
   argument, it reads `BACKEND_HOST` from the running proxy and handles only
   `/etc/letsencrypt/live/<BACKEND_HOST>`.

For upgrades from an installation that kept allowed origins in Compose source,
the deployment script copies the running API's `ALLOWED_ORIGINS` into the private
environment file before replacing source. Existing configured values are kept.
Unless `RETRUCO_HEALTH_URL` is set, the public health check uses the running
API's first allowed origin followed by `/api/health`. Explicitly override this
URL when changing domains. See [automated deployment](deploy/GITHUB-ACTIONS.md),
[voice configuration](deploy/VOICE.md), and [capacity and migration notes](deploy/CAPACITY.md).

Run only one API instance: match ownership and subscriptions are held in memory.
Plan a maintenance window for migrations and deployment, and test backups and
restores before relying on them.
