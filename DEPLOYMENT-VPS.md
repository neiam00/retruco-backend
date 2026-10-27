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
   HTTPS. `compose.production.yaml` hard-codes the origin, so override or edit
   that value for your installation. Consult the NGINX templates when adapting
   the configuration.
3. Start PostgreSQL, apply the migrations, and start the API and proxy. Keep
   database volumes and maintain backups outside the server.
4. Configure the frontend's `BACKEND_ORIGIN` and `API_PROXY_SECRET`, using the
   same secret as the backend. Verify the public `/api/health` endpoint, login,
   gameplay, WebSocket connections, and voice from separate devices.
5. Set up certificate renewal and verify that NGINX reloads after renewal. The
   `deploy/renew-certificate.sh` hook accepts the certificate lineage as its
   first argument and the Compose helper as its second; its defaults are tied
   to an existing installation and must be reviewed.

The production template and deployment scripts contain installation-specific
hostnames and paths. In particular, set `RETRUCO_HEALTH_URL` for the deployment
script and review the GitHub Actions workflow's environment URL and remote
invocation before enabling automated deployments. See [automated deployment](deploy/GITHUB-ACTIONS.md),
[voice configuration](deploy/VOICE.md), and [capacity and migration notes](deploy/CAPACITY.md).

Run only one API instance: match ownership and subscriptions are held in memory.
Plan a maintenance window for migrations and deployment, and test backups and
restores before relying on them.
