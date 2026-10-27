# Deploy retru.co.ve on Cloudflare Pages and a fresh DigitalOcean VPS

Use this guide from the beginning after the VPS reset. It replaces the temporary tunnel setup. The frontend stays in your existing Pages project, and the backend runs PostgreSQL, Go, and NGINX on the VPS.

| Address | Destination |
| --- | --- |
| `https://retru.co.ve` | Cloudflare Pages frontend |
| `https://retru.co.ve/api` | Pages server proxy, forwarding to the backend |
| `https://origin.retru.co.ve` | Cloudflare proxy → NGINX on your DigitalOcean droplet |

Assumptions: a fresh **Ubuntu 24.04 LTS** droplet with systemd and root access (or `sudo -i`). Docker installation is included. Check the operating system before running the Ubuntu-specific installation commands; a different distribution needs its own Docker installation procedure. These commands are instructions to run; no remote deployment or DNS changes have been performed by Codex.

The VPS reset means this is a new database unless you separately restore a backup. Existing Pages settings may still contain an old tunnel URL and secret; step 9 replaces them. For an already-running DNS-only deployment, use [the migration checklist](#migrate-an-existing-dns-only-deployment) instead of resetting secrets or the database.

## 1. Connect the domain to Cloudflare and configure DNS

Replace these placeholders throughout:

| Placeholder | Meaning |
| --- | --- |
| `DROPLET_IP` | Your droplet's public IPv4 address |
| `you@example.com` | Your certificate renewal contact email |
| `ADMIN_IP/32` | Your trusted administrative public IPv4 address in CIDR notation (use `/128` for a single IPv6 address) |

The hostnames in this guide are already correct: use `retru.co.ve` for the website and `origin.retru.co.ve` for the backend. Do not replace either with your IP.

1. In the Cloudflare account containing your Pages project, add `retru.co.ve` as a website/zone if it is not already there.
2. At the registrar, set the domain's nameservers to the **two exact nameservers Cloudflare assigns to your zone**. Skip this if it is already configured.
3. Wait for Cloudflare to show the zone as **Active**. Domain registration being active does not itself confirm that nameserver delegation is complete.
4. Open **Workers & Pages → the project serving `retruco.pages.dev` → Custom domains → Set up a domain**, enter `retru.co.ve`, and complete the setup. Let Pages create its apex DNS record.
5. In **retru.co.ve → DNS → Records**, create this backend record:

| Type | Name | Content | Proxy status | TTL |
| --- | --- | --- | --- | --- |
| A | `origin` | Your droplet's actual public IPv4 address | **Proxied** (orange cloud) | Auto |

**Do not point the apex `@` / `retru.co.ve` at the VPS.** The apex belongs to Pages; only `origin` points at the droplet. Do not add an AAAA record for `origin` unless IPv6 is correctly configured. Remove an obsolete `origin` address if it points to the pre-reset server's old IP.

An apex Pages custom domain requires the domain to be a Cloudflare zone with Cloudflare nameservers. Add it through the Pages custom-domain flow, not just by manually creating a DNS record. [Cloudflare custom-domain documentation](https://developers.cloudflare.com/pages/configuration/custom-domains/).

Configure these Cloudflare settings for the backend:

- Set **SSL/TLS → Overview → Full (strict)**. NGINX will serve a valid Let's Encrypt certificate obtained in step 6. Verify the Cloudflare edge certificate covering `origin.retru.co.ve` is **Active** before testing HTTPS. A fresh origin will be unavailable until its certificate and NGINX are ready.
- Add a **Cache Rule** matching `http.host eq "origin.retru.co.ve"` with **Cache eligibility: Bypass cache**. Ensure no later conflicting rule overrides it. Keep API/session responses uncached on the public `retru.co.ve/api/*` route too.
- Enable **Network → WebSockets**. Keep WAF protection, but configure challenge-producing rules so authenticated Pages-to-origin requests do not receive browser challenges. Pages cannot solve a JavaScript challenge or an interactive Access login. Use Security Events to diagnose narrowly; do not disable the entire WAF.
- Keep `API_PROXY_SECRET` enforcement. Cloudflare IP allowlisting admits Cloudflare's network, not just your Pages project. Do not strip `X-Retruco-Proxy-Secret` or `X-Retruco-Client-IP` in Transform Rules. The app already authenticates those headers; no NGINX real-IP rewrite is needed for its rate limiting.

References: [Full (strict)](https://developers.cloudflare.com/ssl/origin-configuration/ssl-modes/full-strict/), [cache rules](https://developers.cloudflare.com/cache/how-to/cache-rules/), [WebSockets](https://developers.cloudflare.com/network/websockets/), [challenge limitations](https://developers.cloudflare.com/cloudflare-challenges/challenge-types/challenge-pages/).

### Cloudflare-only HTTPS firewall

In **DigitalOcean → Networking → Firewalls**, create or edit the firewall attached to this droplet. Use these **inbound allow rules**:

| Protocol | Ports | Sources |
| --- | --- | --- |
| TCP | `22` (or your configured SSH port) | `ADMIN_IP/32`, your other trusted administrative CIDRs, and the fixed egress IP of your deployment runner if used |
| TCP | `443` | All current Cloudflare IPv4 and IPv6 CIDRs from the official lists below |

Fetch the current lists on your computer; enter each CIDR as a source for HTTPS:

```bash
curl --fail --silent --show-error https://www.cloudflare.com/ips-v4
curl --fail --silent --show-error https://www.cloudflare.com/ips-v6
```

Keep the default outbound allow rules for TCP, UDP, and ICMP to all IPv4/IPv6 destinations. The host needs outbound DNS, HTTPS for Cloudflare/ACME, package/image downloads, and optional TURN relay traffic. Do not add inbound allows for **80**, **8080**, **5432**, or **54329**. DNS-01 needs no inbound validation connection. The production Compose file publishes only NGINX port 443; API/database ports remain bound to loopback.

Remove broad inbound HTTPS/HTTP rules (`0.0.0.0/0`, `::/0`, or “All IPv4/IPv6”) from **every** firewall attached directly or through tags. Another allow rule can leave a bypass open. Do not add an explicit deny-all rule that overrides the Cloudflare allow rules; unmatched traffic is already denied. Check both address families even if the DNS record is IPv4-only. Review the official Cloudflare lists periodically and update the firewall when they change.

Keep the current SSH session open and verify a **second SSH session to `DROPLET_IP`** before removing old SSH rules. Use the droplet IP or a separate DNS-only SSH hostname for all SSH/rsync/CI operations; the orange-cloud origin hostname cannot carry SSH. Standard GitHub-hosted runners have changing egress IPs: configure a compatible runner with static egress and set the workflow's `VPS_RUNNER` repository variable to its label before restricting CI SSH access. See [the CI setup](./deploy/GITHUB-ACTIONS.md).

Use the provider firewall as the enforcement boundary for Docker's published ports. UFW alone may not filter Docker-forwarded traffic. If an additional host firewall is enabled, align its forwarding rules with this policy without blocking Cloudflare. Optional TURN requires separate public-port exceptions described below; do not apply the Cloudflare source restriction to TURN.

References: [Cloudflare IP ranges](https://www.cloudflare.com/ips/), [DigitalOcean firewall rules](https://docs.digitalocean.com/products/networking/firewalls/how-to/configure-rules/), [Docker and UFW](https://docs.docker.com/engine/network/packet-filtering-firewalls/#docker-and-ufw).

## 2. Install Docker and server prerequisites

From your computer:

```bash
ssh root@DROPLET_IP
```

On the droplet, check the OS:

```bash
cat /etc/os-release
```

For the fresh Ubuntu server, install prerequisites and configure Docker's official package repository:

```bash
apt-get update
apt-get install -y ca-certificates curl openssl rsync dnsutils nano certbot python3-certbot-dns-cloudflare
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker
docker run --rm hello-world
docker compose version
```

The test container should print its success message, and Compose should print a version. These are the [official Docker installation steps for Ubuntu](https://docs.docker.com/engine/install/ubuntu/).

Create the deployment directory and check the backend DNS and ports:

```bash
mkdir -p /opt/retruco/backend
dig +short origin.retru.co.ve A
ss -ltnp '( sport = :443 )'
```

The DNS lookup should print **Cloudflare edge addresses**, not your droplet IP. Confirm the record's underlying content is the droplet IP in the Cloudflare dashboard. An empty result means DNS is not ready. On a fresh host the port command should show only its header; resolve an existing port-443 listener before starting NGINX. DNS-01 issuance does not need either port 80 or 443 available.

## 3. Copy your backend code

In a **second terminal on your Mac**, run:

```bash
rsync -av \
  --exclude='.git' \
  --exclude='.env*' \
  --exclude='.agents' \
  --exclude='.agent' \
  --exclude='bin' \
  --exclude='coverage.out' \
  /Users/naim/Projects/retruco/backend/ \
  root@DROPLET_IP:/opt/retruco/backend/
```

This transfers your current local backend source without copying local environment files. No Go installation is needed on the droplet: the Dockerfile builds the binaries.

Return to the **droplet terminal** for the remaining server commands.

## 4. Create the production environment file

Create fresh secrets for the reset server. This command preserves the file if you run it again; changing an environment password later does not change the password inside an already-initialized PostgreSQL volume.

```bash
cd /opt/retruco/backend
umask 077
if [ ! -f .env.production ]; then
  cat > .env.production <<EOF
POSTGRES_PASSWORD=$(openssl rand -hex 32)
API_PROXY_SECRET=$(openssl rand -hex 32)
BACKEND_HOST=origin.retru.co.ve
TLS_CERT_DIR=/etc/letsencrypt
ALLOWED_ORIGINS=https://retru.co.ve
EOF
fi
chmod 600 .env.production
nano .env.production
```

Confirm the values, keeping the generated secrets. Save with Ctrl+O, Enter, and exit with Ctrl+X. If reusing an existing environment file, replace any temporary-tunnel hostname/origin settings with the values above.

| Variable | What it does |
| --- | --- |
| `POSTGRES_PASSWORD` | Password for the app's PostgreSQL user; generated as URL-safe hex |
| `API_PROXY_SECRET` | Authenticates Pages-to-backend requests; copy the exact same value into Pages later |
| `BACKEND_HOST` | `origin.retru.co.ve`, without `https://` or a path |
| `TLS_CERT_DIR` | Full Certbot directory, mounted read-only into NGINX by this runbook |
| `ALLOWED_ORIGINS` | `https://retru.co.ve`, without a trailing slash |

Compose builds `DATABASE_URL` automatically. You do not need to add it to this file. Production already sets `COOKIE_SECURE=true` and the API listen address.

## 5. Add the VPS-specific configuration

Add a final Compose override to read the allowed origin from your environment, restart PostgreSQL after a server reboot, and use the Certbot-aware NGINX template:

```bash
cat > compose.vps.yaml <<'EOF'
services:
  db:
    restart: unless-stopped
  api:
    environment:
      ALLOWED_ORIGINS: ${ALLOWED_ORIGINS:?Set the exact Pages origin}
  proxy:
    volumes:
      - ./deploy/nginx.vps.conf.template:/etc/nginx/templates/default.conf.template:ro
EOF
```

Create a server-specific NGINX template using Certbot's certificate paths:

```bash
sed \
  -e 's|/etc/nginx/tls/fullchain.pem|/etc/nginx/tls/live/${BACKEND_HOST}/fullchain.pem|' \
  -e 's|/etc/nginx/tls/privkey.pem|/etc/nginx/tls/live/${BACKEND_HOST}/privkey.pem|' \
  deploy/nginx.conf.template > deploy/nginx.vps.conf.template
```

Mounting all of `/etc/letsencrypt` preserves the relative links between Certbot's `live` and `archive` directories. Mounting only a `live/<hostname>` directory would leave those links broken inside the container. Keep the Cloudflare token in `/root/.secrets/certbot/`, outside this mount. The production template has no HTTP listener; perform HTTP-to-HTTPS redirects at Cloudflare if desired.

Create a shortcut so every command uses the same project name, environment file, and three Compose files:

```bash
cat > /usr/local/sbin/retruco-compose <<'EOF'
#!/bin/sh
set -eu
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH
exec docker compose \
  --project-name retruco \
  --project-directory /opt/retruco/backend \
  --env-file /opt/retruco/backend/.env.production \
  -f /opt/retruco/backend/compose.yaml \
  -f /opt/retruco/backend/compose.production.yaml \
  -f /opt/retruco/backend/compose.vps.yaml \
  "$@"
EOF
chmod 755 /usr/local/sbin/retruco-compose
```

Docker applies later Compose files over earlier ones. Use this shortcut for the rest of the deployment. [Docker's multiple-file documentation](https://docs.docker.com/compose/how-tos/multiple-compose-files/merge/).

Do not add `compose.quick-tunnel.yaml` or run the temporary tunnel shortcut for this deployment.

## 6. Obtain the HTTPS certificate with DNS-01

Create a Cloudflare **API token**, with **Zone → DNS → Edit** limited to the **specific zone `retru.co.ve`**. Use this token only for Certbot; do not use the Global API Key. If you restrict the token's source IP, allow the droplet's actual outbound address. Keep this token on the VPS, out of Pages, GitHub Actions, the application environment, and Docker containers.

On the droplet, create a protected credentials file without putting its token in shell history:

```bash
install -d -m 700 /root/.secrets/certbot
umask 077
touch /root/.secrets/certbot/cloudflare.ini
chmod 600 /root/.secrets/certbot/cloudflare.ini
nano /root/.secrets/certbot/cloudflare.ini
```

Enter this single line, replacing the placeholder inside the editor:

```ini
dns_cloudflare_api_token = PASTE_YOUR_ZONE_SCOPED_TOKEN
```

Verify `certbot plugins` lists `dns-cloudflare`. For a new certificate, replace `you@example.com` with your contact email and run:

```bash
certbot certonly --dns-cloudflare \
  --dns-cloudflare-credentials /root/.secrets/certbot/cloudflare.ini \
  --dns-cloudflare-propagation-seconds 60 \
  --cert-name origin.retru.co.ve \
  -d origin.retru.co.ve \
  --email you@example.com \
  --agree-tos --non-interactive
```

Certbot creates a temporary `_acme-challenge` TXT record and saves the certificate under `/etc/letsencrypt/live/origin.retru.co.ve/`. The A record stays **Proxied** throughout issuance and renewal; port 80 stays closed. Cloudflare separately manages the public edge certificates.

The certificate name must match `BACKEND_HOST` for the generated VPS template. Certbot remembers the DNS plugin and credentials path for renewal. If validation fails, check token scope/expiry, authoritative DNS, outbound connectivity, and TXT propagation; do not turn off proxying or open inbound HTTP. For an existing standalone certificate, use the migration checklist instead of creating a second lineage. [Certbot Cloudflare plugin](https://certbot-dns-cloudflare.readthedocs.io/en/stable/).

## 7. Start the backend

```bash
retruco-compose config --quiet
retruco-compose up -d --build
retruco-compose ps -a
retruco-compose logs --tail=80 migrate api proxy
```

Expected state: `db` is healthy, `migrate` exited with code **0**, and `api` and `proxy` are running. An exited migration container is normal. The API waits for successful migrations before starting.

Check the public backend without supplying the shared secret:

```bash
curl -i https://origin.retru.co.ve/api/health
```

Expect an **application JSON 403**: the backend deliberately rejects requests that do not authenticate as the Pages proxy. A Cloudflare HTML challenge/error, TLS error, connection refusal, or 502 is not the expected result. The backend `/` URL returns 404 by design. Test the direct-IP firewall separately in step 10.

## 8. Enable and test certificate renewal

Install the repository's deploy hook **after NGINX is running**. It handles only the `origin.retru.co.ve` certificate, runs `nginx -t`, then gracefully reloads NGINX. Existing WebSocket/SSE connections continue on old workers. Renewal must not stop or restart the proxy.

```bash
install -d -m 755 /etc/letsencrypt/renewal-hooks/deploy
install -m 755 /opt/retruco/backend/deploy/renew-certificate.sh \
  /etc/letsencrypt/renewal-hooks/deploy/50-retruco
systemctl enable --now certbot.timer
systemctl list-timers certbot.timer
certbot renew --cert-name origin.retru.co.ve --dry-run --run-deploy-hooks
retruco-compose ps
```

The dry run uses the ACME staging service and runs the deploy hook against the **current live certificate**, not a staging certificate. Confirm both DNS validation and hook execution succeed. Real renewal updates the mounted `live` symlinks before the reload. Inspect `/var/log/letsencrypt/letsencrypt.log` and `journalctl -u certbot.service` for renewal or hook failures; a renewed certificate on disk alone does not prove NGINX loaded it. After fixing a reload failure, rerun the hook with `RENEWED_LINEAGE=/etc/letsencrypt/live/origin.retru.co.ve /etc/letsencrypt/renewal-hooks/deploy/50-retruco` and check HTTPS. Review token expiry and certificate expiry as part of routine operations. [Certbot renewal and hooks](https://eff-certbot.readthedocs.io/en/stable/using.html#renewing-certificates).

## 9. Connect Cloudflare Pages

Open **Cloudflare dashboard → Workers & Pages → the project serving `retruco.pages.dev` → Settings → Variables and Secrets**. Select the **Production** environment. The project name may differ from the name in `wrangler.toml`; select the existing deployed project.

Add these server-only runtime bindings:

| Name | Value |
| --- | --- |
| `BACKEND_ORIGIN` | `https://origin.retru.co.ve` |
| `API_PROXY_SECRET` | The new exact value from the reset droplet's `.env.production`; select **Encrypt** |

Open `.env.production` on the droplet to copy the secret into Cloudflare. Do not paste it into chat, commit it, or put it in a public frontend variable.

Replace the old `trycloudflare.com` value and old proxy secret if present. The VPS reset generated a new secret; the previous Pages value will not authenticate.

For the public build values:

| Name | Initial setting |
| --- | --- |
| `API_ORIGIN` | Remove it or leave it empty |
| `CARD_ASSET_ORIGIN` | Keep your working public B2/CDN base URL, or leave it empty to use bundled cards |
| `GOOGLE_ANALYTICS_ID` | Optional; leave empty to disable analytics |

`BACKEND_ORIGIN` has no `/api` suffix and must not point to `retru.co.ve`. Browser requests use `https://retru.co.ve/api`; the Pages server supplies the secret when forwarding to `origin.retru.co.ve`.

Save the settings and **redeploy the production Pages deployment**. Secrets must be configured before the deployment that uses them. [Cloudflare Pages bindings and secrets](https://developers.cloudflare.com/pages/functions/bindings/).

Under **Custom domains**, verify `retru.co.ve` is **Active**, with HTTPS working. If it is still pending, recheck the nameserver and Pages domain setup in step 1.

If you have not built the Pages project yet, use the frontend repository root (or `frontend` for the parent repository), build command `pnpm build`, and output `.svelte-kit/cloudflare`. The repo specifies Node 26 and pins `pnpm@12.11.2`; match those in the Pages build environment. [Cloudflare build-tool versions](https://developers.cloudflare.com/pages/configuration/build-image/).

## 10. Verify the complete connection

From your computer or the droplet:

```bash
curl -i https://retru.co.ve/api/health
```

Expect **200**, unlike the direct backend's unauthenticated 403. Open `https://retru.co.ve`, register/login, create a table, join from a second browser, and test play/chat/reconnection. Successful browser actions verify origin checks, cookies, and WebSockets as well as basic HTTP connectivity. A login cookie from `retruco.pages.dev` does not transfer to the new domain, and users from the old database will be absent unless you restored a backup.

From your computer **outside the droplet and Cloudflare**, verify HTTPS cannot bypass the edge (replace `DROPLET_IP`):

```bash
curl --noproxy '*' --connect-timeout 5 --max-time 10 \
  --resolve origin.retru.co.ve:443:DROPLET_IP \
  https://origin.retru.co.ve/api/health
```

Expect a connection timeout/refusal, **not any HTTP response**, including 403. `--resolve` supplies the correct TLS hostname while targeting the droplet; a certificate error alone is not proof of a blocked connection. Repeat against the droplet's public IPv6 address, if assigned, using `--resolve 'origin.retru.co.ve:443:[DROPLET_IPV6]'`. Also verify direct port 80 cannot connect. Keep a working SSH session and test the CI runner's SSH access after the firewall change.

Verify SSE updates arrive promptly and WebSocket connections survive an idle period and the step-8 renewal dry run. API responses must not show `CF-Cache-Status: HIT`. The browser should continue using `retru.co.ve/api`, never the origin hostname.

| Symptom | Check |
| --- | --- |
| Pages `/api/health` returns 503 | Missing runtime bindings, invalid backend URL, backend connectivity/TLS, or a Pages deployment created before the bindings were saved |
| Pages `/api/health` returns 403 | For application JSON, compare `API_PROXY_SECRET` and allowed origins. For Cloudflare HTML/challenges, inspect Security Events and the applicable WAF/Access rules |
| Cloudflare returns 521/522 | NGINX must listen on 443; check the droplet IP in DNS, attached firewalls, and the complete current Cloudflare source ranges |
| Cloudflare returns 525/526 | Check Full (strict), the origin certificate's hostname/expiry/chain, mounted `live`/`archive` paths, and whether the renewal hook reloaded NGINX |
| Direct-IP HTTPS returns any HTTP response | A provider/host firewall still permits bypass traffic; review every attached firewall, tags, and IPv4/IPv6 rules |
| Health succeeds but login/game actions fail | Use `https://retru.co.ve`, matching `ALLOWED_ORIGINS`; the old Pages address and `www` are separate origins |
| NGINX returns 502 | `retruco-compose logs --tail=80 api migrate` and the database/migration status |
| NGINX fails to start | DNS/certificate name mismatch, missing certificate files, or another service already using 443 |

By default, browser API access is allowed only from `https://retru.co.ve`. If you want the old production Pages address to remain usable too, set `ALLOWED_ORIGINS=https://retru.co.ve,https://retruco.pages.dev`, then run `retruco-compose up -d api` followed by `retruco-compose restart proxy` to refresh NGINX's upstream address after API recreation. Preview URLs require their own exact origins plus separate Pages **Preview** bindings. The `www.retru.co.ve` hostname also needs an explicit Pages custom-domain setup and an allowed-origin entry if you choose to serve the app there.

## Optional: enable reliable voice across networks

The current backend also supports voice signaling. For audio across NATs and mobile networks, deploy the repository's coturn relay; the basic HTTPS deployment above does not configure a TURN relay.

1. Add a **DNS-only A record** named `turn`, pointing to the same droplet IP.
2. Add these values to `/opt/retruco/backend/.env.production`, replacing `DROPLET_IP` and the secret placeholder:

```dotenv
VOICE_STUN_URLS=stun:turn.retru.co.ve:3478
VOICE_TURN_URLS=turn:turn.retru.co.ve:3478?transport=udp,turn:turn.retru.co.ve:3478?transport=tcp
VOICE_TURN_REALM=turn.retru.co.ve
VOICE_TURN_EXTERNAL_IP=DROPLET_IP
VOICE_TURN_SECRET=PASTE_A_NEW_RANDOM_SECRET
```

Generate the separate voice secret with `openssl rand -hex 32`. It stays on the backend; do not put it in Pages public variables.

3. Edit `/usr/local/sbin/retruco-compose` and insert this extra line immediately before `"$@"`, keeping the trailing backslash:

```sh
  -f /opt/retruco/backend/compose.voice.yaml \
```

4. Allow inbound **TCP/UDP 3478** and **UDP 49160–49260** from clients on the Internet in the DigitalOcean and host firewalls. These are explicit exceptions to the HTTPS source restriction; keep TCP 443 Cloudflare-only.
5. Run `retruco-compose config --quiet`, then `retruco-compose up -d --build` and `retruco-compose restart proxy`.
6. Test voice with two devices on different networks, including mobile data. To verify TURN itself, inspect the selected WebRTC ICE candidate and confirm a relay candidate is used.

TURN traffic goes directly to the droplet, not through the Cloudflare HTTP proxy or NGINX. The DNS-only TURN record exposes this droplet's IP, so origin protection depends on the HTTPS firewall, not IP secrecy. Cloudflare's HTTP proxy does not protect the public TURN ports. The supplied overlay has no TLS TURN listener; networks allowing only TLS egress need additional `turns:` infrastructure. See [the repository's voice deployment guide](./deploy/VOICE.md) for relay details and verification.

## Migrate an existing DNS-only deployment

Use this sequence to preserve the existing database, secrets, and certificate lineage. Keep an SSH session open and ensure DigitalOcean console access is available.

1. Copy the updated backend files as in step 3, preserving `.env.production`, `compose.vps.yaml`, and any operator customization. Change any CI `VPS_HOST` value of `origin.retru.co.ve` to the **droplet IP** and update its verified `VPS_KNOWN_HOSTS` entry. Arrange the runner's fixed SSH egress before restricting SSH.
2. Install `python3-certbot-dns-cloudflare` alongside the existing apt-installed Certbot and create the scoped credentials file from step 6. Do not mix apt and snap installations. `certbot reconfigure` requires Certbot 2.3 or newer (Ubuntu 24.04's package supports it).
3. Pause the renewal timer with `systemctl stop certbot.timer` and let any currently running `certbot.service` finish. Review and remove the old guide's two proxy-stop/start hooks; retain unrelated hooks:

   ```bash
   rm -f /etc/letsencrypt/renewal-hooks/pre/10-retruco \
     /etc/letsencrypt/renewal-hooks/post/10-retruco
   ```

   If you added stop/start hooks to `/etc/letsencrypt/cli.ini` or another scheduler, remove those too. Reconfigure the existing lineage, clearing any stored pre/post command hooks:

   ```bash
   certbot reconfigure --cert-name origin.retru.co.ve \
     --dns-cloudflare \
     --dns-cloudflare-credentials /root/.secrets/certbot/cloudflare.ini \
     --dns-cloudflare-propagation-seconds 60 \
     --pre-hook '' --post-hook '' --non-interactive
   ```

   This tests DNS validation before saving the new renewal settings. Resolve failures before continuing; do not leave renewal disabled after the maintenance work.
4. Keep `TLS_CERT_DIR=/etc/letsencrypt`. Regenerate `deploy/nginx.vps.conf.template` with the step-5 `sed` command so it uses the latest HTTPS-only template and Certbot paths. Preserve any deliberate local changes and keep the optional voice overlay in the helper. Run `retruco-compose config --quiet`, then `retruco-compose up -d --no-deps --force-recreate proxy`. This one-time container recreation removes the old published port 80 and briefly disconnects streams; routine renewals do not.
5. Install the new guarded deploy hook and enable/test the timer as in step 8. Confirm the DNS-01 dry run and graceful reload succeed while a game is connected.
6. Configure Cloudflare Full (strict), the edge certificate, cache bypass, and compatible security rules from step 1; switch `origin` to **Proxied**. Verify the Pages health check and a full browser session.
7. Apply the Cloudflare-only HTTPS firewall from step 1, remove old public HTTP/HTTPS allows, then run both the proxied and direct-IP checks from step 10. Verify SSH and CI still connect.

If the edge configuration fails, fix its TLS/rule settings first. Switching back to DNS only while keeping the Cloudflare-only firewall will make the backend unreachable. A deliberate rollback requires restoring the prior firewall access as well, which exposes the origin again. DNS-01 renewal can remain enabled through a rollback.

Reference: [changing Certbot renewal settings](https://eff-certbot.readthedocs.io/en/stable/using.html#modifying-the-renewal-configuration-of-existing-certificates).

## Later updates and backups

Once the initial deployment is healthy, use the backend repository's
[GitHub Actions setup](./deploy/GITHUB-ACTIONS.md) to deploy pushes to
`master` automatically. It uses the same server environment, Compose helper,
certificate configuration, and backup directory described here.

Before an update, take a database backup and retain a protected copy outside the droplet:

```bash
mkdir -p /opt/retruco/backups
chmod 700 /opt/retruco/backups
umask 077
retruco-compose exec -T db pg_dump -U retruco -d retruco -Fc \
  > "/opt/retruco/backups/retruco-$(date -u +%Y%m%dT%H%M%SZ).dump"
```

Copy updated code with the same `rsync` command, preserving the server's environment and VPS override files. Review schema changes before updating. For an update that requires migration before the new API starts, use a maintenance window:

```bash
retruco-compose build api migrate
retruco-compose stop api
retruco-compose run --rm migrate
```

Only if the migration succeeds:

```bash
retruco-compose up -d --no-deps api
retruco-compose restart proxy
```

Restarting NGINX also refreshes its upstream resolution if Docker assigned the recreated API a new address. Recheck the Pages health endpoint and a browser session. If migration fails, inspect it before restarting the application.

The PostgreSQL volume persists across container recreation. **Do not run `retruco-compose down -v`** unless you intentionally want to delete the database.
