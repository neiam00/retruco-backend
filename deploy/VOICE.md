# Voice with Cloudflare Realtime TURN

Voice uses WebRTC audio with Opus. The Go API handles room-scoped signaling at
`https://retru.co.ve/api/matches/{code}/voice`. Audio travels directly between
browsers when possible, or through Cloudflare Realtime TURN when relaying is
needed. The VPS does not run a media relay or record audio.

Everyone in a room can listen; the room creator controls spectator speaking
permission. These permissions and the browser signaling protocol are unchanged.

## Backend configuration

In Cloudflare Realtime TURN, obtain a **TURN Token ID** and its associated
**API token**. This is separate from the zone-scoped DNS-01 token used by Certbot
and from the Turnstile keys used for login.

Edit `/opt/retruco/backend/.env.production` on the VPS with a text editor:

```dotenv
CLOUDFLARE_TURN_TOKEN_ID=YOUR_TURN_TOKEN_ID
CLOUDFLARE_TURN_API_TOKEN=YOUR_TURN_API_TOKEN
```

Keep the file mode `600`. Do not paste the API token into chat, source code,
shell command arguments, Pages variables, or frontend configuration. Compose
passes both values only to the API container. Set both or neither: incomplete
configuration prevents API startup. With neither set, local development can
still use direct peer connections, but production audio may fail across NATs.

Deploy the updated backend through the normal deployment workflow. For a
manual update, follow the [VPS update procedure](../DEPLOYMENT-VPS.md#later-updates-and-backups).
No frontend rebuild is required for this provider change. After changing only
the token values, recreate the API and refresh NGINX's upstream resolution:

```sh
retruco-compose up -d --no-deps api
retruco-compose restart proxy
```

This disconnects active sessions briefly; use a maintenance window. API tokens
are read at startup, so editing the environment file alone does not rotate them.

## Credential lifecycle

After checking the session, origin, and room, the backend calls
`POST https://rtc.live.cloudflare.com/v1/turn/keys/{TURN_TOKEN_ID}/credentials/generate-ice-servers`
with bearer authentication and `{"ttl":3600}`. Each voice join gets temporary
credentials; the browser requests replacements every 30 minutes and restarts
ICE after installing them. Only the returned ICE configuration reaches clients.
The long-lived API token and provider response bodies are never logged.

Issuance has a five-second timeout and a shared per-user limit of four attempts
per minute across joins and refreshes. Redirects are refused. Responses must
contain usable TURN credentials; browser-blocked port 53 URLs are filtered out.
The provider supplies UDP, TCP, and TLS relay addresses, including TLS on 443.

An initial provider failure returns HTTP 503 before the voice socket opens;
issuance throttling returns 429. A failed refresh closes the voice connection,
and the existing UI asks the user to rejoin. Game state remains unaffected.

## Migrate from local Coturn

Do this **before deploying a version that removes `compose.voice.yaml`**.
Automatic deployment refuses to replace source while the old helper still
includes the local `turn` service.

1. Put the two Cloudflare values in `.env.production`. Keep the old `VOICE_*`
   values until the next step so the old overlay can still be parsed.
2. During a voice maintenance window, use the **old** helper and overlay to stop
   and remove only the relay container:

   ```sh
   retruco-compose stop turn
   retruco-compose rm -f turn
   ```

3. Remove the `-f /opt/retruco/backend/compose.voice.yaml` line from
   `/usr/local/sbin/retruco-compose`. Keep the other Compose files and `"$@"`.
   Remove the obsolete `VOICE_STUN_URLS`, `VOICE_TURN_URLS`, `VOICE_TURN_SECRET`,
   `VOICE_TURN_REALM`, and `VOICE_TURN_EXTERNAL_IP` entries from `.env.production`.
   Run `retruco-compose config --quiet` to verify the helper.
4. Deploy the new backend, then test voice as described below. The old backend
   does not understand the new Cloudflare settings; relayed audio is unavailable
   between stopping Coturn and completing the deployment.
5. Remove the droplet's inbound TCP/UDP **3478** and UDP **49160–49260** firewall
   rules and delete the old DNS-only `turn.retru.co.ve` record if nothing else
   uses them. These ports are no longer used by this application.

No TURN hostname or extra inbound port is needed on the VPS. Keep origin HTTPS
on 443 restricted to Cloudflare's HTTP proxy ranges, with administrative SSH
separate. The API needs outbound HTTPS to `rtc.live.cloudflare.com`; browsers
connect to the relay addresses returned by Cloudflare. Do not point an orange-
cloud record at a local relay or at Cloudflare's TURN service.

## Verification

Use two devices on different networks, including cellular data. Join one room,
enable audio, and test open mic and push-to-talk both ways. Check the WebRTC ICE
details in `chrome://webrtc-internals` or `about:webrtc` and confirm a **relay**
candidate is selected when direct connectivity is unavailable. A same-network
direct connection alone does not verify the Cloudflare credentials.

Keep a voice session active past 30 minutes to verify refreshed ICE credentials.
Check Cloudflare Realtime TURN analytics for relay usage. If joins fail, check
the backend's sanitized `voice credentials unavailable` log: HTTP 401/403 points
to the key/token pair; timeouts or 5xx indicate connectivity/provider issues.
Review usage and billing in Cloudflare when enabling the service.

Reference: [Cloudflare credential API](https://developers.cloudflare.com/realtime/turn/generate-credentials/).
