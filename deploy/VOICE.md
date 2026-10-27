# Voice deployment

Voice uses WebRTC audio with Opus. The Go API only relays room-scoped signaling;
it does not receive or record audio. Players and spectators explicitly join the
audio. Everyone can listen; the room creator chooses whether spectators may
speak. Existing rooms default to listening-only spectators.

The browser requires HTTPS outside localhost. Keep the API and voice signaling
under `https://retru.co.ve/api`; the existing WebSocket proxy handles
`/api/matches/{code}/voice`.

## Self-hosted coturn

The `compose.voice.yaml` overlay runs coturn on the Linux backend host. Use a
DNS-only hostname for TURN that resolves directly to that host's public IP.
Neither Cloudflare's HTTP proxy nor NGINX can proxy this UDP service.
Keep `origin.retru.co.ve` proxied for HTTPS and use a separate DNS-only TURN
hostname such as `turn.retru.co.ve`. Its DNS/ICE addresses reveal the relay IP;
if it shares the VPS, the HTTPS firewall must still prevent direct-origin access.

Set these values in the backend `.env` (replace the example hostname/IP):

```dotenv
VOICE_STUN_URLS=stun:turn.example.com:3478
VOICE_TURN_URLS=turn:turn.example.com:3478?transport=udp,turn:turn.example.com:3478?transport=tcp
VOICE_TURN_REALM=turn.example.com
VOICE_TURN_EXTERNAL_IP=203.0.113.10
VOICE_TURN_SECRET=<a-long-random-secret>
```

Generate the secret locally, for example with `openssl rand -hex 32`. Use the
same secret in the API and coturn. Never expose it in a frontend environment
variable. The API issues per-user HMAC-SHA1 TURN credentials valid for one hour
and refreshes them every 30 minutes while voice is connected.

Open inbound TCP/UDP 3478 and UDP 49160–49260 on the host and its cloud firewall,
then start:

```sh
docker compose -f compose.yaml -f compose.voice.yaml up -d --build
```

Those relay ports must accept Internet clients, not just Cloudflare IP ranges.
Retain the Cloudflare-only source restriction on HTTPS port 443. In the VPS
deployment, add this overlay to `retruco-compose` as described in the
[VPS deployment guide](../DEPLOYMENT-VPS.md) instead of starting a separate Compose project.

For a host behind NAT, use coturn's `PUBLIC_IP/PRIVATE_IP` form for
`VOICE_TURN_EXTERNAL_IP` and forward the same port ranges without translation.
The relay blocks private, loopback, link-local, and multicast destinations.
Tune the allocation quotas and relay range together when increasing capacity.

WebRTC media remains encrypted with DTLS-SRTP over UDP or TCP TURN. The supplied
overlay does not provide a TLS TURN listener. Networks allowing only TLS egress
need a `turns:` endpoint with a valid certificate, typically on a dedicated IP
at port 443; configure that listener in coturn and include its URL in
`VOICE_TURN_URLS`. Do not point a `turns:` URL at the plain TCP listener.

## Verification

Use two devices on different networks, including a cellular connection. Join
the same room, enter audio, and verify open mic and push-to-talk in both
directions. Check `chrome://webrtc-internals` or `about:webrtc`: the audio codec
must be Opus. To verify the relay rather than a direct connection, test with
direct UDP blocked and confirm the selected ICE pair uses a relay candidate.

Local development can leave all voice environment variables empty. Direct
connections then depend on the local network; a passing local test alone does
not prove that the production TURN firewall or DNS is configured correctly.

Reference: [coturn Docker deployment](https://github.com/coturn/coturn/blob/master/docker/coturn/README.md).
