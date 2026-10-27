#!/bin/sh
# Certbot deploy hook: install only after the proxy is running.
# Optional arguments support installations with a different lineage or helper.
set -eu

expected_lineage=${1:-/etc/letsencrypt/live/origin.retru.co.ve}
compose=${2:-/usr/local/sbin/retruco-compose}

: "${RENEWED_LINEAGE:?Run as a Certbot deploy hook with RENEWED_LINEAGE set}"
# Directory hooks run for every certificate on the host. Ignore other sites.
[ "$RENEWED_LINEAGE" = "$expected_lineage" ] || exit 0

# A failed check must not reload NGINX. Propagate failures for operator attention.
# Graceful reload keeps existing WebSocket/SSE connections on the old workers.
"$compose" exec -T proxy nginx -t < /dev/null
"$compose" exec -T proxy nginx -s reload < /dev/null
