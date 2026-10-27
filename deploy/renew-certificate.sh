#!/bin/sh
# Certbot deploy hook: install only after the proxy is running.
# Optional arguments select a certificate lineage and Compose helper.
set -eu

expected_lineage=${1:-}
compose=${2:-/usr/local/sbin/retruco-compose}

: "${RENEWED_LINEAGE:?Run as a Certbot deploy hook with RENEWED_LINEAGE set}"
# Without an explicit lineage, use the running proxy's configured hostname.
if [ -z "$expected_lineage" ]; then
  backend_host=$("$compose" exec -T proxy printenv BACKEND_HOST < /dev/null)
  case "$backend_host" in
    ''|*[!a-zA-Z0-9.-]*) echo 'Invalid proxy BACKEND_HOST' >&2; exit 1 ;;
  esac
  expected_lineage="/etc/letsencrypt/live/$backend_host"
fi
# Directory hooks run for every certificate on the host. Ignore other sites.
[ "$RENEWED_LINEAGE" = "$expected_lineage" ] || exit 0

# A failed check must not reload NGINX. Propagate failures for operator attention.
# Graceful reload keeps existing WebSocket/SSE connections on the old workers.
"$compose" exec -T proxy nginx -t < /dev/null
"$compose" exec -T proxy nginx -s reload < /dev/null
