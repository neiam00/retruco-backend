#!/usr/bin/env bash
# Runs on the VPS after the one-time setup in the deployment guide.
set -Eeuo pipefail
umask 077

archive="${1:?Usage: deploy-vps.sh SOURCE_ARCHIVE COMMIT_SHA}"
revision="${2:?Supply the deployed commit SHA}"
deploy_dir="${RETRUCO_DEPLOY_DIR:-/opt/retruco/backend}"
compose="${RETRUCO_COMPOSE:-/usr/local/sbin/retruco-compose}"
public_health="${RETRUCO_HEALTH_URL:-https://retru.co.ve/api/health}"

[[ "$revision" =~ ^[a-f0-9]{40}$ ]] || { echo "Invalid commit SHA" >&2; exit 1; }
[[ "$deploy_dir" = /* && "$deploy_dir" != / && -d "$deploy_dir" ]] || { echo "Deployment directory is missing or invalid" >&2; exit 1; }
[[ -f "$archive" && -x "$compose" ]] || { echo "Archive or VPS Compose helper is missing" >&2; exit 1; }
[[ -f "$deploy_dir/.env.production" && -f "$deploy_dir/compose.vps.yaml" &&
   -f "$deploy_dir/deploy/nginx.vps.conf.template" ]] || {
  echo "Complete the VPS guide first: .env.production, compose.vps.yaml and the VPS NGINX template are required." >&2
  exit 1
}

for tool in flock rsync tar curl; do
  command -v "$tool" > /dev/null || { echo "Missing VPS command: $tool" >&2; exit 1; }
done
parent_dir="$(dirname "$deploy_dir")"
exec 9>"$parent_dir/.deploy.lock"
flock -n 9 || { echo "Another deployment is already running on this VPS" >&2; exit 1; }

stage="$(mktemp -d "$parent_dir/.deploy-source.XXXXXX")"
api_stopped=0
migration_started=0
cleanup() {
  status=$?
  trap - EXIT
  if (( status != 0 )); then
    if (( api_stopped == 1 && migration_started == 0 )); then
      # The schema is unchanged: restart the existing container, without recreating it.
      "$compose" start api || true
    fi
    echo "Deployment failed. Inspect the VPS; no database/schema rollback was attempted." >&2
    if (( migration_started == 1 )); then
      echo "Migrations may have changed the schema. Fix forward or review the database backup before recovery." >&2
    fi
  fi
  rm -rf -- "$stage"
  rm -f -- "$archive"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

"$compose" config --quiet
[[ -n "$("$compose" ps -q db)" ]] || { echo "PostgreSQL must already be running; complete the initial deployment first." >&2; exit 1; }
tar -xzf "$archive" -C "$stage"
[[ -f "$stage/Dockerfile" && -f "$stage/compose.yaml" ]] || { echo "Archive is not the backend repository" >&2; exit 1; }

# Only application source is managed here. Preserve host secrets and operator files.
# Excluded destination files are also protected from rsync's --delete.
rsync --archive --delete \
  --exclude='/.env' --exclude='/.env.*' \
  --exclude='/.git/' --exclude='/.agents/' --exclude='/.agent/' \
  --exclude='/.deployed-revision' \
  --exclude='/compose.vps.yaml' \
  --exclude='/deploy/nginx.vps.conf.template' \
  --exclude='/bin/' --exclude='/coverage.out' \
  "$stage/" "$deploy_dir/"

"$compose" config --quiet
echo "Building backend commit $revision while the current API continues running."
"$compose" build api migrate

backup_dir="$parent_dir/backups"
mkdir -p "$backup_dir"
chmod 700 "$backup_dir"
backup="$backup_dir/retruco-$(date -u +%Y%m%dT%H%M%SZ)-${revision:0:12}-$$.dump"
"$compose" stop api
api_stopped=1
# Stop writes before taking the recovery snapshot and applying migrations.
if ! "$compose" exec -T db pg_dump -U retruco -d retruco -Fc > "$backup.partial"; then
  rm -f "$backup.partial"
  exit 1
fi
[[ -s "$backup.partial" ]] || { rm -f "$backup.partial"; echo "Database backup is empty" >&2; exit 1; }
mv "$backup.partial" "$backup"
echo "Database backup saved at $backup"

migration_started=1
"$compose" run --rm --no-deps migrate
"$compose" up -d --no-build --no-deps api
api_stopped=0

healthy=0
for ((attempt = 1; attempt <= 30; attempt++)); do
  # Read the proxy secret inside the container; never send it through Actions logs.
  if "$compose" exec -T api sh -c \
    'wget -q -O /dev/null --header="X-Retruco-Proxy-Secret: $API_PROXY_SECRET" http://127.0.0.1:8080/api/health' \
    > /dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 2
done
(( healthy == 1 )) || { echo "New API did not become healthy" >&2; exit 1; }

services="$("$compose" config --services)"
if grep -qx turn <<< "$services"; then
  "$compose" up -d --no-deps turn
fi
# Recreate NGINX to pick up template changes and resolve the new API address.
"$compose" up -d --no-build --no-deps --force-recreate proxy
curl --fail --silent --show-error --retry 10 --retry-delay 3 \
  --retry-connrefused --max-time 10 "$public_health" > /dev/null
printf '%s\n' "$revision" > "$deploy_dir/.deployed-revision"
echo "Deployed $revision; API and public Pages health checks passed."
