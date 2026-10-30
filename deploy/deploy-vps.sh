#!/usr/bin/env bash
# Runs on the VPS after the one-time setup in the deployment guide.
set -Eeuo pipefail
umask 077

archive="${1:?Usage: deploy-vps.sh SOURCE_ARCHIVE COMMIT_SHA IMAGE_ARCHIVE IMAGE_SHA256}"
revision="${2:?Supply the deployed commit SHA}"
image_archive="${3:?Supply the CI-built image archive}"
image_sha256="${4:?Supply the CI-built image archive SHA-256}"
image="retruco-backend:$revision"
deploy_dir="${RETRUCO_DEPLOY_DIR:-/opt/retruco/backend}"
compose="${RETRUCO_COMPOSE:-/usr/local/sbin/retruco-compose}"
public_health="${RETRUCO_HEALTH_URL:-}"

[[ "$revision" =~ ^[a-f0-9]{40}$ ]] || { echo "Invalid commit SHA" >&2; exit 1; }
[[ "$image_sha256" =~ ^[a-f0-9]{64}$ ]] || { echo "Invalid image archive SHA-256" >&2; exit 1; }
[[ "$deploy_dir" = /* && "$deploy_dir" != / && -d "$deploy_dir" ]] || { echo "Deployment directory is missing or invalid" >&2; exit 1; }
[[ -f "$archive" && -f "$image_archive" && -x "$compose" ]] || { echo "Source/image archive or VPS Compose helper is missing" >&2; exit 1; }
[[ -f "$deploy_dir/.env.production" && -f "$deploy_dir/compose.vps.yaml" &&
   -f "$deploy_dir/deploy/nginx.vps.conf.template" ]] || {
  echo "Complete the VPS guide first: .env.production, compose.vps.yaml and the VPS NGINX template are required." >&2
  exit 1
}

for tool in flock rsync tar curl docker sha256sum; do
  command -v "$tool" > /dev/null || { echo "Missing VPS command: $tool" >&2; exit 1; }
done
parent_dir="$(dirname "$deploy_dir")"
exec 9>"$parent_dir/.deploy.lock"
flock -n 9 || { echo "Another deployment is already running on this VPS" >&2; exit 1; }

stage="$(mktemp -d "$parent_dir/.deploy-source.XXXXXX")"
api_stopped=0
migration_started=0
deployment_complete=0
notice_sent=0

notify_update() {
  # Expand the proxy secret inside the container, never in the host shell.
  # shellcheck disable=SC2016
  "$compose" exec -T api sh -c '
    health=$(wget -q -T 10 -O - --header="X-Retruco-Proxy-Secret: $API_PROXY_SECRET" http://127.0.0.1:8080/api/health) || exit 1
    case "$health" in *\"deploymentNotices\":true*) ;; *) exit 44 ;; esac
    wget -q -T 10 -O /dev/null --header="X-Retruco-Proxy-Secret: $API_PROXY_SECRET" \
      --header="Content-Type: application/json" --post-data="$1" \
      http://127.0.0.1:8080/internal/deployment
  ' sh "{\"updating\":$1}" < /dev/null
}

cleanup() {
  status=$?
  trap - EXIT
  if (( status == 0 && deployment_complete == 0 )); then
    echo "Deployment exited before completing all health checks." >&2
    status=1
  fi
  if (( status != 0 )); then
    if (( notice_sent == 1 && api_stopped == 0 && migration_started == 0 )); then
      notify_update false || true
    fi
    if (( api_stopped == 1 && migration_started == 0 )); then
      # The schema is unchanged: restart the existing container, without recreating it.
      "$compose" up -d --no-build --no-deps --no-recreate api || true
    fi
    echo "Deployment failed. Inspect the VPS; no database/schema rollback was attempted." >&2
    if (( migration_started == 1 )); then
      echo "Migrations may have changed the schema. Fix forward or review the database backup before recovery." >&2
    fi
  fi
  rm -rf -- "$stage"
  rm -f -- "$archive" "$image_archive"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# Capture the installed origin configuration before replacing the old Compose
# file. Older installations kept it in source instead of .env.production.
needs_origin_migration=0
if ! grep -Eq '^[[:space:]]*(export[[:space:]]+)?ALLOWED_ORIGINS[[:space:]]*=' "$deploy_dir/.env.production"; then
  needs_origin_migration=1
fi
if [[ -z "$public_health" ]] || (( needs_origin_migration == 1 )); then
  installed_origins="$("$compose" exec -T api printenv ALLOWED_ORIGINS < /dev/null)" || {
    echo "Cannot read installed frontend origins; configure ALLOWED_ORIGINS in .env.production and RETRUCO_HEALTH_URL before deploying." >&2
    exit 1
  }
  [[ -n "$installed_origins" ]] || { echo "Installed ALLOWED_ORIGINS is empty" >&2; exit 1; }
  [[ "$installed_origins" != *$'\n'* && "$installed_origins" != *$'\r'* ]] || {
    echo "Installed ALLOWED_ORIGINS must be a single line" >&2
    exit 1
  }
  IFS=, read -r -a origin_list <<< "$installed_origins"
  for origin in "${origin_list[@]}"; do
    [[ "$origin" =~ ^https?://([a-zA-Z0-9.-]+|\[[a-fA-F0-9:]+\])(:[0-9]+)?$ ]] || {
      echo "Installed ALLOWED_ORIGINS must contain exact HTTP(S) origins" >&2
      exit 1
    }
  done
  if (( needs_origin_migration == 1 )); then
    printf '\nALLOWED_ORIGINS=%s\n' "$installed_origins" >> "$deploy_dir/.env.production"
    echo "Preserved installed frontend origins in .env.production."
  fi
  public_health="${public_health:-${origin_list[0]}/api/health}"
fi

"$compose" config --quiet
services="$("$compose" config --services)"
if grep -qx turn <<< "$services"; then
  echo "Retire the local Coturn service and remove compose.voice.yaml from retruco-compose before deploying. See deploy/VOICE.md." >&2
  exit 1
fi
[[ -n "$("$compose" ps -q db)" ]] || { echo "PostgreSQL must already be running; complete the initial deployment first." >&2; exit 1; }
tar -xzf "$archive" -C "$stage"
[[ -f "$stage/Dockerfile" && -f "$stage/compose.yaml" ]] || { echo "Archive is not the backend repository" >&2; exit 1; }

# Verify the exact CI archive before loading it. Image IDs differ between
# containerd and classic Docker stores; the archive checksum is portable.
checksum="$(sha256sum -- "$image_archive")"
[[ "${checksum%% *}" == "$image_sha256" ]] || {
  echo "CI image archive checksum does not match" >&2
  exit 1
}
echo "Verified CI image archive checksum $image_sha256."
docker image load --input "$image_archive" < /dev/null
platform="$(docker version --format '{{.Server.Os}}/{{.Server.Arch}}')"
metadata="$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}} {{.Os}}/{{.Architecture}}' "$image")"
[[ "$metadata" == "$revision $platform" ]] || {
  echo "CI image commit label or platform does not match this deployment/VPS" >&2
  exit 1
}

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
echo "Using CI-built backend commit $revision."
docker image tag "$image" retruco-backend:production

backup_dir="$parent_dir/backups"
mkdir -p "$backup_dir"
chmod 700 "$backup_dir"
backup="$backup_dir/retruco-$(date -u +%Y%m%dT%H%M%SZ)-${revision:0:12}-$$.dump"
notice_sent=1
notice_status=0
notify_update true || notice_status=$?
if (( notice_status == 44 )); then
  # The first rollout installs this capability in the running backend.
  notice_sent=0
  echo "The running API predates update notices; bootstrapping this deployment."
elif (( notice_status != 0 )); then
  echo "Could not notify players; keeping the current API running." >&2
  exit 1
else
  echo "Players notified. Pausing for 10 seconds before restarting the API."
  sleep 10
fi
"$compose" stop api
api_stopped=1
# Stop writes before taking the recovery snapshot and applying migrations.
# Compose otherwise forwards stdin, which may contain the rest of a script
# streamed over SSH. Backup and migration commands never need interactive input.
if ! "$compose" exec -T db pg_dump -U retruco -d retruco -Fc < /dev/null > "$backup.partial"; then
  rm -f "$backup.partial"
  exit 1
fi
[[ -s "$backup.partial" ]] || { rm -f "$backup.partial"; echo "Database backup is empty" >&2; exit 1; }
mv "$backup.partial" "$backup"
echo "Database backup saved at $backup"

migration_started=1
"$compose" run --rm --pull never --no-deps migrate < /dev/null
"$compose" up -d --no-build --pull never --no-deps api
api_stopped=0

healthy=0
for ((attempt = 1; attempt <= 30; attempt++)); do
  # Read the proxy secret inside the container; never send it through Actions logs.
  # shellcheck disable=SC2016
  if "$compose" exec -T api sh -c \
    'wget -q -O /dev/null --header="X-Retruco-Proxy-Secret: $API_PROXY_SECRET" http://127.0.0.1:8080/api/health' \
    < /dev/null > /dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 2
done
(( healthy == 1 )) || { echo "New API did not become healthy" >&2; exit 1; }

# Recreate NGINX to pick up template changes and resolve the new API address.
"$compose" up -d --no-build --no-deps --force-recreate proxy
curl --fail --silent --show-error --retry 10 --retry-delay 3 \
  --retry-connrefused --max-time 10 --header 'Cache-Control: no-cache' \
  --get --data-urlencode "deployment=$revision-$$" "$public_health" > /dev/null
printf '%s\n' "$revision" > "$deploy_dir/.deployed-revision"
deployment_complete=1
echo "Deployed $revision; API and public Pages health checks passed."
