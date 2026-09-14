#!/bin/bash
# Stream the production serving stack from the Mac Studio to the M3 MacBook Pro.
#
# Runs ON THE STUDIO. It is read-only with respect to production: nothing is
# stopped, restarted, rebuilt, or pruned here. Everything is streamed over SSH,
# so no staging copy lands on the Studio's disk.
#
# Usage:
#   scripts/mbp_migration/export_to_mbp.sh                 # all steps
#   STEPS="images volumes" scripts/mbp_migration/export_to_mbp.sh   # subset
#
# Environment overrides:
#   MBP          ssh target (default isaac@isaacs-MacBook-Pro.local)
#   REMOTE_REPO  checkout path on the MBP (default same path as here)
#   STEPS        space-separated subset of:
#                precheck repo images create volumes cloudflared bootstrap
set -euo pipefail

MBP="${MBP:-isaac@isaacs-MacBook-Pro.local}"
LOCAL_REPO="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE_REPO="${REMOTE_REPO:-$LOCAL_REPO}"
STEPS="${STEPS:-precheck repo images create volumes cloudflared bootstrap}"
COMPOSE_FILES="-f docker-compose.yml -f docker-compose.mbp.yml"

# Exact images serving prod today. Shipping them verbatim means the MBP runs
# the same bits (backend/frontend built 2026-09-10 from 94d7aee), no rebuild.
IMAGES=(
  genizah_search-backend:latest
  genizah_search-embedding:latest
  genizah_search-frontend:latest
  neo4j:2026.04.0
  docker.elastic.co/elasticsearch/elasticsearch:8.18.2
  docker.elastic.co/kibana/kibana:8.18.2
)

# Named volumes copied byte-for-byte (embedding model weights + cache).
# Deliberately NOT copied:
#   * neo4j_data  - the MBP starts with a clean, empty graph that the
#                   historical-document-analysis pipeline rebuilds from scratch
#                   (decision 2026-09-14; see docs/MBP_MIGRATION.md).
#   * elasticsearch_data - a live Lucene directory cannot be tarred safely; every
#                   index is mirrored with reindex-from-remote in bootstrap_mbp.sh.
VOLUMES=(
  genizah_search_hf_home
  genizah_search_embedding_cache
)

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
remote() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$MBP" "$@"; }

step_precheck() {
  log "Checking SSH + Docker on $MBP"
  remote 'printf "%s macOS %s, RAM %s GB\n" "$(hostname)" "$(sw_vers -productVersion)" "$(( $(sysctl -n hw.memsize) / 1073741824 ))"'
  remote 'docker info --format "Docker {{.ServerVersion}}, VM RAM {{.MemTotal}} bytes, arch {{.Architecture}}"'
  remote 'df -h / | tail -1'
  remote "mkdir -p '$REMOTE_REPO' ~/.cloudflared"
}

step_repo() {
  log "Syncing repo working tree -> $MBP:$REMOTE_REPO (with .env files; without venv/data/logs)"
  rsync -az --info=progress2 --delete \
    --exclude .venv --exclude node_modules --exclude __pycache__ --exclude .DS_Store \
    --exclude logs/ --exclude backups/ --exclude data/ --exclude deployment.log \
    "$LOCAL_REPO/" "$MBP:$REMOTE_REPO/"
  log "Syncing data/visualization (2 GB precomputed cache; avoids a cold precompute on first boot)"
  rsync -az --info=progress2 "$LOCAL_REPO/data/visualization/" "$MBP:$REMOTE_REPO/data/visualization/"
  remote "mkdir -p '$REMOTE_REPO/logs' '$REMOTE_REPO/backups/elasticsearch' && chmod 600 '$REMOTE_REPO/.env' '$REMOTE_REPO/src/backend/.env'"
}

step_images() {
  for img in "${IMAGES[@]}"; do
    local id size
    id="$(docker image inspect "$img" --format '{{.Id}}')"
    if remote "docker image inspect '$img' --format '{{.Id}}' 2>/dev/null" | grep -q "$id"; then
      log "Image $img already on the MBP with the same ID, skipping"
      continue
    fi
    size="$(docker image inspect "$img" --format '{{.Size}}' | awk '{printf "%.1f GB", $1/1e9}')"
    log "Streaming image $img ($size) ..."
    docker save "$img" | gzip -1 | remote 'gunzip | docker load'
  done
}

step_create() {
  log "Creating containers on the MBP without starting them (so named volumes exist for restore)"
  remote "cd '$REMOTE_REPO' && docker compose $COMPOSE_FILES create --no-build"
}

step_volumes() {
  for vol in "${VOLUMES[@]}"; do
    log "Streaming volume $vol ..."
    docker run --rm -v "$vol:/v:ro" alpine tar -C /v -cf - . | gzip -1 \
      | remote "gunzip | docker run --rm -i -v '$vol:/v' alpine sh -c 'find /v -mindepth 1 -delete; tar -C /v -xf -'"
  done
}

step_cloudflared() {
  log "Copying cloudflared tunnel credentials + ingress config (tunnel is NOT started on the MBP yet)"
  rsync -az "$HOME/.cloudflared/" "$MBP:.cloudflared/"
}

step_bootstrap() {
  log "Running bootstrap on the MBP"
  remote "cd '$REMOTE_REPO' && bash scripts/mbp_migration/bootstrap_mbp.sh"
}

for step in $STEPS; do
  "step_$step"
done

log "Export finished. Next: scripts/mbp_migration/cutover_mbp.sh on the MBP (see docs/MBP_MIGRATION.md)."
