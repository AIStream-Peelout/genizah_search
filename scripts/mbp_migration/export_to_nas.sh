#!/bin/bash
# Stage the production serving stack on the NAS so the M3 MacBook Pro can pull it
# without any SSH between the machines.
#
# Runs ON THE STUDIO. Read-only for production: nothing is stopped, restarted,
# rebuilt or pruned. The one exception is documented in step 'es': the ES
# snapshot needs `path.repo` active, which requires a one-time recreate of the
# elasticsearch container (docker-compose.yml now carries the setting); this
# script never does that itself, it just tells you when it is missing.
#
# Usage:
#   scripts/mbp_migration/export_to_nas.sh                 # all steps
#   STEPS="es" scripts/mbp_migration/export_to_nas.sh      # subset
#
# Environment:
#   STAGE   staging dir on the NAS (default /Volumes/home/genizah_migration)
#   STEPS   subset of: images volumes repo data cloudflared es manifest
set -euo pipefail

LOCAL_REPO="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="${STAGE:-/Volumes/home/genizah_migration}"
STEPS="${STEPS:-images volumes repo data cloudflared es manifest}"

# Exact images serving prod today (backend/frontend built 2026-09-10 from 94d7aee).
IMAGES=(
  genizah_search-backend:latest
  genizah_search-embedding:latest
  genizah_search-frontend:latest
  neo4j:2026.04.0
  docker.elastic.co/elasticsearch/elasticsearch:8.18.2
  docker.elastic.co/kibana/kibana:8.18.2
)

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

[ -d "$(dirname "$STAGE")" ] || { echo "NAS not mounted at $(dirname "$STAGE")"; exit 1; }
mkdir -p "$STAGE"/{images,volumes,repo,data,cloudflared,es_snapshot}

step_images() {
  for img in "${IMAGES[@]}"; do
    local id file
    id="$(docker image inspect "$img" --format '{{.Id}}')"
    file="$STAGE/images/$(echo "$img" | tr '/:' '__').tar.gz"
    if [ -f "$file" ] && grep -q "$id" "$STAGE/images/manifest.txt" 2>/dev/null; then
      log "Image $img already staged (same ID), skipping"
      continue
    fi
    log "Saving image $img ($(docker image inspect "$img" --format '{{.Size}}' | awk '{printf "%.1f GB", $1/1e9}')) -> $file"
    docker save "$img" | gzip -1 > "$file.part" && mv "$file.part" "$file"
    grep -v " $img\$" "$STAGE/images/manifest.txt" 2>/dev/null > "$STAGE/images/manifest.tmp" || true
    echo "$id $img" >> "$STAGE/images/manifest.tmp"; mv "$STAGE/images/manifest.tmp" "$STAGE/images/manifest.txt"
  done
}

step_volumes() {
  # Embedding model weights + cache, streamed out of the running embedding
  # container with docker cp (no helper image, read-only). Neo4j is deliberately
  # not exported: the MBP starts with a clean graph (docs/MBP_MIGRATION.md).
  for path in hf_home embedding_cache; do
    log "Exporting volume $path from genizah_search-embedding-1:/app/$path"
    docker cp "genizah_search-embedding-1:/app/$path" - | gzip -1 > "$STAGE/volumes/$path.tar.gz.part"
    mv "$STAGE/volumes/$path.tar.gz.part" "$STAGE/volumes/$path.tar.gz"
  done
}

step_repo() {
  log "Archiving the repo working tree (branch $(git -C "$LOCAL_REPO" rev-parse --abbrev-ref HEAD), incl. .git and .env files)"
  tar -C "$(dirname "$LOCAL_REPO")" -czf "$STAGE/repo/genizah_search.tar.gz" \
    --exclude='.venv' --exclude='node_modules' --exclude='__pycache__' --exclude='.DS_Store' \
    --exclude='genizah_search/cache' --exclude='genizah_search/logs' --exclude='genizah_search/backups' --exclude='genizah_search/data' \
    --exclude='genizah_search/deployment.log' \
    "$(basename "$LOCAL_REPO")"
  chmod 600 "$STAGE/repo/genizah_search.tar.gz"
}

step_data() {
  log "Syncing data/visualization (2 GB precomputed cache)"
  rsync -a "$LOCAL_REPO/data/visualization/" "$STAGE/data/visualization/"
}

step_cloudflared() {
  log "Archiving ~/.cloudflared (tunnel credentials; the MBP must NOT start the tunnel before cutover)"
  tar -C "$HOME" -czf "$STAGE/cloudflared/cloudflared.tar.gz" .cloudflared
  chmod 600 "$STAGE/cloudflared/cloudflared.tar.gz"
}

step_es() {
  log "Elasticsearch snapshot of every non-system index -> backups/elasticsearch -> NAS"
  python3 "$LOCAL_REPO/scripts/mbp_migration/es_migrate.py" --snapshot
  log "Copying the snapshot repository to the NAS"
  rsync -a "$LOCAL_REPO/backups/elasticsearch/" "$STAGE/es_snapshot/"
}

step_manifest() {
  {
    echo "staged: $(date -u +%FT%TZ) from $(hostname)"
    echo "repo commit: $(git -C "$LOCAL_REPO" rev-parse HEAD) ($(git -C "$LOCAL_REPO" rev-parse --abbrev-ref HEAD))"
    du -sh "$STAGE"/* 2>/dev/null
  } > "$STAGE/MANIFEST.txt"
  cp "$LOCAL_REPO/scripts/mbp_migration/MBP_CLAUDE_PROMPT.md" "$STAGE/README_MBP.md"
  cat "$STAGE/MANIFEST.txt"
}

for step in $STEPS; do
  "step_$step"
done

log "Done. On the MBP, mount the NAS and follow $STAGE/README_MBP.md"
