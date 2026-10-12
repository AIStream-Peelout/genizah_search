#!/bin/bash
# Bring the serving stack up on the M3 MacBook Pro from the NAS staging dir
# written by export_to_nas.sh. Idempotent: re-run after a partial failure.
#
# Runs ON THE MBP, from inside the extracted repo:
#   tar -xzf /Volumes/home/genizah_migration/repo/genizah_search.tar.gz -C ~/Documents/GitHub
#   cd ~/Documents/GitHub/genizah_search && scripts/mbp_migration/bootstrap_mbp.sh
#
# Steps (STEPS="..." to run a subset):
#   check images create volumes data es_repo up es verify cloudflared
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
STAGE="${STAGE:-/Volumes/home/genizah_migration}"
STEPS="${STEPS:-check images create volumes data es_repo up es verify cloudflared}"
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.mbp.yml"
ES_TOOL="python3 scripts/mbp_migration/es_migrate.py"

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

wait_http() {
  # $1 = url, $2 = label, $3 = timeout seconds
  local i=0
  until curl -s -o /dev/null -f "$1"; do
    i=$((i + 5)); [ "$i" -ge "$3" ] && { echo "TIMEOUT waiting for $2 at $1"; return 1; }
    sleep 5
  done
  echo "$2 is up"
}

step_check() {
  log "Sanity checks"
  [ -d "$STAGE/images" ] || { echo "NAS staging dir not found at $STAGE (mount smb://the_vault.local/home)"; exit 1; }
  docker info >/dev/null 2>&1 || { echo "Docker Desktop is not running."; exit 1; }
  local mem_gb; mem_gb=$(( $(docker info --format '{{.MemTotal}}') / 1073741824 ))
  [ "$mem_gb" -ge 16 ] || echo "WARNING: Docker VM has ${mem_gb} GB; set 20-24 GB in Docker Desktop > Settings > Resources."
  [ -f .env ] && [ -f src/backend/.env ] || { echo "Missing .env / src/backend/.env; extract repo/genizah_search.tar.gz from the NAS."; exit 1; }
  chmod 600 .env src/backend/.env
  cat "$STAGE/MANIFEST.txt"
}

step_images() {
  while read -r id img; do
    if docker image inspect "$img" --format '{{.Id}}' 2>/dev/null | grep -q "$id"; then
      echo "image $img already loaded"; continue
    fi
    log "Loading $img"
    gunzip -c "$STAGE/images/$(echo "$img" | tr '/:' '__').tar.gz" | docker load
  done < "$STAGE/images/manifest.txt"
}

step_create() {
  log "Creating containers (not started) so the named volumes exist"
  $COMPOSE create --no-build
}

step_volumes() {
  for path in hf_home embedding_cache; do
    log "Restoring volume genizah_search_$path"
    gunzip -c "$STAGE/volumes/$path.tar.gz" \
      | docker run --rm -i -v "genizah_search_$path:/v" alpine sh -c 'find /v -mindepth 1 -delete; tar -C /v --strip-components=1 -xf -'
  done
}

step_data() {
  log "Copying data/visualization cache"
  mkdir -p data/visualization logs
  rsync -a "$STAGE/data/visualization/" data/visualization/
}

step_es_repo() {
  log "Copying the ES snapshot repository into backups/elasticsearch"
  mkdir -p backups/elasticsearch
  rsync -a "$STAGE/es_snapshot/" backups/elasticsearch/
}

step_up() {
  log "Starting the stack"
  $COMPOSE up -d --no-build
  $COMPOSE ps
}

step_es() {
  log "Elasticsearch: waiting, restoring the snapshot, creating the backend user + Kibana token"
  $ES_TOOL --wait --restore-snapshot latest --setup-users --kibana-token
  $COMPOSE up -d kibana
}

step_verify() {
  log "Neo4j: fresh graph"
  for _ in $(seq 1 24); do
    docker exec genizah_search-neo4j-1 sh -c 'cypher-shell -u "${NEO4J_AUTH%%/*}" -p "${NEO4J_AUTH#*/}" "RETURN 1"' >/dev/null 2>&1 && break
    sleep 5
  done
  local nodes
  nodes=$(docker exec genizah_search-neo4j-1 sh -c 'cypher-shell -u "${NEO4J_AUTH%%/*}" -p "${NEO4J_AUTH#*/}" --format plain "MATCH (n) RETURN count(n)"' | tail -1)
  echo "Neo4j is up with $nodes nodes (expected 0; rebuilt by historical-document-analysis via bolt://$(hostname -s).local:7681)"

  log "Service health"
  wait_http http://localhost:8001/health "embedding service" 600
  wait_http http://localhost:8000/health "backend" 600
  wait_http http://localhost:3000/ "frontend" 60
  curl -s http://localhost:8001/health; echo
  curl -s http://localhost:8000/health; echo
}

step_cloudflared() {
  log "Installing tunnel credentials into ~/.cloudflared (NOT starting the tunnel)"
  tar -C "$HOME" -xzf "$STAGE/cloudflared/cloudflared.tar.gz"
  chmod 700 "$HOME/.cloudflared"; chmod 600 "$HOME/.cloudflared"/*
  ls -la "$HOME/.cloudflared"
}

for step in $STEPS; do
  "step_$step"
done

log "Bootstrap complete. The stack serves on this MBP; the public tunnel still points at the Studio."
echo "Next: scripts/mbp_migration/cutover_mbp.sh (see docs/MBP_MIGRATION.md)"
