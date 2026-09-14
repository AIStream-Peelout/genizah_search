#!/bin/bash
# Bring the serving stack up on the M3 MacBook Pro.
#
# Runs ON THE MBP, after export_to_mbp.sh has delivered the images, the
# neo4j/hf_home volumes, the repo (with .env files) and data/visualization.
# Idempotent: safe to re-run after a partial failure.
#
# Steps:
#   1. sanity checks (Docker running, VM memory, env files present)
#   2. docker compose up (MBP override, no build)
#   3. Elasticsearch: create the backend's user, mirror EVERY non-system index
#      from the Studio with reindex-from-remote, verify doc counts
#   4. Kibana: mint a service-account token for this cluster
#   5. verify Neo4j is up (fresh, empty graph), backend/embedding/frontend health
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.mbp.yml"
STUDIO_ES="${STUDIO_ES_HOSTPORT:-192.168.8.120:9200}"
MIRROR="python3 scripts/mbp_migration/mirror_es_indexes.py --source http://$STUDIO_ES --dest http://localhost:9200"

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

# --- 1. checks ---------------------------------------------------------------
log "Sanity checks"
docker info >/dev/null 2>&1 || { echo "Docker Desktop is not running on this machine."; exit 1; }
mem_gb=$(( $(docker info --format '{{.MemTotal}}') / 1073741824 ))
if [ "$mem_gb" -lt 16 ]; then
  echo "WARNING: the Docker VM has ${mem_gb} GB. Set 20-24 GB in Docker Desktop > Settings > Resources, then re-run."
fi
[ -f .env ] && [ -f src/backend/.env ] || { echo "Missing .env or src/backend/.env (export_to_mbp.sh step 'repo')."; exit 1; }

# --- 2. up -------------------------------------------------------------------
log "Starting the stack"
$COMPOSE up -d --no-build
$COMPOSE ps

# --- 3. Elasticsearch ----------------------------------------------------------
log "Elasticsearch: waiting, creating the backend user, mirroring indexes from $STUDIO_ES"
$MIRROR --wait --setup-users
$MIRROR --mirror --verify

# --- 4. Kibana token -------------------------------------------------------------
log "Kibana: minting a service token for this cluster"
$MIRROR --kibana-token
$COMPOSE up -d kibana

# --- 5. verification -------------------------------------------------------------
log "Neo4j: waiting for the fresh graph"
for _ in $(seq 1 24); do
  docker exec genizah_search-neo4j-1 sh -c 'cypher-shell -u "${NEO4J_AUTH%%/*}" -p "${NEO4J_AUTH#*/}" "RETURN 1"' >/dev/null 2>&1 && break
  sleep 5
done
nodes=$(docker exec genizah_search-neo4j-1 sh -c 'cypher-shell -u "${NEO4J_AUTH%%/*}" -p "${NEO4J_AUTH#*/}" --format plain "MATCH (n) RETURN count(n)"' | tail -1)
echo "Neo4j is up with $nodes nodes (expected 0: the KG is rebuilt from scratch by the"
echo "historical-document-analysis pipeline; point its .env at bolt://$(hostname).local:7681)."

log "Service health"
wait_http http://localhost:8001/health "embedding service" 600
wait_http http://localhost:8000/health "backend" 600
wait_http http://localhost:3000/ "frontend" 60
curl -s http://localhost:8001/health; echo
curl -s http://localhost:8000/health; echo

log "Bootstrap complete. The stack is serving on this MBP but the tunnel still points at the Studio."
echo "Next: scripts/mbp_migration/cutover_mbp.sh"
