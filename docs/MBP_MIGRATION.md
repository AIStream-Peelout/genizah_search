# Moving the serving stack to the M3 MacBook Pro

Goal: `cairogenizah.ai` / `api.cairogenizah.ai` / `elastic.cairogenizah.ai` are served
from the always-on M3 MacBook Pro (48 GB) instead of the Mac Studio, so a Studio
crash only takes down chat synthesis (LM Studio stays on the Studio for now).
Design discussed 2026-09-06 (multi-device session); kit written 2026-09-14 after
another Studio crash.

## What moves

| Piece | Studio today | On the MBP | How |
|---|---|---|---|
| backend / embedding / frontend images | built 2026-09-10 (`94d7aee`), 2026-08-09 | same images, no rebuild | `docker save \| ssh docker load` |
| neo4j (71,201 nodes / 175,830 rels on the Studio) | `genizah_search_neo4j_data` | **fresh, empty graph** (decision 2026-09-14), rebuilt by the historical-document-analysis pipeline | nothing copied; see "Rebuilding the KG" below |
| Elasticsearch (**this container is prod**: the tunnel maps `elastic.cairogenizah.ai` → `localhost:9200`) | 7.9 GB volume, 21 indexes | **every non-system index** | ES snapshot → NAS → restore (LAN reindex as fallback) |
| embedding weights (`hf_home`, 1.2 GB) | volume | same volume | `docker cp` tar → NAS |
| `data/visualization` (2 GB cache) | bind mount | bind mount | rsync via NAS |
| `.env`, `src/backend/.env` | | same | inside the repo tarball (mode 600) |
| cloudflared tunnel `adc9e5c6…` | two `cloudflared tunnel run` processes in terminals | LaunchAgent, second connector on the same tunnel | `~/.cloudflared` tarball, `cutover_mbp.sh` |
| LM Studio | stays | reached at `http://192.168.8.120:1234` via `docker-compose.mbp.yml` | LM Studio must have **Serve on Local Network** on |

Not copied: the Neo4j data (by decision), Kibana data, Ollama. To skip ES indexes
the snapshot is all-or-nothing; the Studio's ES volume stays intact either way.

## Transport: the NAS, no SSH

Nothing talks machine-to-machine. The Studio stages a bundle on the NAS
(`/Volumes/home/genizah_migration`, share `smb://the_vault.local/home`) and the MBP
pulls it, driven by its own Claude Code session with the prompt in
`scripts/mbp_migration/MBP_CLAUDE_PROMPT.md` (also copied to the bundle as `README_MBP.md`).

Bundle layout: `images/*.tar.gz` (+ `manifest.txt` with image IDs), `volumes/hf_home.tar.gz`,
`volumes/embedding_cache.tar.gz`, `repo/genizah_search.tar.gz` (working tree incl. `.git`
and both `.env` files), `data/visualization/`, `cloudflared/cloudflared.tar.gz`,
`es_snapshot/` (an ES filesystem snapshot repository), `MANIFEST.txt`.

## Prerequisites

- **Studio**: NAS mounted at `/Volumes/home`. LM Studio → Developer tab → server
  settings → **Serve on Local Network** (it came back loopback-only after the
  2026-09-14 crash; the MBP verifies with `curl http://192.168.8.120:1234/v1/models`).
- **Studio, one-time, needs approval**: the ES snapshot needs `path.repo`, which
  `docker-compose.yml` now sets but only applies on a container recreate:
  `docker compose up -d elasticsearch` (about a minute of search outage while ES
  restarts; the backend and Kibana reconnect on their own). Without it the `es`
  step refuses and the LAN reindex fallback (`es_migrate.py --mirror`) is the alternative.
- **MBP**: Docker Desktop with 20–24 GB, NAS mounted, never sleep + auto-login.

## Run

On the Studio (read-only for prod; ~13 GB of images/volumes plus the ~8 GB snapshot):

```bash
scripts/mbp_migration/export_to_nas.sh
```

Steps: `images volumes repo data cloudflared es manifest`; subset with
`STEPS="es manifest" scripts/mbp_migration/export_to_nas.sh`. Re-runs skip images
already staged with the same ID. The `es` step registers the `migration` repository,
snapshots every non-system index (21 today) with `include_global_state=false`, then
rsyncs `backups/elasticsearch/` to the NAS.

On the MBP, from its Claude Code session (prompt above): extract the repo tarball,
then `scripts/mbp_migration/bootstrap_mbp.sh`, steps
`check images create volumes data es_repo up es verify cloudflared`. It restores the
snapshot with `es_migrate.py --restore-snapshot latest` (replicas 0, skips indexes
already present), creates the backend's ES user and a Kibana token, checks Neo4j is
up with 0 nodes, and probes `/health` on 8000/8001/3000. The tunnel credentials are
installed but the tunnel is **not** started.

Everything above leaves the public site untouched: the MBP backend still talks to
`elastic.cairogenizah.ai`, which the tunnel resolves on whichever machine runs a
connector.

## Rebuilding the KG on the MBP (clean graph)

The Studio's graph accumulated experimental work, so the MBP's Neo4j starts empty
and is rebuilt by the `historical-document-analysis` pipeline
(`docs/kg_pipeline_runbook.md` there; every import script reads `NEO4J_URI` /
`NEO4J_USER` / `NEO4J_PASSWORD` / `NEO4J_DATABASE` from that repo's `.env`).

1. Once `bootstrap_mbp.sh` has passed, change one line in
   `historical-document-analysis/.env` on the Studio:
   `NEO4J_URI=bolt://isaacs-MacBook-Pro.local:7681` (was `bolt://127.0.0.1:7681`;
   same user/password, `NEO4J_DATABASE=neo4j`). The MBP publishes bolt on host
   port 7681 exactly like the Studio did. Scripts run with `PYTHONPATH=.` and the
   sibling venv as before; `load_dotenv` does not override exported vars, so the
   Studio's own Neo4j can still be targeted per-command with `NEO4J_URI=bolt://127.0.0.1:7681`.
2. Run the passes in the runbook's order. The site needs at least: Fragment nodes
   carrying `es_doc_id` (KG↔ES join), the merged-shelfmark import, the PGP person
   relations import (`pgp_person_relations_import.py`), and the geocoded places
   (map People/Places layers). Until those land the KG-backed endpoints on the MBP
   return empty results while search itself works.
3. The Studio's graph stays untouched as a reference; compare counts
   (`MATCH (n) RETURN labels(n), count(*)`) before retiring it.

Sibling ES writes: `ELASTIC_SEARCH_HOST=elastic.cairogenizah.ai` in that `.env`
already follows the tunnel, so after cutover they land on the MBP with no change.
Scripts that default to `localhost:9200` (`page_coverage_report.py`,
`two_reader_lines.py` via `ES_URL`) need `--es-host isaacs-MacBook-Pro.local` /
`ES_URL=http://isaacs-MacBook-Pro.local:9200` after the Studio's stack is stopped.

## Cutover (≈ zero downtime)

1. On the MBP: `scripts/mbp_migration/cutover_mbp.sh` → starts a second connector,
   prints `cloudflared tunnel info` (expect two connectors) and probes
   `api.cairogenizah.ai/health`.
2. Keep both connectors for a few minutes; watch `docker logs -f genizah_search-backend-1`
   on the MBP for real traffic and try a chat query on the site (exercises MBP → Studio LM Studio).
3. **On the Studio, with explicit approval:** `pkill -f 'cloudflared tunnel run'`.
   From then on every request lands on the MBP. Writes to `elastic.cairogenizah.ai`
   (AI-transcription loader, sibling project) now hit the MBP's cluster, so do the
   cutover while no loader is running.
4. Confirm on the MBP: `cloudflared tunnel info <id>` lists only the MBP;
   `curl https://api.cairogenizah.ai/health`; load the site.

## Rollback

Start `cloudflared tunnel run` on the Studio again (its stack is still up and its ES
still holds everything). The export changed nothing on the Studio except adding
`path.repo` and a snapshot repository to its ES.

## After cutover (follow-ups)

- Update `CLAUDE.md` / `docs/SHARED_RUNTIME.md`: the MBP is production; the Studio
  keeps LM Studio + kraken. Sibling scripts that used `localhost:9200` / `localhost:7681`
  on the Studio must point at the MBP (`isaacs-MacBook-Pro.local`).
- Shrink the Studio's Docker Desktop VM (58 GB → ~8 GB) and, with approval, stop the
  Studio's compose stack.
- Point the MBP backend at its own ES (`ELASTICSEARCH_SCHEME=http`, host
  `elasticsearch`) once an image with commit `5ce9f65`+ is deployed; that removes the
  Cloudflare round-trip per query.
- Give the DHCP lease for the MBP a reservation (it is `192.168.8.208` on Wi-Fi today;
  the wired plan was `.11`) and the Studio `.120`/`.10`; adjust `STUDIO_*` in `.env`
  if the Studio's address changes.
- Neo4j and ES on the MBP listen on `0.0.0.0` (7681/7475/9200) behind the router
  NAT only, with the auth from `.env`; that is what lets the Studio's pipeline write
  to them.
- Backups: nightly `neo4j-admin dump` + ES snapshot to the NAS (`/Volumes/home`).
