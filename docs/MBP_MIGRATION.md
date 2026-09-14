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
| Elasticsearch (**this container is prod**: the tunnel maps `elastic.cairogenizah.ai` → `localhost:9200`) | 7.9 GB volume, 21 indexes | **every non-system index** (~7.5 GB; serving ones first) | reindex-from-remote over the LAN |
| embedding weights (`hf_home`, 1.2 GB) | volume | same volume | tar stream |
| `data/visualization` (2 GB cache) | bind mount | bind mount | rsync |
| `.env`, `src/backend/.env` | | same | rsync (mode 600) |
| cloudflared tunnel `adc9e5c6…` | two `cloudflared tunnel run` processes in terminals | LaunchAgent, second connector on the same tunnel | rsync `~/.cloudflared`, `cutover_mbp.sh` |
| LM Studio | stays | reached at `http://192.168.8.120:1234` via `docker-compose.mbp.yml` | LM Studio must have **Serve on Local Network** on |

Not copied: the Neo4j data (by decision), Kibana data, Ollama. To skip ES indexes
use `mirror_es_indexes.py --exclude …`; the Studio's ES volume stays intact either way.

## Prerequisites on the MBP (manual, once)

1. **Remote Login on**: System Settings → General → Sharing → Remote Login.
2. **SSH key from the Studio**, run in a Studio terminal (prompts for the MBP password once):
   ```bash
   ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 && ssh-copy-id isaac@isaacs-MacBook-Pro.local
   ```
3. **Docker Desktop** installed, running, **20–24 GB** in Settings → Resources, and
   "Start Docker Desktop when you sign in" enabled.
4. **Never sleep**: System Settings → Energy → prevent sleep when display off / on power;
   auto-login for the user so the LaunchAgent tunnel comes back after a reboot.
5. On the **Studio**, LM Studio → Developer tab → server settings → **Serve on Local
   Network** (it came back loopback-only after the 2026-09-14 crash; verify with
   `curl -s http://192.168.8.120:1234/v1/models | head -c 200` from the MBP).

## Run

On the Studio (read-only for prod; ~15 GB streams over the LAN, allow 20–40 min on Wi-Fi):

```bash
scripts/mbp_migration/export_to_mbp.sh
```

It runs, in order: `precheck repo images create volumes cloudflared bootstrap`.
The ES mirror is the slow part (~7.5 GB through reindex-from-remote; the serving
indexes land first, so the site can be checked while `merged_v2..v4` and the old
text indexes are still streaming).
Re-run any subset with `STEPS="volumes bootstrap" scripts/mbp_migration/export_to_mbp.sh`.
The last step runs `bootstrap_mbp.sh` on the MBP, which starts the stack with
`docker-compose.mbp.yml`, creates the backend's ES user, mirrors the indexes,
mints a Kibana token, checks Neo4j answers (with 0 nodes) and `/health` on 8000/8001/3000.

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
still holds everything). Nothing on the Studio was modified by the export.

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
