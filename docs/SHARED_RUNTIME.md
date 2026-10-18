# Shared Runtime — READ BEFORE RUNNING ANYTHING

**This Mac is production.** `api.cairogenizah.ai` is served from the containers and
processes described below, through a `cloudflared` tunnel running on this machine.
There is no separate prod host. Stopping, rebuilding, pruning, or restarting any of
this takes the live site down, and other long-running jobs (model evals, checkpoint
audits, re-indexing) share the same machine.

Any agent working in this repo should treat everything below as **live shared
infrastructure owned by the user**, not as disposable local dev state.

---

## 1. Docker containers (`docker-compose.yml`)

Compose project name is `genizah_search`, so container names are `genizah_search-<service>-1`.

| Service | Container | Host port → container | What it is |
|---|---|---|---|
| `backend` | `genizah_search-backend-1` | **8000** → 8000 | FastAPI/uvicorn API. **This is production** — the cloudflared tunnel points `api.cairogenizah.ai` at it. |
| `frontend` | `genizah_search-frontend-1` | **3000** → 80 | nginx serving the React production build. Built with `REACT_APP_API_URL=https://api.cairogenizah.ai`, so it talks to **prod**, not localhost:8000. |
| `embedding` | `genizah_search-embedding-1` | **8001** → 8001 | Query-embedding service for the default search (`genizah_merged_v8`), the bibliography and chat retrieval. Pinned to `Qwen/Qwen3-Embedding-0.6B` at a fixed revision — the pin is part of the embedding contract, do not "upgrade" it. |
| `embedding-v9` | `genizah_search-embedding-v9-1` | **127.0.0.1:8002** → 8001 | Opt-in semantic **beta** embedder (profile `v9beta`, only started when named; see `docs/v9_semantic_beta.md`). Despite its name it serves the **v10** revision `isaacmg/genizah-embed-qwen3-0.6b@af0252255d3e9b8b8432e09de7c5f0689d7ada34`, which built `genizah_merged_v10`; the v9 revision was abandoned. Only the beta uses it; the backend starts and serves the default without it. Own volumes `embedding_v9_cache`, `hf_home_v9` (beta data only). |
| `neo4j` | `genizah_search-neo4j-1` | **7681** → 7687 (bolt), **7475** → 7474 (browser) | Knowledge graph. Note the non-default host ports. |
| `elasticsearch` | `genizah_search-elasticsearch-1` | **9200** → 9200 | Local ES 8.18.2, single-node. |
| `kibana` | `genizah_search-kibana-1` | **5601** → 5601 | Kibana for the local ES. |

**The local ES container IS the production search index.** The backend queries
`elastic.cairogenizah.ai:443` (indexes `genizah_merged_v5`,
`bibliography_text_only_0.7`, `genizah_ai_transcriptions_v*`), and the cloudflared
ingress on this machine maps that hostname to `localhost:9200`, i.e. this container
(verified 2026-09-14: same cluster UUID). Every query round-trips through Cloudflare
and back. Destroying `elasticsearch_data` takes search down; treat it like `neo4j_data`.

**Migration in progress (2026-09-14):** this whole stack is being moved to the M3
MacBook Pro — see `docs/MBP_MIGRATION.md`.

### Named volumes (destroying these costs hours to days)

`neo4j_data`, `elasticsearch_data`, `hf_home`, `embedding_cache`, `kibana_data`.
`hf_home` / `embedding_cache` hold downloaded model weights; the DB volumes hold
imported graph and index data that is expensive to rebuild.

---

## 2. Host processes (not in Docker)

| Process | Port | Role |
|---|---|---|
| **LM Studio** | `127.0.0.1:1234` | The production LLM server. Backend reaches it at `host.docker.internal:1234`. |
| **cloudflared** | — | Tunnel mapping `api.cairogenizah.ai` → `localhost:8000`. Kill it and prod goes dark. |
| **tunnel-watchdog** (LaunchAgent) | — | Every minute: restarts cloudflared, and **force-restarts Docker Desktop / a hung backend container** when the stack is down. Pause it with the `MAINTENANCE` file (§4). See `docs/MBP_MIGRATION.md`. |
| Ollama | `11434` | Legacy path (`ollama_rag_service.py`). Not the live chat backend. |

---

## 3. LM Studio models

Configured in `src/backend/.env`:

| Role | Model | Notes |
|---|---|---|
| `ROUTER_MODEL` | `qwen/qwen3-4b-2507` | Query routing / planning / follow-up resolution. ctx 32768. |
| `SYNTHESIS_MODEL` | `qwen/qwen3.8-27b` | Answer synthesis. ctx 32768. |
| `VERIFICATION_MODEL` | `qwen/qwen3.6-35b-a3b` | Citation/claim verification. |

Code-level defaults in `lms_agentic_search.py` differ (`c4ai-command-r-v01` for
synthesis); the `.env` values win. Don't "fix" the code default to match.

### LM Studio rules

- **Do not eject or unload models.** The two chat models above are *manually pinned*
  (`lms load`) so they survive JIT churn. Pins do not persist across LM Studio
  restarts. Unloading them → prod chat returns 500s.
- **Never eject `qwen3-vl-8b-heb-v19a-step1300` or `qwen3-vl-8b-heb-v20a-step1800`.**
  They belong to the checkpoint-audit and offline-transcription work in the sibling
  `historical-document-analysis` project (they replaced `v18b-step700`, 2026-09).
- **AI transcriptions are never computed by this backend.** The site only reads
  finished records from the `AI_TRANSCRIPTIONS_INDEX` side index (default
  `genizah_ai_transcriptions_v1`); loading them is `scripts/load_ai_transcriptions.py
  --apply`, which is a production data write. See
  `docs/planned_features/ai-transcriptions.md`.
- **Never load `google/gemma-4-31b-qat`.** It has crashed the entire LM Studio server.
- **Don't load additional large models casually.** LM Studio keeps several models
  resident (~54 GB observed on this 128 GB machine); memory pressure makes its
  guardrails evict the pinned prod models.
- Every request carries a TTL (`LM_STUDIO_MODEL_TTL`, default 3600s) so idle
  JIT-loaded models self-unload. There is **no HTTP load/unload endpoint** — loading
  only happens via JIT on an inference request.

---

## 4. Do not run these without asking

```
docker compose down                  # takes prod offline
docker compose restart / up --force-recreate   # same, plus a live deploy
docker system prune / volume rm / docker rm    # can destroy graph + index data
docker kill, Docker Desktop restart
pkill cloudflared                    # kills the prod tunnel
lms unload --all  / ejecting models in the LM Studio UI
```

**Stopping things on purpose:** the tunnel-watchdog LaunchAgent relaunches a
stopped or hung Docker Desktop within ~3 minutes. Before quitting Docker,
restarting its VM, or any `compose down`, run
`touch ~/Library/Application\ Support/cairogenizah/MAINTENANCE` and remove the
file when done.

Also note: `docker compose build backend && up -d backend` **is a production
deploy**. The backend container has no source volume mount (its `--reload` is moot),
so backend code changes only reach prod via a rebuild. Get explicit approval first.

## 5. Safe ways to work

- **Test backend changes without touching prod:** `scripts/dev_backend_local.py`
  runs a venv uvicorn on **:8010** against the same compose services.
  (`.claude/launch.json` → `backend-local`.)
- **Frontend:** there is no Node/npm on the host; build only via
  `docker compose build frontend`. Rebuilding the frontend is lower-risk than the
  backend but still recreates a container — ask first.
- **Reading state is fine:** `docker ps`, `docker logs genizah_search-backend-1`
  (these are the prod chat logs), `curl localhost:8000/health`,
  `curl http://192.168.8.122:1234/api/v0/models` (LM Studio lives on the Mac Studio).
- **Is chat's LLM up?** `GET https://api.cairogenizah.ai/lms_uptime` — 200 only while
  LM Studio answers and the router/synthesis/verification models are loaded, else 503.
  Reads the model list only, never runs or loads a model; meant for the external uptime
  monitor. Send `X-API-Key: $CHAT_API_KEY` to also see every resident model.
- **Never modify production databases or delete data** without explicit user
  confirmation (see `AGENTS.md`).
