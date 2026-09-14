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
| `embedding` | `genizah_search-embedding-1` | **8001** → 8001 | Query-embedding service. Pinned to `Qwen/Qwen3-Embedding-0.6B` at a fixed revision — the pin is part of the embedding contract, do not "upgrade" it. |
| `neo4j` | `genizah_search-neo4j-1` | **7681** → 7687 (bolt), **7475** → 7474 (browser) | Knowledge graph. Note the non-default host ports. |
| `elasticsearch` | `genizah_search-elasticsearch-1` | **9200** → 9200 | Local ES 8.18.2, single-node. |
| `kibana` | `genizah_search-kibana-1` | **5601** → 5601 | Kibana for the local ES. |

**Search does not run against the local ES container.** The backend queries remote
`elastic.cairogenizah.ai:443` (indexes `genizah_merged_v4` and
`bibliography_text_only_0.7`). The local ES/Kibana pair is for experimentation and
backups.

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
| Ollama | `11434` | Legacy path (`ollama_rag_service.py`). Not the live chat backend. |

---

## 3. LM Studio models

Configured in `src/backend/.env`:

| Role | Model | Notes |
|---|---|---|
| `ROUTER_MODEL` | `qwen/qwen3-4b-2507` | Query routing / planning / follow-up resolution. ctx 32768. |
| `SYNTHESIS_MODEL` | `qwen/qwen3.6-35b-a3b` | Answer synthesis. ctx 32768. |
| `VERIFICATION_MODEL` | `qwen/qwen3.6-35b-a3b` | Citation/claim verification. |

Code-level defaults in `lms_agentic_search.py` differ (`c4ai-command-r-v01` for
synthesis); the `.env` values win. Don't "fix" the code default to match.

### LM Studio rules

- **Do not eject or unload models.** The two chat models above are *manually pinned*
  (`lms load`) so they survive JIT churn. Pins do not persist across LM Studio
  restarts. Unloading them → prod chat returns 500s.
- **Never eject `qwen3-vl-8b-heb-v18b-step700`.** It belongs to the checkpoint-audit
  work in the sibling `historical-document-analysis` project.
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
  `curl localhost:1234/api/v0/models`.
- **Never modify production databases or delete data** without explicit user
  confirmation (see `AGENTS.md`).
