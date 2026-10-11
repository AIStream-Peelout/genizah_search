# Semantic search beta (v10) — runbook

> **v10 replaced v9 on 2026-10-11.** The v9 beta (`genizah_merged_v9`, revision
> `dbfab73f…`, text representation `semantic_v3`) was abandoned after its festival-query
> regression (`docs/handoffs/2026-10-08_v9_holiday_regression/`). This file keeps its old
> name, and the beta embedder keeps its v9-era names (`embedding-v9` service, profile
> `v9beta`, `.env.v9beta`, volumes `embedding_v9_cache` / `hf_home_v9`), so links, URLs and
> volumes stay put. Requests to `genizah_merged_v9` now get a 409: no running embedder
> serves its revision.

**What it is.** An opt-in "v10 beta" semantic search over `genizah_merged_v10`, which was
embedded with the fine-tuned private model
`isaacmg/genizah-embed-qwen3-0.6b@af0252255d3e9b8b8432e09de7c5f0689d7ada34` (tag
`v10-lora-s0211`, text representation `semantic_v4`). v10 has 79,524 records. 65,522 of
them have a vector. The other 14,002 have `semantic_eligible: false` and no vector, so they
never reach a cosine `script_score`; semantic search never returns them, and hybrid search
finds them only through its keyword leg.

The v10 embedder was tuned mainly on fragment descriptions and their concepts, so v10 ships
the way v9 did: as the opt-in beta only.

**What does not change.** Everything except this one opt-in path stays on v8:

- The default index stays `genizah_merged_v8`.
- The default embedder stays `Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
  (the `embedding` service, `:8001`).
- Chat/RAG retrieval and the bibliography (`bibliography_text_only_0.7`) keep their indices
  and the stock embedder. Keyword, shelfmark and hybrid search keep their default index.
- The startup canary gate still checks v8 only.

The production `embedding` service, its volumes (`embedding_cache`, `hf_home`) and its
model are untouched.

## v10 index contract

The beta contract check reads the v10 mapping `_meta`:

| Key | Value |
|---|---|
| `embedding_model` / `embedding_model_revision` | `isaacmg/genizah-embed-qwen3-0.6b` / `af0252255d3e9b8b8432e09de7c5f0689d7ada34` |
| `text_representation_version` | `semantic_v4` |
| query prompt | `"Instruct: Given a search query, retrieve relevant passages\nQuery: "` |
| `canary`, `query_canary`, `document_canary` | stored vectors the embedder must reproduce (cosine > 0.99) |

Per document: `semantic_eligible`, `embedded_text_sha256`, and three new text fields
(multilingual analyzer plus `.keyword`): `scholarly_frames` (7,021 docs), `subjects`
(21,818) and `genres` (15,180). Results are **not** collapsed on `embedded_text_sha256`.

## Moving parts

| Piece | Where | Notes |
|---|---|---|
| `embedding-v9` compose service | `docker-compose.yml` (profile `v9beta`) | Same image/code as `embedding`, different model env. Serves the v10 revision `af0252…` since 2026-10-11 (`BETA_EMBEDDING_MODEL_REVISION` default). Own volumes `embedding_v9_cache`, `hf_home_v9`. Port `127.0.0.1:8002` → 8001 (local checks only). The backend reaches it at `http://embedding-v9:8001`. |
| `HF_TOKEN` | repo-root `.env.v9beta` (gitignored by `.env.*`, mode 600) | Read-only token for the private model repo. sentence-transformers 5.6 only picks it up from the process environment. Only `embedding-v9` reads this file (`env_file: .env.v9beta`, optional). Keep the token **out of** the root `.env`: `embedding`, `backend` and `kibana` load that whole file, so a token there would reach them and change their compose config hash (a plain `up -d` would then recreate the production embedder). |
| Backend env | `BETA_SEMANTIC_INDEX` (default `genizah_merged_v10`), `BETA_TEXT_REPRESENTATION_VERSION` (default `semantic_v4`), `BETA_EMBEDDING_SERVICE_URL` (default `http://embedding-v9:8001`), `SEMANTIC_BETA_ENABLED` (code default `true`; `src/backend/.env` sets it) | The backend does **not** `depends_on` `embedding-v9`, so it starts and serves v8 without it. Turning the beta on in `src/backend/.env` is a separate, approved deploy step. |
| `GET /semantic-beta` | backend (`src/backend/semantic_beta.py`) | `{"available", "index", "model", "revision", "text_representation_version", "eligible_documents", "reason"}`. `available` is true only after the beta contract check passed: text version `semantic_v4`, `/health` model@revision equal to the `_meta` one, and all three canaries reproduced with cosine > 0.99. Each canary cosine is logged at INFO (`Semantic beta genizah_merged_v10 <canary> cosine <x> (<mode> mode)`). |
| Shelf-mark routing | `/search` and `/search-hybrid`, beta index only | Shelf marks were masked out of the beta vectors. A query that is only a shelf mark, a shelf-mark prefix or a volume (`T-S 13J20.20`, `T-S 13J`, `T-S NS 324`, `T-S Ar. 30`, `ENA 2558`, `Bodl. MS Heb. d 66`, `Mosseri VII`) is answered by shelf-mark search and marked `"routed_to": "shelfmark"`. The marks under a prefix or volume rank first (`Mosseri VII` lists all `Moss. VII,…` records before any `VIII`; `Mosseri` and `Moss.` are both tried), and the results page like semantic search. When shelf-mark search finds nothing, the query runs as typed. A Roman volume counts only as `I`–`X` with an optional `a` (`Ia`, `IXa`). Bare collection names with no number (`Mosseri collection`, `the Taylor-Schechter collection`) and questions that mention a shelf mark stay semantic. The default index never routes, and the shelf-mark endpoint and chat retrieval keep their shelf-mark search unchanged. |
| Keyword fields | `KEYWORD_SEARCH_FIELDS` (`search_service.py`) | The keyword `multi_match` of `/search-keyword` and of the BM25 leg of `/search-hybrid` also searches `scholarly_frames^2`, `subjects^1.5` and `genres` (boost 1). The fields do not exist on v8, where `multi_match` skips them. |
| Facets (backend only) | `GET /filters?index_name=genizah_merged_v10`; filter keys `genres`, `subjects` | Aggregates `genres.keyword` / `subjects.keyword`; both lists are empty on v8. A `genres` / `subjects` filter is an exact term on the `.keyword` field. No facet UI yet. |
| Frontend | "v10 beta" switch (`AdvancedSearch.jsx`, `semanticBeta.js`) | Sends the normal semantic request with the index set to the beta index that `/semantic-beta` names. Results use the normal shape and may carry `"semantic_beta": true`. The `/fragment/` page reads only the catalogue index (v8), and 6,368 v10 doc ids are not in it, so each beta result carries `"catalogue_page"` (the backend looks the page's ids up in v8) and gets a `/fragment/` permalink only when it is true (`hasFragmentPage` in `utils.js`). A plain click opens every beta result in the modal. If a beta search fails, the page re-reads `/semantic-beta`; when the beta is no longer available the switch hides and the same query is re-run once on the default index. |

The embedder is shared code (`src/embedding_service/`). Weights are pinned to fp32, the
sequence length to 8192, and vectors are L2-normalized. Documents are embedded raw (no
prompt). Queries get the prompt `"Instruct: Given a search query, retrieve relevant passages\nQuery: "`.
`GET :8002/health` reports the model, revision, dtype and query instruction.

## Deploy

> This Mac is production (`docs/SHARED_RUNTIME.md`). Every compose command takes **both**
> `-f` files. Rebuilding the backend is a production deploy, so get explicit approval first.
> So is setting `SEMANTIC_BETA_ENABLED=true` in `src/backend/.env` (step 2 below).
> `src/backend/.env` holds `SEMANTIC_BETA_ENABLED=false` until then, and the backend loads it,
> so without that step `/semantic-beta` reports `available: false`.

The beta embedder already serves the v10 revision (commit `9177d06`; canaries against the
v10 `_meta` verified at cosine 1.000000). Check before deploying:

```bash
curl -s http://127.0.0.1:8002/health    # model isaacmg/genizah-embed-qwen3-0.6b, revision af0252255d3e…, dtype torch.float32
```

If it still reports `dbfab73…`, it is running the v9 revision: rebuild/recreate only that
service (`--profile v9beta ... up -d --no-deps embedding-v9`) and wait for
`"model_loaded":true` before going on.

```bash
cd ~/Documents/GitHub/terraform_gcp_search

# 0. Rollback tags for the images that are running now
docker tag genizah_search-backend:latest  genizah_search-backend:rollback-pre-v10beta
docker tag genizah_search-frontend:latest genizah_search-frontend:rollback-pre-v10beta

# 1. Safe pre-deploy check on the dev backend (uvicorn on :8010, not production).
#    It runs on the host, so point it at the published beta port.
#    SEMANTIC_BETA_ENABLED=true overrides the false in src/backend/.env (load_dotenv never
#    overrides a variable that is already set) for this process only.
SEMANTIC_BETA_ENABLED=true BETA_EMBEDDING_SERVICE_URL=http://127.0.0.1:8002 PYTHONPATH=. .venv/bin/python scripts/dev_backend_local.py
#   in another shell (allow ~15 s for the background contract check):
curl -s http://127.0.0.1:8010/semantic-beta   # expect "available": true, "index": "genizah_merged_v10", "text_representation_version": "semantic_v4"

# 2. Turn the beta on (needs approval): in src/backend/.env change the line
#    SEMANTIC_BETA_ENABLED=false to SEMANTIC_BETA_ENABLED=true (edit the file, do not append a
#    second line). The recreate in step 3 picks it up.
grep -c '^SEMANTIC_BETA_ENABLED=true$' src/backend/.env   # must print 1

# 3. Production deploy of the backend and frontend (needs approval)
docker compose -f docker-compose.yml -f docker-compose.mbp.yml build backend frontend
docker compose -f docker-compose.yml -f docker-compose.mbp.yml up -d --no-deps backend frontend
```

One-time token setup (already done for v9; the same token reads the v10 revision). The
token goes in `.env.v9beta`, never in the root `.env`. To enter it, create the file with
private permissions and paste one line `HF_TOKEN=hf_...` in an editor (not on the command
line, so it stays out of shell history):

```bash
cd ~/Documents/GitHub/terraform_gcp_search
(umask 077; touch .env.v9beta); chmod 600 .env.v9beta
open -e .env.v9beta                 # paste HF_TOKEN=hf_... on its own line, save
grep -c '^HF_TOKEN=.' .env.v9beta   # must print 1 (never print the value)
grep -c '^HF_TOKEN=' .env           # must print 0: the shared root .env must not hold it
```

If the logs show `Application startup failed` or a 401/`RepositoryNotFound` error, `HF_TOKEN`
is wrong, empty, expired or lacks read access to `isaacmg/genizah-embed-qwen3-0.6b`. Fix
`.env.v9beta` and recreate `embedding-v9`. Do not leave the container restart-looping:
`restart: unless-stopped` restarts it forever, including after Docker Desktop restarts.

Pass `--profile v9beta` **only** for commands that name `embedding-v9`. A plain
`up -d` leaves `embedding-v9` alone: it does not start it and does not stop it.

## Check

```bash
curl -s http://localhost:8000/semantic-beta            # after step 2: "available": true, index genizah_merged_v10
curl -s https://api.cairogenizah.ai/semantic-beta
curl -s http://localhost:8000/health                   # the v8 canary gate is unchanged
docker logs --tail 400 genizah_search-backend-1 2>&1 | grep 'Semantic beta'   # the three canary cosines
docker logs --tail 50 genizah_search-embedding-v9-1
```

If `available` is false, `reason` says why. The usual causes are:

- `embedding-v9` is down or still loading.
- The model or revision does not match the v10 `_meta` (for example the service still runs
  the v9 revision `dbfab73…`).
- The index records another `text_representation_version` than `semantic_v4`.
- A canary cosine is below the threshold.

## Turn it off

Pick one. v8 search is unaffected in every case.

- **Hide the beta but keep the container:** set `SEMANTIC_BETA_ENABLED=false` in
  `src/backend/.env`, then recreate the backend with
  `docker compose -f docker-compose.yml -f docker-compose.mbp.yml up -d --no-deps backend`.
  `/semantic-beta` then reports `available: false`. Setting it in the root `.env` does not
  work: compose loads `.env` first and `src/backend/.env` second, so the later file wins.
- **Stop the embedder:**
  `docker compose -f docker-compose.yml -f docker-compose.mbp.yml --profile v9beta stop embedding-v9`.
  `/semantic-beta` reports `available: false` once the backend notices that the embedder is unreachable.
- Stop it before freeing memory or disk. Removing its volumes later (`embedding_v9_cache`,
  `hf_home_v9`) is safe because they hold beta data only. **Never** remove `embedding_cache`
  or `hf_home`.

## Roll back

```bash
docker tag genizah_search-backend:rollback-pre-v10beta  genizah_search-backend:latest
docker tag genizah_search-frontend:rollback-pre-v10beta genizah_search-frontend:latest
docker compose -f docker-compose.yml -f docker-compose.mbp.yml up -d --no-deps backend frontend
```

Older rollback tags (`rollback-pre-v9beta`, `rollback-pre-fragments`) may also exist.
