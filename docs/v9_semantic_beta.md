# v9 semantic search beta — runbook

**What it is.** An opt-in "v9 beta" semantic search over `genizah_merged_v9`, which was
embedded with the fine-tuned private model `isaacmg/genizah-embed-qwen3-0.6b@dbfab73f1e91a6b95eba728541cc5a391a88d933`
(text representation `semantic_v3`). v9 has 78,113 records. 63,652 of them have a vector.
The other 14,461 have `semantic_eligible: false` and no vector, so semantic search never
returns them.

**What does not change.** Everything except this one opt-in path stays on v8:

- The default index stays `genizah_merged_v8`.
- The default embedder stays `Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
- Chat/RAG, bibliography (`bibliography_text_only_0.7`), keyword, shelfmark and hybrid
  search keep their current defaults.
- The startup canary gate still checks v8 only.

The production `embedding` service, its volumes (`embedding_cache`, `hf_home`) and its
model are untouched.

## Moving parts

| Piece | Where | Notes |
|---|---|---|
| `embedding-v9` compose service | `docker-compose.yml` (profile `v9beta`) | Same image/code as `embedding`, different model env. Own volumes `embedding_v9_cache`, `hf_home_v9`. Port `127.0.0.1:8002` → 8001 (local checks only). The backend reaches it at `http://embedding-v9:8001`. |
| `HF_TOKEN` | repo-root `.env.v9beta` (gitignored by `.env.*`, mode 600) | Read-only token for the private model repo. sentence-transformers 5.6 only picks it up from the process environment. Only `embedding-v9` reads this file (`env_file: .env.v9beta`, optional). Keep the token **out of** the root `.env`: `embedding`, `backend` and `kibana` load that whole file, so a token there would reach them and change their compose config hash (a plain `up -d` would then recreate the production embedder). |
| Backend env | `BETA_SEMANTIC_INDEX` (default `genizah_merged_v9`), `BETA_EMBEDDING_SERVICE_URL` (default `http://embedding-v9:8001`), `SEMANTIC_BETA_ENABLED` (default `true`) | The backend does **not** `depends_on` `embedding-v9`, so it starts and serves v8 without it. |
| `GET /semantic-beta` | backend | `{"available", "index", "model", "revision", "text_representation_version", "eligible_documents", "reason"}`. `available` is true only after the beta contract check against the v9 mapping `_meta` canaries has passed. |
| Frontend | "v9 beta" option | Sends the normal semantic request with the index set to the beta index. Results use the normal shape and may carry `"semantic_beta": true`. If a beta search fails, the page re-reads `/semantic-beta`; when the beta is no longer available the switch hides and the same query is re-run once on the default index, so open tabs fall back to v8 without a reload. |

The embedder is shared code (`src/embedding_service/`). Weights are pinned to fp32, the
sequence length to 8192, and vectors are L2-normalized. Documents are embedded raw (no
prompt). Queries get the prompt `"Instruct: Given a search query, retrieve relevant passages\nQuery: "`.
`GET :8002/health` reports the model, revision, dtype and query instruction.

## Deploy

> This Mac is production (`docs/SHARED_RUNTIME.md`). Every compose command takes **both**
> `-f` files. Rebuilding the backend is a production deploy, so get explicit approval first.

One-time token setup. The token goes in `.env.v9beta`, never in the root `.env`. To
enter it, create the file with private permissions and paste one line `HF_TOKEN=hf_...`
in an editor (not on the command line, so it stays out of shell history):

```bash
cd ~/Documents/GitHub/terraform_gcp_search
(umask 077; touch .env.v9beta); chmod 600 .env.v9beta
open -e .env.v9beta                 # paste HF_TOKEN=hf_... on its own line, save
```

If the token was already put in the root `.env`, move it without printing it:

```bash
(umask 077; grep '^HF_TOKEN=' .env > .env.v9beta); chmod 600 .env.v9beta
sed -i '' '/^HF_TOKEN=/d' .env
# The prod embedder and kibana now match their running containers again:
docker compose -f docker-compose.yml -f docker-compose.mbp.yml config --hash embedding,kibana
docker inspect -f '{{.Name}} {{index .Config.Labels "com.docker.compose.config-hash"}}' \
  genizah_search-embedding-1 genizah_search-kibana-1
```

```bash
cd ~/Documents/GitHub/terraform_gcp_search
grep -c '^HF_TOKEN=.' .env.v9beta   # must print 1: the line exists and is not empty (never print the value)
grep -c '^HF_TOKEN=' .env           # must print 0: the shared root .env must not hold it

# 0. Rollback tags for the images that are running now
docker tag genizah_search-backend:latest  genizah_search-backend:rollback-pre-v9beta
docker tag genizah_search-frontend:latest genizah_search-frontend:rollback-pre-v9beta

# 1. Start the beta embedder. On first start it downloads about 1.2 GB into hf_home_v9.
docker compose -f docker-compose.yml -f docker-compose.mbp.yml --profile v9beta build embedding-v9
docker compose -f docker-compose.yml -f docker-compose.mbp.yml --profile v9beta up -d --no-deps embedding-v9
ok=0
for i in $(seq 1 120); do   # up to about 10 minutes for the ~1.2 GB first download
  if curl -sf http://127.0.0.1:8002/health | grep -q '"model_loaded":true'; then ok=1; break; fi
  sleep 5
done
if [ "$ok" -ne 1 ]; then
  docker logs --tail 50 genizah_search-embedding-v9-1
  docker compose -f docker-compose.yml -f docker-compose.mbp.yml --profile v9beta stop embedding-v9
  echo "embedding-v9 never became healthy; stopped it (see logs above)"
fi
curl -s http://127.0.0.1:8002/health    # model isaacmg/genizah-embed-qwen3-0.6b, revision dbfab73f…, dtype torch.float32

# 1b. Safe pre-deploy check on the dev backend (uvicorn on :8010, not production).
#     It runs on the host, so point it at the published beta port.
BETA_EMBEDDING_SERVICE_URL=http://127.0.0.1:8002 PYTHONPATH=. .venv/bin/python scripts/dev_backend_local.py
#   in another shell (allow ~15 s for the background contract check):
curl -s http://127.0.0.1:8010/semantic-beta   # expect "available": true before deploying the backend

# 2. Production deploy of the backend and frontend (needs approval)
docker compose -f docker-compose.yml -f docker-compose.mbp.yml build backend frontend
docker compose -f docker-compose.yml -f docker-compose.mbp.yml up -d --no-deps backend frontend
```

If the logs show `Application startup failed` or a 401/`RepositoryNotFound` error, `HF_TOKEN`
is wrong, empty, expired or lacks read access to `isaacmg/genizah-embed-qwen3-0.6b`. Fix
`.env.v9beta` and rerun step 1. Do not leave the container restart-looping:
`restart: unless-stopped` restarts it forever, including after Docker Desktop restarts.

Pass `--profile v9beta` **only** for commands that name `embedding-v9`. A plain
`up -d` leaves `embedding-v9` alone: it does not start it and does not stop it.

## Check

```bash
curl -s http://localhost:8000/semantic-beta            # expect "available": true
curl -s https://api.cairogenizah.ai/semantic-beta
curl -s http://localhost:8000/health                   # the v8 canary gate is unchanged
docker logs --tail 50 genizah_search-embedding-v9-1
```

If `available` is false, `reason` says why. The usual causes are:

- `embedding-v9` is down or still loading.
- The model or revision does not match the v9 `_meta`.
- A canary cosine is below the threshold.

On 2026-10-08 the canaries were reproduced against the v9 `_meta` with this image in a
throwaway container. All three cosines (`canary` in document mode, `query_canary` in query
mode, `document_canary` in document mode) were 1.000000. The checkpoint stores fp32
weights and has no `default_prompt_name`.

## Turn it off

Pick one. v8 search is unaffected in every case.

- **Hide the beta but keep the container:** set `SEMANTIC_BETA_ENABLED=false` in
  `src/backend/.env` (or the root `.env`), then recreate the backend with
  `docker compose -f docker-compose.yml -f docker-compose.mbp.yml up -d --no-deps backend`.
  `/semantic-beta` then reports `available: false`.
- **Stop the embedder:**
  `docker compose -f docker-compose.yml -f docker-compose.mbp.yml --profile v9beta stop embedding-v9`.
  `/semantic-beta` reports `available: false` once the backend notices that the embedder is unreachable.
- Stop it before freeing memory or disk. Removing its volumes later (`embedding_v9_cache`,
  `hf_home_v9`) is safe because they hold beta data only. **Never** remove `embedding_cache`
  or `hf_home`.

## Roll back

```bash
docker tag genizah_search-backend:rollback-pre-v9beta  genizah_search-backend:latest
docker tag genizah_search-frontend:rollback-pre-v9beta genizah_search-frontend:latest
docker compose -f docker-compose.yml -f docker-compose.mbp.yml up -d --no-deps backend frontend
docker compose -f docker-compose.yml -f docker-compose.mbp.yml --profile v9beta stop embedding-v9
```

The older `rollback-pre-fragments` tags still exist (they come from the fragment-pages
deploy).
