# ⚠️ SHARED MACHINE — THIS MAC IS PRODUCTION

`api.cairogenizah.ai` is served from Docker containers + LM Studio + a cloudflared
tunnel **running on this machine**. Read [docs/SHARED_RUNTIME.md](docs/SHARED_RUNTIME.md)
before starting/stopping/rebuilding anything.

Never do these without explicit user approval:
- `docker compose down` / `restart` / `up --force-recreate`, `docker system prune`,
  `docker volume rm`, `docker kill`, or restarting Docker Desktop.
- Killing the `cloudflared` process (it is the prod tunnel).
- Unloading/ejecting LM Studio models, or loading new large ones (evicts the pinned
  prod chat models). Never load `google/gemma-4-31b-qat`; never eject
  `qwen3-vl-8b-heb-v18b-step700`.
- Rebuilding the `backend` image — that is a production deploy.

To test backend changes safely: `scripts/dev_backend_local.py` (uvicorn on :8010).

---
# General Guidelines and Coding Standards
1. All functions should have doc-strings and type hints. Doc strings should follow the Sphinx ReadTheDocs style (eg :param some_param: This is a parameter etc..)
2. When uncertain ask clarifying questions.
3. Before making large changes form a plan and ask user for approval. 
4. Code should be modular and testable. 
5. Refrain from excessive try except blocks. These should only be used in limited circumstances.

## Project Context
See README.md for full project overview, architecture, and data sources.