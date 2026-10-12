# Prompt for Claude Code on the M3 MacBook Pro

Paste everything below the line into a Claude Code session on the MBP (any
directory; it creates `~/Documents/GitHub/genizah_search`).

---

I'm moving the production serving stack of the Cairo Genizah search site
(`cairogenizah.ai`, `api.cairogenizah.ai`, `elastic.cairogenizah.ai`) from my Mac
Studio to this MacBook Pro. The Studio has already staged everything on our NAS;
this machine only pulls from the NAS. Nothing on this machine is production yet, so
you can run Docker freely here. Do not start `cloudflared` until I explicitly say
"cut over": the tunnel credentials are in the bundle and starting them makes this
laptop serve the public site.

Steps:

1. Mount the NAS share `smb://the_vault.local/home` (Finder → Go → Connect to
   Server, user `isaac`) so `/Volumes/home/genizah_migration` exists. Read
   `/Volumes/home/genizah_migration/MANIFEST.txt` and `README_MBP.md` (this file).
2. Make sure Docker Desktop is installed and running with **20–24 GB** memory
   (Settings → Resources) and "Start Docker Desktop when you sign in" enabled.
   Check `docker info --format '{{.MemTotal}}'`.
3. Extract the repo (branch `rag-fixes`, includes the two `.env` files, mode 600):
   ```bash
   mkdir -p ~/Documents/GitHub && tar -xzf /Volumes/home/genizah_migration/repo/genizah_search.tar.gz -C ~/Documents/GitHub
   ```
4. Read `docs/MBP_MIGRATION.md` and `scripts/mbp_migration/bootstrap_mbp.sh` in the
   extracted repo, then run the bootstrap from the repo root:
   ```bash
   cd ~/Documents/GitHub/genizah_search && scripts/mbp_migration/bootstrap_mbp.sh
   ```
   It loads the six Docker images, restores the embedding-weight volumes, copies the
   visualization cache and the Elasticsearch snapshot repository, starts the stack
   with `docker-compose.mbp.yml`, restores every index from the snapshot, creates
   the backend's ES user and a Kibana token, checks Neo4j is up (empty on purpose:
   the knowledge graph is rebuilt from scratch on this machine by the
   historical-document-analysis pipeline running on the Studio), and probes
   `/health` on ports 8000, 8001 and 3000. Re-run a failed subset with
   `STEPS="es verify" scripts/mbp_migration/bootstrap_mbp.sh`.
5. Then verify for real:
   - `curl -s localhost:8000/health` and `curl -s localhost:8001/health` are 200.
   - ES has the same indexes as the manifest lists (`es_migrate.py --verify` compares
     doc counts against the Studio at `http://192.168.8.120:9200` over the LAN if it
     is reachable).
   - The backend can reach LM Studio on the Studio: `curl -s http://192.168.8.120:1234/v1/models | head -c 300`.
     If that fails, tell me; "Serve on Local Network" must be on in LM Studio there.
   - A search request against `localhost:8000` returns results (the backend still
     talks to `elastic.cairogenizah.ai`, which is the Studio until cutover, so
     search working here proves the tunnel path; the local ES is exercised after cutover).
6. Set the machine up as an always-on server: System Settings → Energy → never sleep
   on power, and enable auto-login so the tunnel LaunchAgent comes back after a reboot.
   Report anything that needs my password.
7. Report back with: `docker compose ps`, ES index list with doc counts, the three
   health responses, and anything that failed. Stop there and wait for me to say
   "cut over" before running `scripts/mbp_migration/cutover_mbp.sh`.
