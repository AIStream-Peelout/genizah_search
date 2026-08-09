# Feature prompt: clickable scholar names with info panel

Use this prompt in a fresh session/branch of `genizah_search`.

---

Implement clickable scholar/author names in chat answers that open an
information panel, mirroring the map's scholar detail panels and the existing
book-title popovers.

## Backend

1. New endpoint `GET /scholar-info?name=...` in `src/backend/app.py`:
   - Resolve via `neo4j_service.find_scholars` (already connectivity-ranked)
     and fetch `neo4j_service.get_scholar_detail(name)` (exists; used by
     `/map/scholar/{name}`): publications, referenced fragments, places.
   - Merge with the bibliography index: distinct works actually indexed in ES
     for this author (reuse `_person_name_variants` from
     `src/backend/search_bibliography.py` — it covers initials/inverted
     forms), with per-work page counts, so the panel can say which works are
     readable in-app vs. catalog-only.
   - Include `bio_summary` from the Scholar node when present (the
     historical-document-analysis repo is adding curated bios —
     see its docs/KG_ENHANCEMENTS_TODO.md item 1; the field may not exist yet,
     return null gracefully).
   - Return WorldCat/Scholar locator links per work (reuse the /book-info
     title-cleaning logic — consider extracting it to a helper).

2. Scholar link annotation at FINALIZE time only (never before verification,
   same rule as shelfmark linkify) in `_finalize_response_node` in
   `src/backend/lms_agentic_search.py`:
   - The pipeline already knows the scholars in play: resolved graph scholars
     (`state["graph_results"][i]["scholar"]["name"]` + its `resolution`) and
     bibliography author fields. Build a name→canonical map including surname
     and initials variants.
   - Wrap first occurrences in the answer as `[<as written>](scholar:<canonical name>)`.
     Do not wrap text inside existing markdown links or flag markers
     (`⟦flag:N⟧…⟦/flag⟧`). Follow `_linkify_all_shelfmarks` for the
     no-double-link regex pattern.

## Frontend (`src/frontend/src/ChatUI.jsx`)

3. Extend the link parser in `MarkdownText.parseMarkdown` (which already
   special-cases `doc:` URLs) with a `scholar:` scheme → render a
   `ScholarNameSpan` (follow the `BookTitleSpan` pattern: click → fetch
   `/scholar-info` once, session-cache in a Map, popover). Panel contents:
   - bio_summary when present, else "profile from indexed corpus";
   - works list: indexed works clickable (trigger a chat query or bibliography
     search), catalog-only works with WorldCat links;
   - counts: fragments referenced/studied; places (link to map view:
     `navigate('/map')` with the scholar preselected if MapView supports it).
   - "Ask about this scholar" button that inserts a prompt into the chat input.

4. Keep it responsive: the popover pattern must work inside the mobile chat
   overlay (`.mobile-chat-overlay`); use the alignRight logic from
   BookTitleSpan and `max-width: min(360px, 86vw)`.

## Tests

5. Backend: pytest (container: `docker exec genizah_search-backend-1 python3 -m
   pytest /app/src/backend/tests/ ...`) —
   - scholar-link annotation wraps known scholars once, never inside existing
     links/flag markers, never before verification (annotate in finalize only);
   - `/scholar-info` merges graph + ES works and handles unknown names (404 or
     found=false).
   Follow the style of `tests/test_shelfmark_linkification.py`
   (TestShelfmarkResolutionGuarantees) and `tests/test_graph_rag_phase1.py`.

6. The eval runner (`scripts/run_agentic_rag_eval.py`) and judge must strip
   `scholar:` links like any markdown before judging (verify `bounded_evidence`
   / judge input unaffected).

## Deployment notes

Code is baked into images: `docker compose build backend frontend && docker
compose up -d backend frontend`. For fast iteration `docker cp` files into the
running containers (backend runs uvicorn --reload). Backend startup runs an
embedding canary check — expect "canary verified" in logs after restart.
