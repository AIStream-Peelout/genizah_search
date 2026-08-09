# Feature prompt: context-reactive chat suggestions from info panels

Use this prompt in a fresh session/branch of `genizah_search`. Best built
AFTER the clickable-scholars feature (docs/planned_features/
clickable-scholars-info-panel.md), which introduces the info-panel concept.

---

When the user opens an info panel (scholar, fragment/document, or book), the
chat's suggested prompts should update to context-relevant suggestions —
e.g. after asking about ketubbot and then opening Friedman's panel, suggest
"Summarize Friedman's work on ketubbot"; after opening a fragment, suggest
"Summarize the literature on this fragment" / "Who wrote about T-S 8J22.22?".

## Search-triggered flow (not only chat)

Opening a fragment from SEARCH results (DocumentModal) must also produce
context: fire a background lookup and, when literature exists, surface a
"Scholarship on this fragment" strip in the modal plus chat suggestions.

New endpoint `GET /fragment-context/{doc_id}` (backend), called async on
modal open (never blocks rendering):

- **Two linkage paths — use both** (verified live 2026-07-27):
  1. KG: `MATCH (f:Fragment {es_doc_id: $doc_id})` →
     `(f)<-[:REFERENCES]-(b:BookArticle)<-[:WROTE]-(s:Scholar)` and
     `(f)<-[:STUDIED]-(s:Scholar)`. Coverage: 44,896/49,074 fragments carry
     es_doc_id (≈2/3 of the 67,467 ES docs resolve); 11,956 fragments have
     ≥1 referencing book. PREREQUISITE (KG repo): create a Neo4j index on
     Fragment.es_doc_id — none exists today, so this lookup would scan.
  2. ES bibliography: pages whose `shelf_marks_mentioned` match the
     fragment's shelf mark (match_phrase on the display form; also try
     `ShelfmarkNormalizer` variants — dialects differ across sources).
     This is page-level and works for fragments with NO KG node
     (e.g. Halper 331 → 42 pages across multiple works).
- Response: `{ kg_fragment_found, books: [...], scholars: [...],
  literature_page_count, literature_works: [{title, authors, page_count}] }`.
- Cache per doc_id in the frontend session; treat all-empty as "no
  suggestions" — never show a lit suggestion the corpus can't back.

Suggestions when literature exists: "Summarize the scholarly literature on
{shelfmark}", "What does {top work author} say about {shelfmark}?"; when only
KG books exist (no ES pages): "Which publications reference {shelfmark}?"
(catalog-level phrasing — don't promise summaries we can't ground).

## Architecture

1. **Active-context state, lifted to app level** in
   `src/frontend/src/react_app.jsx`:
   `activeContext = { type: 'scholar'|'fragment'|'book', id, displayName } | null`.
   Publishers: DocumentModal open (fragment; use shelf_mark as displayName),
   scholar info panel open, book popover open. Clear on close. Pass down to
   `ChatUI` alongside the existing `examplePrompts`.

2. **Template-first suggestions — no LLM call on panel open** (instant, and
   avoids LM Studio contention with the RAG pipeline on this machine):
   - In `ChatUI.jsx`, derive suggestions when `activeContext` changes:
     - scholar: "Summarize {name}'s work{ on {topic}}", "What fragments did
       {name} study?", "Who has written about {name}?"
     - fragment: "Summarize the scholarly literature on {shelfmark}",
       "Who wrote about {shelfmark}?", "What is {shelfmark} about?"
     - book: "What does {title} argue?", "Which fragments does {title} discuss?"
   - `{topic}`: extract 1-3 content keywords from the LAST user query
     (drop stopwords and question words); omit the clause when empty.
   - Fall back to the static examplePrompts when no context is active.

3. **Context-carrying queries**: clicking a suggestion should send a query the
   RAG planner can route precisely. Prefix fragment queries with the shelf
   mark verbatim ("Regarding T-S 8J22.22: ..."), scholar queries with the
   CANONICAL graph name (the panel has it) so `graph_scholar` resolution is
   deterministic.

4. **Optional phase 2 (separate commit, feature-flagged)**: an LLM suggestion
   refresher — debounce ~2s after panel open, call a new backend endpoint that
   asks the ROUTER model (small, always loaded: qwen3-4b) for 3 suggestions
   given {activeContext, last 2 turns}; replace templates when it returns.
   Must fail silently to templates; never block UI; skip entirely while a chat
   request is in flight (one LM Studio queue).

## Backend (only if phase 2)

`POST /chat/suggestions {context, recent_turns}` → uses `_call_llm` with the
router model and a JSON schema (see `_VERIFIER_RESPONSE_FORMAT` pattern in
`src/backend/lms_agentic_search.py`; note the reasoning-channel recovery via
`extract_json_object(..., anchor_key=...)` — reuse `recover_reasoning=True`).

## Tests

- Unit-test the suggestion templating (topic-keyword extraction, each entity
  type, fallback to static prompts) — pure functions, extract them from the
  component for testability.
- Test that suggestion clicks produce context-prefixed queries.
- Frontend has no test harness currently — put the pure logic in
  `src/frontend/src/utils.js` (or a new suggestions.js) and keep components
  thin, so at minimum node-based unit tests are possible later; backend
  endpoint (phase 2) gets pytest coverage like /book-info.

## UX notes

- The "TRY ASKING:" block in ChatUI already renders examplePrompts — reuse it;
  animate/label the swap ("Suggestions for {name}") so users notice the
  context change. Keep it working inside the mobile chat overlay.
- Suggestions must never auto-send; they populate the input.
