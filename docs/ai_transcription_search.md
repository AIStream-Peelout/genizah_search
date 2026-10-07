# AI-transcription search (Layer 1) — separate full-text search over the two-reader line reads

Status: **built 2026-09-14, not yet cut over.** Backend, loader, frontend and
tests are on the branch; the production side index is still
`genizah_ai_transcriptions_v1` and the search endpoint answers
`available: false` against it until the user-run cutover below.

This is Layer 1 of a three-layer plan. Layers 2 (folding agreed text into the
merged catalogue index) and 3 (semantic search over reads, hybrid fusion) are
**out of scope** here; the ordering rule for them is recorded at the end so
they cannot quietly change the catalogue ranking.

## What it does

`POST /search-ai-transcriptions` `{query, limit, offset, include_unconfirmed, vlm_model?}`
searches the AI line reads (Kraken × Qwen3-VL, see
`docs/planned_features/ai-transcriptions.md`) and returns one card per image
with the line(s) that matched, highlighted, with their evidence box, so the
`/read` viewer can jump straight to the line (`/read?doc=…&image=…&line=…`).

- Filters exactly as the viewer: `surfaced = true`, `published = true`.
- **Default: agreed lines only.** The nested query is restricted to
  `status = agreed`, so every hit resolves to a line two readers agreed on;
  `include_unconfirmed = true` widens to all lines (`text_all`) and marks
  single-reader lines as such on the card.
- Ranking inside a line: exact phrase (×3) → all terms, at most one edit per
  term and none for tokens under four letters (`AUTO:4,99`; plain `AUTO` and
  even a flat `1` are too loose for 3-letter Hebrew tokens) → all terms on the
  `.confusable` subfield at ×0.25, recall only. A doc-level phrase match on
  `text_agreed` / `text_all` adds ×2 for ordering but can never create a hit.
- **One card per image.** Records of the same `(doc_id, image_index)` by
  different checkpoints are collapsed on `read_key`; the best-scoring record
  is shown, the newer one wins ties. The `models` facet in the response counts
  records per checkpoint and `vlm_model` pins one. (Chosen over "newest
  checkpoint wins": for a search the record that reads the query text more
  cleanly is the more useful one, and the facet keeps the choice visible.)
- Cards are labelled with `shelf_mark` / `description` (the merged index has
  no `title` field) from one `mget` against `ELASTICSEARCH_INDEX` per page.
- The endpoint is separate from `/search`, `/search-keyword`, `/search-hybrid`.
  Nothing in `search_service.py`'s `multi_match` lists changed, and the
  `transcription_source` filter (`search_service.py`) is untouched.
- Disabled (`AI_TRANSCRIPTIONS_ENABLED=false`) → `{enabled: false, available: false}`,
  the same shape as `GET /ai-transcriptions/{doc_id}`. Missing index, or an
  index without the v2 mapping → `{enabled: true, available: false, message}`.
  It is therefore safe to deploy the backend before the index switch.

Frontend: a fifth tab on the search page, **🤖 AI transcriptions (beta)**,
with a Hebrew input, the *include unconfirmed lines* checkbox and the same
caveat wording as the viewer (`src/frontend/src/read/caveat.js` is shared by
both). Result cards show shelf mark + catalogue description, the matched
line(s) with `<mark>` highlights (rendered as text nodes, never as HTML),
the viewer's agreement badge, and deep links. `ReadFragment.jsx` honours
`?line=N`: it zooms the stage to that line's box, pulses the box and the
text row, and scrolls the text panel to it (unconfirmed lines have a box too
since rule v2, so they get the same treatment).

## Index v2 mapping (`genizah_ai_transcriptions_v2`)

`INDEX_MAPPING` in `src/backend/ai_transcriptions.py`. Differences from v1:

| | v1 | v2 | why |
|---|---|---|---|
| `ai_read.lines` | `object, enabled: false` | `nested` (`index`, `text`, `status` keyword, `agreement` float, `bbox` integer; `htr_text`/`htr_fragments` kept in `_source`, not indexed) | a hit must resolve to *the* line that matched, with its box; `status` must be filterable per line |
| `text` | plain `text` | removed | replaced by the two fields below; the strict mapping means v2 records cannot be written into v1 by accident |
| `text_agreed` | — | Hebrew-folded `text` | join of agreed lines; the default doc-level field. A record with no agreed line has `""` and is never surfaced anyway |
| `text_all` | — | Hebrew-folded `text` | join of all lines (= v1 `text`); used only when `include_unconfirmed` |
| `read_key` | — | `keyword` | `<doc_id>__<image_index>`, the collapse key |
| analysis | none | `hebrew_fold`, `hebrew_confusable` | see below |

All three derived fields are computed in `AiTranscriptionRecord._derive`
(always recomputed from the lines, so they cannot drift), and the loader
reports the `text_agreed` non-empty count.

### Hebrew analyzer

`hebrew_fold` = char_filter `pattern_replace` `[֑-ׇ]` → "" (points,
cantillation, other Hebrew combining marks) → `mapping` ך→כ ם→מ ן→נ ף→פ ץ→צ →
`standard` tokenizer → `lowercase`. Applied to `text_agreed`, `text_all` and
`ai_read.lines.text` at index *and* query time, so a pointed query matches an
unpointed read and either form of a final letter matches.

`.confusable` subfield (`hebrew_confusable`) adds one more `mapping` filter
after the final-form fold: ר→ד, כ→ב, ס→מ — the letter pairs both readers were
seen to confuse in the 2026-09-08 probe (ד/ר, ב/כ, ם/ס; samekh folds to
medial mem because final mem already folded there). It is used only as the
lowest-boosted clause, never alone, so it widens recall without promoting a
confused reading over an exact one.

Both tables live once in Python (`FINAL_FORMS`, `CONFUSABLES`) and generate
the ES rules; `fold_hebrew()` mirrors them for tests. Verified against ES
8.18.2 (`_analyze`): `בְּרֵאשִׁית בָּרָא אֱלֹהִים דבר כסף` →
`hebrew_fold` `[בראשית, ברא, אלהימ, דבר, כספ]`, `hebrew_confusable`
`[בדאשית, בדא, אלהימ, דבד, במפ]`.

## Loader

`scripts/load_ai_transcriptions.py` now targets `SEARCH_INDEX`
(`genizah_ai_transcriptions_v2`) by default; `--index` overrides. Dry run
remains the default; `--apply` creates the index with the v2 mapping if
missing and upserts (idempotent, keyed `(doc_id, image_index, vlm_model)`).
`--es-url/--es-user/--es-password` point the load at another cluster (used
for the local experimentation container during development; without them
the target is `ELASTICSEARCH_HOST`, i.e. **prod**).

Dry run of 2026-09-14 (both pipeline files, sibling repo):

| file | records | invalid | surfaced | lines / agreed | `text_agreed` non-empty (surfaced+published) |
|---|---|---|---|---|---|
| `ai_reads_qwen3-vl-8b-heb-v21b-step1200.jsonl` | 1,558 | 0 | 533 | 31,629 / 5,694 | 749 (533) |
| `ai_reads_qwen3-vl-8b-heb-v20a-step1800.jsonl` | 6,065 | 0 | 698 | 87,688 / 5,599 | 1,274 (698) |

(The v21b file is still growing; re-running the loader is the append path.)

## Cutover — every step is user-run

1. **Load v2 on prod** (production data write; v1 is not touched):
   ```bash
   PYTHONPATH=. .venv/bin/python scripts/load_ai_transcriptions.py --input ../historical-document-analysis/src/datasets/raw_data/cairo_genizah/ai_reads/ai_reads_qwen3-vl-8b-heb-v20a-step1800.jsonl --apply
   ```
   ```bash
   PYTHONPATH=. .venv/bin/python scripts/load_ai_transcriptions.py --input ../historical-document-analysis/src/datasets/raw_data/cairo_genizah/ai_reads/ai_reads_qwen3-vl-8b-heb-v21b-step1200.jsonl --apply
   ```
   Expect `index created: genizah_ai_transcriptions_v2` on the first run and
   `0 errors` on both. Until step 3 nothing serves from it.
2. **Point the backend at v2:** set `AI_TRANSCRIPTIONS_INDEX=genizah_ai_transcriptions_v2`
   for the backend container (compose `environment` override, the same way
   `ELASTICSEARCH_INDEX` is set; the baked `src/backend/.env` alone went stale
   before). The viewer, the `transcription_source` filter and the new search
   all read this one variable, so they switch together.
3. **Rebuild + recreate backend and frontend** — this is a production deploy
   (`docs/SHARED_RUNTIME.md`): `docker compose build backend frontend` then
   `docker compose up -d --force-recreate backend frontend`.
4. Check: `POST https://api.cairogenizah.ai/search-ai-transcriptions {"query":"אללה"}`
   returns `available: true` with hits; `GET /ai-transcriptions/<doc>` still
   lists reads; a keyword search with `transcription_source: ai` still filters.
5. Later, once nothing points at v1, delete `genizah_ai_transcriptions_v1`
   (explicitly, by hand; not part of this change).

Rollback: set `AI_TRANSCRIPTIONS_INDEX` back to `genizah_ai_transcriptions_v1`
and recreate the backend; the search tab then shows "not available yet".

## Ordering rule for later layers (Layers 2 and 3)

- **A catalogue match must always outrank an AI-only match.** Whatever fusion
  Layer 2/3 uses, a document that matched on scholar transcription, description
  or any catalogue field keeps its rank; AI text may only *add* documents
  **below** the last catalogue hit, never reorder the ones above it.
- AI text never enters the default `multi_match` field lists of
  `search_service.py`; if Layer 2 denormalises `text_agreed` / `has_ai_read`
  into the merged index, it is queried as a separate, lower-priority clause
  (e.g. a second pass appended after the catalogue results), and only
  `text_agreed`, never `text_all`.
- Unconfirmed lines stay opt-in everywhere.

## Verification done on this branch (2026-09-14)

- Backend: `src/backend/tests/test_ai_transcriptions.py` (record derivation,
  mapping/analyzer tables, query builder, service with a fake ES, endpoint via
  `TestClient` with the ES client mocked) and the untouched
  `test_transcription_source_filter.py`.
- End-to-end against the **local** experimentation Elasticsearch (which holds
  mirrors of `genizah_merged_v5` and `genizah_ai_transcriptions_v1`): both
  files loaded into a local `genizah_ai_transcriptions_v2` with
  `--apply --es-url http://localhost:9200`; a probe over six frequent tokens
  confirmed every hit has ≥1 line with a box and a highlight containing the
  folded query token, `include_unconfirmed=false` never returned an
  unconfirmed line across three pages per query, widened totals ≥ strict
  totals, an image read by both checkpoints appears once, the model pin
  holds, and a pointed query returns the same total as the bare one.
  Dev backend on :8010 (`ELASTICSEARCH_SCHEME=http`, local host/creds,
  `AI_TRANSCRIPTIONS_INDEX=genizah_ai_transcriptions_v2`) served the endpoint,
  the existing viewer endpoints and a keyword search with the AI filter.
