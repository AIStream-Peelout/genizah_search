# Handoff prompt — production follow-ups from the embedder fine-tune work (2026-10-05)

> Paste everything below the line into a Claude session on the MBP (production checkout, `prod-mbp` branch).
> It is self-contained. Nothing here is urgent: items 1–4 only become **required** when a fine-tuned or
> shelf-mark-free embedder is ready to ship. The training work itself happens elsewhere (the Studio session,
> `genizah_search/evals/embedding_audit/`, report `docs/EMBEDDING_FINETUNE_AUDIT_2026-10.md` on branch `sukk`).

---

You are working on the Cairo Genizah search app (`genizah_search`, production branch `prod-mbp`) and its sibling
`historical-document-analysis` (indexers, text representation, embedding contract). A separate project is
fine-tuning the text embedder (`Qwen/Qwen3-Embedding-0.6B`) and an image encoder. That work produced the
production-side findings below. **Verify each claim against the current code before changing anything** — the
line numbers were read from `origin/prod-mbp` on 2026-10-05 and may have moved. Plan each item, get Isaac's
approval, and follow the repo's CLAUDE.md rules (no backend rebuild, container restart or index swap without
explicit approval). One PR per item.

## Why this matters

The fine-tuned embedder will be trained on record text **without** the `Document ID:` / `Shelf Mark:` lines. Shelf
marks carry no meaning: they are numbering schemes plus whichever collection a fragment was shipped to. Measured on
the current production vectors, those lines are a median 37% of each record's embedded characters. 55% of a
record's 10 nearest neighbours come from the same collection series, against 9% by chance. For records with real
content (200+ characters besides the IDs) the figure is still 31%. Removing the lines makes the vectors group by
subject, but some features silently rely on shelf marks being inside the vector today.

## 1. Shelf-mark and collection routing in the default search box (prerequisite for any shelf-mark-free re-embed)

- **Problem:** the site's default search mode is pure vector search. Typing "T-S 13J" or "Mosseri letters" into the
  main box works today only because shelf marks are embedded.
- **Evidence:**
  - `src/frontend/src/react_app.jsx:51` sets the default `searchMode` to `'semantic'`, and `handleSearch`
    (~l.318–354) posts the raw query to `/search`.
  - `app.py` ~l.747 `/search` → `search_service.search()` is a `script_score` cosine over `match_all`
    (`search_service.py` ~l.845–890), with no shelf-mark detection.
  - The `primary_hybrid` keyword fields (~l.1740–1752) do not include `shelf_mark` or `doc_id`.
  - The chat router schema (`lms_agentic_search.py` ~l.2236–2255) has no collection/filters field.
  - Explicit shelf-mark search (`/search-shelfmark`, `search_by_shelfmark`, `_build_shelfmark_query`) and the chat
    route for bare shelf marks (`lms_agentic_search.py` ~l.2158) already work without vectors.
- **Fix:**
  - (a) In `/search` and `/search-hybrid`, run `shelfmark_normalizer.detect_shelfmarks` on the query. Send a bare shelf
    mark to `search_by_shelfmark`; for a mixed query, add a shelf-mark filter or boost.
  - (b) Map collection names and abbreviations (Mosseri, T-S, ENA, Bodl., Halper, AIU …) to collection/series filters,
    in `/search` and in the chat executor. Add `shelf_mark` and the series to the hybrid keyword fields.
  - (c) In chat, when the user's message contains a shelf mark the router did not extract, inject it deterministically.
- **Acceptance:** about 30 shelf-mark and collection queries return the same or better results via
  `scripts/dev_backend_local.py` (:8010) against the local ES mirror, both before and after a shelf-mark-free re-embed.

## 2. Records with no content left once IDs are removed

- **Problem:** without the ID lines, many records have nothing to embed:
  - 2,533 records become empty (74% Manchester, e.g. "Gaster Printed series 398").
  - About 5,900 more keep only boilerplate. The largest group is 2,267 copies of
    "Newly treated and encapsulated, must be examined" (100% JTS ENA NS).
  - 31% of records share identical text with another record, and records under 60 characters go from 114 to about
    11,000.

  These would become identical vectors that crowd vague queries.
- **Fix:** never embed an empty string. Index a `semantic_eligible: false` flag (or no vector) for content-free
  records, and filter on it in every vector leg. Those records stay reachable by shelf mark and BM25. The training
  side uses the same rule: `evals/embedding_audit/semantic_text.py` (`semantic_text`, `boilerplate_descriptions`,
  `is_eligible`) once it lands, so production and training can share one definition.

## 3. Version the embedding text, not just the model

- **Problem:** the index `_meta` contract records the model, revision, instruction and a canary query vector, but
  not which text representation built the document vectors. A stripped/unstripped mix would pass the startup gate.
- **Evidence:**
  - `historical-document-analysis/src/embeddings/qwen_text_embedding.py` ~l.205–245 (`build_index_meta`).
  - `index_merged_genizah.py` ~l.73 defaults `index_name` to the live index and upserts.
  - `index_bibliography.py` ~l.319 skips ids that already exist.
  - `verify_reembedded_indexes.py` ~l.125–146 rebuilds text with `create_text_representation`.
  - The startup gate is `app.py` ~l.176–215; on any mismatch `embedding_client.py` ~l.61 disables embeddings for
    every caller.
- **Fix:**
  - Use one versioned function, e.g. `create_text_representation(version="semantic_v2")`, for the indexer, the
    verifier and training.
  - Add `text_representation_version` plus a document canary to `_meta`, and have writers refuse to write on a
    mismatch.
  - Remove the live-index default and require a new index name for every re-embed.
  - Re-embed **both** indices (merged + bibliography) as one cutover, because the gate checks both against the
    single embedder.

## 4. Recalibrate cosine-dependent constants when the model changes

- **Problem:** fine-tuning shifts the absolute scale of cosine scores.
- **Evidence:**
  - `lms_agentic_search.py` ~l.231 and ~l.2739–2806: `SIMILARITY_THRESHOLD = 0.4` on raw cosine triggers the
    fallback and then "no relevant sources".
  - Hybrid scoring adds `weight * (cosineSimilarity + 1)` to BM25 (`search_service.py` ~l.1731, ~l.1776), so a
    scale change silently re-weights semantic vs keyword.
- **Fix:** at cutover, recompute the threshold from the new model's relevant/irrelevant cosine distributions and
  retune the hybrid weights. Re-run the chat eval sets (agentic_rag v1, crosslingual, multiturn) on the dev backend
  against a locally re-embedded copy.

## 5. Bibliography index: keep its shelf marks

- **Rule:** the shelf-mark strip applies to the **primary** (fragment) index only. In scholarship pages, the
  `Shelf marks mentioned:` line and inline marks are the primary↔secondary bridge.
- **Why:** chat sends a bare shelf mark to `bibliography_hybrid` at 50/50 RRF weights (`search_bibliography.py`
  ~l.335–480, `lms_agentic_search.py` ~l.2158–2161, 2251–2252). If the shelf marks were stripped, the semantic leg
  would rank noise. The line is only 3.1% of page characters.
- **Fix:** drop only the `Source PDF:` filename line from bibliography embedding text, if anything.

## 6. Subjects as an app feature (facets and chat filters)

- **What:** the training side is building a subject table: which records are about what.
  - Sources: KTIV catalogue headings (extended to Hebrew-script and festival-liturgy headings), KTIV domain, curated
    PGP tags, and Sefaria topics reached via catalogue heading → Sefaria ref.
  - Coverage: today only about 2.2% of records carry a usable subject. The PGP tags alone cover about 16,000 records
    (≈290 tags with ≥ 20 records: illness, ketubba, trade, marriage, tax, magic, charity, medical, responsum,
    partnership …).
- **Production use:** subject facets in search, a `subjects` filter in the chat router schema, and "more on this
  subject" links.
- **Inputs (in the Studio checkout):**
  - `evals/embedding_audit/build_subjects.py`
  - the PGP-tag curation sheet `evals/embedding_audit/results/pgp_tag_curation.csv` — have Isaac review it first.
  - output `subjects_v1.jsonl` / `subject_vocab.json` on the NAS under
    `/Volumes/home/studio_offload/genizah_search_embedding_audit/v3/`

## 7. Image similarity, whenever an image feature is built

- **Mask the backdrop.** Compute image vectors on the backdrop-masked fragment, not the whole photo. Masking halves
  the "same photography studio" confound for every encoder tested.
- **Encoder choice:**
  - The fine-tuned Hebrew VLM vision towers are the best handwriting encoders (v22b-step1200: scribe P@1 0.457,
    collection confound 0.169).
  - DINOv2 (masked + ink patches) is best for physical join matching.
  - Fused, they give joins R@10 ≈ 0.43.
- **Code:** recipe and code in `evals/embedding_audit/image_features.py` (`fragment_mask`) and `eval_images.py`. The
  image fine-tune in progress will supersede these numbers.

## 8. Data bug: FJP join lists

About 26% of FJP join lists in the merged data are corrupted by a shelf-mark-prefix scrape bug. The audit filtered
the suspects out (`joins_fjp_clean` in `build_image_manifest.py`). The fix belongs in the merge/scrape pipeline of
`historical-document-analysis`.
