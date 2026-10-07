# Handoff prompt — build index v9 with the fine-tuned text embedder (2026-10-07)

> Paste everything below the line into the MBP session that builds v9. It is self-contained. The model, eval and
> text function come from the embedder work on branch `eval-retrieval` of `genizah_search`
> (`docs/EMBEDDING_FINETUNE_AUDIT_2026-10.md`, sections 2.4–2.6). That branch must be pushed for this session to read
> `evals/embedding_audit/semantic_text.py`.

---

v9 should be embedded with the fine-tuned text embedder. It is a fine-tune of the current Qwen3-Embedding-0.6B with
the same architecture and output size, but its vectors are **not** compatible with the current ones: documents and
queries must both come from the new model, and both indices (merged fragments + bibliography) switch together because
the startup gate checks both against one embedder. Plan this as one cutover, get Isaac's approval for each production
step, and follow the repo's CLAUDE.md (no backend rebuild, index swap or container restart without explicit approval).

## 1. The model (pin exactly this)

- Hugging Face repo `isaacmg/genizah-embed-qwen3-0.6b` (private; use the `HF1_TOKEN` from historical-document-analysis'
  `.env`, or copy the downloaded folder). Revision **`dbfab73f1e91a6b95eba728541cc5a391a88d933`** (tag `v3-final-run1`).
  Pin the commit sha, not the tag, in `embedding_models.py` and in the index `_meta`.
- Contract (unchanged from today apart from the weights):
  - 1024-d, last-token pooling, L2-normalised, cosine.
  - Documents are embedded raw (no prefix).
  - Queries are prefixed with exactly `Instruct: Given a search query, retrieve relevant passages\nQuery: `.
  - `max_seq_length = 8192`.
- **Trap:** the saved model's default `query` prompt (in `config_sentence_transformers.json`) is the stock Qwen one,
  "Given a **web** search query…". The model was trained with the prefix above. Always pass the prefix explicitly and
  never use `prompt_name="query"`.
- Also set `max_seq_length = 8192` explicitly when loading. Training used 384-token windows, and the evaluation that
  produced the numbers below ran at 8192.

## 2. The document text (this matters as much as the model)

The model was trained and evaluated **only** on "semantic text":
- the production `create_text_representation()` output with the `Document ID:` and `Shelf Mark:` lines removed;
- editor credits stripped (`[Editor: …]` and the PGP `Editor: Surname, Given` form);
- field lines left empty after that dropped;
- the record's own shelf mark, cited shelf marks of other fragments, and PGPIDs inside descriptions or join notes
  masked as `[shelfmark]`.

It was never evaluated on today's text, where ID lines are about 37% of the characters. Embedding today's text would
waste much of the gain.

- **Reference implementation:** `genizah_search/evals/embedding_audit/semantic_text.py` (branch `eval-retrieval`).
  `semantic_text(create_text_representation(doc))` is exactly what was trained on. Port it into
  historical-document-analysis as a versioned text function, e.g. `create_text_representation(version="semantic_v3")`.
  Use that one function for the indexer, `verify_reembedded_indexes.py` and any future training, and keep its shelf-mark
  regexes identical. They include the production `shelfmark_normalizer` pattern plus extra formats it misses.
- **Eligibility:**
  - A record is ineligible if its semantic text is empty, under 40 characters, or only a boilerplate label plus
    Language / Document Type / Date lines.
  - Use the frozen list in `docs/handoffs/v9_embedder/boilerplate_descriptions_v3.json` (64 labels) for parity with the
    eval.
  - About 13,000 of the 73,543 v8-era records were ineligible: 2,533 empty, and 2,267 "Newly treated and
    encapsulated…".
  - Ineligible records still go into the index for shelf-mark and BM25 lookup, but get no vector, or
    `semantic_eligible: false` filtered out of every vector leg. Never embed an empty string.
- **Bibliography index:** re-embed it with the new model too (the gate requires it), but **keep** its
  `Shelf marks mentioned:` line and inline marks; they are the primary↔secondary bridge. Drop at most the
  `Source PDF:` filename line. The fine-tune was not trained on scholarship pages, but the bibliography check below shows no
  harm and a significant gain on Hebrew queries.

## 3. The index build

- Build new index names (`genizah_merged_v9` and a new bibliography index name). Never upsert into the live index;
  the default index name in `index_merged_genizah.py` points at the live one, so pass the new name explicitly.
- In `_meta`, record:
  - the model repo and revision sha;
  - `query_prompt`;
  - `text_representation_version: "semantic_v3"`;
  - a query canary (as today);
  - a **document canary**: one real record's semantic text and its vector.
- Writers should refuse to write when the model or text version differs from `_meta`.
- Run `verify_reembedded_indexes.py` with the same text function before any swap.

## 4. Things that must ship with (or before) the switch

- **Search-box shelf-mark and collection routing.** The site's default search mode is pure vector search. Typing
  "T-S 13J" or "Mosseri" works today only because shelf marks are inside the vector, and with semantic text they no
  longer are. Detect shelf marks and collection names in `/search` and `/search-hybrid`, and route them to
  `search_by_shelfmark` or a collection filter. See `docs/handoffs/EMBEDDER_PRODUCTION_HANDOFF_2026-10-05.md` item 1.
- **Keep `SIMILARITY_THRESHOLD = 0.4`.** Measured and verified on 2026-10-07, read from origin/prod-mbp:
  - It is not compared with raw cosine. It applies to the deduplicated bibliography `similarity_score`, which is the
    normalised weighted RRF from `search_bibliography.search_hybrid`: `61 * Σ w/(60+rank)`, rank-only.
  - Changing the embedder cannot move it.
  - Don't add a raw-cosine gate at 0.4. The tuned model lowers cosines by about 0.1 on the bibliography index (median
    known-item top-1 0.60 → 0.49).
- **Hybrid weights:**
  - Bibliography: no change (RRF).
  - Primary `/search/hybrid`, `sw*(cos+1) + kw*hit`: no change at the 50/50 default, or anywhere down to 70/30. A
    lexical hit outranks a non-hit under either model, and the tuned score spread is only 1.2× the base one.
  - Only the 80/20 and 90/10 slider settings would shift slightly.
- **Bibliography index:** re-embed it with the tuned model; this is verified safe.
  - Eval: 4,944 pages, 300 hand-written known-item queries.
  - No metric's paired 95% CI is below zero in any setting: dense, endpoint 60/40, endpoint 30/70, or the chat's own
    50/50 top-5.
  - Hebrew and cross-lingual known-item retrieval improve significantly: R@10 0.34 → 0.58.
  - English is flat. Watch English top-1 on paraphrased queries (R@1 −0.06, not significant). If v9 spot checks show
    regressions, the cheap fix is a heavier keyword weight in `bibliography_hybrid`, not a second model.
  - Keeping the base model for bibliography is not an option without a second embedding container, client and drift
    state. The tuned model scores 0.41 on the bibliography canary, so a mixed setup disables semantic search for both
    indices.
- **Deployment:** switch the `EMBEDDING_MODEL_NAME` / `EMBEDDING_MODEL_REVISION` compose env vars and the defaults in
  `embedding_models.py`. The repo is private, so the embedding container needs an HF token or a pre-filled `hf_home`
  volume.
- **Production bugs found on the way** (separate fixes, not blockers):
  - The chat's `primary_hybrid` action always fails: `SearchRequest` has no `semanticWeight` field.
  - `search_service.search_hybrid` with Advanced Search filters builds a filter-only bool query. That scores 0, and
    `boost_mode: multiply` then zeroes every hybrid score.
  - `search_by_author` scores `(cos+1)/2` and sorts those together with RRF scores. That's harmless in observed plans;
    merge by rank if plans start mixing them.
- Re-run the chat eval sets (agentic_rag v1, crosslingual, multiturn) on the dev backend against the v9 indices before
  the swap.

## 5. What to expect (frozen eval, 60,493 records)

Scored on records the model never trained on:

| | today's model + text | new model + semantic text |
|---|---|---|
| Sukkot / Pesach / kashrut / partnership / slavery / geonic academies, AP | 0.064 | 0.198 |
| nuanced queries for those ("laws of lulav" style), AP | 0.019 | 0.161 |
| all other subjects, AP | 0.132 | 0.377 |
| finding a specific record, MRR / R@10 | 0.193 / 0.295 | 0.282 / 0.426 |
| junk/duplicate records in the top 10 | 22% | 15% |
| same-collection neighbours (lift over chance) | 3.85× | 2.76× |
