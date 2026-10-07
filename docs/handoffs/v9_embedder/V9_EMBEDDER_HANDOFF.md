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
  `Source PDF:` filename line. Note that the fine-tune was not trained or evaluated on scholarship pages (a bibliography
  retrieval check is being run in the audit session before cutover; wait for its numbers).

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
- **Recalibrate cosine-dependent constants.**
  - `SIMILARITY_THRESHOLD = 0.4` in `lms_agentic_search.py` triggers the no-relevant-sources fallback.
  - The hybrid score adds `weight * (cosine + 1)`.
  - Fine-tuning shifts the cosine scale. Recommended values from the new model's relevant/irrelevant cosine
    distributions are being computed in the audit session; use those, not 0.4.
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
