# AI reads ("Transcribe with AI (beta)") — two-reader line confirmation

Status: **deployed 2026-09-08.** Index `genizah_ai_transcriptions_v1` exists on the
prod cluster and is fed from the pipeline's `ai_reads_qwen3-vl-8b-heb-v20a-step1800.jsonl`
by re-running the loader (idempotent upsert). Backend + frontend rebuilt with the
feature and the shelf-mark ranking fix (exact/suffix matches first).

## Evidence (probe of 2026-09-08, sibling repo)

24 KTIV Talmud pages (474 ground-truth lines) read by two independent readers:
`qwen3-vl-8b-heb-v20a-step1800` with the grounded-page prompt (21/24 pages
parseable, 375 lines) and the site's Kraken service (`MiDRASH_Gen_01`,
`POST /transcribe_lines`, ~20 s/page CPU). Reference implementation:
`historical-document-analysis/src/datasets/evaluations/grounding_eval/line_agreement_probe.py`.

Matching rule (`lines-v1-20260908`): a Kraken fragment belongs to a VLM line
when ≥ 50% of its height lies in the line's vertical band and ≥ 50% of its
width in its horizontal extent; one line per fragment (best vertical overlap);
fragments concatenated right-to-left. Agreement = 1 − Levenshtein / max(len)
on Hebrew letters only.

| τ | lines accepted | CER ≤ 0.10 | CER ≤ 0.20 | CER > 0.5 leaked | box on right line |
|---|---|---|---|---|---|
| 0.6 | 64 (22%) | 78% | 92% | 4.7% | 95% |
| **0.8** | **45 (15%)** | **84%** | **96%** | **4.4% (0% text-aligned)** | **96%** |
| 0.9 | 30 (10%) | 97% | 100% | 0% | — |

Findings that shaped the design:

- **τ = 0.8.** Nothing with a badly wrong reading leaks; residual error is
  shared letter confusions (ד/ר, ב/כ, ם/ס), so the badge says "two readers
  agree", never "verified". Same precision on v1.9a → not checkpoint-tuned.
- **Coverage is low on Talmud hands** (Kraken fragments median 4 letters; 31%
  of VLM lines had nothing to compare with) → badge on ~1 line in 8. Expect
  more on documentary PGP hands (MiDRASH's training genre).
- **VLM boxes alone are not display-ready**: only 58% sit on the line whose
  text they carry; text is much better than geometry (median line CER 0.14).
  Snapping to Kraken does not fix it. → **Green solid boxes for agreed lines;
  yellow dashed "caution" boxes for unconfirmed lines (decided 2026-09-08: a box
  still helps the reader find the line even when the text has errors); Kraken
  fragments in blue, off by default.**
- Single decode; a rerun moves individual lines. Say so in the UI.

## Rule v2 (`lines-v2-20260909`, pipeline-side, 2026-09-09)

Triggered by the "narrow boxes" question on L-G Ar. II.70. Corrected diagnosis:
that page is two columns; the model box covered ~60% of its column line, not a
fifth. But the VLM boxes are a **template**: across the first 115 pilot pages,
87 have ≥ 50% of their lines sharing one x-range (one range stepped down the
page). Under v1 the horizontal ≥ 50% test then dropped edge Kraken fragments,
which only ever *lowered* agreement (recall loss, no false confirmations).

v2 changes: (1) fragment assignment is band-first, clustered into columns by
horizontal gaps > 60‰, the cluster overlapping the model box most taken whole;
(2) `bbox` is the **evidence box** = model box ∪ assigned fragments; (3)
`htr_fragments` holds the whole row of the line's column. `agreement`,
`AGREED_MIN = 0.8`, statuses, schema and the surfacing rule are unchanged.
Probe pages: precision kept (100% of confirmed lines at CER ≤ 0.20, box on the
right line 98% vs 96%), lines without evidence 95 → 59 of 340. The site
accepts both rule versions (`ACCEPTED_RULE_VERSIONS`); records carry theirs.

## Decisions

1. **Offline batch, not realtime.** Kraken `/transcribe_lines` + one grounded
   v2.0a read per image (25–90 s on the Studio, one request at a time, at
   night) → match → one sidecar per image. Visitors never touch LM Studio.
2. **Own images only.** Each record names the exact `image_url`
   (`image_urls[image_index]` of the merged doc) and its pixel size after EXIF
   orientation. The viewer shows the same file; 0–1000 boxes scale cleanly.
3. **Elasticsearch side index** `genizah_ai_transcriptions_v1`, key
   `(doc_id, image_index, ai_read.vlm_model)`. Not a field on the merged index
   (production data) and not Neo4j yet (promote lines to nodes when entity
   linking needs them).
4. **Surfacing rule** (derived at load time into `surfaced`): `parsed` and
   (`n_agreed ≥ 3` or `n_agreed ≥ 25% of n_lines`). Everything else is loaded
   but hidden; `published` is the maintainer's manual switch on top.
5. **The old page-level consensus gate** stays the acceptance path for
   whole-page documentary transcriptions; it is not used here.
6. **Private live portal** (uploads, real VLM calls, prompt experiments) is a
   separate later phase behind a server-side admin key. Not built.

## Record schema

`src/backend/ai_transcriptions.py` (`AiTranscriptionRecord`), enforced by the
strict ES mapping (`INDEX_MAPPING`) and a test that the two agree. The
`ai_read` block is the pipeline sidecar **verbatim**; the loader adds the
envelope and derives `surfaced`/`text`.

```jsonc
{
  "doc_id": "Cambridge_CUL_T_S_8J22_22",      // ES _id in the merged index
  "source_index": "genizah_merged_v4",
  "image_index": 0,                           // position in image_urls
  "image_url": "https://…/default.jpg",
  "image_width": 2800, "image_height": 1785,  // after EXIF orientation
  "image_sha256": "…",
  "ai_read": {
    "vlm_model": "qwen3-vl-8b-heb-v20a-step1800", "vlm_revision": "af9df6a0",
    "htr_model": "MiDRASH_Gen_01", "rule_version": "lines-v1-20260908",
    "decoded_at": "2026-09-08T16:31:00", "parsed": true,
    "n_lines": 16, "n_agreed": 3,
    "lines": [
      {"index": 0, "text": "…", "bbox": [104, 237, 928, 356], "agreement": 0.86,
       "status": "agreed", "htr_text": "…", "htr_fragments": [[x1,y1,x2,y2]]},
      {"index": 1, "text": "…", "bbox": [104, 356, 928, 472], "agreement": 0.22,
       "status": "unconfirmed", "htr_text": "…", "htr_fragments": []}
    ]
  },
  "text": "…\n…",        // derived; indexed but NOT in default site search
  "surfaced": true,      // derived from the surfacing rule
  "published": true,     // maintainer switch
  "created_at": "2026-09-08T17:00:00Z"
}
```

Validation guarantees: `status` must equal what the 0.8 rule gives for
`(agreement, htr_text)`; `rule_version` must be the served rule; `n_lines`/
`n_agreed` must match `lines`; indices are `0..n-1`; boxes clamped to 0–1000.

### Loader

`scripts/load_ai_transcriptions.py --input records.jsonl [--apply] [--unpublished]`
— dry run by default; `--apply` creates the index if missing and bulk-upserts
(a production data write); `--unpublished` loads hidden for review.

## Site components

| Piece | Where |
|---|---|
| Schema, rule, surfacing, ES mapping, read service | `src/backend/ai_transcriptions.py` |
| `GET /ai-transcriptions/{doc_id}` → `{enabled, available, items[]}` | `src/backend/app.py` |
| `GET /ai-transcriptions/{doc_id}/{image_index}?model=` → full record | `src/backend/app.py` |
| Tests (rule, surfacing, validation, mapping↔schema, visibility) | `src/backend/tests/test_ai_transcriptions.py` |
| Viewer route `/read?doc=&image=&index=&model=` | `src/frontend/src/read/ReadFragment.jsx` |
| Button on the document modal (only when `available`) | `src/frontend/src/core_results/DocumentModel.jsx` |

Viewer behaviour: header "N of M lines confirmed by a second reader (Kraken)";
green boxes for agreed lines and yellow dashed boxes for unconfirmed ones, numbered
in reading order, hover ↔ text; unconfirmed lines grey in the text panel; badge tooltip "Two
independent readers produced the same text (agreement 0.86). Both can still
share small letter confusions; not a scholarly transcription."; toggles for
boxes / numbers / Kraken fragments / Kraken reading; beta banner on first
load; details disclosure with model, revision, rule version, decode time and
the raw record; credits (NLI KTIV, Princeton Geniza Project, holding
libraries). Missing index or feature off → "not available", never an error.

## Open items

- [x] Pipeline: `two_reader_lines --from-consensus high` writes envelope + sidecar JSONL.
- [x] First load + deploy (2026-09-08); records are published as they arrive.
- [x] Serving index bumped to `genizah_merged_v5` (2026-09-08): `ELASTICSEARCH_INDEX` in
      `src/backend/.env`, frontend `KG_ES_INDEX` default; canary verified on startup.
- [ ] Review coverage on documentary hands once a few hundred records are in.
- [ ] Later: thumbs-up/down per line (correction set), Mirador annotation export,
      Neo4j line nodes, "compare with v1.9a", private live portal.
