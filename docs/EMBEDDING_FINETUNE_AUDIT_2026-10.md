# Embedding fine-tuning audit — text retrieval + fragment images (2026-10-02/03)

_Status: complete (probes finished 2026-10-03 ~04:25). Written by Claude while Isaac was offline; numbers are
reproducible from `evals/embedding_audit/` (Appendix A). Nothing was committed or deployed._

Audit code: [`evals/embedding_audit/`](../evals/embedding_audit/) · raw results: `evals/embedding_audit/results/*.json` ·
large artifacts (corpus, vectors, images, model caches) on the NAS:
`/Volumes/home/studio_offload/genizah_search_embedding_audit/`.

---

## TL;DR

**Text (semantic retrieval).**
1. **The embedder knows topic names but not Jewish concepts.** Queries that imply a topic without naming it
   ("laws of lulav", "Kol Nidre", "בדיקת חמץ") reach the right one of 15 topics **22 %** of the time; the topic
   name itself works 95 % of the time. "Kol Nidre" lands on *niddah*, "the four questions" on Purim. bge-m3 and
   multilingual-e5 are no better (18–21 %), so swapping models won't fix it. It has to be trained in.
2. **Over the real 73.5k-record corpus, the vector leg is no better than BM25** for these queries
   (nDCG@10 0.21 vs 0.20 on catalogue-identified records). **With the query "understood"** (topic name appended,
   an oracle for a perfect query encoder) **it triples to 0.62**. The documents are embedded well enough; query
   understanding is the bottleneck, which is exactly what fine-tuning or a concept-expansion layer addresses.
3. **Our own corpus can fix the document side but not the query side.** A LoRA pilot on 3k corpus-only pairs
   (frames, titles, Hebrew transcription spans → labels; 34 min on the Studio GPU) left nuanced-query retrieval
   and the concept probe unchanged. But on held-out records it made documents easier to find once the topic is
   named: framed AP +30 %, and **recall of human-verified Sukkot items roughly doubled** (R@1000 0.24 → 0.41; for
   items that never say "Sukkot", 0.18 → 0.34). The query-side concept bridge has to come from external data:
   Sefaria topics and bilingual texts, and LLM-written realistic queries.
4. **Some items are unreachable from the query side.** Of 125 human-verified Sukkot items, even the oracle finds
   only 22 % in the top 1,000, because letters/calendars dated by the festival carry no Sukkot signal in their
   embedded text (metadata + 1,000 transcription characters). That needs document-side enrichment (fuller AI
   transcriptions, Sefaria text for identified-but-untranscribed fragments, frame→topic words).

**Images (similarity, joins, scribes).**
5. **The old image vectors were useless for avoidable reasons.** They were ColNomic late-interaction patches
   mean-pooled to 128-d, mostly text-weighted, from 820 docs, with a cache bug, and never evaluated.
6. **Off-the-shelf encoders give a real but weak join signal, dominated by photography.** On a 5,176-image
   gallery with 1,661 known join pairs, DINOv2 finds the partner in the top 10 for **39 %** of fragments (top 1:
   25 %), but only **16 %** across collections. The same vectors predict the holding collection 46–60 % of the time
   (chance 8 %): backdrop colour (blue CUL boxes, black Bodleian), labels, rulers and layout. **Masking the
   backdrop** halves that confound (27 %) and raises cross-collection joins to 20 %; adding ink patches from
   inside the mask gives the best join retrieval (**42 % top-10**) and the best scribe signal (P@1 0.37 vs chance
   0.08). CLIP and a classical handwriting descriptor are weaker. **The VLM's own vision tower (stock
   Qwen3-VL-8B), fed the masked fragment and mean-pooled after the merger, is the best *handwriting/script*
   encoder tested** (scribe P@1 0.42, AUC 0.64, script style 0.50) though weaker for joins. Fusing it with DINOv2
   gives the best join retrieval (R@10 0.42, R@1 0.30). So "the VLM image embeddings are useless" holds for the
   old pipeline, not for the model.
7. **Handwriting lives at line level, and supervision pays off fast.** On KTIV line crops, a single line finds
   another page of the same manuscript as its top hit 34 % of the time (chance 0.08 %), with almost no library
   confound. **A small head trained in 30 s on frozen features from 782 other manuscripts nearly doubles
   held-out mAP (0.22 → 0.42)**, strong evidence that fine-tuning a Genizah handwriting encoder is worth the GPU time.

**Data quality found on the way.** About 26 % of FJP join lists (619/2,411) are corrupted by a shelfmark-prefix
scrape bug ("Or.1080 4.5" carries "4.55"'s list), and some FJP image files are mis-filed (label in photo ≠
filename).

**What to do** (§5–§6):
- Fix the plumbing so vector gains reach users: the agent hybrid crash, a graded keyword leg, and concept
  expansion applied to the embedded query.
- Build a scholar-validated eval set.
- Fine-tune the 0.6B embedder on Colab with Sefaria + synthetic queries + corpus pairs.
- Re-embed both indexes.
- For images: clean the join ground truth, then train a line/patch handwriting encoder (DINOv2 + metric learning
  with same-collection negatives), and build join discovery as candidate generation + rerank + scholar review.
  Text adjacency alone already yields 42 Bavli join candidates to review.

---

## 0. How this audit was run (and what it did not touch)

* **Nothing in production was changed.** No ES writes, no container/LM Studio actions, no index switches.
  All probes run as local torch/MPS processes in the sibling repo's `.venv` (it already pins the exact
  embedding contract: torch 2.8.0 / transformers 4.57.6 / sentence-transformers 5.6.1).
* **Offline corpus = production.** Everything was computed from
  `historical-document-analysis/src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl` (73,543
  records): for each record the audit rebuilds the string the indexer embeds
  (`GenizahDocument.from_merged_format(...).create_text_representation()`) and embeds it with the pinned contract.
  ES reads were blocked by the session's permission classifier during the first run. **Verified 2026-10-04 with
  Isaac's OK:** live `genizah_merged_v8` has exactly 73,543 top-level documents (the 132k in `_cat/indices` counts
  nested transcription/translation/bibliography sub-docs), and the audit's vectors equal the stored
  `embedding_vector`s (cosine 1.000 on 400 random docs). So every T2 number above is production behaviour.
  `genizah_chat_feedback_v1` holds 2 entries, one a thumbs-down on "What do Genizah fragments tell us about Hoshana
  Rabbah?", which is this gap exactly.
* **Memory safety / coordination with the sibling sessions.** Every probe ran under `guard.py`:
  - footprint cap of 8–10 GB, plus an MPS high-watermark cap;
  - killed and resumed when free RAM < 25 %, swap > 12 GB or disk < 25 GB;
  - paused during the sibling's PEFT merge / MLX convert;
  - **no GPU use during its v22b checkpoint evals**, which were agreed with the "19b/20 fine-tuning" HDA session;
    GPU jobs ran only between evals, about 2–3 GPU-hours in total, as agreed;
  - since 22:06 on 10-02, an audit-wide mutex enforces **one audit process at a time**, CPU jobs capped at 4 threads.

  Status was published in `~/box_jobs/genizah_search_embedding_audit.status`. All model caches and images went to
  the NAS, so the internal disk (their 30 GiB gate) was not consumed. No OOM, no swap growth beyond 9.4 GB, and
  LM Studio was never touched.
  **One lapse:** around 21:00–22:05 on 10-02, three of my processes ran at once (CPU embed, CPU line pilot, GPU
  Qwen3-VL). The sibling reported its consensus pipeline slowing 2–4× (35–40 s/page → 56–323 s). I fixed it with
  the mutex at 22:06; the sibling logged it in its status journal.

---

## 1. Current state

### 1.1 Text retrieval as built today

| Piece | Fact | Where |
|---|---|---|
| Embedder | `Qwen/Qwen3-Embedding-0.6B@97b0c61`, 1024-d, cosine; queries prefixed `Instruct: Given a search query, retrieve relevant passages\nQuery: `, docs raw | `src/embedding_service/embedding_models.py` |
| What a fragment vector encodes | one labelled string: ID + shelfmark, description (PGP EN > KTIV HE title > TEI > FJP), KTIV alt titles (incl. frame text like "Hilkhot ha-Rif: Sukkah"), language, heuristic doc type, date, transcription **cut to 1000 chars**, translation cut to 500, people/places. Median length **209 chars**; p90 1,069 | HDA `genizah_document.py:1375-1529` |
| …so mostly metadata | 34k records have a PGP English description; ~5.8k FJP + ~2.1k KTIV transcriptions in total | merged jsonl |
| Fragment search | brute-force `script_score` cosine; hybrid = `semW·(cos+1)` + **constant** keyword bonus (not BM25-graded) | `search_service.py` |
| Bibliography (chat default) | weighted RRF k=60; alias expansion added to the **lexical leg only** — it never reaches the query vector | `search_bibliography.py:335-486` |
| Terminology | 20 concept groups of *name synonyms*; Sukkot = `sukkot, sukkoth, tabernacles, hoshana, סוכות`; no lulav/etrog/sukkah/four species/Shemini Atzeret/Simhat Torah; substring matching (`get` fires on "together") | `genizah_terminology.py` |
| Canary | backend embeds one canary per index with **one shared embedder** and disables semantic search if cos ≤ 0.99 → a fine-tuned model means re-embedding **both** indexes (or serving two embedders) | `app.py:176-216` |
| Known bug (already in memory notes) | agent `primary_hybrid` builds `SearchRequest(semantic_weight=…)`, `search_hybrid` reads `request.semanticWeight` → AttributeError → swallowed; confirmed on `origin/prod-mbp` | `lms_agentic_search.py:2688-2701` |

### 1.2 Image embeddings as built before

* Model: `nomic-ai/colnomic-embed-multimodal-7b` (ColQwen2.5 late-interaction). Its multi-vector page
  output was **mean-pooled into a single 128-d vector** — which discards exactly what late-interaction
  models are for — and the default "multimodal" vector was `0.7·text + 0.3·image`.
* Cache bug: cache key = md5(text) only, and the text-weighted vector was cached before the
  `image_only` branch → an "image-only" request could be served the text-weighted vector.
* Only 820 docs were ever indexed (`cairo_genizah_image_v_1.1.4`), never evaluated on joins/scribes;
  random-pair median cosine ≈ 0.6 (poor discrimination). Replaced 2026-07-24 by text-only Qwen3.
* So "the image embeddings are useless" is right, but for avoidable reasons: wrong pooling for that model
  class, a model trained for *printed-page text retrieval*, whole-photo input, and no evaluation.

---

## 2. Text audit results

### 2.1 T1 — concept probe: does a nuanced query land on the right topic?

`concept_probe_v1.json`: 15 topics (Sukkot, Pesach, Yom Kippur, Rosh Hashanah, Shabbat, Hanukkah, Purim,
Shavuot, Tisha be-Av, marriage, divorce, kashrut, mourning, tefillin/mezuzah, niddah). Each topic has 4
catalogue-style anchor descriptions (2 EN, 2 HE; embedded as documents), **130 nuanced queries that never name the
topic** ("laws of lulav", "the four species", "Kol Nidre", "בדיקת חמץ", "לולב הגזול והיבש פסול"), and 82 spelling
variants of the topic names (Sukkos/Succoth/סֻכּוֹת…). Score = is the right topic the nearest centroid (15-way; chance 0.07).
_Written by Claude for this audit; needs a scholar's review before it is used as a gate._

| model | instruction | nuanced top-1 | top-3 | MRR | Hebrew q | Latin q | topic **names** top-1 |
|---|---|---|---|---|---|---|---|
| **Qwen3-Emb-0.6B (prod)** | production | **0.22** | 0.46 | 0.41 | 0.16 | 0.27 | **0.95** |
| Qwen3-Emb-0.6B | domain-specific | 0.15 | 0.42 | 0.35 | 0.12 | 0.18 | 0.77 |
| Qwen3-Emb-0.6B | none | 0.19 | 0.35 | 0.35 | 0.14 | 0.23 | 0.71 |
| bge-m3 | — | 0.18 | 0.45 | 0.37 | 0.14 | 0.22 | 0.87 |
| multilingual-e5-large-instruct | production | 0.21 | 0.42 | 0.38 | 0.16 | 0.25 | 0.83 |
| Qwen3-Emb-4B | — | not run: ≈8 GB of fp16 weights exceeds the per-probe RAM budget agreed with the sibling session; test on Colab with the real fine-tune | | | | | |

Robustness: per-topic z-scoring (removes "hub" topics) gives 0.20–0.22; nearest single anchor instead of centroid
gives 0.23–0.24. Same conclusion.

What it means:
* **Names are fine, concepts are not.** Every model maps the festival's *name* (incl. Sukkos/Succoth/Hebrew) to the
  right topic 83–95 % of the time, but the practices, liturgy and texts that *imply* the topic only 18–22 %.
  Per topic (prod model): Sukkot 5/19, Pesach 2/12, Yom Kippur **0/10**, Rosh Hashanah 0/8, Shavuot 0/8,
  divorce 0/6; "Kol Nidre"/"כל נדרי" → niddah (string similarity), "the four questions" → Purim,
  "Haggadah" → Hanukkah, "the four species" → kashrut. **Hebrew queries are worse than English (0.16 vs 0.27).**
* It is **not a model-choice problem**: two strong multilingual alternatives are no better. The knowledge has to
  be trained in (or supplied at query time).
* **Do not change the query instruction to a "Jewish domain" prompt** without training — it lowered top-1
  0.22 → 0.15. (Instruction changes alone would not need doc re-embedding, which is why it was tested.)

### 2.2 T2 — the same queries against the real corpus (73,543 records)

The 130 nuanced queries of the 11 topics with enough labelled records, run against the full offline corpus
(production text + production embedding contract). Labels: **strict** = the record's KTIV identification
maps to the topic (e.g. a `[Talmud Bavli]: Sukkah` frame, Hoshanot) or it is in the human Sukkot seed; **lenient** =
strict, or the record's embedded text itself names the topic. The **framed pool** is the 5,661
catalogue-identified records, where labels are close to complete, so AP/nDCG there are meaningful (random AP 0.025).

| retriever | P@10 strict | P@10 lenient | framed AP | framed nDCG@10 | framed R@100 | Sukkot gold R@1000 | …never-says-Sukkot R@1000 | median rank of gold |
|---|---|---|---|---|---|---|---|---|
| BM25 (keyword ceiling) | 0.080 | 0.302 | 0.060 | 0.196 | | | | |
| **Qwen3-0.6B (production)** | 0.061 | 0.296 | 0.081 | 0.213 | 0.076 | 0.112 | 0.081 | 13,748 / 73,543 |
| Qwen3-0.6B, "domain" instruction | 0.040 | 0.261 | 0.074 | 0.193 | 0.076 | 0.176 | 0.122 | 11,375 |
| **Qwen3-0.6B + oracle expansion** (gold topic name appended to the query) | **0.210** | **0.823** | **0.275** | **0.616** | 0.284 | 0.216 | 0.149 | 8,774 |
| multilingual-e5-large-instruct | 0.062 | 0.291 | 0.085 | 0.218 | 0.086 | 0.144 | 0.122 | 9,302 |
| multilingual-e5-large-instruct + oracle expansion | 0.142 | 0.615 | 0.210 | 0.482 | 0.224 | 0.184 | 0.176 | 7,151 |

Per topic (production, framed AP / nDCG@10): Purim 0.23 / 0.55 and kashrut 0.13 / 0.47 work somewhat; **Yom Kippur
0.035 / 0.015, niddah 0.019 / 0.029, tefillin 0.022 / 0.048, divorce 0.026 / 0.09 are essentially random.**

What it means:
1. **For concept queries the semantic leg is no better than keyword search** (nDCG@10 0.213 vs BM25 0.196). The
   hybrid gets nothing from the vector side exactly where it should help. Swapping to multilingual-e5 changes
   nothing (0.218), and its oracle ceiling is lower (0.48 vs 0.62), so Qwen3-0.6B stays the right base to fine-tune.
2. **Query understanding is the bottleneck, and the ceiling is high.** Appending the right topic name to the
   query (what a perfect concept-expander or a fine-tuned query encoder would do) **triples** framed nDCG@10
   (0.21 → 0.62) and AP (0.08 → 0.28), and lifts lenient P@10 from 0.30 to 0.82. The documents are already
   embedded well enough to be found *once the query means the right thing*. The record-to-record space agrees:
   a topic-labelled record's 10 nearest neighbours share its topic 57 % of the time vs a 3.9 % base rate among
   identified records (14.6× lift); domain 75 % vs 15.6 % (4.8×); language 90 % vs 77 % (1.2×). The
   Sukkot-page tech note's "vectors cluster by genre and language rather than festival" holds for
   unidentified records, not for catalogued ones.
3. **Some gold items cannot be reached from the query side at all.** Even the oracle finds only 22 % of the
   125 human-verified Sukkot items in the top 1,000; for the 74 records that never name Sukkot it's 15 %. Those
   are letters dated by the festival, calendars, Hoshana Rabbah pilgrimages: the embedded text (metadata + 1,000
   transcription characters) carries no Sukkot signal. **That part is a document-side problem**: fuller transcriptions
   (AI reads), Sefaria text for identified-but-untranscribed records, frame→topic enrichment. Fine-tuning the
   query side alone will not fix it.

### 2.3 P1 — corpus-only LoRA pilot (does our own data teach the concepts?)

`build_train_pairs.py`: 80/20 split of the 5,661 framed records by doc-id hash (Sukkot seed forced into test);
**15,166 pairs** from the 4,507 train records: 12,556 *label* pairs (frame/title/topic phrase → record) and 2,610
*content* pairs (a Hebrew span of the record's transcription → its label-only text). Because of the shared GPU,
the pilot trained on a random **3,000 pairs**: LoRA r=16 on all projections, CachedMNRL batch 64, 1 epoch, bf16,
34 min on MPS.

| concept probe (T1) | nuanced top-1 | Hebrew q | Latin q | names |
|---|---|---|---|---|
| production | 0.22 | 0.16 | 0.27 | 0.95 |
| corpus-only LoRA pilot | 0.21 | 0.18 | 0.23 | 0.89 |

Held-out corpus retrieval: the 1,154 test-side framed records plus all 67.9k unframed records; train records removed
from the pool. Same T2 queries.

| held-out pool | queries | framed AP | framed nDCG@10 | framed R@100 | Sukkot gold R@1000 | never-says-Sukkot R@1000 | median rank of those |
|---|---|---|---|---|---|---|---|
| production | nuanced | 0.092 | 0.173 | 0.199 | 0.120 | 0.081 | 13,260 |
| **corpus-only LoRA pilot** | nuanced | 0.093 | 0.162 | 0.187 | **0.248** | **0.189** | 12,419 |
| production | oracle (topic named) | 0.254 | 0.481 | 0.438 | 0.240 | 0.176 | 8,653 |
| **corpus-only LoRA pilot** | oracle (topic named) | **0.329** | **0.518** | **0.502** | **0.408** | **0.338** | **4,250** |

**Reading:** the two halves of the problem respond differently, and that is the most useful result of the pilot.
* **Query side: no change.** For nuanced queries, framed AP and nDCG@10 are flat (0.09 / 0.16–0.17), and the concept
  probe is flat. The corpus has almost no English text *about* religious practice (PGP is documentary, FJP
  translations are empty), so nothing in it teaches "four species → Sukkot". **The concept bridge has to come from
  outside the corpus**: Sefaria topics and bilingual texts, and LLM-written queries.
* **Document side: clear gain from 3k pairs and 34 minutes.** Once the query names the topic, the pilot finds
  identified records better (AP +30 %, R@100 0.44 → 0.50). The human-verified Sukkot items roughly **double**
  (R@1000 0.24 → 0.41), including the ones that never say "Sukkot" (0.18 → 0.34; median rank 8,653 → 4,250). The
  *content → label* pairs taught the model that Hebrew content about lulav/sukkah/hoshanot *is* Sukkot material.
* So the production recipe in §5 needs both: external concept data for the query side, and corpus pairs for the
  document side.

### 2.4 Step 0 — removing the Document ID / Shelf Mark lines, base model, no training (2026-10-05)

Production embeds each record with its `Document ID:` and `Shelf Mark:` lines. Those lines are a median 37% of the
characters, and today 55% of a record's 10 nearest neighbours share its collection (chance 9%). Step 0 re-embedded
the 60,493 eligible records with those lines removed (53,108 unique texts), using the same base model and contract,
and scored both versions with the frozen v3 eval (`results/step0_original_vs_semantic.json`; paired bootstrap 95% CI):

| metric | with ID lines | without | Δ [95% CI] |
|---|---|---|---|
| held-out subjects AP (gate, 6 subjects) | 0.127 | 0.110 | −0.017 [−0.033, −0.004] |
| seen subjects AP (663 subjects) | 0.127 | 0.126 | −0.001 (n.s.) |
| known-item MRR (1,961 queries) | 0.193 | 0.196 | +0.003 (n.s.) |
| collection lift over chance (all / long records) | 3.77 / 2.88 | **2.77 / 2.26** | −27% / −21% |
| duplicate or short records in top-10 | 22.5% | 24.3% | slightly worse |

(Caveat: this local run used the eval and semantic text from before the v3 review. The review also masked shelf
marks inside descriptions for 3,947 eligible records and rebuilt the eval: 15,674 held-out eligible records and 78
linked subjects. The v3 Colab notebook recomputes base-with-IDs and base-without on the final eval, so its numbers are
the authoritative version of this table.)

Removing the lines cuts collection clustering by about a quarter, but it is **not** a free win for subject retrieval
on the base model. The small loss on held-out subjects (Sukkot, Pesach, partnership, slavery) is consistent with
collection acting as a proxy for subject in parts of this corpus; for example, 47% of the Bodleian framed records are
Sukkot. That proxy is exactly the shortcut a growing index should not rely on. Fine-tuning has to supply the meaning
instead, and the v3 notebook evaluates base-with-IDs, base-without and tuned-without side by side. All absolute subject APs are low (≈0.11–0.13 over a
60k-record pool), which is the gap the fine-tune targets.

### 2.5 Local LoRA trial of the v3 recipe (2026-10-05)

`pilot_v3_local.py` setup:
- **Model:** LoRA r16 on all projections of the real Qwen3-Embedding-0.6B, on MPS.
- **Training:** 12,000 v3 pairs, 1 epoch, batch 64, masked GradCache loss, no hard negatives, one seed; 189 steps in
  124 min.
- **Eval:** the frozen v3 eval on a fixed pool of 25,674 records (all 15,674 held-out eligible records plus 10,000
  sampled others, the same pool for every model). Absolute APs are higher than on the full 60k pool, so compare
  columns, not with §2.4.

Results are in `results/pilot_v3_local.json`.

| metric | base, production text (today) | base, semantic text | **tuned, semantic text** |
|---|---|---|---|
| held-out subjects AP (gate, 6) | 0.176 | 0.157 | **0.231** |
| held-out subjects P@10 | 0.393 | 0.387 | **0.484** |
| known-item MRR / R@10 (1,961 queries) | 0.229 / 0.346 | 0.233 / 0.344 | **0.249 / 0.379** |
| within-subject known-item median percentile ↓ | 0.033 | 0.032 | **0.016** |
| collection lift (all) ↓ | 3.41 | 2.51 | **2.30** |
| duplicate/short records in top-10 ↓ | 23.1% | 24.7% | **21.2%** |

Paired bootstrap, tuned minus base (95% CI, clustered by subject):
- **Gate (held-out subjects).**
  - Against base on semantic text: AP **+0.074 [+0.042, +0.117]**.
  - Against base on production text, i.e. today's search: AP **+0.055 [+0.026, +0.092]**, P@10 +0.091.
- **Every held-out subject improves on its own:** Sukkot +0.057, Pesach +0.145, kashrut +0.031, partnership +0.075,
  slavery +0.088, geonic academies +0.049. Every CI is above 0.
- **Seen and linked subjects:** seen AP +0.068 [+0.055, +0.082]; linked AP +0.030 [+0.002, +0.065].
- **Known-item retrieval:** R@10 +0.035 [+0.003, +0.067]. Implicit queries gain most: English +0.061 and Hebrew
  +0.046 MRR. Specific and transliterated queries are flat (n.s.).
- **No memorisation:** the train-side vs held-out gap does not widen (−0.012 [−0.030, +0.006]).

So the subject-first recipe generalises to subjects the model never saw. It gains on held-out subjects about as much
as on seen ones, without collapsing same-subject records together, and it more than recovers the step-0 loss from
removing the ID lines while clustering less by collection. This is a minimal run (LoRA, 12k pairs, one epoch, no hard
negatives, one seed). The Colab full fine-tune on all 40,843 pairs with mined negatives is the real test.

### 2.6 Colab full fine-tune v3-run1 (2026-10-06)

The model is `isaacmg/genizah-embed-qwen3-0.6b` at tag `v3-run1` (commit 51afe8da), trained on data tag `v3`:
- **Training:** full fine-tune, fp32 master weights with bf16 autocast, all 40,843 v3 pairs; 162 steps at batch 256
  in 84 min on an A100. Hard negatives were mined for 34,391 pairs, 3,352 of them same-subject.
- **Eval:** the frozen v3 eval on the full 60,493-record eligible pool at the production window.
- **Files:** `results/colab_v3-run1/`.

| metric | today (base, production text) | base, semantic text | **v3-run1** |
|---|---|---|---|
| held-out subjects AP (gate, 6) | 0.127 | 0.110 | **0.182** |
| held-out subjects P@10 | 0.334 | 0.313 | **0.420** |
| held-out probe queries AP ("laws of lulav" style) | 0.039 | 0.035 | 0.056 |
| held-out name queries AP | 0.190 | 0.167 | 0.271 |
| linked subjects AP (78) | 0.173 | 0.171 | **0.281** |
| seen subjects AP (662) | 0.127 | 0.127 | **0.378** |
| known-item MRR / R@10 (1,961) | 0.193 / 0.295 | 0.196 / 0.297 | **0.266 / 0.406** |
| within-subject known-item median percentile ↓ | 0.029 | 0.027 | **0.007** |
| collection lift (all) ↓ | 3.85 | 2.80 | 2.76 |
| duplicate/short records in top-10 ↓ | 22.5% | 24.3% | **12.5%** |

Paired bootstrap 95% CIs:
- **Gate:** vs today's search, AP **+0.055 [+0.024, +0.090]**; vs base on semantic text, +0.073 [+0.033, +0.118].
- **Held-out subjects one by one:**
  - Significant: Pesach +0.141, partnership +0.126, Sukkot +0.055, slavery +0.054, geonic academies +0.042.
  - Kashrut: +0.018, not significant.
- **Known-item:** RR +0.073 [+0.046, +0.099] vs today. Every query type improves, including Hebrew implicit
  (MRR 0.085 → 0.169), transliterated (0.145 → 0.203) and specific (0.485 → 0.559).

What it does not fix yet:
1. **Nuanced phrasings of unseen subjects stay weak.** Held-out probe-query AP is only 0.056, while the subject names
   reach 0.27: the model learned subject names more readily than paraphrased concepts for subjects it never saw.
   A production model would train on these subjects too; the held-out score is the generalisation floor.
2. **A small memorisation gap.** Within seen subjects, trained records outscore held-out records by +0.026 [+0.010,
   +0.042] AP (+0.048 for records used as positives). Both rose from about 0.08 to about 0.3, so most of the gain is
   generalisation.
3. **Collection clustering is unchanged by training.** All of the drop from 3.85 to about 2.8 comes from removing the
   ID lines.

Against the local LoRA trial (§2.5): the held-out gain is the same (+0.073 vs +0.074), while seen subjects and
known-item gain far more with full fine-tuning. So generalisation to new subjects is limited by the variety of the
training data, not by model capacity. Levers for v4:
- concept-level paraphrase queries for many more subjects;
- the cleaned Sefaria passage pairs;
- same-collection, different-subject hard negatives;
- early stopping on held-out records.

## 3. Image audit results

### 3.1 Ground truth assembled (and two data-quality defects found on the way)

`build_image_manifest.py` → `image_manifest_v1.json`; `fetch_images.py` downloaded one image per fragment
(max side 1600 px) for a probe set of **5,176 fragments** (168 URLs 404'd):
* **join pairs: 1,661 with images on both sides** — 880 clean FJP literary pairs + 781 PGP documentary pairs
  (fragments sharing a PGP document); 2,032 fragments have ≥1 known partner. 80–88 % of pairs are in the same
  collection, so every metric is also reported for **cross-collection** pairs.
* **scribes:** 501 fragments, 22 PGP scribes with ≥5 fragments (Ḥalfon b. Menashshe capped at 60).
* **script:** KTIV style (Square / Semi-Cursive / Cursive / Naskhi / Rabbinical, n=1,057) and region (n=663).
* **distractors:** 1,499 random fragments stratified to the join set's collection mix.

**Defect 1 — FJP join lists are corrupted by a prefix bug.** A shelfmark that is a string prefix of another
inherits the longer one's join list: `Or.1080 4.5` (a letter/ketubbah) carries the exact list of `Or.1080 4.55`
(Mishnah Sotah) — T-S E1.91, E1.98, E1.102, E2.65 (Mishnah Nazir)…; likewise NLI `577.2/1` ↔ `577.2/16`,
`577.5/2` ↔ `577.5/22`. **619 of 2,411 FJP items with join lists (26 %) match this pattern.** These lists feed the KG's
join notion and the unfinished `joins_vis` view. The probe uses only the 880 pairs from non-suspect items.

**Defect 2 — some FJP image files are mis-filed.** `images/Or_1080_15_4_1r.jpg` shows the CUL label
"Or. 1080.15.3". The yellow CUL label is in nearly every CUL photo, so an OCR pass over it would validate the
image↔shelfmark mapping cheaply.

### 3.2 Global and patch embeddings (I1)

Gallery = 5,176 images. Joins: R@k = known partner within the top k (first partner); AUC = P(join pair more
similar than a random **same-collection** non-join pair). Scribes: P@1 / AUC over scribe-labelled fragments with
same-document pairs excluded (chance P@1 0.078). Script: kNN-10 balanced accuracy (chance 0.20). **Collection =
how well the vector predicts the holding collection (chance 0.083) — high = it encodes the photography setup.**

| encoder / view | joins R@1 | R@10 | R@100 | cross-coll R@10 | FJP R@10 | PGP R@10 | AUC | scribe P@1 | scribe AUC | style | region | collection ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CLIP B/32, whole photo | 0.152 | 0.272 | 0.460 | 0.113 | 0.262 | 0.273 | 0.702 | 0.297 | 0.605 | 0.494 | 0.298 | 0.595 |
| CLIP B/32, crop | 0.147 | 0.254 | 0.458 | 0.119 | 0.235 | 0.268 | 0.706 | 0.243 | 0.608 | 0.462 | 0.268 | 0.499 |
| CLIP B/32, backdrop masked | 0.142 | 0.254 | 0.456 | 0.151 | 0.217 | 0.296 | 0.734 | 0.317 | 0.641 | 0.472 | 0.273 | 0.247 |
| handcrafted (orientation/hinge/stroke width/layout/colour) | 0.185 | 0.311 | 0.461 | 0.113 | 0.271 | 0.350 | 0.699 | 0.268 | 0.543 | 0.462 | 0.221 | 0.533 |
| DINOv2-B, whole photo | 0.261 | 0.397 | 0.590 | 0.148 | 0.361 | 0.433 | 0.794 | 0.309 | 0.589 | 0.472 | 0.316 | 0.528 |
| DINOv2-B, fragment crop | 0.251 | 0.389 | 0.583 | 0.161 | 0.353 | 0.429 | 0.793 | 0.309 | 0.585 | 0.477 | 0.267 | 0.455 |
| **DINOv2-B, backdrop masked** | 0.257 | 0.387 | 0.570 | 0.203 | 0.342 | 0.443 | 0.794 | 0.315 | 0.596 | 0.469 | 0.244 | **0.272** |
| DINOv2-B, 16 ink patches from crop | 0.204 | 0.327 | 0.484 | 0.161 | 0.246 | 0.421 | 0.699 | 0.319 | 0.564 | 0.453 | 0.327 | 0.404 |
| DINOv2-B, 16 ink patches inside mask | 0.249 | 0.384 | 0.550 | 0.196 | 0.338 | 0.436 | 0.757 | 0.365 | 0.594 | 0.391 | 0.352 | **0.245** |
| **Qwen3-VL-8B vision tower** (stock, masked, 512² px, mean-pooled after merger) | 0.203 | 0.314 | 0.470 | 0.183 | 0.279 | 0.360 | 0.758 | **0.417** | **0.644** | **0.502** | 0.292 | 0.258 |
| DINOv2 masked + masked patches | 0.281 | 0.420 | 0.598 | 0.228 | 0.375 | 0.475 | 0.786 | 0.351 | 0.600 | 0.452 | 0.310 | 0.294 |
| **DINOv2 masked + patches + Qwen3-VL** | **0.295** | **0.424** | 0.597 | **0.228** | **0.380** | **0.479** | 0.792 | 0.381 | 0.622 | 0.482 | 0.311 | 0.298 |
| Qwen3-VL tower, **our fine-tuned heb-v21b** (step 1200, masked) | – | 0.325 | – | **0.241** | 0.298 | 0.367 | 0.746 | 0.439 | 0.660 | 0.496 | 0.271 | 0.175 |
| Qwen3-VL tower, **our fine-tuned heb-v22b** (step 1200, masked) | – | 0.326 | – | 0.238 | 0.295 | 0.371 | 0.749 | **0.457** | **0.661** | 0.494 | 0.263 | **0.169** |
| DINOv2 masked + patches + v21b tower | – | **0.434** | – | **0.241** | **0.383** | **0.497** | 0.784 | 0.397 | 0.620 | 0.503 | 0.303 | 0.298 |
| DINOv2 masked + patches + v22b tower | 0.298 | 0.432 | 0.606 | **0.241** | 0.381 | 0.495 | 0.785 | 0.391 | 0.621 | 0.509 | 0.311 | 0.294 |
| SigLIP2-so400m | not run: 7× slower while the GPU is shared with the sibling's VLM pipeline; CLIP-family was already weakest | | | | | | | | | | | |

(Fusions with handcrafted or CLIP features did not help; fusing complementary views did — see last rows.)

What the table says:
* **Backdrop masking is the single most important preprocessing step.** It cuts the collection confound in half
  for every encoder (DINOv2 0.53 → 0.27, CLIP 0.60 → 0.25) while keeping same-collection joins and *raising*
  cross-collection joins (0.15 → 0.20). Every production image vector should be computed on the masked fragment.
* **Two different kinds of similarity.** DINOv2 (generic self-supervised) is best at *physical/page* similarity:
  fragment shape, substrate, layout, so it finds join partners. **The VLM's own vision tower** (trained to read
  Hebrew script) is best at *hand/script* similarity: scribe P@1 0.42 and AUC 0.64, script style 0.50, lowest
  confound. **So "the VLM's image embeddings are useless" is not right.** Mean-pooled after the merger on the
  masked fragment, they are the best handwriting signal of everything tested. The old ColNomic pipeline failed
  because of how it pooled and what it was fed.
* **Our own Hebrew-reading fine-tunes improve the tower further** (2026-10-05, same 5,176 images; the towers come
  from the LM Studio MLX exports, bf16, unquantised). Compared with stock, the v22b tower raises scribe P@1 from
  0.417 to 0.457 and cross-collection joins from 0.183 to 0.238. It also cuts the collection confound from 0.258 to
  0.169, the lowest of any encoder, so it is matching hands rather than photography. v21b and v22b are equal on
  joins, and v22b is better on scribes. This is the starting point for the image fine-tune.
* Fusing the two views gives the best join retrieval: **R@10 0.42, R@1 0.30, cross-collection 0.23**. With a
  fine-tuned tower in place of the stock one, R@10 rises to 0.43 and cross-collection joins to 0.24.
  Literary FJP joins (0.38) stay harder than documentary PGP joins (0.48).
* Absolute numbers are modest: 58 % of fragments don't find their known partner in the top 10 of 5k. Off-the-shelf
  features are a candidate generator, not a join detector. The supervised results in §3.3 show where the headroom is.

![nearest neighbours](../evals/embedding_audit/results/neighbours_dinov2-base__crop.jpg)

*Nearest neighbours (DINOv2-B crop) for six join queries: query, known partner (with its rank), top-5.* What the
generic encoder matches on is visible: the **backdrop** (row 3: every neighbour of a blue-boxed T-S AS fragment is
blue-boxed), **layout** (row 5: two-column bifolios), the yellow CUL label/mylar stitching (rows 1, 4), and for a
Bodleian image on black, nothing useful (partner at rank 3,062). That is why masking the backdrop matters.

### 3.3 Line-level handwriting (I2) and a supervision pilot (I3)

The KTIV Kraken line crops on the NAS (49k lines, 1,699 manuscripts, 160 px tall) remove the photograph
almost entirely. **I2**: 600 manuscripts × 2 pages × 3 lines = 3,600 lines; each line must retrieve lines of the
**same manuscript on a different page** (same-page lines excluded). Line vector = mean of square tiles.

| line encoder | P@1 (other page, same MS) | mAP | chance P@1 | AUC vs same-library lines | AUC vs other-library lines |
|---|---|---|---|---|---|
| DINOv2-B, 6 tiles | 0.344 | 0.282 | 0.0008 | 0.822 | 0.846 |
| CLIP B/32, 6 tiles | 0.226 | 0.182 | 0.0008 | 0.823 | 0.864 |
| our v22b tower, final (merger) output, whole line at 96 px | 0.067 | 0.054 | 0.0008 | 0.687 | 0.666 |
| our v22b tower, **deepstack layer 8** | **0.371** | **0.305** | 0.0008 | 0.761 | 0.768 |
| our v22b tower, layer 16 | 0.248 | 0.191 | 0.0008 | 0.739 | 0.732 |
| our v22b tower, layer 24 | 0.153 | 0.120 | 0.0008 | 0.712 | 0.704 |
| DINOv2 + v22b merger output | 0.355 | 0.288 | 0.0008 | 0.799 | 0.813 |

**Readout matters, and in opposite directions at the two scales** (2026-10-05). On a single line, the tower's final
merger output encodes *what is written* (the VLM was trained to read). Two lines of the same manuscript say different
things, so it scores near chance. The early layer-8 features encode *how* it is written and beat DINOv2. On whole
fragments (§3.2, same 5,176 images) the averaging over a page washes the text content out, and the order flips:

| v22b tower on masked fragments | joins R@10 (cross-coll.) | scribe P@1 | collection ↓ |
|---|---|---|---|
| merger output | 0.326 (0.238) | **0.457** | **0.169** |
| layer 8 | 0.330 (0.212) | 0.357 | 0.199 |
| DINOv2 masked+patches + merger | **0.432** (0.241) | 0.391 | 0.294 |
| DINOv2 masked+patches + layer 8 | 0.425 (0.228) | 0.359 | 0.291 |

So: lines → layer 8, fragments → merger output. A trained image encoder should read both, e.g. pool layer 8 and the
merger output and learn the mix, rather than commit to one.

Same-library and other-library AUCs are close, so at line level the signal is the hand, not the library's
camera. This is the representation joins and scribe attribution should be built on.

**I3 — does a little supervision help?** A 2-layer projection head (1536→768→256) trained with a supervised
contrastive loss (positive = same manuscript) on **frozen** DINOv2-B line features from 9,087 lines of 782 *other*
manuscripts (disjoint from the I2 test set), 40 epochs, **≈30 s on CPU**:

| same 3,600 held-out lines (4 tiles) | P@1 | mAP |
|---|---|---|
| raw DINOv2-B | 0.274 | 0.219 |
| **+ trained head** | **0.436 (+59 %)** | **0.424 (×1.9)** |

A head on frozen features is the weakest form of fine-tuning; that it nearly doubles mAP on unseen manuscripts
says that fine-tuning the backbone on Genizah handwriting (Step 2 of §6) is very likely worth the GPU time.

The same head on our fine-tuned **v22b tower** (2026-10-05, same 9,087 train lines / 3,600 held-out test lines,
tower lines at 96 px; DINOv2 rows use 4 tiles as above):

| frozen line features | raw P@1 / mAP | + trained head P@1 / mAP |
|---|---|---|
| DINOv2-B | 0.274 / 0.219 | 0.436 / 0.424 |
| v22b merger output | 0.067 / 0.054 | 0.250 / 0.239 |
| v22b layer 8 | 0.371 / 0.305 | 0.448 / 0.431 |
| v22b layer 16 | 0.248 / 0.191 | 0.426 / 0.396 |
| v22b layer 24 | 0.153 / 0.120 | 0.276 / 0.263 |
| DINOv2 + v22b merger | 0.355 / 0.288 | 0.495 / 0.473 |
| DINOv2 + v22b layer 8 | 0.349 / 0.286 | 0.548 / 0.526 |
| **DINOv2 + v22b layer 16** | 0.350 / 0.285 | **0.570 / 0.537** |
| stock tower merger / layer 8 / 16 / 24 | 0.136 / 0.364 / 0.273 / 0.167 (P@1) | 0.320 / 0.458 / 0.470 / 0.333 (P@1) |
| DINOv2 + stock merger / layer 8 / layer 16 | 0.352 / 0.363 / 0.371 (P@1) | 0.493 / 0.545 / 0.565 (P@1) |

DINOv2 and the tower's early/middle layers carry complementary handwriting signals. Fused raw, they are no better than
either alone, but a small trained head combining them (best: DINOv2 + layer 16) retrieves a same-manuscript line
from another page as its top hit for 57% of held-out lines (chance 0.08%), from frozen features alone. At line scale,
the stock and the fine-tuned v22b tower are equivalent within single-seed noise (±0.01). The Hebrew-reading fine-tune
pushed the merger output further toward *content*: stock merger P@1 0.136 vs v22b 0.067. Its gains show at fragment
scale (scribe P@1 0.457 vs 0.417, §3.2). The image fine-tune's head should therefore take
DINOv2 features alongside the tower taps. (The I3 head trains with any same-manuscript line as a positive, including
same-page lines; the test protocol counts other-page lines only. This holds equally for every row.)

## 4. Training data: what we have vs what we need

Counts are from direct parses of the files on disk (2026-10-02). "Pairs" means (anchor → positive)
examples for contrastive training; "labels" means class/grouping supervision.

### 4.1 Text — what we HAVE (corpus-internal)

| Source | Supervision it yields | Size | Location | Caveats |
|---|---|---|---|---|
| **KTIV scholarly "frames"** (Sussmann/Lieberman/Zulay identifications) | identification ↔ fragment: `[Talmud Bavli]: Sukkah 29a–30b`, `[Hilkhot ha-Rif]: Sukkah`, `(Hoshanot): ''למען אב ידעך''` | 20,266 frames on **5,661 records**; Bavli 1,070 records (941 with KTIV transcription), Mishneh Torah 799 (764 transcribed), Bible 974 (4), Piyyut ≈1,040 (≈40), Mishnah 275 (85), Rif 213 (15) | merged jsonl → `sources.ktiv.scholarly_entries` | the gold source of topic labels; the audit's 13 topic labels (403 Sukkot, 323 Shabbat, 222 kashrut, 146 marriage, 117 Pesach…) all come from here |
| KTIV domain hierarchy | genre labels (Bavli, Piyyut, MT, Liturgy, Bible, Letters…) | 29,084 entries on 10,670 records | same | KTIV `subjects` are useless ("Cairo Genizah fragments" on 11.7k) |
| FJP/FGP "Title. Domain" descriptions | description ↔ genre/ref (`Genesis 20: 6-9`, `משנה, טהרות פ"ו`) | 49,037 descriptions; ≈9.5k Bible, ≈4.5k Piyyut, ≈2.7k Liturgy, ≈1.2k Bavli, ≈1k Mishnah | `fjp_all/merged_princeton_friedberger_all_documents_final.json` | refs are free text; transcriptions on only 5.8k |
| PGP documents | English description ↔ fragment; doc type (Letter 11.4k, Legal 8.2k, List 5.7k, Literary 2.7k…) | 36,204 docs (avg ≈295 chars) | `pgp-metadata/data/documents.csv` | mostly documentary; tags noisy |
| **PGP editions + translations** | Hebrew/Judeo-Arabic transcription ↔ English translation/description (cross-lingual pairs) | editions for 7,649 docs; EN translations ≈1,965 | `pgp-metadata/data/footnotes.csv` | the best JA↔EN pairs we own |
| **Scholarship pages** | book page ↔ fragments it discusses | 4,945 pages / 73 PDFs; 2,144 pages carry 10,323 shelfmark mentions; 779 pages contain editions | `raw_data/.../academic_literature/**/page_*_structured.json` | shelfmarks not normalised (`T_S_Ar_53_12`); the natural primary↔secondary bridge pairs |
| KG relations v3 | page-anchored triples with evidence quotes | 12,708 accepted; 4,622 touch a Fragment | `academic_literature/relations_v3/` | LLM-extracted; some templated |
| PGP footnotes / biblio.json | fragment ↔ (book, page) | 7,327 footnotes; 9,908 shelfmarks × 28,758 citations | `pgp-metadata`, `retained_json/biblio.json` | printed→sequence page mapping needed |
| AI transcriptions | VLM reads (Kraken-agreed lines) | 14,405 docs (v21b 19.3k images / 46.9k agreed lines) | `raw_data/.../ai_reads/*.jsonl` | mostly documentary queues; ≈230 Bible / 41 Bavli docs |
| Holiday curation | topic → fragments | Sukkot seed **125 human-verified** (74 never name Sukkot), KTIV Sukkot queues 724 + 294, Yom Kippur 13 | `origin/feature/sukkot-page:…/sukkot_seed.json`, `ktiv-scraper/scraping_csv/` | the only human topic labels; used here as **held-out test only** |
| Eval cases | query → bibliography targets | 20 agentic cases (11 + 4 cross-lingual + 5 multi-turn) | `evals/agentic_rag_*.json` | must stay held out; no fragment-level targets |

### 4.2 Text — what we NEED and don't have

| Need | Why | How to get it | Size target |
|---|---|---|---|
| **Concept knowledge**: English/transliterated practice terms → topic ("laws of lulav", "four species", "Kol Nidre", "bedikat chametz") and Hebrew content → topic | Probes T1/T2 show every off-the-shelf model lacks it; our corpus contains almost no English text about religious content (PGP is documentary; FJP translations are empty) | **Sefaria**: topics graph (e.g. *Lulav* topic → 120 linked sources incl. Mishnah Sukkah 4:7, MT Shofar-Sukkah-Lulav 7:23) + bilingual texts of Mishnah/Tosefta/Bavli/MT/SA/siddur. API verified reachable. License per version (many PD/CC-BY; Davidson Talmud is CC-BY-NC) | ~50–150k topic↔passage and HE↔EN pairs |
| **Canonical text for identified-but-untranscribed fragments** | 974 Bible + ~1k piyyut + 190 Mishnah/Rif framed records have no text; the vector sees only the label | Fetch Sefaria text by the KTIV/FJP ref (`Genesis 20:6-9`, `Bavli Sukkah 29a`), use as doc-side enrichment and as positives | ~3–10k fragments |
| **Realistic queries** | We have zero real user queries on disk (logs empty; chat feedback is in ES index `genizah_chat_feedback_v1`, unread — prod reads were blocked) | (a) export questions from `genizah_chat_feedback_v1`; (b) LLM-synthesised queries per document (Claude via API — still blocked on `ANTHROPIC_API_KEY` — or the local 35B off-hours with Isaac's OK); (c) round-trip filtering | 3–5 queries × 20–30k docs |
| **Hard negatives** | in-batch negatives alone teach "festival vs letter", not "Sukkot vs Pesach" | mine top-k from the current model, drop same-topic/same-frame hits (false-negative filter) | 1–3 per pair |
| **A human-validated eval set** | the concept probe here is hand-written by me (Claude) and needs a scholar's review | Isaac/scholar writes or approves ~300 queries; pool top-20 of BM25 + current + candidate models; judge (Opus judge or Isaac) | 300 queries, graded relevance |

### 4.3 Images — what we HAVE

| Source | Supervision | Size (with images) | Caveats |
|---|---|---|---|
| PGP multi-shelfmark documents (fragments sharing a PGP id) | **join pairs** (documentary) | 2,231 pairs (groups ≤6); 781 with images on both sides in the v1 subset | groups >6 dropped (dossiers, not physical joins) |
| FJP `joins_data.joinedManuscripts` | **join pairs** (literary) | 2,411 items with lists → 1,567 resolvable pairs → **880 clean pairs with images** | **≈26 % of lists (619/2,411) are corrupted by a prefix bug** (see §3); 2,168 shelfmarks unresolved |
| PGP person→document `Scribe` relations | **scribe labels** | 2,324 rows / 104 scribes → 1,501 imaged fragments; 22 scribes with ≥5, 15 with ≥10 (Ḥalfon b. Menashshe 848) | documentary hands only; heavily skewed |
| KTIV palaeographic fields | script style (Semi-Cursive 1,117 / Square 644 / Cursive 97 / Naskhi 80 / Rabbinical 63), region (Oriental 903 / Spanish 154 / Yemenite 28…), material (Paper 4,792 / Vellum 1,013) | ~3–6k fragments | coarse classes |
| KTIV "same hand as …" notes | scribe/same-MS pairs in free text | 73 notes | needs parsing |
| KTIV Kraken line crops (NAS) | **same-manuscript line groups** | 49,352 lines, 1,699 MSS, 3,587 pages | local, clean, 160 px lines |
| Images | pixels | GCS: FJP 106k files / 45.6k docs, KTIV 49k / 9.3k docs; 54,163 merged records resolve to an image URL | some FJP images are mis-filed (§3) |
| External | TAU "Bag of Bags" Genizah join benchmark (Gogawale…Dershowitz 2026, CC BY 4.0, `github.com/TAU-CH/midrash_bob`) | public join retrieval set | not yet downloaded |

### 4.4 Images — what we NEED

* **Clean join ground truth at scale**: fix the FJP prefix bug, finish shelfmark resolution, add the TAU set, mine
  KTIV "מצטרף / may join" notes; target ≥5k verified pairs with images.
* **Scribe labels beyond documentary hands** (literary scribes are where joins live): KTIV `author_characteristics.scribe`
  (107), colophons, scholarly attributions in the bibliography pages.
* **Image↔shelfmark integrity check**: OCR the yellow CUL shelfmark label in each photograph and compare with the
  filename (found `Or_1080_15_4_1r.jpg` showing label "Or. 1080.15.3").
* **Compute**: self-supervised pre-training on ~150k Genizah images needs a GPU box (Colab A100 per the sibling's
  policy); local is fine for feature extraction/eval only.

## 5. Step-by-step plan — text embedder

Each step has an exit gate. Steps 0–1 are cheap and should happen whether or not we fine-tune;
without them a better embedder cannot show up in the product.

**Step 0 — Plumbing so embedding gains reach users (days, no training).**
1. Fix the agent's `primary_hybrid` crash (`semantic_weight` vs `semanticWeight`) — fragment hybrid search from chat
   currently always fails. (Known; on the chat-false-negative plan.)
2. Make the fragment hybrid's keyword leg BM25-graded (today a constant bonus) and fuse with RRF like the
   bibliography path, so semantic and lexical rankings combine instead of the semantic order dominating.
3. Concept-implication lexicon (narrower term → topic) applied **to the query text that is embedded**, not only to the
   lexical leg: lulav/etrog/hadas/aravah/schach/sukkah/hoshanot/Shemini Atzeret/Simhat Torah → Sukkot; chametz/matzah/
   haggadah/seder/omer-start → Pesach; Kol Nidre/Avodah/Neilah/Yoma → Yom Kippur; … Fix the substring matcher
   (`get` ⊂ "together"). The `oracle_expansion` row in §2 is the ceiling of this step.
4. Doc-side enrichment at index time: render KTIV frames as topic words ("Talmud Bavli, tractate Sukkah — festival of
   Sukkot") and fetch canonical text (Sefaria) for identified-but-untranscribed fragments into the embedded string.
   Re-embedding is needed anyway for step 5.
   *Gate:* T2 framed-pool nDCG@10 and Sukkot-gold recall improve; no regression on the agentic eval.

**Step 1 — Build the evaluation before the model (1–2 weeks of Isaac's time is the bottleneck).**
1. Keep the probes built here as regression tests: concept probe (130 nuanced queries + 82 name variants / 15 topics), T2 framed-pool topic
   retrieval (5,661 identified records, 11 topics), Sukkot gold (125, 74 without the word), BM25 baseline.
2. Scholar-validated query set (~300): festivals/halakha/liturgy/piyyut/documentary/people/places, English + Hebrew +
   transliteration variants; pooled judgments (top-20 of BM25 + current + candidates).
3. Export real questions from `genizah_chat_feedback_v1` (needs a production read — Isaac's call).
   *Gate:* the set exists and reproduces the failures seen in §2 on the current model.

**Step 2 — Assemble training data (≈1 week engineering).**
Mixture target ≈150–250k pairs, every pair tagged with its family for ablations:
| family | source | anchor → positive | share |
|---|---|---|---|
| synthetic nuanced queries | LLM per document (Claude API preferred; local 35B only off-hours with approval) + round-trip filter | query → record text | 35 % |
| concept bridge | Sefaria topics + bilingual texts | topic/English practice term → Hebrew/Aramaic passage; EN ↔ HE passage | 25 % |
| identification labels | KTIV frames, FJP titles/domains, Bodleian TEI | "Bavli Sukkah", "מסכת סוכה", "Hoshanot" → record | 15 % |
| content → label | transcription spans (KTIV 1.6k framed + FJP 5.8k + AI reads) | Hebrew span → label-only text | 10 % |
| primary↔secondary | scholarship page ↔ fragment (10.3k mentions), PGP footnotes | page passage → record; record → page | 10 % |
| cross-lingual | PGP editions ↔ translations/descriptions | JA/HE edition → EN description | 5 % |
Hard negatives mined with the current model, minus same-frame/same-topic hits. **Split by fragment and by
manuscript group** (joins!) so test fragments never leak through a join partner; the Sukkot seed and every eval
set stay out of training entirely.

**Step 3 — Train (hours of A100, not local).**
* Base `Qwen3-Embedding-0.6B` (keeps the 1024-d contract, CPU-servable on the MBP). Full fine-tune, lr ≈1e-5, 1–2 epochs,
  InfoNCE with in-batch + mined negatives, effective batch ≥512 via GradCache (`CachedMultipleNegativesRankingLoss`),
  max_seq 512 (median doc is 209 chars), keep the production query instruction (§2 shows a "domain" instruction
  hurts zero-shot). LoRA is enough for a pilot (§2.3) but full FT is cheap at 0.6B.
* Benchmark Qwen3-Embedding-4B on Colab next to the fine-tuned 0.6B (not testable locally, §2.1). Only switch if the
  0.6B ceiling is the limit: 4B costs ≈7× query latency on the MBP CPU and ≈8 GB RAM in the embedding container.
* Run on Colab A100 (the sibling project's rule: training never runs locally). The local Studio is fine for the
  LoRA-scale pilot only.
  *Gate:* beats base on every §2 metric; no regression on known-item / agentic eval; cross-lingual not worse.

**Step 4 — Ship (a production deploy — needs Isaac's go).**
1. New contract: weights in a private HF repo (e.g. `isaacmg/genizah-embed-v1`) pinned by revision; update
   `embedding_models.py` defaults + `requirements` stay pinned; bump the canary.
2. Re-embed **both** indexes into new versions (fragment `genizah_merged_v9`, bibliography `…_0.8`) — the backend's
   canary check uses one shared embedder. ≈136k texts ≈ 30–60 min on the Studio GPU.
3. Switch `ELASTICSEARCH_INDEX` / bibliography index together with the embedding service image; rerun the viz precompute.
4. Keep v8 + the old embedder image for instant rollback.

## 6. Step-by-step plan — image similarity, joins, scribes

**Step 0 — Use the right representation (this audit, §3).** The backdrop-masked fragment, not the whole photograph;
handwriting features from ink-bearing patches/lines plus the VLM tower; never a mean of a late-interaction model's
patch vectors. This alone would make a usable "visually similar fragments" feature today: fused DINOv2 + Qwen3-VL
vectors on masked fragments in an ES `dense_vector` field, presented as suggestions.

**Step 1 — Ground truth (≈1 week).**
1. Fix the FJP joins prefix bug at the scraper/merge level; recompute join groups; resolve the 2,168 unmatched
   shelfmarks with `ShelfmarkNormalizer`.
2. Import the TAU Genizah join benchmark as an external test set.
3. Validate image↔shelfmark mapping by OCR of the yellow CUL label (cheap, catches mis-filed photos).
4. Build scribe sets: PGP scribes + KTIV scribe/"same hand" notes; same-manuscript groups from KTIV pages/lines.

**Step 2 — Train a Genizah handwriting encoder (A100 days, not local).**
* Two complementary encoders, not one: a **hand/script encoder** and a **physical/page encoder**. For the hand,
  start from the VLM vision tower (best scribe/script signal here). Our own fine-tuned Hebrew VLMs (vision LoRA
  trained since v1.7) are the obvious next probe. That needs a merged bf16 checkpoint, which the sibling's
  eval harness produces at every merge, so ask it to keep one. For the page, start from DINOv2.
* Backbone for training: DINOv2 (or DINOv3) ViT-B/14 and/or the Qwen3-VL tower. Self-supervised continued pre-training on ink-bearing patches from the
  ~150k GCS images (grayscale/binarisation/colour-jitter so photography setup is not learnable), then metric learning
  with positives = same manuscript page/line, join partners, same scribe; negatives drawn **from the same holding
  collection** (forces it off imaging cues). Aggregate local features (VLAD/GeM) — the writer-retrieval literature
  (Raven et al. 2024; Peer et al. 2023) shows local foreground features + VLAD beat a CLS token.
* Evaluate on held-out joins (cross-collection subset reported separately), scribes (same-document pairs excluded),
  line-level same-MS retrieval (I2), script style/region kNN.

**Step 3 — Join discovery as a pipeline, not a single embedding.**
1. Candidate generation: ANN over handwriting vectors **∧** compatible metadata (material, script style/region,
   lines per page, page size) **∧**, for identified literary texts, text adjacency (consecutive Bavli amudim, Bible
   verses, Mishnah chapters). The text-adjacency heuristic alone already proposes 42 Bavli pairs with no metadata
   conflict (`results/join_candidates_bavli_review.html`, unverified; 27 of the 42 are Berakhot, so expect low precision).
2. Rerank with a pairwise model (both images + metadata), then a scholar review queue (the unfinished `joins_vis` UI).
3. Accepted joins feed back as training positives.

**Step 4 — Scribe attribution** is the same encoder with a kNN/prototype head over labelled scribes, reported as
"similar hand to …" suggestions with evidence, never as an attribution.

## 6b. Phase 2 (started 2026-10-04): building the fine-tune

Decisions taken by Isaac on 2026-10-04:
- ES reads are allowed.
- Sefaria is fine to use.
- Synthetic queries are written by Claude sub-agents.
- Isaac validates queries in a review UI.
- Both indexes get re-embedded.
- Training runs on Colab A100; artifacts go to private HF repos under `isaacmg`.

| piece | status | where |
|---|---|---|
| Synthetic queries, round 1 | 1,547 records sampled; 1,424 got 4 queries each (implicit / topical / Hebrew / specific), 123 skipped as too thin. 287 eval-pool records (incl. the Sukkot seed) and 1,137 training records | NAS `synthetic_queries/r1/` |
| Synthetic queries, round 2 | 1,787 more training records → 1,651 with 4 queries (136 skipped). Total across rounds: 3,075 records, 12,300 queries | NAS `synthetic_queries/r2/` |
| Writing spec | v2: four query types, explicit skip/grade/inference rules | `evals/embedding_audit/query_writing_spec.md` |
| Review UI | 587 records / 2,348 queries; verdicts stored in the artifact's db | https://claude.ai/artifact/6GWLqX4VZXdX3FFTr11J5g |
| Baseline on held-out synthetic queries (known item among 73.5k; R@10) | implicit 0.18 · topical 0.35 · Hebrew 0.23 (BM25 0.31) · specific 0.58 | `results/t3_synthetic_qwen3-0.6b.json` |
| Sefaria | full topic graph (4,412 topics with links: lulav *participates-in* Sukkot, Passover *has-participant* haggadah/chametz/matzah/four cups…), curated passages (≤10 per topic) and texts for 2,467 KTIV-frame refs (Bavli/Mishnah/Bible/MT) | NAS `sefaria/` |
| Scholarship ↔ fragment pairs | 3,302 resolved mentions on 2,140 bibliography pages (read-only from ES) | NAS `scholarship_pairs.jsonl` |
| Training mix | **v1** (corpus + synthetic r1+r2 + scholarship): 30,607 pairs — label 12,556, synthetic 11,151, content 2,610, scholarship 4,290. **v2** = v1 + Sefaria pairs (pending the curated-text pull), so the Colab runs double as a with/without-Sefaria ablation | NAS `hf_dataset/v1/`, `build_training_mix.py` |
| Colab | `colab/train_genizah_embedder.ipynb` + `train_embedder.py`. Hard negatives use false-negative guards; full FT with CachedMNRL; eval of base vs tuned on held-out records (T3 by type, T2 topics, Sukkot gold); pushed to a private HF repo with its eval report. CPU smoke test passed for load/eval/mining. | `evals/embedding_audit/colab/` |
| Catalogue side product | 139 records where the query writers found the catalogue label contradicting the transcription (e.g. "Proverbs 5–6" whose text is eruv law) | `results/catalogue_flags_from_query_writing.csv` |

## 7. Decisions needed from Isaac

_Decided on 2026-10-04: items 1–3 and 5 (see §6b). Item 4: Isaac validates in the review UI._

1. **Production reads for evaluation.** Should future audits be allowed read-only access to the MBP's ES (index
   stats, the live v8 vectors, `genizah_chat_feedback_v1` questions)? This audit was blocked from it and worked
   from the merged JSONL instead. A permission rule for read-only `_search`/`_cat` on the MBP would unblock it.
2. **Synthetic-query generator.** Claude via API (needs `ANTHROPIC_API_KEY`; best quality, no load on the Studio) or
   the local 35B in off-hours (free, but contends with prod chat and the sibling pipelines)?
3. **Sefaria as training data.** OK to pull Sefaria topics and bilingual texts? (Licences vary per text version;
   Davidson Talmud is CC-BY-NC.)
4. **Who validates the eval set?** About 300 queries need a scholar's eye, ideally yours: the concept probe here was
   written by Claude.
5. **Embedding contract change.** A fine-tuned embedder means re-embedding both indexes and deploying a new
   embedding image on the MBP. When, and do we serve two embedders during transition?
6. **Image track priority.** Joins/scribes need (a) the FJP prefix-bug fix, (b) a GPU budget for backbone
   fine-tuning (Colab A100), and (c) a line-segmentation pass over fragment images. The sibling's Kraken service on
   :8002 could do (c) but is in continuous use by the consensus pipeline.

## Appendix A. Files and how to re-run

All in `evals/embedding_audit/` (untracked; nothing committed). Every probe runs under `guard.py`. The guard:
- holds an audit-wide mutex, so only one process runs at a time;
- enforces memory, swap and disk gates;
- yields to the sibling's merge/convert and to its GPU checkpoint evals.

`AUDIT_DEVICE=cpu AUDIT_THREADS=4` runs any probe on CPU.

| file | what |
|---|---|
| `guard.py` | resource/sibling-aware wrapper (`--max-gb`, `--cpu-only`) |
| `embed_utils.py` | model registry, cached text encoder, device selection |
| `concept_probe_v1.json`, `probe_text_concepts.py` | T1 concept probe |
| `build_corpus.py` | offline corpus = production embedding text + topic labels (→ NAS `corpus_v1.jsonl`) |
| `embed_corpus.py`, `eval_corpus.py`, `eval_bm25.py` | T2 corpus retrieval (dense + BM25), oracle expansion, neighbourhood structure |
| `build_train_pairs.py`, `train_text_lora.py` | P1 corpus-only LoRA pilot (80/20 split by doc; Sukkot seed held out) |
| `build_image_manifest.py`, `fetch_images.py` | image ground truth (joins/scribes/script) + 5,176-image probe set on the NAS |
| `image_features.py`, `handcrafted_features.py`, `eval_images.py`, `render_neighbours.py` | I1 image probes |
| `line_probe.py`, `line_head_pilot.py` | I2 line-level handwriting, I3 supervised-head pilot |
| `join_candidates_text.py`, `render_join_candidates.py` | Bavli text-adjacency join candidates + review sheet |
| `run_queue_v*.sh`, `run_cpu_*.sh` | the queues actually run (v1–v4 superseded; v5–v7 + CPU queues produced the results) |
| `results/*.json`, `results/*.jpg`, `results/*.html` | raw results |

NAS (`/Volumes/home/studio_offload/genizah_search_embedding_audit/`): `corpus_v1.jsonl`, `corpus_vectors/`,
`image_manifest_v1.json`, `images/max1600/` (5,176 JPEGs), `cache/` (text/image/line features), `hf/` (model
weights), `pilot_text_v1/` (pairs, splits, pilot model), `logs/`.
