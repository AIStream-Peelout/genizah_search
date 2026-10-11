# Handoff: v9 holiday-search regression, and what the next model and index need (2026-10-08)

**For:** the owner of `historical-document-analysis` (index builder, `semantic_v3`) and of `evals/embedding_audit/` on
`eval-retrieval` (read at 7c85a29). **From:** the serving side (Isaac's MBP production host). Same repo,
`AIStream-Peelout/genizah_search`; the MBP checkout is named `terraform_gcp_search`, and this folder is uncommitted
there (branch `seo-plan-2`). Isaac decides where it is committed.

| File | What it is |
|---|---|
| `holiday_eval_seed.json` | 54 queries with v8 / v9 / fusion P@10 and top-10 lists, relevance regexes, v8 and v9 silver lists, gates, probes (keys: section 8) |
| `regenerate_silver.py` | stdlib, read-only: `build` silver lists for a new index; `score` a run and print the gate |

## 0. Summary

- **Serving is correct:** the live top 20 equalled a local exact re-score in 108 of 108 runs.
- **The regression comes from the model; the label mechanism behind it is likely (read from code).** v9 discounts
  `Description:` and trusts `Alternate titles:`, which 80.1% of eligible records lack. It also maps bare holiday names
  to their scripture or tractate.
- **Size, 44 holiday queries:** description-only P@10 0.489→0.077; Purim 1.0→0.0, Passover 1.0→0.3, פסח 1.0→0.2.
  Strict P@10 0.686→0.657 is within noise (paired SE 0.053). Controls 0.97→0.93.
- **Site state:** the beta switch has been hidden since 2026-10-08; everything serves v8 with stock
  Qwen3-Embedding-0.6B.

## 1. Asks, in order

**[beta]** = needed to re-enable the opt-in beta on a new model; **[default]** = only to make it the site default.

1. Run the smoke test (6.1, no ES).
2. Add the eval items P4 to the frozen eval, before training. **[beta]**
3. Fix labels (P1) and negatives (P2). **[beta]**
4. Text function and `semantic_text_hash` (P3.1–2) **[beta]**; eligibility and window (P3.3–4) **[default]**.
5. Train; pass probes M1–M2 (6.5) on the checkpoint; build the index.
6. Produce the run (6.2) and the new silver lists (6.3); run `score --gate beta`.
7. Send back the run file (with hashes), both printouts and the index `_meta`. The site re-checks, measures the
   default gate and its own fusion, and decides.

## 2. What was measured

| | Index | Embedder | Text | Scoring |
|---|---|---|---|---|
| v8 | `genizah_merged_v8` (73,543) | `Qwen/Qwen3-Embedding-0.6B@97b0c61` | production, with ID lines | exact cosine, all records |
| v9 | `genizah_merged_v9` (63,652 of 78,113 eligible) | `isaacmg/genizah-embed-qwen3-0.6b@dbfab73` | `semantic_v3` | exact cosine, eligible only |
| fusion | offline only | RRF (k=60) over the top 100 of v8, v9 and BM25 (production fields, no aliases, no dedupe) | | |

- **Method.** Live `POST /search` on 2026-10-08 (v9 via `index_name: "genizah_merged_v9"`; the response said
  `semantic_beta: true`), re-scored with the production `script_score`: same top 20 in 108/108 runs. The API path is
  closed now; reproduce with the embedder plus ES (6.2).
- **Queries.** 44 holiday queries over 9 holidays, plus 10 controls. 32 are **bare names** (english 4, transliterated
  16, hebrew 10, Ashkenazi 1, typo 1); 12 are descriptive. Isaac's screenshot queries: "Pesach", "Rosh Hashnnah",
  "simchat torah".
- **Relevance ("silver")** comes from automatic regexes over each index's `description` and `alt_titles`, not human
  judgement. Relevant = in the **union** of the v8 and v9 sets (seed `relevance_rules`). **strict:** topic named
  outside tractate or law-code context. **broad:** strict plus tractate/law-code mentions plus Pesahim and Sukkah,
  never scripture. **desc-only** (`desc_strict`): strict on `description` alone, which is all the result card shows.

| Mean P@10 | v8 | v9 | fusion |
|---|---|---|---|
| holiday strict | 0.686 | 0.657 | 0.736 |
| holiday broad | 0.755 | 0.802 | 0.814 |
| holiday desc-only | 0.489 | **0.077** | 0.407 |
| control strict | 0.97 | 0.93 | 0.99 |

- **Noise.** Paired against v8: v9 strict −0.030 (SE 0.053, W/L/T 14/17/13), fusion strict +0.050 (SE 0.029),
  v9 desc-only −0.411 (SE 0.046). One 0.1 step moves a 44-query mean by 0.0023; that is resolution, not noise.
- **Worst sets (strict):** Purim 0.70→0.23, Pesach 0.93→0.63, English names 0.60→0.28 (all tables: seed `aggregates`).
  P@20 0.635→0.658, MRR@100 0.883→0.772, capped R@100 0.437→0.530 (relevant in top 100 / min(|relevant|, 100), which
  is P@100 for 7 of 9 sets and not the frozen eval's recall).
- **Keep the gains:** descriptive queries 0.683→0.850; שמחת תורה 0.2→1.0; frozen eval focus AP 0.064→0.198.
- **Proxy set** (`supplementary_proxy_set`, lenient rule, not comparable): "Peysakh" 0.8→**0.0**, "Feast of Booths"
  0.4→1.0.

## 3. Root causes

### 3.1 v9 trusts `Alternate titles:` and discounts `Description:`

This is the main cause of the desc-only collapse and the buried prose records; most of the strict loss is 3.3. Same
content under each label, document mode, scored against the query:

| Query (content) | v8 Description / Alt titles | v9 Description / Alt titles |
|---|---|---|
| Pesach ("Pesach") | 0.762 / 0.741 | **0.279 / 0.681** |
| Passover Haggadah (same) | 0.757 / 0.692 | **0.271 / 0.674** |
| Rosh Hashanah ("Prayers for Rosh ha-Shanah") | 0.636 / 0.577 | **0.240 / 0.552** |
| divorce document get ("Bill of divorce (get).") | 0.763 / 0.572 | 0.460 / 0.630 |
| letter from a merchant in India ("Letter from a merchant in India…") | 0.840 / 0.714 | 0.724 / 0.719 |

- The `Description:` penalty is smaller or absent for documentary subjects (all six pairs: seed `probes.label_swap`).
  Likely, not tested: their PGP tags pass `--min-pgp 20` (India 489, Marriage 411, divorce 117), so they were
  labelled on records that hold the subject under `Description:`.
- 80.1% of eligible v9 records (50,961) have no `alt_titles`, including every non-KTIV one. Yet all 50 of v9's top 50
  have them for "Pesach", "Passover Haggadah", "Rosh Hashanah" and "Simchat Torah" (v8: 30, 2, 13, 49), and 88% of
  its strict-relevant holiday hits name the holiday only there (v8: 33%).
- Median cosine to "Pesach" among records whose description or `alt_titles` name Pesach: v9 0.451 / 0.208 / 0.153,
  v8 0.519 / 0.469 / 0.346, for KTIV-any / FJP-without-KTIV / PGP-only (v9 n=147/225/36; v8 103/227/39).

Exact ranks of records that name the holiday in `description` and have no `alt_titles` (all 10 cases, with cosines:
seed `probes.rank_regressions`):

| Record (description) | Query | v8 | v9 |
|---|---|---|---|
| Paris_AIU_IV_A_256 (הגדה של פסח) | Pesach | 1 | 179 |
| Manchester_JRL_A_83 (Passover Haggadah) | Passover Haggadah | 14 | 3,853 |
| Philadelphia_CAJS_Halper_213 (Haggadah for Passover) | Passover Haggadah | 7 | 625 |
| Cambridge_CUL_T_S_AS_62_1 (Poems for Passover) | Pesach | 15 | 4,452 |
| Cairo_ENL_MIAC_74 (Prayers … Rosh ha-Shanah) | Rosh Hashanah | 1 | 92 |
| Cambridge_CUL_T_S_AS_161_14 (דיני שופר בראש השנה) | Rosh Hashanah | 4 | 1,978 |
| Philadelphia_CAJS_Halper_251 (Mahazor for Hoshana raba … Simhat Torah) | Simchat Torah | 47 | 1,807 |

### 3.2 Likely mechanism: labels and negatives

**Read from the code at 7c85a29** (the training rows were not available):
- **Labels** (`build_subjects.py`). `topic:` ids come from the case-sensitive `TOPIC_RULES_V2` (line 58), applied only
  to KTIV frames (line 1056) and, as weak labels, KTIV general titles (line 1062). PGP tags need ≥ 20 records
  (`--min-pgp 20`, line 1031): Passover has 11, the other holidays 3–5, so all are dropped. The Sukkot gold seed
  (125 ids, `docs/planned_features/sukkot_page/sukkot_seed.json` on `feature/sukkot-page`) adds `topic:sukkot`
  (line 1058) to about 60 non-KTIV records (read-only ES lookup), but `build_eval_v3.py` final mode holds every gold
  record out of training. **So no FJP-only or PGP-only record in the training pool carries a holiday subject.**
- **Mined negatives** (`colab/train_embedder_v3.py`). `mine_negatives` (line 536) ranks the 47,963-record train pool
  (`semantic_v3`, max_seq 512) with the base model (stock Qwen3, v8's embedder) against each row's anchor, skips the
  top 3 and picks among the first 10 valid candidates; 42,537 of 51,264 rows got one (`training_info.json`).
  `choose_negative` (line 495): subject-level rows (33,129) reject candidates sharing a non-broad subject id;
  doc-level rows (18,135) exclude only the record and its text, and **prefer** a same-subject negative (5,143 rows);
  content rows also reject an empty label part.
- **In-batch.** `MaskedCachedMNRL` (line 356) masks a candidate only when its ids overlap the anchor's `seg0`.
- **Eval.** Outside the focus subjects (Pesach, Sukkot, …), an unlabelled record naming the holiday counts as
  **non-relevant**, so the eval rewarded burying it.

**Inferred, not verified:**
- The subject-row guard cannot see an unlabelled holiday record. The base model ranks such records high on v8
  (Manchester A 83 is #14 for "Passover Haggadah"), so they are plausible mining candidates; mining used different
  text, pool and anchors, and whether any was drawn is unverified. All rank-regression records are on the train side
  of the md5 split (`build_query_batches.is_test`) except Halper 251, a held-out Sukkot gold record; eval-pool,
  dup-closure and leak-filter exclusions are unverified.
- A possible second contributor: doc-level rows deliberately take a labelled same-subject record as the negative.
- Field bias follows: positives are KTIV records (holiday under `Alternate titles:`, genre list under
  `Description:`), many negatives are FJP/PGP records (holiday under `Description:`), so the label is the cheapest
  separator.

### 3.3 Bare holiday names drift to their scripture or tractate

| Query | Strict v8→v9 (fusion) | v9's top hits (checked in ES) |
|---|---|---|
| Purim | 1.0→**0.0** (1.0) | Bible: Esther 1–4; Talmud Bavli: Megillah |
| פסח | 1.0→**0.2** (0.7) | Hilkhot ha-Rif: Pesahim (6 of 10); Talmud Bavli: Pesahim |
| Passover | 1.0→**0.3** (1.0) | Talmud Bavli / Mishnah: Pesahim |
| Pesah | 0.8→**0.2** (0.9) | Hilkhot ha-Rif / Talmud: Pesahim |
| Feast of Tabernacles | 0.5→**0.0** (0.8) | Bible: Numbers 29, 11, 13, 18; one Mishneh Torah record |
| Pentecost | 0.3→**0.0** (0.4) | Bible: Exodus (incl. 19:3–20:24), Esther; Talmud: Pesahim |

This drift accounts for most of the strict-P@10 loss. The broad rule hides only its tractate part (broad−strict:
v9 0.145, v8 0.069); scripture is never counted, so Purim, Feast of Tabernacles and Pentecost stay low under broad.
"Canon" records (`alt_titles` starting Bible:, Talmud, Mishnah …; seed `aggregates.top20_record_style`) fill 35% of
v9's holiday top 20, against 10% for v8 and an 8% base rate.

**Likely mechanism (code, unverified):** `TOPIC_RULES_V2` gives `purim` to `]: Megillah`, `[Bible]: Esther` and
`מגילת אסתר`, and `pesach` to `]: Pesahim` (likewise Sukkah, Yoma, Rosh ha-Shanah). `build_training_mix_v3.py`
`subject()` (line 964) drops subject names that appear in the carrier record, so "Piyyut (Purim)" cannot pair with
"Purim" but "Bible: Esther 3:1–5" can. The frozen eval counts those carriers as relevant (seen-topic AP 0.087→0.485).
Numbers/Exodus may come from `sefaria_subjects` (line 933) or "tabernacle" as mishkan.

### 3.4 Duplicates and minor issues

- **Identical texts.** Without ID lines, multi-part shelf marks become identical (Oxford heb f 54 parts 56–63, ENA
  2695.1–7); v9's top 10 averages 7.80 distinct scores. Collapsing ties leaves holiday strict P@10 at 0.657 in
  simulation: some tied groups are relevant (Oxford heb f 54), others not (Numbers, Exodus). Training already hashes
  text (`T:<md5>`, line 168); the index does not expose it.
- **No vector.** 55 strict-relevant holiday records are ineligible (seed `ineligible_strict`), e.g. Or 1081.2.33
  "הגדה של פסח (קטע)", v8's #8 for "Pesach". `MIN_CHARS = 40` is a guess.
- **Window.** Training 384 tokens, mining 512, eval and serving 8192. **Scale.** Random doc–doc cosine is 0.071
  (v9) vs 0.431 (v8), so lower on-screen percentages are scale, not quality.

## 4. Ruled out

From the diagnosis' serving audit and embedding probe, not re-run except the 108/108 match.
- **Serving path:** routing, query prompt, `cosineSimilarity + 1` with no boosts, separate caches, 0 vector/eligibility
  mismatches, 108/108 (re-checked). ES `int8_hnsw` knn agrees 10/10 with exact scoring on the 4 screenshot/name queries
  tested (9/10 on the typo).
- **Prompt, tokenizer:** `embed(q, query)` = `embed(PROMPT+q, document)` at 1.000000; tuned tokenizer = stock.
- **Shelf-mark masking:** five holiday records rebuilt at 1.000000, no holiday word masked, no `[shelfmark]` token.
  Separately 5 of 10 random records rebuilt at 1.000000, including T-S NS 292.18, whose masking changes the vector
  (unmasked 0.956).
- **Wrong document text:** each vector is closer (by 0.03–0.53) to tuned-model `semantic_v3` text than to the old
  production text (0.47–0.97); stock model 0.26–0.64. Shortfalls (0.74–0.998) are most likely fields ES does not
  store (unverified).
- **Transliteration, short text:** v9 pulls spellings together (Rosh Hashanah~ראש השנה 0.33→0.80); "Description:
  Pesach" alone scores 0.279.

## 5. Requested changes

**P1. Labels [beta]**
1. Run the holiday rules (which already have Haggadah, Passover, Hoshanot, Hoshana Rabbah) over every record's
   `Description` (FJP, PGP, Bodleian, TEI), case-insensitively ("PRAYERS FOR ROSH HA-SHANAH" is missed today). Add
   the spellings seen in the regressed records (Hagadah, Hoshana raba, Ros ha-sana) and the missing phrases (seder,
   matzah, "Passover night", New Year, Day of Atonement). Use them as strong labels, or at least as mention ids for
   P2.
2. Do not give festival topics to scroll or tractate frames (`[Bible]: Esther`, `]: Megillah`, `]: Pesahim`, Sukkah,
   Yoma, Rosh ha-Shanah); `work:` ids cover them.
3. Revisit the label-copy rule in `subject()`: cap scripture/tractate carriers per festival subject, or require
   liturgy or festival content for festival-name anchors.

**P2. False-negative guard [beta]**
1. In `choose_negative`, reject candidates whose folded Description or Alternate titles contain a surface form of the
   anchor's subject (`subject_vocab.json` names plus `build_eval_v3.CONCEPT_TERMS`-style regexes for every festival).
   Decide whether doc-level rows should keep preferring same-subject festival negatives.
2. In-batch, `candidate_masks` compares `seg0` with `seg1|seg2`, so a mention id must go on both sides: `M:topic:pesach`
   in `seg0` of rows whose anchor stands for `topic:pesach`, and in the identity of every record mentioning Pesach.
   (Or put `topic:pesach` itself on those records' identity side.)
3. Log rejections per subject.

**P3. Text function** (next `semantic_v*`, in `historical-document-analysis` and `semantic_text.py`)
1. **[beta]** Remove the field-label shortcut: one block for Description and Alternate titles, or randomise/drop the
   label in training. Index with the same function (M1 checks it).
2. **[beta]** Store `semantic_text_hash`: keyword, sha256 hex of the embedded text, null when ineligible, algorithm in
   `_meta`. Needed for B8/D8 and the site's dedupe.
3. **[default]** Eligibility: first confirm with `semantic_text()` why Or 1081.2.33 and the other 54 are ineligible,
   then admit short descriptions that match a subject or topic rule.
4. **[default]** Window: train at 8192, or for eligible records over 384 tuned-tokenizer tokens report their count,
   the median and 5th-percentile cosine between 384- and 8192-token vectors, and focus AP / known-item MRR at both.

**P4. Eval additions [beta], before the next training run**
1. A text relevance rule ("the record names the topic", as strict and desc-only here) next to label relevance; gate on
   both.
2. Every name form for every major holiday: English, transliterated, Hebrew, Ashkenazi (Peysakh, Sukkos, Shavuos,
   Rosh Hashono, Simchas Torah), typos.
3. Holiday-versus-text checks (Purim vs Esther/Megillah, Passover vs Pesahim/Exodus 12, Sukkot vs Numbers 29,
   Shavuot vs Exodus 19–20, …) with two shares: tractate/law-code (broad − strict; v8 0.069, v9 0.145) and canon in the
   top 20 (v8 0.105, v9 0.348).
4. The alt_titles-only share of relevant top-10 hits (v8 0.288, v9 0.882); only 19.9% of eligible records have
   `alt_titles` at all.
5. Distinct `semantic_text_hash` values in the top 10 (v9 7.80).

## 6. Acceptance

Proposed thresholds; Isaac confirms them. Stored in seed `acceptance`.

### 6.1 Smoke test (in this folder, no ES)

```bash
python3 -c "import json;d=json.load(open('holiday_eval_seed.json'));json.dump({q['query']:q['top10']['v9'] for q in d['queries']},open('run_v9_seed.json','w'),ensure_ascii=False)"
python3 regenerate_silver.py score --seed holiday_eval_seed.json --run run_v9_seed.json --gate beta
```

Expected: B1 FAIL (0.657), B3 PASS (0.802), B8 NOT MEASURED, `overall: FAIL (B1, B2, B4, B5, B6, B7, B8)`, and a
paired line `diff -0.030  SE 0.053`.

### 6.2 The run

A JSON object keyed by the exact `queries[].query` strings (Hebrew included; "simchat torah" and "Simchat Torah" are
two queries). Each value is ≥ 10 hits, best first, as ids or `{"doc_id", "semantic_text_hash"}` (hashes needed for
B8/D8). Sketch:

```python
import json, regenerate_silver
from sentence_transformers import SentenceTransformer
seed = json.load(open("holiday_eval_seed.json"))
model = SentenceTransformer(MODEL_DIR); model.max_seq_length = 8192
body = seed["protocol"]["script_score_body"]   # eligible filter; cosineSimilarity(..., 'embedding_vector') + 1.0
run = {}
for q in seed["queries"]:
    body["query"]["script_score"]["script"]["params"]["query_vector"] = model.encode(
        q["query"], prompt=seed["protocol"]["query_prompt"], normalize_embeddings=True).tolist()
    hits = regenerate_silver.es_search(ES_URL, NEW_INDEX, body)["hits"]["hits"]
    run[q["query"]] = [{"doc_id": h["_source"]["doc_id"],
                        "semantic_text_hash": h["_source"].get("semantic_text_hash")} for h in hits]
json.dump(run, open("run_new.json", "w"), ensure_ascii=False)
```

### 6.3 Silver lists for the new index

```bash
python3 regenerate_silver.py build --index <new> --out silver_new.json --seed holiday_eval_seed.json
python3 regenerate_silver.py score --seed holiday_eval_seed.json --run run_new.json --silver silver_new.json --gate beta
```

Only the new index needs ES (`ES_URL`, `ES_USER`, `ES_PASSWORD`; `_search` only); v8/v9 lists are in the seed.
`build` pages through ~78k records (≈50 MB), so run it once per index or use `--from-dump`. v8/v9 live in the
production ES on the MBP: ask Isaac before querying it.

### 6.4 Gate

- **Who measures.** The beta gate is measured on the model alone (your run). The default gate is measured by the
  site on whatever would serve as the default (the model, or its fusion without v8); the model must also pass beta.
- **Rules.** NOT MEASURED counts as FAIL. Set drop = v8's mean strict P@10 minus the run's, per holiday, max over the
  9 sets. Bare-name drop = the same per bare-name query. The v8 side is re-scored on the run's relevance union.
- **Point thresholds.** B1 and D4 equal a baseline, so one lost hit fails them. Read B1, D1 and D5 against the paired
  SE (≈ 0.05).

✗B / ✗D = fails the beta / default threshold; n/m = not measured. Row *n* is B*n* and D*n* in `score`.

| Metric | Beta | Default | v8 | v9 today | fusion |
|---|---|---|---|---|---|
| 1 holiday strict P@10 | ≥ 0.686 | ≥ 0.736 | 0.686 ✗D | 0.657 ✗B✗D | 0.736 |
| 2 holiday desc-only P@10 | ≥ 0.40 | ≥ 0.45 | 0.489 | 0.077 ✗B✗D | 0.407 ✗D |
| 3 holiday broad P@10 | ≥ 0.80 | ≥ 0.80 | 0.755 ✗B✗D | 0.802 | 0.814 |
| 4 control strict P@10 | ≥ 0.95 | ≥ 0.97 | 0.97 | 0.93 ✗B✗D | 0.99 |
| 5 worst holiday-set drop | ≤ 0.10 | ≤ 0.05 | 0 | 0.467 Purim ✗B✗D | 0.10 ✗D |
| 6 worst bare-name drop | ≤ 0.30 | ≤ 0.20 | 0 | 1.0 Purim ✗B✗D | 0.5 יום הכפורים ✗B✗D |
| 7 alt_titles-only share | ≤ 0.50 | ≤ 0.40 | 0.288 | 0.882 ✗B✗D | 0.448 ✗D |
| 8 distinct text groups in top 10 | ≥ 9.0 | ≥ 9.5 | n/m (no ties) | 7.80 ✗B✗D | n/m |
| 9 seed queries missing | = 0 | = 0 | 0 | 0 | 0 |

The planned site fusion fails B6, D2, D5, D6 and D7.

### 6.5 Manual probes (these block too)

| Id | Gate | Probe | Pass |
|---|---|---|---|
| M1 | beta | seed `probes.label_swap` (6 pairs) | Description and Alternate titles scores within 0.10 in every pair |
| M2 | beta | seed `probes.rank_regressions` (10 cases) | every case at rank ≤ 50 among eligible records (v8: all ≤ 47) |
| M3 | default | `compare_reports` (`colab/train_embedder_v3.py` line 902) vs `v3-final-run1` | no paired CI entirely below 0 for the primary group (`focus` in the final run; AP 0.198), and no fall in known-item MRR (0.282) |
| M4 | default | site check on `scripts/dev_backend_local.py` | "Pesach", "Rosh Hashnnah", "simchat torah" look right with `alt_titles` shown (the site does this) |

The cutover items in `docs/handoffs/v9_embedder/V9_EMBEDDER_HANDOFF.md` section 4 still apply.

## 7. Site side

Planned, not built. Each change is tested on `scripts/dev_backend_local.py`; backend rebuilds need Isaac's approval.
- Show `alt_titles` on result cards (`_extract_metadata` / `DocumentMetadata`, `src/backend/search_service.py`).
- Collapse identical-text duplicates: tied cosines for v9, `semantic_text_hash` for the next index.
- Rank the beta by RRF of v8 + v9 + BM25, factoring out the inline weighted-RRF block in
  `search_bibliography.search_hybrid` (around line 438). Offline: 0.736 / 0.407 / 0.99; known loss יום הכפורים
  0.9→0.4. With alias expansion Purim falls to 0.2–0.3, so expansion stays off until `genizah_terminology` drops
  esther, אסתר, megillah, megillat esther, weeks, new year and seder.

**What the site expects from the next index:** `alt_titles` unchanged; `semantic_text_hash` on every document;
`semantic_eligible` and the v9 `_meta` contract (model revision, `query_prompt`, bumped
`text_representation_version`, canaries); the deliverables in step 7 of section 1.

## 8. Seed keys

| Key | Contents |
|---|---|
| `systems`, `protocol` | section 2, plus `query_prompt`, `script_score_body`, run-file keys, noise and metric definitions |
| `relevance_rules` | every regex and context rule |
| `aggregates` | P@10 by group, holiday and form; tractate/law-code share; canon share; paired SE |
| `acceptance` | `beta` / `default` criteria, `definitions`, `manual_probes`, `baseline_metrics` |
| `probes` | label swap, rank regressions, query-to-query cosines, anisotropy |
| `queries` | per query: form, P@10 per rule and system, P@20 / capped R@100 / MRR, top-10 lists |
| `supplementary_proxy_set` | the lenient-rule set |
| `silver` | `note`, then `{set: {index: {strict, broad, desc_strict[, ineligible_strict]}}}` |

Scratchpad filenames in the seed (`out06_probe.txt`, `syn_fusion.py`, …) are provenance only; that scratchpad is
deleted.

## 9. Unverified

- Whether any description-only holiday record was a mined or in-batch negative (training files: NAS / private HF).
- The label-copy pairs (Purim→Esther, Passover→Pesahim) and the cause of the Numbers/Exodus drift.
- Why 55 holiday records are ineligible; whole-index vector parity; the 384/8192 effect on long records.
- The silver sets are regexes (passing mentions and a few personal names slip in) and the fusion was not tuned: read
  the numbers as direction.
- T-S AS 111.208 "piyyut for Passover": v8 #9 for "Pesach" (seed), v9 rank > 1000 (saved in the diagnosis); the exact
  #2,075 was console-only. It is on the held-out side of the md5 split, so it was never mined.
