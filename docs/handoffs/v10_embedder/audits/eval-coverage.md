# Pre-index acceptance gate for the next text embedder: inventory, gaps and spec

The gate below can be computed offline in Colab with pure numpy. Seven findings shape it:

- **Field label, source and language move together.** `Alternate titles:` exists only on KTIV records: 74.3% of KTIV-eligible records have it, every other source has 0%. KTIV is 19.9% of the pool (12,018 of 60,493), and 64.8% of KTIV descriptions are Hebrew-only. So any check for the field-label shortcut is also a check by source and by language.
- **Holidays are not the only case of unlabelled records being scored as wrong answers.** For 12 non-festival concepts, 37–92% of the records that name the concept in Description or Alternate titles carry no matching subject label. Today's subject eval counts all of them as irrelevant (table in 2b).
- **FJP-only records are invisible to the subject eval.** They are 31% of the pool (18,762), only 1.7% carry a strong label even with the new labels, and they are 15% of known-item targets (42 of 287). KTIV is 45% of the targets (128 of 287).
- **The frozen eval rewarded the holiday regression.** v9 against base_semantic: seen-topic AP 0.082 → 0.485, Pesach name-query AP 0.074 → 0.582. The reason is that canon records (Esther, Pesahim) counted as relevant.
- **Some holiday-section queries are training anchors.** In `new/mix.jsonl`, exact anchors include Purim (15), פורים (16), Rosh Hashanah (15), Hanukkah (13), the typo "Yom Kipur" (6) and the Ashkenazi "Shavuos" (6).
- **Today's merged file adds 4,957 ids, almost all KTIV.** 4,955 are KTIV, 4,954 have Hebrew-only descriptions, and only 646 have a description of 40 characters or more. The pool shifts toward KTIV and Hebrew.
- **Two of the handoff's thresholds are infeasible as written.** M1 ("within 0.10 in every pair") already fails v8: divorce 0.763/0.572 (gap 0.191), India 0.840/0.714 (gap 0.126). B8 (distinct text groups ≥ 9.0) cannot be met on some sets by any model: Simhat Torah's 209 strong carriers have only 116 distinct texts, and the largest identical-text group has 37 records.

## 1. Inventory

**`eval_v3.py`** (all sections rank the frozen eligible pool; ties are pessimistic):
- **subjects** (`subject_retrieval`, :296). AP, P@10 and R@100 per subject, macro by status and by kind. Weak and unjudged carriers are removed from the ranking (:328-346). `focus` subjects are scored over held-out records only.
- **memorisation** (:400): train-side AP minus held-out AP.
- **known_item** (:462): rank of the target, identical-text twins count as hits; MRR and R@k by type; rank within its subject.
- **collection** (:521): collection-series lift over chance.
- **hubness** (:569): share of duplicate or short records in the top 10.
- **holiday** (:601): new and uncommitted. Per query: P@10 and P@20 against strong carriers (weak carriers count as misses), MRR@100, `weak_share@20` and `alt_only_share`; aggregated by form and festival, plus the worst festival.
- `compare()` (:244) does paired, clustered bootstrap for the subject groups, known-item and memorisation. It does **not** support holiday.

**`build_eval_v3.py`** query sources:
- `probe`: concept_probe_v1 queries for the focus topics.
- `llm_t2`: 36 queries (8 EN + 4 HE) for 3 PGP subjects, at :154.
- `name`: up to 6 EN + 4 HE names per subject. 2,096 seen name queries; final mode has 6 focus subjects and 801 seen.
- known_item: 2,003 synthetic queries, 7 types, 287 targets; `shelfmark_query` = 0.
- `HOLIDAY_NAME_FORMS` (:212): 9 festivals, about 53 queries.
- `holiday_queries.json` (:237) uses label relevance only.

**Handoff 6.4 / 6.5.** It exists only as text pasted into this session. `holiday_eval_seed.json` and `regenerate_silver.py` are only on the MBP (uncommitted, `terraform_gcp_search`, branch `seo-plan-2`).
- Beta gate B1–B9: strict ≥ 0.686, desc-only ≥ 0.40, broad ≥ 0.80, control ≥ 0.95, worst-set drop ≤ 0.10, bare-name drop ≤ 0.30, alt-only share ≤ 0.50, distinct ≥ 9.0, missing = 0.
- Probes: M1 label swap (6 pairs, within 0.10); M2 rank regressions (10 cases at rank ≤ 50); M3 `compare_reports` against v3-final-run1; M4 a site check.

**Other query material:**
- `concept_probe_v1.json`: 15 topics, 130 nuanced queries plus 82 spelling variants. All the queries are in the leak filter (`probe_scope all`, build_training_mix_v3.py:459); the variants are not.
- `v9_ood_queries.json`: 60 unanswerable queries.
- Chat eval sets: `agentic_rag_v1` (11), crosslingual (4), multiturn (5).
- `rag_audit_20261007/chat_sessions.jsonl`: 27 organic sessions and 42 distinct router queries. They include real typos (Channukah, Shavout, Tish B'Av), Hebrew with a prefix letter (הפורים) and literal-quote searches.
- `bib_v9_check.py`: 4,944 pages and 300 hand-written known-item queries.
- `genizah_terminology.CONCEPT_ALIASES`.
- **No search-box query log exists anywhere in the repo.**

**Last run:** `colab_runs/v3-final-run1/genizah_eval/` holds base_original, base_semantic, v3_run1 and final results with per-query lists. Document vectors were not saved.

## 2. Gaps

### a. Query families
1. **Holiday section:**
   - Relevance is labels only; the handoff's P4 text rule was never added.
   - No desc-only view.
   - `weak_share` lumps canon and incidental mentions together.
   - No distinct@10, and no paired comparison.
   - Trained and untrained name forms are mixed together.
   - `alt_only_share` is not normalised. Its base rate among strong carriers runs from 0.04 (fast days) through 0.58 (Sukkot) to 0.76 (Simhat Torah), so a 0.50 cap fails some festivals for any model.
   - If fix 4 merges the two fields, `text_field("Alternate titles")` returns "" and the metric silently disappears.
2. **No canon queries** ("Book of Esther", "Tractate Pesahim") to catch overcorrection. The subject section removes weak (canon) records from the ranking, so it cannot see canon drift at all.
3. **Documentary descriptive queries:** only the 36 llm_t2 queries for 3 subjects. Nothing like "letter from a merchant in India" or "divorce document get".
4. **Missing families:** people, places, scripts and languages; cross-lingual cuts by the language of the record's description; typo coverage (6 hand-written typos, one of them trained); real-user phrasing; the OOD set.
5. Shelf-mark queries: correctly excluded. Keep them report-only.

### b. Record populations (old corpus, eligible records)

| group | n | Alternate titles | <100 chars | strong-labelled (new labels) | known-item targets |
|---|---|---|---|---|---|
| fjp_only | 18,762 | 0% | 55% | 1.7% | 42 |
| fjp_pgp | 16,314 | 0% | 8% | 31% | 69 |
| ktiv_any | 12,018 | 74% | 23% | 84% | 128 |
| pgp_only | 11,982 | 0% | 16% | 24% | 33 |

Concepts named in a record's Description or Alternate titles but not labelled (regex-level counts, not audited):

| concept | records naming it | not labelled (scored irrelevant today) |
|---|---|---|
| divorce | 709 | 464 (0.65) |
| ketubba | 1,358 | 712 (0.52) |
| India | 567 | 209 (0.37) |
| medicine | 1,523 | 901 (0.59) |
| magic | 621 | 382 (0.62) |
| responsa | 1,190 | 807 (0.68) |
| piyyut | 3,235 | 2,215 (0.68), 1,834 of them FJP-only |
| haftarah | 328 | 233 (0.71) |
| Alexandria | 1,331 | 1,219 (0.92) |

Length and duplicates:
- About 6.4k eligible records are 1,000 characters or longer. The 384-token training window against 8,192-token serving is untested.
- 16.7% of eligible records sit in identical-text groups, which inflates P@10.

### c. Failure modes not covered
- Field-label bias: no automated probe. M1 has 6 pairs and lives on the MBP.
- M2's records are mostly on the train side. Of the 9 I found, 7 are train side and 2 held out; Or 1081.2.33 is ineligible. With the new labels they are now strong carriers, so after retraining M2 measures fitting, not generalisation.
- Anisotropy and score scale are not measured (random doc–doc cosine: v9 0.071, v8 0.431).
- The bibliography check runs separately (`bib_v9_check.py`).

## 3. Gate spec ("broad regression gate", gate_v1)

### Setup
- **Models:** C = candidate; B = base_semantic (stock Qwen3-0.6B@97b0c61 on C's text); P = base_original (stock on production text with ID lines; the offline v8 analogue); V = v9 (dbfab73 on semantic_v3).
- **Pool:** all four models are scored on one frozen pool: the eligible ids of C's text function, built from today's merged file (inferred: about 63.6k).
- **Text hashes:** check each model against its own text version. The pool must carry a hash per text version, otherwise V and P will raise a mismatch.
- **Dedupe:** rank, then collapse identical `text_hash` into one slot. The primary metric is P@10_dd; raw values are kept.
- **Relevance tiers**, using the record's source Description and Alternate titles fields as stored before any field merge:
  - `relevant` = text_strict ∪ label_strong.
  - `canon` = festival named only in a canon frame or context (FESTIVAL_CANON_RE build_subjects.py:149, CANON_WORKS :210). Counts as a **miss** for festival queries and as **relevant** for canon queries.
  - `unjudged` = (label_weak ∪ transcription-only mentions ∪ incidental-type mentions, :154) minus relevant. Removed from the ranking.
- **Views:** union (primary), text-only, label-only, desc-only (alt-only relevant records become unjudged).
- **Statistics:** clustered paired bootstrap (`eval_v3.paired_bootstrap`, 2,000 resamples). Clusters are concepts; for known-item, target records. SE = (ci_high − ci_low) / 3.92.
- **Integrity:** NOT MEASURED = FAIL.
- **Leak control:** every gate query is tagged `trained: blocked|label_ok`. Only bare subject names are `label_ok`. Blocked queries go into the leak filter **before** the mix is built.

### Families

| id | queries (source, est. n) | relevance | metric |
|---|---|---|---|
| G1 holiday names | HOLIDAY_NAME_FORMS, concept_probe festival variants, seed bare names, Hebrew-prefixed forms (בפסח, הפורים); ~100 | union, canon = miss | P@10_dd; cuts by form, festival, trained/untrained |
| G2 holiday descriptive / nuanced | seed descriptive, concept_probe nuanced for 9 festivals + Shabbat; ~100, blocked | union | AP, P@10_dd |
| G3 canon (overcorrection guard) | ~40 written ("Book of Esther", "Tractate Pesahim", "Exodus 12", "Numbers 29", "Mishnah Yoma") | `work:` labels | AP, P@10 |
| G4 documentary | 115 PGP name subjects + ~120 new descriptive queries (30 concepts, 3 EN + 1 HE, `query_writing_spec`) + concept_probe marriage/divorce/kashrut/mourning | PGP label ∪ text | AP, P@10_dd |
| G5 genres / works | existing domain/work/sefaria names + ~60 text-rule queries for FJP-heavy concepts (piyyut, haftarah, responsa, calendar, magic, medicine, Karaite, Saadya) | label ∪ text | AP, P@10_dd |
| G6 people / places | ~70 EN + HE names with ≥ 10 records in Related People / Places | field ∪ description text | P@10_dd, R@100 |
| G7 scripts / languages | ~20 | `Language:` field | P@10 |
| G10 known-item | existing 2,003 | target + twins | MRR, R@10 |
| G11 real users | ~45 (chat router queries, agentic variants, "Pesach", "Rosh Hashnnah", "simchat torah") | mapped to concept rules | per query |
| G12 OOD | 60 | none | AUC of top-1 score, in-domain vs OOD |
| G13 shelf-mark / collection | ~20 | none | report only |

Cuts applied across G1–G7 and G10:
- query language × description language (Hebrew-only vs Latin);
- typo/clean pairs (deterministic generator plus the real typos from the chat logs);
- source group;
- length bucket (<100, 100–300, 300–1k, ≥1k characters);
- held-out vs train side;
- new-since-v9 records.

Rule audit before freezing: per concept, 20 matched records plus 20 unmatched records from B's top 20 go to `review_sample.csv`. A rule is frozen only at precision ≥ 0.9.

### Hard criteria (all must pass)

**H1 – Family non-inferiority.** For each of G1–G7 and G10, against B and against P: Δ ≥ −δ and Δ ≥ −2·SE. δ is 0.03 for P@10_dd and 0.02 for AP or MRR. Expected paired SE at about 100 queries is about 0.035, extrapolated from the handoff's SE of 0.053 at n = 44.

**H2 – Cells.** Applies to every cut with ≥ 10 queries or ≥ 50 query–document pairs: Δ ≥ −max(0.10, 2·SE). At most one cell per cut may fall below −2·SE, and none below −3·SE or −0.20.

**H3 – Holiday, offline analogues of B1–B7:**
- strict P@10_dd ≥ P − 1·SE;
- desc-only ≥ 0.82 × P (the handoff's 0.40 / 0.489 ratio);
- broad ≥ P;
- worst festival-set drop against P ≤ 0.10;
- worst bare-name drop ≤ 0.30, among queries where P ≥ 0.3;
- alt-only share ≤ 0.50 (macro);
- alt-only lift (share among relevant top-10 hits ÷ share among relevant records) ≤ 1.25 macro, and ≤ 1.5 per festival with ≥ 30 relevant records.

**H4 – Canon.** canon_share@20 ≤ P + 0.05 (v8 0.105, v9 0.348). Tractate / law-code share ≤ P + 0.03 (v8 0.069, v9 0.145). G3 must pass H1.

**H5 – M1, automated.**
- Seed pairs: |cos_Desc − cos_Alt| ≤ max(0.10, P's gap + 0.05).
- Record swap: take 300 real relevant records (150 KTIV with the concept under Alternate titles, 150 FJP/PGP with it under Description). Swap only the label, re-embed with a document encoder, and compute Δz = Δscore / σ_q, where σ_q is the spread of the query's pool scores. Pass when |mean Δz| ≤ 0.25 and ≤ |B| + 0.10, in each direction.

**H6 – M2, automated.**
- The seed's 10 cases stay at rank ≤ 50.
- Held-out sweep over (query, relevant record) pairs where P's rank ≤ 20: ≥ 95% of them have C's rank ≤ max(50, 3 × P's rank); ≤ 2% fall below rank 500; in each family ≤ 10% fall below rank 50.

**H7 – Population.** For each source group, R@100 ratio C/P and C/B ≥ 0.90. KTIV lift@10 ≤ P's lift + 0.25.

**H8 – M3, keep v9's gains (V re-scored on the new eval dir).**
- focus AP: paired CI against V not entirely below 0;
- known-item MRR: Δ ≥ −0.01 and CI upper bound ≥ 0;
- seen-subject AP: Δ ≥ −0.02;
- G2: Δ ≥ −0.05;
- Hebrew and transliterated known-item types: MRR Δ ≥ −0.02 each.

**H9 – Structure.**
- collection lift ≤ B + 0.10;
- hubness share ≤ B;
- no single record in the top 10 of more than 2% of gate queries;
- raw distinct@10 ≥ B − 0.3 (replaces B8's absolute 9.0);
- memorisation_trained net gap ≤ 0.06.

**H10 – Bibliography.** `bib_v9_check score` in Colab: no metric's paired CI against base entirely below 0, for dense, 60/40, 30/70 and chat 50/50.

**H11 – Integrity.**
- text-hash mismatch = 0;
- every query scored;
- minimum sizes met;
- no blocked gate query appears as an exact or contained training anchor (recomputed from the mix).

**Soft (reported):** typo robustness (mean Δ typo − clean ≥ B's − 0.05); OOD AUC ≥ B − 0.05; score-spread ratio against B between 0.5× and 2× (affects hybrid weights); mean random cosine and PC1 share; G11 per query; G13.

**After the pre-index gate passes:** build the index, then run the handoff's official 6.2/6.3 B1–B9 with the absolute thresholds, plus M4 and the chat eval sets on `dev_backend_local.py`.

### Files (in the new eval dir, `gate_v1/`)
- `queries.jsonl`: `{qid, text, family, concept, form, lang, source, pair_of, trained, control}`
- `concepts.json`: `{cid: {family, label_ids, text_rule{en, he, fields}, canon_rule, incidental_types, relevant, relevant_text, relevant_label, desc_mention, alt_only, canon, unjudged, n_by_group}}`
- `records.jsonl`, one row per pool record: `{doc_id, group, has_alt, desc_lang, len_bucket, new_since_v9, held_out, text_hash}`
- `probes.json`: `{label_swap_pairs, label_swap_records[{doc_id, qids, orig_text, swapped_text, direction}], rank_regressions_seed, rank_sweep{ref: "base_original", max_base_rank: 20}}`
- `thresholds.json`: the frozen criteria; Isaac confirms them.
- Outputs per model: `gate_<label>.json`, `vectors_<label>.f16.npy` with `ids.json` (about 130 MB each), and `gate_verdict.{json,txt}`.

### Where to implement (order matters)
1. Rebuild corpus, text, subjects and `build_eval_v3 --final` from today's merged file.
2. New `build_gate_v1.py` (json only). Reuses FESTIVAL_TERMS (build_subjects.py:129), HB/HE (:61-62), CONCEPT_TERMS (build_eval_v3.py:104), HOLIDAY_NAME_FORMS (:212) and CONCEPT_ALIASES. Imports `holiday_eval_seed.json`, which first has to be copied from the MBP.
3. Extend `build_training_mix_v3.build_leak_filter` (:428) with the gate queries tagged `blocked`. Then build the mix.
4. New `eval_gate.py`, numpy only: `evaluate_gate(doc_ids, M, qfn, eval_dir, doc_fn)` and `verdict(reports, cand, refs)`.
5. `colab/train_embedder_v3.py`:
   - `evaluate()` (:725): also call eval_gate, save the vectors, and score V and P;
   - `compare_reports()` (:904): print the verdict;
   - `doc_fn` reuses `encode(is_query=False)` (:477).
6. `package_dataset_v3.py`: ship `gate_v1/`.
7. Optionally keep `eval_v3.holiday` for continuity. `compare()` has no holiday group.

### Decisions for Isaac
- Confirm the thresholds.
- Copy the seed from the MBP.
- Is a search-box query log available? None is in the repo.
- Whether to also re-run the changed recipe on the v3 held-out-subject split as a leave-concept-out check: gate AP ≥ 0.182 − 1 SE, at the cost of one more A100 run.

Scratch scripts and outputs (read-only inputs) are in `/private/tmp/claude-501/-Users-isaac-Documents-GitHub-genizah-search/6e892054-8bdc-4cf3-9c1c-de16a1b5474a/scratchpad/audit_gate/`: `merged_pop.py`, `merged_pop.json`, `pop.py`, `cover.py`, `hidden.py`.