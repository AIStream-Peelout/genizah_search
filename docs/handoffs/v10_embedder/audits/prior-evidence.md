# v9 (v3-final-run1) evidence, open risks and cutover checklist

**Bottom line:**
- v9 beats the stock model on almost every aggregate number.
- The frozen eval is biased in exactly the way that caused the holiday regression. It scored "Purim" as a 10x gain, so it cannot certify the next model on holidays.
- Retraining the embedder does force re-embedding the bibliography. But per the 2026-10-08 handoff, `bibliography_text_only_0.8` had not been built yet, so a v10 costs one bibliography embed (4,944 pages), not two.

Sources: `evals/embedding_audit/results/colab_v3-final-run1/{comparison.txt,summary.json,training_info.json}`, the per-model JSONs in `AUDIT_ROOT/colab_runs/v3-final-run1/genizah_eval/`, `results/v9_bibliography_check.json` and `results/v9_cosine_calibration.json`. My recomputations are in `SCR/audit_evidence/{losses.py,ki_losses.py,festival_carriers.py}`.

Column meanings below:
- **orig** = stock model on today's production text (v8).
- **sem** = stock model on semantic text.
- **v9** = the tuned model.

Confidence intervals are the authors' paired bootstrap 95% intervals.

## 1. Evidence table

**Frozen eval v3-final.** Pool of 60,493 eligible records; scored on 12,337 held-out eligible records.

| Metric | orig | sem | v9 | v9 minus sem [CI] (v9 minus orig) |
|---|---|---|---|---|
| Focus AP (6 subjects) | 0.064 | 0.051 | 0.198 | +0.147 [+0.059, +0.238] (+0.134 [+0.046, +0.222]) |
| Focus P@10 / R@100 | 0.178 / 0.124 | 0.146 / 0.107 | 0.378 / 0.275 | +0.232 [+0.096, +0.374] / +0.168 [+0.094, +0.255] |
| Focus probe-query AP | 0.019 | 0.016 | 0.161 | Optimistic: 17 leaked rows and 27 near-duplicates; `v3-final2` was never built (`hf_dataset/` has no such tag) |
| Sukkot / Pesach / kashrut AP | 0.051 / 0.058 / 0.022 | | 0.256 / 0.346 / 0.178 | +0.216 [+0.153, +0.277] / +0.307 [+0.201, +0.400] / +0.155 [+0.052, +0.280] |
| Slavery / geonic academies AP | 0.074 / 0.008 | | 0.195 / 0.034 | +0.139 [+0.080, +0.218] / +0.027 [+0.011, +0.047] |
| Partnership AP | 0.170 | 0.143 | 0.179 | +0.036 [−0.015, +0.101], not significant; −0.035 vs v3-run1 |
| Seen AP (740 subjects, name queries, includes training-side records) | 0.132 | 0.131 | 0.377 | +0.246 [+0.226, +0.264] |
| Seen AP by kind | domain 0.092, pgp 0.077, sefaria 0.020, topic 0.087, work 0.281 | | 0.484, 0.319, 0.107, 0.485, 0.610 | Sefaria stays weak |
| Seen subjects one by one (my count vs sem) | | | | 655 better, 59 tie, **26 worse** |
| Known-item MRR / R@10 / R@100 (1,961 queries) | 0.193 / 0.295 / 0.494 | 0.196 / 0.297 / 0.498 | 0.282 / 0.426 / 0.711 | RR +0.086 [+0.058, +0.114]; R@10 +0.129 [+0.090, +0.167] |
| Known-item by type (MRR) | Hebrew-implicit 0.085, Hebrew-topical 0.133, Hebrew 0.148, implicit 0.105, specific 0.485, topical 0.247, transliterated 0.145 | | 0.157, 0.255, 0.251, 0.146, 0.587, 0.330, 0.250 | Every type improves |
| Known-item one by one (my count vs sem) | | | | 1,294 better, **453 worse**, 214 same; 384 entered the top 10, **131 dropped out** |
| Duplicate/short records in top 10 | 22.5% | 24.3% | 14.6% | v3-run1 had 12.5%, so v3-final is worse than run1 |
| Memorisation gap (training-side minus held-out AP) | −0.003 | −0.006 | +0.032 | Net **+0.038 [+0.024, +0.054]**; trained positives +0.058 [+0.039, +0.077] |
| Step 0 alone (removing ID lines, stock model) | | | | Focus AP **−0.013 [−0.025, −0.004]** for sem vs orig |

**Losses to the stock model hidden inside the aggregates.** Seen-subject AP, sem → v9:
- **Festival liturgy:**
  - `common-prayers-sukkot` 0.614 → 0.298
  - `common-prayers-yom-ha-kippurim` 0.496 → 0.337
  - `musaf-sukkot-amidah` 0.684 → 0.604
  - `shemini-atzeret` 0.195 → 0.122
  - `piyyut-maariv-for-sukkot` 0.107 → 0.043
  - `talmud-bavli-sukkah` 0.900 → 0.866
- **Rare works looked up by name:**
  - Avraham min ha-Har commentary 1.000 → 0.280
  - Yehonatan of Lunel on the Rif 0.860 → 0.211
  - Tanhuma printed version 0.875 → 0.282
  - Pesiqta de-Rav Kahana 0.904 → 0.744
- **Shabbat-labour Sefaria topics:** saving lives 0.159 → 0.057; work done on Shabbat 0.116 → 0.067.
- **Against orig as well:** `biblical-exegesis-rabbanite-psalms` 1.000 → 0.638; `sefaria:laws-of-the-sukkah` 0.670 → 0.591; `domain:astrology` 0.347 → 0.273.

**Single-record hub.** v9's top hub is `Cambridge_CUL_T_S_NS_329_413`, a Bavli Shabbat record. It fills 81 top-10 slots, 3.8% of 2,129 queries, which is larger than any hub under the stock model (largest 2.8%). That fits the "canon-frame pull" root cause (inferred).

**v3-final lost ground against v3-run1 while still beating the stock model:**
- implicit MRR 0.155 → 0.146
- Hebrew-implicit MRR 0.167 → 0.157
- seen work AP 0.620 → 0.610
- duplicate/short share in the top 10: 12.5% → 14.6%

**Bibliography check** (4,944 pages, 300 hand-written known-item queries; `v9_bibliography_check.json`; verdict `safe_to_reembed_bibliography_with_tuned`):
- **Gains:**
  - Dense R@10 0.573 → 0.703 (+0.13 [+0.053, +0.203]).
  - Hebrew R@10 0.34 → 0.58 (+0.24 [+0.153, +0.320]).
  - Cross-lingual R@10 0.281 → 0.544 (+0.263).
  - 60/40 hybrid, Hebrew RR +0.053 [+0.005, +0.100].
  - Chat 50/50 setting, Hebrew R@10 +0.093 [+0.020, +0.167].
  - Subject-name AP 0.111 → 0.223 (+0.111 [+0.094, +0.128]).
- **"Watch" items** (negative point estimates, intervals include 0):
  - English dense RR −0.018 [−0.078, +0.041]; English R@1 0.507 → 0.473.
  - 60/40, English RR −0.012.
  - 30/70, all RR −0.010; Hebrew R@1 0.10 → 0.06.
- **Cosine scale:** v9 lowers cosines by about 0.1 (known-item top-1 median 0.601 → 0.486). The canary cosine between the two models is 0.41.

## 2. Collection clustering: what v9 changed

| Same-collection share of top-10 neighbours (lift over chance) | orig | sem | v9 |
|---|---|---|---|
| All records (n=3,000; chance 11.4%) | 43.8% (3.85) | 31.9% (2.80) | 31.5% (2.76) |
| Long records (n=1,444) | 32.7% (3.14) | 25.1% (2.42) | 25.7% (2.47) |

- The "55% vs 9%" figure (EMBEDDER_PRODUCTION_HANDOFF:22-24) was measured on the 73.5k production vectors with collection series as the unit. The eval above uses a different sample and chance rate, so the two are not directly comparable.
- v9 cuts clustering by about 28% against today. **All of that comes from removing the ID lines.** Training itself was neutral overall (−0.04 lift) and slightly worse on long records (+0.055). The audit says the same (EMBEDDING_FINETUNE_AUDIT:338-339, 368).
- Collection works as a proxy for subject in parts of the corpus: 47% of the Bodleian framed records are Sukkot (audit:260-261).
- The audit proposed two fixes (audit:343-347): same-collection, different-subject hard negatives, and early stopping. **Neither is implemented**; grep finds no such code in `build_training_mix_v3.py` or `colab/train_embedder_v3.py`.

## 3. Why the frozen eval missed the holiday regression

This matters for the gate on the next model.

- **Purim scored as a 10x gain.** Seen `topic:purim` AP went 0.058 → 0.584, while live search sends Purim to Esther/Megillah.
  - In `v3/subjects_v1.jsonl`, 95 of 138 strong Purim carriers (69%) also carry a Bavli/Bible/Mishnah/Mishneh Torah work subject.
  - The same holds for 91 of 224 Pesach carriers (41%; 22 of the 43 held-out ones) and 299 of 523 Shabbat carriers (57%).
  - The labels come from the committed `build_subjects.py:61,70` (`\]:\s*Pesahim\b`, `\]:\s*Megillah\b|\[Bible\]:\s*Esther`).
  - The older T1/T2 probe labels use the same mapping (`build_corpus.py:43,47`).
- **Records that name the holiday were invisible by design.** The focus eval leaves "unjudged concept mentions" out of the ranking (`eval_final/focus_subjects.json` rule): 391 Pesach and 343 Sukkot records. The "holiday named in the Description gets buried" failure could not show up.
- **The broader probes were never re-run on v9.** `results/` has T1/T2 files only for the stock model, the pilot, bge and e5. That covers the 15-topic concept probe (Purim, Yom Kippur, "Kol Nidre → niddah") and the T2 framed pool.

## 4. Known risks raised earlier and still open

1. **Fragment-pool cosine calibration was never run on v9.** The status line reads "fragment pool: base_only". The 80/20 and 90/10 hybrid mapping and the out-of-domain queries are listed as pending (`v9_cosine_calibration.json` `pending`); the run was blocked by guard.py's disk gate.
2. **Probe-query leak.** The leak-fixed `v3-final2` dataset was never built (audit:375-377).
3. **No scholar-validated query set.** About 300 queries were planned (audit:590, 748-749).
4. **No document-side enrichment.** Frame-to-topic words and Sefaria text were planned (audit:630-632). Even the oracle reaches only 22% of the Sukkot gold items in the top 1,000.
5. **Training/inference length mismatch.** Training used 384-token windows (`training_info.json` `args.max_seq`) while serving uses 8192. The eval ran at 8192, so this was measured for fragments, but bibliography pages of up to 7,655 tokens were never trained on.
6. **No scholarship pairs in the v3-final mix.** It contains only content, subject, synthetic_*, t2t_* and d2d families, so the bibliography is covered only by the out-of-domain check.
7. **Prompt trap.** The saved model's default query prompt is the stock "web search" one (`colab_runs/v3-final-run1/model_cfg/config_sentence_transformers.json`). It must never be used.
8. **Field-label shortcut (fix 4) is not done.**
9. **Corpus drift.** The eval corpus is the 73,543-record file of 2026-10-05; the v9 index holds 78,113 documents. About 4.6k new records and about 1,550 changed ones were never evaluated (inferred).
10. **Machine-generated descriptions are planned into the v9 index** (HDA `docs/handoffs/v9_index_switch_mbp.md:74-81`). That text type is untested, and the writer would need retargeting to a v10 index.
11. **The memorisation gap widened** (+0.038). Isaac's concept-first rule says it matters as the index grows.
12. **Bibliography vectors were made with lowered disk gates.** `run_bib_v9.sh` produced the tuned vectors after lowering guard.py's disk gates (`v9_cosine_calibration.json` `provenance_warning`).
13. **Other open items:**
    - Qwen3-Embedding-4B was never benchmarked.
    - The FJP join-list prefix bug is in the data.
    - Three production bugs: `primary_hybrid` uses the wrong field name (`semanticWeight`); filtered hybrid search zeroes all scores; `search_by_author` sorts (cos+1)/2 scores together with RRF scores.

## 5. Does retraining force re-embedding the bibliography?

**Yes.**
- **One query embedder, one gate.**
  - At startup, `app.py:194-231` re-embeds both indices' canaries through the single `embedding_client`.
  - A cosine at or below 0.99 sets a global `drift_reason` (`embedding_client.py:92-97`), after which `get_embedding` raises for every caller (`:61-62`). Semantic search then goes off for both indices.
  - I read this on the local `origin/prod-mbp` ref (e765768, 2026-09-22), which may lag the live branch.
- **Any new weights will fail the canary.** v9 against the stock model is 0.41; a v10 will also land far below 0.99 (inferred).
- **Keeping the stock model for the bibliography** would need a second embedding container and client, separate drift state per client, about 2.4 GB more RAM, and two contracts to maintain (`v9_bibliography_check.json` `keep_base_would_require`).
- **The cost is small.** As of 2026-10-08 01:06, `genizah_merged_v9` was built (78,113 docs, 63,652 with vectors, 14,461 marked `semantic_eligible:false`) and `bibliography_text_only_0.8` was not (HDA handoff:13-19). I could not check whether it has been built since, because ES queries are off-limits. Superseding v9 before the switch means one bibliography embed of about 4,944 pages.
- **The "safe" verdict does not carry over.** It was specific to the v9 weights, so `bib_v9_check.py` has to be re-run on v10.

## 6. Cutover checklist for the next model

Type key: **A** = automated check exists; **A\*** = script exists but has to be run by hand or extended; **M** = manual.

**Model and contract**
1. **M, then A.** Pin the repo and commit sha, never the tag, in the `embedding_models.py` defaults (currently `:28-29`), the compose `EMBEDDING_MODEL_NAME`/`_REVISION`, and the index `_meta`. The canary verifies it.
2. **A in the verifier, M at startup.** Pass the query prefix explicitly and never use `prompt_name="query"`.
   - `verify_reembedded_indexes.py:174-190` checks the query canary.
   - The startup gate does not yet check the query canary or `text_representation_version`; that has to be added (HDA handoff:52-53).
3. **M.** Set `max_seq_length=8192` and fp32 explicitly. Supply the HF token or a pre-filled `hf_home`, and export `HF_TOKEN` (the 401 trap on `processor_config.json`). Do not set `fix_mistral_regex`.

**Document text**
4. **A.** Bump the text-function version (e.g. `semantic_v4` for the field-label fix). Keep it byte-identical across `genizah_search/evals/embedding_audit/semantic_text.py` and `HDA/src/embeddings/semantic_text.py`. The regex constants are identical today (my diff), and HDA's `tests/test_semantic_text.py` pins them.
5. **A.** Eligibility uses the frozen 64-label boilerplate list, and an empty string is never embedded. `check_eligibility_vectors` (verifier `:223+`) checks it.
6. **M.** The bibliography keeps `Shelf marks mentioned:`; drop only `Source PDF:`.

**Index build**
7. **M, then A.** Create new names for both indices; the HDA constants (`index_merged_genizah.py:67`, `index_all_bibliography.py:35`, `qwen_text_embedding.py:44-45`) need bumping. Those HDA files hold someone else's uncommitted work, so this needs Isaac's OK. The writer guard refuses a mismatched `_meta`.
8. **A\*.** `_meta` must hold model, revision, query prompt, text version, max_seq and three canaries. Run `verify_reembedded_indexes` on **both** indices before any swap.

**Query-side code** (genizah_search, prod-mbp)
9. **M.** Every vector leg filters `semantic_eligible:true`. Without it, `script_score` over `match_all` errors on documents with no vector (`search_service.py:858-869, 1724-1731, 1772-1776`; `search_bibliography.py:232-239, 417-420, 557-560`).
10. **M.** Shelf-mark and collection routing in `/search` and `/search-hybrid`, plus shelf-mark injection in chat. The default search mode is pure vector (`react_app.jsx:51`). Acceptance is about 30 queries via the dev backend on :8010 against the local ES mirror; no script exists for this.
11. **No change.** Keep `SIMILARITY_THRESHOLD=0.4` (`lms_agentic_search.py:231`, rank-only RRF) and add no raw-cosine gate. Hybrid weights stay as they are from 50/50 to 70/30.
12. **A\*, pending.** Run `v9_cosine_calibration.py` on the new model's fragment vectors to settle 80/20, 90/10 and the out-of-domain queries.

**Evaluation before the swap**
13. **A.** `eval_v3.py`, every section, against both orig and sem. Add the uncommitted holiday section, with labels that do not map tractates or Bible books to festivals. Report per-subject losses as well as the aggregates (26 seen subjects lost under v9).
14. **A\*.** Re-run `bib_v9_check.py` for the new weights.
15. **A\*, never run on a tuned model.** T1 concept probe (`probe_text_concepts.py`) and the T2 framed pool (`eval_corpus.py`); the T2 labels need de-biasing first.
16. **A\*, Opus judge.** Chat eval sets agentic_rag v1, crosslingual and multiturn on the dev backend against the new indices.
17. **M.** Four smoke queries (Hebrew query, English paraphrase, shelf mark, bibliography) on dev, then on prod after the switch. A scholar-validated set is still missing.

**Deployment**
18. **M.**
    - Isaac's approval for each production step (CLAUDE.md).
    - Re-run the viz precompute (audit:673).
    - Keep v8 and the old embedder for rollback (audit:674).
    - Retarget the generated-description writer to the new index.
    - Fix the three production bugs separately; they do not block the switch.