# Training recipe audit (v9 = v3-final-run1) and a conservative config for the next run

**Bottom line:** v9 did not collapse. It over-specialised. It improved on in-distribution metrics (seen subjects AP +0.25, known-item RR +0.086) while:
- memorisation grew (net gap +0.0575 [+0.039, +0.077]);
- geometry drifted a lot;
- gains concentrated on the name forms it trained on.

The recipe gives four levers for a safer run: limit drift (LoRA, or a lower learning rate plus a pull back toward the base model), save checkpoints and pick one on a broad gate, close the 384-token training window gap, and fix how negatives are chosen for record-level rows.

## 1. The recipe as run (code plus v3-final-run1/training_info.json)
- **Full fine-tune.** fp32 master weights, bf16 autocast, gradient checkpointing, GradCache mini-batch 8 (`train_embedder_v3.py:644-693`).
- **Optimiser settings.** lr 2e-5, warmup_ratio 0.05 (≈10 steps, :686), cosine schedule, 1 epoch = 203 steps, batch 256, scale 20 (:646). Proportional sampler, one dataset per batch: q2d_hn 155 steps, t2t 34, d2d_hn 12, q2d 1, d2d 1. NO_DUPLICATES batches, seed 13. Run time 6,196 s.
- **Weight decay and clipping.** Weight decay and max_grad_norm are not set, so the HF defaults apply (0 and 1.0). Logged grad norms were 13–30, so inferred: every step was clipped.
- **`save_strategy="no"` (:690).** Only the final weights exist, so no early stopping or checkpoint choice is possible.
- **Loss.** Fell from 4.48 (step 10) to 2.44 (step 200), with spikes at step 110 (3.46, grad norm 24.6) and step 160 (3.00, grad norm 27.0). Each logged value averages mixed datasets, so it cannot show per-family overfitting.
- **Token windows.** Training 384 (:81), mining 512 (:538), eval and serving 8192 (:80).
- **Mining.** 42,537 rows got a negative; 5,143 of them `same_subject` (content 2,763, specific 989, translit 721, hebrew 670).
- **Notebook.** Uses the defaults: `te.mine_negatives(base, data)` and `te.train(..., epochs=1.0, lr=2e-5, batch=256, mini_batch=8)` (`make_notebook_v3.py:116,124`).
- **Model selection.** The decision metric is focus AP alone: 6 subjects, 6 clusters (:818-825, :978).

## 2. What got worse, with numbers (`colab_runs/v3-final-run1/genizah_eval`)

**Tuned vs base_semantic:**
- **Seen subjects:** 33 of 740 got worse, 15 by more than 0.05.
  - 6 of those 15 are festival liturgy works: common-prayers-sukkot −0.315 (0.614→0.298), yom-ha-kippurim −0.159, musaf-sukkot-amidah −0.080, shemini-atzeret −0.073, piyyut-maariv-sukkot −0.064, yom-ha-kippurim-amidah −0.051. This matches root cause 1.
  - 3 small work subjects (5–8 relevant records each) dropped 0.59–0.72.
- **Known-item:** net better, but 453 of 1,961 queries got a worse rank (23%). 131 fell out of the top 10 (384 entered it), and 113 got more than 10× worse. Every type has 53–76 worse queries.
- **Memorisation (trained positives), base gap → tuned gap by kind:**

| Kind | Base gap | Tuned gap |
|---|---|---|
| topic | 0.0068 | 0.1026 |
| domain | −0.0099 | 0.0692 |
| sefaria | 0.0114 | 0.0697 |

  The local LoRA trial had −0.012 [−0.030, +0.006].
- **Focus subjects:** partnership got worse on 6 of 15 queries (5 of them llm_t2 paraphrases); kashrut got worse on 4 of 17.
- **Hubness:** the overall share improved (0.243→0.146). But the single top hub went from 59 to 81 slots (3.8% of the 2,129 queries). The top 4 hubs are all T-S NS 329 records: KTIV Talmud records with an "Alternate titles" line and a generic Description.
- **Collection lift:** flat (2.80→2.76). Not a problem.

**v3-final vs v3-run1 (same recipe, 25% more data):**
- 283 of 740 seen subjects got worse, 107 by more than 0.05, 72 of those work subjects. Examples: mishneh-torah-shehitah −0.444 (103 relevant records), bible-genesis −0.283 (193 relevant).
- Focus AP split by query source: llm_t2 paraphrases were flat (0.0952→0.0962) while name queries rose 0.172→0.311. Partnership got worse on 12 of 15 queries and geonic-academies on 13 of 20 (10 llm_t2 each).
- Reading: gains landed on the name forms trained on, and run-to-run interference is large. This argues for choosing among checkpoints.

**Out of domain (bibliography; I measured this, `audit_recipe/aniso.py`):**

| Measure | Base | Tuned |
|---|---|---|
| random page-page cosine, mean | 0.473 | 0.347 |
| query-page cosine, mean | 0.235 | 0.109 |
| participation ratio (effective dimensions) | 36.9 | 29.5 |
| top-10 eigenvalue share | 0.426 | 0.466 |

- Same-text cosine, base vs tuned: canary 0.41, median 0.609.
- English known-item R@1 0.507→0.473 (RR −0.018 [−0.078, +0.041], not significant).
- I did not re-measure the fragment numbers you were given (doc-doc 0.431→0.071): no tuned fragment vectors exist locally.

**Blind spot in the eval:** the `seen` group is scored over the whole pool, including the train-side records, with the same name vocabulary used as training anchors (`eval_v3.py:318-329`; only focus subjects exclude train-side records). So the seen AP gain of +0.246 is mostly in-distribution and should not drive model selection. The new `holiday` section (`eval_v3.py:601`) is also whole-pool, and its relevance labels come from the same regex as the training labels.

## 3. Token windows (exact Qwen counts; my pure-Python BPE matched the bibliography total of 5,258,578 tokens exactly)
- **Eligible corpus:** median 55 tokens, p90 389, p99 858, max 1,938. 10.1% exceed 384 tokens.
- **q2d positives:** median 109, p90 473, p99 902.

| Window | Positives truncated | Tokens lost |
|---|---|---|
| 384 | 19.0% | 16.3% |
| 512 | 7.9% | 8.1% |
| 1024 | 0.44% | 0.5% |

- **What 384 cuts:** Transcription in 5,268 of 7,777 records, Translation in 1,146 of 1,257, Related People and Places about half. Description is always the first line and is never cut. So the window does not cause the field-label shortcut; it is a train/serve mismatch for 10–19% of texts.
- **Cost of 1024 vs 384:** 2.59× document tokens with today's padding. Inferred: the ST collator pads each column to the longest text in the 256-row batch. With padding trimmed per mini-batch of 8 it is about 1.43×.
- **Mining at 512** matches neither training nor serving. Every record is under 2,048 tokens, so mining at 8192 is the full text.

## 4. Field-label shortcut: a contributing cause found
- 38.2% of v9's q2d positives contain "Alternate titles:", against 14.9% of the train-side pool (2.6×). In the subject family it is 54.5% (3.7×).
- After fixes 1–3 it is still 36.6% overall and 48.4% in the subject family.
- So the training data over-represents records with that line. Fix 4 is not done, and a training-side counter is in the config table.

## 5. Q4: is the `same_subject` negative preference still sound? Keep it only for `synthetic_specific`; drop it elsewhere
How it works now: `choose_negative` (:529-532) prefers any candidate that shares a non-broad subject with the positive, for every doc-level family in `DOC_LEVEL` (`build_training_mix_v3.py:110`).

- **synthetic_hebrew and synthetic_translit are topical, not record-specific.** The spec defines them that way (`query_writing_spec.md:42-47`).
  - 46.7% and 45.5% of these anchors are 3 tokens or fewer.
  - Examples: "הלכות כשרות", "חשבונות מסחר", "Rif Yoma". Some anchors point at several different records: "כתובה קראית" ×8, "שטר חוב" ×3.
  - As doc-level rows, every other record of the topic is an in-batch negative and a preferred mined negative.
  - Only 15–20% of these rows carry `anchor_topic_ids`, and those cover festivals only.
- **content rows** (skipped by the festival guard, `build_training_mix_v3.py:946`):
  - 63% have a same-subject candidate.
  - 1,788 of 11,113 (16.1%) have a candidate whose label text is ≥0.8 token-Jaccard of the positive's label; 408 have 10 or more such candidates.
  - The negative is that candidate's label text, so the model is pushed to separate near-identical catalogue strings (inferred link to the work-subject regressions above).
  - 504 content rows have a positive carrying a festival topic and no guard at all.
- **Festival records still unguarded:** 273 synthetic doc-level rows have a festival-topic positive but an anchor without a festival term.
- **synthetic_specific is fine:** median 9 tokens, 0.2% are 3 tokens or fewer, and the queries are built from facts in the record.
- **Why keep it there:** within-subject known-item results improved (MRR 0.378→0.490, median percentile 0.027→0.005). The LoRA trial, with no mined negatives at all, also halved the median percentile, so in-batch negatives provide part of this.

**Change:** keep the preference for `synthetic_specific` only, and for all doc-level rows:
- ban candidates sharing the positive's festival topics;
- skip candidates whose label text is ≥0.6 token-Jaccard of the positive's;
- move hebrew and translit to subject-level masks.

## 6. Recommended configuration
Run two arms in one Colab session, save 4 checkpoints per arm, and pick one with a broad gate. A full evaluation takes about 5 minutes (`meta.embed_seconds` 260–340), so scoring 8 checkpoints is cheap compared with re-indexing.

| Parameter | v3-final value | Recommended | Why |
|---|---|---|---|
| Update type | full fine-tune | **Arm A (preferred): LoRA r16, α32, dropout 0.05, on q/k/v/o/gate/up/down, merged into fp32 before saving. Arm B: full fine-tune** | The LoRA trial matched full fine-tune on held-out subjects (+0.074 vs +0.073) with no memorisation (−0.012 vs +0.026 and +0.038 net). Merging keeps the serving contract unchanged. |
| Learning rate | 2e-5 | A: 1e-4 (as in the trial); B: 1e-5 | Less drift. Bibliography same-text cosine was 0.41–0.61 after 203 steps. |
| Warmup | 5% (≈10 steps) | 10% | Grad norm was already 22 at step 10. |
| Schedule / epochs | cosine / 1 epoch (203 steps) | cosine / at most 1 epoch | Shorter training comes from choosing an earlier checkpoint. |
| Checkpoints | none (`save_strategy="no"`) | every 25% of steps, `save_only_model=True` | Lets you pick a checkpoint; v3-run1 → v3-final showed large run-to-run swings. |
| Selection | focus AP (6 subjects) | broad gate (below), ties go to the earliest checkpoint or Arm A | Focus AP missed the holiday regression. |
| Batch / mini-batch | 256 / 8 | 256 / 8 (drop to 4 if out of memory) | Masks were validated at 256. |
| Scale (1/temperature) | 20 | 20 | Every run, including the trial, used 20; changing it moves the cosine scale and confounds the comparison. |
| Training window | 384 | 1024, with padding trimmed per mini-batch | Truncated positives drop from 19% to 0.44%, for about 1.43× compute. |
| Mining window | 512 | 8192 (full text) | Matches serving. |
| Eval window | 8192 | 8192 | Matches serving. |
| Same-subject negative preference | all doc-level families | `synthetic_specific` only, plus festival ban and near-duplicate guard | See section 5. |
| Doc-level families | specific, hebrew, translit, content | specific, content | Hebrew and translit queries are topical. |
| Field-label augmentation | none | Fold "Alternate titles" into the Description line in 50% of positive and negative texts | The training data has 2.6× more records with that line than the pool. |
| Pull toward base (Arm B) | none | About 10% of steps: cosine to precomputed base embeddings on about 20k texts (train-side records, bibliography pages not in its eval) | Limits drift without adding external data. I did not choose general retrieval pairs: they would be a new source to vet. |
| Saved query prompt | stock "web search" prompt | QUERY_PROMPT | Removes the handoff trap where `prompt_name="query"` gives the wrong prefix. |
| Sampler / seed / clipping | proportional / 13 / 1.0 | unchanged; log loss per dataset | Logged loss mixes datasets today. |

**Broad gate** (vs base_semantic, scored on the eval rebuilt from today's merged file, with v9 also scored on it):
1. Holiday MRR@100 and P@10 at least base level (CI low > −0.02); weak_share@20 and alt_only_share no higher than base. Add a held-out-only variant and recall on the Sukkot gold seed, which has human labels.
2. Focus AP improves (CI > 0), and llm_t2 and probe query sources are no worse than v9.
3. Known-item RR improves (CI > 0) and no query type has a lower point estimate. Read hebrew/translit known-item with caution, since those queries are topical.
4. Seen subjects worse by more than 0.05: at most about 2% (v9 had 15 of 740).
5. Trained-positive memorisation net gap at most +0.03.
6. Top-hub share no higher than base (0.0277); collection lift no higher than base.
7. Bibliography: 300 known-item queries, English R@1 and RR non-inferior.
8. Report drift without gating on it: median same-text cosine base vs tuned, random doc-doc cosine, participation ratio. The LoRA trial's mean same-text cosine was 0.72.

## 7. Code changes needed
All in `evals/embedding_audit/` unless a path says otherwise.

**Required:**
1. **Training arguments and LoRA.** In `colab/train_embedder_v3.py` `train()` (:644-719, arguments at :684-693): add save steps, `save_only_model`, warmup 0.10, and an `lora_r` option. LoRA wrapping can follow `train_text_lora.py:46-52`, including `enable_input_require_grads`. Call `merge_and_unload` before `model.save` (:700-701). Pin `peft` in the pip line (`make_notebook_v3.py:72`).
2. **Padding trim.** In `MaskedCachedMNRL` (:358-471), cut each mini-batch's features to its own longest attention mask. Inferred that ST 5.6.1 does not already do this; check its source. Raise `TRAIN_MAX_SEQ` (:81) to 1024.
3. **Mining window.** `mine_negatives(max_seq=512)` (:538) → `EVAL_MAX_SEQ`.
4. **Negative choice.** In `choose_negative` (:497-535): a per-family same-subject preference (specific only), ban the positive's festival topics, and add the 0.6 Jaccard guard on label text.
5. **Data:** `build_training_mix_v3.py:110` `DOC_LEVEL = {"synthetic_specific", "content"}`. This changes masks for about 4.5k rows; the check at :1249 follows the constant.
6. **Prompts.** Before `model.save` (:700-701), set `model.prompts = {"query": QUERY_PROMPT, "document": ""}`.
7. **Gate and checkpoint loop.** A `gate(reports)` next to `compare_reports` (:904-980), a drift metric (`evaluate` keeps the document matrix), and a notebook loop that evaluates every checkpoint (`make_notebook_v3.py` `final_cells`, :186-249).
8. **Holiday eval.** Add a held-out-only variant of `holiday()` (`eval_v3.py:601-656`) plus Sukkot-gold recall.

**Strongly recommended:**
9. Field-label augmentation in `build_datasets` (:286-320), deterministic per row.
10. Loss logged per dataset and family in `MaskedCachedMNRL.stats`.
11. Package the bibliography known-item eval (`v9_biblio/pages.jsonl` plus `results/v9_bib_known_item_queries.jsonl`) with the dataset.

**Arm B only:**
12. A precomputed base-embedding anchor dataset and a small cosine-to-teacher loss passed to `train()`, computed in the notebook while the base model is still loaded after mining.

**Not verified and inferred:**
- The ST padding and trimming behaviour.
- The cost estimates: token counts only, attention cost ignored.
- The link between near-duplicate content negatives and the work-subject regressions.
- Lengths come from the 2026-10-05 corpus; re-run `lengths.py` and `padcost.py` on today's corpus.
- Data outside my scope: 332 synthetic rows still pair a festival-named query with a scripture/tractate record, plus one t2t pair "אסתר↔פורים".

Scratch scripts and outputs are in `/private/tmp/claude-501/-Users-isaac-Documents-GitHub-genizah-search/6e892054-8bdc-4cf3-9c1c-de16a1b5474a/scratchpad/audit_recipe/`:
- bpe.py
- lengths.py
- lengths_v3corpus.json
- padcost.py
- aniso.py
- regress.py
- regress.json
- q4.py
- q4.json