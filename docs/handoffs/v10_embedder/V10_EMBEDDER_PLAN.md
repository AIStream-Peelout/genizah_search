# v10 embedder round: one plan (lead synthesis, 2026-10-09)

## 0. Two things to tell Isaac before any rebuild

**1. The v9 text contract is uncommitted, and v9 is being changed in place.**
- In HDA, `src/embeddings/semantic_text.py`, `text_representation.py`, `tests/test_semantic_text.py` and `tests/test_text_representation.py` are untracked (`git status` shows `??`).
- `elastic_index_genizah.py`, `index_merged_genizah.py`, `verify_reembedded_indexes.py`, `qwen_text_embedding.py` and `genizah_document.py` are modified but not committed. HDA HEAD is `069af54` (2026-10-07 18:18); v9 was built on 2026-10-08 01:06. So v9 was built from a working copy that exists nowhere in git.
- HDA `docs/handoffs/v9_index_switch_mbp.md` §0 says the "2026-10-09 catalogue enrichment" is in progress. It writes KTIV scholarly frames and a `Subjects:` line into v9 texts and re-embeds them, still labelled `semantic_v3`.
- The text-pipeline audit measured the effect at its 13:02 snapshot: 28,321 texts changed, and only 48,332 of 73,156 shared ids still match the training text.
- **Recommendation:**
  - Have the HDA owner stop writing into v9 and commit the code.
  - Do not switch the site to v9. Go from v8 straight to v10.
  - Do not build `bibliography_text_only_0.8` with v9 (the handoff says it is not built yet). Build it once, with v10.

**2. Record counts (the audits use different numbers; they agree once split this way).**
- Today's file: 78,113 records = 73,156 shared with the Oct-5 corpus + 4,957 new ids.
- 387 old ids are gone, so the net change is +4,570.
- The eligible pool grows from 60,493 (old corpus) to 63,652 (HEAD text on today's file).


## Where things live (2026-10-10)

| What | Where |
|---|---|
| Frozen inputs (merged file + sha, HDA source snapshot) | NAS `genizah_search_embedding_audit/v4/` |
| Corpus, labels, eval dirs, gate, training mixes | NAS `genizah_search_embedding_audit/v4/` |
| Training datasets (what Colab reads) | private HF dataset repo `isaacmg/genizah-embed-train`, tags `v4-final` (e9131c05), `v4-final-domk` (0c87a15d), `v4-lco` (4bf724d6) |
| Run outputs (reference reports, checkpoints, reports, vectors, `selection.json`) | private HF repo `isaacmg/genizah-embed-runs`, one folder per run (its README describes the layout) |
| Selected model | private HF repo `isaacmg/genizah-embed-qwen3-0.6b`, tag `v10-lora-s0211`, commit `af0252255d3e9b8b8432e09de7c5f0689d7ada34` (LoRA step 211 merged on CPU in fp32; weights sha256 107d00fd…). Local copy: NAS `v4/models/lora_s0211` (steps 106 / 159 merged beside it). The v9 tag `v3-final-run1` (dbfab73f) is untouched |
| Notebook | `evals/embedding_audit/colab/train_genizah_embedder_v4final.ipynb` (`make_notebook_v3.py --v4`) |
| Index input (v10 build) | the LIVE latest merge (`historical-document-analysis/.../merged/merged_shelfmarks.jsonl`; Isaac: every index is built from the newest merge with all scraped data). At build time: the 2026-10-11 01:24 merge, 79,397 records, sha256 9ed38c95…, 65,423 eligible; provenance copy NAS `merged_snapshots/merged_shelfmarks_20261011.jsonl`, recorded in the index `_meta.source`, text `semantic_v4` with scholarly frames OFF (`INDEX_SCHOLARLY_FRAMES = False` in HDA `index_merged_genizah`), matching the training text |

Everything a notebook reads or writes is on Hugging Face: no Google Drive, and no credential beyond the `HF_TOKEN`
secret the notebook already needs to read the data and push the model.

---

## A. Must-fix bugs in the current diff (confirmed findings → exact edits), in priority order

**A1. Typed weak labels** (findings 1, 5, 6, 9, 17; C1, C2 and E depend on this). Fixes the cause: one untyped weak set feeds both training masks and eval relevance.
- **`build_subjects.main` (:1156-1182):** replace `weak[doc_id] = {...}` (:1182) with a `weak_reason` map:
  - `canon - topics` → `"canon"`
  - `desc_weak - topics` → `"incidental"` if `incidental` else `"canon"`
  - `title_only - topics` → `"title"`
- Write each `subjects_v1.jsonl` row as `{doc_id, subjects, weak, weak_reason, canon, mentions}`. `weak` stays equal to `sorted(weak_reason)` so existing readers keep working; `canon` and `mentions` are defined in C1 and C3.
- **sukkot_gold (:1160):** keep the gold record strong (a documented override), but emit `gold_override: true`. The holiday ranking then treats a gold record that is canon-only or incidental as unjudged (finding 9).
- **`build_eval_v3.holiday_queries` (:237-270):** per festival, emit:
  - `relevant` = strong carriers;
  - `canon` = weak_reason canon;
  - `unjudged` = title ∪ incidental ∪ mention-only ∪ gold-override canon/incidental.
- **`eval_v3.holiday` (:601):** drop unjudged records from the ranking, count canon records as misses, and replace `weak_share@20` with `canon_share@20`.
- **Focus subjects (finding 1):** add `rel_mode` to every query in `subject_queries.json`:
  - `"bare"`: festival names with no law cue; canon carriers count as misses.
  - `"law"`: names matching `LAW_NAME_RE` (C2), plus every probe and llm_t2 query; canon carriers count as relevant.
  - `eval_v3.subject_retrieval` (:296) picks the relevant set per query by `rel_mode`.
  - In the first Colab session, re-score orig, sem and v9 on both label sets (`SCR/legacy/eval_final`, `SCR/new/eval_final`).

**A2. Yom Kippur matched as Purim (finding 2).**
- `HB` (:61) allows כ as a prefix letter, so הכפורים matches Purim.
- In `FESTIVAL_TERMS["purim"]` (:139), and in every other `{HB}פורים` rule, use `{HB}(?<!כ)פורים{HE}`.
- Change Yom Kippur to `{HB}(?:יום[- ]ה?כי?פורי?ם?|כי?פור|נעילה|יוה["״]כ){HE}`.
- Add asserts:
  - יום הכפורים, ליום הכפורים, יום-הכפורים → no purim;
  - יום-הכפורים → yom_kippur;
  - לפורים, ופורים → purim.

**A3. Canon detector (findings 3, 11, 12).** In `FESTIVAL_CANON_RE` (:149-152):
- Before the canon test, strip KTIV domain-label tokens with a `KTIV_DOMAIN_TOKEN_RE` built from the `domain:` vocab names ("Bible: Texts", "Mishnah: Texts", "Aramaic Targumim").
- Skip the canon demotion when `LITURGY_CUE_RE = piyyut|פיוט|liturg|prayer|תפיל|סידור|מחזור|haggad|הגד|kinot|קינות|selih|סליח` also matches.
- Replace the alternatives that never fire with `(?:rosh ha-?shanah|sukkah?|yoma|pesa[hḥ]im|megill?ah|ta'?anit) \d+[ab]?\b`.
- Add the Hebrew rules `{HB}(?:מסכת|מס['׳]){HE}` and `ע["״][אב]`, and bound the Hebrew canon tokens with HB/HE.
- Add `ימים נוראים|High Holy Days` to Yom Kippur and Rosh Hashanah, and `{HB}ר["״']ה{HE}` to Rosh Hashanah.

**A4. Festival term coverage and false positives (findings 4, 7, 8).** In `FESTIVAL_TERMS` (:129-146):
- **Additions:**
  - `{HB}ל(?:חג )?שבועות{HE}` and `חג שבועות`
  - `\bshavu[ʿ'’`]?(?:ot|os)\b`
  - `{HB}יוה?["״]כ{HE}`
  - `ne[ʿ'’`]?ilah?`
  - `sim(?:[hḥ]|ch)at torah`
  - `shemini [ʿ'’`]?a(?:tz|ts|ṣ|z|c)eret`
  - `(?:[hḥ]|ch)a?metz`
  - `tish[ʿ'’`]?a?h? b[e'’]?[-' ]?[ʿ'’`]?av` and `(?:ninth|9th) of av`
  - `\bsukot\b`
  - `hosha[ʿ'’`]?na rabb`
- **Replacements:**
  - `hoshan` becomes `\bhosha[ʿ'’`]?n(?:ot|os)\b|hosha[ʿ'’`]?n(?:a|ah) rabb`;
  - bare `matsa` becomes `matz(?:ah|ot|os)|matsot`.
- **Exclusions:** `(?:muslim|islamic|coptic) new year` and `יום כיפור קטן`.
- **Treated as incidental:** PGP type "Paraliterary text", and descriptions starting `dated` / `written on`.
- **Test:** a table-driven `--self-test` with each finding's positive and negative strings.
- **Order dependency:** `eligibility_admit_v4.json` (B3) is generated from the final `FESTIVAL_TERMS`, so regenerate it after this step.

**A5. Parse source fields, not the embedded text (finding 10; needed by B1).**
- These read Description, Alternate titles and Document Type from the corpus `text`:
  - `build_subjects` :1169-1171
  - `build_eval_v3` :256 and `catalogue_lines` / `concept_mentions` (:614-650)
- Under v4, `Alternate titles:` and `Document Type:` no longer exist in `text`, so these readers would silently return "".
- Switch them to the new corpus column `text_v3`, after `fold_field_lines` so continuation lines are included (1,828 records). Better still for Document Type: use the merged record, as `pgp_info` already does.
- Assert in `main()` that `records_with_description_festival > 0`, and that the Document-Type hit share is within ±20% of the v3 run.

**A6. Festival and mention guard on subject-level rows (finding 14); over-masking of canon known-item queries (finding 18).**
- `build_training_mix_v3.emit` (:946): drop `row["mask_level"] == "doc"`, keeping `family != "content"`.
- `anchor_festivals`: return [] when the anchor matches `FESTIVAL_CANON_RE` or a folio pattern `\b\d+[ab]\b|דף|ע["״][אב]`, or when the positive carries the festival only as canon-weak or not at all.
- `train_embedder_v3.row_segments` (:223): `seg0 = relevance_of(pos, keep=[row.get("anchor_subject"), row.get("anchor_mention_id")] + list(row.get("anchor_topic_ids") or []))`. This also carries C3.
- Add a subject-level tagged row to the trainer self-test and to `verify()`.

**A7. Bare festival names still trained onto canon records through `sefaria:*` (finding 13).**
- In `subject()`, in the allowed-names loop (:1031-1046), skip name `n` when all of these hold:
  - `festival_mentions(n)` is non-empty;
  - `not LAW_NAME_RE.search(n)`;
  - `d` is not a strong `topic:<festival>` carrier.
- Data outside the subject family:
  - drop t2t pairs where one side matches `FESTIVAL_TERMS` and the other is a canon name ("אסתר↔פורים");
  - drop synthetic rows whose bare festival anchor has a positive that carries that festival only as canon. The training-config audit counted 332 such rows.
- Post-build assert: 0 rows with a bare festival anchor and a canon-only positive.

**A8. Eval plumbing (findings 15, 16, 19-23).**
- **15:** `step0_local.query_texts` (:151-153) adds holiday and gate query texts.
- **16:** `train_embedder_v3._metric_rows` (:849), `compare_reports` (:904) and `format_table` (:983) get holiday rows: macro P@10, MRR@100 and canon_share@20, per-festival P@10, and the worst festival. Add a paired bootstrap clustered by festival (`eval_v3.paired_bootstrap` :205).
- **19:** `eval_v3.holiday` (:629-630) breaks ties pessimistically with `np.lexsort((is_rel[top], -s[top]))`.
- **20:** report `alt_only` base rate and lift per festival, computed from `text_v3`, with the macro weighted by hits.
- **21:** add a held-out-only holiday variant.
- **22:** add a torch-free mix self-test covering:
  - anchor_festivals;
  - doc-level and subject-level tagging;
  - `--no-festival-guard`;
  - `verify()` catching a planted held-out id.
- **23:** `package_dataset_v3`: `mask_overlap` counts `anchor_topic_ids` and `M:` ids; stats gain `festival_guard` and `mention_guard`; the README states the masking contract.

---

## B. Index-affecting changes (all ship in this one round)

The HDA working copy holds someone else's uncommitted work, so the HDA owner makes every HDA edit. We never modify HDA.

| # | Change | Decision | Where |
|---|---|---|---|
| B1 | `semantic_v4` = layout over `semantic_v3(v8)`: fold continuation lines; drop `Subjects:` and `Document Type:`; fold `Alternate titles:` parts into `Description:` (normalised whole-word dedupe, `MAX_ALT_CHARS=1000`). Text-pipeline Option A. | **Adopt.** Removes the field-label shortcut at the source. `Document Type` is a keyword guess, absent on KTIV-only records (0%) and present on 73% of FJP-only records, with 70 mid-word misfires. `Subjects:` puts the training labels literally into the text (domain names 38%→99%). Training-config's "fold in 50% of training texts" augmentation is unnecessary once the index text folds 100%. | gs `semantic_text.py`, new v4 block after `ineligible_reason` (:260). HDA `src/embeddings/semantic_text.py` (same block); `text_representation.py`: `TEXT_VERSION_SEMANTIC_V4`, `embed_text_for_version`, `text_sha256`. |
| B1a | KTIV scholarly frames in the embedded text (from the HDA working copy) | **Decide with the stock-model ablation (F7) before the mix is built.** Default: include if non-inferior. My measurement (`SCR/audit_lead/frames_literal.py`): the working copy changes the text of 100% of strong work pairs (6,076) and sefaria pairs (3,086), and 65% of domain pairs. Work pairs whose every name is literal rise from 3,406 to 3,976 (+9.4 pp); domain from 4,236 to 5,001. Inferred: about 21% fewer trainable work carriers, and canon names added exactly on canon records (the Purim→Esther direction). The HDA owner must store frames as their own attribute, not inside `alt_titles`, with a render flag (`--no-scholarly-frames` for the ablation). | HDA `genizah_document.py`; gs `build_corpus.py` passes the flag |
| B1b | KTIV `general_title` in the text | **No.** Adds KTIV-only text to 6,902 records; only 71 name a festival only there. | — |
| B1c | KTIV language fill in `from_merged_format` (:1635) | **Adopt.** It removes the KTIV layout tell (Language line on 0% of KTIV-only vs 86-91% of others), makes about 707 more records eligible, and corrects the ES `language` field on about 13.9k records. All 18,327 KTIV records carry `basic_catalog.language` (my scan): Hebrew 11,837; Judeo-Arabic 2,510; "Hebrew Judeo-Arabic" 1,366; Arabic 1,276; Aramaic 568. The proposed `.split()` breaks "Greek, ancient" / "Greek, Modern" (9 records), so use `re.split(r"(?<!,)\s+", s)`. Also map Judeo-→Judaeo-, pol→Polish, mul→drop. Freeze the map now: the Language line is embedded, so a later spelling fix means re-embedding. | HDA `genizah_document.py` |
| B1d | FJP string transcriptions → `TranscriptionSection` | **Adopt** (65 records, 74 items currently dropped). | HDA `genizah_document.py` (`validate_transcriptions` :350) |
| B1e | KTIV `full_catalog.notes` | **No** (provenance boilerplate). | — |
| B2 | Per-document text hash | **ES field `embedded_text_sha256`** (HDA's existing name, `update_catalogue_enrichment.py:114,177-184`, moved into `text_representation.py`). Keyword type, written on every document including ineligible ones. Corpus gets a new `text_sha256`. **Disagreement with text-pipeline:** keep `text_hash` = md5. md5 is recomputed and compared at `build_training_mix_v3.py:1092,1231`, `eval_v3.py:708` and `package_dataset_v3.py:239`; switching to sha256 would silently disable the held-out-text leak checks. | HDA `elastic_index_genizah.py` (`TEXT_VERSION_MAPPING` :44, `process_documents` ~:288, `_get_versioned_embeddings` :567); gs `semantic_text.main` (:273-332) |
| B3 | Eligibility v4 | **Adopt the text-pipeline rule:** decided on the v3 text, measured without `Subjects:`, frozen 64 labels (`boilerplate_descriptions_v3.json`, sha `d57204f7…`), never recomputed, plus a frozen admission list for short texts naming a festival or topic. Expected: +127 on HEAD text; +707 via the language fill and +30 admitted; about 64.4k eligible; **assert 0 removals versus v9.** `eligibility_admit_v4.json` is generated after A4 and sha-pinned in both repos. | gs `semantic_text.py` (`semantic_v4_ineligible_reason`, `content_lines_v4`, `admit_pattern`); HDA same; data file `docs/handoffs/v10_embedder/` + HDA `src/embeddings/data/` |
| B4 | Duplicate texts (55,833 unique of 63,652; 7,819 redundant vectors) | Keep one vector per fragment and add no identifiers back. **Embed once per `embedded_text_sha256`** in the indexer. Train and eval already de-duplicate by `text_hash`. Collapsing duplicates on the site is a later, query-side change (the field exists). | HDA `_get_versioned_embeddings` |
| B5 | Token windows | Train 1024 (`TRAIN_MAX_SEQ` :81, with a padding trim per mini-batch); mine with the full text (`mine_negatives max_seq` → `EVAL_MAX_SEQ`, :538); eval and serving 8192. Before `model.save` (:701), call `backend_tokenizer.no_truncation()` and `.no_padding()` and set `model.prompts={"query": QUERY_PROMPT, "document": ""}`. Keep the caps: transcription 1,000 characters, translation 500, alternate titles 1,000. | gs `colab/train_embedder_v3.py` |
| B6 | Contract and verification | `_meta`: `text_representation_version=semantic_v4`, `text_hash` rule, `document_canary.text_sha256`, `text_function{version, golden_fixture_sha256, hda_commit}`, `source{merged_records, merged_sha256}`, final `counts`. Add `text_hash` to `REQUIRED_META_KEYS` (verifier :66). `check_self_consistency_genizah` (:315) compares the stored sha before re-embedding. `DEFAULT_TEXT_VERSION` flips in the same commit as the v10 `FINETUNED_MODEL_*` pin. Callers to migrate: `describe_from_reads.py:1177-1185`, `update_generated_descriptions.py:211,439`, `update_catalogue_enrichment.py`. | HDA |

**B7. What must stay byte-identical** (the genizah_search `evals/embedding_audit/semantic_text.py` and HDA `src/embeddings/semantic_text.py` copies):
- **Constants:** `ID_PREFIXES`, `META_PREFIXES`, `ALT_PREFIX`, `MASK`, `MIN_CHARS`, `EDITOR_RE`, `PGP_EDITOR_RE`, `PRE_ID_RE`, `PROD_SHELFMARK_RE`, `EXTRA_ID_RE`, and the new `DESC_PREFIX`, `SUBJECTS_PREFIX`, `FIELD_PREFIXES`, `V4_DROP_PREFIXES`, `TITLE_SEP`, `MAX_ALT_CHARS`, `TITLE_KEY_RE`.
- **Functions:** `semantic_text`, `own_shelfmark`, `mask_shelfmarks`, `content_lines`, `alt_title_parts`, `fold_field_lines`, `title_key`, `merge_title_lines`, `semantic_v4_text`, `content_lines_v4`, `semantic_v4_ineligible_reason`, `admit_pattern`.
- **Data files:** the boilerplate json and the admission json, by sha.
- **The v8 producer** (`GenizahDocument.from_merged_format` / `create_text_representation`, and `merge_shelfmarks` description precedence) has a single copy, in HDA. It is pinned by commit sha, not by copying.
- **Enforcement:**
  - regex fingerprints as today;
  - a shared golden fixture of about 200 stratified real records `{v8_text, v4_sha256, reason}` (`tests/data/semantic_v4_golden.jsonl` and `evals/embedding_audit/test_semantic_v4.py`), plus a v3 fixture;
  - a full-corpus parity script (F5).

---

## C. Label and mix changes beyond festivals

| # | Change | Decision | Evidence |
|---|---|---|---|
| C1 | R1: route every topic matched on a `CANON_WORKS` frame to a per-record `canon` list. Festival canon stays weak; non-festival canon stays strong but is tagged. Add `rosh_hodesh` to `FESTIVAL_TOPICS` (:126) with its terms. Generalise `FESTIVAL_NAME_DROP` (:156) to `^(mishnah\|talmud\|tosefta\|megillat\|scroll of\|tractate\|משנה\|תלמוד\|מגילת\|מסכת)\b` for topic and sefaria vocabularies. | **Adopt** | rosh_hodesh pairs: 65% go to Mishneh Torah Qiddush ha-Hodesh. Sefaria labels are canon by construction (1,415 of 1,428 records). |
| C2 | R2: name-form gate in `subject()`: carriers keyed `(s, cls)`; canon carriers (including weak festival canon) take only `LAW_NAME_RE` names; non-canon carriers take only bare names; separate `ceil(k·√n)` quotas. `load_subjects` (:493) reads `canon`. | **Adopt** (it implements the training half of finding 1 and A7) | marriage: 68% of pairs canon vs 92% documentary among records naming it. divorce: 74% vs 91%. "bill of divorce": 8 of 9 pairs go to Gittin. In `mix_label`, festival law names went to Haggadot. About 707 bare→canon pairs removed (5.6%). |
| C3 | R3: mention mask. A second pass in `build_subjects` matches kept names (with Hebrew prefixes, minus the C4 stoplist) plus `FESTIVAL_RE` against `text_v3` Description and Alternate titles, and writes `mentions`. The mix adds `M:<id>`, `anchor_mention_id` and `M:` entries in `anchor_topic_ids`. Trainer: `BROAD_PREFIXES=("domain:","M:")` (:77); row_segments as in A6. Eval: mentions → `unjudged` for every subject (generalising `concept_mentions`, build_eval_v3 :624). | **Adopt.** This is the general fix for root cause 3 and subsumes finding 14. | 111,505 ids, mean 1.84 per record. 95% of domain rows have ≥10 literal-name negative candidates. 37-92% of records naming 12 concepts are unlabelled (eval-coverage table). |
| C4 | R4: stoplist `{אבל, עד, כשר, מס, ספר, נשים, מזון, לוח, עבד, ברכה, שמע, Get, Praise, Fast}` in `subject_name_table` (:584) | **Adopt** | 43 of 64 נשים hits are Seder Nashim; 48 of 55 מזון hits are Birkat ha-Mazon. |
| C5 | Same-subject negative preference (`choose_negative` :497-535) | **Restrict to `synthetic_specific`.** For every doc-level row: ban the positive's festival topics, and skip candidates whose label text is ≥0.6 token-Jaccard of the positive's. | content rows: 1,788 of 11,113 have a ≥0.8 near-duplicate candidate. 504 content rows with a festival positive are unguarded. |
| C6 | `DOC_LEVEL` (:110): training-config wants hebrew/translit moved to subject-level masks | **Partly adopt (my resolution).** `emit` sets `mask_level="subject"` for `synthetic_hebrew` / `synthetic_translit` only when the anchor has ≤3 normalised tokens; longer anchors stay doc-level. `verify()`'s `bad_mask_level` (:1249) must follow the same rule. | 46.7% / 45.5% of these anchors are ≤3 tokens (topical, e.g. "כתובה קראית" ×8), against 0.2% for specific. v9's Hebrew/translit known-item gains (MRR 0.148→0.251, 0.145→0.250) are an H8 gate item, so a full switch is not taken blind. |
| C7 | FJP/PGP Document Type → strong `domain:` labels | **No, not this round.** C3 masks cover the negative side, and H7 catches residual KTIV bias. Pre-build a fallback mix with the domain quota k halved, in case H7 fails. | Document Type is a keyword guess (70 misfires). It would change domain quotas with no eval yet. |
| C8 | Same-collection, different-subject hard negatives (audit:343-347, never implemented) | **No.** Covered by H9. | Collection lift with v9 training was neutral (2.80→2.76). |
| C9 | Leak filter: gate queries tagged `blocked` go into `build_leak_filter` (:428) before the mix is built. The v3-final2 probe leak (17 rows) disappears because `--probe-scope all` is rebuilt. | **Adopt** (required for gate validity) | 105 training rows use exact holiday query texts (finding 21). |
| C10 | Rebuild `frame_refs.jsonl` from the new corpus (`frames_to_sefaria.py` hard-codes `AUDIT_ROOT/corpus_v1.jsonl` → `AUDIT_ROOT/frame_refs.jsonl`, :104-115, and would overwrite v3 inputs) | **Adopt;** add `--corpus` / `--out-dir` | The current file has 5,618 rows from 2026-10-04 only. Without a rebuild, new KTIV records lack `sefaria:` labels and become unlabelled negatives. No audit checked this. |

---

## D. Training recipe (conservative)

| Parameter | v3-final (v9) | v10 | Basis |
|---|---|---|---|
| Arms | full fine-tune | **A (preferred): LoRA r16, α32, dropout 0.05, on q/k/v/o/gate/up/down, merged into fp32 before save.** **B: full fine-tune.** | LoRA trial: held-out +0.074 vs +0.073, memorisation −0.012 vs v9's +0.038 net |
| Learning rate | 2e-5 | A 1e-4; B 1e-5 | Drift: bibliography same-text cosine 0.41-0.61 after 203 steps |
| Warmup / schedule / epochs | 5%, cosine, 1 epoch | 10%, cosine, ≤1 epoch | Grad norm was 22 at step 10 |
| Checkpoints | `save_strategy="no"` (:690) | every 25% of steps, `save_only_model=True`: 4 per arm | v3-run1→v3-final swings (283 of 740 seen subjects worse) |
| Selection | focus AP (6 subjects) | pre-registered rule over the E gate | Focus AP missed the holiday regression |
| Batch / mini-batch / scale | 256 / 8 / 20 | 256 / 8 (4 if out of memory) / 20 | Masks validated at 256; scale held fixed for comparability |
| Training window | 384 | 1024, with a padding trim per mini-batch in `MaskedCachedMNRL` (:358-471) | Truncated positives 19%→0.44%; about 1.43× compute (inferred; ST trimming behaviour unverified) |
| Mining window | 512 | `EVAL_MAX_SEQ` (full text) | **Resolves disagreement:** text-pipeline said 1024, training-config 8192. Every v4 text is ≤1,933 tokens, so the two are identical except for cost. |
| Negatives | same-subject preferred for every doc-level row | C5 + C6 + A6 + C3 masks | — |
| Field-label augmentation | — | **none** | Superseded by B1 (the index text folds 100%) |
| Pull toward base (Arm B) | — | **deferred** (disagrees with training-config) | New loss code and a new data source; Arm A already is the low-drift option |
| Saved prompts / tokenizer | stock "web search" prompt, truncation 384 baked in | QUERY_PROMPT; no truncation, no padding; window recorded in `training_info.json` and the model card | Prompt and TEI traps |
| Logging | mixed-dataset loss | loss per dataset and family | — |
| LoRA merge check | — | merged vs adapter cosine ≥0.9999 on 100 texts before push; pin `peft` in `make_notebook_v3.py:72` | — |
| Seed / clipping | 13 / 1.0 | unchanged | — |

**Compute (inferred):**
- about 2.5 h per arm (6,196 s × 1.43);
- about 5 minutes per full eval: P, B3, B, V and 8 checkpoints, so about 1.2 h;
- run one Colab session per arm, plus one leave-concept-out run (H12).

---

## E. Acceptance gate before re-indexing (gate_v1)

**What to build:**
- **`evals/embedding_audit/build_gate_v1.py`** (Mac side; stdlib only, json only), writing to `$V4/eval_final/gate_v1/`:
  - `queries.jsonl`, `concepts.json`, `records.jsonl`, `probes.json`, `thresholds.json`, `review_sample.csv`.
  - Relevance comes from source fields (`text_v3` and the merged record), never from the v4 text, so it holds for every text variant.
- **`evals/embedding_audit/eval_gate.py`** (numpy only): `evaluate_gate(...)` and `verdict(reports, cand, refs)`.
- **Hooks:**
  - `train_embedder_v3.evaluate` (:725) calls `eval_gate` and saves `vectors_<label>.f16.npy` to Drive, not the Hub;
  - `compare_reports` (:904) prints the verdict;
  - `package_dataset_v3` `EVAL_FILES` ships `gate_v1/`;
  - `make_notebook_v3.final_cells` (:186-249) loops over the checkpoints.

**Reference models, all scored on the same pool** (v4-eligible ids, each model on its own text):
- **P** = stock model on v8 text;
- **B3** = stock model on the v9 text (semantic_v3 HEAD content);
- **B** = stock model on the chosen v4 text;
- **V** = v9 on semantic_v3.

The eval marks `v9_trained` ids (anchors, positives and negatives in `v3/train_mix_v3final.jsonl`) and passes them as `ignore_ids` when scoring V. Text regrouping can move records V trained on into the held-out side.

**Families, in build-priority order:**
- **P1 (required before training):**
  - G1 holiday names (about 100, including the Hebrew-prefixed forms הפורים / בפסח and the real typos from `rag_audit_20261007/chat_sessions.jsonl`);
  - G2 holiday descriptive and nuanced (seed + concept_probe, blocked from training);
  - G3 canon (about 40, written: "Book of Esther", "Tractate Pesahim", "Numbers 29", …);
  - G10 known-item (2,003);
  - every existing eval_v3 section;
  - G11 real-user queries (about 45);
  - the H5/H6 probes;
  - population cuts (source group × description language × length × held-out × new-since-v9).
- **P2 (also required: "work well all round"):**
  - G4 documentary descriptive (about 120, written by a Claude session to `query_writing_spec.md`; no LLM server);
  - G5 FJP-heavy genre text rules (about 60);
  - G6 people and places (about 70);
  - G7 scripts and languages (about 20).
- **Rule audit:** 20 matched and 20 unmatched records per concept, with precision ≥0.9 before freezing. Text-only reading; marked as inferred precision.
- **Report-only:** G12 out-of-domain (60 queries) and G13 shelf marks.

**Hard criteria.** All must pass; NOT MEASURED counts as FAIL.

| id | Criterion |
|---|---|
| H0 (text function, stock model only, before mix build) | B(v4) vs B3 per family Δ ≥ −δ. Frames variant chosen when Bf vs Bnf: every family Δ ≥ −δ, canon_share@20 ≤ Bnf + 0.03, and G3/G10 Δ ≥ 0. Otherwise no frames. |
| H1 | Each of G1-G7 and G10, vs B and vs P: Δ ≥ −δ and Δ ≥ −2·SE (δ 0.03 for P@10_dd, 0.02 for AP/MRR). |
| H2 | Cells with ≥10 queries: Δ ≥ −max(0.10, 2·SE); at most one cell per cut below −2·SE; none below −3·SE or −0.20. |
| H3 holiday (vs P) | strict P@10_dd ≥ P − 1 SE; desc-only ≥ 0.82·P; broad ≥ P; worst-festival drop ≤ 0.10; worst bare-name drop ≤ 0.30 (where P ≥ 0.3); alt-only lift ≤ 1.25 macro and ≤ 1.5 per festival with ≥30 relevant. The absolute alt-only share ≤ 0.50 is dropped because base rates run 0.04-0.76. |
| H4 canon | canon_share@20 ≤ P + 0.05 (v8 0.105, v9 0.348); tractate/law share ≤ P + 0.03; G3 passes H1 (over-correction guard). |
| H5 | Title-position probe on v4 (the concept moved from the appended-title position to the head): \|mean Δz\| ≤ 0.25 and ≤ \|B\| + 0.10. The old label-swap probe runs on V/B3 for diagnosis only, since v4 has no label. |
| H6 | Seed's 10 cases stay at rank ≤50. Held-out sweep over pairs where P ranks ≤20: ≥95% have candidate rank ≤ max(50, 3·P rank); ≤2% fall below rank 500. |
| H7 | Per source group, R@100 ratio vs P and vs B ≥ 0.90; KTIV lift@10 ≤ P + 0.25; the new-since-v9 cell passes H2. |
| H8 keep v9's gains | Focus AP: paired CI vs V not entirely below 0. Known-item MRR Δ ≥ −0.01 vs V (specific / topical / implicit). **Hebrew/translit known-item: Δ ≥ −0.05 vs V and CI-low > 0 vs B.** This resolves eval-coverage's −0.02 against C6, and the training-config audit's warning that these queries are topical. Seen-subject AP Δ ≥ −0.02 vs V. Seen subjects worse by >0.05 vs B: ≤2%. |
| H9 structure | Collection lift ≤ B + 0.10. Duplicate/short share ≤ B. **Top-hub share ≤ max(B, 0.02)**: eval-coverage's absolute 2% fails base itself (2.8%). distinct@10 ≥ B − 0.3. **Memorisation net gap ≤ min(V, 0.04)**: reconciles 0.03 (training-config) with 0.06 (eval-coverage); v9 measured +0.038. |
| H10 | `bib_v9_check score`: no paired CI vs base entirely below 0, for dense, 60/40, 30/70 and chat 50/50. |
| H11 integrity | Text-hash mismatch 0; every query scored; minimum sizes met; no blocked gate query is an exact or contained training anchor (recomputed from the mix). |
| H12 leave-concept-out | Required (Isaac's concept-first rule): the chosen recipe trained on the non-final split. Held-out subject AP CI-low > 0 vs B, and ≥ 0.182 − 1 SE. |

**Soft (reported, not gating):** typo robustness, out-of-domain AUC, score-spread ratio 0.5-2× vs B, drift (same-text cosine, random doc-doc cosine, participation ratio), G11 per query, G13.

**Pre-registration:**
- `thresholds.json` is frozen and its sha printed in the notebook before training.
- Selection: among checkpoints passing every hard criterion, take the highest mean H1 Δ vs B. Within ±0.005, prefer the earlier checkpoint, then Arm A.
- If nothing passes, do not index. Report the failing criteria and retrain (the fallback mix is pre-built). Retraining is cheap; re-indexing is not.

**After the pre-index gate passes:**
- run the handoff's official B1-B9 with absolute thresholds, plus M4;
- run the chat eval sets (agentic_rag v1, crosslingual, multiturn) on `dev_backend_local.py` (:8010) against the new indices (ask first);
- run `v9_cosine_calibration.py` on the fragment vectors.

---

## F. Rebuild order (nothing overwrites v3; all outputs go under `$V4`)

```
GS=/Users/isaac/Documents/GitHub/genizah_search/evals/embedding_audit
HDA=/Users/isaac/Documents/GitHub/historical-document-analysis
AR=/Volumes/home/studio_offload/genizah_search_embedding_audit ; V4=$AR/v4
MERGED=$HDA/src/datasets/raw_data/cairo_genizah/merged/merged_shelfmarks.jsonl
```

**Steps 0-2: prerequisites and frozen inputs**

0. **Prerequisites (owner and Isaac):**
   - Commit A, C, and the gs side of B on `eval-retrieval`.
   - The HDA owner commits the B changes on a branch and records `$HDA_SHA`; the HDA text code is frozen until v10 is indexed.
   - Copy `holiday_eval_seed.json` and `regenerate_silver.py` from the MBP into `$GS/data/`.
1. **Freeze the inputs** (the merged file changes as KTIV scraping continues):
   - `mkdir -p $V4 && cp $MERGED $V4/merged_shelfmarks_20261008.jsonl && shasum -a 256 $V4/merged_shelfmarks_20261008.jsonl > $V4/merged.sha256`
   - `git -C $HDA archive $HDA_SHA | tar -x -C $SCR/hda_$HDA_SHA` (read-only on HDA).
2. **v8 corpus** (new `build_corpus.py` arguments `--hda-root` and `--merged`; it writes the HDA sha and merged sha to `.stats.json` and refuses a dirty tree):
   - `$HDA/.venv/bin/python $GS/build_corpus.py --hda-root $SCR/hda_$HDA_SHA --merged $V4/merged_shelfmarks_20261008.jsonl --seed $AR/sukkot_seed_feature_branch.json --out $V4/corpus_v1.jsonl`
   - Ablation corpus: add `--no-scholarly-frames --out $V4/ablation/corpus_v1_nf.jsonl`.
   - Confirm `sukkot_seed_feature_branch.json` is the v3 seed (inferred).

**Steps 3-6: text, labels and eval data**

3. **Frame refs:** `python3 $GS/frames_to_sefaria.py --corpus $V4/corpus_v1.jsonl --out-dir $V4` (new arguments; report the refs missing from `curated_refs`).
4. **v4 text:**
   - `python3 $GS/semantic_text.py --corpus $V4/corpus_v1.jsonl --text-version semantic_v4 --boilerplate-file docs/handoffs/v9_embedder/boilerplate_descriptions_v3.json --admit-file docs/handoffs/v10_embedder/eligibility_admit_v4.json --out $V4/corpus_semantic.jsonl`
   - Writes `text, text_v3, eligible, reason, text_hash(md5), text_sha256, …`.
   - Run the same for the ablation corpus, to `$V4/ablation/corpus_semantic_nf.jsonl`.
   - Write `$V4/ablation/corpus_semantic_v3.jsonl` (the v9 text, for V and B3).
5. **Parity:**
   - `python3 $GS/test_semantic_v4.py` (golden fixture);
   - new `parity_v4.py`: imports both semantic_text copies and asserts equal `text_sha256` for all 78,113 records, plus 0 eligibility removals versus v9;
   - HDA `tests/test_semantic_text.py` and `tests/test_text_representation.py`. Ask Isaac first: pytest on this Mac.
6. **Subjects, eval and gate:**
   - `python3 $GS/build_subjects.py --self-test`
   - `python3 $GS/build_subjects.py --semantic $V4/corpus_semantic.jsonl --merged $V4/merged_shelfmarks_20261008.jsonl --frame-refs $V4/frame_refs.jsonl --out-dir $V4 --curation-csv $V4/pgp_tag_curation.csv`. `--curation-csv` is an output (:1223-1224); the default would overwrite the repo's `results/pgp_tag_curation.csv`.
   - `python3 $GS/build_eval_v3.py --final --semantic $V4/corpus_semantic.jsonl --subjects $V4/subjects_v1.jsonl --vocab $V4/subject_vocab.json --corpus-v1 $V4/corpus_v1.jsonl --merged $V4/merged_shelfmarks_20261008.jsonl --v3-eval-dir $AR/v3/eval --out-dir $V4/eval_final`
   - Run the same command without `--final`, with `--out-dir $V4/eval_lco`.
   - `python3 $GS/build_gate_v1.py --eval-dir $V4/eval_final --semantic $V4/corpus_semantic.jsonl --subjects $V4/subjects_v1.jsonl --seed $GS/data/holiday_eval_seed.json --chat-sessions evals/results/rag_audit_20261007/chat_sessions.jsonl --v9-mix $AR/v3/train_mix_v3final.jsonl --out-dir $V4/eval_final/gate_v1`
   - Then the rule audit, and Isaac freezes `thresholds.json`.

**Step 7: choose the text (Colab session 1, stock and v9 models only, about 30 minutes)**

7. Package an eval-only dataset `v4-ablation` (`package_dataset_v3.py --name v4-ablation …`; it ships the three corpus variants).
   - Score P, B3, Bf, Bnf and V. Apply H0.
   - Also re-score orig, sem and v9 on the legacy and new label sets (A1).
   - **If no-frames wins,** redo steps 4-6 from `corpus_v1_nf` into `$V4` (move the frames outputs to `$V4/rejected_frames/`).

**Steps 8-9: mix and dataset**

8. **Mix:**
   - `python3 $GS/build_training_mix_v3.py --self-test`
   - `python3 $GS/build_training_mix_v3.py --semantic $V4/corpus_semantic.jsonl --subjects $V4/subjects_v1.jsonl --vocab $V4/subject_vocab.json --eval-dir $V4/eval_final --corpus-v1 $V4/corpus_v1.jsonl --merged $V4/merged_shelfmarks_20261008.jsonl --frame-refs $V4/frame_refs.jsonl --gate-dir $V4/eval_final/gate_v1 --v3-eval-dir $AR/v3/eval --v3-stats $AR/v3/train_mix_v3final.stats.json --out $V4/train_mix_v4final.jsonl`
   - Then `--verify $V4/train_mix_v4final.jsonl`.
   - Also build `--subject-k 2.5`-for-domain as `$V4/train_mix_v4final_domk.jsonl` (the fallback), and `--eval-dir $V4/eval_lco --out $V4/train_mix_v4lco.jsonl`.
9. **Package:**
   - `python3 $GS/colab/package_dataset_v3.py --name v4-final --mix $V4/train_mix_v4final.jsonl --eval-dir $V4/eval_final --corpus-v1 $V4/corpus_v1.jsonl`. Output goes to `$AR/hf_dataset/v4-final`; the script refuses to overwrite.
   - `DATA_FILES` (:84) gains `corpus_semantic_v3.jsonl` for V and B3.
   - The owner uploads it as the new dataset tag `v4-final` (publishing step: Isaac's approval; `v3-final` is never moved).
   - Repeat for `v4-lco`.

**Steps 10-11: training (Colab)**

10. **Notebook:** `python3 $GS/colab/make_notebook_v3.py --final --data-revision v4-final --run-name v4-final-run1 --previous-revision v3-final --previous-label v9 --out $GS/colab/train_genizah_embedder_v4final.ipynb` (with the arm, checkpoint and gate cells added).
11. **Sessions:**
    - Session 2 (Arm A) and session 3 (Arm B): mine with the base model at full text → train → evaluate every checkpoint (eval_v3 + gate) → verdict.
    - Then `bib_v9_check score` on the winner.
    - Session 4: the winning recipe on `v4-lco` (H12).
    - The owner pushes the merged winner to `isaacmg/genizah-embed-qwen3-0.6b` as a new commit and pins the sha (tag `v10`; never move the v9 tag).

**Step 12: index (Studio, HDA, Isaac's approval for each production step)**

12. One HDA commit flips `DEFAULT_TEXT_VERSION=semantic_v4` and `FINETUNED_MODEL_*`. Then:
    - `python -m src.datasets.indexing.index_merged_genizah --merged $V4/merged_shelfmarks_20261008.jsonl --index genizah_merged_v10`
    - `python -m src.datasets.indexing.bibliography.index_all_bibliography` with a new constant name
    - `python -m src.datasets.indexing.verify_reembedded_indexes` (both indices)
    - a read-only scroll checking that the corpus `text_sha256` equals the index `embedded_text_sha256` for every doc_id
    - re-run `update_generated_descriptions` on v10
    - re-run the viz precompute
    - backend changes: startup gate adds `text_representation_version=semantic_v4` and the query canary; `semantic_eligible` filters; shelf-mark routing
    - dev backend on :8010 (ask first) → Isaac's approval → switch.
    - Keep v8 and the old embedder for rollback.

---

## G. Open decisions for Isaac

1. **v9 in-place enrichment and uncommitted HDA text code.** Recommend:
   - stop writing into v9;
   - the HDA owner commits;
   - never switch the site to v9;
   - go v8 → v10.
2. **Bibliography index.** Recommend: do not build it with v9; build it once with v10 (4,944 pages).
3. **Scholarly frames in the embedded text.** Recommend: let H0 decide (stock-model ablation). Include them if non-inferior: they are high-precision identifications, and BM25 gets them either way.
4. **KTIV `general_title` in the text.** Recommend: no.
5. **KTIV language fill** (the site's language facet changes for about 13.9k records; the Language line enters the text). Recommend: yes, with the frozen normalisation map.
6. **FJP/PGP Document Type as strong domain labels.** Recommend: no this round. The fallback mix covers an H7 failure.
7. **Gate thresholds.** Confirm `thresholds.json`, including my resolutions:
   - H8 Hebrew/translit −0.05 vs V and > B;
   - memorisation ≤ min(V, 0.04);
   - top hub ≤ max(B, 0.02);
   - the alt-only absolute cap dropped.
8. **Inputs only Isaac can supply:**
   - copy `holiday_eval_seed.json` and `regenerate_silver.py` from the MBP (`terraform_gcp_search`, `seo-plan-2`);
   - is any search-box query log available? None is in the repo.
9. **Compute.** Two arms plus one leave-concept-out run: about 4 A100 sessions. Recommend yes. H12 follows Isaac's "gate on leave-concept-out" rule.

---

## H. Completeness critic: what could still force a second re-embed

1. **Merged-file churn.** The indexer must read the frozen `$V4/merged_shelfmarks_20261008.jsonl`, not the live file; otherwise texts differ from the training corpus. Merge precedence (PGP > KTIV title > TEI > FJP) is part of the text contract and must be frozen too.
2. **HDA text code drift.** It already happened today (28,321 texts). The golden fixture plus the HDA commit pin are the only real guard; regex fingerprints did not catch it.
3. **Machine-generated descriptions.** They are planned for the 14,461 vector-less documents. The pilot format (130 rows, `generated_descriptions_pilot_passA.jsonl`) is `Description: … [machine-generated from automatic transcription]\nLanguage: …\nDocument Type: …`.
   - Under v4 the Document Type line is dropped.
   - This text type is in no training set and no gate.
   - Fix its v4 layout now and add the 130 pilot rows as a gate population. A later layout change means a version bump and a full rebuild.
4. **Consensus or AI transcriptions entering fragment text.** prod-mbp `ai_transcriptions.py` (e765768) has no vector code (inferred: BM25-only side index). If they are ever to go into `Transcription:` for the 12,108 imaged documents, decide the layout now.
5. **Shelf-mark masking on the 4,957 new records** (for example NLI system-number ids such as `987007606066605171`). Unchecked by every audit. Before freezing, count residual id-like tokens (≥6-digit runs, `Ms.` / `Cod.` patterns) in the v4 texts of new records. A leak found later forces a text version bump.
6. **Sefaria label coverage for new records** (C10). Not index-affecting, but it does affect the model.
7. **Language map spellings** (B1c): the value is embedded text, so freeze it.
8. **Bibliography text function.**
   - Keep `Shelf marks mentioned:`, drop `Source PDF:`, and settle chunking now.
   - Pages run to 7,655 tokens, while training stops at 1,024; H10 measures retrieval but cannot justify a later chunking change.
9. **V contamination.** v4 regrouping and the Language lines move records across the text-grouped split (`split_side`, build_eval_v3 :380). Without `v9_trained` as ignore_ids, V is over-scored and H8 is too lenient.
10. **ES vector storage.** `dense_vector`, `index: true`, cosine (`elastic_index_genizah.py:797-800`); grep finds no `_source` excludes. So a later mapping change (int8 or kNN tuning) can reindex without re-embedding (inferred). Keep it that way.
11. **Base model choice.** Qwen3-Embedding-4B was never benchmarked. Moving to it later means a full re-embed. Accept 0.6B explicitly for this round (serving RAM).
12. **Tokenizer and prompts saved in the model** (B5). A TEI or raw-`tokenizers` consumer would otherwise truncate at 384 tokens.
13. **Non-forcing items** (no re-embed needed): hybrid weights and cosine calibration, the site's duplicate collapse, query-canary and text-version startup checks, a later change to which documents are marked eligible, the viz precompute.

Scratch outputs from this synthesis are in `/private/tmp/claude-501/-Users-isaac-Documents-GitHub-genizah-search/6e892054-8bdc-4cf3-9c1c-de16a1b5474a/scratchpad/audit_lead/`: `ktiv_lang.py` (KTIV language values) and `frames_literal.py` (frames effect on literal-name drops).