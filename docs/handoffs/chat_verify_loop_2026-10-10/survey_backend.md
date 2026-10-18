# Backend/frontend reference survey: quote rule (N2a-delete) + N3

Prepared 2026-10-10. Every line number was re-checked against the read-only snapshot of `origin/prod-mbp` @ `e765768`, which is `e76576869013ee41775223a7415d4714c80003b7` (`git rev-parse origin/prod-mbp` today).
Snapshot root: `scratchpad/prod-mbp`. The audit's line numbers still match exactly, because prod-mbp has not changed since the audit.

Abbreviations: `lms` = `src/backend/lms_agentic_search.py` (4,763 lines), `app` = `src/backend/app.py`, `ChatUI` = `src/frontend/src/ChatUI.jsx` (2,733 lines).

Companion file: `handoff/proto_quote_check.py`. It is a reference prototype of the confirmed quote rule, not production code. It was run read-only over the P2 replay drafts; results are in section A.6.

---

## A. The quote matcher today, and what the confirmed rule needs

### A.1 Current code

| What | Where | Notes |
|---|---|---|
| `_normalize_verification_text` | lms:865-884 | NFKC + lower. It maps curly quotes, gershayim, geresh and maqaf to ASCII, strips nikud/cantillation (U+0591-05C7) and collapses whitespace. It does **not** remove punctuation, so `"India Book."` vs `'India Book,'` fails. It is also used for claim anchoring (`locate_claim_sentence`), dedup and protected-quote checks, so **leave it alone** and add a new quote-only normaliser. |
| `extract_direct_quotes(text) -> List[str]` | lms:887-908 | Regex `r'"([^"\n]{8,})"\|“([^”\n]{8,})”'` (lms:894), plus gershayim quotes at word boundaries (lms:899). It returns bodies only, with **no offsets**. Because of the `{8,}` minimum, a short quote such as `"הכפישה"` is skipped and its closing mark pairs with the next opening mark, which turns prose into a "quote". Reproduced: `' is derived from Lamentations 3:16 ('` (C2 `tisha_bav_kinnot`, where 7 correct sentences were deleted). |
| `bound_direct_quote` | lms:911-923 | Exhaustion fallback only. |
| `find_quote_source(quote, sources) -> Optional[int]` | lms:926-952 | Normalised substring check against `source["quoteable_text"]`. For quotes of 40+ chars it also accepts a sliding-window `SequenceMatcher` ratio ≥ 0.93 (lms:943-951). That fuzzy leg contradicts "no fuzzy tolerance by default", so remove it in delete mode. It is also O(n·m). |
| Where `quoteable_text` comes from | lms:766-836 `build_bibliography_source_context` | `extract_quoteable_main_text` (lms:747-763) keeps the text after `Main text:`. It is capped per source at `quote_cap = min(1800, max(400, quote_budget // n))`, or `hebrew_quote_cap = max(500, quote_cap // 3)` when the page is more than half Hebrew (lms:786-796, 818-820). The dict is stored at lms:829-835: `source_number`, `citation`, `prompt_text`, `evidence_text`, and `quoteable_text` (lms:834). Only bibliography sources have `quoteable_text`. |
| Graph sources | lms:839-862 `build_verification_sources` | Appends Neo4j sources with `citation = "Neo4j knowledge graph record for {scholar}"` and an `evidence_text` built by `build_graph_prompt_context` (lms:695-744), capped at 2,400 chars. Work titles appear as `- {title}{year}; article_id=…` lines (lms:711-724). There is no `quoteable_text`, so graph text can never verify a quote, which is correct. |
| Citation line (for title exemption) | lms:801-807 | `citation = f"{author_text}, *{title}*" + (f", p. {page}")`, where `title = bibliography.get("title")` is the full index title. It is available in every source dict as `source["citation"]`, and the raw titles are in `state["bibliography_results"][i]["title"]`. |
| Shelf-mark check | lms:1194-1210 `find_shelfmark_source`; lms:1166-1191 `shelfmarks_equivalent`; `shelfmark_normalizer.py:72 ShelfmarkNormalizer.to_canonical_id`, `:181 are_equivalent`, `:266 detect_shelfmarks` | It searches `source["evidence_text"]` (summary + quoteable + shelf-mark list, and the graph text), first for the canonical id as a substring and then by canonical comparison of detected marks. Draft marks come from `detect_shelfmarks(draft)` (lms:3723). Note that `detect_shelfmarks("Or. 1081 J 67")` returns only `Or. 1081`; underscore labels such as `T_S_12_243` are not detected in prose, but their canonical form matches a graph record by substring. |
| User question | `state["user_query"]` (set at lms:4532 / 4610); `state["resolved_query"]` (4B rewrite); prior turns via `conversation_turns(state)` (lms:1280-1287); `conversation_grounding_text(state)` (lms:1382-1394) joins them all | The confirmed rule exempts spans copied from **the user's question**, so use `user_query` only. Adding `resolved_query` or prior user turns is optional. Do not use assistant turns: they would let an earlier unverified quote launder itself. |
| Synthesis prompt, quote rules | lms:3294-3298 (rule 1, "Format quotes as: **Author**, p. X: "quote text"" at 3297); lms:3319-3323 (rule 6, "verbatim (or near-verbatim)"); lms:3342-3346 (rule 11, original script inside straight double quotes) | Add to rule 1: put punctuation outside the closing quote mark; italicise titles, never double-quote them; copy quotations exactly; use … only for omissions. In rule 6, drop "(or near-verbatim)". |
| Verifier prompt | lms:3768-3822; "TITLES COUNT" at 3788; "Do not separately return direct quotes or shelf-mark identifiers" at 3807 | No change is needed; the verifier keeps ignoring quotes and shelf marks. |

### A.2 Sentence splitter: two gaps that become hard prerequisites once deletion is immediate

`split_sentences` (lms:972-995) uses `_SENTENCE_BOUNDARY_REGEX = (?<=[.!?])\s+(?=[\"“(\[]?[A-Z0-9א-ת])` (lms:959). Verified with a scratch run of the same regex:

1. **No boundary after a closing quote.** `Goitein called it the "India Book." He began in 1948.` stays one sentence, so deleting the quote's sentence would also delete the next one.
2. **No boundary before a bold name.** `First sentence. **Goitein** wrote x.` stays one sentence. The synthesis prompt mandates `**Author**`, so this pattern is everywhere, in English and in Hebrew (`**לוין** מציין`).

Proposed regex, tested on both cases plus Hebrew:
`(?:(?<=[.!?])|(?<=[.!?][\"”’״)\]]))\s+(?=(?:\*\*|[*_])?[\"“״(\[]?[A-Z0-9א-ת])`

Abbreviation masking (lms:964-968) is unchanged. `split_sentences` is shared with flag anchoring (`annotate_answer_with_flags`, `locate_claim_sentence`), so re-run `test_sentence_split_keeps_citations_and_abbreviations_intact` (tests:1064) and `test_flag_wrapping_covers_whole_citation_sentence` (tests:1084). Both should still pass. A dedicated splitter for deletion only is the fallback if anchoring regressions appear.

**Delete by offset, not by fuzzy lookup.** `remove_sentences_containing` (lms:1121-1140) uses `locate_claim_sentence` (lms:998-1028). That function returns the first sentence where the claim is in the sentence *or the sentence is in the claim*, and otherwise falls back to a SequenceMatcher ≥ 0.55 match. With a mis-paired span this deleted unrelated sentences (C2). The new extractor should return `(start, end, body)` offsets. Deletion should then remove exactly the sentence(s) of that line that overlap `[start, end)`, found with `split_sentences(line)` plus `line.find(sentence, cursor)`. A paragraph or bullet left empty is dropped, and so is a heading left with no body.

### A.3 The confirmed rule, mapped onto code

All of this is deterministic and involves no LLM, except the single smoothing call in A.4.

1. **Extract (offsets, position pairing).**
   - A `"` or `״` opens only after start, whitespace, `( [ { — – - : / * _`. It closes only before end, whitespace, `. , ; : ! ? ) ] } — – - * _ … ' ’`.
   - A mark with letters on both sides is an abbreviation when exactly one letter follows it (`ע"ו`, `כ"י`, `ור"ד`, `י"ג-יד`).
   - **A Hebrew prefix quote opens a quotation:** 1-3 prefix letters `[ושהכלבמ]` at word start, then a mark, then a multi-letter word (`ש"נשתמרו …"`, `ל"תנ"ה, צר 119"`). The prototype found this pattern in real 35B and 27B drafts; without the rule a real quotation is silently skipped.
   - Curly `“ ”` and `« »` pair directly. Single quotes are never delimiters (nested `'Old List'`, apostrophes).
   - Pairing resets at `\n`.
   - Spans under 3 words are terms: they are not checked and not deleted, but their marks are consumed so they cannot mis-pair. This alone retires the `"India Book."`, `"Old List,"` and `"גאון הגאונים"` false positives.
2. **Normalise quote and source identically.**
   - NFKD, then drop all combining marks (category Mn: Hebrew nikud and cantillation, and also Latin diacritics, so `Yijū` = `Yiju`), then NFKC, then casefold.
   - Editorial brackets `[ ]` are removed with no space (`במעשי[ו]`). Every other Unicode punctuation or symbol (categories P* and S*) becomes a space; that covers all quote styles, geresh/gershayim, maqaf, dashes and guillemets.
   - Fold Hebrew finals (ך→כ etc.).
   - Collapse whitespace, then **compare with spaces removed**. This one choice absorbs OCR line-break hyphenation (`where- abouts`, `Ara- bic`, both seen in the Goitein sources), maqaf, and `י"ג-יד` vs `י ג י ד`. A changed letter or digit still fails: `between 889 and 988` vs a source with `998` fails in the prototype.
   - Ellipses (`...`, `. . .`, `…`) split the quote into segments, and each segment must occur in order.
3. **Verified iff** every segment is a substring of the normalised `quoteable_text` of any bibliography source. Today's code accepts any source rather than only the cited one, which is simpler and needs no citation parsing; keep that. There is no fuzzy leg.
4. **Exemptions** (not quotations, never deleted):
   - the span is a substring of the normalised citation lines plus graph `evidence_text` (titles, including graph work titles: P09);
   - the span is a substring of the normalised `user_query`.
5. **Shelf marks.** A detected mark with no `find_shelfmark_source` match, and not equivalent (via `shelfmarks_equivalent`) to a mark in `detect_shelfmarks(user_query)` (P06), is treated like an unverifiable quote: its sentence is deleted.
6. **Delete** each sentence containing an unverifiable quote or shelf mark, then run one guarded smoothing pass (A.4) and add a closing note (A.5).

### A.4 Where it runs: a new node before the 35B

**Recommended: a new graph node `check_quotes`.** The edges are built at lms:1729-1757:

- replace `workflow.add_edge("synthesize_answer", "verify_claims")` (lms:1736) with `synthesize_answer → check_quotes → verify_claims`;
- add the node to the list at lms:1719-1729;
- add `"check_quotes"` to `node_status_map` (lms:4645-4654). Labels name the *next* step: `synthesize_answer` → "Checking quotations...", `check_quotes` → "Verifying claims...".

Everything the check needs is in state at that point: `draft_answer` (set at lms:3468), `bibliography_results`, `graph_results`, `user_query`, and `build_verification_sources` (the same function the verifier uses at lms:3697, so the evidence is identical). The 35B therefore only ever sees the cleaned, smoothed text.

- **Skip conditions.** Copy the verify node's guards (lms:3681-3694): return immediately on `error_type in ("NO_RELEVANT_SOURCES", "NO_ANSWER_GENERATED")` or an empty draft.
- **Rollback flag.** In `QUOTE_FAILURE_MODE=repair` the node is a pass-through, and the verify node keeps today's hard gates (lms:3709-3735 and 3840-3859).
- **In delete mode the verify node must not re-run the old matcher.** If it did, title- and question-exempt spans would become hard failures again. Instead it reads the node's results from state:
  - verified quotes and marks are recorded as `SUPPORTED` with `supported_evidence_units` (same shape as lms:3849-3856), so the "N verified" list is unchanged;
  - deleted ones are recorded as `NOT_SUPPORTED` with reasoning "removed from the answer: not found in the sources" if you want them visible in the claims panel. They are **not** added to `hard_failures` and **not** added to `soft_flags`. Unanchored soft flags render in the red "Additional unverified claims" box (ChatUI:1373-1388), which would be wrong for removed text.
- **Repair path (contradicted verdicts only).** Route `repair_answer → check_quotes → verify_claims` in place of lms:1756, so a repaired paragraph gets the same check.
- **Repair no-op guard** (applies in both modes). `_repair_answer_node` (lms:3502-3665) unconditionally clears `error`/`error_type` at lms:3657-3658, and also at 3568-3571 when every rejection was unlocatable.
  - Before clearing, set `state["repair_noop"] = True` when `state["draft_answer"]` equals the pre-repair `draft` (lms:3519), or when `claims_by_paragraph` is empty. In the no-op case, keep `error_type = "FABRICATED_CLAIMS"` and `excluded_claims`.
  - Replace the plain edge with a conditional edge: `repair_noop` → `finalize_response`, where the FABRICATED_CLAIMS branch at lms:4224-4267 applies today's exhaustion semantics; otherwise → `check_quotes`.
  - Unlocatable rejections are already moved into soft flags at lms:3547-3565.
- **Effect.** After this change, `FABRICATED_CLAIMS` and the repair loop fire only on a model `contradicted` verdict (lms:3942-3944), which the audit counted at about 1 in 1,900 claims. `MAX_VERIFICATION_REPAIR_ATTEMPTS = 3` (lms:232) and `should_retry_verification` (lms:469-479, used at lms:1738-1744) stay as they are.

**Alternative:** run the same logic at the top of `_verify_claims_node` before the prompt is built (lms:3696-3735). This also covers re-verify. The cost is that the smoothing call gets attributed to the `verify_claims` stage in metrics. A separate node keeps the timing visible through `_timed_node` (lms:1644-1672).

**Smoothing call (single, guarded, paragraph-scoped):**

- **Wrapper.** `_call_llm(messages, model, temperature=0.7, response_format=None, recover_reasoning=False, max_tokens=8192) -> str` (lms:1818-1908).
  - It holds the `lm_studio_gateway.slot` (lms:1859), calls `resolve_model` on each attempt (lms:1863) and records metrics (lms:1872).
  - It sends `reasoning_effort` **only** when `self.reasoning_effort` (env `SYNTHESIS_REASONING_EFFORT`, lms:1537; prod = `low`) is set **and** the model is the synthesis or verification model (lms:1851-1852).
  - Add an optional `reasoning_effort: Optional[str] = None` parameter that overrides the global value.
- **Model: the 35B (`self.verification_model`, prod `qwen/qwen3.6-35b-a3b`) with `reasoning_effort="none"`.** Evidence from `evals/results/rag_audit_20261007/P2_replay_bench.md` §1 E2/E3 and `p2_raw/E2b*.json`, `E2d*.json`, `E3*.json`:
  - The API accepts only `none|minimal|low|medium|high|xhigh`. `"off"` returns HTTP 400 (`p2_raw/E2b_35b_synth_off_HTTP400.json`).
  - **On the 35B, every value except `none` falls back to thinking ON.** `low` gives 3,159-4,526 reasoning tokens (E2a/E2c).
  - On the 27B, `none` turns thinking off and `minimal` maps to `low`.
  - Uncontended speed:
    - 35B with `none`: about 98-100 tok/s decode, about 1,400 tok/s prefill, 8-10 s for 800-940 answer tokens;
    - 27B with `none`: about 27 tok/s decode, about 250 tok/s prefill.
    - A 150-400-token paragraph rewrite is therefore about 2-5 s on the 35B, against about 10-20 s on the 27B. Under VLM contention the 35B decoded at about 20 tok/s, giving about 10-20 s.
  - The 35B is pinned, so there is no JIT load.
  - The 35B with `none` fabricated quotes when asked to **synthesise** (E2d). Smoothing gives it no sources, and the deterministic guard rejects any new quote, number or name, so that failure mode is contained.
  - **Never** use `qwen/qwen3-4b-2507` with `response_format`, because it segfaulted the MLX engine (P2 §0).
- **Shape.** Make one call for all affected paragraphs, at most 3, and reuse the existing `_REPAIR_RESPONSE_FORMAT` json_schema (lms:3477-3499; `{revisions:[{paragraph_index, revised_text}]}`) with `extract_json_object(raw, anchor_key="revisions")` (lms:1031-1063), `temperature=0.0`, `recover_reasoning=True`, `max_tokens≈1024`. The 35B handles json_schema well: P2 E4b shows the JSON lands in `content` with `none`. Plain text per paragraph is an equally simple alternative.
  - Prompt contents: the paragraph after deletion, with a `⟦gap⟧` marker where text was removed.
  - Instruction: "Fix only the transition at ⟦gap⟧ and any now-dangling reference; keep every other word; add no facts, names, numbers, quotations, citations or shelf marks; remove the marker".
- **Deterministic guard.** Compare each revised paragraph with its pre-smoothing text; on any violation keep the unsmoothed paragraph. Violations:
  1. new digit runs: `set(re.findall(r"\d+", post)) - set(...pre)`;
  2. any quotation span of 1+ words in post that is not in pre, or any pre quotation missing verbatim from post (protected quotes, as in the repair node at lms:3525-3529 and 3641-3651);
  3. new bold or italic spans (`**…**`, `*…*`), brackets or parentheses (the citation markers), `doc:` links or `⟦…⟧` markers;
  4. new shelf marks, compared by canonical `detect_shelfmarks`;
  5. every word in post that is not in pre (normalised) must be in a small EN+HE connective allowlist ("however", "also", "moreover", "this", "these", "in", "addition", "furthermore", "similarly", "בנוסף", "כמו", "כן", "עם", "זאת", "לעומת", "גם", …). This rule also blocks invented names in Hebrew, which has no capitalisation;
  6. `len(post) <= len(pre) + 80` and `len(post) >= 0.6 * len(pre)`;
  7. `finish_reason == "length"`, or any exception (`ModelUnavailableError`, `httpx.HTTPError`). This is one narrow try/except, justified because failing to smooth must never fail the answer.
- **Closing note.** Add it in `_finalize_response_node` after flag annotation and before linkify (lms:4271-4284), not in the draft. Otherwise the 35B would judge the note as a claim and could flag it.
  - The note is one sentence, in italics, in the answer's language (`hebrew_char_ratio(prose) > 0.5`, imported from `genizah_terminology`). English: "_Some quoted statements were removed because they could not be found in the sources._" Hebrew equivalent: "_חלק מהציטוטים הוסרו משום שלא נמצאו במקורות._"
  - Add it only when `sentences_deleted > 0`.
- **Empty-result edge.** If deletion leaves no prose, or less than about 200 chars (the same threshold as the exhaustion branch at lms:4250), skip the verifier. Finalize with `_build_verified_evidence_fallback` (lms:4127-4195), or `_GRACEFUL_FALLBACK` (lms:4120-4124) plus the normal appendices. This mirrors the exhaustion branch at lms:4250-4267.

### A.5 Metrics and state

- **New state keys.** Add `AgenticRAGState` keys (lms:397-442), for example `quote_check: Dict[str, Any]` and `repair_noop: bool`, and initialise them in **both** initial states: `chat()` at lms:4531-4562 and `chat_stream()` at lms:4609-4640. `chat_stream` accumulates node updates (lms:4659, 4710) and builds the response from `accumulated` (lms:4739-4759).
- **Counters.** Put them in `_build_metrics` (lms:1674-1712, return dict at 1701-1712), next to `verification_cycles`/`repair_attempts` (lms:1708-1709):
  - `quote_check: {mode, quotations, verified, exempt_title, exempt_question, terms_skipped, unverified, shelf_marks_unverified, sentences_deleted, smoothing_applied, smoothing_rejected_reason}`;
  - `noop_repairs`.
  - The smoothing call shows up automatically under `llm_by_stage["check_quotes"]`. `metrics` is free-form (`AgenticRAGResponse.metrics`, lms:386-391), and the frontend does not read `verification_summary` keys (only `verified_claims[].verification_status`, ChatUI:1404), so adding keys is safe.
- **Logging.** Also log one `logger.info("Quote check: …")` line per turn, like the `Verification PASSED` line at lms:4014-4016, so log-based analysis works. Add a `processing_steps` entry as well.
- **Also from N0.** Add `finish_reason` to `record_llm_call` (lms:50-73). It is in `result["choices"][0]["finish_reason"]`.

### A.6 Prototype run (read-only, no network)

`python3 -I handoff/proto_quote_check.py …/rag_audit_20261007/p2_raw` rebuilds sources from the full replay verifier prompts `p18_verify_request.json` and `p07_verify_request.json`, and checks every replay draft.

| draft | old extractor spans | new quotations | unverified after the rule |
|---|---|---|---|
| prod P18 (27B) | 13 | 9 | 0 |
| E1a (27B low) | 15 | 13 | 0 (was `"Old List,"`/`"New List."` plus OCR `where- abouts`) |
| E1b | 12 | 12 | 0 |
| E3a (27B none) | 5 | 3 | 0 (was `"India Book."`) |
| E2a (35B on) | 2 | 0 | 0 |
| E2b (35B none) | 6 | 2 | 0. The spliced `…seminal study … ... did more to illuminate …` passes, because each ellipsis segment is verbatim and in order. That is the confirmed ellipsis rule; P2 had counted it as a real failure. |
| prod P07 (27B, Hebrew) | 14 | 8 | **1**: `"משובשת בחשבונותיה בספר ההוא והגהתי"` vs the source `והגהותי` (plene/defective spelling of one word). Under the strict rule this sentence is deleted. |
| E1c P07 | 12 | 8 | 0 |
| E2c P07 | 6 | 3 | 0 (old mis-pairs gone) |
| E2d P07 (35B none) | 11 | 7 | **1**: the invented `"אין לסלוח על כך… יתלצו ויתלעסו הרוסים"`. Correctly caught. |

Unit-style checks that passed:

| case | result |
|---|---|
| `ע"ו ו-כ"י ור"ד` | no quotes |
| `"הכפישה" … ("ויגרס בחצץ שני") … "the other one here"` | 3 correctly paired spans |
| Kol-Nidré title | extracted, then exempt as a title |
| curly quotes and guillemets | paired |
| `**Levin**, p. 68: "קרא רובי…"` | extracted |
| nested `'Old List'` | kept inside the outer quote |
| a changed date | fails |
| pointed vs unpointed source | verifies |
| `Yijū` vs `Yiju` | verifies |
| `India Book,` vs `'India Book.'` | verifies |

**Residual decision worth showing Isaac (not an assumption about the design):** Hebrew matres-lectionis variants (`והגהתי`/`והגהותי`) are the only false deletion found in this replay. Options: accept them as strict, or add an opt-in fold that drops ו/י inside words on both sides. The fold weakens discrimination only marginally for quotes of 3+ words.

Cases still to replay at implementation time (drafts are in the LM Studio logs):

| case | expected under the rule | where |
|---|---|---|
| 09-28 ketubbot `"…between 889 and 988,"` | deleted (genuine) | `2026-09/2026-09-28.1.log` around line 127152 (`rv_n2a/l52.txt`) |
| 09-23 Sukkot gloss inside a quotation | deleted (genuine) | `rv_n2a/l46.txt` (2026-09-23.1.log:73666) |
| `"formulary of laymen"` near-quote | deleted (genuine) | `rv_n2a/l55.txt` (2026-10-06.1.log:4537) |
| s49 Hoshana `"קרא רובי כרתי סילקא תמרי"` (Levin p.68) | depends on the page text | `rv_n2a/l49.txt` (2026-09-28.1.log:8010-10230) |
| Goitein `"India Book."` turn | term, so cleared | 2026-10-07.1.log:~15317-19160 (`rv_n2a/d1007.txt`) |

---

## B. Hard-failure routing today (for reference and rollback mode)

| Item | Where |
|---|---|
| `MAX_VERIFICATION_REPAIR_ATTEMPTS = 3` | lms:232 |
| `should_retry_verification(failure_count)`, `0 < n <= 3` | lms:469-479 |
| Graph: nodes and edges | lms:1715-1759. Nodes at 1719-1729. Edges 1731-1736: `synthesize_answer→verify_claims` at **1736**. `_route_after_verify` at 1738-1744 ("retry" when `FABRICATED_CLAIMS` and retries remain, else "abort"). Conditional edges at 1746-1754 (retry→`repair_answer`; continue/abort→`finalize_response`). **`repair_answer→verify_claims` at 1756.** `finalize→END` at 1757. |
| Verify node | lms:3668-4019 |
| Deterministic quote and shelf claims built | lms:3709-3735 (quotes 3710-3722, shelf marks 3723-3735) |
| Verifier call | lms:3824 |
| Hard gate loop | lms:3840-3859 (unmatched quote or mark → `NOT_SUPPORTED` plus `hard_failures`) |
| Model verdicts | lms:3861-3951 |
| Verdict `supported` with no valid source | evidence_limitation is accepted (3872-3885), otherwise downgraded to unsupported (3886-3887) |
| Citation-corroboration downgrade to a **soft flag** | lms:3889-3931 |
| `contradicted` | **hard failure**, lms:3942-3944 |
| `unsupported` | **soft flag**, lms:3945-3951. This must stay unchanged. |
| `parse_error` | `verification_error` soft flag, lms:3953-3958 |
| Summary | lms:3971-3982 |
| Hard-failure branch | lms:3984-4005 (sets `excluded_claims`, `retry_count`, feedback history, `error_type="FABRICATED_CLAIMS"`) |
| Pass branch | lms:4006-4017 |
| Repair node `_repair_answer_node` | lms:3502-3665. Builds sources at 3521. Protected quotes at 3525-3529. Anchoring by `locate_claim_sentence` at 3537-3545. **Unlocatable → soft flags at 3547-3565.** Early return when nothing is locatable at 3567-3571. 27B call with `_REPAIR_RESPONSE_FORMAT` at 3615-3628 (`except Exception` → deterministic removal). Splice-back and protected-quote fallback at 3630-3652. Clears the error at 3657-3658. |
| Finalizer | lms:4197-4348. NO_ANSWER at 4203-4206. NO_RELEVANT_SOURCES at 4208-4222. **FABRICATED_CLAIMS exhaustion at 4224-4267** (`remove_sentences_containing` at 4233-4235; unlocatable → soft flags at 4238-4250; keep if ≥ 200 chars at 4250-4258, else extractive fallback at 4259-4267). Shelf-mark resolution at 4269. **Flag anchoring at 4271-4282** (`annotate_answer_with_flags`, lms:1076-1118, wraps one sentence per flag with `⟦flag:N⟧…⟦/flag⟧`, markers at lms:955-957). Linkify at 4284. Appendices at 4286-4342. |
| Verified-evidence fallback | lms:4127-4195 (uses `extract_direct_quotes`/`find_quote_source` at 4143-4153; switch it to the new functions) |
| UI flag rendering | `FLAG_MARKER_REGEX` (ChatUI:87-88). `FlaggedSpan` (ChatUI:161). The `verification_error` banner at ChatUI:1362-1371. Unanchored flags at ChatUI:1373-1388. |

---

## C. N3: never lose finished work (assumption to confirm)

Isaac did not answer this question. The default below follows the audit's recommendation.

| Item | Where | Today / change |
|---|---|---|
| `ModelUnavailableError` | lms:146-152; message `LOCAL_MODEL_CAPACITY_MESSAGE` lms:155-164; `_is_model_unavailable_error` lms:167-190 | |
| `_call_llm` failure modes | lms:1856-1888 | Raised on timeout (1867-1869), on a 5xx or unloaded model after one re-resolve (1874-1885, raise at 1884-1885), and on resolve failure. Other 4xx raise `httpx.HTTPStatusError` (1886). `httpx.ConnectError` and other transport errors propagate **uncaught**. |
| `_call_verifier` | lms:4062-4099 | Two attempts **only on a parse error** (4074-4098). `max_tokens` is the default 8192. There is no `finish_reason` check, so on `length` it repeats the identical temperature-0 call. |
| Where the verifier is called | lms:3824 (`model_claims, model_summary, parse_error = await self._call_verifier(verifier_prompt)`) | Not wrapped. Any exception aborts the graph. |
| `resolve_model` | lms:1581-1623 | Exact id → `:N` instance → **in-request JIT `load_model`** (1614-1623, raises `ModelUnavailableError` on failure). |
| `load_model` | lms:1910-1955 | Sends a 1-token "ping" with `ttl`, timeout 300 s. **The gateway slot is already held** by the caller. |
| `ttl` in payloads | lms:1779-1780 (tools), **1853-1854** (`_call_llm`), 1940-1941 (`load_model`) | All gated by `self.model_ttl_seconds > 0` (env `LM_STUDIO_MODEL_TTL`, default 3600, lms:1540; `.env.example:26` = 3600). **Setting `LM_STUDIO_MODEL_TTL=0` on prod drops ttl with no code change.** |
| `chat_stream` failure path | lms:4673-4680 (`_produce` puts the exception on the queue); **4706-4707** `if isinstance(event, Exception): raise event` | The exception propagates out of the generator into app.py, and `accumulated` (draft plus sources) is discarded. |
| app `/chat-stream` | app:1463-1502; `except ModelUnavailableError` at **1484-1495** sends a `final` with `answer=str(e)`, `success False`, `MODEL_UNAVAILABLE` (no sources); generic `except` at 1496-1500 sends `{"type":"error"}` | |
| app `/chat` | app:1406-1460 (`ModelUnavailableError` → same capacity answer, 1444-1452; other errors → HTTP 500) | `/internal/eval/chat` is at app:1362-1403. |
| Frontend `/chat` fallback | ChatUI `handleSend` (917): `catch (streamErr)` at **1028-1062** re-POSTs `/chat`. `handleExampleClick` (1103): duplicate at **1209-1243**. Both are reached on any stream error, including the backend's `{"type":"error"}` event, which is re-thrown at 1019-1024 / 1200-1205. `streamStarted` is set (967/973) but never used. A stream that ends without `final` (`if (done) break;` at 971 / 1154) silently produces no message. | **Frontend change** = frontend container rebuild (no Node on the host; `docker compose build frontend` on the MBP, with Isaac's OK; `docs/SHARED_RUNTIME.md` §5). |

**Proposed N3 shape (smallest):**

1. Wrap the verifier call at lms:3824 (and, in effect, every re-verify, since it is the same node) in `except (ModelUnavailableError, httpx.HTTPError)`.
   - On failure, set `model_claims = []`, keep the deterministic and quote results, and add a `verification_error` soft flag with the reason "the verification model was unavailable".
   - The existing ChatUI:1362 banner then shows. The node returns normally, so `chat()` and `chat_stream()` both keep the draft and sources, and app.py is reached only for failures before the draft exists.
   - The existing `test_verifier_invalid_json_retries_then_flags_visibly` (tests:747) is the template.
2. Backstop in `chat_stream`. At lms:4706-4707, when `accumulated.get("draft_answer")` exists and the exception is not a pre-draft failure, add the `verification_error` flag, run `_finalize_response_node(accumulated)` (Neo4j and ES only, no LM Studio), and yield a normal `final` with `success=False, error_type="VERIFICATION_INCOMPLETE"`.
3. `LM_STUDIO_ALLOW_JIT_LOAD`, default `1` (tests at tests:1032 expect JIT); prod sets it to `0`. In `resolve_model`, when the flag is off, raise `ModelUnavailableError` at lms:1614 instead of calling `load_model`.
4. No identical retry after `length`.
   - Have `_call_llm` expose `finish_reason`. Recommended: move the body into a new `_chat_completion(...) -> (content, finish_reason)` and keep `_call_llm` as a wrapper with an unchanged signature. About 30 existing tests monkeypatch `_call_llm` with `AsyncMock(return_value=str)` (`test_graph_rag_phase1.py`, `test_multiturn_rag.py`, `test_judge_v2_and_language.py`), so changing its return type would break them.
   - `_call_verifier` uses `_chat_completion` and breaks the loop on `length`, which becomes a parse error and therefore a `verification_error` flag. Update the verifier tests (tests:493, 546, 603, 665, 708, 747) to patch the new method.
5. Remove the `/chat` fallback (ChatUI:1028-1062, 1209-1243), or allow it only when `!streamStarted`. Treat end-of-stream without `final` as an error, with a Retry button.

---

## D. Tests

- **Config.** `pytest.ini` (repo root on prod-mbp) sets `asyncio_mode = auto` and `testpaths = src/backend`.
- **Do not run bare `pytest`.** It collects `src/backend/tests/test_agentic_rag.py`, which builds a real `AgenticRAGService()` with `LLM_STUDIO_URL=http://127.0.0.1:1234` (test file line 25) and calls `service.chat(...)` (e.g. line 216). On the Studio that is **prod LM Studio inference plus ES**. `test_fetch_*` and `test_primary_source_fetching.py` likely hit live ES as well (not checked).
- **Run only the relevant files.** From the temp clone root, use the existing venv (pytest 9.0.1, langgraph and pytest-asyncio present in `/Users/isaac/Documents/GitHub/genizah_search/.venv`):

```
PYTHONPATH=. /Users/isaac/Documents/GitHub/genizah_search/.venv/bin/python -m pytest -q \
  src/backend/tests/test_graph_rag_phase1.py \
  src/backend/tests/test_pipeline_metrics.py \
  src/backend/tests/test_judge_v2_and_language.py \
  src/backend/tests/test_shelfmark_linkification.py \
  src/backend/tests/test_multiturn_rag.py
```

  Add the new test module, for example `src/backend/tests/test_quote_check.py`.
- **Relevant existing tests** (all in `src/backend/tests/test_graph_rag_phase1.py` unless stated). Every fixture is an inline `state` dict plus a `monkeypatch.setattr(service, "_call_llm", AsyncMock(...))`; there is no conftest.

| test | line |
|---|---|
| `test_synthesis_and_verification_share_identical_bibliography_evidence` | 452 |
| `test_only_main_text_is_quoteable_bibliography_evidence` | 470 |
| `test_verifier_checks_quotes_deterministically_and_claim_propositions` | 493 |
| `test_hallucinated_citation_is_not_trusted` | 546 |
| `test_accurate_limitation_statements_verify_as_supported` | 603 |
| `test_contradicted_claim_triggers_targeted_repair` | 665 |
| `test_prior_rejections_are_advisory_not_a_blacklist` | 708 |
| `test_verifier_invalid_json_retries_then_flags_visibly` | 747 (N3 template) |
| `test_retry_prompt_lists_unsupported_attributions` | 774 |
| `test_span_repair_rewrites_only_the_offending_paragraph` | 805 |
| `test_span_repair_falls_back_to_deterministic_removal` | 865 |
| `test_locate_claim_sentence_matches_paraphrased_claims` | 900 |
| `test_annotate_answer_with_flags_wraps_sentences_with_markers` | 915 |
| model resolution tests | 996, 1012, 1032 |
| `test_model_unavailable_error_is_recognised` | 1050 |
| `test_sentence_split_keeps_citations_and_abbreviations_intact` | 1064 |
| `test_flag_wrapping_covers_whole_citation_sentence` | 1084 |
| `test_hebrew_quotes_and_pointed_text_verify` | 1174 (uses `extract_direct_quotes`/`find_quote_source` directly; keep it passing, or port it to the new API) |
| `test_remove_sentences_containing_deletes_only_offending_sentences` | 1470 |
| `test_retry_exhaustion_retains_only_verified_short_quotes` | 1482 |
| `test_retry_policy_allows_three_targeted_repairs` | 1511 |
| `test_terminal_fallback_groups_verified_facts_and_quotes_by_source` | 1519 |
| `test_verified_quotes_are_limited_to_thirty_words` | 1555 |
| `test_pipeline_metrics.py`: `test_build_metrics_reports_cycles_and_per_stage_usage` | 41 |
| `test_pipeline_metrics.py`: `test_chat_response_carries_metrics` | 77 (patches `service.graph.ainvoke`) |
| `test_judge_v2_and_language.py`: `test_verifier_refuses_to_verify_an_empty_draft` | 306 |
| `test_judge_v2_and_language.py`: `test_finalize_returns_honest_failure_without_appendices` | 327 |

  **New unit cases:** the A.6 table rows, plus P06 (shelf mark echoed from the question), P09 (graph work title), delete-by-offset (the quote's sentence goes and the next sentence survives), the smoothing guard (each of the 7 rejections), the empty-after-deletion fallback, the no-op repair edge, and verifier `ModelUnavailableError` → draft kept with a `verification_error` flag.

- **Replay material.** All of it is on the Studio, read-only.
  - LM Studio logs: `~/.lmstudio/server-logs/YYYY-MM/YYYY-MM-DD.N.log`. Request bodies are truncated in the logs (`<Truncated in logs>`, P2 §0), but predictions (drafts, repairs) are logged in full.
  - Full sources for the 18 probes: `evals/results/rag_audit_20261007/p1/finals/P01..P18.json`, each with `bibliography_results` (including `full_text`) and `graph_results`. Rebuild prompts with `build_verification_sources`, as `p2_build_prompts.py` does.
  - Full replay prompts: `p2_raw/p18_verify_request.json` and `p07_verify_request.json`. Replay drafts: `p2_raw/E*.json` (`content`).
  - Repair diffs: `python3 rv_n2a/repair_triggers.py <log> <start> <end>` (pre-extracted slices: `rv_n2a/d1007.txt`, `l46`, `l49`, `l52`, `l55`, `pre_a`, `pre_b`).
  - Turn classification: `rv_n2a/n2a_bundles.py` (`CLASSIFICATION`, lines 22-44).
  - P17's 35B "invented support" for 889/988: `2026-10/2026-10-07.1.log:65379`.
- **Opus non-regression run.** `scripts/run_agentic_rag_eval.py` on prod-mbp has these flags:

| flag | default / meaning |
|---|---|
| `--dataset` | `evals/agentic_rag_v1.json` |
| `--judge-prompt` | judge prompt file |
| `--api-base-url` | `http://localhost:8000` |
| `--judge-base-url`, `--judge-model` | judge endpoint and model |
| `--judge-provider {lmstudio,anthropic}` | `anthropic` reads `ANTHROPIC_API_KEY` |
| `--synthesis-model` | uses `/internal/eval/chat` and needs `EVAL_API_KEY` |
| `--case` | repeatable |
| `--output`, `--no-judge`, `--timeout` | default timeout 300 |
| `--wait-for-idle`, `--idle-quiet-seconds`, `--idle-max-wait-seconds` | idle gate |

  **`--stream` does not exist on prod-mbp.** It exists only on the unmerged branch `origin/yk-holiday` (a3aaed7, 2026-09-22). The prod-mbp runner posts to non-streaming `/chat`, which is fine against the local dev backend (no Cloudflare edge) but not against `api.cairogenizah.ai`. Command, run on the MBP with Isaac's OK, quiet hours, one run at a time:

```
PYTHONPATH=. .venv/bin/python scripts/run_agentic_rag_eval.py \
  --api-base-url http://127.0.0.1:8010 --judge-provider anthropic --judge-model <opus id> \
  --wait-for-idle --output evals/results/<run>.json
```

  The dev backend for this is `scripts/dev_backend_local.py` (`.claude/launch.json` → `backend-local`), with `LLM_STUDIO_URL=http://192.168.8.122:1234`. Either cherry-pick `--stream` from yk-holiday or keep the eval on :8010.

## E. Env flags and where defaults live

- **Module constants and `__init__` env reads.** Defaults live in code:
  - `EVIDENCE_CHAR_BUDGET` (lms:236), `SYNTHESIS_MAX_TOKENS` (240), `SYNTHESIS_RETRY_MAX_TOKENS` (244), `MAX_VERIFICATION_REPAIR_ATTEMPTS` (232, a constant, not an env var);
  - `AgenticRAGService.__init__` (lms:1529-1548): `LLM_STUDIO_URL`, `ROUTER_MODEL`, `SYNTHESIS_MODEL`, `VERIFICATION_MODEL`, `SYNTHESIS_REASONING_EFFORT`, `LM_STUDIO_MODEL_TTL`, `LM_STUDIO_REQUEST_TIMEOUT`;
  - `LM_STUDIO_MAX_CONCURRENCY` (lms:88);
  - `CHAT_STREAM_HEARTBEAT_SECONDS` (lms:4685).
  - New flags go next to these: `QUOTE_FAILURE_MODE` (default `delete`), `QUOTE_SMOOTHING_MODEL` (default = `VERIFICATION_MODEL`), `QUOTE_SMOOTHING_REASONING_EFFORT` (default `none`), and `LM_STUDIO_ALLOW_JIT_LOAD` (default `1`).
- **Prod values.** These are **not in git**. The MBP's `src/backend/.env` is loaded via compose `env_file` (`docker-compose.yml` backend service lines 38-47, `required: false`), plus the root `.env`. `docker-compose.yml` sets no model or reasoning vars. `docker-compose.mbp.yml` overrides only `LLM_STUDIO_URL=${STUDIO_LM_STUDIO_URL:-http://192.168.8.120:1234}` and `OLLAMA_URL`.
  - Observed prod values (C1 table, `C1_pipeline_callgraph.md:229-232`): `SYNTHESIS_MODEL=qwen/qwen3.8-27b`, `VERIFICATION_MODEL=qwen/qwen3.6-35b-a3b`, `SYNTHESIS_REASONING_EFFORT=low`, `LM_STUDIO_MODEL_TTL=3600`.
  - Note the mbp default IP `.120` vs the `.122` in memory and the audit. Whatever is in the MBP's `.env` wins.
  - **Changing an env value = a container recreate on the MBP** (compose `env_file` is read at create). That is a prod action and needs Isaac's OK. Code changes need a backend image rebuild, which is a prod deploy.

## F. Things the implementer must not trip on

- The `_normalize_verification_text` change radius: 17 call sites in lms (dedup, anchoring, protected quotes, claim terms). Add a new `normalize_for_quote_match` instead of editing it.
- Both initial-state dicts (lms:4531, 4609) must get any new state key, or `chat_stream`'s `accumulated` will lack it.
- `_repair_answer_node` clears `error_type` before returning (lms:3657-3658, 3569-3570). The no-op guard must decide before that line.
- The verifier sees the draft only through `verifier_prompt` (lms:3768-3822). If check_quotes runs first, nothing else is needed for "the 35B checks the final text".
- `json_schema` must never go to `qwen/qwen3-4b-2507`. Smoothing on the 35B with `_REPAIR_RESPONSE_FORMAT` is fine.
- Context: smoothing is a small prompt (affected paragraphs only), with no evidence, so it does not grow the synthesis or verifier context.
