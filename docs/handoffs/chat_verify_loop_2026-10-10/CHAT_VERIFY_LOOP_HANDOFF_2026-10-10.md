# Handoff: chat quote check, no-op repair guard, and loss-proofing (genizah_search)

Written 2026-10-10 for a fresh Claude Code session. You need nothing from the session that wrote it.
Target: **one PR into `prod-mbp`** in `AIStream-Peelout/genizah_search`, **opened, not merged**. `prod-mbp` is the production build branch: the next backend rebuild ships whatever is merged there. You open the PR (C1-C6), plus a separate PR for C7 if Isaac wants it; **Isaac reviews and merges**. Never merge, push to `prod-mbp` directly, or enable auto-merge.

---

## 0. Paste this to start

> Read `CHAT_VERIFY_LOOP_HANDOFF_2026-10-10.md` (this file) end to end, then `CLAUDE.md` and `docs/SHARED_RUNTIME.md` in the repo. Production runs on the M3 MBP (Docker + cloudflared); LM Studio on this Mac Studio serves production inference too. `docs/SHARED_RUNTIME.md` ("this Mac is production") predates the 2026-09-15 migration and describes the MBP.
>
> Implement items C1-C6 of the handoff and **open** one PR into `prod-mbp`; do not merge it (Isaac merges). C7 is an optional, separate frontend PR.
> - Build in a **temporary clone** off `origin/prod-mbp` on branch `fix/chat-verify-loop`. Do not use git worktrees, and do not touch `/Users/isaac/Documents/GitHub/genizah_search`: it holds Isaac's uncommitted work.
> - Run only the named unit-test files (§6). Never run bare `pytest`: it calls prod LM Studio.
> - Do not start the dev backend, run evals, rebuild images, or run any docker compose command without asking Isaac.
> - Before coding, post a short plan and list the open items in §9 for Isaac to answer.

---

## 1. Goal and why

An audit on 2026-10-07 found the following (`docs/RAG_AUDIT_2026-10.md`; data in `evals/results/rag_audit_20261007/` in Isaac's main checkout, untracked, read-only for you):

- Chat takes a median of **222 s** and a p90 of **675 s**.
- **27 %** of chat seconds go to the quote-repair loop.
  - The deterministic quote matcher flags format-only mismatches, e.g. a draft with `"India Book."` against a source reading `India Book,`.
  - Each flag triggers a 27B repair and then a full 35B re-verify, up to 3 rounds, with no guard against a repair that changes nothing.
  - After 3 rounds, the finalizer deletes whatever still fails.
  - Its extractor mis-pairs quote marks, so prose becomes a "quote". In one turn this deleted **7 correct sentences** (C2, Tisha b'Av).
- 57 % of chat seconds is GPU contention with the sibling project. That is out of scope here (§10).
- A crash in the verifier after the draft exists throws away the draft **and** its sources. Seven answers were lost on 09-22/23.

This PR does the following:

1. Replaces the quote loop with **one simple deterministic check before the 35B verifier**. Quotes are verified by normalised substring. A sentence whose quote cannot be found is deleted. One guarded smoothing pass follows, then one closing note.
2. Adds a **no-op guard** to the remaining repair loop.
3. Makes a late verifier failure **keep the answer**, under the existing "couldn't be double-checked" banner.
4. Adds the **metrics** needed to measure all of the above.

---

## 2. Owner decisions (Isaac)

**Quote verification: CONFIRMED on 2026-10-10.** Isaac's words:

> "The quotes should be verified and if they can't be verified they should be deleted. The main goal of the verifier is to catch fabricated quotes. But if it is small things off we don't want it falsely flagging say due to punctuation. That shouldn't be that complicated."

He said earlier:

> "if a late check fails it should remove the questionable part and simply keep the verified parts, maybe with some filler so it isn't as jarring."

What follows from that:

- No LLM decides whether a quote is real.
- Quote failures get no 27B repair rounds.
- Matching ignores punctuation, case, whitespace and nikud. It is not fuzzy.
- A sentence with an unverifiable quote is deleted. A guarded smoothing pass then fixes the transition ("filler"), and one closing note is added.
- Keep the code small and readable.

**Claims the 35B judges `unsupported`: unchanged.** They stay soft-flagged and highlighted, and are never deleted. A `contradicted` verdict still goes through the existing repair path.

**ASSUMPTION TO CONFIRM (the only one): N3 crash handling.** Isaac has not answered this. The default follows the audit:

- keep the draft and its sources, and show the existing `verification_error` banner;
- no JIT model load inside a request;
- no identical retry after `finish_reason=length`;
- no silent `/chat` fallback in the frontend.

List it in the PR description as needing Isaac's confirmation. You never merge; Isaac decides whether C5 ships as is.

---

## 3. Ground rules (non-negotiable)

- **Coding standards** (`CLAUDE.md`):
  - every function gets a Sphinx-style docstring (`:param x:`, `:returns:`, `:rtype:`) and type hints;
  - keep the code modular and testable;
  - use try/except only where justified; this brief names each one.
- **No git worktrees.** Clone into the session scratchpad:
  ```
  git clone https://github.com/AIStream-Peelout/genizah_search.git <scratch>/gs_verify
  cd <scratch>/gs_verify && git checkout -b fix/chat-verify-loop origin/prod-mbp
  git rev-parse origin/prod-mbp   # expect e76576869013ee41775223a7415d4714c80003b7; if it differs, re-check every line ref below
  ```
  Never check out, stash or modify `/Users/isaac/Documents/GitHub/genizah_search`; it is on branch `eval-retrieval` with uncommitted work. The same applies to the sibling repo `historical-document-analysis`.
- **No production actions without Isaac's explicit OK.** This covers:
  - merging anything into `prod-mbp` (you only open PRs);
  - backend or frontend image rebuilds (each one is a prod deploy);
  - `docker compose up/down/restart/--force-recreate`;
  - edits to the MBP's `src/backend/.env`.

  Prod runs on the M3 MBP. LM Studio on the Mac Studio is prod too.
- **On the Studio, ask first** before any of these: dev containers, node builds, `scripts/dev_backend_local.py`, eval runs, or full pytest sweeps. Run only the unit-test files of the modules you change (§6).
- **LM Studio:**
  - never `lms load/unload/eject`;
  - never load `google/gemma-4-31b-qat`;
  - never eject `qwen3-vl-8b-heb-v18b-step700`;
  - **never send `response_format` json_schema to `qwen/qwen3-4b-2507`** (it segfaulted the MLX engine on 2026-10-07);
  - no traffic to LM Studio inference or the prod chat API from your session; read-only `~/.lmstudio/bin/lms ps --json` is fine.
- **Do not grow the synthesis or verifier context.** The context is 32k, with an evidence budget of about 15k chars. The smoothing prompt carries no evidence.
- PR description ends with: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Commits end with: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

---

## 4. Orientation

`lms` means `src/backend/lms_agentic_search.py` at `e765768`. All line refs below were verified on 2026-10-10.

The graph is built at lms:1715-1759:

```
resolve_query → route_query → execute_searches → link_primary_secondary → synthesize_answer
  → verify_claims (35B) ─retry→ repair_answer (27B) → verify_claims …   (lms:1736, 1738-1756)
                       └continue/abort→ finalize_response → END
```

The current quote path:

1. `extract_direct_quotes` (lms:887-908) uses the regex `"([^"\n]{8,})"|“([^”\n]{8,})”` (lms:894) and returns bodies with no offsets.
2. `find_quote_source` (lms:926-952) runs `_normalize_verification_text` (lms:865-884, which does **not** strip punctuation) and a substring check, plus a SequenceMatcher ≥0.93 fuzzy leg for quotes of 40+ chars (lms:943-951).
3. The verify node builds deterministic quote and shelf-mark claims (lms:3709-3735) and turns misses into **hard failures** (lms:3840-3859). Each hard failure leads to `FABRICATED_CLAIMS`, then a repair, then a re-verify, up to `MAX_VERIFICATION_REPAIR_ATTEMPTS=3` (lms:232, 469-479).
4. On exhaustion the finalizer runs `remove_sentences_containing` (lms:4224-4267), which locates sentences with fuzzy `locate_claim_sentence` (lms:998-1028, ≥0.55).

Reference material, all read-only:

- `proto_quote_check.py`, in the same folder as this handoff: a working prototype of the rule. Its core is reproduced in **Appendix A** in case the folder is gone.
- `survey_backend.md`, in the same folder: a fuller line-by-line survey.
- Under `/Users/isaac/Documents/GitHub/genizah_search/evals/results/rag_audit_20261007/`:
  - `p2_raw/p18_verify_request.json` and `p07_verify_request.json`: full verifier prompts, i.e. sources plus draft;
  - `p2_raw/E*.json`: replay drafts, in `content`;
  - `p1/finals/P01..P18.json`: full `bibliography_results` and `graph_results`;
  - `rv_n2a/{l46,l49,l52,l55,d1007}.txt`: repair-loop slices from the LM Studio logs;
  - `rv_n2a/repair_triggers.py`;
  - `C2_verify_loop.md` (its reading of l46 as an inserted gloss and of l55 as a near-quote is corrected in §7 #3, from on-disk sources).
- LM Studio logs: `~/.lmstudio/server-logs/2026-0{9,10}/*.log`. Predictions are logged in full; request bodies are truncated.

---

## 5. Changes

### C1. New pure module `src/backend/quote_check.py` (no I/O, no LLM)

Port Appendix A into this module and add the functions below.

#### Module layout (avoids an import cycle)

`lms_agentic_search` will import `quote_check`, so `quote_check` must **never** import `lms_agentic_search`. Its imports are limited to the standard library and `src.backend.shelfmark_normalizer` (which itself imports only `re` and `typing`). The helpers it needs that live in lms today move out of lms, and lms re-imports them under the same names:

| helper (current location) | moves to |
|---|---|
| `_SENTENCE_BOUNDARY_REGEX` (lms:959), `_ABBREVIATION_REGEX` (lms:964-968), `_ABBREVIATION_SENTINEL` (lms:969), `split_sentences` (lms:972-995) | `quote_check.py` |
| `shelfmarks_equivalent` (lms:1166-1191), `find_shelfmark_source` (lms:1194-1210) | `shelfmark_normalizer.py` (they only need `ShelfmarkNormalizer` and `detect_shelfmarks`, already there) |

In lms add `from src.backend.quote_check import split_sentences, ...` and `from src.backend.shelfmark_normalizer import shelfmarks_equivalent, find_shelfmark_source` (extend the existing import at lms:24). The re-export matters: `test_graph_rag_phase1.py:1070, 1081` call `agent_module.split_sentences`, and `test_shelfmark_linkification.py:235` does `from src.backend.lms_agentic_search import shelfmarks_equivalent`. No test monkeypatches these helpers, so moving them is safe. Lms call sites (`locate_claim_sentence` lms:1016, lms:3071, 3724, 4413) keep working through the imported names.

#### `normalize_for_quote_match(text: str) -> str`

Applies these steps in order:

1. NFKD.
2. Drop every combining mark (category `Mn`). This covers Hebrew nikud and cantillation, and Latin diacritics too, so `Yijū` = `Yiju`.
3. NFKC.
4. `casefold()`.
5. Delete editorial brackets `[]⟨⟩` without inserting a space (`במעשי[ו]` → `במעשיו`).
6. Replace every other char of category `P*` or `S*` with a space. That covers straight, curly and single quotes, geresh and gershayim, maqaf, dashes, guillemets and ellipses.
7. Fold Hebrew final letters (ךםןףץ → כמנפצ).
8. Collapse whitespace.

Matching then compares **with all spaces removed** (`.replace(" ", "")`). This absorbs OCR line-break hyphenation in the sources (`where- abouts`, `Ara- bic`, `for-\nmulary`), maqaf, and abbreviation marks, and a changed letter or digit still fails.

Do **not** modify `_normalize_verification_text`. It has 17 call sites (anchoring, dedup, protected quotes).

#### `extract_quotations(text: str, min_words: int = 3) -> List[Tuple[int, int, str]]`

Pairs quote marks by position and returns `(start, end, body)` offsets into `text`, where `start` is the opening mark and `end` is one past the closing mark.

- **Straight marks.** A `"` or `״` can open only after start, whitespace, or one of `( [ { — – - : / * _ « „ “`. It can close only before end, whitespace, or one of `. , ; : ! ? ) ] } — – - * _ … ' ’ »`.
- **Abbreviations.** A mark with a letter on both sides, followed by exactly one letter and then a non-letter, is an abbreviation and is skipped (`ע"ו`, `כ"י`, `רמב"ם`, `ור"ד`).
- **Hebrew prefix quotes.** A mark after 1-3 prefix letters `[ושהכלבמ]` at word start, followed by a multi-letter word, **opens** a quotation (`ש"נשתמרו …"`, `ל"…"`). These occur in real 27B and 35B drafts.
- **Curly and guillemets.** `“ ”` and `« »` pair directly.
- **Single quotes** are never delimiters. They cover apostrophes and nested `'Old List'`.
- **Newlines.** Pairing resets at `\n`.
- **Terms.** Spans of fewer than `min_words` (3) words are *terms*, not quotations. They are not returned, but their marks are consumed, so they cannot mis-pair with later marks. This alone retires `"India Book."`, `"Old List,"` and `"גאון הגאונים"`, and fixes the Tisha b'Av mis-pairing.
- **Known coverage limit (state it in the PR description).** Single-quoted quotations and quotations under 3 words stay unchecked, exactly as with today's extractor (which only matched double/curly quotes of 8+ chars). Reviewers should not assume full coverage.

#### `quote_found(body: str, haystack_nospace: str) -> bool`

- **Attempt 1 (plain).** Split the body on ellipses (`...`, `. . .`, `…`). Normalise each segment; segments of **fewer than 2 words are ignored** (a lone "the" or "and" between ellipses proves nothing). Each remaining segment, with spaces removed, must occur in the haystack **in order**. If no segment of 2+ words remains, attempt 1 fails (fail closed).
- **Attempt 2 (editorial interpolation), only if the body contains `[`.** Replace each draft-side `[...]` with an ellipsis gap and run the same segment rule. This accepts `"the [Geniza] documents of the eleventh century show"` against a source reading `the documents of the eleventh century show` (Isaac's "small things off"). Source-side brackets are still dropped with their content kept (normalisation step 5), and attempt 1 already covers a draft that copies the bracketed word from the source.
- The quote is verified if either attempt succeeds.
- No fuzzy tolerance. "between 889 and 988" against a source with "998", or with no date at all, fails.

#### `check_quotations(draft, sources, question) -> QuoteCheckResult`

Signature: `check_quotations(draft: str, sources: List[Dict[str, Any]], question: str) -> QuoteCheckResult`. Use a small dataclass or TypedDict for the result.

`sources` is the output of `build_verification_sources` (lms:839-862), the same list the verifier uses.

- **Quote haystacks.** Normalised, spaceless `source["quoteable_text"]` of each bibliography source (lms:834). Graph sources have no `quoteable_text` key and can never verify a quote.
- **Title haystack.** All `source["citation"]` lines (lms:807: `Author, *Title*, p. N`), plus the `evidence_text` of **graph sources only**, identified by the absence of the `"quoteable_text"` key (their citation starts `Neo4j knowledge graph record`, lms:854). Do **not** add the `evidence_text` of bibliography sources whose `quoteable_text` is empty: every bibliography source has the key (lms:834), and its `evidence_text` contains `Generated catalog summary (orientation only; never quote)` (lms:815). Under a looser rule a quote lifted from a generated summary would become `exempt_title` and ship unflagged, where today `find_quote_source` rejects it.
- **Question haystack.** `state["user_query"]` only. Do not use assistant turns: they would let an earlier unverified quote launder itself.
- All three haystacks are built the same way: `normalize_for_quote_match(...).replace(" ", "")`. The title and question tests compare the quote's normalised, space-stripped body by substring.

Classify each quotation, checking in this order:

| order | test | class |
|---|---|---|
| 1 | `quote_found` in any quote haystack | `verified` (record `source_number`) |
| 2 | substring of the title haystack | `exempt_title` |
| 3 | substring of the question | `exempt_question` |
| 4 | none of the above | `unverified` |

"Any source" matches today's behaviour; do not parse which source the sentence cites.

**Shelf marks.** Do **not** locate marks with `detect_shelfmarks(draft)` plus `draft.find(mark, cursor)`: `detect_shelfmarks` (`shelfmark_normalizer.py:266-279`) de-duplicates case-insensitively and rstrips `.,`, so `find` reaches only the first occurrence and a second sentence carrying the same unverifiable mark would survive. Instead iterate **every occurrence**:

```python
for match in SHELFMARK_PATTERN.finditer(draft):
    raw = match.group(1)
    mark = raw.strip().rstrip(".,")
    start = match.start(1) + (len(raw) - len(raw.lstrip()))
    end = start + len(mark)
```

(`SHELFMARK_PATTERN` is the module-level pattern at `shelfmark_normalizer.py:211`, used by `detect_shelfmarks` at :272.) Classify each occurrence (cache the verdict per lowercased mark):

- `verified` if `find_shelfmark_source(mark, sources)` matches;
- else `exempt_question` if `shelfmarks_equivalent(mark, q)` holds for some `q` in `detect_shelfmarks(question)`;
- else `unverified`; every occurrence becomes an unverified span.

The result carries:

- `unverified_spans: List[Tuple[int, int, str, str]]` (start, end, text, kind `quote|shelf_mark`);
- `verified: List[Dict]`, shaped like today's `deterministic_claims` items (lms:3711-3735: `type`, `text`, `supported=True`, `source_number`, `reasoning`), each also carrying its `(start, end)` span;
- counts for metrics (`quotations`, `verified`, `exempt_title`, `exempt_question`, `terms_skipped`, `unverified`, `shelf_marks_unverified`).

#### Sentence splitter fix (prerequisite)

`_SENTENCE_BOUNDARY_REGEX` (lms:959) has two gaps:

- it never splits after a closing quote: `…the "India Book." He began…` stays one sentence;
- it never splits before `**Author**`.

With immediate deletion, either gap would delete the following sentence too. Replace it with:

```
(?:(?<=[.!?])|(?<=[.!?][\"”’״)\]]))\s+(?=(?:\*\*|[*_])?[\"“״(\[]?[A-Z0-9א-ת])
```

`_ABBREVIATION_REGEX` (lms:964-968) also has a gap that matters once deletion is by offset: it lacks `b.` (ben), `d.` (died), `c.`/`ca.` (circa) and `fl.` (floruit). Names such as `Nahray b. Nissim`, `Khalaf b. Isaac` and `Joseph b. Abraham` are frequent in the audit drafts and finals, and `split_sentences` breaks them after `b.`. So for `Nahray b. Nissim wrote that "<unverifiable quote>".` offset deletion would remove only `Nissim wrote that "…".` and leave the remnant `Nahray b.`; `(d. 1985)` splits the same way. Add `b|d|c|ca|fl` to the alternation (e.g. `…|mss|r|v|b|d|c|ca|fl)\.\s`).

`split_sentences` (lms:972-995) also feeds flag anchoring, so `test_sentence_split_keeps_citations_and_abbreviations_intact` (tests:1064) and `test_flag_wrapping_covers_whole_citation_sentence` (tests:1084) must stay green. If anchoring regresses with either change, give deletion its own splitter (boundary regex plus the extended abbreviation list) in `quote_check.py` and leave the shared one alone. Unit cases (in `test_quote_check.py`, case 24): `Nahray b. Nissim wrote that "there was no court in Fustat at all." He travelled widely.` deletes the whole first sentence with no `Nahray b.` remnant; `Goitein (d. 1985) wrote that "<fabricated>". His work continues.` likewise.

#### `delete_sentences_at(draft, spans) -> Tuple[str, List[int]]`

Signature: `delete_sentences_at(draft: str, spans: List[Tuple[int, int]]) -> Tuple[str, List[int]]`. It returns the new draft and the indices of paragraphs (split on `\n\n`) that lost a sentence but still have text.

- Delete **by character offset**, never by fuzzy lookup. It uses the `split_sentences` that now lives in `quote_check.py` (or the deletion-only splitter, see above).
- Per line: take a leading list marker or indent (`^\s*(?:[-*+]|\d+\.)\s+`) aside, `split_sentences(rest)`, and locate each sentence with `rest.find(sentence, cursor)`. Remove every sentence overlapping any `[start, end)`, and rejoin the rest with single spaces, keeping the list marker.
- Drop lines left empty, and markdown headings left with no body before the next heading or the end.
- Collapse runs of blank lines.

### C2. New graph node `check_quotes` (before the 35B)

- **Registration.** Add `("check_quotes", self._check_quotes_node)` to the node list (lms:1719-1729), wrapped by `_timed_node` like the others.
- **Wiring.** Replace `add_edge("synthesize_answer", "verify_claims")` (lms:1736) with:
  - `synthesize_answer → check_quotes`;
  - a conditional edge from `check_quotes`: `"finalize"` when `(state.get("quote_check") or {}).get("answer_too_short")`, else `"verify"` → `verify_claims`. Use `.get`, never `["answer_too_short"]`: the repair-mode pass-through and the guard returns store no such key.
- **Status labels.** Add to `node_status_map` (lms:4645-4654): `synthesize_answer` → "Checking quotations...", `check_quotes` → "Verifying claims...", `repair_answer` → "Checking revised quotations..." (it now leads to `check_quotes`). Labels name the next step.

`_check_quotes_node(state)` does the following:

1. **Guards.** Mirror the verify node's guards (lms:3681-3694): the `error_type in ("NO_RELEVANT_SOURCES", "NO_ANSWER_GENERATED")` check at lms:3681, and the empty-draft branch at lms:3683-3694. But `check_quotes` only **passes through** in both cases: return the state unchanged and let `verify_claims` handle them. In particular do not copy the empty-draft branch, which sets `NO_ANSWER_GENERATED` and replaces the draft; `verify_claims` still does that.
2. **Repair mode.** If `self.quote_failure_mode == "repair"`, set `state["quote_check"] = {"mode": "repair"}` and return. This is a pass-through: today's behaviour.
3. **Check.** `sources = build_verification_sources(...)` (the same call as lms:3697), then `check_quotations(draft, sources, state["user_query"])`.
4. **Delete.** If anything is unverified, run `delete_sentences_at` on the unverified spans. Then filter `quote_check["verified"]` down to entries whose span lies in a surviving sentence (a verified quote can share a sentence with an unverified one). Otherwise the verify node records `SUPPORTED` units for text that is no longer in the answer, and `_build_verified_evidence_fallback` can resurrect it through `supported_evidence_units`. Simplest: re-run `check_quotations` on the post-deletion draft and keep its `verified` list (it is cheap and pure); keep the counts from the first run.
5. **Too short.** Only when `sentences_deleted > 0` **and** the remaining prose is under 200 chars, set `answer_too_short = True` and stop. A legitimately short draft with nothing deleted must go to the 35B as today. 200 is the same threshold as the exhaustion branch at lms:4250.
   - **Finalizer branch.** `error_type` is `None` here, so without a new branch the short draft would ship. In `_finalize_response_node`, before the `FABRICATED_CLAIMS` branch (lms:4224), add: if `(state.get("quote_check") or {}).get("answer_too_short")`, set `final_answer = self._build_verified_evidence_fallback(state, quote_failure_mode=self.quote_failure_mode, quotes_removed=True) or self._GRACEFUL_FALLBACK`, append a `processing_steps` entry, and `return state` (same shape as lms:4259-4267).
   - **Fallback signature.** `_build_verified_evidence_fallback` is a `@staticmethod` (lms:4126-4127) and cannot read `self`. Add a keyword parameter `quote_failure_mode: str = "repair"`. With `"repair"` it behaves exactly as today, so `test_retry_exhaustion_retains_only_verified_short_quotes` (tests:1482, calls it with `state` only) and `test_terminal_fallback_groups_verified_facts_and_quotes_by_source` (tests:1519) stay green unchanged. With `"delete"` it collects quotes via `extract_quotations` + `quote_found` (instead of `extract_direct_quotes`/`find_quote_source`, lms:4143-4153). Both callers in the finalizer (the existing exhaustion call at lms:4260 and the new branch) pass `self.quote_failure_mode`.
   - **Lead text.** The fallback's lead (lms:4190-4192) says "did not pass verification after three targeted repairs", which is false when quote deletion caused the fallback. Add a second keyword parameter `quotes_removed: bool = False`; the new `answer_too_short` branch passes `quotes_removed=True` and gets: "Several quoted statements in the drafted answer could not be found in the sources and were removed, leaving too little to present. The following source-linked account includes only claims and short quotations that were verified:". The exhaustion branch keeps today's text (in delete mode it is still reached by a model `contradicted` verdict after repairs, where "three targeted repairs" stays true).
6. **Smooth.** Otherwise, if `sentences_deleted > 0`, run one smoothing call over the affected paragraphs (C3).
7. **Record.** Store `draft_answer`, `state["quote_check"] = {"mode": "delete", "verified": [...claim dicts...], "unverified_spans": [...], "counts": {...}, "sentences_deleted": int, "smoothing": str, "answer_too_short": bool}` (keep the counts under `counts` so they do not collide with the `verified` list), one `processing_steps` entry, and one `logger.info("Quote check: …")` line.

**Verify node in delete mode** (the only change there). At lms:3709-3735, the rule is:

```python
quote_check = state.get("quote_check") or {}
if quote_check.get("mode") == "delete" and "verified" in quote_check:
    deterministic_claims = list(quote_check["verified"])
else:
    ...today's extract_direct_quotes / find_quote_source / find_shelfmark_source loop, unchanged...
```

Many tests call `_verify_claims_node` directly with no `quote_check` key (`test_graph_rag_phase1.py:530, 589, 646, 699, 736, 765, 1286`; `test_judge_v2_and_language.py:320`), and the repair-mode pass-through stores `{"mode": "repair"}` with no `verified` key. Both must take today's path. The `delete`-mode claims are all `supported=True`, so the hard-gate loop at lms:3840-3859 records them as `SUPPORTED` with `supported_evidence_units` and adds no hard failures. Do not add the deleted spans to `soft_flagged_claims`: unanchored soft flags render in the red "Additional unverified claims" box (ChatUI:1373-1388), which is wrong for text that is no longer there.

**Effects:**

- In delete mode `FABRICATED_CLAIMS` comes only from a model `contradicted` verdict (lms:3942-3944).
- `unsupported` stays a soft flag (lms:3945-3951), unchanged.
- Repair mode is byte-for-byte today's verify node.

**Repair re-check.** Route `repair_answer → check_quotes` instead of `→ verify_claims` (lms:1756), via the no-op guard in C4. A repaired paragraph then gets the same quote check.

**Closing note.** Add it in `_finalize_response_node`, after flag anchoring (lms:4271-4282) and before `_linkify_all_shelfmarks` (lms:4284), only when `quote_check.sentences_deleted > 0`. The 35B never sees it.

- One italic line, in the answer's language (`hebrew_char_ratio(answer) > 0.5`; already imported at lms:29).
- English: `_Some quoted statements were removed because they could not be found in the sources._`
- Hebrew: `_חלק מהציטוטים הוסרו משום שלא נמצאו במקורות._`

**Synthesis prompt** (small and cheap; it reduces deletions):

- In rule 1 (lms:3294-3298), add: put punctuation outside the closing quote mark; italicise work titles and never double-quote them; copy quotations exactly; use … only for omissions.
- In rule 6 (lms:3319-3323), drop "(or near-verbatim)".
- Leave rule 11 (lms:3342-3346) as is.

### C3. One guarded smoothing call

Smoothing is optional for correctness: if it fails, the unsmoothed text ships.

**Model.** `QUOTE_SMOOTHING_MODEL`; when unset, default = `self.verification_model` (prod `qwen/qwen3.6-35b-a3b`, already pinned, so no JIT load). Setting it explicitly to an empty string disables smoothing.

**4B safety guard (hard rule, in `__init__`).** The code default of `VERIFICATION_MODEL` is `qwen/qwen3-4b-2507` (lms:1532), the same as `ROUTER_MODEL` (lms:1530). In any environment without the MBP's env files (a temp clone, a dev backend), smoothing would therefore send `_REPAIR_RESPONSE_FORMAT` (a `json_schema`, lms:3475-3499) to the 4B, which segfaulted the MLX engine on 2026-10-07 and is forbidden. So: if the resolved smoothing model is empty, equals `self.router_model`, or contains `qwen3-4b` (case-insensitive), set it to `""` and record `smoothing: "disabled"`. Log one warning at init. Unit test (case 17b): a service built with `VERIFICATION_MODEL=qwen/qwen3-4b-2507` and no `QUOTE_SMOOTHING_MODEL` has smoothing disabled and makes no `_chat_completion` call when a sentence is deleted.

**Reasoning.** `QUOTE_SMOOTHING_REASONING_EFFORT`, default `none`. On this 35B, **only `none` turns thinking off**: `low` falls back to thinking on (3-4.5k reasoning tokens), and `"off"` returns HTTP 400 (P2 §1 E2/E3; `p2_raw/E2b_35b_synth_off_HTTP400.json`). With `none` it decodes at about 98 tok/s uncontended, so a 150-400-token rewrite takes about 2-5 s, or 10-20 s under VLM contention.

**Call path.** Smoothing calls the new `_chat_completion(...) -> Tuple[str, Optional[str]]` introduced in C5.4, **not** `_call_llm`: `_call_llm` returns only `str`, and the guard needs `finish_reason`. `_chat_completion` takes the same parameters as `_call_llm` plus `reasoning_effort: Optional[str] = None`:

- send `effort = reasoning_effort or (self.reasoning_effort if model in (self.synthesis_model, self.verification_model) else None)` and set `payload["reasoning_effort"] = effort` when it is truthy. Today only the global value is sent, and only to those two models (lms:1851-1852);
- `_call_llm` stays a thin wrapper with its current signature (no new parameter needed), returning `(await self._chat_completion(...))[0]`.

Test 17 mocks `_chat_completion` (returning `(content, finish_reason)` tuples), not `_call_llm`.

**One call** for at most 3 affected paragraphs. Further paragraphs stay unsmoothed.

- Reuse `_REPAIR_RESPONSE_FORMAT` (lms:3475-3499, `{revisions:[{paragraph_index, revised_text}]}`) with `extract_json_object(raw, anchor_key="revisions")` (lms:1031-1063).
- Call parameters: `temperature=0.0`, `recover_reasoning=True`, `max_tokens=1024`, `reasoning_effort=self.quote_smoothing_reasoning_effort`.
- For each paragraph, the prompt carries the post-deletion text with a `⟦gap⟧` marker where a sentence was removed. The marker exists only in the prompt, never in the draft. The prompt carries no evidence.
- Instruction: "Fix only the transition at ⟦gap⟧ and any reference left dangling by the removal. Keep every other word. Add no facts, names, numbers, quotations, citations or shelf marks. Remove the marker."

**Deterministic guard.** Run it per paragraph. On any violation, keep that paragraph's unsmoothed text and record the reason.

1. **Vocabulary.** Every normalised word of the output that is not in the input must be in a small connective allowlist. A starter list: EN "however, also, moreover, furthermore, in, addition, similarly, this, these, that, it, he, she, they, and, but, which, the, a, an"; HE "בנוסף, כמו, כן, גם, עם, זאת, לעומת, אך, אולם, לכן, אף, ו". Before the Hebrew check, strip one or two leading prefix letters `ו`/`ה` from a new Hebrew word if the remainder is in the list, so `ובנוסף`, `וכן`, `והם` pass. This blocks new numbers, names, shelf marks and quoted words. Hebrew has no capitalisation, so a word-level check is required.
2. **Markup.**
   - Every quotation in the input (`extract_quotations(..., min_words=1)`) appears verbatim in the output, and the output has none that is not in the input.
   - The counts of `**`, `[`, `(`, `doc:` and `` ` `` must not increase.
   - No `⟦` may remain.
3. **Length.** `0.6 * len(in) <= len(out) <= len(in) + 80`.
4. **Call health.** Reject on `finish_reason == "length"`, a parse failure, or an exception. Use **one** narrow `try/except (ModelUnavailableError, httpx.HTTPError)`: failing to smooth must never fail the answer.

#### C3a. Guard revision from a live test (2026-10-10; supersedes guard items 1 and 3 above)

A real smoothing call was measured on the prod 35B (`reasoning_effort=none`, 228 tokens in, 167 out, no reasoning; 17.8 s with the sibling VLM batch running, about 2-3 s uncontended). Raw record: `evals/results/rag_audit_20261007/kvtest_20261010/k3_35b_smoothing.json` in Isaac's checkout.

**What happened.** The removed first sentence left a dangling "His transformation…". The model added one bridging sentence, "The preface to the India Book recounts how Goitein's career shifted dramatically.", and returned every other sentence verbatim.

**Why the guard above is wrong.** It would **reject** this output. The new words `how, s, career, shifted, dramatically` are not on the allowlist, and the output is 81 chars over the input against a +80 limit. Every realistic bridge needs a few ordinary words, so as written smoothing would almost never apply.

**Use these rules instead:**

1. **Diff confinement.** Split the input and output into sentences. Every sentence not adjacent to a `⟦gap⟧` must be byte-identical in the output. Only the sentence before and the sentence after each gap may change, plus at most one inserted sentence per gap.
2. **No new facts in the changed text.**
   - No new digits.
   - No new Latin-script word starting with a capital letter that is absent from the paragraph (new proper nouns).
   - No new quotation, `**`, `[`, `(`, `doc:` or backtick.
   - No new shelf mark (`detect_shelfmarks`).
   - At most **8 new words** per gap (Hebrew: at most 4, after stripping one or two leading `ו`/`ה`).
3. **Length.** `0.6 * len(in) <= len(out) <= len(in) + 160`.
4. **Safety net.** `check_quotes` runs before the 35B verifier, so any claim in the filler (such as "career shifted dramatically") is still checked by the 35B and soft-flagged if unsupported. Keep that ordering.

**Prompt tweak.** Ask for the *smallest* fix: "prefer replacing a pronoun left without its referent by the name already used in the paragraph, or adding one short connective; add a new sentence only if nothing else works; do not summarise."

**Test case 17 additions:**
- the measured K3 input/output pair, which is accepted under C3a;
- an output that alters a non-adjacent sentence, which is rejected;
- an output adding `in 1952`, which is rejected (digit);
- an output adding `Mann`, which is rejected (new proper noun).

**Top-level `import httpx`.** `lms_agentic_search` imports `httpx` only locally, inside `_loaded_model_ids`, `_call_llm_with_tools`, `_call_llm`, `load_model` and `list_available_models` (lms:1565, 1785, 1858, 1933, 4476). An `except (ModelUnavailableError, httpx.HTTPError)` in the smoothing code or the verify node (C5.1) would raise `NameError` on exactly the failure path. Add `import httpx` at module top (next to lms:9-20); the local imports can stay or go. Tests 17 and 21 raise `httpx.ConnectError` so this path is covered.

### C4. `QUOTE_FAILURE_MODE` and the repair no-op guard

**Env flag.** `QUOTE_FAILURE_MODE`, read in `AgenticRAGService.__init__` next to the other env reads (lms:1529-1548). Values: `delete` (default) or `repair`. `repair` means today's behaviour plus the no-op guard below; it is the rollback.

**No-op guard** (applies in both modes). `_repair_answer_node` (lms:3502-3665) clears `error`/`error_type` in two places, and the guard must decide **before** each:

- the no-anchor clear at **lms:3568-3571** (`if not claims_by_paragraph: state["error"] = None; state["error_type"] = None; return state`);
- the unconditional clear at **lms:3659-3660**, right after the `draft_answer` join at lms:3656-3658.

Cases:

- **Nothing anchorable** (`claims_by_paragraph` empty, lms:3568). The unanchorable rejections were already moved to soft flags at lms:3547-3565. Clear the error as today, but set `state["repair_noop"] = True`. This skips a pointless 35B re-verify.
- **Draft unchanged.** After the join at lms:3656-3658 and before the clear at lms:3659-3660, compare the new `state["draft_answer"]` with the pre-repair `draft` (lms:3519). They are equal when the 27B returned identical text, or when its revision was rejected (parse failure, or it dropped a protected quote, lms:3640-3652) and the deterministic `remove_sentences_containing` fallback (lms:3636-3639, 3648-3651) found nothing to remove. When the deterministic removal does remove something, the draft changed and this is not a no-op.
  - Set `repair_noop = True`.
  - **Keep** `error_type = "FABRICATED_CLAIMS"` (skip the clear at lms:3659-3660).
  - Set `excluded_claims` to the anchorable claims only (the union of `claims_by_paragraph` values), so the unanchorable ones are not soft-flagged twice.
  - The finalizer's exhaustion branch (lms:4224-4267) then deletes those sentences, which is today's end state reached without the extra rounds.
- **Otherwise** set `repair_noop = False` and keep today's behaviour.
- **Edge.** Replace `add_edge("repair_answer", "verify_claims")` (lms:1756) with a conditional edge: `state.get("repair_noop")` → `finalize_response`, else → `check_quotes`.

A no-op repair ends the loop (the edge goes straight to finalize), so there is at most one per request. The metric can be a bool (`noop_repair`) rather than a count.

### C5. N3: never lose a finished draft (ASSUMPTION TO CONFIRM, see §2)

1. **Wrap the verifier call** (lms:3824, `await self._call_verifier(verifier_prompt)`, currently unwrapped) in `except (ModelUnavailableError, httpx.HTTPError)`. This needs the module-level `import httpx` (C3).
   - On failure, set `model_claims = []`, `model_summary = ""`, `parse_error = None`, set `state["verification_unavailable"] = True`, and keep the deterministic and quote results.
   - Append a `verification_error` soft flag with the reason "The verification model was unavailable; this answer could not be double-checked." Use the same shape as the parse-error flag at lms:3953-3958.
   - The node returns normally, so both `chat()` (`/chat`) and `chat_stream()` (`/chat-stream`) keep the draft and sources, and the existing ChatUI banner (ChatUI:1362-1371) shows.
   - Other exception types (e.g. the `RuntimeError` at lms:1888, a bug) still propagate; in `chat_stream()` they hit the backstop below, in `chat()` they fail as today.
   - Template test: `test_verifier_invalid_json_retries_then_flags_visibly` (tests:747).
2. **Stream backstop.** At lms:4706-4707 (`if isinstance(event, Exception): raise event`):
   - **Which node failed.** `accumulated` only holds completed updates, so it cannot say which node raised. Record it in the per-request metrics dict, which the producer task shares by reference: in `_timed_node` (lms:1658-1669), set `sink["active_stage"] = name` (where `sink = _REQUEST_METRICS.get()`, if not `None`) before `await node(state)`, and set it back to `None` after the node returns normally (inside the existing `try`, no new `except`). After an exception, `request_metrics.get("active_stage")` names the failing node.
   - **Run the backstop only when** `accumulated.get("draft_answer")` is non-empty **and** the failing node is not `finalize_response` (re-running the finalizer would just re-raise). Otherwise re-raise as today.
   - **Reset before finalizing.** Set `accumulated["error_type"] = None`, `accumulated["error"] = None` and `accumulated["excluded_claims"] = []` before calling the finalizer. If the crash came after a verify pass that set `FABRICATED_CLAIMS` (e.g. inside a repair), the finalizer's exhaustion branch (lms:4224-4267) would otherwise delete sentences or swap in the extractive fallback.
   - Add the same `verification_error` soft flag to `accumulated["soft_flagged_claims"]`, set `accumulated["verification_unavailable"] = True`, then `await self._finalize_response_node(accumulated)` (it touches Neo4j and ES only, never LM Studio).
   - Break out of the event loop and build the normal `final` from `accumulated`, but set `success=False` and `error_type="VERIFICATION_INCOMPLETE"` **only on the yielded response** (do not write `VERIFICATION_INCOMPLETE` into `accumulated` before finalizing; the finalizer branches on `error_type`).
   - app.py's handlers (app:1463-1502) are then reached only for failures before a draft exists, or in the finalizer.
   - `chat()` gets no backstop: it is only the frontend's fallback path, and C5.1 already covers the expected verifier failures there. Acceptance #6 is worded accordingly.
3. **`LM_STUDIO_ALLOW_JIT_LOAD`**, default `1` (today's behaviour; the tests at tests:996-1032 expect JIT). In `resolve_model`, when it is `0`, raise `ModelUnavailableError` at lms:1614 instead of calling `load_model` (lms:1910-1955). An in-request load holds the gateway slot for up to 300 s and evicts pinned models. Recommended prod value: `0`. The Studio watchdog re-pins the 4 prod models.
4. **No identical retry after `length`.**
   - Move the body of `_call_llm` into a new `_chat_completion(...) -> Tuple[str, Optional[str]]` that returns `(content, finish_reason)` from `result["choices"][0]["finish_reason"]`.
   - Keep `_call_llm` as a thin wrapper with **unchanged signature and return type**: about 30 tests monkeypatch it with `AsyncMock(return_value=str)`.
   - `_call_verifier` (lms:4062-4099) switches to `_chat_completion` and stops retrying when `finish_reason == "length"`. The result is a parse error, which becomes the `verification_error` flag.
   - Update every test that drives the verifier through a `_call_llm` mock to patch `_chat_completion` instead (returning `(content, "stop")`): in `test_graph_rag_phase1.py` the patches at tests:498-500 (test at 493), 552-554 (546), 609-611 (603), 670-672 (665), 714 (708), 753 (747) and **1258-1260 (`test_crossscript_citation_not_downgraded`, test at 1252, call at 1286)**. Tests that patch `_call_llm` for synthesis, routing or repair (774, 805, 865, 1143, `test_multiturn_rag.py`, `test_judge_v2_and_language.py:286`) are unaffected, because those callers keep calling `_call_llm`. `test_judge_v2_and_language.py:313` patches `_call_verifier` directly and is unaffected.
5. **`ttl`: env only, no code change.** `LM_STUDIO_MODEL_TTL=0` drops `ttl` from all payloads (gated at lms:1779-1780, 1853-1854 and 1940-1941; default 3600 at lms:1540).
   - A read-only `lms ps --json` on 2026-10-10 showed the three prod chat models (`qwen3.8-27b`, `qwen3.6-35b-a3b`, `qwen3-4b-2507`) at `ttlMs: null`, i.e. pinned, despite prod sending `ttl=3600`. Dropping it is probably cosmetic.
   - Recommend it only together with `LM_STUDIO_ALLOW_JIT_LOAD=0`, and only with Isaac's OK.

### C6. Metrics

- **New state keys.** Add `quote_check: Dict[str, Any]`, `repair_noop: bool` and `verification_unavailable: bool` to `AgenticRAGState` (lms:397-442), and initialise them (`{}`, `False`, `False`) in **both** initial states: `chat()` at lms:4531-4562 and `chat_stream()` at lms:4609-4640. If you miss one, the stream's `accumulated` lacks the key.
- **`_build_metrics`** (lms:1674-1712; add to the return dict at 1701-1712, next to `verification_cycles`/`repair_attempts`):
  - `quote_check` (flattened from `state["quote_check"]`: `mode` plus the `counts`, never the claim lists): `{mode, quotations, verified, exempt_title, exempt_question, terms_skipped, unverified, shelf_marks_unverified, sentences_deleted, smoothing: "applied"|"rejected:<reason>"|"skipped"|"disabled", answer_too_short}`;
  - `noop_repair` (bool: at most one no-op per request, since the no-op edge goes straight to finalize);
  - `verification_unavailable` (bool, from C5.1).
- **`record_llm_call`** (lms:50-73) receives the raw `result`. Add `finish_reason` per call, aggregated in `llm_by_stage`. The smoothing call shows up under `llm_by_stage["check_quotes"]` automatically.
- `metrics` is free-form (lms:386-391), and the frontend reads none of these keys, so adding them is safe.

### C7. OPTIONAL, separate frontend item (not in the backend PR)

In `src/frontend/src/ChatUI.jsx`:

- remove the silent re-POST to `/chat` on any stream error, at **1028-1062** (`handleSend`) and **1209-1243** (`handleExampleClick`), or allow it only when `!streamStarted`; the flag is set at 967/973 and never read today;
- treat a stream that ends without `final` (`if (done) break;` at 971 and 1154) as an error with a Retry button.

This needs a frontend container rebuild on the MBP (there is no Node on the host; see `docs/SHARED_RUNTIME.md` §5), so it is a prod deploy and needs Isaac's OK. Put it in a separate small PR or commit that Isaac can schedule on its own.

---

## 6. Tests

**Never run bare `pytest`.** `pytest.ini` has `testpaths = src/backend`, and `src/backend/tests/test_agentic_rag.py` builds a real service against `http://127.0.0.1:1234` (test file :25, :216). On the Studio that is prod LM Studio inference.

Run only named files, from the temp clone root, with the main checkout's venv (it has pytest 9.0.1, pytest-asyncio and langgraph):

```
PYTHONPATH=. /Users/isaac/Documents/GitHub/genizah_search/.venv/bin/python -m pytest -q \
  src/backend/tests/test_quote_check.py \
  src/backend/tests/test_graph_rag_phase1.py \
  src/backend/tests/test_pipeline_metrics.py \
  src/backend/tests/test_judge_v2_and_language.py \
  src/backend/tests/test_shelfmark_linkification.py \
  src/backend/tests/test_multiturn_rag.py
```

Fixtures elsewhere are inline `state` dicts plus `monkeypatch.setattr(service, "_call_llm", AsyncMock(...))`; there is no conftest. Follow that pattern; for verifier and smoothing calls patch `_chat_completion` with `AsyncMock(return_value=(content, "stop"))`.

Existing tests that must stay green (or be ported to the new API with the same intent), all in `src/backend/tests/test_graph_rag_phase1.py`:

| test | line |
|---|---|
| `test_verifier_checks_quotes_deterministically_and_claim_propositions` | 493 |
| `test_contradicted_claim_triggers_targeted_repair` | 665 |
| `test_verifier_invalid_json_retries_then_flags_visibly` | 747 |
| `test_span_repair_rewrites_only_the_offending_paragraph` | 805 |
| `test_span_repair_falls_back_to_deterministic_removal` | 865 |
| splitter and flag-wrap tests | 1064, 1084 |
| `test_hebrew_quotes_and_pointed_text_verify` | 1174 |
| `test_crossscript_citation_not_downgraded` (patch moves to `_chat_completion`, C5.4) | 1252 |
| exhaustion tests, incl. `_build_verified_evidence_fallback` called with `state` only | 1470-1555 |

Also `test_judge_v2_and_language.py:320` (calls `_verify_claims_node` with no `quote_check`; patches `_call_verifier`, unaffected by C5.4) and `test_shelfmark_linkification.py:235` (imports `shelfmarks_equivalent` from lms; works via the re-export, C1).

Tests that exercise today's quote hard gate should run with `QUOTE_FAILURE_MODE=repair` (set on the service instance), so the rollback path stays covered.

**New `test_quote_check.py`.** Write each case with a small inline source dict (`quoteable_text`, `citation`, `evidence_text`, `source_number`).

| # | case | expected |
|---|---|---|
| 1 | Draft `Goitein called it the "India Book." He began in 1948.` | term, not a quotation; nothing deleted; the splitter yields 2 sentences |
| 2 | Draft `"documents of the India Book."` vs source `…documents of the India Book, which…` | verified |
| 3 | Curly `“letters of the India traders”` vs a straight-quoted source, and the reverse | verified |
| 4 | Hebrew pointed draft `"בְּרֵאשִׁית בָּרָא אֱלֹהִים"` vs unpointed source, and the reverse; `Yijū` vs `Yiju` | verified |
| 5 | `ראה כ"י אוקספורד, ע"ו ורמב"ם` | zero quotations; `כ״י` with real gershayim also zero |
| 6 | Hebrew prefix quote `הוא כתב ש"נשתמרו בגניזה רבות מתשובותיהם" בספרו` | one quotation, body `נשתמרו בגניזה רבות מתשובותיהם` |
| 7 | Title: `In "Kol Nidre: Studies and Essays" he argues…` with citation `Gershon, *Kol Nidre: Studies and Essays*, p. 12` | `exempt_title`, nothing deleted |
| 8 | Question echo: user_query `Why did Goitein call it "a Mediterranean society in the Genizah"?`, and the draft repeats the span; the source lacks it | `exempt_question`, nothing deleted |
| 9 | Tisha b'Av mis-pairing: `The word "הכפישה" is derived from Lamentations 3:16 ("ויגרס בחצץ שני"), and "the other one here" too.` | exactly the spans `ויגרס בחצץ שני` and `the other one here`; no span contains `is derived from` |
| 10 | Ellipsis: `"seminal study of the documents … did more to illuminate the period"`, with both segments in the source in order | verified; segments in reverse order → unverified |
| 11 | OCR hyphenation: source `their where- abouts in the Ara- bic letters`, draft `"their whereabouts in the Arabic letters"` | verified |
| 12 | Fabricated: `Goitein wrote that "there was no rabbinic court in Fustat at all." He later revised this.` (not in the source) | first sentence deleted; `He later revised this.` survives; `sentences_deleted == 1` |
| 13 | 09-28 misquote: draft `It is "a rare surviving early medieval Iraqi ketubbah produced in Baghdad sometime between 889 and 988," as noted.` vs source `…a rare surviving early medieval Iraqi ketubbah produced in Baghdad, now…`; repeat with a source reading `between 889 and 998` | deleted in both |
| 14 | Shelf mark: draft `TS 12.243` vs source `T-S 12.243` | verified. A mark found in neither the sources nor the question → sentence deleted. A mark echoed from the question → exempt |
| 15 | Hebrew spelling variant: draft `"משובשת בחשבונותיה בספר ההוא והגהתי"` vs source `…והגהותי` | deleted under the strict default; documents the open decision in §9 |
| 16 | `delete_sentences_at`: bullet line keeps its `- ` marker; heading left without a body is dropped; empty paragraph removed | as stated |
| 17 | Smoothing guard: one test per rejection (new number, new name, new or removed quotation, new `(`/`[`/`**`, leftover `⟦`, too long/short, `finish_reason="length"`, raised `httpx.ConnectError`), plus one acceptance case: `ובנוסף` / `However,` added at the gap passes. All via a `_chat_completion` mock | unsmoothed paragraph kept, reason recorded (acceptance case: smoothed text kept) | unsmoothed paragraph kept, reason recorded |
| 17b | 4B guard: `VERIFICATION_MODEL=qwen/qwen3-4b-2507`, no `QUOTE_SMOOTHING_MODEL` (and separately: smoothing model == `ROUTER_MODEL`) | `smoothing == "disabled"`; `_chat_completion` never called; deletion still applied |
| 18 | Node: deletion leaves under 200 chars | routes to finalize via the new `answer_too_short` branch; fallback returned with the delete-mode lead text; verifier mock **not** called. Second case: a 150-char draft with no quotes and nothing deleted → **not** too short, goes to the verifier |
| 19 | Node: delete mode → verify node gets the verified quotes as `SUPPORTED` and no quote hard failures; mocked 35B returns all supported → no repair | as stated |
| 20 | No-op guard: the 27B mock returns the paragraph unchanged | `repair_noop`, edge to finalize, exhaustion removes the sentence, verifier called once |
| 21 | N3: `_call_verifier` raises `ModelUnavailableError` and `httpx.ConnectError` | draft and sources kept, `verification_error` flag present, `success` path unchanged |
| 22 | N3: `_chat_completion` returns `finish_reason="length"` | no second verifier attempt |
| 23 | Metrics: `_build_metrics` includes `quote_check`, `noop_repair`, `verification_unavailable`, and per-call `finish_reason` | as stated |
| 24 | Splitter abbreviations: `Nahray b. Nissim wrote that "there was no court in Fustat at all." He travelled widely.` and `Goitein (d. 1985) wrote that "<fabricated three-plus words>". His work continues.` | whole first sentence deleted, no `Nahray b.` / `Goitein (d.` remnant; second sentence survives |
| 25 | Same unverifiable shelf mark (e.g. `T-S 99.999`) in two different sentences | both sentences deleted |
| 26 | Editorial interpolation: draft `"the [Geniza] documents of the eleventh century show"` vs source `the documents of the eleventh century show` | verified; and vs a source reading `the [Geniza] documents of…` also verified |
| 27 | Ellipsis with a trivial segment: `"the … there was no court in Fustat"` vs a source with `the` but not the clause | unverified (the 1-word segment is ignored, the clause must match) |
| 28 | Generated-summary quote: bibliography source with `quoteable_text=""` and `description` containing `a decisive turning point in Fatimid trade`; draft quotes that phrase | **unverified** (not `exempt_title`): only graph-source `evidence_text` joins the title haystack |
| 29 | Verified-list filtering: one sentence holds a verified and an unverified quote | sentence deleted; `quote_check["verified"]` no longer contains the verified quote |
| 30 | Stream backstop (`chat_stream` with the graph nodes monkeypatched): (a) `verify_claims` raises `RuntimeError` after a draft exists → `final` with the draft, sources, `verification_error` flag, `error_type="VERIFICATION_INCOMPLETE"`; (b) state carried `error_type="FABRICATED_CLAIMS"` and `excluded_claims` from an earlier verify, then `repair_answer` raises → no sentence deleted, no extractive fallback; (c) `finalize_response` raises → re-raised, no second finalize | as stated |
| 31 | Log fixtures from on-disk sources (see §7 #3): l46, l55 | as stated in §7 #3 |

**Replay check (offline, no LM Studio).**

- Write a small script, e.g. `evals/replay_quote_check.py`. Commit it only if Isaac wants it.
- It imports **only** `src.backend.quote_check` (which pulls in `shelfmark_normalizer`, both pure). It must not import `lms_agentic_search`, which needs langgraph and the service singletons.
- It parses `p2_raw/p18_verify_request.json` and `p07_verify_request.json` into sources plus draft, using `parse_verifier_prompt` from Appendix B or the prototype, and runs `check_quotations` over the prod drafts and every `p2_raw/E*.json` `content`. In the parsed sources, a graph source is one whose citation starts with `Neo4j knowledge graph record`: drop its `quoteable_text` key so it is treated like `build_verification_sources` output (C1 title haystack).
- Questions: P18 `Who is S.D. Goitein`; P07 `מה ידוע מן הגניזה על תשובות הגאונים?`.
- Run it from the temp clone root with the venv interpreter (do **not** use `python3 -I`: `-I` drops the cwd and `PYTHONPATH`, so `import src.backend.quote_check` fails):
  ```
  PYTHONPATH=. /Users/isaac/Documents/GitHub/genizah_search/.venv/bin/python evals/replay_quote_check.py \
    /Users/isaac/Documents/GitHub/genizah_search/evals/results/rag_audit_20261007/p2_raw
  ```
  It only reads files. The prototype (`proto_quote_check.py`, same folder as this handoff) already reproduces the §7 #2 numbers this way; a variant with the C1 rules of this revision (graph-only title haystack, 2-word ellipsis segments, bracket gaps) gave the same result on 2026-10-10.
- The `rv_n2a/` log slices hold only extracted quote lists and repair diffs, with no full draft and no source text, so they are not replayed. They become unit fixtures instead (case 31, §7 #3).

**Opus non-regression eval.** Optional, and **Isaac sets it up or explicitly approves each step.** Do not start it on your own. What it needs and where it is unsafe as naively written:

- **Host.** This session runs on the Mac Studio (`hostname` = `isaacs-Mac-Studio.local`). Only `kraken-linewise` runs in Docker here: there is no `genizah_search-backend-1`, no prod Neo4j and no prod ES. `scripts/dev_backend_local.py` reads the Neo4j credentials from that container and defaults to `bolt://127.0.0.1:7681`, `LLM_STUDIO_URL=http://127.0.0.1:1234` and `EMBEDDING_SERVICE_URL=http://127.0.0.1:8001`. Which ES (e.g. the local mirror on `localhost:9200`, per Isaac's notes) and which Neo4j to target, and how the embedding service is reached, is Isaac's call. Ask; do not improvise.
- **Model env.** A temp clone has no `src/backend/.env`, so the dev backend would start with the code defaults: `SYNTHESIS_MODEL=c4ai-command-r-v01` (a JIT load that evicts the pinned prod models) and `VERIFICATION_MODEL=qwen/qwen3-4b-2507` (receives `json_schema` → MLX segfault risk). Before any start, get the prod values from Isaac and export explicitly: `SYNTHESIS_MODEL`, `VERIFICATION_MODEL`, `ROUTER_MODEL`, `SYNTHESIS_REASONING_EFFORT`, plus `LM_STUDIO_ALLOW_JIT_LOAD=0` (candidate only; the baseline code ignores it, so for the baseline confirm with `~/.lmstudio/bin/lms ps --json` that all configured models are resident). Then read the startup log and confirm the model ids **before** sending any request.
- **Code under test.** The launch config `backend-local` (`.claude/launch.json`) runs `.venv/bin/python scripts/dev_backend_local.py` relative to the **main checkout**, i.e. Isaac's `eval-retrieval` code. Do not use it. Run the candidate from the temp clone, and the baseline from a second clean clone checked out at `e765768`, each as `PYTHONPATH=. /Users/isaac/Documents/GitHub/genizah_search/.venv/bin/python scripts/dev_backend_local.py` from that clone's root (absolute venv path), one at a time on :8010.
- **Runs.** `ANTHROPIC_API_KEY`, quiet hours, one run at a time, same dataset for baseline and candidate. `scripts/run_agentic_rag_eval.py` on prod-mbp has **no `--stream`** (that exists only on unmerged `origin/yk-holiday` a3aaed7), so target :8010, not the public API. Its default `--timeout` is 300 s while chat p90 is 675 s, so pass `--timeout 900` or more:
  ```
  PYTHONPATH=. /Users/isaac/Documents/GitHub/genizah_search/.venv/bin/python scripts/run_agentic_rag_eval.py \
    --api-base-url http://127.0.0.1:8010 --judge-provider anthropic --judge-model <opus id> \
    --timeout 900 --wait-for-idle --output <scratch>/<run>.json
  ```

---

## 7. Acceptance criteria

1. All named test files pass, including the new cases 1-31 (17b included).
2. **Replay over the 10 P2 drafts:**
   - every format, title and mis-pairing false positive is cleared;
   - exactly **2** quotations are unverified: the E2d invented `"אין לסלוח על כך… יתלצו ויתלעסו הרוסים"` (a true catch), and the prod-P07 spelling variant `והגהתי`/`והגהותי` (strict default; this becomes 0 only if Isaac opts into the fold, see §9);
   - E2b's spliced ellipsis quote is verified.
3. **Log cases, as unit fixtures (case 31) built from on-disk sources.** The `rv_n2a` slices hold only quote lists and diffs, so each case pairs the logged draft quote with a source excerpt copied from the audit's on-disk data:
   - **d1007** Goitein `"India Book."` (`rv_n2a/d1007.txt`; sources in `p1/finals/P18.json`) → a term, no deletion and no repair. Already covered by case 1.
   - **l52** 09-28 `"a rare surviving early medieval Iraqi ketubbah produced in Baghdad sometime between 889 and 988,"` → deleted. Already covered by case 13.
   - **l46** 09-23 Sukkot. Draft quote (from `rv_n2a/l46.txt`): `During Sukkot (the Feast of Tabernacles) a qinṭir arrived from the Land of the Zanj, which had in it new slaves.` Source: Goitein and Friedman, *India Traders*, p. 457 chunk in `p1/finals/P03.json`, main text `During Sukkot (the Feast of Tabernacles) a qințir (?) 19 arrived from the Land\nof the Zanj, which had in it new slaves. 20 I tried…`. **Correction to the audit:** the gloss "(the Feast of Tabernacles)" is in the source, so this was not an inserted gloss. The draft omitted the editorial `(?)` (punctuation, ignored) and the footnote numeral `19`. Under the strict rule the `19` breaks the match → **deleted**. This is a false deletion of the "small things off" kind; it is open item §9.3. The fixture asserts today's strict outcome and is updated if Isaac picks a footnote rule.
   - **l55** 10-06 `"the language or formulary of laymen"`. Source: Friedman, *Jewish Marriage in Palestine*, p. 2 chunk in `p1/finals/P17.json`, main text `…is designated in Tannaitic sources as leshon hedyot, the language or for-\nmulary of laymen.4 It primarily developed…` (position ≈347 of the main text, inside the quoteable cap). **Correction to the audit:** this was an OCR-hyphenation false positive of today's matcher, not a near-quote. Under the new rule it is **verified**, nothing deleted.
   - Whether the 09-23 and 10-06 turns had exactly these chunks in their evidence is not recorded; the fixtures test the rule against the on-disk wording.
4. In delete mode, a quote or shelf-mark miss **never** sets `FABRICATED_CLAIMS` and never calls the 27B. Harness for the replay drafts: build the service with `QUOTE_FAILURE_MODE=delete` and `QUOTE_SMOOTHING_MODEL=""`; monkeypatch the instance methods `_resolve_query_node`, `_route_query_node`, `_execute_searches_node` and `_link_primary_secondary_node` to return the state unchanged, and `_synthesize_answer_node` to inject the replay draft as `draft_answer` (sources as `bibliography_results`/`graph_results` in the shape `build_verification_sources` expects; or monkeypatch `build_verification_sources` in the lms module to return the parsed sources) **before** `_build_graph()` is called (the graph binds the methods at build time, so call `service.graph = service._build_graph()` after patching); mock `_chat_completion` to return an all-`supported` verifier JSON (`{"verified_claims": [], "summary": "ok"}` suffices); stub the finalizer's Neo4j/ES helpers to no-ops (`_resolve_unlinked_shelfmarks` lms:4375, `_build_works_cited` lms:3172 → `[]`, and `_render_catalog_entries` lms:4351 → `[]`), or simply monkeypatch `_finalize_response_node` to set `final_answer = draft_answer`. Run `graph.ainvoke` and check `_build_metrics(...)`: `verification_cycles == 1` and `repair_attempts == 0` for every replay draft (they come from `stage_calls`, filled by `_timed_node`).
5. With `QUOTE_FAILURE_MODE=repair`, the existing quote and repair tests pass unchanged (except where the no-op guard short-circuits).
6. A `ModelUnavailableError` or `httpx.HTTPError` from the verifier after the draft exists returns the draft, the sources and a `verification_error` flag, from both `chat()` and `chat_stream()` (C5.1). Any other exception after the draft exists, in any node except `finalize_response`, returns the same from `chat_stream()` with `error_type="VERIFICATION_INCOMPLETE"` (C5.2); `chat()` re-raises it as today.
7. If Isaac approves the Opus eval: candidate mean ≥ baseline mean − 0.2, no case down by more than 1.5 points, and median `verification_cycles` = 1.

---

## 8. Rollout

**Code defaults:**

| flag | default |
|---|---|
| `QUOTE_FAILURE_MODE` | `delete` (Isaac's decision) |
| `QUOTE_SMOOTHING_MODEL` | `VERIFICATION_MODEL` |
| `QUOTE_SMOOTHING_REASONING_EFFORT` | `none` |
| `LM_STUDIO_ALLOW_JIT_LOAD` | `1` (today's behaviour) |
| C5.1 verifier wrap and C5.2 stream backstop | always on, no flag (loss-proofing only) |

**Recommended prod `.env` additions** on the MBP: `LM_STUDIO_ALLOW_JIT_LOAD=0`, and optionally `LM_STUDIO_MODEL_TTL=0`.

- Prod model and reasoning values are **not in git**. They come from the MBP's env files, read through compose `env_file` (`docker-compose.yml`: the repo-root `.env`, then the optional `src/backend/.env`, which wins on conflicts) at container create, so any change means a container recreate. The new flags go in the same place, by Isaac.
- **`LLM_STUDIO_URL` is not set by either env file in prod.** `docker-compose.mbp.yml` sets it under `environment:` as `${STUDIO_LM_STUDIO_URL:-http://192.168.8.120:1234}`, interpolated from the **repo-root `.env`** on the MBP, and compose `environment:` overrides every `env_file` entry. So editing `src/backend/.env` has no effect on it. The compose default is `.120` while Isaac's notes and the audit say `.122`; the real value is whatever `STUDIO_LM_STUDIO_URL` holds in the MBP's root `.env`. Do not touch it.
- **`.env.example`** (repo root, tracked; placeholders only): add the new variables next to the existing `LM_STUDIO_MODEL_TTL=3600`, one comment line each: `QUOTE_FAILURE_MODE=delete`, `# QUOTE_SMOOTHING_MODEL=` (commented out: unset = use `VERIFICATION_MODEL`; an explicitly empty value disables smoothing; a 4B is always refused), `QUOTE_SMOOTHING_REASONING_EFFORT=none`, `LM_STUDIO_ALLOW_JIT_LOAD=1`. In code, distinguish unset from empty: `os.getenv("QUOTE_SMOOTHING_MODEL")` returning `None` means "use the default", `""` means "disabled".

**Deploy** is a backend image rebuild on the MBP: Isaac's call, at a time he picks. You open the PR; you do not deploy.

**Rollback** is `QUOTE_FAILURE_MODE=repair` plus a container recreate, with no rebuild. A full rollback is a revert of the PR plus a rebuild.

**What to watch** for the first days, in `metrics` and the `Quote check:` log line. Logs live on the **M3 MBP**: `docker logs genizah_search-backend-1` there (plus `./logs` mounted at `/app/logs`). There is no backend container on this Studio host. (`docs/SHARED_RUNTIME.md` says "this Mac is production"; it predates the 2026-09-15 migration and now describes the MBP.)

- `quote_check.unverified` and `sentences_deleted` per turn: expect most turns at 0;
- the rate of `smoothing: rejected:*`;
- `answer_too_short`: should be rare;
- `noop_repair`;
- `verification_unavailable`;
- `finish_reason=length` counts;
- the share of chat seconds in the `repair_answer` stage, which should fall from about 27 % toward about 0.

Spot-read every turn with `sentences_deleted > 0` for the first week, and report false deletions to Isaac.

---

## 9. Open items to raise with Isaac (do not block coding on them)

1. **N3 default** (C5): keep the draft and sources with the banner; no in-request JIT; no identical retry after `length`. This is the one assumption to confirm. The `/chat` fallback removal (C7) is a separate frontend deploy.
2. **Hebrew spelling variants.** The only false deletion found in the P2 replay is `והגהתי` (draft) vs `והגהותי` (source), a plene/defective spelling of one word.
   - The default is strict: the sentence is deleted.
   - The alternative is an opt-in fold that drops ו/י inside Hebrew words on both sides before matching. It costs about 3 lines and slightly weakens discrimination for 3+-word quotes.
   - Ask whether he wants it.
3. **Footnote numerals inside quoted source text.** OCR'd book pages carry footnote markers inline (`a qințir (?) 19 arrived…`, l46, §7 #3). A faithful quote that leaves out the `19` is deleted under the strict rule. Options:
   - keep strict (default);
   - allow the draft to omit a standalone 1-2 digit token of the source inside a matched quote. This is about 5 lines (try again with such tokens removed from the source side only when the plain match fails). It never lets a quote **add or change** a number, so "between 889 and 988" still fails, but a quote that silently drops a short number would pass.
   - Ask which he wants. The l46 fixture follows his answer.
4. **Prod env values** for `LM_STUDIO_ALLOW_JIT_LOAD=0` and `LM_STUDIO_MODEL_TTL=0`. Both need a container recreate.
5. **Opus eval setup** (only if he wants the eval): the prod `SYNTHESIS_MODEL`/`VERIFICATION_MODEL`/`ROUTER_MODEL`/`SYNTHESIS_REASONING_EFFORT` values, and which ES/Neo4j/embedding service a dev backend on the Studio should use (§6).

## 9a. Measured 2026-10-10: conversation-continuation caching (context for reviewers, and for later work)

Isaac asked whether corrections could go back to the model as a follow-up turn, treating verification like a tool call, so that LM Studio reuses the cached prompt. It was tested live with 5 requests: the P18 Goitein prompts, the VLM batch running, contended timings. Write-up: `evals/results/rag_audit_20261007/kvtest_20261010/RESULTS.md`.

| call | reused from cache | time to first token | wall |
|---|---|---|---|
| 27B synthesis, cold | 0 of 5,852 | 44 s | 199 s |
| 27B follow-up "reply only with REPLACE/DELETE patch lines" | **5,632 (96% of the prompt)**; its own draft (1.8k tokens) re-processed | 18 s | 111 s (792 reasoning tokens at `low`) |
| 35B verify as prod | – | 4 s | 97 s |
| 35B follow-up "re-check only these 2 sentences" | whole prompt (LCP 0.984) | 0.6 s | **9 s** |

**Consequences for this PR:**
- **Keep `reasoning_effort` constant within a conversation.** Qwen3.8's template writes it into a system block at the very top of the prompt (`chat_template.jinja:45-85`), so changing it between calls breaks the cache from token 0. Nothing in this PR needs to change it.
- **No re-verify is needed in delete mode**, because `check_quotes` and smoothing run before the single 35B verify.

**For later (not this PR):**
- If a re-check is ever needed (the repair-mode rollback, or a future late edit), send it to the 35B as a continuation of the verifier conversation carrying only the changed sentences. That takes about 9 s instead of a full re-verify of about 100-250 s.
- A 27B patch protocol (`REPLACE`/`DELETE`/`KEEP` lines as a follow-up turn at the same effort) reuses about 96% of the prompt. Its thinking still costs about half of a full rewrite.
- The 27B noticed in its reasoning that a deliberately mis-flagged quote was in the source, so any such protocol needs a `KEEP` answer.

## 10. Out of scope

- **N1 GPU-contention yield gate.** It goes to the sibling repo `historical-document-analysis` through its own handoff: pause long VLM experiments when a chat request starts, dry-run first. It will stop mattering once the AMD workstation takes the experiments.
- **N4-N7**, including N5/N6 retrieval and top-8 truncation fixes: later. See `docs/RAG_AUDIT_2026-10.md`.
- Verifier prompt changes, model swaps, and context or evidence-budget changes.

---

## Appendix A: prototype core (reference, port with standards applied)

Where this appendix and §5 differ, §5 wins: `quote_found` below predates the 2-word-segment and bracket-gap rules (C1), and the prototype's title haystack (in `proto_quote_check.py`) still includes bibliography sources with empty quoteable text, which C1 forbids.

```python
import re, unicodedata
from typing import List, Optional, Tuple

_FINALS = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})
_DROP_NO_SPACE = set("[]⟨⟩")

def normalize_for_quote_match(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    folded = unicodedata.normalize("NFKC", stripped).casefold()
    out = []
    for ch in folded:
        if ch in _DROP_NO_SPACE:
            continue
        out.append(" " if unicodedata.category(ch)[0] in "PS" else ch)
    return re.sub(r"\s+", " ", "".join(out).translate(_FINALS)).strip()

_OPEN_BEFORE = set(" \t([{—–-:/*_«„“")
_CLOSE_AFTER = set(" \t.,;:!?)]}—–-*_…'’»")
_HEBREW_PREFIXES = re.compile(r"(?:^|(?<=[\s(\[—–:]))[ושהכלבמ]{1,3}$")

def _is_abbreviation_mark(text: str, index: int) -> bool:
    after = text[index + 1:index + 3]
    return len(after) >= 1 and after[0].isalnum() and (len(after) == 1 or not after[1].isalnum())

def _follows_hebrew_prefix(text: str, index: int) -> bool:
    return bool(_HEBREW_PREFIXES.search(text[max(0, index - 4):index]))

def extract_quotations(text: str, min_words: int = 3) -> List[Tuple[int, int, str]]:
    spans, open_at, open_char = [], None, ""
    closers = {'"': '"', "״": "״", "“": "”", "«": "»"}
    for index, ch in enumerate(text):
        if ch == "\n":
            open_at = None
            continue
        before = text[index - 1] if index > 0 else " "
        after = text[index + 1] if index + 1 < len(text) else " "
        if ch in ('"', "״"):
            if before.isalnum() and after.isalnum():
                if _is_abbreviation_mark(text, index):
                    continue
                if not _follows_hebrew_prefix(text, index):
                    continue
                can_open, can_close = True, False
            else:
                can_open = before in _OPEN_BEFORE or before == "\n" or index == 0
                can_close = after in _CLOSE_AFTER or after == "\n" or index + 1 == len(text)
            if open_at is not None and open_char == ch and can_close:
                body = text[open_at + 1:index]
                if len(body.split()) >= min_words:
                    spans.append((open_at, index + 1, body))
                open_at = None
            elif open_at is None and can_open and not can_close:
                open_at, open_char = index, ch
        elif ch in ("“", "«"):
            open_at, open_char = index, ch
        elif ch in ("”", "»") and open_at is not None and closers.get(open_char) == ch:
            body = text[open_at + 1:index]
            if len(body.split()) >= min_words:
                spans.append((open_at, index + 1, body))
            open_at = None
    return spans

_ELLIPSIS = re.compile(r"\.\s?\.\s?\.|…")

def quote_found(body: str, haystack_nospace: str) -> bool:
    position = 0
    segments = [normalize_for_quote_match(s).replace(" ", "") for s in _ELLIPSIS.split(body)]
    for segment in [s for s in segments if s]:
        found = haystack_nospace.find(segment, position)
        if found == -1:
            return False
        position = found + len(segment)
    return True
```

## Appendix B: replay-prompt parser (for the offline replay script)

```python
def parse_verifier_prompt(prompt: str):
    chunks = prompt.split("SOURCE CHUNKS:\n", 1)[1].split("\n\nDRAFT ANSWER:\n", 1)
    sources_text, rest = chunks
    draft = rest.split("\n\nThe type must be", 1)[0]
    sources = []
    for block in re.split(r"\n\n(?=\[SOURCE \d+\])", sources_text.strip()):
        head, _, body = block.partition("\n")
        m = re.search(r"Original page text \(quoteable\): (.*?)(?:\nShelf marks cited in this source:|\Z)", body, re.S)
        sources.append({"citation": re.sub(r"^\[SOURCE \d+\] ", "", head),
                        "quoteable_text": m.group(1) if m else "", "evidence_text": body})
    return sources, draft
# load: json.load(open(p))["messages"][0]["content"]
```
