# Cairo Genizah chat: overnight RAG audit (2026-10-07)


> Supporting notes, scripts and data: `evals/results/rag_audit_20261007/` (written as `audit/` below). The two large raw files, `llm_calls.csv` and `i1_requests.jsonl` (about 36 MB), can be regenerated with `parse_lmstudio_logs.py` and should not be committed.

For Isaac. Audit run 02:25-05:30 EDT against the deployed `origin/prod-mbp` @ e765768.
- Every finding was re-checked by an adversarial verifier. Refuted claims are dropped, corrected numbers are used, and estimates are labelled as such.
- Code legend: `lms` = `src/backend/lms_agentic_search.py`, `app.py` = `src/backend/app.py`, `gt` = `src/backend/genizah_terminology.py`, `ChatUI.jsx` = `src/frontend/src/ChatUI.jsx`. Line numbers are from prod-mbp.
- Logs are cited as `~/.lmstudio/server-logs/YYYY-MM/<file>:line`.

---

## 1. TL;DR

- **Speed.** A typical turn takes **222 s; the p90 is 675 s.** That covers 32 steady-state turns since the 09-23 verifier pin, 18 of them tonight's probes. Real users only: 191 / 629 s.
  - For the 6 turns ranked around the median (averaged, then scaled to 222 s), the time goes to routing + retrieval 18 s → 27B synthesis 129 s → 35B verify 43 s → repair loop 30 s. These are band composites, not per-stage medians (Section 3.1).
  - 17 of the 32 turns had no repair at all. When the loop fires it adds a median 164 s.
  - **107 s of the 222 s is GPU contention.**
- **Cause #1: a shared GPU.** The sibling VLM batch had a request in flight **92.9%** of the time from 09-23 to 10-07, with no quiet hour and no priority.
  - Chat calls that overlap it run 1.9-3.2× slower. Contention is **57% of all chat seconds**.
  - The clean A/B is the same Goitein question with one verify pass each: **317 s with the VLM busy (P01) and 124 s on a quiet GPU (P18)**.
  - Your 702 s Goitein turn also carried a 328 s repair loop that neither probe had. Of its 702 s, 488 s was contention.
- **Cause #2: a quote-mark repair loop.** `"India Book."` counts as a fabricated quotation. That forces a 27B repair plus a full re-verify, up to 3 rounds, with no check for a repair that changes nothing.
  - It fires in 47% of steady-state turns, adds a median 164 s, and is **27% of all seconds**. Goitein lost 328 s to it.
  - Not all of it is waste. About 30% of the loop seconds (803 of 2,770 s) are format false positives or re-verifies of unchanged drafts. The rest are cycles that caught real misquotes, or failures that the safe fixes cannot clear. Examples of real misquotes: a date range that is not on the page, and a gloss inserted inside a quotation.
  - For organic traffic it is mostly a p90 problem.
- **"Not found."** Of 24 organic questions since 09-15:
  - **3 got a full answer** and 2 a partial one;
  - 12 were told the sources don't cover it;
  - 7 were lost to errors, all on 09-22/23.

  The bibliography held relevant pages for 14 of the 24. The causes are spread out:
  - candidate generation;
  - the top-8 cut (a weight-default bug plus a junk Hebrew variant);
  - the subject gate (both Chanukah failures);
  - router plan shape;
  - truncation.

  The verifier then labels these negatives "Checked against all retrieved sources", although it saw only 8 truncated pages.

  **What the fixes would recover** (deduplicated; auditor judgement, no Opus run; Section 5):
  - Selection plus aliases and gate (N5 + N6): about **4 of 25** incidents (range 3-5). All were demonstrated on replay.
  - Adding term-centred windows and keyword-first candidates (X1-X3): about **7 of 25** (range 5-10), with low-medium confidence.
  - For the 12 organic questions since the pin: 3 with N5 + N6 (one Hanukkah question asked three times), and about 4-6 with X1-X3.
- **The three changes that matter most:**
  1. **The sibling batch yields to chat.** This is a sibling-repo change only. Modelled: median 222 → 114 s, p90 675 → 229 s.
  2. **Safe quote-matcher fixes with a no-op guard (N2a), and keep the draft on late failures (N3).**
     - N2a alone gives 191 / 629 s. The p90 barely moves, because the slow tail is genuine catches plus contention. With (1): **109 / 228 s**.
     - A blanket de-quote (N2b) would reach 191 / 428 s, or 100 / 165 s with (1). But it ships real misquotes as paraphrases that only the 35B checks, and the 35B has invented verbatim support before (P17). That is your decision (Section 6).
  3. **Fix selection, aliases and the gate.** Prove the fixes first on a GPU-free replay harness, then on an Opus non-regression run.
- **Perceived and combined.**
  - A provisional draft at 128 s (82 s with the gate) costs no GPU, but 47% of drafts change afterwards.
  - The sources list could stream at 16 s. It is a reading list, not an answer: the UI already shows the "8 bibs" chip at that point.
  - (1) + N2a together give about **109 / 228 s**. Adding a lighter verifier and pre-synthesis cleanup gives about **98 / 198 s**; the verifier part is an estimate. With N2b instead of N2a these become 100 / 165 and 89 / 151.
- **Audit incident.** A replay-bench request crashed the 4B router at 04:41:31. The watchdog reloaded it in about 2.4 min, and no prod request arrived meanwhile.

## 2. How this was measured

**LM Studio logs, 09-15 15:45 → 10-07.**
- Coverage: 33 files, 82% of the period (gaps of 8.5, 25.6 and 60 h).
- 41,365 logged calls. 357 of them belong to the rebuilt chat turns; the rest are VLM batch, evals and tests.
- Rebuilt into 62 turns: 27 organic, 11 of your smoke tests, 20 eval rows, 2 config tests and 2 probes.
- The rebuild matches all 20 eval rows to within −0.4 to +4.9 s.

**Live traffic.**
- 18 sequential `/chat-stream` probes, 02:30-04:13, with 0 failures.
  - 13 ran with the VLM busy: P01-P12 and P16. 5 ran quiet: P13-P15, P17 and P18.
  - v23a-step1500 sent its last request at 03:55:52 and TTL-unloaded at about 03:57.
  - A single JIT request to a third checkpoint, `qwen3-vl-8b-heb-v22i-a50`, ran from 04:05:10 to about 04:07 and ended in a Channel Error. It overlapped P16 (`2026-10/2026-10-07.1.log:63103`).
  - No VLM request arrived after that, up to the end of the log at 05:54.
- 16 uncontended LM Studio replays on rebuilt prod prompts.
- Read-only search replays and 108 link GETs.

**Simulation.** `audit/a1/a1_attribution.py` re-prices every call at quiet rates. It over-predicts quiet turns by 4%, so the contention figures are conservative.

**Incidents.** 25 "not found" incidents (14 organic, 7 probes, 4 eval), each given one primary cause.

| Population | n | median s | p90 s |
|---|---|---|---|
| Steady state (SS) since 09-23 pin: 14 real turns + 18 probes | 32 | 222 | 675 |
| Real interactive since the pin | 14 | 191 | 629 |
| Organic only since the pin (mostly short "not found" answers) | 11 | 175 | 342 |
| All interactive successes, 09-19 → 10-07 | 23 | 188 | 503 |
| 09-20 eval harness, quiet GPU | 20 | 162 | 386 |

**Limits.**
- Organic traffic is mostly you, and n=24 is small (about ±10-15 points).
- The probes repair more often than organic turns.
- Prompts are truncated in the logs, so the evidence windows are reconstructed.
- The MBP env, the backend logs and Neo4j were unreachable.
- Quality calls are auditor judgement; no Opus judge was run.
- Hebrew-language coverage is thin. Only one probe was in Hebrew (P07, a full answer), and no organic Hebrew-language failure was analysed. The Hebrew-prefix gap is shown only behaviourally, because the ES analyzer could not be inspected.
- The follow-up path is only partly audited (Section 4.4).

## 3. Latency

### 3.1 Waterfall

Seconds. Contention is shown in parentheses and is included in the totals.

**How to read the first two columns.** They are band composites, not per-stage medians or percentiles (`a1/a1_attribution.py`, `band()` and `composite()`). Each column averages the stage times of the turns ranked around that percentile and scales them to the percentile wall time.
- **Median band:** 6 turns (P05, L54, P06, P14, P12, L50), scaled to 222 s.
- **p90 band:** 5 turns (P07, P10, P08, L59, P04), scaled to 675 s. Four of the five are probes.

This is why routing + retrieval reads 16 s at the p90 and 18 s at the median. It is also why "repair loop 30 s" is a band average: 17 of the 32 steady-state turns had no repair.

| Stage | Median band composite | p90 band composite | Goitein 10-07 02:04 |
|---|---|---|---|
| Network + 4B routing + retrieval/linking | 18 (7) | 16 (7) | 12 (3) |
| 27B synthesis: prefill / reasoning / answer | 129 (68): 22 / 79 / 28 | 247 (148): 35 / 105 / 107 | 194 (105): 28 / 98 / 68 |
| 35B verify #1 | 43 (26) | 134 (100) | 163 (128) |
| Repair + re-verify loop | 30 (7) | 275 (181) | 328 (253) |
| Gaps, finalize, transfer | 3 | 4 | 6 |
| **Total** | **222 (48% contention)** | **675 (64%)** | **702 (70%)** |
| *Sources exist (not shown)* | *16* | *32* | *11* |
| *Full draft exists (not shown)* | *128* | *291* | *205* |

The Goitein repair changed `"India Book."` to `'India Book.'`. It took 78 s and forced a 250 s re-verify at 14.8 tok/s (`2026-10/2026-10-07.1.log:18195-19148`). Your "Verifying revised claims… · 552 s" came 98 s into that pass.

### 3.2 Ranked findings

| # | Finding | Verdict | Size (corrected) | Fix alone, SS median / p90 (s) |
|---|---|---|---|---|
| LAT-1 | The sibling VLM shares the GPU with no priority. The backend cannot see it (`queued_behind` counts only its own semaphore, lms:76-143). | confirmed | 57% of SS seconds (63% for real users). Verify 3.2×, synthesis 2.4×, repair 1.9× slower. 35B decode 88 → 32 tok/s (~8 during VLM prefill). | **114 / 229**, only while the batch stays this busy. Tonight it went quiet at about 03:57, apart from one 2-minute request at 04:05 that overlapped P16. |
| LAT-3 | Quote and shelf-mark false positives trigger a repair plus a full re-verify, with no no-op guard (lms:894, 3838-3859, 1756). | partial | Repaired turns: 15/32 SS, 5/14 real, 2/11 organic, 7/7 of your eval-style questions. About half the triggers are matcher artefacts, but by seconds only about 30% of the SS loop (803 of 2,770 s) is false positives or no-op re-verifies. The verifier itself said "contradicted" only once in about 2,550 verdicts. | **N2a (safe fixes): 191 / 629**; real users 191 / 416 (p90 from 629). **N2b (blanket de-quote): 191 / 428**; real users 191 / 354. |
| LAT-2 | Sources and the draft are hidden until `final` (lms:4709-4762). | partial | 48% of wall time comes after the draft exists. | Provisional draft visible at **128 s** median (82 s with N1), but 47% change afterwards. The sources list at 16 s is a reading list, not an answer. End-to-end unchanged. |
| LAT-5 | Dense 27B synthesis. | partial | 46% of seconds, about 21% of it the model's own. Reasoning is 54-58% of tokens even at `low`, the floor. | Thinking off: 158 / 577, rejected |
| LAT-4 | The verifier echoes each claim and adds a rationale (about 125 tok/claim). | partial | Clean decode is about 5% of turn seconds. | Estimated 189 / 469 (measured −17%: 210 / 625); about 15 s after LAT-1 |
| LAT-6 | No prompt-prefix reuse. | partial | Prefill is 15.7% of seconds; 6.4% is reachable. | 204-213 / 617-661 |
| LAT-8 | Serial pre-synthesis work; a verbose 4B router. | partial | 5.5% of seconds; router retries are 0.2-0.5%. | About 216 / 669 |
| LAT-7 | Failure paths discard or duplicate work. | partial | 7 turns lost on 09-22/23; 0 since the pin. | Insurance |
| LAT-9 | A single gateway slot (lms:88). | partial | 2 overlaps in 3 weeks, both duplicate submissions. Abandoned turns do get cancelled: on 09-17 a resubmitted ketubbot question waited 76 s for the slot until the first turn's verifier was cancelled. LM Studio logged "Client disconnected. Stopping generation…" (`2026-09/2026-09-17.1.log:150483`), and the waiting turn started the same second. Prefill still runs to completion. The cancel came about 80 s after the resubmission, for an unknown reason. | Negligible today. Dedupe (X6) removes the wait. |

Four earlier beliefs turned out wrong; see Appendix B.

## 4. "Not found"

### 4.1 What users got

Since the pin, 11 of 12 organic answers were negative. The eval set does better (11/20 on the Opus judge). Topic density explains only part of that gap:
- Some eval topics have dedicated works whose pages carry the term early: ketubbot (Friedman), Goitein, Kol Nidre (Wallenstein).
- But 6 of the 11 single-turn eval cases are festival or liturgy rows, and 4 of those 6 fail under Opus.
- Hoshana Rabbah (הושענא: 56 hits) and Sukkot (135 hits) are not lexically sparse. Only Hanukkah and Simhat Torah are.

So organic traffic fails more because it asks the festival questions the pipeline handles worst, not only because the bibliography lacks them.

Organic questions are about:
- festivals and liturgy (Hoshana Rabbah ×7, Sukkot ×4, Chanukah ×3);
- manuscripts of a specific text;
- method.

Repeats of failing questions are 33% of organic wall time. Six of the 24 are true negatives for the bibliography, e.g. Simhat Torah and a Yerushalmi passage.

### 4.2 Root causes (25 incidents, primary cause after re-check)

The **n** column gives each incident one primary cause. The last column is the verifiers' estimate of how many incidents each planned fix plausibly flips. It is not the same as n: several causes need two fixes to land together. The deduplicated expectation is in Section 5.

| Cause | n (primary) | Demonstrated example | Plan | Addressable by plan (verifier estimate) |
|---|---|---|---|---|
| **Candidate generation**: collection words plus the semantic leg pull in namesakes; alias gaps; 5 results per action | 8.5 (~5 distinct questions) | Kalman *Genesis Rabbah* took 19 of 40 read slots across 5 Hoshana Rabbah turns. "Hoshana Rabbah in the Genizah" returns 7/10 Kalman. Dropping "in the Genizah" gives 0/8 Kalman, but the slots then fill with one work (Tirosh-Becker), so phrase aliases and a per-work cap are also needed. | N5, N6, X3 | **About 3-4** (QUAL-1). The no-context-word control recovered none of P04's targets. |
| **Top-8 cut**: incomparable raw scores (lms:669-692, 785). The `semantic_weight=50` default (lms:288) turns keyword 70 into 41/59, capping the user's own hits at 0.59, while the 4B Hebrew variant's hits score 0.70. | 3 | s56 "Chanukkah", exact replay: Gil p319, in-window, at #9. P16: endecha at #9. Eval kinnot: Friedman p141 at #9. | N5 | **3** on replay. The pooling was reimplemented, so N0 must re-prove it. |
| **Subject gate / recovery**: substring matching, thin aliases, and the retry *replaces* the pool (lms:2844) | 2 (s57, s58) | s58: in-window Hanukkah pages read at #7/#8 were discarded, and the retry passed on "want"/"learn". s57: "channukkah" isn't an alias, so the answer was canned. | N6 | **2** on replay |
| *Disputed: P06* (T-S Ar. 38.11) | 1 | The two verdicts disagree and this audit did not settle it:<br>- QUAL-4: Hathaway p119, tagged T-S Ar. 38.11, ranked #1 in the first pass and the gate discarded it (a gate case).<br>- QUAL-5: the target page (Krakowski & Stern p626) was never in the first-pass pool, so the replacement lost nothing proven (shelf mark unsearchable). | N6 merge, or the X4/X8 shelf-mark path | 0-1 |
| **Router / plan shape** | 3 | s10: only one side of a comparison searched; it also has a follow-up co-cause (4.4). s53: "joins" read as SQL. P05: Schechter given an author-only search (lms:2626-2630), 0 pages. | X4 | **About 2**, only 1 of them organic (QUAL-4). Untested. |
| **Head truncation**: 1,175 Latin / 500 Hebrew chars (lms:785-797) | 1.5 to 3. This table uses 1.5; the QUAL-2 verdict says 2-3, because s35 and s44 also have the term past the cap. | Eval Purim: Smithuis at char 1,464 vs a 1,442 cap. Half of P03. | X1 | 1.5-2 |
| Unproven / unreachable | 5 | s36 and s45 (replays don't reproduce); P14 (weak); P15 = eval passover (target lexically unreachable). | — | 0 |
| Draft lost to availability | 1 | s37 | N3 | 1, if the fault recurs |

**The verifier amplifies; it does not cause.**
- Rule 5 (lms:3793-3799) treats the 8 windows as "the complete universe". 136 of 138 limitation claims passed and were filed under "✅ Verified Claims" (lms:3876). Three caveats apply to that count:
  - 89 of the 138 come from eval traffic.
  - Many of the passes are correct true negatives, e.g. s11, s41, s46, s47.
  - 31 of 120 verifier predictions did not parse, so the count covers only the parsed ones.
  
  The problem is the label's scope, not that every pass was wrong.
- Rejecting them would only soft-flag them, so relabelling is the honest fix; recall has to be fixed upstream.
- It also passed one wrong positive: in P04, Levin p68 (Rosh-Hashanah *simanim*) was read as a Hoshana Rabbah rite.

### 4.3 The other readings

- **(c) Removed content.** In eval, 5 of 20 answers lost 12 sentences after exhausting repairs, 7 of them probably correct. In organic traffic about 0 were lost; there the cost is latency.
- **(d) Links and citations.** `/document/{id}` resolves 107 of 108 links. The 24% dead-link rate occurs only on the orphan `/chat` route. The real defects:
  - a wrong "p. 119" (5 of 18 answers);
  - raw graph labels (2 of 18);
  - more than 8 references listed although only 8 are read (about 55% of answers);
  - about 5% of "Verified" entries are NOT_SUPPORTED.
- **(e) Failures.** JIT ping-pong between the unpinned 35B and the VLM cost 7 drafts on 09-22/23 (`2026-09/2026-09-22.1.log:38464`), discarding 671 s of synthesis (lms:1612-1623, 4706-4707; app.py:1484-1495). There have been 0 since the pin, but the code path is still live.
- **Scope.** The catalogue is never searched: 0 of 26 plans used it, and `primary_hybrid` always raises (lms:2688-2701). About 2 delivered answers were affected.

### 4.4 Follow-up questions (partly audited)

15 of the 62 rebuilt turns ran the 4B `followup_resolver`: 8 organic, 6 eval and 1 probe. The data is `chat_sessions.jsonl` (`aux_outputs`, `router_plan`). The resolver's output is a `FOLLOWUP: yes/no` line plus a restated `QUESTION`.

**Correct restatements: 8 turns** (s10, s11, s26, s27, s29, s33, s36, s43). s33 is a Hebrew follow-up to an English turn; it needed two resolver calls and was answered.

**Narrowing to one book: 3 turns.** In each case the router then planned an author-only `graph_scholar` search.
- s28: "…the India trade **in Letters of Medieval Jewish Traders**".
- P02: "…the India trade **in *India Traders of the Middle Ages***".
- s30: Arrant's thesis title.

No "not found" resulted: P02 was a full answer, and Opus scored the s28/s30 eval rows 8/10 on "question answered". The risk is the same mechanism as P05: evidence confined to the scholar's own pages.

**Co-causes of incidents: 2 of 25, never the sole primary cause.**
- **s54:** the resolver appended "of Genizah fragments, based on the Cairo Geniza materials". The router then searched "joining Genizah fragments", which put 1 term page in the top 8. Keyword-only "joins" gives 7/8 in-window term pages.
- **s10:** the restatement was correct, but the router's `is_followup` reasoning treated Goitein as "previously discussed" and searched only Moshe Gil.

**Misclassification: 2 turns, no recall loss.**
- s41: the resolver marked a standalone question `FOLLOWUP: yes`, but the restated question equals the input, so this was harmless.
- s42: the user typed "יקךךם", which is "hello" typed on a Hebrew keyboard layout. The resolver said `FOLLOWUP: no` but replaced the question with the user's previous Yerushalmi question, so the pipeline re-ran that true negative. s40, the same input, was searched as "ikkkem".

**Not audited:**
- the conversation history passed to synthesis (truncated in the logs);
- the effect of carrying sources over from earlier turns;
- any follow-up Hebrew failure.

**Fix (fold into X4):**
- The restated question must not add collection words or titles that the user did not write.
- A comparison must search every named side, even one discussed in an earlier turn.
- A `FOLLOWUP: no` result must keep the user's text.

## 5. Recommendations: one plan

Every item keeps the 32k context, the 15k-char / 8-slot evidence budget, keyword-first ranking, bibliography-first answers and `reasoning_effort=low`.

Backend items are prod deploys behind env flags. Tests run off the Studio, or on `:8010` with your OK. Details are in Appendix F.

### Now (this week)

| # | Change | Expected impact | Effort / risk | Test |
|---|---|---|---|---|
| N0 | GPU-free replay harness (logged quote checks; retrieval fixtures that call the prod selection code) plus a turn trace | Makes every later item measurable in seconds | S / none | — |
| N1 | The sibling batch yields to chat: a gate in `lms_transcriber.py` on `lms ps --json` status, with a 15-min cap and fail-open | SS **114 / 229**; real users 94 / 213; Goitein about 224. Residual +5-20 s (untested) | S-M / sibling throughput −1-3% (modelled) | 24 h dry run; tok/s before and after |
| N2a | **Safe quote-matcher fixes** (the subset the verifiers endorsed):<br>- strip trailing `.,;:` inside quotes;<br>- pair quote marks by position and ignore gershayim;<br>- match title-like spans against the source title/citation;<br>- exempt spans and shelf marks echoed from the user's question;<br>- no-op guard: when a repair changes no paragraph, skip the re-verify and finalize with today's exhaustion semantics.<br>Unmatched quotes still get today's LLM repair. | SS **191 / 629**; real users 191 / 416; all-interactive 163 / 368. Goitein 702 → about 375. SS loop seconds 2,770 → 1,967. Genuine catches are kept. Output is unchanged except where a quote is verbatim apart from formatting. | S-M / low | Replay the logged drafts on N0: every cleared span must be verbatim apart from formatting. Opus non-regression. |
| N2b | *Owner decision, not recommended by default.* **Blanket de-quote:** every quote still unmatched after N2a is de-quoted and soft-flagged instead of repaired; never delete a format-only failure. | On top of N2a: SS 191 / 428; real users 191 / 354; with N1 100 / 165. | S / **medium**. Known genuine catches would ship as paraphrases judged only by the 35B:<br>- the 09-28 "…between 889 and 988" date range, which is not on the page;<br>- the 09-23 and P03 Sukkot gloss inside a quotation;<br>- the eval `hebrew_ketubbot` "לשון הדייט" misquote.<br>The 35B has invented verbatim support: in P17 it passed the same date claim as "Source 8 states: '…sometime between 889 and 988'" (`2026-10/2026-10-07.1.log:65379`). | Only after N2a, and only with an Opus A/B on the misquote cases |
| N3 | Never lose finished work: draft + sources on a late failure; no in-request JIT; no identical retry; no `/chat` fallback | 0 on success; would have saved 7/7 drafts on 09-22/23 | S / low | Fault-injection unit tests |
| N4 | Progressive SSE: `sources` and provisional `draft` events, a pass counter, a SUPPORTED-only chip, the read references marked | Provisional draft at 128 s median (82 s with N1), but 47% of drafts change afterwards. The sources list (16 s) is a reading list, not an answer. | S backend, S-M frontend / low | Scratch-container frontend build (ask first) |
| N5 | Selection: weight default `None`, rank-fair pooling with a per-work cap, the Hebrew variant sanitised and capped at 2 slots | Recovers the 3 demonstrated top-8 incidents on replay | S / medium-low; shadow-log first | N0 fixtures, then an **Opus non-regression run** (eval 20 + organic topics) before the flag goes on |
| N6 | Aliases and gate: word boundaries, Hanukkah spellings and the substring bug (gt:125), Hoshana phrase forms, merge instead of replace | Flips s57 and s58 on replay | S / low | N0 + true-negative controls, then an **Opus non-regression run** |
| N7 | Honest labels and citations: "checked only against the passages read", a page-label sentinel, linkifier boundaries, no raw graph labels | Fixes "p. 119" (5/18) and graph labels; honesty, not recall | S / low | Snapshot tests on probe finals |

### Next (2-4 weeks)

| Item | What | Notes |
|---|---|---|
| X1 | Term-centred windows at the same caps | Ships with or after N5; about 2-3 incidents plus fewer false flags |
| X2 | Pre-synthesis coverage check | An ES phrase count swaps an unread term page into a term-less slot (still 8; about 10-50 ms). Negatives are labelled with counts |
| X3 | Keyword-first chat candidates | Narrow collection-word strip; lexical admission ≤15 or a concept form required; semantic only re-orders. Shadow mode, then an Opus run with n ≥ 40 |
| X4 | Plan shape | "Who is X" also retrieves pages about X. Comparisons search both sides, even one discussed in an earlier turn. The follow-up resolver must not add collection words or titles (4.4). The router returns its plan as plain-content JSON, because LM Studio drops malformed tool calls (`2026-09/2026-09-28.1.log:7264-7270`). Drop the router's `reasoning` field, which is 64% of its tokens |
| X5 | Verifier output | No rationale for supported claims, and send `none`. A sentence-ID contract only after an Opus A/B. Re-verify changed sentences only, with at most 1 repair and a deadline. Do this before X1-X3 |
| X6 | Concurrency | Dedupe identical in-flight questions; per-model locks; per-request queue position |
| X7 | Prefix-stable prompts | About 5-12 s per turn; low priority after N2a |
| X8 | Catalogue, cautiously | Fix `primary_hybrid`. A manuscript-intent trigger shows phrase-matched hits in the existing "Related catalog entries" appendix (lms:4286), outside the evidence budget |

### Later

- After N1, Opus-judged A/Bs of 35B synthesis and of a prose length budget.
- Page-number re-extraction and index v7; the Neo4j bridge query (prior #6, not re-verified); a re-ranker on the MBP only if N0 shows headroom; resumable jobs.
- Infra items from I1, not adversarially verified:
  - MLX prompt-cache churn (about 2 TB/day of writes, about 36 GB of unlinked temp files);
  - RAM at its limit;
  - the stale Docker VM;
  - the private-Wi-Fi-MAC outage.

### Expected combined effect (A1 simulation)

How N2a was modelled (`audit/rv_n2a/n2a_bundles.py`, `rv_n2a/n2a_bundles.json`):
- Each of the 15 repaired steady-state turns was classified from the diff of its logged repair (`rv_n2a/repair_triggers.py`).
- N2a drops the cycles in L59 and P06. It drops P09's cycle (a volume title present in the graph record) and the cycles after a no-op repair in L49, P04 and P14.
- It keeps the genuine or uncleared catches: L46, L52, L55, P02, P03, P07, P08, P10 and P17.
- The conservative variant also keeps P09. It moves only the median (196 s instead of 191).

| Bundle | SS median / p90 (s) | All-interactive median / p90 (s) |
|---|---|---|
| Today | 222 / 675 | 188 / 503 |
| **N2a** (safe matcher fixes + no-op guard) | **191 / 629** | 163 / 368 |
| N2b (blanket de-quote; owner decision) | 191 / 428 | 150 / 297 |
| Code bundle: N2a + N3 + lighter verifier (−40%, estimate) + router/ES cleanup | 172 / 524 | 135 / 306 |
| *Same code bundle with N2b* | *168 / 375* | *128 / 261* |
| N1 | 114 / 229 | 119 / 275 |
| **N1 + N2a** | **109 / 228** | 116 / 186 |
| *N1 + N2b* | *100 / 165* | *102 / 159* |
| **N1 + code bundle (N2a)** | **98 / 198** | 105 / 162 |
| *N1 + code bundle (N2b)* | *89 / 151* | *91 / 139* |

N2a barely moves the p90 on its own. The slow tail (P04, P07, P08, P10, L52) is genuine catches multiplied by contention. Its value is mostly in combination with N1 and X5, which also shortens the cycles it keeps.

**Expected "not found" recovery (deduplicated).**
- Inputs: the per-incident attribution (A2) and the verifier verdicts (QUAL-1: 3-4 addressable; QUAL-2: 2-3; QUAL-3: 3; QUAL-4: 2; QUAL-5: 2).
- Auditor judgement only; no Opus run.
- The 12 organic questions since the pin include 2 true or partly correct negatives (s46, s47), so at most 10 are recoverable.

| Bundle | Incidents of 25 (expected, range) | Organic questions since the pin, of 12 | Which | Confidence |
|---|---|---|---|---|
| N5 + N6 | **≈4 (3-5)** | **3**: s56, s57, s58, which are one Hanukkah question asked three times | Top-8 cut: s56, P16, eval kinnot. Gate and alias: s57, s58. P06 only if the QUAL-4 reading holds. | Medium-high. Exact replays, but the pooling was reimplemented (`a2_replay_pools.py`), so N0 must re-prove it. |
| + X1-X3 | **+≈3 (2-5), giving ≈7 (5-10)** | **+1-3, giving ≈4-6** | Joins s53/s54: keyword-only "joins" puts 7/8 in-window term pages in the top 8. Eval Purim: Smithuis is just past the cap. Half of P03. Possible only: the Hoshana Rabbah family (s44, s49-51, P04), because removing the context words recovered none of P04's targets. | Low-medium |
| + X4 (plan shape) | +1-2 | +0-1 (s53 overlaps X3) | s10, P05 | Low; untested |
| + X8 (catalogue appendix) | 0 as scholarship answers | 0-1 partial (s44 asks for fragments) | Adds catalogue hits for manuscript-intent questions. About 2 delivered sessions are affected (QUAL-10), mostly scope questions outside the 25 (s43, s39/41). | Low |
| N3 | +1 only if the 09-22/23 fault recurs | 0 | s37 | High for the mechanism; 0 losses since the pin |

Read this as: the near-term retrieval fixes repair the Hanukkah failures and the three top-8 cases. The festival and liturgy questions you ask most (Hoshana Rabbah, Sukkot) need X1-X3 and are still uncertain.

**Trade-off.** Recovered answers carry 15-35 claims instead of 2-5, which adds 15-45 s per verify pass (40-120 s contended). Ship N2a, ideally with N1 and X5, before X1-X3.

## 6. Decisions needed from Isaac

1. **Yield gate (N1).** Approve the sibling change and its dry run. Set the maximum hold. Should the sibling's own 35B judge calls yield too? Accept about 1-3% batch throughput loss.
2. **Quote failures (N2).**
   - Approve N2a: the safe matcher fixes plus the no-op guard. It keeps today's behaviour for genuine misquotes.
   - Separately, decide on N2b, the blanket de-quote with a soft flag instead of an LLM rewrite. It would cut the SS p90 from 629 to 428 s (165 s with N1). The cost: real misquotes ship as flagged paraphrases that only the 35B checks, and the 35B invented verbatim support for an unsupported date range in P17.
   - If N2b is wanted, gate it on an Opus A/B over the known misquote cases (09-28 ketubbot, 09-23 / P03 Sukkot gloss, eval `hebrew_ketubbot`).
3. **Provisional draft (N4).** Show it, given that about 47% change afterwards? If a later step fails, keep it visible as "unverified"?
4. **Verifier.** Keep the 35B; drop rationales for supported claims; send `none`; allow at most 1 repair plus a deadline.
5. **Thinking.** Synthesis stays at `low`. After N1, run an Opus A/B of 35B synthesis and of a prose length budget?
6. **Catalogue.** Start with appendix-only hits for manuscript-intent questions?
7. **Deploys.** Three flag-guarded backend rebuilds, one frontend build, and `:8010` runs in quiet hours.
8. **Housekeeping.** The stale Docker VM, the MLX disk cache, and dropping `ttl` from prod payloads (lms:1853-1854).

---

## 7. Appendix

### A. Live probes (P1; tonight, sequential)

| ID | Question | Wall s | Draft at s | Verify passes | Repairs | VLM busy | Outcome (after re-check) |
|---|---|---|---|---|---|---|---|
| P01 | Who is S.D. Goitein | 317 | 215 | 1 | 0 | 1.00 | full; reproduced your chips (8 bibs, 1 graph, 31 verified); read only preface and *Letters* front matter |
| P02 | Follow-up: India trade | 425 | 171 | 2 | 1 | 1.00 | full |
| P03 | Sukkot | 573 | 330 | 2 | 1 | 1.00 | partial false negative (retrieval + truncation) |
| P04 | Hoshana Rabbah | 720 | 227 | 4 | 3 | 1.00 | false negative (5/8 Kalman namesakes) + verified wrong positive (Levin p68) |
| P05 | Who was Solomon Schechter | 193 | 129 | 1 | 0 | 1.00 | false negative: author-only search, 0 pages |
| P06 | Scholars on T-S Ar. 38.11 | 208 | 103 | 4 | 0 (3 no-op cycles) | 1.00 | false negative: shelf mark unsearchable; gate turned the topic into "scholars" |
| P07 | תשובות הגאונים (Hebrew) | 616 | 294 | 2 | 1 | 1.00 | full |
| P08 | Ben Yiju letters | 680 | 314 | 2 | 1 | 1.00 | full |
| P09 | Mark Cohen on poverty | 292 | 121 | 2 | 1 | 1.00 | true negative (books not indexed); box-level links |
| P10 | Fragments Goitein discusses for India trade | 630 | 262 | 2 | 1 | 1.00 | full; 1 truncation false flag; 1 corrupted link |
| P11 | Which synagogue | 161 | 112 | 1 | 0 | 1.00 | full; core claim flagged (verifier strictness) |
| P12 | Yom Kippur piyyut literature | 256 | 228 | 1 | 0 | 0.82 | full |
| P13 | Tu B'Av (control) | 66 | 53 | 1 | 0 | 0.00 | true negative |
| P14 | Purim fragments | 237 | 95 | 4 | 3 | 0.00 | weak partial negative |
| P15 | Passover seder | 95 | 76 | 1 | 0 | 0.00 | negative; target lexically unreachable |
| P16 | Tisha B'Av kinnot | 131 | 98 | 1 | 0 | 1.00 (one v22i-a50 request, 04:05:10 to about 04:07) | false negative: endecha at #9 (top-8 cut) |
| P17 | Ketubbot overview | 188 | 118 | 2 | 1 | 0.00 | full |
| P18 | Goitein rerun (quiet GPU) | 124 | 82 | 1 | 0 | 0.00 | full; A/B with P01: 317 → 124 s |

Totals:
- median wall 246 s;
- 10 of 18 repaired;
- 0 failures;
- `queued_behind` was 0 throughout.

Whole-call speeds, VLM busy (13 probes) vs idle (5):
- synthesis: 8.2 vs 19.0 tok/s;
- verifier: 19.7 vs 77.7 tok/s.

### B. Verification ledger

**Earlier beliefs that turned out wrong:**
- There is no hard 300 s timeout: a 334 s call completed on 09-28.
- The status labels are forward-mapped on purpose (lms:4642-4654).
- The 27B's MLX prompt cache is per process and is not shared with the VLMs.
- `reasoning_effort=low` sent to the 35B falls back to "on" and does nothing (`2026-10/2026-10-07.1.log:17008`).

| Finding | Verdict | Main correction |
|---|---|---|
| LAT-1 | confirmed | "Every call" should read "calls that overlap the VLM". The ~8 tok/s figure applies only during VLM prefill; during VLM decode it is about 32. The batch is a rotating set of JIT checkpoints, so its duty cycle varies. |
| LAT-2 | partial | Labels are forward-mapped on purpose; the gap is wording plus a missing pass counter. Backend queue waits are already shown; GPU contention is not. The draft + F1 figure (82/118 s) is modelled. |
| LAT-3 | partial | The repair rate depends on the question mix: 18% for organic turns since the pin vs 56% for the probes. About half the triggers are matcher false positives; the other half are real non-verbatim quotes. After N1, each cycle takes about 50-60 s. |
| LAT-4 | partial | The −65% is assumed; only −17% was measured (n=1-2). The 8,192-token truncations were eval-only. The claim echo is reused for sentence anchoring and repair. |
| LAT-5 | partial | Answers are not long: median 446 answer tokens, organic 178. The 16,384 cap never binds. The bandwidth ceiling is about 80% of theoretical, not 73%. The 21% "clean" share is this finding's own; the rest is LAT-1. |
| LAT-6 | partial | The cache is per process. It reuses 256-token chunks at checkpoints, and 1,280 tokens were restored on 09-20. Only 6.4% of prefill is reachable by reordering. |
| LAT-7 | partial | 7 of the 9 turns inside the incident window were lost; 7 of 38 counts all interactive turns. Parse-retry hit 8 of 105 verifier calls in 4 turns, 2 of them interactive. The /chat fallback was not observed. |
| LAT-8 | partial | A backend lenient parse is infeasible. The real saving (about 6 s) comes from ES batching and async calls. The router's own verbosity is the larger lever. |
| LAT-9 | partial | Both overlaps were duplicate submissions. A separate 4B lane is the wrong fix; per-model locks and dedupe are better. `queued_behind` is a global count. |
| QUAL-1 | partial | Addressable: about 3-4 of 25. The router mostly copies the user's "Genizah" rather than adding it. The 41/59 default is a co-cause. "Chanukkah" fails because of the substring bug in `expand_query_aliases`. |
| QUAL-2 | partial | Primary in 2-3 of 25, not 26%. s35/s37 were availability, s49-51 retrieval. Several hidden "answers" are passing mentions. |
| QUAL-3 | partial | Demonstrated in 3 of 25 (s56, P16, kinnot). s44 was namesakes plus truncation; s36/s45 are unproven. The Hebrew variant sometimes rescues answers (s58). |
| QUAL-4 | partial | 3 of 25 (s10, s53, P05). P06 belongs to the gate and the shelf-mark path. The other co-cause claims do not hold. |
| QUAL-5 | partial | 2 of 25 (s57, s58). Of the "false pass in ≥9", only 2 caused harm. Judging re-plan terms as proposed is risky. For P06 it found no proven loss, because the target was never in the first-pass pool. That conflicts with QUAL-4 (Hathaway p119 at #1, discarded), so 4.2 shows P06 as disputed. |
| QUAL-6 | partial | A labelling and trust problem; it does not change content. At most 21 of 25 incidents reached the verifier. The relabel recovers 0 incidents. |
| QUAL-7 | partial | Correct organic deletions: about 0. Waste is about 30-35% of organic verify time, not 54%. Some repairs were genuine catches. Blanket "never delete supported" is unsafe. Hence the split into N2a and N2b. The re-simulation agrees: N2a removes 29% of SS loop seconds. |
| QUAL-8 | partial | One incident, one root cause (JIT ping-pong). Weak evidence for the 300 s timeout. 0 losses since the pin. |
| QUAL-9 | partial | The 24% dead-link figure applies only to the orphan /chat route. P10 makes one corrupted link, not two. References overstate what was read in about 55% of answers. |
| QUAL-10 | partial | Negatives are not worded corpus-wide. About 2 delivered sessions were affected. A noisy catalogue keyword search needs phrase matching. |
| QUAL-11 | partial | Misreads come from namesake retrieval priming, not the window. Effect on found/not-found is about 0. A deterministic festival/name check is the useful fix. |
| QUAL-12 | partial | The Purim oracle is correct (בפורים at offset 1,152). The harness must call prod code, pin router actions, and record the index version. |

### C. Data files produced (all under `audit/`)

- **Reviser additions:** `rv_n2a/repair_triggers.py` (diff and reasoning of every steady-state repair) and `rv_n2a/n2a_bundles.py` → `rv_n2a/n2a_bundles.json` (N2a what-ifs).
- **Logs → turns:** `parse_lmstudio_logs.py` produces `llm_calls.csv` (41,365 calls; 357 in chat turns), `chat_sessions.jsonl` (62 turns), `tg3_samples.jsonl` and `vlm_busy_summary.json`. Also `i1_*` (infra) and `L1_stats.json`.
- **Attribution:**
  - `a1/a1_attribution.py` → `a1_attribution.json`, `a1_extras.json`, `a1_mechanisms.json`, `a1_tables.md`;
  - `d/d_bundles.py` → `d_bundles.json`;
  - `a2/a2_replay_pools.py`, `a2/a2_probe.py` and their outputs.
- **Probes:** `probe_client.py`, `live_probes.jsonl`, `p1/finals/`, `p1/events/`, `p1/tables.md`, `p1/negative_checks.json`.
- **Replay bench:** `p2_bench.py` and `p2_raw/` (per-request JSON, `summary.json`, `verdict_agreement.json`, `synthesis_quality.json`).
- **Forensics:** `q1_sessions/`, `c3/`, `c4/`.
- **Verifier checks:** `vlat3_repair_diffs.py`, `vlat4/`, `vlat6/`, `v_lat8/`, `v_qual1/`, `v_qual2/`, `v_qual3/`, `v_qual5/`, `qual10_verify/`, `rv_qual4_*`, `v6_limitation_count.py`, `v11_window.py`.
- **Lead notes:** A1, A2, C1-C4, I1, L1, P1, P2, Q1, and the three `D_*.md` roadmaps.

### D. Not verified

- **MBP environment values** (`LM_STUDIO_REQUEST_TIMEOUT`, `LM_STUDIO_MAX_CONCURRENCY`, `EVIDENCE_CHAR_BUDGET`) were inferred from the logs.
- **Systems not reachable from here:**
  - MBP backend and access logs, including /chat route usage;
  - Neo4j: the prior #6 bridge and the Outhwaite graph claim;
  - prod ES analyzers: the Hebrew-prefix gap is shown only behaviourally.
- **Catalogue counts** come from local 10-06 dumps.
- **Modelled, not measured:**
  - the yield-gate residual;
  - `lms ps` status behaviour mid-generation;
  - the −40% and −65% verifier cuts;
  - the early-exit and cache savings.
- **Approximate replays:** the 09-23 replays (s44, s45) run against today's index.
- **N2a cycle classification** comes from reading each repair's diff, which is auditor judgement. L8, L10 and P09 are the least certain; the conservative variant in Section 5 keeps P09.
- **Quality-bundle estimate** (Section 5) is auditor judgement over the verifier verdicts, with no Opus run.
- **Follow-up path** (Section 4.4): the synthesis-side conversation history and source carry-over were not audited.
- **Hebrew-language questions:** one probe (P07) only.
- **Not run:** no Opus judge.
- **Partly checked:** client disconnect. LM Studio cancels decoding on disconnect but finishes prefill first. One case, on 09-17, shows the cancel reaching LM Studio through the whole chain (LAT-9). Whether every browser abandonment propagates through Cloudflare and the backend is unverified without the backend logs.
- **Not checked:**
  - the localStorage-quota crash risk (C4);
  - speculative-decoding support.

### E. Audit footprint and incident

**Traffic sent:**
- 18 chat turns;
- 16 LM Studio requests, to loaded models only;
- several dozen read-only search requests;
- 108 link GETs;
- read-only `lms ps`.

Nothing was changed in containers, launchd, models, the repo or the sibling project.

**Incident.** P2 request #11 sent the verifier's strict `json_schema` to `qwen/qwen3-4b-2507` and segfaulted its MLX engine at 04:41:31. The watchdog re-pinned it at about 04:43:52. No prod request arrived in that window. Rule: never send `response_format: json_schema` to the 4B.

### F. Implementation notes per plan item

- **N0 harness.**
  - (a) Stream the logs and extract each logged verifier prompt's draft and "Original page text (quoteable)" (emitted at lms:822). Run old vs new `extract_direct_quotes`/`find_quote_source` over them: 47 repair cycles, 52 drafts, Goitein at `2026-10-07.1.log:~16180`.
  - (b) Build retrieval fixtures from read-only public searches. They must call the prod pooling, cut and window functions; `a2_replay_pools.py` reimplements them, so it would miss regressions. Pin router actions and the index version.
  - Assert must-read-in-window for:
    - Gil p319 and Hornkohl p232 (Hanukkah);
    - Gottheil p200 (Hoshana Rabbah);
    - Arrant p109 (joins);
    - Gutwirth endecha and Friedman p141 (kinnot);
    - Vorderstrasse p115 (Schechter);
    - Krakowski & Stern p626 (T-S Ar. 38.11);
    - Levin p52 and Smithuis (Purim).
  - Controls: Simhat Torah and מעשה רוקם.
  - Mark past-window targets as expected-fail: Levin p44 at char 1,594, Friedman JMP p52 at 3,132, India Book p410 at 1,619.
  - (c) Add to `record_llm_call`/`_build_metrics` (lms:50-73, 1674-1712): `finish_reason`, decode tok/s, a contended flag, `time_to_sources`, `time_to_draft`, passes, no-op repairs and sentences deleted.
- **N1 yield gate.** In historical-document-analysis `src/models/ocr/lms_transcriber.py`, at both `async with _get_lms_lock()` sites (about :275 and :364):
  - Poll `~/.lmstudio/bin/lms ps --json` read-only every 3-5 s.
  - Hold while `qwen/qwen3-4b-2507`, `qwen/qwen3.8-27b` or `qwen/qwen3.6-35b-a3b` is processingPrompt, generating or queued. Use `lastUsedTime` within 120-180 s only to bridge gaps between calls, because it updates when a request *ends*.
  - Exclude the sibling's own in-flight 35B judge calls (tracked with a lock file).
  - Cap at 15 min; fail open on any `lms ps` error; log every wait.
  - The fcntl file lock shares only the gate decision; it must not serialise sibling concurrency.
  - Modes: `LMS_YIELD_MODE=off|dryrun|on`.
  - Bypass paths to cover later: `probe_forgetting.py`, `confusable_probe_fonts.py`, `talmud_evaluation.py:813`.
  - The 2026-10-05/06 private-Wi-Fi-MAC outage shows the Studio network also needs a fixed MAC or a DHCP reservation.
- **N2a quotes (safe subset).**
  - In `extract_direct_quotes` (lms:887-910), pair quote marks by position: an opening mark after the start, a space, `(`, `—` or `:`; a closing mark before a space, punctuation or the end.
  - Ignore word-internal `\w"\w` (ASCII gershayim, as in ע"ו). Replace the `{8,}` skip with "ignore spans under 3 words".
  - In `find_quote_source` (lms:926-953), strip leading and trailing `.,;:!?…'"`.
  - Treat a span that matches a source's title or citation line as supported by that source. This includes work titles listed in a graph record, as in P09.
  - Exempt spans and shelf marks copied from the user's question.
  - Quotes that still fail stay hard failures and get today's LLM repair. Shelf marks are compared after `shelfmark_normalizer`.
  - Replace `workflow.add_edge("repair_answer","verify_claims")` (lms:1756) with a conditional edge. When the repaired paragraphs hash-equal the originals, or every rejection was unanchorable (lms:3547-3571), go straight to finalize. Keep today's exhaustion semantics there, so the output is identical and only the redundant passes go.
  - Add synthesis rule 1 (lms:3297): punctuation goes outside closing quotes, and titles are italicised, not double-quoted.
  - Unit cases:
    - `"India Book."` vs `'India Book,'`;
    - the Kol-nidré title span;
    - `"הכפישה"` mis-pairing;
    - `ע"ו`;
    - the comma-inside-quote ketubbot quote, where the dates must still fail;
    - the P09 volume title;
    - the P06 shelf mark echoed from the question.
  - Log per turn: cleared spans by reason, no-op stops.
- **N2b blanket de-quote (only if approved).**
  - Run the check before the verifier prompt is built (lms:3709-3735). Every quote still unmatched after N2a is de-quoted in the draft and gets a soft flag instead of an LLM repair.
  - In the finalizer (lms:4225-4258), never delete a sentence whose only failure is a quote mismatch.
  - Flag: `QUOTE_FAILURE_MODE=repair|dequote`. Default `repair`.
  - Gate on an Opus A/B over the 09-28 ketubbot, 09-23 / P03 Sukkot and eval `hebrew_ketubbot` cases. The 35B judge alone is not enough: see P17, `2026-10-07.1.log:65379`.
- **N3 availability.**
  - Catch `ModelUnavailableError` and transport errors around `_call_verifier` (lms:3824) and in the re-verify path. Route them to the existing `verification_error` soft flag (ChatUI.jsx:1362).
  - In `chat_stream` (lms:4700-4707), send a normal `final` with the draft once `draft_answer` exists. Only failures before the draft reach app.py:1484-1495, which can still return sources.
  - Add `LM_STUDIO_ALLOW_JIT_LOAD=0` for the verifier in `resolve_model` (lms:1581-1623). Drop `ttl` for the prod models (lms:1853-1854).
  - `_call_llm` returns `finish_reason`. `_call_verifier` (lms:4074-4098) must not repeat a temperature-0 call after `length`. The observed runaways looped inside the free-text `reasoning` field, so cap or drop that field, or set `max_tokens` to about 4k (the observed maximum since 09-20 is 3,510).
  - Remove the `/chat` fallback (ChatUI.jsx:1028-1062, 1209-1243), or allow it only before the first event. Treat a stream that ends without a `final` as an error.
- **N4 SSE.**
  - `state['bibliography_results']` and `state['draft_answer']` are already in `updates` when those nodes finish (lms:4709-4733), so the backend events are cheap. The current frontend ignores unknown event types, so the backend can ship first.
  - Count SUPPORTED only in `verified_claims_count` (lms:4732). Group the panel by status (ChatUI.jsx:1390-1413). Mark the read references (ChatUI.jsx:1465-1475).
  - For the GPU notice, use the tok/s of the last call compared with the model's norm.
- **N5 selection.**
  - In `SearchAction` (lms:288-289) and the router schema (lms:2251-2252), drop the defaults of 50.
  - Tag each result with `action_index` and `action_rank` in `_run_bib_search` (lms:2560-2584). Round-robin across actions in `deduplicate_bibliography_results` (lms:669-692), user-language actions first. Cap at most 2-3 read slots per work and at most 2 for the variant.
  - In the Hebrew variant (lms:2408-2477), reject Latin or Arabic letters and question forms.
  - Raise `num_results` only together with X3 admission. In s56 the answer was already at #9 and `[:8]` would still drop it.
  - Flags: `POOLING_MODE`, `HEBREW_VARIANT_MODE`.
- **N6 aliases and gate.**
  - `find_concepts` (gt:94) uses word-boundary regex.
  - `expand_query_aliases` (gt:125) must not skip "hanukkah" just because it is a substring of "chanukkah".
  - Add channukah, channukkah, chanuka and hanukka.
  - Add a `hoshana_rabbah` concept: hoshana rabbah/rabba, hosha'na rabbah, hoshanot, הושענא רבה, הושענה רבה, יום הושענא/הושענה, הושענות. Remove "hoshana" from sukkot (gt:66).
  - Stop list (lms:521-531): want, learn, contain, mention, discuss, scholars. Keep shelf-mark tokens (lms:550-553).
  - Recovery (lms:2844, 2911, 2925): merge concept-bearing first-pass pages ahead of retry pages, and judge against the original concept's forms, including fuzzy-mapped ones.
- **N7 labels and citations.**
  - lms:3876 wording. Rule 10 wording ("in the scholarship consulted here"). Link the site's catalogue search instead of external indexes.
  - Page sentinel at lms:803-812: omit the page when the header says "Page: 0", or when more than one page of a work in the pool shares a number.
  - Linkifier (lms:4441-4458): add a negative lookahead `(?![\s.]?\d)`, and never match inside an existing `(doc:…)`.
  - Convert underscore graph labels (`T_S_10J16_15` → `T-S 10J16.15`) before they enter `shelf_mark_lookup` (lms:2613-2624) or the prompt.
  - The orphan `/chat` route: redirect to `/`, or pass doc ids (react_app.jsx:1646-1650).
- **X1 windows.**
  - In `build_bibliography_source_context` (lms:766-836): a 150-200-char head stub, then a window starting about 20% of the cap before the densest curated concept/alias cluster.
  - Anchors come from the alias table, never from router queries or the 4B variant.
  - Avoid name collisions (Esther vs "Esther-Miriam Wagner"). Pages with no hit keep the head window.
  - Shrinking the summary (lms:813-817) and lowering `per_source_fixed_chars` (lms:790) is a separate A/B.
  - The same window feeds `quoteable_text` (lms:834), the verifier (lms:839-862) and `find_quote_source`.
- **X2 coverage.**
  - After the gate (about lms:2862): one ES search (`size` 5, `track_total_hits`, `match_phrase` on concept/alias/prefix forms, `_source_excludes: embedding_vector`).
  - Swap up to 2 term pages into term-less slots. Record `index_hits` and `read_hits`, and a LIMITED_TO_READ citation with the counts.
  - Needs N6 forms first; with today's substring matching it would swap in divorce pages via "get".
- **X3 keyword-first.**
  - New `chat_retrieval.keyword_first_search`, used by `_run_bib_search` when `CHAT_CANDIDATES=kwfirst`.
  - Lexical: `multi_match` without author fields; phrase boosts; prefix forms.
  - Semantic: `script_score` filtered to the admitted ids instead of `match_all` (sb:416-430).
  - Public `sb.search_hybrid` stays untouched.
- **X4 plan shape.**
  - Use Neo4j scholar resolution for person detection.
  - For "who is X", cap the author-constrained search at 4 slots and demote front matter. Run the full question as hybrid with `must_not` on X's author variants, minding co-authored works.
  - Ask for the router plan with `tool_choice` none and parse it with `extract_json_object` (lms:1031).
  - Reject mixed-script tokens, not every Arabic query: Judeo-Arabic is a legitimate topic.
- **X5 verifier.**
  - Schema (lms:4015-4060): `reasoning` is "" for supported claims and ≤20 words otherwise.
  - Then `{id, verdict, source_number, note}` per sentence, with a "partial" verdict carrying the offending span.
  - Carry verdicts forward by sentence hash. Set `MAX_VERIFICATION_REPAIR_ATTEMPTS` to 1 (lms:232). Use `VERIFY_DEADLINE_S`.
- **X6 concurrency.** Replace the single semaphore (lms:76-143) with per-model locks. The memory cost is about 0 for the 35B (its KV pool is preallocated) and about 0.5 GB for one extra 27B request. Report a per-request queue position, not the global waiter count (ChatUI.jsx:1534).
- **X7 prefix.**
  - Use one evidence join (synthesis uses `\n`, repair and verify use `\n\n`).
  - Put shared evidence first and the per-call parts last. Move the verifier's exclusion note after SOURCE CHUNKS.
  - Check `Prompt cache restore cached_tokens` before and after.
- **X8 catalogue.**
  - Fix `primary_hybrid` (lms:2688-2701): build the request model with `semanticWeight`/`keywordWeight`, or call `search_by_keyword`.
  - Detect manuscript intent deterministically (manuscript(s), fragment(s), כתב יד, קטע, "do you have").
  - Run one phrase/AND catalogue query; plain OR + fuzzy returns noise such as "ספר העיתים" for הר הזיתים.
  - In-prompt records (≤2 × 300 chars, labelled "Catalogue record (metadata, not scholarship)", 8 → 7 bibliography slots) only after you review samples.
