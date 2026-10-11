# Handoff: chat verify loop, Studio follow-up (2026-10-11)

For a fresh Claude Code session on the **Mac Studio**. Read this file, `CLAUDE.md` and
`docs/SHARED_RUNTIME.md` first. Production runs on the M3 MBP (Docker + cloudflared). **LM Studio on
the Studio is production inference** — `127.0.0.1:1234` there is prod.

## 1. What is already done (branch `chatbot-fixes`)

Built and reviewed on the MBP on 2026-10-11, from the 2026-10-10 handoff
(`CHAT_VERIFY_LOOP_HANDOFF_2026-10-10.md`, items C1-C7). Nothing is deployed.

| commit | what |
|---|---|
| `8590e96` | Deterministic quote check before the 35B (`src/backend/quote_check.py`, graph node `check_quotes`), guarded smoothing, closing note, no-op repair guard, verifier-outage loss-proofing, stream backstop, `LM_STUDIO_ALLOW_JIT_LOAD`, no retry after `finish_reason=length`, metrics |
| `aa3ee0d` | `evals/replay_quote_check.py` — offline replay over the audit drafts (run it here, §2) |
| `c29c525` | ChatUI: no silent `/chat` re-run once a stream started; Retry button (separate commit; frontend rebuild = prod deploy) |
| `63363a3` | Repair KEEP protocol: the 27B must revise a contradicted claim or KEEP it with a verbatim source quote that is checked deterministically; unchanged sentences are removed by offset |

Isaac's decisions folded in (beyond the 10-10 handoff):

- **Not strict on spelling:** Hebrew ו/י after a word's first letter are ignored when matching quotes
  (והגהתי = והגהותי); a quote may omit a source's standalone 1-2 digit footnote marker (l46). A quote
  that adds or changes a number still fails.
- **KEEP protocol** (option 1). Blunt delete-and-smooth for contradicted claims was rejected:
  coherence over speed.
- Deferred ideas (not started): anchor-quote splicing (model cites source# + start/end words, code
  splices exact text); repair as a KV-cache continuation turn (§4).

Tests: 392 pass; 2 failures pre-date this work and are unrelated
(`test_synthesis_receives_graph_evidence_with_provenance_rules`,
`test_works_cited_links_do_not_depend_on_italic_titles`).

## 2. Run on the Studio (read-only, no LM Studio traffic)

**Unit tests** — never bare `pytest`. Keep the LM Studio blackhole: `lms_agentic_search` loads
`src/backend/.env` on import (no override of existing env), and on the Studio `127.0.0.1:1234` is prod.

```
LLM_STUDIO_URL=http://127.0.0.1:9 WEAVE_ENABLED=false WANDB_MODE=disabled PYTHONPATH=. \
  <venv>/bin/python -m pytest -q -p no:cacheprovider \
  src/backend/tests/test_quote_check.py src/backend/tests/test_graph_rag_phase1.py \
  src/backend/tests/test_pipeline_metrics.py src/backend/tests/test_judge_v2_and_language.py \
  src/backend/tests/test_shelfmark_linkification.py src/backend/tests/test_multiturn_rag.py
```

**Replay** over the audit data (`evals/results/rag_audit_20261007/` in Isaac's main checkout):

```
PYTHONPATH=. <venv>/bin/python evals/replay_quote_check.py <path>/rag_audit_20261007/p2_raw
```

Expected with Isaac's leniency decisions: **exactly one** unverified quotation (the invented E2d
"אין לסלוח על כך… יתלצו ויתלעסו הרוסים"); E2b's spliced ellipsis quote verified; the P07 spelling
variant now verified. If the E-draft pairing guesses wrong, pass `--pair E2x=p07`. Report any other
unverified item to Isaac with the source text — it is either a true catch or a false deletion to fix.

## 3. Live checks (each needs Isaac's explicit OK; quiet hours; one request at a time)

1. **KEEP protocol compliance.** Does the 27B return `keeps` with verbatim quotes, and how often is a
   flagged sentence force-removed vs kept vs rewritten? Watch `metrics["repair"]`
   (`keeps_verified`, `keeps_rejected`, `forced_removals`, `deterministic`).
2. **Smoothing on the real 35B** (`reasoning_effort=none`): rate of `applied` vs `rejected:<reason>`
   in `metrics["quote_check"]["smoothing"]`; any `rejected:length` (max_tokens 1024 for up to 3
   paragraphs).
3. **Opus non-regression eval** (optional; §6 of the 10-10 handoff): ask Isaac for the prod model env
   values and which ES/Neo4j/embedding a dev backend on the Studio should use. Never start a dev
   backend with code-default models (the code default `VERIFICATION_MODEL` is the 4B; smoothing
   refuses it, but other calls would still JIT-load).

## 4. KV-cache experiments (Isaac wants these)

LM Studio reuses the longest common prompt prefix with the previous prompt **to the same model**.
Fixed instructions already come first (synthesis ~1.5k tokens, verifier ~0.7k), but any other prompt
to that model in between evicts the cache (the 27B repair call; on the 35B, the new smoothing call
right before the verifier; concurrent chats). Measured 10-10: 27B follow-up turn reused 96% of the
prompt (TTFT 44 s → 18 s); 35B follow-up re-check 9 s vs 100-250 s for a full re-verify.

1. Check the LM Studio server logs (`~/.lmstudio/server-logs/`) for how many prompt tokens prod
   synthesis calls actually reuse.
2. Time a synthesis call right after another synthesis vs right after a repair.
3. Prototype the KEEP repair as a **follow-up turn** of the synthesis conversation (same
   `reasoning_effort`, which Qwen3.8's template writes at the very top of the prompt). The repair
   prompt already puts the numbered evidence first so this is a small change.

## 5. Deploy (Isaac's call; MBP)

- Backend: image rebuild on the MBP = prod deploy. New env flags (`.env.example` documents them):
  `QUOTE_FAILURE_MODE=delete` (default; `repair` = rollback without rebuild),
  `QUOTE_SMOOTHING_MODEL` (unset = `VERIFICATION_MODEL`; empty = off; a 4B is always refused),
  `QUOTE_SMOOTHING_REASONING_EFFORT=none`, `LM_STUDIO_ALLOW_JIT_LOAD` (recommended `0` in prod; the
  Studio watchdog re-pins the models), optionally `LM_STUDIO_MODEL_TTL=0`.
- Frontend (C7): container rebuild, separate decision.
- After deploy, watch in metrics / the `Quote check:` log line (`docker logs genizah_search-backend-1`
  on the MBP): `quote_check.unverified` and `sentences_deleted` (most turns 0), smoothing
  `rejected:*` rate, `answer_too_short`, `noop_repair`, `repair.*`, `verification_unavailable`,
  `finish_reason` length counts, and the share of chat seconds in `repair_answer` (was ~27%).
  Spot-read every turn with `sentences_deleted > 0` for the first week.

## 6. Open decisions for Isaac

1. **N3 crash handling** (10-10 §2): keep draft + sources under the "couldn't be double-checked"
   banner, no in-request JIT load, no identical retry after length. Implemented; confirm.
2. **`answer_too_short`** (10-10 C2 step 5): when quote deletion leaves < 200 chars, the code skips
   the verifier and returns the fallback, which in practice is the generic failure text (no verify has
   run yet). Recommended change: only when nothing survives; otherwise the short remainder goes to the
   35B and ships with the closing note. One-line change plus test 18's first case. Pending Isaac.
3. Push / PR / the big merge of `chatbot-fixes` with the SEO work: Isaac's call.

## 7. Known limits (documented in `quote_check.py`)

- Single-quoted quotations and quotations under 3 words are unchecked.
- Digitless shelf-mark captures (e.g. "L-G Arabic 1.99" captured as "L-G Arabic") are left to the 35B.
- A number truncated exactly at a quote/ellipsis edge ("…numbered 700" vs 7000) still verifies.
- A ו/י standing for a source prefix letter verifies (ימלך vs המלך).
- A KEEP only proves the quote exists in a source, not that it supports the claim; the sentence then
  ships highlighted as disputed.
- ChatUI: error turns are still sent as conversation history on later questions (pre-existing).
