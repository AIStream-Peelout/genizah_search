# Synthesis model comparison — analysis (2026-08-19)

Run: `qwen/qwen3.6-35b-a3b` (current prod, MoE, 3B active) vs `qwen/qwen3.6-27b`
(dense, 4-bit MLX) as the **synthesis** model only; router `qwen/qwen3-4b-2507`
and verifier `qwen/qwen3.6-35b-a3b` unchanged for both. Datasets:
`agentic_rag_v1` (11 single-turn cases) + `agentic_rag_multiturn_v1` (5 cases).
Judge: `qwen/qwen3-4b-2507` at ctx 32768. Both candidates loaded at ctx 32768.
Model-major order (35B first, 07:34–08:00 UTC; 27B 08:00–09:00 UTC). Raw data:
`summary.md`, `summary.json`, per-run `*.jsonl` (full responses + metrics).

## Headline

| | 35B-A3B (prod) | 27B dense |
|---|---|---|
| Deterministic pass (single / multi) | 7/11 · 5/5 | 7/11 · 5/5 (same cases) |
| Judge mean (single / multi) | 3.97 · 4.00 | 3.91 · 3.97 |
| Judge pass rate | 0.64 · 1.00 | 0.64 · 1.00 |
| Critical failures | 0 | 0 |
| **Latency mean** (single / multi) | **74 s · 89 s** | **171 s · 278 s** |
| Synthesis throughput | ~80 tok/s | ~23 tok/s |
| Synthesis completion tokens / case | ~3.5k (92 % reasoning) | ~3.4k (91 % reasoning) |
| Verification cycles (total, 16 cases) | 27 | 21 |
| Repair attempts (total) | 11 | 5 |
| First-pass verification rate (single / multi) | 0.64 · 0.40 | 0.91 · 0.60 |
| Flagged claims (total) | 1 | 2 |

**Quality is a statistical tie.** Same deterministic outcomes (the four single-turn
failures — purim, passover seder, tisha b'av ×2 — are retrieval/routing misses,
independent of the synthesis model), judge means within 0.06, no critical failures
either way. The judge is saturating near 4.0 (a 4B judge), so it has little
discriminative power; the 27B's deductions were citation-style nits (a
near-verbatim quotation, a graph count cited as if from page text).

**Speed is not a tie.** The 35B-A3B is ~2.3× faster end-to-end (synthesis
728 s vs 2527 s summed over 16 cases) because the dense 27B generates at ~23 tok/s
vs ~80 tok/s for the MoE. The 27B's worst case (multi-turn "verses" follow-up)
took 619 s with 4 verification cycles and 3 repairs.

**The 27B writes more verifiable drafts.** 21 vs 27 verification cycles and 5 vs 11
repairs; on single-turn cases it passed verification first time 91 % vs 64 %.
That partly offsets its slower generation (verification is the 35B in both cases:
395 s vs 526 s summed — the 27B's longer answers cost verification time too).

## The dominant cost is thinking, for both models

~91–92 % of synthesis completion tokens are reasoning tokens (LM Studio
`completion_tokens_details.reasoning_tokens`): ~3.1k–3.3k reasoning tokens per
case for ~300–400 tokens of answer. Synthesis is ~60 % (35B) to ~77 % (27B) of
wall time. Disabling or capping thinking for the synthesis call (Qwen3.6 supports
non-thinking mode / `enable_thinking: false`, or a reasoning budget) is the single
biggest latency lever available — potentially a 5–8× reduction in synthesis time
for either model — and should be evaluated with this same harness before any
distillation/LoRA decision, since it changes what the student would learn.

## Recommendation

- **Keep `qwen/qwen3.6-35b-a3b` as the production synthesis model**: equal quality,
  2.3× faster, already resident/pinned.
- **Next experiment (cheap, high value)**: rerun this comparison with thinking
  disabled/capped for synthesis (and separately for verification, which is also
  the 35B). If quality holds, prod latency drops dramatically.
- **Distillation / LoRA**: the 27B's higher first-pass verification rate suggests
  its drafts are a better *teacher* signal for "verifiable synthesis" than the
  35B's; but its raw speed makes it unattractive as a deployed student. A
  plausible target: a small dense model (e.g. 4B–9B) LoRA-tuned on verified
  35B/27B answers with thinking off, judged by this harness (judge + verification
  cycles + tok/s). Training data is already in the JSONL (answers, evidence,
  verification outcomes, flagged claims).

## Caveats

- The box was shared: the Hebrew VL checkpoint audit was running on LM Studio
  during the run (and prod chat can arrive at any time). Affects absolute
  timings, in principle equally for both legs; relative numbers are robust.
- n = 16 cases; judge is a 4B model near ceiling. Treat ±0.1 judge differences as
  noise. Deterministic checks are the more reliable quality signal here.
- Two 35B rows (Goitein cases) were judged after the run via
  `scripts/rejudge_eval_results.py` because the unbounded graph evidence
  (~224k chars) overflowed the judge context; the evidence bounding was fixed
  (`bounded_graph_result`) before the 27B leg, so the 27B was judged live with
  the same bounding the re-judge used.
- The 27B had no warm-up; the first 27B case (217 s) is not an outlier vs its
  mean, so cold-start effects look negligible.
