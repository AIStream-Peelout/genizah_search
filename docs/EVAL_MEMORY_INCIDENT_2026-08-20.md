# Eval RAM incident & memory-budget report — 2026-08-20

Diagnostic write-up of the OOM that killed the eval driver (and the Claude Code
session) during the Qwen 3.8 synthesis-model comparison, plus a full accounting
of the memory each piece needs on this shared 128 GB Mac Studio and the delays
that limited RAM caused across the day's runs.

## 1. What happened (short version)

The box ran out of RAM while three things competed for it at once:

1. the **synthesis-model comparison** eval, which keeps **three LLMs resident**
   (router 4B + verifier 35B-A3B + the candidate 27B) for the *entire* run;
2. the **Qwen-VL checkpoint audit** the user submitted the same day, which loads
   its own 8B VL checkpoints into the same LM Studio;
3. the **production stack** (backend container, Neo4j, Elasticsearch, embedding)
   plus Docker Desktop's VM and normal desktop apps.

The macOS OOM killer fired and killed the largest recent allocator — the eval's
Python driver and the Claude Code process. Collateral: **LM Studio evicted every
model** (including the pinned prod chat models), and Docker Desktop went
sluggish. Production stayed *up* (API health 200, tunnel alive) but **cold** —
the next chat request JIT-reloads the 35B synthesis model from disk.

The Anthropic API "Server disconnected" errors seen in the minutes before the
kill were almost certainly a **symptom**, not a separate fault: the Python
process could not complete TLS handshakes while the box was thrashing.

## 2. The core design flaw this exposed

The eval harness has an **idle gate** (added after the *first* memory scare)
that waits for LM Studio to be idle before each case. But it gates on
**generation** (compute/latency contention), **not residence** (memory). Two
big models can sit *resident* side by side without either *generating* — the
gate sees "idle" and proceeds, while their combined weights + KV caches have
already overcommitted RAM. **The gate prevents concurrent compute, not
concurrent memory.** That is the bug.

## 3. Memory footprint accounting

### Per-model weights (measured this session, `lms ps --json`)

| Model | Role | Weights | Loaded context | KV cache @ that ctx (est.) |
|---|---|--:|--:|--:|
| `qwen/qwen3-4b-2507` | router + local judge | 2.3 GB | 32k | ~1 GB |
| `qwen/qwen3.6-35b-a3b` | prod synthesis **and** eval verifier | 22.1 GB | 32k | ~6–10 GB |
| `qwen/qwen3.8-27b` (or 3.6-27b) | eval candidate | 16.1 GB | 32k | ~6–10 GB |
| `qwen3-vl-8b-heb-*` | VL audit checkpoints | 9.9 GB each | 8k | ~1–2 GB each |

KV-cache figures are estimates; the point is that a **32k loaded context roughly
doubles the marginal cost** of the large models versus an 8k context, and it is
the swing factor we control.

### Supporting stack (steady state)

| Component | Approx. RSS |
|---|--:|
| Elasticsearch (2 GB heap) | ~3–4 GB |
| Neo4j | ~2 GB |
| embedding service | ~1 GB |
| Kibana | ~1 GB |
| Docker Desktop VM overhead | ~3–5 GB |
| macOS + desktop apps (WebStorm ~1.6 GB, browsers, …) | ~12–18 GB |

### The overcommit at the moment of the crash (~22:43 UTC)

| Bucket | GB (low–high) |
|---|--:|
| Eval models resident (4B + 35B-A3B + 27B, weights) | 40.5 |
| Their KV caches @ 32k | 13–21 |
| VL audit checkpoints (2–4 × 9.9 GB + KV) | 22–44 |
| Docker + prod stack | 10–14 |
| macOS + apps + Python drivers + Claude Code | 15–22 |
| **Total** | **~101–142 GB** |

Against **128 GB physical**, the high end of that range is already past the
ceiling. As generation grew the KV caches and macOS wired memory, the live set
crossed the line and the OOM killer fired. This is why it was intermittent —
it depended on how many VL checkpoints happened to be resident and how deep the
thinking contexts had grown.

## 4. Delays caused by limited RAM (measured across the day)

| Run (dir) | Cases | Wall span | Σ case time | Idle-gate waits | Notes |
|---|--:|--:|--:|--:|---|
| `…073415Z` (35B + 3.6-27B, ungated) | 32 | 1.4 h | 75 min | — | before the gate existed |
| `…023915Z` (3.8, gated, 300s cap) | 16 | 1.3 h | 68 min | 18 events / 9.0 min | 10/16 answers empty (budget bug) |
| `…052326Z` (3.8, gated, 900s cap) | 16 | 1.7 h | 85 min | 18 events / 9.0 min | valid answers; slow |
| `…182617Z` (3.8, gated, 16k budget) | 9 (died) | **4.3 h** | 110 min | 13 events / 6.6 min | OOM mid-run |

Three distinct RAM-driven delays:

1. **Idle-gate waiting** — the direct, *intended* cost of sharing the box:
   ~25 min of explicit "waiting for LM Studio to be idle" across the gated runs.
   This is correctness-over-speed and is working as designed; it is only
   necessary *because* the box is memory/compute-shared.
2. **Thrashing overhead** — in the run that OOM'd, wall span (4.3 h) far exceeds
   Σ case work (1.83 h) + gate waits (0.11 h). Much of that ~2.4 h gap is the
   whole box slowing down (memory compression / swap) as RAM filled toward the
   ceiling — the insidious, unlogged tax of running near the limit.
3. **Thinking-budget compounding** — raising `SYNTHESIS_MAX_TOKENS` to 16384
   (the fix for the empty-answer bug) made the thinking-heavy 3.8 run **mean
   733 s/case, max 1842 s** (31 min for one answer). Not a RAM cost itself, but
   it keeps the memory-heavy models resident **far longer**, widening the window
   in which a concurrent VL-audit load can tip the box over.
4. **Cold-load penalty** — every eviction/TTL churn costs a ~10–12 s reload;
   post-OOM, the first prod chat request pays a full cold 35B load (~30–60 s).

## 5. Recommendations

Ordered by impact. None of these are done yet — they are the proposed fixes.

1. **Add a residence/memory gate to the eval harness**, complementing the idle
   gate. Before loading a candidate model, read free RAM (`vm_stat`) and the
   resident set (`lms ps --json`), estimate the candidate's footprint
   (`lms load --estimate-only` already exists), and **refuse to proceed unless
   there is headroom** (e.g. keep ≥ 20 GB free). This directly prevents the
   overcommit the idle gate misses.
2. **Do not run the model-comparison eval concurrently with the VL audit.** Their
   combined *residence* (~55 GB eval + ~25–45 GB VL) is the specific thing that
   overcommits. Serialize them, or run the eval only when the audit is not
   scheduled.
3. **Lower eval context to what the eval needs.** The 32k loaded context on the
   35B and 27B is the biggest lever we control. Synthesis prompts here are
   ~5k tokens; loading these models at, say, 12k instead of 32k could save
   ~8–14 GB of KV cache with no effect on eval fidelity.
4. **Prefer the Claude/Opus judge — it needs ~0 local RAM.** It is a network API
   call, so it never competes for the box. The whole "trustworthy judge"
   deliverable can be produced from the *already-saved* answers with **no local
   model loaded at all**. This is the memory-safe path and should be the default
   for decision-grade scoring.
5. **Re-pin the prod chat models** (4B + 35B-A3B) after any OOM so production
   isn't left cold — but only deliberately, and not while the VL audit is
   loading, or it just re-triggers the contention.
6. **Cap the eval's synthesis budget per model.** A thinking model that spends
   8k+ tokens reasoning for a 400-token answer should hit a budget ceiling and
   fail honestly (as the empty-answer fix now does) rather than run 31 minutes
   and keep RAM pinned.

## 6. Current state (as of this report)

- **Production:** UP (API health 200, tunnel alive) but LM Studio is **cold** —
  prod chat models will JIT-reload on the next request. Not re-pinned yet
  (awaiting user decision).
- **Memory:** fully recovered — 52.5 GB free, 0 GB compressed, 17 GB wired.
- **LM Studio:** no models resident.
- **Eval work:** the 3.8 rerun died at 9–10/16; its partial results are on disk.
  The 35B-A3B baseline (16/16, valid) and the empty-answer-fix code are intact.
- **Opus judge:** acceptance-validated as trustworthy (3/3 gold) earlier; the
  full Opus re-judge of saved runs is the remaining, memory-safe deliverable.
