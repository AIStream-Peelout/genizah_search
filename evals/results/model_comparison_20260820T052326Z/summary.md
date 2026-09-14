# Synthesis model comparison

Generated 2026-08-20T07:03:12.767691+00:00 · models: qwen/qwen3.6-35b-a3b, qwen/qwen3.8-27b · judge: qwen/qwen3-4b-2507 · run order: model-major

- `qwen/qwen3.8-27b`: loaded by script in 11.2s (ctx 32768)

## Per dataset × model

| Dataset | Model | Cases | Errors | Det. pass | Judge pass | Judge mean | Critical | Latency mean s | Latency median s | Synth s | Synth tok/s | Synth tokens | Synth reasoning tokens | Verify cycles | 1st-pass verify | Repairs | Flagged | Answer chars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cairo_genizah_agentic_rag_v1 | `qwen/qwen3.6-35b-a3b` (baseline, earlier run) | 11 | 0 | 0.64 | 0.64 | 3.97 | 0 | 73.89 | 80.03 | 43.99 | 79.58 | 3532.00 | 3252.91 | 1.54 | 0.64 | 6 | 0.09 | 3116.27 |
| cairo_genizah_agentic_rag_multiturn_v1 | `qwen/qwen3.6-35b-a3b` (baseline, earlier run) | 5 | 0 | 1.00 | 1.00 | 4.00 | 0 | 88.63 | 106.41 | 48.74 | 80.09 | 3922.80 | 3571.40 | 2.00 | 0.40 | 5 | 0.00 | 2747.80 |
| cairo_genizah_agentic_rag_v1 | `qwen/qwen3.8-27b` | 11 | 0 | 0.64 | 0.64 | 3.76 | 1 | 290.64 | 346.71 | 230.67 | 23.43 | 5661.09 | 5633.00 | 1.00 | 1.00 | 0 | 1.27 | 2059.82 |
| cairo_genizah_agentic_rag_multiturn_v1 | `qwen/qwen3.8-27b` | 5 | 0 | 1.00 | 1.00 | 3.93 | 0 | 381.07 | 350.36 | 296.51 | 24.32 | 7220.40 | 7135.60 | 1.00 | 1.00 | 0 | 0.40 | 1913.80 |

## Judge dimensions (mean 0–4)

| Dataset | Model | citation_and_quote_quality | groundedness_and_accuracy | intent_and_relevance | restraint_and_limitations | retrieval_evidence_coverage | synthesis_and_coherence |
|---|---|---|---|---|---|---|---|
| cairo_genizah_agentic_rag_v1 | `qwen/qwen3.6-35b-a3b` | 4.00 | 4.00 | 4.00 | 4.00 | 3.82 | 4.00 |
| cairo_genizah_agentic_rag_multiturn_v1 | `qwen/qwen3.6-35b-a3b` | 4.00 | 4.00 | 4.00 | 4.00 | 4.00 | 4.00 |
| cairo_genizah_agentic_rag_v1 | `qwen/qwen3.8-27b` | 3.64 | 3.64 | 4.00 | 4.00 | 3.54 | 3.73 |
| cairo_genizah_agentic_rag_multiturn_v1 | `qwen/qwen3.8-27b` | 3.60 | 4.00 | 4.00 | 4.00 | 4.00 | 4.00 |

## Per case

| Case | qwen/qwen3.6-35b-a3b (det/judge/latency s/verify cycles/repairs) | qwen/qwen3.8-27b (det/judge/latency s/verify cycles/repairs) |
|---|---|---|
| cairo_genizah_agentic_rag_multiturn_v1::arrant_fragments_followup | ✓ / 4.00 / 52.82 / 1 / 0 | ✓ / 4.00 / 288.45 / 1 / 0 |
| cairo_genizah_agentic_rag_multiturn_v1::goitein_india_trade_followup | ✓ / 4.00 / 111.72 / 4 / 3 | ✓ / 3.83 / 350.36 / 1 / 0 |
| cairo_genizah_agentic_rag_multiturn_v1::topic_change_after_goitein | ✓ / 4.00 / 112.99 / 2 / 1 | ✓ / 3.83 / 533.28 / 1 / 0 |
| cairo_genizah_agentic_rag_multiturn_v1::yom_kippur_shelfmark_followup | ✓ / 4.00 / 59.22 / 1 / 0 | ✓ / 4.00 / 207.42 / 1 / 0 |
| cairo_genizah_agentic_rag_multiturn_v1::yom_kippur_verses_followup | ✓ / 4.00 / 106.41 / 2 / 1 | ✓ / 4.00 / 525.85 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::bible_codices_overview | ✓ / 4.00 / 116.83 / 4 / 3 | ✓ / 4.00 / 388.05 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::estara_arrant_profile | ✓ / 4.00 / 84.30 / 1 / 0 | ✓ / 4.00 / 523.76 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::goitein_identity_and_contributions | ✓ / 4.00 / 128.52 / 2 / 1 | ✓ / 3.67 / 367.85 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::ketubbot_overview | ✓ / 4.00 / 91.92 / 2 / 1 | ✓ / 4.00 / 346.71 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::passover_seder_evolution | ✗ / 4.00 / 87.94 / 2 / 1 | ✗ / 3.00 / 136.96 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::purim_fragments | ✗ / 3.83 / 32.02 / 1 / 0 | ✗ / 3.83 / 102.22 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::shavuot_dairy_custom | ✓ / 4.00 / 42.48 / 1 / 0 | ✓ / 4.00 / 83.74 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::tisha_bav_fragments | ✗ / 3.83 / 61.92 / 1 / 0 | ✗ / 3.67 / 252.70 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::tisha_bav_kinnot | ✗ / 4.00 / 50.74 / 1 / 0 | ✗ / 3.33 / 515.43 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::tu_bav_history | ✓ / 4.00 / 36.06 / 1 / 0 | ✓ / 4.00 / 62.15 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::yom_kippur_piyyut_literature | ✓ / 4.00 / 80.03 / 1 / 0 | ✓ / 3.83 / 417.42 / 1 / 0 |

Legend: det = deterministic routing/retrieval/resolution checks; judge = mean judge score (0–4); verify cycles = verify_claims node executions (1 = passed first time); repairs = repair_answer executions.
