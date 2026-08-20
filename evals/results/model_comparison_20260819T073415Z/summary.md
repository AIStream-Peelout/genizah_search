# Synthesis model comparison

Generated 2026-08-19T09:00:07.211970+00:00 · models: qwen/qwen3.6-35b-a3b, qwen/qwen3.6-27b · judge: qwen/qwen3-4b-2507 · run order: model-major

- `qwen/qwen3.6-35b-a3b`: already resident (ctx 32768)
- `qwen/qwen3.6-27b`: already resident (ctx 32768)

## Per dataset × model

| Dataset | Model | Cases | Errors | Det. pass | Judge pass | Judge mean | Critical | Latency mean s | Latency median s | Synth s | Synth tok/s | Synth tokens | Synth reasoning tokens | Verify cycles | 1st-pass verify | Repairs | Flagged | Answer chars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cairo_genizah_agentic_rag_v1 | `qwen/qwen3.6-35b-a3b` | 11 | 0 | 0.64 | 0.64 | 3.97 | 0 | 73.89 | 80.03 | 43.99 | 79.58 | 3532.00 | 3252.91 | 1.54 | 0.64 | 6 | 0.09 | 3116.27 |
| cairo_genizah_agentic_rag_multiturn_v1 | `qwen/qwen3.6-35b-a3b` | 5 | 0 | 1.00 | 1.00 | 4.00 | 0 | 88.63 | 106.41 | 48.74 | 80.09 | 3922.80 | 3571.40 | 2.00 | 0.40 | 5 | 0.00 | 2747.80 |
| cairo_genizah_agentic_rag_v1 | `qwen/qwen3.6-27b` | 11 | 0 | 0.64 | 0.64 | 3.91 | 0 | 170.77 | 170.53 | 144.15 | 22.71 | 3389.00 | 3059.18 | 1.09 | 0.91 | 1 | 0.09 | 3460.55 |
| cairo_genizah_agentic_rag_multiturn_v1 | `qwen/qwen3.6-27b` | 5 | 0 | 1.00 | 1.00 | 3.97 | 0 | 277.56 | 213.31 | 188.22 | 23.81 | 4526.40 | 4209.80 | 1.80 | 0.60 | 4 | 0.20 | 2679.80 |

## Judge dimensions (mean 0–4)

| Dataset | Model | citation_and_quote_quality | groundedness_and_accuracy | intent_and_relevance | restraint_and_limitations | retrieval_evidence_coverage | synthesis_and_coherence |
|---|---|---|---|---|---|---|---|
| cairo_genizah_agentic_rag_v1 | `qwen/qwen3.6-35b-a3b` | 4.00 | 4.00 | 4.00 | 4.00 | 3.82 | 4.00 |
| cairo_genizah_agentic_rag_multiturn_v1 | `qwen/qwen3.6-35b-a3b` | 4.00 | 4.00 | 4.00 | 4.00 | 4.00 | 4.00 |
| cairo_genizah_agentic_rag_v1 | `qwen/qwen3.6-27b` | 3.82 | 4.00 | 4.00 | 4.00 | 3.73 | 3.91 |
| cairo_genizah_agentic_rag_multiturn_v1 | `qwen/qwen3.6-27b` | 3.80 | 4.00 | 4.00 | 4.00 | 4.00 | 4.00 |

## Per case

| Case | qwen/qwen3.6-35b-a3b (det/judge/latency s/verify cycles/repairs) | qwen/qwen3.6-27b (det/judge/latency s/verify cycles/repairs) |
|---|---|---|
| cairo_genizah_agentic_rag_multiturn_v1::arrant_fragments_followup | ✓ / 4.00 / 52.82 / 1 / 0 | ✓ / 4.00 / 163.89 / 1 / 0 |
| cairo_genizah_agentic_rag_multiturn_v1::goitein_india_trade_followup | ✓ / 4.00 / 111.72 / 4 / 3 | ✓ / 3.83 / 213.31 / 1 / 0 |
| cairo_genizah_agentic_rag_multiturn_v1::topic_change_after_goitein | ✓ / 4.00 / 112.99 / 2 / 1 | ✓ / 4.00 / 260.20 / 2 / 1 |
| cairo_genizah_agentic_rag_multiturn_v1::yom_kippur_shelfmark_followup | ✓ / 4.00 / 59.22 / 1 / 0 | ✓ / 4.00 / 131.29 / 1 / 0 |
| cairo_genizah_agentic_rag_multiturn_v1::yom_kippur_verses_followup | ✓ / 4.00 / 106.41 / 2 / 1 | ✓ / 4.00 / 619.10 / 4 / 3 |
| cairo_genizah_agentic_rag_v1::bible_codices_overview | ✓ / 4.00 / 116.83 / 4 / 3 | ✓ / 4.00 / 239.95 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::estara_arrant_profile | ✓ / 4.00 / 84.30 / 1 / 0 | ✓ / 3.83 / 211.39 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::goitein_identity_and_contributions | ✓ / 4.00 / 128.52 / 2 / 1 | ✓ / 3.83 / 285.43 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::ketubbot_overview | ✓ / 4.00 / 91.92 / 2 / 1 | ✓ / 4.00 / 286.28 / 2 / 1 |
| cairo_genizah_agentic_rag_v1::passover_seder_evolution | ✗ / 4.00 / 87.94 / 2 / 1 | ✗ / 3.67 / 126.61 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::purim_fragments | ✗ / 3.83 / 32.02 / 1 / 0 | ✗ / 3.83 / 75.17 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::shavuot_dairy_custom | ✓ / 4.00 / 42.48 / 1 / 0 | ✓ / 4.00 / 73.02 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::tisha_bav_fragments | ✗ / 3.83 / 61.92 / 1 / 0 | ✗ / 3.83 / 170.53 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::tisha_bav_kinnot | ✗ / 4.00 / 50.74 / 1 / 0 | ✗ / 4.00 / 111.06 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::tu_bav_history | ✓ / 4.00 / 36.06 / 1 / 0 | ✓ / 4.00 / 81.70 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::yom_kippur_piyyut_literature | ✓ / 4.00 / 80.03 / 1 / 0 | ✓ / 4.00 / 217.35 / 1 / 0 |

Legend: det = deterministic routing/retrieval/resolution checks; judge = mean judge score (0–4); verify cycles = verify_claims node executions (1 = passed first time); repairs = repair_answer executions.
