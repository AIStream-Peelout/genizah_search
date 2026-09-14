# Synthesis model comparison

Generated 2026-08-20T04:00:08.353877+00:00 · models: qwen/qwen3.6-35b-a3b, qwen/qwen3.8-27b · judge: qwen/qwen3-4b-2507 · run order: model-major

- `qwen/qwen3.8-27b`: loaded by script in 10.2s (ctx 32768)

## Per dataset × model

| Dataset | Model | Cases | Errors | Det. pass | Judge pass | Judge mean | Critical | Latency mean s | Latency median s | Synth s | Synth tok/s | Synth tokens | Synth reasoning tokens | Verify cycles | 1st-pass verify | Repairs | Flagged | Answer chars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cairo_genizah_agentic_rag_v1 | `qwen/qwen3.6-35b-a3b` (baseline, earlier run) | 11 | 0 | 0.64 | 0.64 | 3.97 | 0 | 73.89 | 80.03 | 43.99 | 79.58 | 3532.00 | 3252.91 | 1.54 | 0.64 | 6 | 0.09 | 3116.27 |
| cairo_genizah_agentic_rag_multiturn_v1 | `qwen/qwen3.6-35b-a3b` (baseline, earlier run) | 5 | 0 | 1.00 | 1.00 | 4.00 | 0 | 88.63 | 106.41 | 48.74 | 80.09 | 3922.80 | 3571.40 | 2.00 | 0.40 | 5 | 0.00 | 2747.80 |
| cairo_genizah_agentic_rag_v1 | `qwen/qwen3.8-27b` | 11 | 0 | 0.18 | 0.18 | 1.76 | 18 | 244.57 | 301.95 | 73.77 | 23.89 | 1779.64 | 3839.20 | 1.00 | 0.46 | 0 | 0.00 | 1298.73 |
| cairo_genizah_agentic_rag_multiturn_v1 | `qwen/qwen3.8-27b` | 5 | 0 | 0.20 | 0.20 | 0.80 | 4 | 279.07 | 302.57 | 33.28 | 23.20 | 772.20 | 3750.00 | 1.00 | 0.20 | 0 | 0.00 | 803.60 |

## Judge dimensions (mean 0–4)

| Dataset | Model | citation_and_quote_quality | groundedness_and_accuracy | intent_and_relevance | restraint_and_limitations | retrieval_evidence_coverage | synthesis_and_coherence |
|---|---|---|---|---|---|---|---|
| cairo_genizah_agentic_rag_v1 | `qwen/qwen3.6-35b-a3b` | 4.00 | 4.00 | 4.00 | 4.00 | 3.82 | 4.00 |
| cairo_genizah_agentic_rag_multiturn_v1 | `qwen/qwen3.6-35b-a3b` | 4.00 | 4.00 | 4.00 | 4.00 | 4.00 | 4.00 |
| cairo_genizah_agentic_rag_v1 | `qwen/qwen3.8-27b` | 1.82 | 1.73 | 1.82 | 1.82 | 1.64 | 1.73 |
| cairo_genizah_agentic_rag_multiturn_v1 | `qwen/qwen3.8-27b` | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 |

## Per case

| Case | qwen/qwen3.6-35b-a3b (det/judge/latency s/verify cycles/repairs) | qwen/qwen3.8-27b (det/judge/latency s/verify cycles/repairs) |
|---|---|---|
| cairo_genizah_agentic_rag_multiturn_v1::arrant_fragments_followup | ✓ / 4.00 / 52.82 / 1 / 0 | ✗ / 0.00 / 302.48 / — / — |
| cairo_genizah_agentic_rag_multiturn_v1::goitein_india_trade_followup | ✓ / 4.00 / 111.72 / 4 / 3 | ✗ / 0.00 / 302.57 / — / — |
| cairo_genizah_agentic_rag_multiturn_v1::topic_change_after_goitein | ✓ / 4.00 / 112.99 / 2 / 1 | ✗ / 0.00 / 308.22 / — / — |
| cairo_genizah_agentic_rag_multiturn_v1::yom_kippur_shelfmark_followup | ✓ / 4.00 / 59.22 / 1 / 0 | ✓ / 4.00 / 178.09 / 1 / 0 |
| cairo_genizah_agentic_rag_multiturn_v1::yom_kippur_verses_followup | ✓ / 4.00 / 106.41 / 2 / 1 | ✗ / 0.00 / 303.98 / — / — |
| cairo_genizah_agentic_rag_v1::bible_codices_overview | ✓ / 4.00 / 116.83 / 4 / 3 | ✗ / 0.00 / 303.44 / — / — |
| cairo_genizah_agentic_rag_v1::estara_arrant_profile | ✓ / 4.00 / 84.30 / 1 / 0 | ✗ / 0.00 / 301.95 / — / — |
| cairo_genizah_agentic_rag_v1::goitein_identity_and_contributions | ✓ / 4.00 / 128.52 / 2 / 1 | ✗ / 0.00 / 302.10 / — / — |
| cairo_genizah_agentic_rag_v1::ketubbot_overview | ✓ / 4.00 / 91.92 / 2 / 1 | ✗ / 0.00 / 302.88 / — / — |
| cairo_genizah_agentic_rag_v1::passover_seder_evolution | ✗ / 4.00 / 87.94 / 2 / 1 | ✗ / 3.67 / 181.71 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::purim_fragments | ✗ / 3.83 / 32.02 / 1 / 0 | ✗ / 3.67 / 182.10 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::shavuot_dairy_custom | ✓ / 4.00 / 42.48 / 1 / 0 | ✓ / 4.00 / 141.80 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::tisha_bav_fragments | ✗ / 3.83 / 61.92 / 1 / 0 | ✗ / 0.00 / 307.45 / — / — |
| cairo_genizah_agentic_rag_v1::tisha_bav_kinnot | ✗ / 4.00 / 50.74 / 1 / 0 | ✗ / 4.00 / 247.15 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::tu_bav_history | ✓ / 4.00 / 36.06 / 1 / 0 | ✓ / 4.00 / 111.09 / 1 / 0 |
| cairo_genizah_agentic_rag_v1::yom_kippur_piyyut_literature | ✓ / 4.00 / 80.03 / 1 / 0 | ✗ / 0.00 / 308.64 / — / — |

Legend: det = deterministic routing/retrieval/resolution checks; judge = mean judge score (0–4); verify cycles = verify_claims node executions (1 = passed first time); repairs = repair_answer executions.
