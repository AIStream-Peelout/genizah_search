# Sibling-repo reference survey for the chat-yield gate (N1 / L1)

Surveyed 2026-10-10 on `isaacs-Mac-Studio.local` (Mac16,9, 128 GB), read-only.
Sibling repo: `/Users/isaac/Documents/GitHub/historical-document-analysis`, branch `kg/pipeline-version-stamp`
@ `069af54`, **47 dirty/untracked paths** (see section g). All refs are `path:line` in that working tree
unless prefixed `genizah_search:` (prod-mbp snapshot at
`/private/tmp/claude-501/.../scratchpad/prod-mbp`, identical to the genizah_search working tree for the lines cited).

---------------------------------------------------------------------------------------------------------------

## (a) The choke point: `src/models/ocr/lms_transcriber.py` (416 lines, last commit 2bf4bb2 2026-10-05, clean)

| what | ref | notes |
|---|---|---|
| `LMStudioConfig` dataclass | `src/models/ocr/lms_transcriber.py:43-80` | `base_url="http://localhost:1234/v1"` (:46), default model `qwen/qwen3-vl-8b` (:49-51), `timeout_s=600` (:64), `connect_timeout_s=5` (:69), `max_retries=4` (:72), `retry_delay_s=5` (:79). `DEFAULT_CONFIG` singleton :83. |
| lock globals | `:94-95` | `_lms_lock`, `_lms_lock_loop` |
| `_get_lms_lock()` | `:97-112` | `asyncio.Lock`, lazily recreated per running event loop. **Process-local only** (does not serialise across the several sibling processes). |
| `check_lm_studio_health()` | `:119` | GET `/models` via aiohttp; no inference; does not take the lock. |
| `chat_payload()` | `:183-236` | builds the OpenAI body; rejects `extra_payload` overriding `model`/`messages` (:232). |
| `transcribe_with_lm_studio()` | def `:239`, lock `:275` | Async. **aiohttp** (`import aiohttp` :35), new `ClientSession` per attempt (:278), POST `{base_url}/chat/completions` (:271/279). Lock held across ALL retries (up to 5 attempts, sleeps inside the lock :291/:313/:318). Image as base64 data URI. Models: whatever the caller passes (VLM checkpoints, e.g. `qwen3-vl-8b-heb-v22i-a50`, `-v21b-step1200`, `-v18b-step700`). |
| `complete_text()` | def `:328`, lock `:364` | Async, text-only, aiohttp, single attempt (no retry), default timeout 120 s, `print` on failure. Used for **judge / segmentation** text calls (see d: can be the 35B). |
| `transcribe_batch()` | `:383-416` | sequential loop over model ids calling `transcribe_with_lm_studio` (:411). |

**Gate insertion point:** immediately after `async with _get_lms_lock():` at :275 and :364 (i.e. before the first
POST; the lock is held so the process's own call is never in flight while it checks). For `transcribe_with_lm_studio`
put the check **inside** the retry loop's first iteration only, or before the loop (once per call) — retries are rare.
Logging in this module is `print(...)` with the `✗`/`⏱` prefix style (no `logging`).

**Importers (21 = 20 src modules + 1 test; matches the audit's "21"):**

```
src/datasets/consensus/corpus_runner.py:37            (transcribe_with_lm_studio; call :127, per-record loop :212)
src/datasets/consensus/describe_from_reads.py:90      (DEFAULT_CONFIG only — its requests BYPASS the module, see b)
src/datasets/consensus/linewise.py:32                 (transcribe_with_lm_studio; per-LINE loop :151, call :157)
src/datasets/consensus/retry_escalates.py:31          (check_lm_studio_health only)
src/datasets/consensus/run_linewise_benchmark.py:31   (check_lm_studio_health only)
src/datasets/consensus/two_reader_lines.py:77         (THE running consensus batch; stage2 call :523; job loop :1072-1084)
src/datasets/evaluations/error_classifier.py:32       (complete_text as judge; call :217-219, 2-attempt loop :211)
src/datasets/evaluations/fragment_evals.py:28         (transcribe_batch :945, complete_text :252 segmentation)
src/datasets/evaluations/grounding_eval/grounding_eval.py:193 (lazy import)
src/datasets/evaluations/helper_eval_scripts/preflight_final_benchmark.py:39 (health only)
src/datasets/evaluations/helper_eval_scripts/probe_column_transcription.py:40
src/datasets/evaluations/helper_eval_scripts/run_arabic_benchmark.py:38
src/datasets/evaluations/helper_eval_scripts/run_religious_benchmark.py:64
src/datasets/evaluations/helper_eval_scripts/run_vqa_parse_eval.py:43
src/datasets/evaluations/talmud_crop_evaluation.py:36
src/datasets/evaluations/talmud_evaluation.py:77      (transcribe_with_lm_studio :1175; ALSO a bypass, see b)
src/finetuning/qwen_hebrew/eval_harness/layout_qa_eval.py:41
src/finetuning/qwen_hebrew/eval_harness/lite_eval.py:33
src/finetuning/qwen_hebrew/eval_harness/run_pgp131_v19b.py:20
src/finetuning/qwen_hebrew/eval_harness/v22_task_eval.py:53 (call :510 in per-row loop :502)
tests/test_reread.py:11                                (monkeypatches lms.aiohttp.ClientSession :269 and runs
                                                        transcribe_with_lm_studio via asyncio.run :272/:276)
```

Consequence for the handoff: `tests/test_reread.py:269-277` drive `transcribe_with_lm_studio` with a fake session;
a gate that shells out to `lms ps` would run inside these tests. The gate must be injectable / mode-controlled
(e.g. `LMS_YIELD_MODE` default `off` until the dry run is reviewed, and tests pin it off or monkeypatch the check).

---------------------------------------------------------------------------------------------------------------

## (b) LM Studio call sites that BYPASS lms_transcriber (no lock, no gate)

Corrections to the audit's candidate list first:
- `fragment_evals.py` is **not** a bypass: all its LM Studio traffic goes through `lms_transcriber`
  (`transcribe_batch` :945-950, `complete_text` :251-256, judge via `error_classifier` -> `complete_text`).
  Only constants `LM_STUDIO_BASE_URL` (:96) live there.
- `offline_error_analysis.py` is **not** a bypass: it calls `error_classifier.classify_transcription_errors`
  (:114) -> `lms_transcriber.complete_text` (`error_classifier.py:217-219`). But it is a sibling user of the
  **35B** chat model (see d).
- `preflight_final_benchmark.py` only calls `check_lm_studio_health` (:123) — no inference against LM Studio
  (its `client.chat.completions.create` at :102 is the **OpenAI cloud** preflight).
- `v22_task_eval.py` inference goes through `transcribe_with_lm_studio` (:510); its own urllib use is only
  `model_served()` GET `/models` (:540-551).
- `auto_eval_series.py` makes no inference calls itself; it is an orchestrator (subprocess stages) — see c.

Real bypasses (inference POSTs not via the module):

| file | call function / line | client | model(s) | per-item loop (where a one-line wait goes) | notes |
|---|---|---|---|---|---|
| `src/datasets/consensus/describe_from_reads.py` (**untracked file, uncommitted owner work**) | `requests_transport()` :1192-1209, POST at **:1204**; `LMStudioChatClient.chat()` :1310-1352 (url :1326, transport call :1332) | `requests` (sync) | **`qwen/qwen3.8-27b`** (`DEFAULT_MODEL` :93) | document loop in `run()` **:1614** (`for n, doc in enumerate(todo, 1)`), `describe_document` :1620 -> `client.chat` :1442 / repair :1451. Best hook: top of `LMStudioChatClient.chat()` (:1326) or the loop at :1614. | sends `response_format: json_schema` to the **27B** (fine; the json_schema ban is only for the 4B). Can double max_tokens on `finish_reason=length` (:1338-1342). |
| `src/datasets/evaluations/talmud_evaluation.py` | `_call_lms_text_only()` :792-822, POST **:812-816** | aiohttp (async), **no lock** | segmentation model passed by CLI (`--segmentation_model` + `--segmentation_base_url` :1658-1662) | called sequentially per section from `_call_ocr_segmentation._call_one` (:873-876; loop :907-908 region, `for section in SECTION_KEYS` then retry); doc loop :1409 | same shape as `complete_text`; simplest fix is to replace with `lms_transcriber.complete_text` (gets lock + gate). |
| `src/datasets/evaluations/helper_eval_scripts/probe_forgetting.py` | `call_model()` :54-82, `client.chat.completions.create` **:68** | `openai.OpenAI` sync (`:125`, base `LM_STUDIO_URL` :32) | VLM checkpoints (`--models`) | nested loops `for model` :134 / `for i, s` :135 -> call :138 | retries sleep 20 s (:81). |
| `src/datasets/evaluations/helper_eval_scripts/confusable_probe_fonts.py` | `run_model()` :219-253, `OpenAI(...)` :234, `call_model(...)` **:246** (imported from probe_forgetting :51) | openai sync | VLM checkpoints (default `qwen3-vl-8b-heb-v18b-step700,…v17-step800` :308) | per-row loop :241 | **Already has an idle gate**: `lms_busy()` :97-107 (parses TEXT `lms ps` for `GENERATING/PROCESSING/LOADING`), `wait_for_idle()` :110-126 (5 quiet polls x 60 s, 6 h cap) — but only once at start (`main` :327-328), not per item. |
| `src/models/llm/academic/coreference_resolver.py` | `_call_lms()` :308-~400, `requests.post` **:352**; `_classify_ambiguous_llm()` :422, `requests.post` **:455** and retry **:458** | `requests` sync | **`qwen/qwen3-4b-2507`** (`_LMS_MODEL` :109) by default; `run_kg_overnight.py:48/102` passes `qwen3.6-35b-a3b` | dedup batch loop `for batch_start in range(…, self._BATCH_SIZE)` **:700** (batch 25 names, :675), call :707; classify call from :768 | `_call_lms` sends `response_format: {"type": "text"}` (:345) — **not** json_schema, so the 4B segfault rule is respected; `_classify_ambiguous_llm` sends `{"type": "json_object"}` (:452), which LM Studio 400s, then retries without it (:456-458) — a wasted 4B round trip per call, still not json_schema; timeout 600 s (:115); up to 3 attempts with max_tokens 16384 doubling (:350-380). |
| `src/models/llm/academic/llm_client.py` (`LLMClient._complete_lms`) | :149-195; **lmstudio SDK** WebSocket: `lms.Client(host_port)` :179, `client.llm.model(self.lms_model)` **:180**, `model.respond(...)` **:181** | `lmstudio` Python SDK (sync) | default `qwen3:8b` (:92); callers: `relationship_extractor.py:959` default **`qwen/qwen3.6-35b-a3b`**; `entity_tagger.py:670` `qwen3:8b`; `run_kg_overnight.py:48` `qwen3.6-35b-a3b`; `complete_pipeline_example.py:54` / `fill_initial_stages.py:69` `qwen3.6-35b-a3b` | callers loop: `entity_tagger.py:487` (per structured page, call :529); `relationship_extractor.py:791-792/804` (per entity, call :824); `enrich_node_relations.py:208` (per node, call :176) | **Hazard:** in the lmstudio SDK `client.llm.model(key)` is get-**or-load**; an unloaded key (e.g. `qwen3:8b`) would JIT-load a model on the Studio (evicts / RAM). Sends JSON-schema constrained decoding via `response_format=response_schema` (:184) — must never be pointed at `qwen/qwen3-4b-2507`. |
| `src/models/llm/academic/structured_json_llm.py` | pydantic-ai `OpenAIChatModel` :146-152; `agent.run(...)` **:582, :593, :750** | pydantic-ai -> openai async | `qwen3-vl:8b` default (:90/:1034); `fill_initial_stages.py:462` passes `qwen3.6-35b-a3b` | per-page loop in `process_ocr_results` **:895** | Pass-1 of the bibliography/KG pipeline. |
| shell: `src/finetuning/qwen_hebrew/eval_harness/hard_eval_ckpt.sh` | `curl /v1/models` :90/:93 (no inference); `lms unload $NAME` :113 | — | its own candidate only | — | stage/cleanup script, no chat-model use. |
| shell: `offload_to_nas.sh:11` | `lms ps \| grep` (read-only abort check) | — | — | — | — |
| `src/finetuning/qwen_hebrew/eval_harness/auto_eval_series.py` | `lm_studio_up()` :93-100 (GET /models); `lms unload` :139 | — | its own candidate | stage loop :275 calls `process()` :188 -> `hard_eval_ckpt.sh` stages (:203, :213) | **`gates_ok()` :103-116** is an existing between-stage gate (disk, RAM, NAS, LM Studio up, no other hard_eval) — a natural place to add "no chat active". |

Shell scripts in `logs/` (not in `src/`) also launch LM Studio jobs (`logs/v23a_eval_chain.sh`,
`logs/routed/pipeline_relaunch_1005.sh`, `logs/arabic_eval_1002*.sh`); they call the Python entry points above, so
they are covered once the Python call sites are gated.

---------------------------------------------------------------------------------------------------------------

## (c) Direct GPU users (not via LM Studio)

| file | GPU path | loop / wait point | frequency |
|---|---|---|---|
| `src/finetuning/qwen_hebrew/mlx_train.py` (last touched 2026-07-17) | `mlx_vlm.trainer.sft_trainer.train` (import :243), called **:388**; `TrainingArgs(... steps_per_report=cfg.steps_per_report)` :354-357 (default 10, :105) | The step loop is inside mlx_vlm (not ours). Only hook without patching mlx_vlm: `_WandbTee.write/_parse` (:175-230) sees each report line every `steps_per_report` steps; a blocking wait there pauses training (print is synchronous). Hacky; not recommended. | **Rare / dormant**: fine-tunes now run on Colab (auto_eval_series polls HF checkpoint repos); no local training process seen. |
| `src/finetuning/qwen_hebrew/export_lmstudio.py` (2026-07-14) | MLX LoRA fuse `fuse_lora_layers` :37-58, `export` :62; optional sanity generation :92 | one-off, minutes | **one-off** |
| `src/finetuning/qwen_hebrew/quick_eval.py` (2026-07-19) | in-process `mlx_vlm.generate` (`transcribe` :63-88, call :78) | per-sample loop **:143**, call :146 — one-line wait goes at top of loop body | rare, small (N samples) |
| `src/finetuning/qwen_hebrew/eval_harness/hard_eval_ckpt.sh` | `merge_ckpt_generic.py` (torch bf16, **CPU**, `low_cpu_mem_usage` :69) then `python -m mlx_vlm convert -q --q-bits 8` (:41/:54/:72) | stage boundaries only | per checkpoint (merge+convert ~20-25 GB RSS spike, minutes); orchestrated by `logs/v23a_eval_chain.sh` / `auto_eval_series.py`. Not worth gating mid-stage; gate at `auto_eval_series.gates_ok()` :103 or before each `hard_eval_ckpt.sh` stage in the chain script. |
| `src/embeddings/qwen_text_embedding.py` (MPS :159-160; batch loop :182-211) | torch MPS embedding builds | per batch group | occasional index (re)builds; dirty file (owner work). |
| `src/embeddings/embedding_models.py` :33/:55, `src/finetuning/clip_hebrew/clip_hebrew.py` :314 (train loop :393), `src/finetuning/legacy_gemma/gemma_three.py` :291 | torch MPS | — | legacy / dormant (2025-11 .. 2026-07) |
| Kraken HTR (`src/models/ocr/kraken_transcriber.py` -> :8002) | Docker container `kraken-linewise`, `--cpus 6 --memory 10g` (`docs/shared_studio_runtime.md:44`) | — | **CPU in the Docker VM, not Metal** — not a GPU contender; stage1 of two_reader_lines (kraken) is prefetched concurrently with stage2 (`two_reader_lines.py:1071/1080`), only stage2 (VLM) needs the gate. |

---------------------------------------------------------------------------------------------------------------

## (d) Sibling's own use of the three CHAT models (gate must not mistake these for a chat)

| model | sibling users (file:line) | path |
|---|---|---|
| `qwen/qwen3-4b-2507` (prod router) | `src/models/llm/academic/coreference_resolver.py:109` (`_LMS_MODEL` default; used by `CoreferenceResolver` :632-650 unless overridden) | `requests.post` :352/:455/:458 (bypass) |
| `qwen/qwen3.8-27b` (prod synthesis) | `src/datasets/consensus/describe_from_reads.py:93` (`DEFAULT_MODEL`, untracked file) — generated-descriptions batch (`docs/generated_descriptions.md`, `tests/test_compare_descriptions.py:60`) | `requests.post` :1204 (bypass) |
| `qwen/qwen3.6-35b-a3b` (prod verifier) | `src/models/llm/academic/relationship_extractor.py:959` (`--lms-model` default) via LLMClient SDK; `run_kg_overnight.py:48` (`"qwen3.6-35b-a3b"`, no `qwen/` prefix) via LLMClient + CoreferenceResolver (:100-102); `src/datasets/indexing/bibliography/complete_pipeline_example.py:54/59`; `src/datasets/indexing/bibliography/fill_initial_stages.py:69` (via StructuredJSONLLM :462); **judge**: `offline_error_analysis.py:12` usage (`--judge-model qwen/qwen3.6-35b-a3b`) and any `fragment_evals`/`talmud_evaluation` run with `--judge_model … --judge_base_url` -> `error_classifier.py:216-219` -> `complete_text`; `docs/shared_studio_runtime.md:46` documents the 35B as the sibling's local judge | SDK (:180-181), `complete_text` (:364), aiohttp |

Implications:
1. Model ids may appear with or without the `qwen/` publisher prefix (`run_kg_overnight.py:48`,
   `fill_initial_stages.py:69`). `lms ps --json` reports `identifier` = `modelKey` = `qwen/qwen3.6-35b-a3b`, so the
   gate should match on `identifier` / `modelKey` / `indexedModelIdentifier` and accept a bare suffix match
   (`endswith("qwen3.6-35b-a3b")`).
2. **Self-trigger risk:** (i) the busy check — another sibling process (e.g. describe_from_reads on the 27B) makes the
   27B `generating`, which two_reader_lines' gate would read as a chat; (ii) the `lastUsedTime` bridge — the same
   process's own previous 35B judge call just ended, so `lastUsedTime` is fresh and the gate would hold after every
   judge call. Hence the audit's "own-call registry": every sibling call to a chat model registers itself
   (pid, model, start, end) in a small shared dir/file (fcntl-locked), and the gate ignores (a) busy status on a model
   that has a registered in-flight sibling call and (b) a `lastUsedTime` within ~5 s of a registered sibling call end.
   Registration points: `lms_transcriber.complete_text` (:364), `describe_from_reads.LMStudioChatClient.chat` (:1326),
   `coreference_resolver._call_lms` (:352) and `_classify_ambiguous_llm` (:455), `LLMClient._complete_lms` (:179),
   `structured_json_llm` agent runs (:582/:593/:750) when pointed at a chat model.
   Owner decision still open in the audit (`docs/RAG_AUDIT_2026-10.md:306`): should the sibling's own chat-model
   jobs (35B judge, 27B descriptions, 4B coreference) ALSO yield to prod chat? Simplest consistent answer: yes —
   they are batch work; the registry only stops them from yielding to *each other*.
3. None of the chat-model jobs is running now (section g).

---------------------------------------------------------------------------------------------------------------

## (e) `lms ps --json` — exact shape (read-only run, 2026-10-10 ~22:13 local)

CLI: `~/.lmstudio/bin/lms ps --json` -> JSON **array**, one object per loaded model. ~0.15-0.2 s per call
(12 polls at 4 s intervals cost ~4.17 s each). Plain `lms ps` prints a table with UPPERCASE status
(`IDLE`, `GENERATING`) — that is what `confusable_probe_fonts.lms_busy()` parses; prefer `--json`.

Fields per entry (all observed): `type` ("llm"), `modelKey`, `format` ("safetensors"|"gguf"), `displayName`,
`publisher`, `path`, `sizeBytes`, `indexedModelIdentifier`, `deviceIdentifier` (null), `paramsString`,
`architecture`, `quantization` {name,bits}, `variants`, `selectedVariant`, **`identifier`**, **`ttlMs`**
(null for pinned, 3600000 for the JIT-loaded VLM), **`lastUsedTime`**, `vision`, `trainedForToolUse`,
`maxContextLength` (262144), **`contextLength`** (32768 chat / 8192 VLM), **`status`**, **`queued`**, **`parallel`** (4).

Loaded right now:

| identifier | status | queued | lastUsedTime | ctx | ttlMs |
|---|---|---|---|---|---|
| qwen/qwen3.8-27b | idle | 0 | 1791531029577 | 32768 | null |
| qwen/qwen3.6-35b-a3b | idle | 0 | 1791363377886 | 32768 | null |
| qwen3-vl-8b-heb-v21b-step1200 | idle | 0 | 1791499543156 | 8192 | null |
| qwen/qwen3-4b-2507 | idle | 0 | 1791362634992 | 32768 | null |
| qwen3-vl-8b-heb-v22i-a50 | generating / processingPrompt | 0 | **null while busy** | 8192 | 3600000 |

Status values observed in a 50 s poll: `idle`, `processingPrompt`, `generating` (camelCase, lower-first).
`confusable_probe_fonts.py:107` also treats `LOADING` (text form) as busy; treat any `status != "idle"` as busy
(that is what genizah_search `busy_models_from_entries` does), plus `queued > 0`.

**`lastUsedTime` semantics (verified by polling):** epoch **milliseconds**; it is **null while the model is
processing/generating** and is set to the request's **end** time when the model returns to idle (observed:
v22i-a50 went idle at poll 1791684929407 with `lastUsedTime=1791684929084`, then back to `null` on the next
request). So: "busy" = `status != idle or queued > 0`; "recently used" = `now_ms - lastUsedTime < HOLD_MS`
(lastUsedTime non-null). A chat's gaps between router -> synthesis -> verify are bridged by the recency window
(audit: 120-180 s). Chat models were last used ~42 h (27B) and ~3.7 days (35B/4B) ago at survey time.

---------------------------------------------------------------------------------------------------------------

## (f) Reusable code

1. **genizah_search `scripts/run_agentic_rag_eval.py`** (identical in the prod-mbp snapshot and working tree for
   lines 1-141):
   - `LMS_CLI = os.path.expanduser("~/.lmstudio/bin/lms")` :18
   - `busy_models_from_entries(entries)` **:59-72** — pure: `status = str(entry.get("status") or "idle").lower()`,
     `queued = int(entry.get("queued") or 0)`, busy if `status != "idle" or queued > 0`; returns descriptions.
   - `lm_studio_busy_models(lms_cli)` **:75-89** — `subprocess.run([lms, "ps", "--json"], timeout=30)`; raises
     `RuntimeError` on non-zero rc (the gate must instead **fail open**).
   - `wait_for_lm_studio_idle(quiet_seconds, max_wait_seconds, poll_seconds, check, log, sleep, clock)` **:92-138** —
     injectable check/log/sleep/clock, logs once per minute, requires a quiet window, raises `TimeoutError` at the cap
     (the gate must instead **proceed** at the cap).
   - Tests to copy: genizah_search `src/backend/tests/test_model_comparison.py:328-341`
     (`test_busy_models_from_entries_flags_generating_and_queued`) and `:344-367`
     (`test_wait_for_lm_studio_idle_requires_a_quiet_window_and_times_out`, fake clock/sleep).
   - Reused already by `scripts/run_synthesis_model_comparison.py:38/676`.
   What to change for the gate: filter entries to the 3 chat identifiers; add the `lastUsedTime` recency rule;
   subtract the sibling's own registered calls; fail-open on any error; cap -> proceed (log), never raise.
2. **genizah_search `evals/embedding_audit/guard.py`** (working tree only, commit 9946996; NOT in prod-mbp) —
   process-level guard (kill/relaunch a child), not an in-request gate. Copyable pieces:
   O_EXCL pid-file mutex with stale-pid takeover `acquire_mutex`/`release_mutex` :192-238; `pgrep -fl` sibling
   phase detection that filters out pgrep/guard/Claude shell-snapshot self-matches :84-102, :104-139;
   `~/box_jobs/*.status` status-file convention :31, `write_status` :240. The pgrep self-match filter is relevant if
   the gate ever detects processes by name.
3. **Sibling `confusable_probe_fonts.py:97-126`** — the pattern the sibling's own runtime doc points to
   (`docs/shared_studio_runtime.md:22-24` "queue behind it (the font probe's `lms ps` idle gate is the pattern)").
   Text parsing and 60 s polls; replace with the JSON version.
4. **Sibling `logs/v23a_eval_chain.sh:60-83`** — `pause_consensus`/`resume_consensus` via `kill -STOP/-CONT` of the
   consensus pipeline pid and its guards, with `trap resume_consensus EXIT`; `:90` waits on `pgrep` for other LM
   Studio jobs. Note SIGSTOP does not stop an LM Studio request already in flight (server keeps generating), so the
   in-request gate is the better mechanism for chat.
5. **Sibling `auto_eval_series.gates_ok()` :103-116** — between-stage gate; add a `chat_active()` condition there.
6. `~/box_jobs/` currently holds `genizah_search_embedding_audit.status` and
   `historical_document_analysis_v22b_evals.status` (sibling session's free-text status, last updated 2026-10-05) —
   a natural place for the gate's dry-run/wait log summary or the own-call registry (or use `logs/` in the sibling).

---------------------------------------------------------------------------------------------------------------

## (g) Sibling conventions and current state

- **Standards:** `CLAUDE.MD` (uppercase .MD) = same coding standards as genizah_search (Sphinx `:param:` docstrings,
  type hints, modular/testable, minimal try/except) + "Shared Machine" section; `AGENTS.md` also present.
  `docs/shared_studio_runtime.md` is the sibling's runtime contract (note: its line 3 still says
  api.cairogenizah.ai is served from the Studio — stale since the 2026-09-15 MBP migration; LM Studio on the Studio
  is still prod).
- **Tests:** flat `tests/` (102 entries), `[tool.pytest.ini_options] testpaths=["tests"], python_files="test_*.py"`
  (`pyproject.toml:94-96`), no `conftest.py`; tests import `from src...` (run from repo root, e.g.
  `.venv/bin/python -m pytest tests/test_<x>.py`); style: plain pytest functions + `monkeypatch` + `asyncio.run`
  (`tests/test_reread.py`), injectable transports/sleeps (`tests/test_describe_from_reads.py:450`
  `LMStudioChatClient(..., transport=..., sleep=lambda s: None)`). Interpreters: `.venv` (main), `.venv-mlx` (MLX).
- **Logging:** mixed. OCR/eval/consensus code uses `print(..., flush=True)` (lms_transcriber, two_reader_lines,
  describe_from_reads); KG/academic code uses `logging.getLogger(__name__)` (81 uses, e.g.
  `coreference_resolver.py:100`). Gate should `print(..., flush=True)` in lms_transcriber context so it lands in the
  pipeline logs (`logs/two_reader_routed_documentary_0928.log`).
- **Config/env:** `.env` at repo root (`dotenv.load_dotenv()` in ~75 modules); env var style `LM_STUDIO_URL`,
  `LM_STUDIO_MODEL` (`enrich_node_relations.py:36-37/268`); config dataclasses (`LMStudioConfig`). A
  `LMS_YIELD_MODE=off|dryrun|on` plus `LMS_YIELD_MAX_WAIT_S`, `LMS_YIELD_RECENT_S` fits.
- **Where the helper lives:** there is **no `src/utils/`**. Put it next to the choke point:
  `src/models/ocr/lms_yield_gate.py` (pure decision function + `lms ps` reader + own-call registry + sync and async
  wait wrappers), imported by `lms_transcriber.py` and the bypass sites; tests in `tests/test_lms_yield_gate.py`.
- **Dirty working tree (47 paths)**: modified files under `src/datasets/indexing/*`, `src/datasets/merging/*`,
  `src/datasets/document_models/*`, `src/embeddings/qwen_text_embedding.py`, several tests; **untracked**
  `src/datasets/consensus/describe_from_reads.py` (a bypass site, 27B user), `compare_descriptions.py`,
  `rover_reads.py`, etc. `lms_transcriber.py`, `two_reader_lines.py`, `coreference_resolver.py`,
  `llm_client.py`, `talmud_evaluation.py`, `probe_forgetting.py`, `confusable_probe_fonts.py` are **clean**.
  Any edit must be coordinated with the sibling's Claude session (no worktrees; no branch switch under running jobs).

**Running sibling processes (read-only `ps`, not touched):**

| pid | elapsed | command |
|---|---|---|
| 10278 | 5d19h | `zsh logs/routed/pipeline_relaunch_1005.sh qwen3-vl-8b-heb-v22i-a50 … qwen3-vl-8b-heb-v21b-step1200 …` — relaunch loop (`while true` :57-59 relaunches `two_reader_lines` from the **working tree** after any exit; then runs the literary pass with v21b :80) |
| 10575 | 5d19h | `python -m src.datasets.consensus.two_reader_lines --ids logs/routed/jobs_routed_documentary.jsonl --source-index genizah_merged_v8 --vlm-model qwen3-vl-8b-heb-v22i-a50 …` — page 5764/8376, VLM 20-95 s/page, kraken 19-55 s/page |
| 10964 | 5d19h | `logs/routed/vlm_guard.sh` — every 5 min; on temp-leak growth it SIGSTOPs 10575 and `lms unload`s the v22i VLM (last fired 2026-10-09 17:38) |
| 10965 / 10966 | 5d19h | `loader_loop.sh` (ES loader of ai_reads), `kraken_guard.sh` |
| 10279 | 5d19h | `logs/status_page/status_loop.sh` |
| 6616 | 5d16h | `zsh logs/v23a_eval_chain.sh` — armed; per checkpoint pauses consensus (SIGSTOP), runs hard_eval_ckpt + Arabic/VQA/PGP benchmarks via LM Studio; last activity 2026-10-07 04:05 |
| 45790 | 1d06h | bare `Python` with cwd = sibling repo, no args visible (unidentified; leave alone) |

Not running: describe_from_reads (27B), coreference/KG jobs (4B/35B), auto_eval_series, any training.

**Effect timing:** Python loads the module at start, so a gate merged into the working tree affects only
**newly started** processes: the next `two_reader_lines` relaunch by pipeline_relaunch_1005.sh (after a crash/STOP
exit, or the literary pass after the documentary pass finishes ~2,600 pages from now), the next v23a chain
benchmark step, or any new job. The running pid 10575 keeps the old code. Conversely, because those relaunches
read the working tree, a half-finished edit or a branch switch in the sibling checkout would be picked up by the
next relaunch — land the change in one step, with `LMS_YIELD_MODE` defaulting to `off`/`dryrun`.

---------------------------------------------------------------------------------------------------------------

## Design notes the handoff should carry (derived from the above)

- Gate decision (pure function, unit-tested): given `lms ps --json` entries, `now_ms`, chat ids, recency window and
  the own-call registry -> `(hold: bool, reason: str)`. Hold iff any chat model (matched by identifier/modelKey,
  suffix-tolerant) has `status != "idle"` or `queued > 0` (minus models with a registered in-flight sibling call), or
  `lastUsedTime` non-null and within the recency window (minus registered sibling call ends).
- Async wrapper for lms_transcriber (`await asyncio.sleep(poll)` while holding `_get_lms_lock()` — fine, it is
  per-process and the process should not send anything else meanwhile); sync wrapper for requests/openai/SDK sites.
- Poll 3-5 s; cap 15 min then proceed; fail open on any `lms ps` error/timeout (subprocess timeout ~10 s);
  log every wait start/end (print, flush) with reason and seconds; modes `off|dryrun|on` (dryrun = log what it would
  have held, never sleep).
- Cross-process coordination: the audit asks for an fcntl file lock that shares **only the gate decision** (one
  process polls `lms ps`, others read its cached verdict), not a serialising lock on requests. Optional; with
  ~0.2 s per `lms ps` and a handful of processes, independent polling is also acceptable.
- Residual: an in-flight VLM page at chat start is not interrupted (p50 29 s, p90 55 s per the audit); retries
  inside `transcribe_with_lm_studio` (:276-318) happen under the lock, so check once per call before the loop.
- When the AMD workstation takes the experiments off the Studio, the gate stops triggering on its own (no local
  chat models in `lms ps` on that box -> nothing to hold for; or `LMS_YIELD_MODE=off`).
