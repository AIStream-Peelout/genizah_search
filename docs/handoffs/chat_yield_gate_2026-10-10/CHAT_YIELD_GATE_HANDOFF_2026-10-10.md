# Handoff: "yield to prod chat" gate for historical-document-analysis (2026-10-10)

For the Claude session working in `/Users/isaac/Documents/GitHub/historical-document-analysis` (the sibling repo).
Written by the genizah_search session "RAG system performance audit". All `path:line` refs are in the sibling working tree
(branch `kg/pipeline-version-stamp` @ `069af54`, surveyed read-only 2026-10-10) unless prefixed `genizah_search:`.

---

## 0. Paste this to start

> Read `/Users/isaac/Documents/GitHub/genizah_search/docs/handoffs/chat_yield_gate_2026-10-10/CHAT_YIELD_GATE_HANDOFF_2026-10-10.md` in full, then
> `CLAUDE.MD` and `docs/shared_studio_runtime.md`. Implement the "yield to prod chat" gate: `src/models/ocr/lms_yield_gate.py`,
> `tests/test_lms_yield_gate.py` and `tests/conftest.py` (§6), then the §5 hooks in priority order. `LMS_YIELD_MODE`
> defaults to `dryrun`, so nothing waits yet.
> The working tree is live (`pipeline_relaunch_1005.sh` relaunches `two_reader_lines` from it), so the gate must never
> raise (§4.8). Run the §7 step 1 smoke test before touching `lms_transcriber.py`; make each hook edit in one step and
> `py_compile` it straight after. Do not edit running shell scripts (`logs/v23a_eval_chain.sh`, `logs/routed/*.sh`).
> Never `lms load`/`unload`/`eject`. Ask Isaac which branch before committing; no branch switches, no worktrees;
> stage only your files. Run only the §6 run list. Report the diff summary, test results and log path.
> The dry run starts by itself when each process next starts. Do not restart, kill, SIGSTOP or relaunch anything.
> Report when the first `yield_gate.jsonl` line from each sibling process appears and count the 1-2 days from then;
> pid 10575 may not reload for about a day, so the window may start late.

---

## 1. Why

From the genizah_search audit (`docs/RAG_AUDIT_2026-10.md` and `evals/results/rag_audit_20261007/` in
`/Users/isaac/Documents/GitHub/genizah_search`; mainly `I1_lmstudio_infra.md` and `D_pipeline-restructure.md` §D3):

- Prod chat (api.cairogenizah.ai) runs its LLMs on **this Mac Studio's LM Studio**: router `qwen/qwen3-4b-2507`,
  synthesis `qwen/qwen3.8-27b` and verifier `qwen/qwen3.6-35b-a3b`. The sibling VLM batch shares that GPU and has
  no priority.
- The sibling VLM had a request in flight **92.9%** of the time from 09-23 to 10-07, with no quiet hour.
- Chat median is **222 s** and p90 **675 s**. **57% of all chat seconds is GPU contention.** Every chat call is
  2.4-3.2× slower. 35B decode falls from 88 to 32 tok/s, and to about 8 tok/s during VLM prefill.
- Clean A/B, the same "Who is S.D. Goitein" question with one verify pass each: **317 s with the VLM busy and
  124 s on a quiet GPU**. The worst real Goitein turn took **702 s**, of which 488 s was contention.
- Modelled effect of this gate: chat median 222 → ~114 s and p90 675 → ~229 s. Sibling throughput cost is about
  1-3%; prod used the chat LLMs 3.9 h in 3 weeks.

The genizah_search backend cannot see the sibling load, so the fix has to be on the sibling side. It is
client-side, fails open, and needs no prod change.

## 2. Owner decision (Isaac, 2026-10-07..10)

- Isaac is fine with pausing long-running sibling experiments and benchmarks while a prod chat request is active.
- **Inference, not an owner decision (confirm with Isaac in the dry-run report):** the sibling's own batch jobs on
  the chat models (the 35B judge, 27B descriptions and 4B coreference) yield to prod chat too. This is the author's
  reading of "pause long-running experiments/benchmarks" (`survey_sibling.md:134-136`, which also notes the audit left it open at `docs/RAG_AUDIT_2026-10.md:306`), not something Isaac said.
  If he says no, drop the gate calls from the §5 rows whose job runs on a chat model (`complete_text`,
  `describe_from_reads`, `coreference_resolver`, `llm_client`, `structured_json_llm`, `talmud_seg`) and keep only
  their `own_call` wrappers. The own-call registry (§4.3) only stops sibling jobs from yielding to *each other*.
- This lands as a sibling-repo change, made by you, **dry run first**.
- An AMD workstation is coming and will take the experiments off the Studio. The gate must then simply stop
  triggering (§4.6), so no removal work is needed.
- Yielding is scheduling only. Never kill or stop a job to make room for chat.

## 3. What the gate does (summary)

Before each unit of sibling GPU work (one page, one LLM call, one eval row or one stage), check whether a prod chat
is active:

- Read `lms ps --json`, which is local and read-only.
- If a chat is active and mode is `on`: sleep and poll every 3 s until it is not, with a hard cap of 15 min. At the
  cap, proceed with that one unit; the next unit re-checks.
- If mode is `dryrun`: log what the gate would have done and proceed immediately.
- On any error: proceed (fail open).

## 4. Design

### 4.1 Module `src/models/ocr/lms_yield_gate.py`

There is no `src/utils/`, so this sits next to the choke point. Follow `CLAUDE.MD`: Sphinx `:param:`/`:returns:`
docstrings, type hints, small pure functions, minimal try/except. Use `print(..., flush=True)` for human-readable
lines, matching `lms_transcriber`'s `✗`/`⏱` style, plus a JSONL log.

```python
CHAT_MODELS: tuple[str, ...] = ("qwen/qwen3-4b-2507", "qwen/qwen3.8-27b", "qwen/qwen3.6-35b-a3b")

@dataclass(frozen=True)
class YieldConfig:
    mode: str              # "off" | "dryrun" | "on"         env LMS_YIELD_MODE (default "dryrun"; unknown value -> "dryrun" + one warning)
    recent_s: float        # 180                               env LMS_YIELD_RECENT_S
    poll_s: float          # 3                                 env LMS_YIELD_POLL_S
    max_wait_s: float      # 900                               env LMS_YIELD_MAX_WAIT_S  (must stay < VLM TTL, see §4.4)
    own_end_tol_ms: int    # 5000  (lastUsedTime within this of an own call's end = ours)
    ps_timeout_s: float    # 10
    lms_cli: str           # ~/.lmstudio/bin/lms               env LMS_CLI
    state_dir: Path        # <repo>/logs/lms_yield             env LMS_YIELD_DIR  (logs/ is gitignored, .gitignore:53)

@dataclass(frozen=True)
class GateDecision:
    hold: bool
    reasons: tuple[str, ...]      # e.g. ("qwen/qwen3.8-27b:generating", "qwen/qwen3-4b-2507:recent 42s")
    ignored_own: tuple[str, ...]  # chat-model signals discarded because they are the sibling's own calls
    error: str | None             # set when lms ps failed -> hold=False (fail open)

def load_config(env: Mapping[str, str] | None = None, env_file: Path | None = None) -> YieldConfig  # never raises, §4.8
def is_chat_model(model: str) -> bool            # suffix-tolerant: "qwen3.6-35b-a3b" == "qwen/qwen3.6-35b-a3b"
def is_local_lms(base_url: str | None) -> bool   # string comparison only, no DNS, never raises, §4.8
def read_lms_ps(lms_cli: str, timeout_s: float) -> list[dict] | None   # None on ANY failure
def decide(entries: list[dict], own_calls: list[OwnCall], now_ms: int, cfg: YieldConfig) -> GateDecision  # PURE
def chat_active(*, cfg=None, reader=read_lms_ps, clock=time.time, registry=None) -> GateDecision
def wait_for_chat_idle(site: str, *, base_url: str | None = None, cfg=None,
                       reader=read_lms_ps, clock=time.time, sleep=time.sleep) -> float      # seconds waited
async def await_chat_idle(site: str, *, base_url: str | None = None, cfg=None,
                          reader=read_lms_ps, clock=time.time, sleep=asyncio.sleep) -> float
@contextmanager
def own_call(model: str, *, cfg=None) -> Iterator[None]   # no-op unless is_chat_model(model); on exit also appends
                                                          # a record to <state_dir>/own_calls.jsonl (§4.3)
def main() -> int   # CLI: --check (exit 0 clear / 3 hold), --wait --site NAME, --summarize [--since ISO]
```

### 4.2 Decision rule (`decide`, pure)

- `lms ps --json` returns an **array**, one object per loaded model. Match an entry to a chat model on
  `identifier`, `modelKey` or `indexedModelIdentifier`, suffix-tolerantly. Sibling code passes the 35B both with and
  without the `qwen/` prefix (`run_kg_overnight.py:48`, `fill_initial_stages.py:69`).
- A chat model is **busy** when `str(status).lower() != "idle"` or `int(queued or 0) > 0`. Observed status values
  are `idle`, `processingPrompt` and `generating`; loading also counts as busy.
- A chat model is **recent** when `lastUsedTime` is non-null and `now_ms - lastUsedTime < recent_s * 1000`.
  - Verified by polling on 2026-10-10: `lastUsedTime` is **epoch milliseconds**. It is **null while a request is in
    flight** and is set to the request's end time when the model returns to idle. So a null value is not an error.
  - 180 s bridges the gaps inside one chat turn: router → retrieval → synthesis → verify → repair.
- Discount the sibling's own calls, using the registry (§4.3):
  - Ignore **busy** on model M if a live in-flight own-call marker exists for M.
  - Ignore **recent** on model M if some own-call marker for M has `|lastUsedTime - end_ms| <= own_end_tol_ms`.
  - Record each ignored signal in `ignored_own`.
- `hold = any remaining busy or recent signal on a chat model`.
- Non-chat models (the VLM checkpoints) are never a reason to hold.
- `queued` and `parallel` are present (parallel = 4). Do not use `contextLength` or `ttlMs`.

`read_lms_ps`:
- Run `subprocess.run([lms_cli, "ps", "--json"], capture_output=True, text=True, timeout=ps_timeout_s)`. It costs
  about 0.2 s.
- Return `None` on a missing CLI, non-zero rc, timeout, invalid JSON or a non-list result. This is one try/except,
  and it is the fail-open point.
- Do **not** copy genizah_search's `lm_studio_busy_models`, which raises.
- The async wrapper runs it through `asyncio.to_thread`, so `two_reader_lines`' concurrent kraken prefetch
  (`two_reader_lines.py:1071/1080`) is not blocked.

### 4.3 Own-call registry (stops self-triggering)

- **Why it is needed.**
  - The sibling also uses the chat models: 4B coreference, 27B descriptions, 35B KG, bibliography and judge (§5,
    table B).
  - Without a registry, `two_reader_lines` would hold whenever `describe_from_reads` is generating on the 27B.
  - A job would also hold after each of its *own* 35B judge calls, because `lastUsedTime` is fresh.
- **Where markers live.** One marker file per call in `<state_dir>/own_calls/`, named
  `<pid>-<counter>-<model_slug>.json`, containing `{"pid", "model", "start_ms", "end_ms": null, "argv0"}`.
  - Write each file atomically: tmp file, then `os.replace`.
  - On context exit, rewrite the marker with `end_ms`. Do not delete it immediately: it must survive the
    `recent_s` window so the `lastUsedTime` match works.
- **Permanent record for the dry-run review (required).** Markers are pruned (below), so by review time nothing
  would show which server-log requests were the sibling's own. So on context exit `own_call` also appends one line
  `{"ts", "pid", "argv0", "model", "start_ms", "end_ms"}` to `<state_dir>/own_calls.jsonl`. Use the same append
  helper as the gate log (§4.4), so it is covered by that helper's single `OSError` try/except. Write it in a
  `finally`, so a call that raised is still recorded.
- **Reading and pruning.** `chat_active` reads the directory and prunes as it goes:
  - Delete markers whose pid is dead. Check with `os.kill(pid, 0)`; `ProcessLookupError` means dead and
    `PermissionError` means alive.
  - Delete markers whose `end_ms` is older than `recent_s + 60 s`.
  - Skip, and delete, unreadable or corrupt files.
  - Registry I/O is the second allowed try/except (`OSError`, `json.JSONDecodeError`). A registry failure means
    "no own calls known", which errs toward holding, and the hold is capped.
- **Scope.** `own_call(model)` is a no-op for non-chat models, so the VLM path pays nothing. `os.getpid()` makes
  the registry multi-process safe.
- **No shared lock.** Each process writes only its own files.

### 4.4 Waiting, modes, cap and logging

- **`off`:** return 0.0 immediately. No `lms ps` call and no log line. Use this in tests.
- **Skip:** if `base_url` is not local (`is_local_lms` is False), log `decision="skip_remote"` once per process and
  return 0.0. This is what keeps the gate harmless on the AMD box (§4.6).
- **`dryrun`:** one `chat_active()` per call, log it, never sleep. When the hold is recency-based, the log records
  `would_wait_s_est = recent_s - age_s` as a lower bound. `--summarize` reconstructs the would-wait spans
  per pid by grouping consecutive hold records.
- **`on`:** loop `chat_active()`; while it holds, call `sleep(poll_s)`.
  - Stop when the result is clear, when there is an error (fail open), or when `waited >= max_wait_s`, in which
    case log `capped=true` and proceed.
  - Print one line at wait start and one at wait end, e.g.
    `    ⏸ [site] yielding to prod chat (qwen/qwen3.8-27b:generating)` / `    ▶ [site] resumed after 74 s`.
  - Do not print every poll.
- **JSONL log:** `<state_dir>/yield_gate.jsonl`. Write one record per gate call, including calls that pass without
  waiting:
  `{"ts", "pid", "argv0", "site", "mode", "decision": "clear|hold|error|skip_remote", "reasons", "ignored_own",
  "error", "waited_s", "polls", "capped", "would_wait_s_est", "ps_ms"}`.
  - Append with `open(..., "a")`. Each line stays under PIPE_BUF, so appends from several processes stay whole.
  - The log write is the third allowed try/except (`OSError`): a failure prints a warning and the gate continues.
    The same `_append_jsonl(path, record)` helper writes `own_calls.jsonl` (§4.3).
  - For `skip_remote`, also log the parsed host, so the review can spot a base_url that is really this machine.
- **Volume and the `lms ps` cache.** Volume is NOT one line per VLM page. The per-call hooks in
  `coreference_resolver` and `llm_client` (the 35B KG per-entity loop, `relationship_extractor.py:791-824`) can
  produce thousands of gate calls per hour per process, each one an `lms ps` subprocess. So:
  - Cache the parsed `read_lms_ps` result in-process for **2 s** (a module-level `(ts, entries)` pair, read
    through the injected `clock`). Cache the raw entries, not the `GateDecision`: `decide` runs fresh each time
    against the registry.
  - `own_call` clears the cache on exit, so a sibling call that just ended is never judged against a snapshot taken
    while it was still in flight.
  - The `on` poll interval (3 s) is longer than the cache, so each poll sees a fresh `lms ps`.
  - Expected volume with the cache: at most one `lms ps` every 2 s per process, and up to a few thousand JSONL
    lines (~300 B each) per hour per busy LLM job, so roughly 1-2 MB/h in the worst case. That is acceptable for
    a 1-2 day dry run under the gitignored `logs/`. The VLM path (`two_reader_lines`) stays at about one line per
    page.
- **Cap semantics:** under continuous chat, the sibling runs one unit per 15 min, and each unit re-checks. This
  matches the audit's D3.
- **Why the 15-min cap is safe, and its upper bound.** The active VLM `qwen3-vl-8b-heb-v22i-a50` is loaded with
  `ttlMs = 3,600,000` (1 h) in `lms ps` (checked 2026-10-10; the chat models and the other VLM show no TTL).
  LM Studio unloads a TTL model after that much idle time. If a capped wait ever left the VLM idle past its TTL,
  the next sibling page would JIT-reload it, and a JIT load can evict the chat models (the known JIT ping-pong,
  `unloadPreviousJITModelOnLoad`). So `max_wait_s` must stay well below the shortest sibling VLM TTL:
  `load_config` rejects `LMS_YIELD_MAX_WAIT_S >= 3000` (falls back to 900 with one warning, §4.8). If a VLM is
  ever loaded with a shorter TTL, lower the cap to match. Do not touch the TTL itself.

### 4.5 The per-process `asyncio.Lock`: leave it process-local

- **What it is today.** `_get_lms_lock()` (`lms_transcriber.py:97-112`) is process-local. The comments at
  `:273-274` and `:362-363` say "LM Studio must never see two inference requests at once", but only within one
  process.
- **The audit's suggestion.** The audit suggested an `fcntl` cross-process lock.
- **Recommendation: do NOT make it cross-process in this change.** The gate does not need it: each process polls
  `lms ps` independently for about 0.2 s per unit.
- **Why not:**
  1. **Deadlock hazard.** `logs/v23a_eval_chain.sh:60-83` SIGSTOPs the consensus pid, and `logs/routed/vlm_guard.sh`
     SIGSTOPs pid 10575 on temp leaks. A stopped process that holds a cross-process flock (held for 20-95 s per page,
     across up to 5 retries, `:275-318`) would block the very benchmark that stopped it, and the chain resumes the
     consensus only after that benchmark finishes.
  2. **Throughput change.** It would serialise concurrent sibling jobs that run in parallel today, a behaviour change
     to running pipelines that Isaac has not asked for.
  3. **Blast radius.** One hung process would stall every sibling LM Studio user.
- **If wanted later:** do it as a separate, owner-approved change with a lock timeout and a stale-holder check.
- **Inside a process:** take the gate *inside* the existing `async with _get_lms_lock():`. That way the process's
  other coroutines also wait, which is intended.

### 4.6 AMD workstation / future

- When the experiments run on the AMD box, they point `base_url` at that box. `is_local_lms` is then False and the
  gate logs `skip_remote` and never holds. If jobs run *on* that box, its `lms ps` has no prod chat models, so
  `decide` never holds either.
- `LMS_YIELD_MODE=off` is the explicit off switch.
- v2 (later, optional): genizah_search may expose an authenticated `GET /chat/activity` with in-flight turns and the
  last turn end. `chat_active` would prefer it and fall back to `lms ps`. This makes the lease exact, starting before
  the 4B call and ending at `final`, and removes the limitation in §4.7. No work is needed now; keep `chat_active`
  as the single seam.

### 4.7 Known limitations (document them in the module docstring)

1. **Own call on a chat model masks prod on that same model.** This applies to every chat model, not only the
   35B. While any sibling process has an in-flight own call on model M, a prod request on M cannot be told apart
   from it: busy is ignored for M, and `lastUsedTime` is null while M is in flight. Concretely:
   - 35B: a sibling judge, KG or bibliography call masks the prod verify.
   - 27B: whenever `describe_from_reads` runs, its in-flight call masks prod synthesis.
   - 4B: during coreference batches, its in-flight call masks the prod router.
   Two things soften this:
   - The prod turn's signals on the *other* chat models still hold the gate (for example the 4B router's
     recency during a 27B-masked synthesis), and the 180 s recency covers the gaps.
   - The sibling's next unit, after its own call, sees the chat's `lastUsedTime`, provided it differs from its own
     `end_ms` by more than 5 s.
   The exact fix is v2 `/chat/activity`.
2. **An in-flight VLM page at chat start is not interrupted** (p50 29 s, p90 55 s per the audit). Only the next unit
   waits. Retries inside `transcribe_with_lm_studio` (`:276-318`) are not re-gated, because the check happens once
   per call, before the loop.
3. The first chat signal is the 4B router call, so the turn's ES retrieval before it is not covered.
4. Anything that touches a chat model counts as chat: Isaac's own tests, prod eval runs, or a watchdog call, if
   the watchdog ever sends inference. The dry-run review (§7) should show whether this causes false holds.

### 4.8 The gate must never raise (it runs inside the live batch)

`pipeline_relaunch_1005.sh:59` relaunches `.venv/bin/python -m src.datasets.consensus.two_reader_lines` from the
working tree, so the successor of pid 10575 imports this module on its next relaunch. An exception anywhere in the
gate path would crash a multi-day batch. Beyond the three try/excepts in §4, make the rest of the path non-raising:

- **`is_local_lms` — string comparison only, no DNS.** Do not call `socket.gethostbyname(...)` or any resolver
  (`socket.gethostbyname(socket.gethostname())` can raise `socket.gaierror` on macOS). Algorithm:
  - `None` or empty → True.
  - Extract the host with one regex, no `urllib` (`urlsplit` raises `ValueError` on malformed brackets):
    `^(?:[A-Za-z][A-Za-z0-9+.-]*://)?(?:[^@/]*@)?(\[[^\]/]*\]|[^:/?#]*)`. This accepts both `http://host:1234/v1`
    and the bare `host:port` form (`llm_client` derives `host_port`, `llm_client.py:166-170`). Strip `[` `]`,
    lowercase. A string that is exactly `::1` also counts.
  - Local iff the host is in `{"localhost", "127.0.0.1", "::1", h, h + ".local", h.removesuffix(".local")}` with
    `h = socket.gethostname().lower()` (`gethostname` does no lookup). On the Studio, `gethostname()` already returns
    `isaacs-Mac-Studio.local`.
  - The Studio's LAN IP is deliberately not in the set. No sibling code uses it today (grep 2026-10-10). If one
    does later, it logs `skip_remote` with the host, and the dry-run review will show it.
- **`load_config` — precedence and validation.** Most hooked entry points never call `load_dotenv`
  (`lms_transcriber`, `describe_from_reads`, `llm_client`, `talmud_evaluation`, `run_kg_overnight.py`,
  `v22_task_eval`, `probe_forgetting`, `quick_eval` and the v23a-chain benchmark scripts). Only
  `two_reader_lines.py:68-70`, `coreference_resolver.py:80-84`, `structured_json_llm.py:25/36` and
  `auto_eval_series.py:242` do. So a `.env` setting would not reach most processes unless the gate reads it itself:
  - For each `LMS_YIELD_*` key (and `LMS_CLI`): the process environment first, then the repo `.env` read with
    `dotenv.dotenv_values(<repo>/.env)` (repo = `Path(__file__).resolve().parents[3]`), then the built-in default.
    Do **not** call `load_dotenv` and do not mutate `os.environ`. `dotenv_values` returns `{}` for a missing file.
  - Take `env` and `env_file` as parameters (defaults `os.environ` and `<repo>/.env`) so tests can inject both.
  - Read the config once per process and cache it (module-level), so the `.env` file is not re-read per call.
  - Validate without raising. Numbers: accept only `re.fullmatch(r"\d+(\.\d+)?", value.strip())`, then apply
    bounds (`recent_s` 10-900, `poll_s` 1-60, `max_wait_s` 0-2999, see §4.4). A bad or out-of-range value → the
    default plus one `print(..., flush=True)` warning per key per process. Mode: lowercase; anything other than
    `off`/`dryrun`/`on` → `dryrun` plus one warning. This uses a regex, so the try/except count stays at three.

## 5. Hook points (priority = how long-running the job is)

Each hook is one line, `wait_for_chat_idle(site=...)` or `await await_chat_idle(site=...)`. Calls to chat models are
also wrapped in `own_call(model)`. Pass `base_url` wherever the call site has it.

### A. LM Studio request paths

| P | file:line | job / loop | edit |
|---|---|---|---|
| 1 | `src/models/ocr/lms_transcriber.py:275` (`transcribe_with_lm_studio`, def :239) | **Running consensus batch** `two_reader_lines` (stage2 :523, job loop :1072, call :1083; pid 10575, 5 days, page 5764/8376), plus `corpus_runner` :127, `linewise` :157, `v22_task_eval` :510, all VLM benchmarks (21 importers) | First statement inside `async with _get_lms_lock():`, before the retry loop: `await lms_yield_gate.await_chat_idle(site=f"transcribe:{model_id}", base_url=base_url)`. No `own_call` (image calls go to VLMs). |
| 1 | `src/models/ocr/lms_transcriber.py:364` (`complete_text`, def :328) | judge (35B via `error_classifier.py:217` ← `offline_error_analysis.py:114`, `fragment_evals.py:252`) and segmentation | Inside the lock: `await await_chat_idle(site=f"complete_text:{model_id}", base_url=base_url)`, then wrap the existing `try` body in `with lms_yield_gate.own_call(model_id):`. |
| 2 | `src/datasets/consensus/describe_from_reads.py:1326` (`LMStudioChatClient.chat`, POST at :1204; doc loop :1614) | 27B description batch (`DEFAULT_MODEL` :93) | Before `url = ...`: `wait_for_chat_idle(site="describe_from_reads", base_url=self.base_url)`. Wrap the `while True` transport loop in `own_call(model)`. **This file is untracked owner work: ask Isaac before editing it.** Its tests inject `transport=`/`sleep=` (`tests/test_describe_from_reads.py:450`), so they must pin `LMS_YIELD_MODE=off`. |
| 2 | `src/models/llm/academic/coreference_resolver.py:352` (`_call_lms`, def :308) and `:455`/`:458` (`_classify_ambiguous_llm`); batch loop :700 | 4B coreference (`_LMS_MODEL` :109); 35B via `run_kg_overnight.py:48/102` | Gate at the top of `_call_lms` and `_classify_ambiguous_llm`, plus `own_call(model)` around the POST and retry block. **Payloads are unchanged**: `text` (:345) and `json_object` (:452) stay; never add `json_schema` for the 4B. |
| 2 | `src/models/llm/academic/llm_client.py:179-181` (`_complete_lms`, lmstudio SDK; `host_port` :166-170, `for ctx_shift in range(3)` :174, `try:` :178, `with lms.Client(host_port)` :179) | 35B KG relations (`relationship_extractor.py:959`, per-entity loop :791-824), `entity_tagger.py:529`, `enrich_node_relations.py:176` | Gate **once, before the `for ctx_shift` loop** (not per context retry): `wait_for_chat_idle(site="llm_client", base_url=self.lms_url)`. Pass `self.lms_url` (it has the scheme), not `host_port`. Wrap the whole loop in `own_call(self.lms_model)`. **Do not change** `client.llm.model(...)`. It is get-or-load and can JIT-load an unloaded key such as the default `qwen3:8b`; that is a pre-existing hazard. Report it to Isaac and do not fix it here. |
| 2 | `src/finetuning/qwen_hebrew/eval_harness/auto_eval_series.py:280-282` (`gates_ok` def :103, called :276; `break` on failure :279; `log(f"gates OK ({why})")` :280; `full = ...` :281; `res = process(step, ...)` :282) | checkpoint eval orchestrator: CPU merge + `mlx_vlm convert` (GPU) per checkpoint via `hard_eval_ckpt.sh` :41/:54/:72 | Insert `wait_for_chat_idle(site=f"auto_eval_series:step{step}")` after :280 and before `full = ...` / `process(...)` at :281-282. This gives cap semantics; leave `gates_ok()` (:103-116) unchanged. The stage's own LM Studio calls are already gated through `lms_transcriber`. |
| 3 | `src/models/llm/academic/structured_json_llm.py:895` (per-page loop; `agent.run` :582/:593/:750) | bibliography pass 1 (35B via `fill_initial_stages.py:69/462`) | The loop at :895 is inside `async def process_ocr_results` (:848), so use the async form. The class can also run on Gemini (`use_gemini`, :125/:130), and `lm_studio_url` is not stored on `self` today (only passed to `_setup_lm_studio`, :133/:146). So: in `__init__` add `self.lm_studio_url = lm_studio_url`; at the top of the page loop body add `if not self.use_gemini: await lms_yield_gate.await_chat_idle(site="structured_json_llm", base_url=self.lm_studio_url)`; wrap the page's agent runs in `own_call(self.model_name)` only when `not self.use_gemini` (use `contextlib.nullcontext()` otherwise). |
| 3 | `src/datasets/evaluations/talmud_evaluation.py:792` (`_call_lms_text_only`, POST :812, no lock) | segmentation per section; doc loop :1409 | `await await_chat_idle(site="talmud_seg", base_url=base_url)` at the top, plus `own_call(model)` around the POST. Do not reroute it through `complete_text`: that would add the lock, which is a behaviour change. |
| 3 | `src/datasets/evaluations/helper_eval_scripts/probe_forgetting.py:54` (`call_model`, create :68; loops :134/:135) | VLM forgetting probe | `wait_for_chat_idle(site="probe_forgetting")` at the top of `call_model`. This also covers `confusable_probe_fonts.py:246`, which imports it. |
| 3 | `src/datasets/evaluations/helper_eval_scripts/confusable_probe_fonts.py:241` (per-row loop) | font probe | Covered via `call_model` above. Keep its start-only text gate (`lms_busy` :97, `wait_for_idle` :110, used :328) as it is. |
| 4 | `src/finetuning/qwen_hebrew/quick_eval.py:143` (in-process `mlx_vlm.generate`, call :146) | rare, small | `wait_for_chat_idle(site="quick_eval")` at the top of the loop body. No base_url, so it counts as local. |

### B. Not gated (record why in the PR)

- `mlx_train.py:388`: the step loop is inside mlx_vlm, it is dormant, and training is on Colab.
- `export_lmstudio.py:37`: a one-off.
- Kraken (`kraken-linewise` Docker): runs on CPU, so it is not a GPU contender.
- `check_lm_studio_health` and `/models` GETs: no inference.
- `preflight_final_benchmark.py:102`: goes to the OpenAI cloud.

### C. Shell orchestration

- **Do not edit `logs/v23a_eval_chain.sh` (pid 6616, armed) or `logs/routed/pipeline_relaunch_1005.sh` (pid 10278)
  while they run.** zsh reads scripts incrementally, so an edit can corrupt a live script.
- Their Python stages are covered once the call sites above are gated.
- For future chain scripts: before each GPU stage, run
  `.venv/bin/python -m src.models.ocr.lms_yield_gate --wait --site <stage>`.

**Corrections to the audit's list** (for the record):
- These are **not** bypasses: `fragment_evals.py`, `offline_error_analysis.py`, `preflight_final_benchmark.py` and
  `v22_task_eval.py`.
- The audit missed these bypasses: `describe_from_reads`, `coreference_resolver`, `llm_client` and
  `structured_json_llm`.

## 6. Tests: `tests/test_lms_yield_gate.py`

Use plain pytest with `monkeypatch`, fake clock/sleep/reader, and `tmp_path` for `state_dir`. There is no
`conftest.py` yet; add the one described below. Run from the repo root (see "Run list" at the end of this
section). Start from
genizah_search `src/backend/tests/test_model_comparison.py:328-367`, the tests for `busy_models_from_entries` and
`wait_for_lm_studio_idle` in `genizah_search:scripts/run_agentic_rag_eval.py:59-138`. Adapt them: filter to chat
models, add recency, fail open, and proceed at the cap instead of raising.

**`decide`:**
- All idle with old `lastUsedTime` → clear.
- 27B `generating` → hold.
- 4B `queued=1` → hold.
- 35B idle with `lastUsedTime` 42 s ago → hold, reason "recent 42s".
- 200 s ago → clear.
- Null `lastUsedTime` while idle → clear.
- VLM `generating` → clear.
- Bare id `qwen3.6-35b-a3b` matches.
- Own in-flight marker on the 35B plus the 35B `generating` → clear, with `ignored_own` set.
- Own marker `end_ms` within 5 s of `lastUsedTime` → clear.
- Own marker `end_ms` 30 s off → hold.
- An own marker on the 27B does not mask the 4B.
- Own in-flight marker on the 27B + 27B `generating` + 4B idle with `lastUsedTime` 30 s ago → hold (the 4B
  recency still catches the turn, §4.7 item 1), with the 27B signal in `ignored_own`.

**Registry:**
- `own_call` writes and then finalises the marker.
- `own_call` appends exactly one well-formed line `{ts, pid, argv0, model, start_ms, end_ms}` to
  `<state_dir>/own_calls.jsonl` on exit, including when the wrapped block raises; nothing is appended for a VLM
  model.
- `own_call` exit clears the 2 s `lms ps` cache (§4.4); with a fake clock, two gate calls 1 s apart call the reader
  once, and 3 s apart call it twice.
- A no-op for a VLM model.
- Dead-pid markers are pruned (use a pid from a finished `subprocess`).
- Expired markers are pruned.
- A corrupt file is skipped.

**`read_lms_ps`:**
- Missing CLI, non-zero rc, timeout, invalid JSON or a dict instead of a list → `None`.
- `chat_active` then returns `hold=False` with `error` set.

**`wait_for_chat_idle` / `await_chat_idle`:**
- `off` → the reader is never called.
- `dryrun` → never sleeps, logs `hold` and `would_wait_s_est`.
- `on` → sleeps in 3 s steps until clear and returns the waited seconds.
- Cap: 900 s of hold → proceeds with `capped=true`.
- A reader error mid-wait → proceeds.
- A remote `base_url` → `skip_remote`, and the reader is not called.
- JSONL lines parse, and the field set matches §4.4.

**`--summarize`:** a synthetic log yields the expected would-wait spans per pid.

**Never-raise (§4.8):**
- `is_local_lms`: `None`, `""`, `http://localhost:1234/v1`, `localhost:1234`, `127.0.0.1:1234`, `http://[::1]:1234`,
  `::1` and `socket.gethostname()` (with and without `.local`, upper-cased) → True; `http://192.168.1.50:1234/v1`,
  `amd-box.local:1234` → False; malformed input such as `http://[::1:1234`, `://`, `@@@` → a bool, no exception.
  Monkeypatch `socket.gethostbyname`/`getaddrinfo` to raise, to prove no lookup is made.
- `load_config` precedence, using a tmp `.env` passed as `env_file` and an injected `env` dict: the env value wins
  over `.env`; with no env value the `.env` value is used (for example `LMS_YIELD_MODE=on` in the file only); with
  neither, the default (`dryrun`); a missing `.env` file → defaults; `os.environ` is unchanged afterwards.
- `load_config` validation: `LMS_YIELD_RECENT_S=abc`, `LMS_YIELD_POLL_S=-1`, `LMS_YIELD_MAX_WAIT_S=1e9`, an empty
  string and `LMS_YIELD_MAX_WAIT_S=3600` (≥ the VLM TTL bound) each → the default plus one warning, no exception;
  `LMS_YIELD_MODE=ON` → `on`; `LMS_YIELD_MODE=bogus` → `dryrun` plus one warning.

**Isolation of every test run: new `tests/conftest.py`.** There is no conftest today. Add one with an autouse
fixture that sets `LMS_YIELD_MODE=off` and `LMS_YIELD_DIR=<tmp_path>` via `monkeypatch.setenv`, and resets the
module's cached config and `lms ps` cache. Without it, existing tests that reach hooked code would shell out to
`lms` and append to the REAL `logs/lms_yield/yield_gate.jsonl`, polluting the dry-run data. Example:
`tests/test_coreference_dedup.py:114-134` calls `_call_lms(..., "http://x/v1", ...)` directly, so with the gate
hooked it would log `skip_remote` there. `tests/test_lms_yield_gate.py` overrides mode per test (`monkeypatch.setenv`
or an explicit `cfg`), and the `load_config` tests pass their own `env` dict.

**Existing tests that drive or patch around hooked code:**
- `tests/test_reread.py:269-277` drives `transcribe_with_lm_studio` with a fake session.
- `tests/test_coreference_dedup.py:114-134` calls `_call_lms` directly.
- `tests/test_two_reader_htr_key.py`, `tests/test_v22_task_eval.py` and `tests/test_structured_json_llm_refusal.py`
  patch around hooked modules.
- `tests/test_describe_from_reads.py` (untracked owner work, injects `transport=`/`sleep=`, :450): only if Isaac
  OKs editing `describe_from_reads.py`.

**Run list** (from the repo root; nothing else, no full sweep):

```
.venv/bin/python -m pytest tests/test_lms_yield_gate.py tests/test_reread.py tests/test_coreference_dedup.py \
    tests/test_two_reader_htr_key.py tests/test_v22_task_eval.py tests/test_structured_json_llm_refusal.py
```

Add `tests/test_describe_from_reads.py` only if Isaac OKs that edit. After the run, check that
`logs/lms_yield/yield_gate.jsonl` and `own_calls.jsonl` did not grow (the conftest worked).

## 7. Rollout and success metrics

**Step 1: land.**
1. Write and test `lms_yield_gate.py` and `tests/conftest.py` (§6).
2. **Pre-hook smoke test (required, before the first hook edit in `lms_transcriber.py`).** `py_compile` alone does
   not catch import or runtime errors in a module the live relaunch will import. From the repo root, run the real
   read-only path once (dryrun, real `lms ps`; neither call sends inference):
   ```
   .venv/bin/python -m src.models.ocr.lms_yield_gate --check
   .venv/bin/python -c 'import asyncio; from src.models.ocr import lms_yield_gate as g; print(asyncio.run(g.await_chat_idle("smoke")))'
   ```
   `--check` exits 0 (clear) or 3 (hold); any traceback or other exit code is a failure. Then check that
   `logs/lms_yield/yield_gate.jsonl` received well-formed lines (each `json.loads`, with the §4.4 field set,
   `site="smoke"` present, `mode="dryrun"`). Do not proceed to the hooks until this passes.
3. Make the P1 edits in `lms_transcriber.py`, then the P2-P4 edits. Make each edit in one step and run
   `.venv/bin/python -m py_compile <file>` straight after. Then run the §6 run list.
4. Mode stays `dryrun` by default, so there is no behaviour change.
5. Running processes keep the old code until they restart:
   - pid 10575 (started Oct 5) picks the gate up at its next relaunch by `pipeline_relaunch_1005.sh`, or at the
     literary pass (about 2,600 pages from now, so possibly about a day).
   - The v23a chain's next benchmark step also picks it up.
   - Do not restart, kill, SIGSTOP or relaunch anything to speed this up.

**Step 2: dry run, 1-2 days (no waiting).** It starts by itself as processes start. Report to Isaac when the first
`yield_gate.jsonl` line from each sibling process (by `pid`/`argv0`) appears, and count the 1-2 days from then, not
from the commit.

**Step 3: review.** Join `logs/lms_yield/yield_gate.jsonl` and `logs/lms_yield/own_calls.jsonl` with the requests in
`~/.lmstudio/server-logs/2026-10/*.log`, read-only.
- **Defining chat turns.** Take the server-log requests to the 3 chat models. Exclude every request whose time falls
  inside a sibling `own_calls.jsonl` span `[start_ms - own_end_tol_ms, end_ms + own_end_tol_ms]` on the same model.
  Group what remains into bursts (gaps < 180 s); each burst is one chat turn. The markers in `own_calls/` are pruned
  at run time, so `own_calls.jsonl` is the only record of which requests were the sibling's own.
- **Active seconds per process-day.** For each pid and day, sum the gaps between consecutive gate records of that
  pid that are ≤ 300 s (longer gaps count as idle).
- Check that:
  - every chat turn in the window has ≥ 1 `hold` record from each active sibling process within one unit of the
    turn's start;
  - `hold` records with no chat turn within ±180 s are rare, and each is explained (§4.7 items 1 and 4);
  - `error` records are rare;
  - for each process, would-wait seconds per process-day divided by that process's active seconds that day is
    **≤ 3%**. Use this one ratio (same process, same day) for both numerator and denominator.

Send the summary (`--summarize` output plus those four numbers) to Isaac, together with the open question from §2
(should the sibling's own chat-model jobs yield too?).

**Step 4: switch to `on`, only with Isaac's OK.** Set `LMS_YIELD_MODE=on` in the repo `.env`. Because `load_config`
reads `<repo>/.env` itself (§4.8), this reaches every hooked entry point, including those that never call
`load_dotenv`. A process-environment value still wins, so check that no launcher exports `LMS_YIELD_MODE`. It
applies to newly started processes only (the config is cached per process).

**Success metrics.** The decode-bucket analysis is done by the genizah_search session with
`evals/results/rag_audit_20261007/i1_parse_lms_logs.py` and `i1_analyze.py`, read-only on server-logs. The live chat
probe is run by Isaac or the genizah_search session. Never send traffic to the prod chat API from this session.

| metric | target | baseline |
|---|---|---|
| chat decode samples in the "no VLM" bucket | **≥ 80%** (35B ~88 tok/s, 27B ~27 tok/s uncontended) | VLM busy 92.9% of the time; 35B decode ~32 tok/s contended |
| "Who is S.D. Goitein" via the live chat | **≤ 150 s** | 317 s busy / 124 s quiet / 702 s worst |
| sibling throughput (`two_reader_lines` pages per hour, 3 days before vs after `on`) | loss **≤ 3%** | VLM 20-95 s/page |
| gate waits that hit the 15-min cap | ~0, apart from genuinely long chat sessions | n/a |

**Rollback:** set `LMS_YIELD_MODE=off`, or `dryrun`, in the repo `.env` (read by the gate itself, §4.8, so it reaches
every hooked entry point). It takes effect at the next process start. Do not revert code without asking Isaac.

## 8. Do-nots (standing rules)

- **The Studio's LM Studio is prod.**
  - Never `lms load`, `unload` or `eject`.
  - Never load `google/gemma-4-31b-qat`.
  - Never eject `qwen3-vl-8b-heb-v18b-step700`.
  - No `lms server` restarts.
  - Read-only `lms ps --json` is fine.
- **Never send `response_format: json_schema` to `qwen/qwen3-4b-2507`.** It segfaulted the MLX engine on
  2026-10-07. Do not change any payload as part of this work.
- **Running jobs.** Never kill, SIGSTOP, restart or edit-in-place any running sibling job or shell script. The gate
  takes effect only when a process starts.
- **Unidentified process.** Leave pid 45790 alone.
- **Git.**
  - No git worktrees and no branch switches; relaunch loops read the working tree.
  - Ask Isaac which branch to commit to.
  - Stage only the files you changed; there are 47 dirty paths of owner work.
- **Resource-heavy actions on the Studio.** Ask before dev containers, node builds or full pytest sweeps. Run only
  the unit tests of the modules you changed.
- **genizah_search.** Touch nothing in `/Users/isaac/Documents/GitHub/genizah_search`: no checkout, edits, docker
  or compose actions, and no traffic to the prod chat API.
- **Coding standards.** Follow `CLAUDE.MD`: Sphinx `:param:` docstrings, type hints, modular and testable code. Use
  exactly the three try/excepts in §4 (lms ps read, registry I/O, log write) and no others. Everything else in the
  gate path must be non-raising by construction (§4.8: no DNS lookups, regex-validated config).
- **Stale doc.** Optional: `docs/shared_studio_runtime.md:3` still says api.cairogenizah.ai is served from the Studio.
  That has been stale since the 2026-09-15 MBP migration; prod is served from the M3 MBP, but LM Studio on the Studio
  is still prod. Fix that line when you add a short "yield gate" section to the same doc.

## 9. Contact

- Origin: the genizah_search Claude session "RAG system performance audit". Report:
  `/Users/isaac/Documents/GitHub/genizah_search/docs/RAG_AUDIT_2026-10.md`, items N1 and §"Decisions" 1, with
  supporting notes in `evals/results/rag_audit_20261007/I1_lmstudio_infra.md` (L1) and `D_pipeline-restructure.md`
  (D3).
- Full survey behind the refs above:
  `/Users/isaac/Documents/GitHub/genizah_search/docs/handoffs/chat_yield_gate_2026-10-10/survey_sibling.md`.
- Report progress (landed / dry-run summary / switched on) to Isaac. If `SendMessage` is available, also report to
  that session; find it with `ListAgents`.
