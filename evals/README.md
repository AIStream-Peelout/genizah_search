# Cairo Genizah Agentic RAG Evaluations

`agentic_rag_v1.json` is a versioned retrieval-and-answer benchmark for the
Genizah chat pipeline. It intentionally contains both well-covered questions
and questions for which the audited corpus does not currently support the
requested conclusion. It is **single-turn**: every case is one question with
no conversation history.

`agentic_rag_multiturn_v1.json` is a separate, smaller **multi-turn** sanity
check. Each case carries a fixed `conversation_history` (`role`/`content`
turns, the shape the chat UI sends) plus a follow-up `question` that only makes
sense in that context ("Can you give some samples of what the verses actually
state?"). It also includes one topic-change case that must *not* be
contextualized. Its cases may declare a `resolution` block, checked
deterministically against the response's `resolved_query`:

- `must_contain_any`: the follow-up was contextualized with the right subject.
- `must_not_be_rewritten`: a topic change passed through unchanged.

Run it with `--dataset evals/agentic_rag_multiturn_v1.json`; the runner prints
each case's `resolved_query` next to its deterministic status.

The benchmark evaluates separate failure surfaces:

1. Query intent and routing.
2. Retrieval of known relevant works, pages, graph identities, and fragments.
3. Rejection of known distractors.
4. Grounded synthesis, citation quality, and quotation accuracy.
5. Appropriate restraint when corpus evidence is partial or absent.

## Corpus snapshot

The initial targets were audited against:

- `bibliography_text_only_0.5`
- `cairo_genizah_text_v_5_3`
- the configured Neo4j graph, using read-only queries

Re-audit and version the target lists whenever these indices are rebuilt.
Do not silently change an existing benchmark's oracle to accommodate a
regression.

## Running

Start the backend and LM Studio, then run:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_agentic_rag_eval.py \
  --dataset evals/agentic_rag_v1.json \
  --api-base-url http://localhost:8000 \
  --judge-base-url http://localhost:1234 \
  --judge-model qwen/qwen3-4b-2507
```

Set `CHAT_API_KEY` in the environment when the backend protects `/chat`.
Use `--no-judge` to run only deterministic routing and retrieval checks, or
`--case CASE_ID` to run one or more selected cases.

To evaluate uncommitted backend changes without rebuilding or restarting the
running backend container, start a second instance from the venv against the
same compose services and point the runner at it:

```bash
PYTHONPATH=. .venv/bin/python scripts/dev_backend_local.py
```

then pass `--api-base-url http://127.0.0.1:8010`.

To compare a downloaded LM Studio synthesis model, configure the same private
`EVAL_API_KEY` in the backend and the runner environment, then add the model ID:

Generate and save a new key in the untracked `.env`, export it into the current
shell for the evaluation runner, and recreate the backend so Docker loads it:

```bash
eval_key="$(openssl rand -hex 32)"
printf '\nEVAL_API_KEY=%s\n' "$eval_key" >> .env
export EVAL_API_KEY="$eval_key"
docker compose up -d --force-recreate backend
```

The key is written without printing it to the terminal. Before running an
evaluation from a later terminal session, load the untracked `.env` variables:

```bash
set -a
source .env
set +a
```

Do not add `EVAL_API_KEY` to a `REACT_APP_*` variable or otherwise expose it in
the frontend bundle.

Then run the comparison with the exact downloaded LM Studio model ID:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_agentic_rag_eval.py \
  --api-base-url http://localhost:8000 \
  --synthesis-model downloaded-model-id \
  --no-judge
```

Model overrides use the hidden, evaluation-only endpoint. Public `/chat` and
`/chat-stream` requests continue to reject model overrides. The evaluation
endpoint is disabled when `EVAL_API_KEY` is unset and accepts only models that
LM Studio reports as already downloaded.

Results are written as JSON Lines beneath `evals/results/` unless `--output`
is supplied. Each row contains the case, raw RAG response, deterministic
checks, and structured judge result. The runner validates every judge score
and computes the pass rule independently; it records whether the judge's own
pass decision agreed with that calculation. Each row also carries a `metrics`
block: request latency, per-stage timings, verification cycles and repair
attempts, and per-stage token usage / tokens-per-second (taken from the
response's `metrics`, which the backend now reports on every chat response).
A judge failure is retried once and then recorded as `judge_error` beside the
response rather than discarding the case.

## Comparing synthesis models

`scripts/run_synthesis_model_comparison.py` runs one or more datasets against
several candidate synthesis models and aggregates quality *and* efficiency
(latency, synthesis tokens/s, verification cycles, repairs) so a model choice —
or a distillation / LoRA target — can be made on both. It starts its own
private backend instance on `--port` (default 8010) with an ephemeral
`EVAL_API_KEY`, loads any model that is not resident with an explicit
`--context-length` (default 32768, so candidates are on equal footing), runs
model-major, and writes `summary.md` / `summary.json` plus per-run JSONL and a
`progress.log` into `evals/results/model_comparison_<timestamp>/`. Router and
verification models stay at their configured defaults.

```bash
PYTHONPATH=. .venv/bin/python scripts/run_synthesis_model_comparison.py \
  --models qwen/qwen3.6-35b-a3b qwen/qwen3.6-27b \
  --datasets evals/agentic_rag_v1.json evals/agentic_rag_multiturn_v1.json \
  --judge-model qwen/qwen3-4b-2507
```

Models the script loaded are unloaded at the end unless
`--keep-models-loaded` is given; `--case ID` restricts a run for smoke tests.

**Shared-box safety:** LM Studio is shared with production chat and with batch
jobs (e.g. checkpoint audits) that can be submitted at any time. By default the
comparison waits for LM Studio to be fully idle (`lms ps --json`: no model
generating or queued, for `--idle-quiet-seconds`, default 30 s) before starting,
before loading each candidate model, and before every case (the runner's
`--wait-for-idle`). This prevents two large models decoding concurrently, which
exhausts RAM. `--no-wait-for-idle` overrides when you know the box is yours;
`--idle-max-wait-seconds` (default 12 h) bounds the wait. A run that sat waiting
logs it in `progress.log`, so pauses are visible in the case timings — per-case
`metrics.elapsed_seconds` is measured after the gate and stays comparable.
The judge model is also checked (and reloaded pinned) for
`--judge-context-length` (default 32768): the judge input is the whole case
plus bounded evidence, ~10k tokens, which overflows an 8k JIT-loaded context.

## Judge specification v2 (0-10, seven dimensions)

As of 2026-08-20 every dataset carries judge spec v2: integer scores **0-10**
on seven dimensions — `question_answered`, `answer_flow`,
`primary_source_linking`, `groundedness_and_accuracy`,
`citation_and_quote_quality`, `retrieval_evidence_coverage`,
`restraint_and_limitations` — with the pass rule computed by the runner from
the dataset's `judge.scale` and `judge.pass_thresholds` (currently: every
dimension >= 5, mean >= 7.5, no critical failures, deterministic checks pass).
The previous 0-4 six-dimension spec is archived as `judge_prompt_v1.md`;
result files judged before the switch keep their 0-4 scores until re-judged,
so do not compare raw judge means across the boundary.

**Judge providers.** The runner and `rejudge_eval_results.py` accept
`--judge-provider anthropic` to judge with the Claude API instead of local
LM Studio (`--judge-model claude-opus-5`; reads `ANTHROPIC_API_KEY` from the
environment). Recommended two-tier flow: the local 4B judge as the free
in-loop smoke tier, and an Opus re-judge over the saved JSONLs for
decision-grade comparisons — judging sends the case, answer, and bounded
evidence excerpts to the API. Example:

```bash
PYTHONPATH=. .venv/bin/python scripts/rejudge_eval_results.py   --results evals/results/model_comparison_<timestamp>/*.jsonl   --judge-provider anthropic --judge-model claude-opus-5 --all
```

## Cross-lingual dataset

`agentic_rag_crosslingual_v1.json` checks that the pipeline answers in the
user's language (Hebrew question → Hebrew answer; an English control guards
the other direction) while still retrieving across scripts. Cases declare a
`language` block that the runner checks deterministically via the Hebrew
character ratio of the answer's prose (appendices excluded). The synthesis
prompt's rule 12 implements the behavior.

## Human-in-the-loop annotation

`scripts/build_annotation_page.py` renders result JSONLs into a
self-contained local HTML page (`evals/annotations/`): all models' answers
side by side per case, 0-10 grades on the judge dimensions, pass verdicts,
notes, and a preferred-model choice, with LLM judge scores hidden by default
so human grades stay blind. Annotations autosave to the browser's
localStorage and export as JSON — use them to calibrate the LLM judge
(human/judge agreement) and as preference data for fine-tuning. Nothing is
uploaded anywhere.

## Re-judging saved results

Because every result row keeps the full RAG response, a failed judge (or a
changed judge prompt/model) can be re-applied without re-running RAG:

```bash
PYTHONPATH=. .venv/bin/python scripts/rejudge_eval_results.py \
  --results evals/results/model_comparison_<timestamp>/*.jsonl \
  --judge-model qwen/qwen3-4b-2507
PYTHONPATH=. .venv/bin/python scripts/run_synthesis_model_comparison.py \
  --summarize-only evals/results/model_comparison_<timestamp>
```

## Maintaining cases

- Keep user wording, spelling, and ambiguity in the primary `question`.
- Put normalized phrasings in `query_variants`; do not replace the original.
- `must_find_any` is a snapshot oracle: at least one listed target should be
  retrieved for covered cases.
- `known_distractors` are rubric guidance, not proof that a source is always
  irrelevant in every context.
- A `not_found` case should reward a concise limitation and penalize model
  memory presented as corpus evidence.
- Retrieval and generation scores must remain separate. A good answer cannot
  excuse failed retrieval, and excellent retrieval cannot excuse unsupported
  synthesis.
