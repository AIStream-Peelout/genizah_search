# Cairo Genizah Agentic RAG Evaluations

`agentic_rag_v1.json` is a versioned retrieval-and-answer benchmark for the
Genizah chat pipeline. It intentionally contains both well-covered questions
and questions for which the audited corpus does not currently support the
requested conclusion.

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
pass decision agreed with that calculation.

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
