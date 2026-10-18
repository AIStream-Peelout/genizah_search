#!/bin/bash
# CPU-only text probes (no GPU) so the text audit progresses while the sibling's checkpoint evals
# hold the GPU back-to-back. Resumable shards + per-model LOCK let the GPU queue pick up where this stops.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
G="/usr/bin/python3 $HERE/guard.py"
V=/Volumes/home/studio_offload/genizah_search_embedding_audit/corpus_vectors
cd "$HERE"
export AUDIT_DEVICE=cpu AUDIT_THREADS=4
step() { echo "=== $(date '+%m-%d %H:%M:%S') $1"; }
step "CPU T2 qwen3-0.6b embed (remaining long shards) + eval"
{ $G --name cpu_t2_embed_qwen06 --cpu-only --max-gb 7 -- $PY embed_corpus.py --model qwen3-0.6b --max-seq 8192 --batch-size 8 \
  && $G --name cpu_t2_eval_qwen06 --cpu-only --max-gb 6 -- $PY eval_corpus.py --model qwen3-0.6b --max-seq 8192 --vec-dir $V/qwen3-0.6b__all__8192; } || echo "!!! STEP FAILED: cpu t2 qwen06"
step "CPU T2 e5-large-instruct embed + eval"
{ $G --name cpu_t2_embed_e5 --cpu-only --max-gb 6 -- $PY embed_corpus.py --model e5-large-instruct --max-seq 512 --batch-size 16 \
  && $G --name cpu_t2_eval_e5 --cpu-only --max-gb 6 -- $PY eval_corpus.py --model e5-large-instruct --max-seq 512 --vec-dir $V/e5-large-instruct__all__512; } || echo "!!! STEP FAILED: cpu t2 e5"
step "CPU T2 bge-m3 embed + eval"
{ $G --name cpu_t2_embed_bgem3 --cpu-only --max-gb 7 -- $PY embed_corpus.py --model bge-m3 --max-seq 8192 --batch-size 8 \
  && $G --name cpu_t2_eval_bgem3 --cpu-only --max-gb 6 -- $PY eval_corpus.py --model bge-m3 --max-seq 8192 --vec-dir $V/bge-m3__all__8192; } || echo "!!! STEP FAILED: cpu t2 bgem3"
step "CPU text queue finished"
