#!/bin/bash
# CPU-only corroboration: multilingual-e5-large-instruct on the full corpus (T2), one process at a time.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
G="/usr/bin/python3 $HERE/guard.py"
V=/Volumes/home/studio_offload/genizah_search_embedding_audit/corpus_vectors
cd "$HERE"
export AUDIT_DEVICE=cpu AUDIT_THREADS=4
echo "=== $(date '+%m-%d %H:%M:%S') CPU T2 e5-large-instruct embed + eval"
{ $G --name cpu_t2_embed_e5 --cpu-only --max-gb 6 -- $PY embed_corpus.py --model e5-large-instruct --max-seq 512 --batch-size 16 \
  && $G --name cpu_t2_eval_e5 --cpu-only --max-gb 6 -- $PY eval_corpus.py --model e5-large-instruct --max-seq 512 --vec-dir $V/e5-large-instruct__all__512; } || echo "!!! STEP FAILED: e5"
echo "=== $(date '+%m-%d %H:%M:%S') queue v7 finished"
