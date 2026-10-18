#!/bin/bash
# Re-run of the qwen T2 tail: the last length-sorted shards need a 10 GB footprint cap on MPS.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
G="/usr/bin/python3 $HERE/guard.py"
V=/Volumes/home/studio_offload/genizah_search_embedding_audit/corpus_vectors
PILOT=/Volumes/home/studio_offload/genizah_search_embedding_audit/pilot_text_v1
cd "$HERE"
echo "=== $(date '+%m-%d %H:%M:%S') T2 qwen3-0.6b tail shards (10 GB cap) + eval + held-out baseline"
{ $G --name t2_embed_qwen06_tail --max-gb 10 -- $PY embed_corpus.py --model qwen3-0.6b --max-seq 8192 --batch-size 2 \
  && AUDIT_DEVICE=cpu $G --name t2_eval_qwen06 --cpu-only -- $PY eval_corpus.py --model qwen3-0.6b --max-seq 8192 --vec-dir $V/qwen3-0.6b__all__8192 \
  && AUDIT_DEVICE=cpu $G --name p1_base_holdout --cpu-only -- $PY eval_corpus.py --model qwen3-0.6b --max-seq 8192 \
       --vec-dir $V/qwen3-0.6b__all__8192 --exclude-ids $PILOT/train_ids.json --tag holdout_base; } || echo "!!! STEP FAILED: t2 qwen06 tail"
echo "=== $(date '+%m-%d %H:%M:%S') queue v6b finished"
