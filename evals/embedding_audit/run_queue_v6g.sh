#!/bin/bash
# Pilot T2 tail: finish the longest shards on CPU (MPS overshoots the footprint cap there), then held-out eval.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
G="/usr/bin/python3 $HERE/guard.py"
V=/Volumes/home/studio_offload/genizah_search_embedding_audit/corpus_vectors
PILOT=/Volumes/home/studio_offload/genizah_search_embedding_audit/pilot_text_v1
cd "$HERE"
while pgrep -f "run_queue_v6f.sh" >/dev/null; do sleep 30; done
rm -f $V/qwen3-0.6b-pilot1__all__8192/LOCK
echo "=== $(date '+%m-%d %H:%M:%S') P1 pilot tail shards on CPU + held-out eval"
{ AUDIT_DEVICE=cpu AUDIT_THREADS=4 $G --name p1_embed_tail_cpu --cpu-only --max-gb 9 -- $PY embed_corpus.py --model qwen3-0.6b-pilot1 \
     --local-path $PILOT/model_lora_merged --max-seq 8192 --batch-size 1 \
  && AUDIT_DEVICE=cpu $G --name p1_eval --cpu-only -- $PY eval_corpus.py --model qwen3-0.6b-pilot1 --local-path $PILOT/model_lora_merged \
     --max-seq 8192 --vec-dir $V/qwen3-0.6b-pilot1__all__8192 --exclude-ids $PILOT/train_ids.json --tag holdout_pilot1; } || echo "!!! STEP FAILED: pilot tail"
echo "=== $(date '+%m-%d %H:%M:%S') queue v6g finished"
