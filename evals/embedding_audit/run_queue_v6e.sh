#!/bin/bash
# Re-run of the LoRA pilot after the ST-5 PEFT wrapping fix (GPU, mutex, between sibling evals).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
G="/usr/bin/python3 $HERE/guard.py"
V=/Volumes/home/studio_offload/genizah_search_embedding_audit/corpus_vectors
PILOT=/Volumes/home/studio_offload/genizah_search_embedding_audit/pilot_text_v1
cd "$HERE"

echo "=== $(date '+%m-%d %H:%M:%S') P1 LoRA pilot (smoke, train, probe, embed, held-out eval)"
{ $G --name p1_smoke --max-gb 10 --max-restarts 1 -- $PY train_text_lora.py --limit 64 --batch 16 --out $PILOT/smoke_model \
  && $G --name p1_train --max-gb 10 --max-restarts 4 -- $PY train_text_lora.py \
  && $G --name p1_t1 -- $PY probe_text_concepts.py --local-path $PILOT/model_lora_merged --models qwen3-0.6b-pilot1 \
  && $G --name p1_embed --max-gb 10 -- $PY embed_corpus.py --model qwen3-0.6b-pilot1 --local-path $PILOT/model_lora_merged --max-seq 8192 --batch-size 4 \
  && AUDIT_DEVICE=cpu $G --name p1_eval --cpu-only -- $PY eval_corpus.py --model qwen3-0.6b-pilot1 --local-path $PILOT/model_lora_merged --max-seq 8192 \
       --vec-dir $V/qwen3-0.6b-pilot1__all__8192 --exclude-ids $PILOT/train_ids.json --tag holdout_pilot1; } || echo "!!! STEP FAILED: pilot1"
echo "=== $(date '+%m-%d %H:%M:%S') queue v6e finished"
