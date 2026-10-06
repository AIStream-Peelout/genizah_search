#!/bin/bash
# Queue v2 (runs after v1): bigger off-the-shelf embedder + corpus-only LoRA pilot.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
G="/usr/bin/python3 $HERE/guard.py"
ROOT=/Volumes/home/studio_offload/genizah_search_embedding_audit
cd "$HERE"
step() { echo "=== $(date '+%m-%d %H:%M:%S') $1"; }

while pgrep -f "run_queue_v1.sh" >/dev/null; do sleep 60; done

PILOT=$ROOT/pilot_text_v1
step "P1 baseline on held-out docs (qwen3-0.6b, same vectors as T2)"
$G --name p1_base_holdout -- $PY eval_corpus.py --model qwen3-0.6b --max-seq 8192 \
   --vec-dir $ROOT/corpus_vectors/qwen3-0.6b__all__8192 --exclude-ids $PILOT/train_ids.json --tag holdout_base

step "P1 LoRA smoke train (64 pairs)"
$G --name p1_smoke --max-gb 9 --max-restarts 1 -- $PY train_text_lora.py --limit 64 --batch 16 --out $PILOT/smoke_model \
  && { step "P1 LoRA pilot train (corpus-only pairs)"; $G --name p1_train --max-gb 9 --max-restarts 3 -- $PY train_text_lora.py; }
if [ -f "$PILOT/model_lora_merged/config.json" ]; then
  step "P1 pilot: concept probe"
  $G --name p1_t1 -- $PY probe_text_concepts.py --local-path $PILOT/model_lora_merged --models qwen3-0.6b-pilot1
  step "P1 pilot: embed corpus"
  $G --name p1_embed -- $PY embed_corpus.py --model qwen3-0.6b-pilot1 --local-path $PILOT/model_lora_merged --max-seq 8192 --batch-size 8
  step "P1 pilot: eval held-out"
  $G --name p1_eval -- $PY eval_corpus.py --model qwen3-0.6b-pilot1 --local-path $PILOT/model_lora_merged --max-seq 8192 \
     --vec-dir $ROOT/corpus_vectors/qwen3-0.6b-pilot1__all__8192 --exclude-ids $PILOT/train_ids.json --tag holdout_pilot1
fi

step "T1+T2 Qwen3-Embedding-4B (fp16, 10 GB cap)"
$G --name t1_qwen4b --max-gb 10 -- $PY probe_text_concepts.py --models qwen3-4b
$G --name t2_embed_qwen4b --max-gb 10 -- $PY embed_corpus.py --model qwen3-4b --max-seq 2048 --batch-size 4
$G --name t2_eval_qwen4b --max-gb 10 -- $PY eval_corpus.py --model qwen3-4b --max-seq 2048 --vec-dir $ROOT/corpus_vectors/qwen3-4b__all__2048
step "queue v2 finished"
