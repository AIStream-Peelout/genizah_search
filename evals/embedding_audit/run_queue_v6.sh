#!/bin/bash
# Queue v6 — GPU steps, agreed with the sibling session: one process at a time (guard mutex),
# only between its checkpoint-eval windows (guard yields to them), ~2-3 GPU-hours in total.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
G="/usr/bin/python3 $HERE/guard.py"
ROOT=/Volumes/home/studio_offload/genizah_search_embedding_audit
V=$ROOT/corpus_vectors
PILOT=$ROOT/pilot_text_v1
cd "$HERE"
step() { echo "=== $(date '+%m-%d %H:%M:%S') $1"; }
fail() { echo "!!! STEP FAILED: $1"; }
while pgrep -f "run_cpu_lines.sh" >/dev/null; do sleep 30; done

step "T2 qwen3-0.6b: finish last long shards on GPU, then eval (+ held-out baseline for the pilot)"
{ $G --name t2_embed_qwen06 -- $PY embed_corpus.py --model qwen3-0.6b --max-seq 8192 --batch-size 8 \
  && AUDIT_DEVICE=cpu $G --name t2_eval_qwen06 --cpu-only -- $PY eval_corpus.py --model qwen3-0.6b --max-seq 8192 --vec-dir $V/qwen3-0.6b__all__8192 \
  && AUDIT_DEVICE=cpu $G --name p1_base_holdout --cpu-only -- $PY eval_corpus.py --model qwen3-0.6b --max-seq 8192 \
       --vec-dir $V/qwen3-0.6b__all__8192 --exclude-ids $PILOT/train_ids.json --tag holdout_base; } || fail t2_qwen06

step "P1 LoRA pilot (smoke, train, probe, embed, held-out eval)"
{ $G --name p1_smoke --max-gb 9 --max-restarts 1 -- $PY train_text_lora.py --limit 64 --batch 16 --out $PILOT/smoke_model \
  && $G --name p1_train --max-gb 9 --max-restarts 4 -- $PY train_text_lora.py \
  && $G --name p1_t1 -- $PY probe_text_concepts.py --local-path $PILOT/model_lora_merged --models qwen3-0.6b-pilot1 \
  && $G --name p1_embed -- $PY embed_corpus.py --model qwen3-0.6b-pilot1 --local-path $PILOT/model_lora_merged --max-seq 8192 --batch-size 8 \
  && AUDIT_DEVICE=cpu $G --name p1_eval --cpu-only -- $PY eval_corpus.py --model qwen3-0.6b-pilot1 --local-path $PILOT/model_lora_merged --max-seq 8192 \
       --vec-dir $V/qwen3-0.6b-pilot1__all__8192 --exclude-ids $PILOT/train_ids.json --tag holdout_pilot1; } || fail pilot1

step "Qwen3-Embedding-4B: concept probe + framed-pool retrieval (subset keeps GPU time small)"
{ $G --name t1_qwen4b --max-gb 10 -- $PY probe_text_concepts.py --models qwen3-4b \
  && $G --name t2_embed_qwen4b --max-gb 10 -- $PY embed_corpus.py --model qwen3-4b --max-seq 2048 --batch-size 4 --subset framed \
  && $G --name t2_embed_qwen06_framed -- $PY embed_corpus.py --model qwen3-0.6b --max-seq 2048 --batch-size 8 --subset framed \
  && AUDIT_DEVICE=cpu $G --name t2_eval_qwen4b --cpu-only --max-gb 10 -- $PY eval_corpus.py --model qwen3-4b --max-seq 2048 --vec-dir $V/qwen3-4b__framed__2048 --tag framed \
  && AUDIT_DEVICE=cpu $G --name t2_eval_qwen06_framed --cpu-only -- $PY eval_corpus.py --model qwen3-0.6b --max-seq 2048 --vec-dir $V/qwen3-0.6b__framed__2048 --tag framed; } || fail qwen4b
step "queue v6 finished"
