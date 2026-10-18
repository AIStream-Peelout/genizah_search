#!/bin/bash
# Embedding-audit probe queue (supersedes v1-v3). Strictly sequential; every GPU step runs under
# guard.py, which yields to the sibling's merge/convert AND to its GPU-bound checkpoint evals.
# Steps are resumable/idempotent; a failed embed skips its dependent eval (&&).
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

step "T2 qwen3-0.6b embed + eval (production setting, max_seq 8192)"
{ $G --name t2_embed_qwen06 -- $PY embed_corpus.py --model qwen3-0.6b --max-seq 8192 --batch-size 8 \
  && $G --name t2_eval_qwen06 -- $PY eval_corpus.py --model qwen3-0.6b --max-seq 8192 --vec-dir $V/qwen3-0.6b__all__8192; } || fail t2_qwen06

step "T2 BM25 baseline (CPU)"
$G --name t2_bm25 --cpu-only --max-gb 6 -- $PY eval_bm25.py || fail bm25

for spec in "dinov2-base crop" "dinov2-base global" "dinov2-base patches" "clip-b32 global" "clip-b32 crop" \
            "siglip2-so400m global" "siglip2-so400m crop" "qwen3vl-vit global" "qwen3vl-vit crop" "dinov2-large patches"; do
  set -- $spec
  step "I1 features $1 $2"
  $G --name "i1_$1_$2" -- $PY image_features.py --encoder "$1" --view "$2" || fail "i1 $1 $2"
done
step "I1 eval"
$G --name i1_eval --cpu-only -- $PY eval_images.py || fail i1_eval

step "I2 line-level handwriting probe"
$G --name i2_lines -- $PY line_probe.py || fail i2

step "P1 pilot: baseline on held-out docs"
$G --name p1_base_holdout -- $PY eval_corpus.py --model qwen3-0.6b --max-seq 8192 \
   --vec-dir $V/qwen3-0.6b__all__8192 --exclude-ids $PILOT/train_ids.json --tag holdout_base || fail p1_base
step "P1 pilot: smoke train then LoRA train (corpus-only pairs)"
{ $G --name p1_smoke --max-gb 9 --max-restarts 1 -- $PY train_text_lora.py --limit 64 --batch 16 --out $PILOT/smoke_model \
  && $G --name p1_train --max-gb 9 --max-restarts 3 -- $PY train_text_lora.py \
  && $G --name p1_t1 -- $PY probe_text_concepts.py --local-path $PILOT/model_lora_merged --models qwen3-0.6b-pilot1 \
  && $G --name p1_embed -- $PY embed_corpus.py --model qwen3-0.6b-pilot1 --local-path $PILOT/model_lora_merged --max-seq 8192 --batch-size 8 \
  && $G --name p1_eval -- $PY eval_corpus.py --model qwen3-0.6b-pilot1 --local-path $PILOT/model_lora_merged --max-seq 8192 \
       --vec-dir $V/qwen3-0.6b-pilot1__all__8192 --exclude-ids $PILOT/train_ids.json --tag holdout_pilot1; } || fail pilot1

step "T2 bge-m3 embed + eval"
{ $G --name t2_embed_bgem3 -- $PY embed_corpus.py --model bge-m3 --max-seq 8192 --batch-size 8 \
  && $G --name t2_eval_bgem3 -- $PY eval_corpus.py --model bge-m3 --max-seq 8192 --vec-dir $V/bge-m3__all__8192; } || fail t2_bgem3
step "T2 e5-large-instruct embed + eval"
{ $G --name t2_embed_e5 -- $PY embed_corpus.py --model e5-large-instruct --max-seq 512 --batch-size 16 \
  && $G --name t2_eval_e5 -- $PY eval_corpus.py --model e5-large-instruct --max-seq 512 --vec-dir $V/e5-large-instruct__all__512; } || fail t2_e5

step "T1+T2 Qwen3-Embedding-4B (fp16, 10 GB cap)"
{ $G --name t1_qwen4b --max-gb 10 -- $PY probe_text_concepts.py --models qwen3-4b \
  && $G --name t2_embed_qwen4b --max-gb 10 -- $PY embed_corpus.py --model qwen3-4b --max-seq 2048 --batch-size 4 \
  && $G --name t2_eval_qwen4b --max-gb 10 -- $PY eval_corpus.py --model qwen3-4b --max-seq 2048 --vec-dir $V/qwen3-4b__all__2048; } || fail qwen4b
step "queue v4 finished"
