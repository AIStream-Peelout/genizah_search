#!/bin/bash
# Sequential, guarded probe queue for the embedding audit (one probe at a time).
# Each step runs under guard.py (memory/disk/sibling-merge gates, resumable relaunch).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
G="/usr/bin/python3 $HERE/guard.py"
ROOT=/Volumes/home/studio_offload/genizah_search_embedding_audit
cd "$HERE"

step() { echo "=== $(date '+%m-%d %H:%M:%S') $1"; }

# wait for the concept probe of the other models to finish (started separately)
while pgrep -f "probe_text_concepts.py" >/dev/null; do sleep 30; done

step "T2 qwen3-0.6b embed (8192 = production setting)"
$G --name t2_embed_qwen06 -- $PY embed_corpus.py --model qwen3-0.6b --max-seq 8192 --batch-size 8
step "T2 qwen3-0.6b eval"
$G --name t2_eval_qwen06 -- $PY eval_corpus.py --model qwen3-0.6b --max-seq 8192 --vec-dir $ROOT/corpus_vectors/qwen3-0.6b__all__8192

step "T2 bge-m3 embed"
$G --name t2_embed_bgem3 -- $PY embed_corpus.py --model bge-m3 --max-seq 8192 --batch-size 8
step "T2 bge-m3 eval"
$G --name t2_eval_bgem3 -- $PY eval_corpus.py --model bge-m3 --max-seq 8192 --vec-dir $ROOT/corpus_vectors/bge-m3__all__8192

step "T2 e5-large-instruct embed"
$G --name t2_embed_e5 -- $PY embed_corpus.py --model e5-large-instruct --max-seq 512 --batch-size 16
step "T2 e5 eval"
$G --name t2_eval_e5 -- $PY eval_corpus.py --model e5-large-instruct --max-seq 512 --vec-dir $ROOT/corpus_vectors/e5-large-instruct__all__512

# images: wait for downloads
while pgrep -f "fetch_images.py" >/dev/null; do sleep 30; done
for spec in "dinov2-base global" "dinov2-base crop" "dinov2-base patches" "clip-b32 global" "clip-b32 crop" \
            "siglip2-so400m global" "siglip2-so400m crop" "qwen3vl-vit global" "qwen3vl-vit crop" "dinov2-large patches"; do
  set -- $spec
  step "I1 features $1 $2"
  $G --name "i1_$1_$2" -- $PY image_features.py --encoder "$1" --view "$2"
done
step "I1 eval"
$G --name i1_eval -- $PY eval_images.py
step "queue v1 finished"
