#!/bin/bash
# CPU: fine-tuned Hebrew VLM vision towers (sibling checkpoints) as image encoders, vs the stock tower.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
G="/usr/bin/python3 $HERE/guard.py"
cd "$HERE"
export AUDIT_DEVICE=cpu AUDIT_THREADS=4
for ck in heb-v21b-step1200 heb-v22b-step1200; do
  echo "=== $(date '+%m-%d %H:%M:%S') CPU features qwen3vl-vit512-$ck masked"
  $G --name "cpu_i1_tower_$ck" --cpu-only --max-gb 6 -- $PY image_features.py --encoder "qwen3vl-vit512-$ck" --view masked || echo "!!! STEP FAILED: $ck"
done
echo "=== $(date '+%m-%d %H:%M:%S') eval stock vs fine-tuned towers"
$G --name i1_eval_towers --cpu-only --max-gb 6 -- $PY eval_images.py --features "qwen3vl-vit512__masked,qwen3vl-vit512-heb-v21b-step1200__masked,qwen3vl-vit512-heb-v22b-step1200__masked,dinov2-base__masked+dinov2-base__mpatches+qwen3vl-vit512-heb-v21b-step1200__masked"
echo "=== $(date '+%m-%d %H:%M:%S') queue v8 finished"
