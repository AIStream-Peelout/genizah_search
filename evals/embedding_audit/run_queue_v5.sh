#!/bin/bash
# Queue v5 — CPU-only remainder of the audit, strictly one process at a time (guard mutex),
# 4 threads, so the sibling's GPU consensus/eval jobs are not slowed. GPU steps are NOT here:
# they run only in a window agreed with the sibling session.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
G="/usr/bin/python3 $HERE/guard.py"
cd "$HERE"
export AUDIT_DEVICE=cpu AUDIT_THREADS=4
step() { echo "=== $(date '+%m-%d %H:%M:%S') $1"; }
# let the older CPU queues (line pilot, text embeds) drain first; the mutex serialises them anyway
while pgrep -f "run_cpu_lines.sh" >/dev/null; do sleep 60; done
step "CPU I1 features qwen3vl-vit512 masked (VLM image features, 512px budget)"
$G --name cpu_i1_qwenvit512_masked --cpu-only --max-gb 6 -- $PY image_features.py --encoder qwen3vl-vit512 --view masked \
  || echo "!!! STEP FAILED: qwenvit512 masked"
step "CPU I1 features clip-b32 masked"
$G --name cpu_i1_clip_masked --cpu-only --max-gb 5 -- $PY image_features.py --encoder clip-b32 --view masked \
  || echo "!!! STEP FAILED: clip masked"
step "CPU I1 eval (all feature sets)"
$G --name cpu_i1_eval --cpu-only --max-gb 6 -- $PY eval_images.py || echo "!!! STEP FAILED: i1 eval"
step "queue v5 finished"
