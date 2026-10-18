#!/bin/bash
# CPU-only image feature extraction (no GPU use) so image probes progress while the
# sibling's GPU-bound checkpoint evals run. 6 threads to leave CPU for everything else.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
G="/usr/bin/python3 $HERE/guard.py"
cd "$HERE"
export AUDIT_DEVICE=cpu AUDIT_THREADS=6
for spec in "dinov2-base crop" "clip-b32 crop" "clip-b32 global" "dinov2-base global" "dinov2-base patches" \
            "siglip2-so400m crop" "siglip2-so400m global"; do
  set -- $spec
  echo "=== $(date '+%m-%d %H:%M:%S') CPU I1 features $1 $2"
  $G --name "cpu_i1_$1_$2" --cpu-only --max-gb 5 -- $PY image_features.py --encoder "$1" --view "$2" || echo "!!! STEP FAILED: cpu $1 $2"
done
echo "=== $(date '+%m-%d %H:%M:%S') CPU I1 eval"
$G --name cpu_i1_eval --cpu-only --max-gb 6 -- $PY eval_images.py || echo "!!! STEP FAILED: cpu i1 eval"
echo "=== $(date '+%m-%d %H:%M:%S') CPU image queue finished"
