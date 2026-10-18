#!/bin/bash
# CPU-only: backdrop-masked views (colour segmentation) for DINOv2 — the confound fix.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
G="/usr/bin/python3 $HERE/guard.py"
cd "$HERE"
export AUDIT_DEVICE=cpu AUDIT_THREADS=4
for spec in "dinov2-base masked" "dinov2-base mpatches" "clip-b32 masked"; do
  set -- $spec
  echo "=== $(date '+%m-%d %H:%M:%S') CPU2 I1 features $1 $2"
  $G --name "cpu2_i1_$1_$2" --cpu-only --max-gb 5 -- $PY image_features.py --encoder "$1" --view "$2" || echo "!!! STEP FAILED: cpu2 $1 $2"
done
echo "=== $(date '+%m-%d %H:%M:%S') CPU2 image queue finished"
