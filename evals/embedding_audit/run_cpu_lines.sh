#!/bin/bash
# CPU-only: line-level handwriting probe (I2) for DINOv2/CLIP, then the supervised head pilot (I3).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
G="/usr/bin/python3 $HERE/guard.py"
cd "$HERE"
export AUDIT_DEVICE=cpu AUDIT_THREADS=4
echo "=== $(date '+%m-%d %H:%M:%S') CPU I2 line probe (dinov2-base, clip-b32)"
$G --name cpu_i2_lines --cpu-only --max-gb 5 -- $PY line_probe.py --encoders dinov2-base,clip-b32 || echo "!!! STEP FAILED: cpu i2"
echo "=== $(date '+%m-%d %H:%M:%S') CPU I3 line head pilot"
$G --name cpu_i3_head --cpu-only --max-gb 5 -- $PY line_head_pilot.py || echo "!!! STEP FAILED: cpu i3"
echo "=== $(date '+%m-%d %H:%M:%S') CPU line queue finished"
