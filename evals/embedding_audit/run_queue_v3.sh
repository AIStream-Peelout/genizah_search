#!/bin/bash
# Queue v3 (runs after v2): line-level handwriting probe on local KTIV line crops.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
G="/usr/bin/python3 $HERE/guard.py"
cd "$HERE"
step() { echo "=== $(date '+%m-%d %H:%M:%S') $1"; }
while pgrep -f "run_queue_v[12].sh" >/dev/null; do sleep 60; done
step "I2 line-level handwriting probe"
$G --name i2_lines -- $PY line_probe.py
step "queue v3 finished"
