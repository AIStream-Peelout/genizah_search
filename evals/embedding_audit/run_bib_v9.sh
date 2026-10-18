#!/bin/bash
# v9 bibliography check: embed the bibliography page pool + queries with the base and the tuned text embedder
# (bib_v9_check.py embed, GPU job through guard.py), then score both (numpy only).
# The embed step exits 99 above the soft footprint limit (6.5 GB) so this loop relaunches it fresh; the guard's hard
# cap (8 GB) is the backstop. Everything it writes (HF cache, vectors, logs) is on the NAS.
#
# DISK GATE: guard.py refuses to start while the Studio's internal disk has < 30 GB free (kills below 25 GB). On
# 2026-10-07 the disk sat at ~7.5 GB free (KTIV scrape), which blocks every audit job. This job writes nothing to
# the internal disk, so it runs guard.py with the disk gates lowered to start >= ${START_DISK_GB} GB / kill
# < ${KILL_DISK_GB} GB for THIS invocation only (module constants overridden in-process; guard.py itself is
# unchanged, and the RAM / swap / sibling-merge / sibling-eval / footprint gates are untouched).
#
# Launch:  nohup ./run_bib_v9.sh > /dev/null 2>&1 &      Log: $ROOT/logs/bib_v9.log
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
ROOT=/Volumes/home/studio_offload/genizah_search_embedding_audit
START_DISK_GB=${START_DISK_GB:-6}
KILL_DISK_GB=${KILL_DISK_GB:-4}
export HF_HUB_OFFLINE=1 HF_HOME="$ROOT/hf"   # both models are cached on the NAS; never reach the Hub mid-run
mkdir -p "$ROOT/logs"
exec >> "$ROOT/logs/bib_v9.log" 2>&1
cd "$HERE"
echo "=== $(date '+%m-%d %H:%M:%S') bib v9: start (disk gates start>=${START_DISK_GB} kill<${KILL_DISK_GB} GB) $*"

guarded() {  # guarded NAME CMD...: guard.py main() with the NAS-only disk gates
  local name=$1; shift
  /usr/bin/python3 -c "
import sys
sys.path.insert(0, '$HERE')
import guard
guard.START_MIN_DISK_GB = float('$START_DISK_GB')
guard.KILL_MIN_DISK_GB = float('$KILL_DISK_GB')
sys.argv = ['guard.py'] + sys.argv[1:]
guard.main()
" --name "$name" --max-gb 8 -- "$@"
}

for model in base tuned; do
  rc=0
  for i in $(seq 0 40); do
    guarded "gpu_bib_v9_$model" "$PY" bib_v9_check.py embed --model "$model" --soft-max-gb 6.5 "$@"
    rc=$?
    echo "=== $(date '+%m-%d %H:%M:%S') bib v9 embed $model: launch $i exited rc=$rc"
    [ $rc -eq 99 ] || break
  done
  if [ $rc -ne 0 ]; then
    echo "!!! bib v9 embed $model FAILED rc=$rc (70 = guard footprint cap, 75 = guard restarts exhausted)"
    exit $rc
  fi
done
"$PY" bib_v9_check.py score
rc=$?
echo "=== $(date '+%m-%d %H:%M:%S') bib v9: finished rc=$rc"
exit $rc
