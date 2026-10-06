#!/bin/bash
# Pilot v3 (local LoRA of the v3 recipe on the real Qwen3-Embedding-0.6B, MPS): pilot_v3_local.py through the guard.
# GPU job: the guard holds the audit mutex, yields to the sibling's checkpoint evals and kills on its merges (the
# pilot resumes from its shards / checkpoints when the guard relaunches it). The pilot exits 99 above the soft
# footprint limit (6.5 GB) so this loop relaunches it fresh; the guard's hard cap (8 GB) is the backstop.
# Extra arguments are passed to pilot_v3_local.py (e.g. --stop-after embed-base, --n-train 12800).
# Launch:  nohup ./run_pilot_v3_local.sh > /dev/null 2>&1 &      Log: $ROOT/logs/pilot_v3.log
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
ROOT=/Volumes/home/studio_offload/genizah_search_embedding_audit
export HF_HUB_OFFLINE=1 HF_HOME="$ROOT/hf"   # weights are local; never reach the Hub mid-run
mkdir -p "$ROOT/logs"
exec >> "$ROOT/logs/pilot_v3.log" 2>&1
cd "$HERE"
echo "=== $(date '+%m-%d %H:%M:%S') pilot v3 local: start $*"
rc=0
for i in $(seq 0 40); do
  /usr/bin/python3 "$HERE/guard.py" --name gpu_pilot_v3 --max-gb 8 -- "$PY" pilot_v3_local.py --soft-max-gb 6.5 "$@"
  rc=$?
  echo "=== $(date '+%m-%d %H:%M:%S') pilot v3 local: launch $i exited rc=$rc"
  [ $rc -eq 99 ] || break
done
[ $rc -eq 0 ] || echo "!!! pilot v3 local FAILED rc=$rc (70 = guard footprint cap, 75 = guard restarts exhausted)"
echo "=== $(date '+%m-%d %H:%M:%S') pilot v3 local: finished rc=$rc"
exit $rc
