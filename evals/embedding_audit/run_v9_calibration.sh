#!/bin/bash
# v9 cosine calibration (v9_cosine_calibration.py): a --small smoke test, then the embed phase (tuned model on the
# eligible pool + both models' queries) through guard.py, then the numpy-only contract and analyze phases.
# GPU job: the guard holds the audit mutex, yields to the sibling's checkpoint evals, kills on its merges and waits
# while the box is under pressure (including its disk gate); the embed phase resumes from its shards and exits 99
# above the soft footprint limit (6.5 GB) so this loop relaunches it fresh; the guard's hard cap (8 GB) is the backstop.
# Launch:  nohup ./run_v9_calibration.sh > /dev/null 2>&1 &      Log: $ROOT/logs/v9_calibration.log
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
ROOT=/Volumes/home/studio_offload/genizah_search_embedding_audit
export HF_HUB_OFFLINE=1 HF_HOME="$ROOT/hf"   # both models are cached on the NAS; never reach the Hub mid-run
mkdir -p "$ROOT/logs"
exec >> "$ROOT/logs/v9_calibration.log" 2>&1
cd "$HERE"
echo "=== $(date '+%m-%d %H:%M:%S') v9 calibration: start $*"
if [ ! -f "$ROOT/v9_calibration_smoke/v9_cosine_calibration.json" ]; then
  /usr/bin/python3 "$HERE/guard.py" --name v9_calib_smoke --small -- \
    "$PY" v9_cosine_calibration.py embed --smoke 300 --smoke-layers 2 --canary-min-cos 0.99 \
    && "$PY" v9_cosine_calibration.py analyze --smoke
  rc=$?
  echo "=== $(date '+%m-%d %H:%M:%S') v9 calibration smoke rc=$rc"
  [ $rc -eq 0 ] || { echo "!!! v9 calibration smoke FAILED rc=$rc"; exit $rc; }
fi
rc=0
for i in $(seq 0 40); do
  /usr/bin/python3 "$HERE/guard.py" --name gpu_v9_calib --max-gb 8 -- \
    "$PY" v9_cosine_calibration.py embed --soft-max-gb 6.5 "$@"
  rc=$?
  echo "=== $(date '+%m-%d %H:%M:%S') v9 calibration embed: launch $i exited rc=$rc"
  [ $rc -eq 99 ] || break
done
if [ $rc -ne 0 ]; then
  echo "!!! v9 calibration embed FAILED rc=$rc (70 = guard footprint cap, 75 = guard restarts exhausted)"
  exit $rc
fi
"$PY" v9_cosine_calibration.py contract; rc=$?
echo "=== $(date '+%m-%d %H:%M:%S') v9 calibration contract rc=$rc (see corpus_vectors/genizah-v3final__semantic__8192/contract_vs_colab.json)"
"$PY" v9_cosine_calibration.py analyze; rc=$?
echo "=== $(date '+%m-%d %H:%M:%S') v9 calibration: finished rc=$rc"
exit $rc
