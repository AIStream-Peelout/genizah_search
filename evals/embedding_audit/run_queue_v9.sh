#!/bin/bash
# Queue v9 — CPU: Qwen3-VL vision towers (fine-tuned Hebrew v22b — best fragment-level scribe encoder — then stock) as LINE-level handwriting
# encoders on the KTIV Kraken line crops. Whole line through the tower at a fixed height (see the
# line_probe.py docstring); one forward gives all readouts (merger + deepstack ds8/ds16/ds24).
#   I2: line_probe on tower readouts + DINOv2 fusion (extracts the 3,600 test lines)
#   I3: line_head_pilot on frozen tower / fusion features (extracts the 9,087 train lines once,
#       then every readout/fusion head is a cache hit)
#   Last, optional: I2-only height ablation for v22b (test lines only).
# One process at a time (guard mutex), 4 threads, never MPS. Extraction is resumable (parts) after
# guard kills; re-running the queue skips finished extraction.
# CPU budget (20-line smoke, 4 threads: tower h96 0.26 s/line, h64 0.15 s/line, + ~0.07 s/line NAS read):
# per tower ~20 min test + ~50 min train extraction + ~15 min probes/heads => ~2.9 h for both towers,
# ~3.2 h with the default h64 ablation (h128 would add ~35 min more).
# Knobs: TOWERS="qwen3vl-vit512-heb-v21b-step1200" (skip stock), ABLATE_HEIGHTS="" (skip ablation),
#        ABLATE_HEIGHTS="64 128", LINE_HEIGHT=96 (main height).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
G="/usr/bin/python3 $HERE/guard.py"
ROOT=/Volumes/home/studio_offload/genizah_search_embedding_audit
H=${LINE_HEIGHT:-96}
TOWERS=${TOWERS-"qwen3vl-vit512-heb-v22b-step1200 qwen3vl-vit512"}
ABLATE_HEIGHTS=${ABLATE_HEIGHTS-"64"}
mkdir -p "$ROOT/logs"
exec >> "$ROOT/logs/queue_v9.log" 2>&1
cd "$HERE"
export AUDIT_DEVICE=cpu AUDIT_THREADS=4
export AUDIT_LINES_LOCAL="${AUDIT_LINES_LOCAL:-$HOME/audit_local/kraken_lines}"   # local mirror of the selected line crops
step() { echo "=== $(date '+%m-%d %H:%M:%S') $1"; }
fail() { echo "!!! STEP FAILED: $1"; }
short() { local s=${1#qwen3vl-vit512-heb-}; s=${s%-step*}; [ "$1" = qwen3vl-vit512 ] && s=stock; echo "$s"; }

for T in $TOWERS; do
  L=$(short "$T")
  step "I2 line probe $T h$H (extracts test lines): merger, ds8, ds16, ds24, dinov2-base fusion"
  $G --name "cpu_i2_tower_$L" --cpu-only --max-gb 6 -- $PY line_probe.py --height "$H" \
     --encoders "$T,$T:ds8,$T:ds16,$T:ds24,dinov2-base+$T" || fail "i2 $L"
  step "I3 head pilot $T h$H merger (extracts train lines)"
  $G --name "cpu_i3_tower_$L" --cpu-only --max-gb 6 -- $PY line_head_pilot.py --encoder "$T" --height "$H" || fail "i3 $L merger"
  for spec in "$T:ds8" "$T:ds16" "$T:ds24" "dinov2-base+$T"; do
    step "I3 head pilot $spec h$H (cached features)"
    $G --name "cpu_i3_head_$L" --cpu-only --max-gb 6 -- $PY line_head_pilot.py --encoder "$spec" --height "$H" || fail "i3 $spec"
  done
done

T=qwen3vl-vit512-heb-v22b-step1200
for AH in $ABLATE_HEIGHTS; do
  step "I2 height ablation $T h$AH (test lines only): merger, ds16, dinov2-base fusion"
  $G --name "cpu_i2_tower_v22b_h$AH" --cpu-only --max-gb 6 -- $PY line_probe.py --height "$AH" \
     --encoders "$T,$T:ds16,dinov2-base+$T" || fail "i2 ablation h$AH"
done
step "queue v9 finished"
