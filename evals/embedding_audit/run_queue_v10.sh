#!/bin/bash
# Queue v10 — fragment-level I1 with the v22b tower read from deepstack layer 8 instead of the final merger.
# Why: on single lines (I2) the merger output scored P@1 0.067 (it encodes WHAT is written) while layer 8 scored
# 0.371 (HOW it is written), beating DINOv2 (0.344). Fragment-level features so far used the merger readout.
# GPU (MPS) job through the guard (sibling OK'd ~1-2 h of MPS tonight next to its consensus pipeline);
# the guard makes CPU queue steps yield to it. Eval is small and CPU-only.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/isaac/Documents/GitHub/historical-document-analysis/.venv/bin/python
G="/usr/bin/python3 $HERE/guard.py"
ROOT=/Volumes/home/studio_offload/genizah_search_embedding_audit
exec >> "$ROOT/logs/queue_v10.log" 2>&1
cd "$HERE"
export AUDIT_IMG_DIR="${AUDIT_IMG_DIR:-$HOME/audit_local/images/max1600}"   # local mirror (the NAS share is on Wi-Fi)
T=qwen3vl-vit512-heb-v22b-step1200@ds8
echo "=== $(date '+%m-%d %H:%M:%S') MPS features $T masked"
$G --name gpu_i1_tower_v22b_ds8 --max-gb 8 -- $PY image_features.py --encoder "$T" --view masked || echo "!!! STEP FAILED: features $T"
echo "=== $(date '+%m-%d %H:%M:%S') eval merger vs ds8 readout (alone and fused with DINOv2)"
$G --name i1_eval_ds8 --small -- $PY eval_images.py --features "qwen3vl-vit512-heb-v22b-step1200__masked,${T}__masked,dinov2-base__masked+dinov2-base__mpatches+qwen3vl-vit512-heb-v22b-step1200__masked,dinov2-base__masked+dinov2-base__mpatches+${T}__masked" || echo "!!! STEP FAILED: eval ds8"
echo "=== $(date '+%m-%d %H:%M:%S') queue v10 finished"
