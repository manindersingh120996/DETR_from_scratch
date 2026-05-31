#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run DETR inference on a single image from the validation split.
# Run:  bash scripts/inference.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

CONFIG="configs/train.yaml"
CHECKPOINT="checkpoints/detr_best.pth"
IDX=0              # dataset index to visualise
THRESHOLD=0.3      # confidence threshold

python -m src.inference \
    --config      "$CONFIG"     \
    --checkpoint  "$CHECKPOINT" \
    --idx         "$IDX"        \
    --threshold   "$THRESHOLD"  \
    --save        "outputs/inference_result.png"
