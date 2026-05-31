#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Train DETR from the project root.
# Run:  bash scripts/train.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

CONFIG="configs/train.yaml"

# ── Option 1: Single GPU, CPU, or Apple Silicon (MPS) ───────────────────────
python -m src.train --config "$CONFIG"

# ── Option 2: Multi-GPU DDP (uncomment and set nproc_per_node) ──────────────
# torchrun --nproc_per_node=4 -m src.train --config "$CONFIG"

# ── Option 3: Resume from a checkpoint ──────────────────────────────────────
# python -m src.train --config "$CONFIG" --resume checkpoints/detr_best.pth
