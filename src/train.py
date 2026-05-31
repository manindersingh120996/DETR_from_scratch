"""
DETR training script.

Usage (single GPU / CPU / Apple Silicon MPS):
    python -m src.train --config configs/train.yaml

Usage (multi-GPU DDP, e.g. 4 GPUs):
    torchrun --nproc_per_node=4 -m src.train --config configs/train.yaml

Config keys can be overridden per-run:
    python -m src.train --config configs/train.yaml --epochs 10 --batch_size 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")   # headless-safe; must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.ops import generalized_box_iou

from src.model import DETR
from src.dataset import CocoDataset, collate_fn, _resolve_paths


# ---------------------------------------------------------------------------
# Utility: box format conversion
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """[cx, cy, w, h] → [x1, y1, x2, y2], all normalised."""
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h,
                        cx + 0.5 * w, cy + 0.5 * h], dim=-1)


# ---------------------------------------------------------------------------
# Utility: device detection  (CUDA → MPS → CPU)
# ---------------------------------------------------------------------------

def get_device(local_rank: int = -1) -> torch.device:
    """Return the correct device.

    For DDP runs (local_rank >= 0) returns cuda:<local_rank>.
    For single-process runs: CUDA → MPS → CPU.
    """
    if local_rank >= 0:
        return torch.device(f"cuda:{local_rank}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Utility: reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Utility: distributed helpers
# ---------------------------------------------------------------------------

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def validate_dataset(dataset_path: str, split: str) -> None:
    """Raise a clear error if the dataset layout is not found."""
    try:
        _resolve_paths(dataset_path, split)
    except FileNotFoundError:
        sys.exit(
            f"\n[ERROR] COCO dataset not found at '{dataset_path}'.\n"
            "  Expected either:\n"
            "    <dataset_path>/annotations/instances_{split}2017.json  +  {split}2017/\n"
            "  or (mini layout):\n"
            "    <dataset_path>/annotations.json  +  images/\n\n"
            "  Please download COCO and update 'dataset_path' in your config YAML.\n"
            "  See README.md → Dataset Setup for full instructions.\n"
        )


# ---------------------------------------------------------------------------
# Hungarian Matcher
# ---------------------------------------------------------------------------

class HungarianMatcher:
    """Finds the optimal one-to-one assignment between predictions and GTs.

    Cost = lambda_class * (-softmax_prob) + lambda_l1 * L1 + lambda_giou * (-GIoU)

    Matching runs inside torch.no_grad(); gradients never flow through it.
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0,
                 cost_giou: float = 2.0):
        self.cost_class = cost_class
        self.cost_bbox  = cost_bbox
        self.cost_giou  = cost_giou

    @torch.no_grad()
    def __call__(self, outputs: dict, targets: list) -> list:
        bs = outputs["pred_logits"].shape[0]
        out_prob = outputs["pred_logits"].softmax(-1)   # (B, Q, C+1)
        out_bbox = outputs["pred_boxes"]                 # (B, Q, 4)

        indices = []
        for b in range(bs):
            prob      = out_prob[b]           # (Q, C+1)
            bbox      = out_bbox[b]           # (Q, 4)
            tgt_ids   = targets[b]["labels"]  # (M,)
            tgt_boxes = targets[b]["boxes"]   # (M, 4)

            if len(tgt_ids) == 0:
                indices.append((torch.empty(0, dtype=torch.long),
                                torch.empty(0, dtype=torch.long)))
                continue

            # raw softmax prob (not log) for numerical stability in cost matrix
            cost_class = -prob[:, tgt_ids]                               # (Q, M)
            cost_bbox  = torch.cdist(bbox, tgt_boxes, p=1)               # (Q, M)
            cost_giou  = -generalized_box_iou(
                box_cxcywh_to_xyxy(bbox), box_cxcywh_to_xyxy(tgt_boxes)
            )                                                             # (Q, M)

            C = (self.cost_class * cost_class
                 + self.cost_bbox  * cost_bbox
                 + self.cost_giou  * cost_giou)

            row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long),
                torch.as_tensor(col_ind, dtype=torch.long),
            ))
        return indices


# ---------------------------------------------------------------------------
# Set-based criterion (loss function)
# ---------------------------------------------------------------------------

class SetCriterion(nn.Module):
    """Computes DETR loss: CE (all 100 slots) + L1 + GIoU (matched slots only).

    Auxiliary losses from intermediate decoder layers are summed in.
    """

    def __init__(self, num_classes: int, matcher: HungarianMatcher,
                 eos_coef: float = 0.1, lambda_l1: float = 5.0,
                 lambda_giou: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher     = matcher
        self.lambda_l1   = lambda_l1
        self.lambda_giou = lambda_giou

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef   # background class gets 0.1 weight
        self.register_buffer("empty_weight", empty_weight)

    def _compute_single(self, outputs: dict, targets: list,
                        indices: list, num_boxes: int) -> dict:
        """Loss for one set of predictions (reused for final + aux layers)."""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        device = outputs["pred_logits"].device

        # Classification loss — all 100 slots
        target_classes = torch.full((bs, num_queries), self.num_classes,
                                    dtype=torch.long, device=device)
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[b, pred_idx] = targets[b]["labels"][gt_idx]

        loss_ce = F.cross_entropy(
            outputs["pred_logits"].transpose(1, 2),   # (B, C+1, 100)
            target_classes,
            weight=self.empty_weight,
        )

        # Box losses — matched slots only
        src_boxes_list, tgt_boxes_list = [], []
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                src_boxes_list.append(outputs["pred_boxes"][b][pred_idx])
                tgt_boxes_list.append(targets[b]["boxes"][gt_idx])

        if src_boxes_list:
            src_boxes = torch.cat(src_boxes_list)
            tgt_boxes = torch.cat(tgt_boxes_list)
            loss_l1   = F.l1_loss(src_boxes, tgt_boxes, reduction="none").sum() / num_boxes
            giou      = generalized_box_iou(box_cxcywh_to_xyxy(src_boxes),
                                            box_cxcywh_to_xyxy(tgt_boxes))
            loss_giou = (1 - giou.diag()).sum() / num_boxes
        else:
            loss_l1   = torch.tensor(0.0, device=device)
            loss_giou = torch.tensor(0.0, device=device)

        total = loss_ce + self.lambda_l1 * loss_l1 + self.lambda_giou * loss_giou
        return {"loss_ce": loss_ce, "loss_l1": loss_l1,
                "loss_giou": loss_giou, "total": total}

    def forward(self, outputs: dict, targets: list) -> dict:
        # num_boxes is shared across layers so loss scale stays consistent
        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)

        # Final decoder layer
        indices = self.matcher(outputs, targets)
        losses  = self._compute_single(outputs, targets, indices, num_boxes)

        # Auxiliary decoder layers (fresh matching per layer)
        if "aux_outputs" in outputs:
            for aux in outputs["aux_outputs"]:
                aux_indices = self.matcher(aux, targets)
                aux_losses  = self._compute_single(aux, targets, aux_indices, num_boxes)
                for k in ("loss_ce", "loss_l1", "loss_giou", "total"):
                    losses[k] = losses[k] + aux_losses[k]

        return losses


# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------

def train_one_epoch(model, criterion, matcher, optimizer, data_loader,
                    device, epoch, warmup_sched, cfg) -> dict:
    """Run one training epoch. Returns averaged losses and accuracy."""
    model.train()
    running = {"total": 0., "loss_ce": 0., "loss_l1": 0., "loss_giou": 0.}
    correct, total_objs = 0, 0

    for batch in data_loader:
        images, masks, targets = batch
        images  = images.to(device)
        masks   = masks.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        outputs = model(images, masks)
        losses  = criterion(outputs, targets)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_max_norm"])
        optimizer.step()

        if epoch == 0:
            warmup_sched.step()   # linear LR ramp over first epoch batches

        for k in running:
            running[k] += losses[k].item()

        # Per-batch accuracy: compare matched prediction labels to GT labels
        with torch.no_grad():
            final_out = {k: v for k, v in outputs.items() if k != "aux_outputs"}
            indices   = matcher(final_out, targets)
            for b, (pred_idx, gt_idx) in enumerate(indices):
                if len(pred_idx) > 0:
                    pred_cls = outputs["pred_logits"][b][pred_idx].argmax(-1)
                    gt_cls   = targets[b]["labels"][gt_idx]
                    correct    += (pred_cls == gt_cls).sum().item()
                    total_objs += len(gt_idx)

    n = len(data_loader)
    stats = {k: running[k] / n for k in running}
    stats["acc"] = correct / max(total_objs, 1)
    return stats


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, criterion, matcher, data_loader, device) -> dict:
    """Run validation using final decoder output only (no auxiliary losses)."""
    model.eval()
    running = {"total": 0., "loss_ce": 0., "loss_l1": 0., "loss_giou": 0.}
    correct, total_objs = 0, 0

    for batch in data_loader:
        images, masks, targets = batch
        images  = images.to(device)
        masks   = masks.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs   = model(images, masks)
        final_out = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        losses    = criterion(final_out, targets)

        for k in running:
            running[k] += losses[k].item()

        indices = matcher(final_out, targets)
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                pred_cls = outputs["pred_logits"][b][pred_idx].argmax(-1)
                gt_cls   = targets[b]["labels"][gt_idx]
                correct    += (pred_cls == gt_cls).sum().item()
                total_objs += len(gt_idx)

    m = len(data_loader)
    stats = {k: running[k] / m for k in running}
    stats["acc"] = correct / max(total_objs, 1)
    return stats


# ---------------------------------------------------------------------------
# Plot training history
# ---------------------------------------------------------------------------

def plot_history(history: dict, output_dir: str) -> None:
    """Save a 2×3 grid of loss and accuracy curves."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_total"]) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    pairs = [
        ("Total loss",    "total"),
        ("CE loss",       "ce"),
        ("L1 loss",       "l1"),
        ("GIoU loss",     "giou"),
        ("Accuracy",      "acc"),
    ]
    for ax, (title, key) in zip(axes.flat, pairs):
        ax.plot(epochs, history[f"train_{key}"], label="train")
        ax.plot(epochs, history[f"val_{key}"],   label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    axes.flat[-1].axis("off")   # 6th cell unused
    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Training curves saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train DETR")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file (e.g. configs/train.yaml)")
    # Allow per-run overrides of any config key
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--dataset_path",            default=None)
    parser.add_argument("--output_dir",              default=None)
    parser.add_argument("--checkpoint_dir",          default=None)
    parser.add_argument("--resume",                  default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    for key in ("epochs", "batch_size", "dataset_path", "output_dir", "checkpoint_dir"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    # ── DDP initialisation ───────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)

    # ── Device ───────────────────────────────────────────────────────────────
    device = get_device(local_rank)
    if is_main_process():
        print(f"[INFO] Device: {device}")

    # ── Reproducibility ──────────────────────────────────────────────────────
    set_seed(cfg.get("seed", 42))

    # ── Validate dataset paths ───────────────────────────────────────────────
    validate_dataset(cfg["dataset_path"], cfg.get("split", "val"))

    # ── Dataloaders ──────────────────────────────────────────────────────────
    from src.dataset import build_dataloaders
    train_loader, val_loader, num_classes = build_dataloaders(
        dataset_path=cfg["dataset_path"],
        split=cfg.get("split", "val"),
        batch_size=cfg["batch_size"],
        train_ratio=cfg.get("train_ratio", 0.8),
        seed=cfg.get("seed", 42),
        num_workers=cfg.get("num_workers", 0),
        img_size=cfg.get("img_size", 480),
    )

    # Replace train DataLoader with DDP-aware sampler when running distributed
    if is_distributed():
        train_dataset = train_loader.dataset
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader  = DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=cfg.get("num_workers", 0),
        )

    # ── Model ────────────────────────────────────────────────────────────────
    model = DETR(
        d_model      = cfg.get("d_model", 256),
        num_heads    = cfg.get("num_heads", 8),
        ffn_dim      = cfg.get("ffn_dim", 2048),
        dropout      = cfg.get("dropout", 0.1),
        num_encoders = cfg.get("num_encoders", 6),
        num_decoders = cfg.get("num_decoders", 6),
        num_queries  = cfg.get("num_queries", 100),
        num_classes  = num_classes,
    ).to(device)

    # torch.compile speeds up training on PyTorch >= 2.0 + CUDA/MPS.
    # suppress_errors=True tells dynamo to silently fall back to eager mode if
    # the backend (inductor/triton) fails during the first forward pass — so
    # training always continues regardless of the environment.
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)
        if is_main_process():
            print("[INFO] torch.compile() enabled (falls back to eager if backend unsupported)")
    except Exception as e:
        if is_main_process():
            print(f"[INFO] torch.compile() skipped ({e}) - using eager mode")

    if local_rank >= 0:
        model = DDP(model, device_ids=[local_rank])

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[INFO] Model parameters: {total_params / 1e6:.1f}M")

    # Optional: resume from checkpoint
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location=device, weights_only=True)
        (model.module if is_distributed() else model).load_state_dict(state)
        if is_main_process():
            print(f"[INFO] Resumed from {args.resume}")

    # ── Matcher + Criterion ──────────────────────────────────────────────────
    matcher   = HungarianMatcher(
        cost_class=cfg.get("lambda_class", 1.0),
        cost_bbox =cfg.get("lambda_l1",    5.0),
        cost_giou =cfg.get("lambda_giou",  2.0),
    )
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher    =matcher,
        eos_coef   =cfg.get("no_object_weight", 0.1),
        lambda_l1  =cfg.get("lambda_l1",  5.0),
        lambda_giou=cfg.get("lambda_giou", 2.0),
    ).to(device)

    # ── Optimizer (differential LRs: backbone 10× lower than transformer) ────
    backbone_params     = [p for n, p in model.named_parameters()
                           if "backbone" in n and p.requires_grad]
    transformer_params  = [p for n, p in model.named_parameters()
                           if "backbone" not in n and p.requires_grad]
    param_dicts = [
        {"params": transformer_params},
        {"params": backbone_params, "lr": cfg.get("lr_backbone", 1e-5)},
    ]
    optimizer = torch.optim.AdamW(
        param_dicts,
        lr=cfg.get("lr_transformer", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    # ── Schedulers ───────────────────────────────────────────────────────────
    # Epoch 0: linear warmup over all batches in first epoch
    warmup_iters   = len(train_loader)
    warmup_sched   = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup_iters)
    )
    # Epoch 1+: drop LR by 10× every lr_drop epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.get("lr_drop", 100), gamma=0.1
    )

    # ── History + checkpointing setup ────────────────────────────────────────
    history = {
        "train_total": [], "train_ce": [], "train_l1": [], "train_giou": [], "train_acc": [],
        "val_total":   [], "val_ce":   [], "val_l1":   [], "val_giou":   [], "val_acc":   [],
    }
    best_val_loss    = float("inf")
    checkpoint_dir   = cfg.get("checkpoint_dir", "checkpoints")
    checkpoint_name  = cfg.get("checkpoint_name", "detr_best.pth")
    output_dir       = cfg.get("output_dir", "outputs")

    if is_main_process():
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────────
    epochs = cfg.get("epochs", 300)

    for epoch in range(epochs):
        if is_distributed():
            train_loader.sampler.set_epoch(epoch)   # for reproducible shuffling

        train_stats = train_one_epoch(
            model, criterion, matcher, optimizer,
            train_loader, device, epoch, warmup_sched, cfg,
        )
        val_stats = evaluate(model, criterion, matcher, val_loader, device)

        # Append to history (stats keys: total, loss_ce, loss_l1, loss_giou, acc)
        key_map = {"total": "total", "ce": "loss_ce", "l1": "loss_l1",
                   "giou": "loss_giou", "acc": "acc"}
        for hist_key, stats_key in key_map.items():
            history[f"train_{hist_key}"].append(train_stats[stats_key])
            history[f"val_{hist_key}"].append(val_stats[stats_key])

        if epoch > 0:
            lr_scheduler.step()

        if is_main_process():
            print(
                f"Epoch {epoch + 1:03d}/{epochs} | "
                f"train {train_stats['total']:.3f} (acc {train_stats['acc']:.2%}) | "
                f"val {val_stats['total']:.3f} (acc {val_stats['acc']:.2%})"
            )
            if val_stats["total"] < best_val_loss:
                best_val_loss = val_stats["total"]
                state = (model.module if is_distributed() else model).state_dict()
                torch.save(state, os.path.join(checkpoint_dir, checkpoint_name))

    # ── Plot + cleanup ───────────────────────────────────────────────────────
    if is_main_process():
        plot_history(history, output_dir)

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
