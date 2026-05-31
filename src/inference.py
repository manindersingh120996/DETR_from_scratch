"""
DETR inference script.

Loads a trained checkpoint, runs inference on one image from the dataset,
and saves a visualisation with ground-truth (green) and predicted (red) boxes.

Usage:
    python -m src.inference \\
        --config configs/train.yaml \\
        --checkpoint checkpoints/detr_best.pth \\
        --idx 0

Options:
    --split       train | val  (which split to sample from, default: val)
    --threshold   confidence threshold for predictions (default: 0.3)
    --save        path to save the figure (default: outputs/inference_result.png)
    --no-show     skip interactive display (useful for headless servers)
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")   # headless-safe; must come before pyplot import
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.model import DETR
from src.dataset import CocoDataset, collate_fn, _resolve_paths

# ImageNet normalisation constants (must match dataset pre-processing)
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unnormalise(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation → uint8 HWC numpy array."""
    img = tensor.cpu().clone()
    img = img * _STD[:, None, None] + _MEAN[:, None, None]
    img = img.permute(1, 2, 0).numpy()
    return (img.clip(0, 1) * 255).astype(np.uint8)


def _yolo_label(ax, x1: float, y1: float, text: str, color: str) -> None:
    """Filled YOLO-style text label pinned to a box corner."""
    ax.text(
        x1, y1, text,
        color="white", fontsize=7, fontweight="bold", va="bottom",
        bbox=dict(facecolor=color, edgecolor=color, alpha=0.85,
                  pad=1.5, boxstyle="square"),
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model: torch.nn.Module, dataset: CocoDataset,
                  idx: int, device: torch.device,
                  threshold: float = 0.3):
    """Run model on one sample and return decoded detections.

    Returns:
        img_rgb  : (H, W, 3) uint8 numpy array
        scores   : (K,) float tensor, sorted descending
        labels   : (K,) long tensor
        boxes    : (K, 4) float tensor, normalised [cx, cy, w, h]
        target   : ground-truth dict
    """
    model.eval()
    image, target = dataset[idx]
    images, masks, _ = collate_fn([(image, target)])
    outputs = model(images.to(device), masks.to(device))

    logits = outputs["pred_logits"][0]   # (100, C+1)
    boxes  = outputs["pred_boxes"][0]    # (100, 4)

    probs          = logits.softmax(-1)
    scores, labels = probs[:, :-1].max(-1)   # exclude background

    keep   = scores > threshold
    scores = scores[keep].cpu()
    labels = labels[keep].cpu()
    boxes  = boxes[keep].cpu()

    order = scores.argsort(descending=True)
    return unnormalise(image), scores[order], labels[order], boxes[order], target


def draw_and_save(img_rgb: np.ndarray, scores: torch.Tensor,
                  labels: torch.Tensor, boxes: torch.Tensor,
                  target: dict, dataset: CocoDataset,
                  save_path: str, show: bool = True) -> None:
    """Draw GT (green solid) and predictions (red dashed), save figure."""
    H, W = img_rgb.shape[:2]

    # Build index → human-readable name map
    idx_to_cat_id = {v: k for k, v in dataset.cat_id_to_idx.items()}
    cat_id_to_name: dict = {}
    try:
        ann_file = getattr(dataset, "annotation_file",
                           getattr(dataset, "_ann_file", None))
        if ann_file and os.path.isfile(ann_file):
            with open(ann_file) as f:
                coco_data = json.load(f)
            cat_id_to_name = {c["id"]: c["name"] for c in coco_data["categories"]}
    except Exception:
        pass

    def label_name(dense_idx: int) -> str:
        cat_id = idx_to_cat_id.get(int(dense_idx), dense_idx)
        return cat_id_to_name.get(cat_id, str(cat_id))

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img_rgb)

    # Ground-truth boxes (green solid)
    for box, lbl in zip(target["boxes"], target["labels"]):
        cx, cy, w, h = box.tolist()
        x1 = (cx - 0.5 * w) * W
        y1 = (cy - 0.5 * h) * H
        bw, bh = w * W, h * H
        ax.add_patch(patches.Rectangle((x1, y1), bw, bh,
                                        linewidth=2, edgecolor="lime",
                                        facecolor="none"))
        _yolo_label(ax, x1, max(y1 - 2, 0), f"GT: {label_name(lbl.item())}", "green")

    # Predicted boxes (red dashed)
    for score, lbl, box in zip(scores, labels, boxes):
        cx, cy, w, h = box.tolist()
        x1 = (cx - 0.5 * w) * W
        y1 = (cy - 0.5 * h) * H
        bw, bh = w * W, h * H
        ax.add_patch(patches.Rectangle((x1, y1), bw, bh,
                                        linewidth=2, edgecolor="red",
                                        facecolor="none", linestyle="--"))
        _yolo_label(ax, x1, min(y1 + bh + 2, H - 1),
                    f"{label_name(lbl.item())} {score:.2f}", "red")

    ax.set_title(f"Green = GT  |  Red dashed = Predictions (threshold {scores.min():.2f}+)"
                 if len(scores) > 0 else "No predictions above threshold", fontsize=9)
    ax.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"[INFO] Inference result saved -> {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DETR inference")
    parser.add_argument("--config",     required=True,
                        help="Path to train YAML config")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--idx",        type=int, default=0,
                        help="Dataset index to run inference on (default: 0)")
    parser.add_argument("--split",      default="val",
                        choices=["train", "val"],
                        help="Dataset split to sample from (default: val)")
    parser.add_argument("--threshold",  type=float, default=0.3,
                        help="Confidence threshold (default: 0.3)")
    parser.add_argument("--save",       default=None,
                        help="Save path for figure (default: outputs/inference_result.png)")
    parser.add_argument("--no-show",    action="store_true",
                        help="Suppress interactive display")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    print(f"[INFO] Device: {device}")

    # Dataset (full, without train/val split — we just need the image + annotations)
    ann_file, img_dir = _resolve_paths(cfg["dataset_path"], cfg.get("split", "val"))
    dataset = CocoDataset(ann_file, img_dir, img_size=cfg.get("img_size", 480))
    # Store annotation file path so draw_and_save can resolve category names
    dataset.annotation_file = ann_file

    # Model
    model = DETR(
        d_model      = cfg.get("d_model", 256),
        num_heads    = cfg.get("num_heads", 8),
        ffn_dim      = cfg.get("ffn_dim", 2048),
        dropout      = cfg.get("dropout", 0.1),
        num_encoders = cfg.get("num_encoders", 6),
        num_decoders = cfg.get("num_decoders", 6),
        num_queries  = cfg.get("num_queries", 100),
        num_classes  = dataset.num_classes,
    ).to(device)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    # Inference
    idx = args.idx % len(dataset)   # wrap around if idx is too large
    img_rgb, scores, labels, boxes, target = run_inference(
        model, dataset, idx, device, threshold=args.threshold
    )
    print(f"[INFO] Sample {idx}: {len(scores)} prediction(s) above threshold {args.threshold}")

    save_path = args.save or os.path.join(
        cfg.get("output_dir", "outputs"), "inference_result.png"
    )
    draw_and_save(img_rgb, scores, labels, boxes, target, dataset,
                  save_path=save_path, show=not args.no_show)


if __name__ == "__main__":
    main()
