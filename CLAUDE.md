# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Virtual environment: `c:\Projects\testenv`

```
c:\Projects\testenv\Scripts\activate
jupyter notebook
```

All dependencies (PyTorch, torchvision, OpenCV, scipy, matplotlib) are installed in `testenv`.

## Notebooks

- `detr_baseline_fixed.ipynb` — the canonical, working implementation. Run this end-to-end.
- `rough_experimentation.ipynb` — early scratch work; contains the original buggy version of the model for reference. Do not treat it as authoritative.

## Data

COCO val2017 (5000 images) is stored locally. A 100-image subset is pre-built at `coco_mini/`:

```
coco_mini/
├── annotations.json     # filtered COCO JSON (100 images, 658 annotations, 80 categories)
└── images/              # 100 JPEG files
```

Raw source data lives at:
- `val2017/` — full COCO val images
- `val2017_annotations/annotations/instances_val2017.json` — full annotations

The mini-dataset creation cell in `detr_baseline_fixed.ipynb` only needs to be run once.

## Architecture

`DETR` (`detr_baseline_fixed.ipynb`, Step 4) is a single `nn.Module` with these components in sequence:

1. **Backbone** — ResNet-50 (pretrained), final avg-pool and FC removed → `(B, 2048, H', W')`
2. **Input projection** — 1×1 conv mapping 2048 → `d_model` (256)
3. **2D sinusoidal positional encoding** — splits `d_model` in half: first 128 dims encode Y position, last 128 encode X. Added to projected features before the transformer.
4. **`nn.Transformer`** — `batch_first=False` (expects `(seq, batch, dim)`). Encoder receives the flattened spatial features; decoder receives the object queries as `tgt`.
5. **Object queries** — 100 learned `nn.Parameter` vectors, used directly as decoder `tgt` input (not zeros).
6. **Class head** — linear → `num_classes + 1` (last index is background/no-object).
7. **Bbox head** — 3-layer MLP → 4, with `Sigmoid` output (normalised `cx cy w h` in `[0,1]`).

Forward signature: `model(images, mask)` where `mask` is a `bool` tensor `(B, H, W)`, `True` at padded positions.

## Key Implementation Details

**Box format**: all boxes are stored and predicted as normalised `[cx, cy, w, h]` in `[0, 1]`. Ground truth is normalised against the *original* image size before resizing. Conversion to `xyxy` for GIoU uses `box_cxcywh_to_xyxy`.

**Padding mask**: images in a batch are padded to the same spatial size. The boolean mask is downsampled with `F.interpolate(..., mode='nearest')` to match the backbone feature map size, then flattened to `(B, H'*W')` and passed as both `src_key_padding_mask` (encoder) and `memory_key_padding_mask` (decoder).

**Category remapping**: COCO category IDs are sparse (1–90 with gaps). The dataset remaps them to dense 0-based indices via `cat_id_to_idx`. The class head output size is `num_classes + 1 = 81`.

**Differential learning rates**: backbone parameters use `lr=1e-5`; all other parameters use `lr=1e-4`. Both use AdamW with `weight_decay=1e-4`.

**Training schedule**: linear LR warmup for the first epoch (`warmup_iters = len(train_loader)`), then `StepLR(step_size=100, gamma=0.1)`. Gradient clipping at `max_norm=0.1` is required for stability.

**Loss weights**: `loss_ce + 5 * loss_l1 + 2 * loss_giou`. Background class weight is `0.1` to counter the ~95:5 background/foreground imbalance across 100 queries.

**Hungarian matching**: uses raw softmax probability (not log) in the cost matrix for numerical stability. The training loss uses full cross-entropy (log) for gradient signal.

## Saved Artifacts

- `detr_best.pth` — best validation checkpoint (lowest val loss over 100 epochs)
- `training_curves.png` — loss and accuracy plots
- `inference_result.png` — visualisation from the last inference run

Load the checkpoint:
```python
model.load_state_dict(torch.load('detr_best.pth', map_location=device, weights_only=True))
```

## Bugs Fixed (baseline vs rough_experimentation)

The baseline notebook documents and corrects five bugs present in the rough version:
1. `tgt = query_embed` (not `torch.zeros_like`) — zero tgt gives the decoder nothing to differentiate queries
2. Removed `hs = hs + query_embed` — double-adding the queries creates a gradient-fighting bias
3. `memory_key_padding_mask=src_mask` — decoder cross-attention must also ignore padded encoder positions
4. Gradient clipping (`max_norm=0.1`) — DETR is sensitive to gradient explosions without it
5. `warmup_iters = len(train_loader)` (≈39), not 1000 — original kept LR near zero for all 10 epochs
