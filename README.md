# DETR from Scratch

A clean, from-scratch implementation of [DETR (Detection Transformer)](https://arxiv.org/abs/2005.12872) in PyTorch.

Runs entirely from the terminal — no Jupyter required. Supports CUDA, Apple Silicon (MPS), and CPU out of the box, with optional multi-GPU training via PyTorch DDP.

NOTE : The detailed blog around it's [implementation can be accessed from here](https://medium.com/@manindersingh120996/building-detr-from-scratch-an-end-to-end-walkthrough-of-object-detection-with-transformers-77f18691d329)

---

## Project Structure

```
.
├── src/
│   ├── model.py        ← DETR model (backbone, encoder, decoder, FFN heads)
│   ├── dataset.py      ← COCO dataset loader + collate + dataloaders
│   ├── train.py        ← training script (CLI entry point)
│   └── inference.py    ← inference + visualisation script
├── configs/
│   └── train.yaml      ← all hyperparameters and paths
├── scripts/
│   ├── train.sh        ← convenience wrapper for training
│   └── inference.sh    ← convenience wrapper for inference
├── pages/
│   ├── 1_Training.py   ← Streamlit training page (live charts, log viewer, architecture diagram)
│   ├── 2_Inference.py  ← Streamlit inference page (checkpoint loader, result viewer, architecture diagram)
│   └── 3_History.py    ← Streamlit history page (past run metrics and curves)
├── app.py              ← Streamlit home page + TensorBoard auto-launch
├── run.bat             ← Windows one-click launcher
├── .streamlit/
│   └── config.toml     ← UI theme (dark navy + amber accent)
├── checkpoints/        ← saved model weights go here
├── outputs/            ← training curves, inference figures, live progress JSON, training log
├── requirements.txt
└── from_scratch_implementation.ipynb   ← learning notebook (reference only)
```

---

## Installation

```bash
# 1. Clone the repo
git clone <repo-url>
cd "DETR from srcatch"

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Browser UI (Streamlit)

A full browser UI is included — no command-line required for day-to-day use.

### Launch

```bash
# Windows (recommended — avoids PATH issues)
run.bat

# Any platform
python -m streamlit run app.py
```

Opens at **http://localhost:8501**. TensorBoard starts automatically in the background and is embedded inside the Training page.

### Pages

| Page | What it does |
|---|---|
| **Home** | Device summary (CUDA / MPS / CPU), quick-start guide |
| **Training** | Start/stop training, live loss charts (auto-refresh every 3 s), training log viewer, TensorBoard embed, architecture diagram |
| **Inference** | Load any checkpoint, pick an image + confidence threshold, view detections vs ground truth, architecture diagram with training/inference toggle |
| **History** | Metrics and interactive loss curves from the last completed training run |

### Cloud / RunPod / SSH

```bash
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Expose port `8501` in your cloud provider's settings. TensorBoard is embedded in the UI so no second port is needed.

---

## Dataset Setup

### Option A — Mini dataset (already included)

The `coco_mini/` folder contains 100 COCO val images and is ready to use.
No download needed. The default config already points to it.

### Option B — Full COCO val2017

```bash
# Download images (~1 GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Download annotations (~241 MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

Expected layout after extraction:
```
your_dataset_root/
├── val2017/
│   ├── 000000000139.jpg
│   └── ...
└── annotations/
    └── instances_val2017.json
```

Update `configs/train.yaml`:
```yaml
dataset_path: "path/to/your_dataset_root"
split: "val"
```

### Option C — Full COCO train2017

```bash
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
```

Layout:
```
your_dataset_root/
├── train2017/
└── annotations/
    └── instances_train2017.json
```

Update config:
```yaml
dataset_path: "path/to/your_dataset_root"
split: "train"
```

**Common mistakes:**
- Wrong `split` name (must match the folder name: `train`, `val`)
- Missing `annotations/` subfolder — ensure you downloaded the annotation zip, not just images
- Pointing `dataset_path` to `val2017/` directly instead of its parent

---

## Training

### Single GPU / CPU / Apple Silicon

```bash
python -m src.train --config configs/train.yaml
```

Or use the shell script:
```bash
bash scripts/train.sh
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 -m src.train --config configs/train.yaml
```

Replace `4` with the number of GPUs available.

### Per-run config overrides

```bash
python -m src.train --config configs/train.yaml --epochs 50 --batch_size 4
```

### Resume from checkpoint

```bash
python -m src.train --config configs/train.yaml --resume checkpoints/detr_best.pth
```

**Outputs after training:**
- `checkpoints/detr_best.pth` — best checkpoint (lowest val loss)
- `outputs/training_curves.png` — loss and accuracy plots

---

## Inference

```bash
python -m src.inference \
    --config      configs/train.yaml \
    --checkpoint  checkpoints/detr_best.pth \
    --idx         0 \
    --threshold   0.3
```

Or:
```bash
bash scripts/inference.sh
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--idx` | `0` | Dataset index to visualise |
| `--split` | `val` | `train` or `val` split |
| `--threshold` | `0.3` | Confidence cutoff |
| `--save` | `outputs/inference_result.png` | Where to save the figure |
| `--no-show` | off | Skip interactive display (for headless servers) |

**Output:** A figure with green boxes (ground truth) and red dashed boxes (predictions), with class labels and confidence scores.

---

## Configuration

All hyperparameters live in `configs/train.yaml`. Key values:

| Key | Default | Description |
|---|---|---|
| `epochs` | 300 | Training epochs |
| `batch_size` | 2 | Batch size per GPU |
| `lr_transformer` | 1e-4 | Transformer learning rate |
| `lr_backbone` | 1e-5 | Backbone learning rate (10× lower) |
| `lr_drop` | 100 | Epoch where LR drops by 10× |
| `lambda_l1` | 5.0 | L1 box loss weight |
| `lambda_giou` | 2.0 | GIoU loss weight |
| `no_object_weight` | 0.1 | CE weight for background class |
| `num_queries` | 100 | Object queries |

---

## Loading a Checkpoint

```python
import torch
from src.model import DETR

model = DETR(num_classes=80)
model.load_state_dict(
    torch.load("checkpoints/detr_best.pth", map_location="cpu", weights_only=True)
)
model.eval()
```

---

## Architecture Summary

```
Input (B, 3, H, W)
  → ResNet-50 backbone          (B, 2048, H/32, W/32)
  → 1×1 conv projection         (B, 256, H/32, W/32)
  → 2D sinusoidal PE            added to Q and K (not V)
  → Transformer encoder (×6)    self-attention on flattened spatial features
  → Transformer decoder (×6)    100 object queries cross-attend to encoder memory
  → Class head (linear)         → (B, 100, 81)  logits, last index = background
  → Bbox head (3-layer MLP)     → (B, 100, 4)   sigmoid [cx, cy, w, h] in [0,1]
```

Hungarian matching assigns each GT object to the unique lowest-cost query at training time. No NMS needed.

---

## Troubleshooting

**`FileNotFoundError` at startup**
Dataset path is wrong. Check `dataset_path` in `configs/train.yaml` and the folder layout described above.

**Loss oscillates without decreasing**
Make sure gradient clipping is on (`grad_clip_max_norm: 0.1` in config). DETR is sensitive to gradient explosions.

**All predictions show background after training**
Expected early in training — DETR converges slowly. Run for 100+ epochs on a real dataset.

**`NCCL error` during DDP**
Your system may not support NCCL. Try `NCCL_DEBUG=INFO torchrun ...` to diagnose, or switch to `gloo` backend by setting `TORCH_DISTRIBUTED_DEFAULT_BACKEND=gloo`.

**Apple Silicon (MPS) crashes**
Ensure PyTorch ≥ 2.0 is installed. Some ops fall back to CPU automatically.

**`Missing key(s) in state_dict: "backbone.backbone.0.weight"` / `Unexpected key(s): "_orig_mod.*"`**
The checkpoint was saved from a `torch.compile()`-d model, which prefixes all keys with `_orig_mod.`. The inference code strips this automatically. If you load a checkpoint manually, strip the prefix before calling `load_state_dict`:
```python
sd = torch.load("checkpoints/detr_best.pth", map_location="cpu", weights_only=True)
sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
model.load_state_dict(sd)
```
