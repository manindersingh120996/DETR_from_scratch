"""Training page — start/stop training, live loss charts, TensorBoard embed."""
import json
import os
import subprocess
import sys

import streamlit as st
import torch
import yaml
from streamlit_autorefresh import st_autorefresh

from src.dataset import _resolve_paths

st.set_page_config(page_title="Training", page_icon="🏋️", layout="wide")

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"] {
    background: #16162A; border: 1px solid #2A2A3E;
    border-radius: 10px; padding: 14px 18px;
}
[data-testid="stButton"] > button[kind="primary"] {
    background: #D4A76A; color: #0D0D14; border: none;
    border-radius: 8px; font-weight: 700;
}
[data-testid="stButton"] > button[kind="primary"]:hover { background: #C49050; }
[data-testid="stSidebar"] { border-right: 1px solid #2A2A3E; }
[data-testid="stProgressBar"] > div > div { background: #D4A76A !important; }
</style>
""", unsafe_allow_html=True)

# ── Load config ───────────────────────────────────────────────────────────────
cfg = yaml.safe_load(open("configs/train.yaml"))

st.title("Training")
st.divider()

# ── A: Device / Mode Selection ────────────────────────────────────────────────
st.subheader("Device & mode")

gpu_count = torch.cuda.device_count()
has_mps   = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

if gpu_count > 1:
    mode = st.radio(
        "Training mode",
        ["Single GPU", f"Multi-GPU DDP  ({gpu_count} GPUs)"],
        help="DDP (DistributedDataParallel) distributes each batch across all GPUs in parallel.",
    )
elif gpu_count == 1:
    st.info("1 GPU detected — Multi-GPU DDP requires 2 or more GPUs.")
    mode = "Single GPU"
elif has_mps:
    st.info("Apple Silicon (MPS) detected — training will run on MPS.")
    mode = "MPS"
else:
    st.warning("No GPU detected — training will run on CPU (significantly slower).")
    mode = "CPU"

st.divider()

# ── B: Run type + hyperparameters ─────────────────────────────────────────────
st.subheader("Run configuration")

run_type = st.radio(
    "Run type",
    ["Test run  (quick sanity check)", "Full training  (use config file)"],
    help="Test run lets you adjust epochs and batch size here. "
         "Full training uses all values from configs/train.yaml.",
)

if run_type.startswith("Test"):
    col1, col2 = st.columns(2)
    epochs     = col1.slider("Epochs", 1, 500, 5)
    batch_size = col2.selectbox("Batch size", [1, 2, 4], index=1)
    st.caption(
        "All other parameters (LR, loss weights, model size, etc.) use the values "
        "in `configs/train.yaml`. Edit that file for full control."
    )
else:
    epochs     = cfg["epochs"]
    batch_size = cfg["batch_size"]
    st.caption(
        f"Using `configs/train.yaml`: **{epochs} epochs**, batch size **{batch_size}**. "
        "Edit that file to change any hyperparameter."
    )

st.divider()

# ── C: Dataset validation ─────────────────────────────────────────────────────
st.subheader("Dataset")

dataset_path = cfg["dataset_path"]
abs_path     = os.path.abspath(dataset_path)
st.write(f"**Configured path:** `{abs_path}`")

try:
    _resolve_paths(dataset_path, cfg.get("split", "val"))
    st.success("Dataset found and ready.")
    dataset_ok = True
except FileNotFoundError:
    st.error("Dataset not found at the configured path.")
    with st.expander("Show expected folder structure"):
        st.code("""
# Option 1 — Standard COCO layout
dataset_root/
├── annotations/
│   └── instances_val2017.json
└── val2017/
    └── *.jpg

# Option 2 — Mini / flat layout  (default: coco_mini/)
coco_mini/
├── annotations.json
└── images/
    └── *.jpg
        """)
    st.warning(
        "Update `dataset_path` in `configs/train.yaml` and re-run. "
        "See **README.md → Dataset Setup** for full download instructions."
    )
    dataset_ok = False

st.divider()

# ── D: Start / Stop ───────────────────────────────────────────────────────────
st.subheader("Start training")

def _build_cmd(mode: str, epochs: int, batch_size: int, gpu_count: int) -> list:
    """Build the terminal command that launches src/train.py."""
    common_args = [
        "--config", "configs/train.yaml",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
    ]
    if "Multi-GPU" in mode:
        return ["torchrun", f"--nproc_per_node={gpu_count}", "-m", "src.train"] + common_args
    return [sys.executable, "-m", "src.train"] + common_args

is_training = st.session_state.get("training", False)

log_path = os.path.join(cfg.get("output_dir", "outputs"), "training.log")

col_start, col_stop = st.columns(2)
with col_start:
    if st.button("Start Training", type="primary", disabled=is_training or not dataset_ok):
        os.makedirs(cfg.get("output_dir", "outputs"), exist_ok=True)
        log_file = open(log_path, "w", buffering=1)
        cmd = _build_cmd(mode, epochs, batch_size, gpu_count)
        st.session_state.train_process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.getcwd(),
        )
        st.session_state.train_log_file = log_file
        st.session_state.training = True
        st.rerun()

with col_stop:
    if st.button("Stop Training", disabled=not is_training):
        proc = st.session_state.get("train_process")
        if proc:
            proc.terminate()
        lf = st.session_state.get("train_log_file")
        if lf:
            lf.close()
        st.session_state.training = False
        st.rerun()

# Show the exact command that will be / was run
preview_cmd = _build_cmd(mode, epochs, batch_size, gpu_count)
st.caption("Command: `" + " ".join(preview_cmd) + "`")

st.divider()

# ── E: Live loss charts ───────────────────────────────────────────────────────
st.subheader("Live training progress")

# Auto-refresh every 3 s while training is active
if st.session_state.get("training"):
    st_autorefresh(interval=3000, key="train_refresh")

    # Check if process finished
    proc = st.session_state.get("train_process")
    if proc and proc.poll() is not None:
        lf = st.session_state.get("train_log_file")
        if lf:
            lf.close()
        st.session_state.training = False

progress_path = os.path.join(cfg.get("output_dir", "outputs"), "training_progress.json")

if os.path.exists(progress_path):
    try:
        with open(progress_path) as f:
            p = json.load(f)
        h = p["history"]
        done = p.get("status") == "completed"

        # Progress bar + epoch counter
        prog_val = p["epoch"] / max(p["total_epochs"], 1)
        label    = (f'Epoch {p["epoch"]} / {p["total_epochs"]}  '
                    + ("(completed)" if done else "(running...)"))
        st.progress(prog_val, text=label)

        # 4 metric cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Train loss", f'{h["train_total"][-1]:.3f}')
        m2.metric("Val loss",   f'{h["val_total"][-1]:.3f}',
                  delta=f'{h["val_total"][-1] - h["val_total"][-2]:.3f}'
                        if len(h["val_total"]) > 1 else None,
                  delta_color="inverse")
        m3.metric("Train acc",  f'{h["train_acc"][-1]:.1%}')
        m4.metric("Val acc",    f'{h["val_acc"][-1]:.1%}',
                  delta=f'{(h["val_acc"][-1] - h["val_acc"][-2]):.1%}'
                        if len(h["val_acc"]) > 1 else None)

        # Charts in 2×2 grid
        c1, c2 = st.columns(2)
        c1.line_chart(
            {"Train": h["train_total"], "Val": h["val_total"]},
            use_container_width=True,
            color=["#D4A76A", "#7C6CD4"],
        )
        c1.caption("Total loss")
        c2.line_chart(
            {"Train": h["train_acc"], "Val": h["val_acc"]},
            use_container_width=True,
            color=["#D4A76A", "#7C6CD4"],
        )
        c2.caption("Accuracy (matched predictions)")

        c3, c4 = st.columns(2)
        c3.line_chart(
            {"CE loss": h["train_ce"], "L1 loss": h["train_l1"]},
            use_container_width=True,
        )
        c3.caption("Component losses (train)")
        c4.line_chart(
            {"GIoU loss": h["train_giou"]},
            use_container_width=True,
            color=["#4A9B6F"],
        )
        c4.caption("GIoU loss (train)")

    except (json.JSONDecodeError, KeyError):
        st.info("Waiting for first epoch to complete...")
elif st.session_state.get("training"):
    st.info("Training started — waiting for first epoch to complete. Check the log below.")
else:
    st.info("Start training above to see live progress here.")

# ── Live log viewer ───────────────────────────────────────────────────────────
if os.path.exists(log_path):
    with st.expander("Training log (last 40 lines)", expanded=st.session_state.get("training", False)):
        try:
            with open(log_path, "r", errors="replace") as f:
                lines = f.readlines()
            last_lines = lines[-40:] if len(lines) > 40 else lines
            st.code("".join(last_lines), language=None)
        except Exception as e:
            st.warning(f"Could not read log: {e}")

st.divider()

# ── F: TensorBoard panel ──────────────────────────────────────────────────────
with st.expander("TensorBoard — full interactive view", expanded=False):
    st.caption("Opens TensorBoard running at http://localhost:6006")
    st.components.v1.iframe("http://localhost:6006", height=720, scrolling=True)

st.divider()

# ── G: Architecture diagram ───────────────────────────────────────────────────
st.subheader("Model architecture (training mode)")


def _detr_dot_train() -> str:
    loss_color = "#5C2A2A"
    return f"""
digraph DETR {{
    rankdir=TB;
    bgcolor="#0D0D14";
    node [style=filled, fontcolor="#E8E3DC", fontname="Helvetica", shape=box,
          margin="0.3,0.15", penwidth=0];
    edge [color="#4A4A6A", arrowsize=0.8];

    Input    [label="Input Images\\n(B, 3, H, W)", fillcolor="#1A1A2E"]
    Backbone [label="ResNet-50 Backbone\\n(pretrained, stride 32)\\n(B, 2048, H/32, W/32)", fillcolor="#5C3D1E"]
    Proj     [label="1x1 Conv Projection\\n(B, 256, H/32, W/32)", fillcolor="#5C3D1E"]
    PE       [label="2D Sinusoidal PE\\nY-dims + X-dims concatenated\\nadded to Q and K only", fillcolor="#1E2A4A"]
    Encoder  [label="Transformer Encoder\\n6 layers | Self-Attention\\n(HW/1024, B, 256)", fillcolor="#3A2060"]
    Queries  [label="100 Object Queries\\n(learned nn.Parameter)\\n(100, B, 256)", fillcolor="#1A3030"]
    Decoder  [label="Transformer Decoder\\n6 layers | Self + Cross Attention\\n(100, B, 256)", fillcolor="#3A2060"]
    ClsHead  [label="Class Head\\nLinear -> (B, 100, 81)\\nraw logits", fillcolor="#1A3A1A"]
    BboxHead [label="BBox Head\\n3-layer MLP -> (B, 100, 4)\\nsigmoid [cx,cy,w,h]", fillcolor="#1A3A1A"]
    HM       [label="Hungarian Matcher\\n(bipartite matching\\nno gradients)", fillcolor="{loss_color}"]
    Loss     [label="Set Loss\\nCE (all 100 slots)\\nL1 + GIoU (matched only)", fillcolor="{loss_color}"]

    Input -> Backbone -> Proj -> Encoder
    PE -> Encoder
    Encoder -> Decoder
    Queries -> Decoder
    Decoder -> ClsHead
    Decoder -> BboxHead
    ClsHead -> HM
    BboxHead -> HM
    HM -> Loss
}}
"""


st.graphviz_chart(_detr_dot_train(), use_container_width=True)

with st.expander("Architecture notes"):
    st.markdown("""
- **Backbone** — ResNet-50 with the last avgpool and FC removed. Output stride is 32, so a 480×640 image becomes ~15×20 spatial positions.
- **Projection** — 1×1 conv maps 2048 channels to 256 (d_model). This is the transformer's working dimension.
- **Positional Encoding** — 2D sinusoidal: the first 128 dims encode the row (Y) position, the last 128 encode the column (X) position. Added to Q and K, not V.
- **Encoder** — 6 self-attention layers. Every spatial position can attend to every other position globally.
- **Object Queries** — 100 learned vectors (`nn.Parameter`). Each query is responsible for detecting at most one object.
- **Decoder** — 6 layers. Each query first self-attends to other queries, then cross-attends to encoder memory.
- **Hungarian Matcher** — finds the minimum-cost bipartite matching between the 100 predictions and GT boxes. No gradients flow through the matcher itself.
- **Set Loss** — CE on all 100 query slots (background weight = 0.1) + L1 and GIoU only on the matched slots.
- **Auxiliary losses** — all 6 decoder layers produce predictions; losses are summed across all layers during training.
""")
