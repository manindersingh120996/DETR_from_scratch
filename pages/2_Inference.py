"""Inference page — run model on any dataset image + architecture diagram."""
import json
import os

import streamlit as st
import torch
import yaml

from src.dataset import CocoDataset, _resolve_paths
from src.inference import draw_and_save, get_device, run_inference
from src.model import DETR

st.set_page_config(page_title="Inference", page_icon="🔍", layout="wide")

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
</style>
""", unsafe_allow_html=True)

cfg = yaml.safe_load(open("configs/train.yaml"))

st.title("Inference")
st.divider()

# ── A: Checkpoint selection ───────────────────────────────────────────────────
st.subheader("Model checkpoint")

default_ckpt = os.path.join(cfg.get("checkpoint_dir", "checkpoints"),
                             cfg.get("checkpoint_name", "detr_best.pth"))

ckpt_path = st.text_input(
    "Path to checkpoint (.pth)",
    value=default_ckpt,
    help="Default: the best checkpoint saved during training.",
)

if not os.path.isfile(ckpt_path):
    st.error(f"Checkpoint not found: `{ckpt_path}`  — train the model first, or provide a valid path.")
    ckpt_ok = False
else:
    size_mb = os.path.getsize(ckpt_path) / 1e6
    st.success(f"Checkpoint ready — `{ckpt_path}` ({size_mb:.1f} MB)")
    ckpt_ok = True

st.divider()

# ── B: Dataset + image selection ──────────────────────────────────────────────
st.subheader("Image selection")

try:
    ann_file, img_dir = _resolve_paths(cfg["dataset_path"], cfg.get("split", "val"))
    dataset = CocoDataset(ann_file, img_dir, img_size=cfg.get("img_size", 480))
    dataset.annotation_file = ann_file
    dataset_ok = True
    st.caption(f"Dataset: `{os.path.abspath(cfg['dataset_path'])}` — {len(dataset)} images")
except FileNotFoundError:
    st.error("Dataset not found. Check `dataset_path` in `configs/train.yaml`.")
    dataset_ok = False

if dataset_ok:
    col1, col2 = st.columns([3, 1])
    idx       = col1.number_input("Image index", min_value=0,
                                   max_value=len(dataset) - 1, value=0, step=1)
    threshold = col2.slider("Confidence threshold", 0.05, 0.95, 0.30, 0.05,
                             help="Only predictions above this score are shown.")

    if st.button("Run Inference", type="primary", disabled=not ckpt_ok):
        with st.spinner("Loading model and running inference..."):
            device = get_device()

            @st.cache_resource(show_spinner=False)
            def _load_model(ckpt: str, num_classes: int):
                m = DETR(
                    d_model      = cfg.get("d_model", 256),
                    num_heads    = cfg.get("num_heads", 8),
                    ffn_dim      = cfg.get("ffn_dim", 2048),
                    dropout      = cfg.get("dropout", 0.1),
                    num_encoders = cfg.get("num_encoders", 6),
                    num_decoders = cfg.get("num_decoders", 6),
                    num_queries  = cfg.get("num_queries", 100),
                    num_classes  = num_classes,
                ).to(device)
                sd = torch.load(ckpt, map_location=device, weights_only=True)
                # Strip _orig_mod. prefix left by torch.compile when saving
                sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
                m.load_state_dict(sd)
                m.eval()
                return m

            model = _load_model(ckpt_path, dataset.num_classes)
            img_rgb, scores, labels, boxes, target = run_inference(
                model, dataset, int(idx), device, threshold
            )

        save_path = os.path.join(cfg.get("output_dir", "outputs"), "inference_result.png")
        draw_and_save(img_rgb, scores, labels, boxes, target, dataset,
                      save_path=save_path, show=False)

        # Display result
        st.image(save_path,
                 caption=f"Image {idx} | {len(scores)} detection(s) above threshold {threshold:.2f}",
                 use_container_width=True)

        # Detection table
        if len(scores) > 0:
            idx_to_cat = {v: k for k, v in dataset.cat_id_to_idx.items()}
            cat_names: dict = {}
            try:
                with open(ann_file) as f:
                    coco_data = json.load(f)
                cat_names = {c["id"]: c["name"] for c in coco_data["categories"]}
            except Exception:
                pass

            def _name(dense_idx: int) -> str:
                cid = idx_to_cat.get(int(dense_idx), dense_idx)
                return cat_names.get(cid, str(cid))

            st.dataframe(
                {
                    "Rank":       list(range(1, len(scores) + 1)),
                    "Label":      [_name(l.item()) for l in labels],
                    "Confidence": [f"{s:.3f}" for s in scores],
                },
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No detections above threshold. Try lowering the confidence slider.")

st.divider()

# ── C: Architecture diagram ───────────────────────────────────────────────────
st.subheader("Model architecture")

arch_mode = st.radio("View for", ["Inference", "Training"], horizontal=True,
                      help="Training mode shows the Hungarian Matcher and loss nodes.")


def _detr_dot(mode: str) -> str:
    """Return a Graphviz DOT string for the DETR architecture."""
    training = mode == "Training"
    loss_color   = "#5C2A2A"   # dark red
    filter_color = "#1E3A2A"   # dark green

    extra_nodes = ""
    extra_edges = ""

    if training:
        extra_nodes = f"""
    HM  [label="Hungarian Matcher\\n(bipartite matching\\nno gradients)", fillcolor="{loss_color}"]
    Loss [label="Set Loss\\nCE (all 100 slots)\\nL1 + GIoU (matched only)", fillcolor="{loss_color}"]
"""
        extra_edges = """
    ClsHead -> HM
    BboxHead -> HM
    HM -> Loss
"""
    else:
        extra_nodes = f"""
    Filter [label="Threshold Filter\\nscore > threshold\\nno NMS needed", fillcolor="{filter_color}"]
    Out    [label="Detections\\n(class, score, box)", fillcolor="#1A2A1A"]
"""
        extra_edges = """
    ClsHead -> Filter
    BboxHead -> Filter
    Filter -> Out
"""

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

    {extra_nodes}

    Input -> Backbone -> Proj -> Encoder
    PE -> Encoder
    Encoder -> Decoder
    Queries -> Decoder
    Decoder -> ClsHead
    Decoder -> BboxHead
    {extra_edges}
}}
"""

st.graphviz_chart(_detr_dot(arch_mode), use_container_width=True)

with st.expander("Architecture notes"):
    st.markdown("""
- **Backbone** — ResNet-50 with the last avgpool and FC removed. Output stride is 32, so a 480×640 image becomes ~15×20 spatial positions.
- **Projection** — 1×1 conv maps 2048 channels to 256 (d_model). This is the transformer's working dimension.
- **Positional Encoding** — 2D sinusoidal: the first 128 dims encode the row (Y) position, the last 128 encode the column (X) position. Added to Q and K, not V.
- **Encoder** — 6 self-attention layers. Every spatial position can attend to every other position globally.
- **Object Queries** — 100 learned vectors (`nn.Parameter`). Each query is responsible for detecting at most one object.
- **Decoder** — 6 layers. Each query first self-attends to other queries, then cross-attends to encoder memory. Positional encoding is added to Q and K in cross-attention.
- **Auxiliary outputs** — all 6 decoder layers produce predictions; losses are summed across all layers during training.
- **No NMS** — DETR enforces unique assignments via Hungarian matching, so non-maximum suppression is not needed.
""")
