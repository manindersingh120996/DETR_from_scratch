"""
DETR Dashboard — home page.

Launch:
    streamlit run app.py

This page also starts TensorBoard as a background process so it is
available inside the Training page's embedded iframe.
"""
import os
import subprocess
import sys

import streamlit as st
import torch

st.set_page_config(
    page_title="DETR Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS (injected once; applies to all pages via session) ─────────────
st.markdown("""
<style>
/* Card-style metric boxes */
[data-testid="stMetric"] {
    background: #16162A;
    border: 1px solid #2A2A3E;
    border-radius: 10px;
    padding: 14px 18px;
}
/* Primary button */
[data-testid="stButton"] > button[kind="primary"] {
    background: #D4A76A;
    color: #0D0D14;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    letter-spacing: 0.03em;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #C49050;
}
/* Sidebar border */
[data-testid="stSidebar"] {
    border-right: 1px solid #2A2A3E;
}
/* Code blocks */
.stCode { border-radius: 8px; }
/* Success / info / warning banners */
[data-testid="stAlert"] { border-radius: 8px; }
/* Progress bar colour */
[data-testid="stProgressBar"] > div > div {
    background: #D4A76A !important;
}
</style>
""", unsafe_allow_html=True)

# ── Start TensorBoard once per Streamlit session ─────────────────────────────
if "tb_process" not in st.session_state:
    tb_logdir = os.path.join("outputs", "tensorboard")
    os.makedirs(tb_logdir, exist_ok=True)
    try:
        st.session_state.tb_process = subprocess.Popen(
            [sys.executable, "-m", "tensorboard",
             "--logdir", tb_logdir,
             "--port", "6006",
             "--host", "0.0.0.0"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        st.session_state.tb_process = None

# ── Page content ─────────────────────────────────────────────────────────────
st.title("DETR Object Detection")
st.caption("Detection Transformer — built from scratch in PyTorch")
st.divider()

# Device summary
gpu_count = torch.cuda.device_count()
has_mps   = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

if gpu_count > 0:
    device_label = f"CUDA — {gpu_count} GPU{'s' if gpu_count > 1 else ''}"
elif has_mps:
    device_label = "Apple Silicon (MPS)"
else:
    device_label = "CPU only"

ddp_ready = "Yes" if gpu_count > 1 else "No"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active device",    device_label)
col2.metric("CUDA GPUs",        gpu_count)
col3.metric("Multi-GPU DDP",    ddp_ready)
col4.metric("PyTorch",          torch.__version__)

st.divider()

# Quick-start instructions
st.subheader("Getting started")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
**1. Training**
Go to the **Training** page in the sidebar.
- Choose single or multi-GPU mode
- Pick a test run or full training
- Watch live loss curves update as training runs
""")
with c2:
    st.markdown("""
**2. Inference**
Go to the **Inference** page.
- Select a checkpoint (.pth file)
- Pick any image from the dataset
- See predictions vs ground truth instantly
""")
with c3:
    st.markdown("""
**3. History**
Go to the **History** page.
- View the last completed training run
- Interactive loss curves by tab
- Quick summary metrics at a glance
""")

st.divider()
st.caption(
    "TensorBoard is running at http://localhost:6006 — "
    "it is also embedded inside the Training page."
)
