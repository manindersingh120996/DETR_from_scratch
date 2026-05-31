"""History page — last training run metrics and interactive loss curves."""
import json
import os

import streamlit as st
import yaml

st.set_page_config(page_title="History", page_icon="📊", layout="wide")

st.markdown("""
<style>
[data-testid="stMetric"] {
    background: #16162A; border: 1px solid #2A2A3E;
    border-radius: 10px; padding: 14px 18px;
}
[data-testid="stSidebar"] { border-right: 1px solid #2A2A3E; }
</style>
""", unsafe_allow_html=True)

cfg = yaml.safe_load(open("configs/train.yaml"))
output_dir    = cfg.get("output_dir", "outputs")
progress_path = os.path.join(output_dir, "training_progress.json")
curves_path   = os.path.join(output_dir, "training_curves.png")

st.title("Training History")
st.caption("Showing the most recent training run. No database — just the last saved progress file.")
st.divider()

if not os.path.exists(progress_path):
    st.info("No training history found yet. Run a training session first.")
    st.stop()

with open(progress_path) as f:
    p = json.load(f)
h = p["history"]

status = p.get("status", "unknown")
if status == "completed":
    st.success(f"Run status: **Completed** ({p['epoch']} / {p['total_epochs']} epochs)")
elif status == "running":
    st.warning(f"Run status: **In progress** — epoch {p['epoch']} / {p['total_epochs']}")
else:
    st.info(f"Run status: {status}")

# ── Summary metrics ───────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Epochs completed",  p["epoch"])
m2.metric("Best val loss",     f'{p["best_val_loss"]:.4f}')
m3.metric("Final val acc",     f'{h["val_acc"][-1]:.2%}')
m4.metric("Final train acc",   f'{h["train_acc"][-1]:.2%}')

st.divider()

# ── Saved PNG ─────────────────────────────────────────────────────────────────
if os.path.exists(curves_path):
    st.subheader("Training curves (saved image)")
    st.image(curves_path, caption="Generated at end of training", use_container_width=True)
    st.divider()

# ── Interactive loss curves (tabbed) ─────────────────────────────────────────
st.subheader("Interactive loss curves")

tab_total, tab_ce, tab_l1, tab_giou, tab_acc = st.tabs(
    ["Total loss", "CE loss", "L1 loss", "GIoU loss", "Accuracy"]
)

_keys = [
    (tab_total, "total", "Total loss"),
    (tab_ce,    "ce",    "CE loss"),
    (tab_l1,    "l1",    "L1 loss"),
    (tab_giou,  "giou",  "GIoU loss"),
    (tab_acc,   "acc",   "Accuracy"),
]

for tab, key, label in _keys:
    with tab:
        train_key = f"train_{key}"
        val_key   = f"val_{key}"
        if train_key in h and val_key in h:
            tab.line_chart(
                {"Train": h[train_key], "Val": h[val_key]},
                use_container_width=True,
                color=["#D4A76A", "#7C6CD4"],
            )
            # Min/max summary
            c1, c2 = tab.columns(2)
            c1.metric(f"Best train {label}", f'{min(h[train_key]):.4f}')
            c2.metric(f"Best val {label}",   f'{min(h[val_key]):.4f}')
        else:
            tab.info(f"No data for '{key}'.")

st.divider()

# ── Raw epoch table ────────────────────────────────────────────────────────────
with st.expander("Raw epoch data (table)"):
    rows = []
    n = len(h.get("train_total", []))
    for i in range(n):
        rows.append({
            "Epoch":      i + 1,
            "Train loss": round(h["train_total"][i], 4),
            "Val loss":   round(h["val_total"][i],   4),
            "Train acc":  f'{h["train_acc"][i]:.2%}',
            "Val acc":    f'{h["val_acc"][i]:.2%}',
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
