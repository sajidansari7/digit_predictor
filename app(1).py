"""
Handwritten Digit Recognizer — Streamlit App
MLP built from scratch (NumPy) + CNN (PyTorch), compared side-by-side.

Author : Sajid Ansari
Project: Handwritten Digit Recognizer
"""

import sys, os, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import torch
from my_library.mlp import MLPClassifier, DenseLayer
from my_library.cnn import DigitCNN, predict_cnn

# ── Auto-train if models don't exist (for Streamlit Cloud) ────────────────
def ensure_models():
    needed = ["models/mlp_weights.npz", "models/cnn_pytorch.pth", "models/eval_data.pkl"]
    if not all(os.path.exists(p) for p in needed):
        st.info("⏳ First run: training models on MNIST (~3 min). Please wait...")
        with st.spinner("Training MLP and CNN..."):
            result = subprocess.run(
                [sys.executable, "train.py"],
                capture_output=True, text=True
            )
        if result.returncode != 0:
            st.error(f"Training failed:\n{result.stderr}")
            st.stop()
        st.success("✅ Models trained! Loading app...")
        st.rerun()

ensure_models()

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

section[data-testid="stSidebar"] {
    background: #060810;
    border-right: 1px solid #161b2e;
}
section[data-testid="stSidebar"] * { color: #c9d1e3 !important; }
.main { background: #080b14; }
.block-container { padding: 2rem 2.5rem; }

.hero-title {
    font-size: 2.6rem; font-weight: 700; letter-spacing: -0.03em;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2; margin-bottom: 0.3rem;
}
.hero-sub { font-size: 1rem; color: #475569; margin-bottom: 1.5rem; }

.stat-card {
    background: #0d1117;
    border: 1px solid #1e2740;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
}
.stat-val {
    font-size: 2rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.stat-val.mlp  { color: #818cf8; }
.stat-val.cnn  { color: #38bdf8; }
.stat-label { font-size: 0.72rem; color: #475569; text-transform: uppercase; letter-spacing: 0.1em; }

.section-head {
    font-size: 0.72rem; font-weight: 600; color: #818cf8;
    text-transform: uppercase; letter-spacing: 0.14em;
    margin: 1.4rem 0 0.7rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid #161b2e;
}

.pred-box {
    border-radius: 12px; padding: 1.2rem 1.5rem;
    text-align: center; margin-bottom: 0.8rem;
}
.pred-digit { font-size: 3.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.pred-conf  { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-top: 0.2rem; }
.pred-model { font-size: 0.7rem; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.1em; }

.tag {
    display: inline-block; background: #0d1117; border: 1px solid #1e2740;
    color: #818cf8; font-size: 0.7rem; padding: 0.12rem 0.45rem;
    border-radius: 20px; margin: 0.1rem;
    font-family: 'JetBrains Mono', monospace;
}
.divider { border: none; border-top: 1px solid #161b2e; margin: 1.2rem 0; }
h1, h2, h3 { color: #e2e8f0 !important; }
p, li       { color: #64748b; }
</style>
""", unsafe_allow_html=True)

# ── Load models ─────────────────────────────────────────────────────────────
def load_mlp_from_weights():
    """Rebuild MLP object and restore weights from .npz (avoids 131MB pickle)."""
    with open("models/mlp_config.json") as f:
        cfg = json.load(f)
    mlp = MLPClassifier(
        layer_sizes  = cfg["layer_sizes"],
        activations  = cfg["activations"],
        epochs=1, verbose=False,
    )
    mlp._build()
    # Restore history
    mlp.train_losses = cfg["train_losses"]
    mlp.val_losses   = cfg["val_losses"]
    mlp.train_accs   = cfg["train_accs"]
    mlp.val_accs     = cfg["val_accs"]
    # Restore weights — keys use original layer indices (including dropout layers)
    weights  = np.load("models/mlp_weights.npz")
    key_iter = iter(sorted(k for k in weights.keys() if k.startswith("W_")),)
    dense_layers = [l for l in mlp.layers if hasattr(l, 'W')]
    for key_w in sorted(k for k in weights.keys() if k.startswith("W_")):
        idx = key_w.split("_")[1]
        key_b = f"b_{idx}"
        layer_idx = sorted(k for k in weights.keys() if k.startswith("W_")).index(key_w)
        dense_layers[layer_idx].W = weights[key_w]
        dense_layers[layer_idx].b = weights[key_b]
    return mlp

@st.cache_resource
def load_models():
    mlp    = load_mlp_from_weights()
    eval_d = joblib.load("models/eval_data.pkl")
    cnn    = DigitCNN()
    cnn.load_state_dict(torch.load("models/cnn_pytorch.pth", map_location="cpu"))
    cnn.eval()
    return mlp, cnn, eval_d

mlp, cnn_model, eval_data = load_models()

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✍️ Digit Recognizer")
    st.markdown("<p style='font-size:0.78rem;color:#334155;'>MLP from scratch · CNN via PyTorch</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", [
        "✍️ Draw & Predict",
        "📊 Model Dashboard",
        "🔬 CNN Internals",
        "📖 Architecture",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#334155; line-height:1.8;'>
    <b style='color:#475569;'>From scratch</b><br>
    <span class='tag'>NumPy backprop</span>
    <span class='tag'>Adam optimizer</span>
    <span class='tag'>Dropout</span><br><br>
    <b style='color:#475569;'>Framework</b><br>
    <span class='tag'>PyTorch CNN</span>
    <span class='tag'>LeNet arch</span><br><br>
    <b style='color:#475569;'>Dataset</b><br>
    <span class='tag'>MNIST 70k images</span><br><br>
    <b style='color:#475569;'>Author</b><br>
    <span style='color:#818cf8;'>Sajid Ansari</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 1: DRAW & PREDICT
# ═══════════════════════════════════════════════════════════════════════════
if page == "✍️ Draw & Predict":
    st.markdown("<div class='hero-title'>Handwritten Digit Recognizer</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Draw a digit — both models predict in real time.</div>", unsafe_allow_html=True)

    from streamlit_drawable_canvas import st_canvas
    from PIL import Image

    col_canvas, col_pred = st.columns([1, 1], gap="large")

    with col_canvas:
        st.markdown("<div class='section-head'>Draw Here</div>", unsafe_allow_html=True)

        canvas_result = st_canvas(
            fill_color  = "rgba(0,0,0,0)",
            stroke_width= 18,
            stroke_color= "#FFFFFF",
            background_color = "#000000",
            height      = 280,
            width       = 280,
            drawing_mode= "freedraw",
            key         = "canvas",
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn2:
            predict_clicked = st.button("✨ Predict", use_container_width=True, type="primary")

        # ── Process drawn image ───────────────────────────────────────────
        drawn_img   = None
        mlp_proba   = None
        cnn_proba   = None
        mlp_pred    = None
        cnn_pred    = None

        if canvas_result.image_data is not None:
            # image_data is RGBA (280,280,4) — extract grayscale from R channel
            img_array = canvas_result.image_data[:, :, 0].astype(np.float32) / 255.0
            has_drawing = img_array.max() > 0.05

            if has_drawing:
                # Resize 280×280 → 28×28 using PIL
                pil_img    = Image.fromarray((img_array * 255).astype(np.uint8), mode="L")
                pil_resized= pil_img.resize((28, 28), Image.LANCZOS)
                drawn_img  = np.array(pil_resized).astype(np.float32) / 255.0

                # Predict
                flat      = drawn_img.flatten().reshape(1, -1)
                mlp_proba = mlp.predict_proba(flat)[0]
                mlp_pred  = int(np.argmax(mlp_proba))

                cnn_preds_live, cnn_proba_live = predict_cnn(cnn_model, drawn_img.reshape(1, 28, 28))
                cnn_proba = cnn_proba_live[0]
                cnn_pred  = int(cnn_preds_live[0])

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-head'>Or try a sample from test set</div>", unsafe_allow_html=True)
        sample_images = np.array(eval_data["sample_images"])
        sample_labels = np.array(eval_data["sample_labels"])
        sample_mlp    = np.array(eval_data["sample_mlp_pred"])
        sample_cnn    = np.array(eval_data["sample_cnn_pred"])

        digit_filter = st.selectbox("Filter by digit", ["All"] + list(range(10)))
        if digit_filter == "All":
            indices = list(range(len(sample_labels)))
        else:
            indices = [i for i, l in enumerate(sample_labels) if l == int(digit_filter)]

        if indices:
            idx = st.slider("Sample index", 0, len(indices)-1, 0)
            chosen    = indices[idx]
            img       = sample_images[chosen]
            true_label= sample_labels[chosen]
            sml_pred  = sample_mlp[chosen]
            scnn_pred = sample_cnn[chosen]

            fig_img = go.Figure(go.Heatmap(z=img[::-1], colorscale="Greys", showscale=False))
            fig_img.update_layout(
                height=180, width=180,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                margin=dict(t=5, b=5, l=5, r=5),
            )
            st.plotly_chart(fig_img, use_container_width=False)
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("True Label", f"  {true_label}")
            with c2: st.metric("MLP Pred", f"  {sml_pred}", delta="✅" if sml_pred==true_label else "❌", delta_color="off")
            with c3: st.metric("CNN Pred", f"  {scnn_pred}", delta="✅" if scnn_pred==true_label else "❌", delta_color="off")

            # Use sample for predictions if nothing drawn
            if drawn_img is None:
                flat      = img.flatten().reshape(1, -1)
                mlp_proba = mlp.predict_proba(flat)[0]
                mlp_pred  = int(np.argmax(mlp_proba))
                cnn_preds_live, cnn_proba_live = predict_cnn(cnn_model, img.reshape(1, 28, 28))
                cnn_proba = cnn_proba_live[0]
                cnn_pred  = int(cnn_preds_live[0])

        with col_pred:
            if mlp_proba is not None and cnn_proba is not None:
                st.markdown("<div class='section-head'>Predictions</div>", unsafe_allow_html=True)

                # MLP result
                st.markdown(f"""
                <div class='pred-box' style='background:linear-gradient(135deg,#1e1b4b,#1a1040);border:1px solid #4338ca;'>
                  <div class='pred-model'>MLP · NumPy from scratch</div>
                  <div class='pred-digit' style='color:#818cf8;'>{mlp_pred}</div>
                  <div class='pred-conf'>Confidence: {mlp_proba[int(mlp_pred)]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

                # CNN result
                st.markdown(f"""
                <div class='pred-box' style='background:linear-gradient(135deg,#0c1a2e,#0a1628);border:1px solid #0369a1;'>
                  <div class='pred-model'>CNN · PyTorch LeNet</div>
                  <div class='pred-digit' style='color:#38bdf8;'>{cnn_pred}</div>
                  <div class='pred-conf'>Confidence: {cnn_proba[int(cnn_pred)]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

                # Probability bars
                st.markdown("<div class='section-head'>Probability Distribution</div>", unsafe_allow_html=True)
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=list(range(10)), y=mlp_proba, name="MLP",
                    marker_color=["#818cf8" if i == mlp_pred else "#1e2740" for i in range(10)],
                ))
                fig_bar.add_trace(go.Bar(
                    x=list(range(10)), y=cnn_proba, name="CNN",
                    marker_color=["#38bdf8" if i == cnn_pred else "#0f2233" for i in range(10)],
                ))
                fig_bar.update_layout(
                    barmode="group", height=250,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080b14",
                    font_color="#64748b",
                    legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94a3b8"),
                    xaxis=dict(tickmode="array", tickvals=list(range(10)), gridcolor="#161b2e"),
                    yaxis=dict(tickformat=".0%", gridcolor="#161b2e"),
                    margin=dict(t=10, b=30, l=50, r=10),
                )
                st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 2: MODEL DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Dashboard":
    st.markdown("# 📊 Model Dashboard")
    st.markdown("<p>Side-by-side evaluation of MLP (NumPy scratch) vs CNN (PyTorch).</p>", unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    mlp_d = eval_data["mlp"]
    cnn_d = eval_data["cnn"]

    # Top stats
    c1, c2, c3, c4 = st.columns(4)
    for col, label, mv, cv, mc, cc in [
        (c1, "Test Accuracy",  f"{mlp_d['accuracy']*100:.2f}%", f"{cnn_d['accuracy']*100:.2f}%", "mlp", "cnn"),
        (c2, "Val Accuracy",   f"{max(mlp_d['val_acc'])*100:.2f}%", f"{max(cnn_d['val_acc'])*100:.2f}%", "mlp", "cnn"),
        (c3, "Best Val Loss",  f"{min(mlp_d['val_loss']):.4f}",  f"{min(cnn_d['val_loss']):.4f}",  "mlp", "cnn"),
        (c4, "Parameters",     "~235K", "~44K", "mlp", "cnn"),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-label'>{label}</div>
              <div style='display:flex;gap:1rem;margin-top:0.5rem;align-items:flex-end;'>
                <div>
                  <div style='font-size:0.6rem;color:#334155;margin-bottom:2px;'>MLP</div>
                  <div class='stat-val {mc}'>{mv}</div>
                </div>
                <div>
                  <div style='font-size:0.6rem;color:#334155;margin-bottom:2px;'>CNN</div>
                  <div class='stat-val {cc}'>{cv}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Learning curves
    st.markdown("<div class='section-head'>Learning Curves</div>", unsafe_allow_html=True)
    lc_col, acc_col = st.columns(2, gap="large")

    def lc_chart(title, mlp_tr, mlp_val, cnn_tr, cnn_val, yaxis_title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=mlp_tr, name="MLP Train", line=dict(color="#818cf8", width=2, dash="dot")))
        fig.add_trace(go.Scatter(y=mlp_val, name="MLP Val",  line=dict(color="#818cf8", width=2)))
        fig.add_trace(go.Scatter(y=cnn_tr, name="CNN Train", line=dict(color="#38bdf8", width=2, dash="dot")))
        fig.add_trace(go.Scatter(y=cnn_val, name="CNN Val",  line=dict(color="#38bdf8", width=2)))
        fig.update_layout(
            title=dict(text=title, font=dict(color="#94a3b8", size=13)),
            yaxis_title=yaxis_title, xaxis_title="Epoch",
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080b14",
            font_color="#64748b",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11)),
            xaxis=dict(gridcolor="#161b2e"), yaxis=dict(gridcolor="#161b2e"),
            margin=dict(t=35, b=35, l=55, r=10),
        )
        return fig

    with lc_col:
        st.plotly_chart(lc_chart("Loss", mlp_d["train_loss"], mlp_d["val_loss"],
                                  cnn_d["train_loss"], cnn_d["val_loss"], "Cross-Entropy Loss"),
                         use_container_width=True)
    with acc_col:
        st.plotly_chart(lc_chart("Accuracy", mlp_d["train_acc"], mlp_d["val_acc"],
                                  cnn_d["train_acc"], cnn_d["val_acc"], "Accuracy"),
                         use_container_width=True)

    # Confusion matrices
    st.markdown("<div class='section-head'>Confusion Matrices</div>", unsafe_allow_html=True)
    cm_sel = st.radio("Select model", ["MLP", "CNN"], horizontal=True)
    cm     = np.array(mlp_d["cm"] if cm_sel == "MLP" else cnn_d["cm"])
    color  = "#818cf8" if cm_sel == "MLP" else "#38bdf8"

    fig_cm = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                            subplot_titles=["Confusion Matrix (counts)", "Confusion Matrix (%)"])
    labels = [str(i) for i in range(10)]
    norm   = cm / cm.sum(axis=1, keepdims=True) * 100

    for j, (data, fmt) in enumerate([(cm, "%d"), (norm, ".1f")], 1):
        fig_cm.add_trace(go.Heatmap(
            z=data, x=labels, y=labels,
            text=np.vectorize(lambda v: f"{v:{fmt[1:]}}")(data),
            texttemplate="%{text}",
            colorscale=[[0,"#080b14"],[0.5,"#1e2740"],[1,color]],
            showscale=False, textfont={"size": 9},
        ), row=1, col=j)

    fig_cm.update_layout(
        height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080b14",
        font_color="#64748b", margin=dict(t=40, b=30, l=40, r=20),
    )
    fig_cm.update_xaxes(gridcolor="#161b2e")
    fig_cm.update_yaxes(gridcolor="#161b2e")
    st.plotly_chart(fig_cm, use_container_width=True)

    # Per-class F1
    st.markdown("<div class='section-head'>Per-class F1 Score</div>", unsafe_allow_html=True)
    digits = list(range(10))
    mlp_f1 = [mlp_d["per_class"][d]["f1"] for d in digits]
    cnn_f1 = [cnn_d["per_class"][d]["f1"] for d in digits]

    fig_f1 = go.Figure()
    fig_f1.add_trace(go.Bar(x=digits, y=mlp_f1, name="MLP", marker_color="#818cf8", opacity=0.85))
    fig_f1.add_trace(go.Bar(x=digits, y=cnn_f1, name="CNN", marker_color="#38bdf8", opacity=0.85))
    fig_f1.update_layout(
        barmode="group", height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080b14",
        font_color="#64748b",
        xaxis=dict(tickmode="array", tickvals=digits, title="Digit", gridcolor="#161b2e"),
        yaxis=dict(range=[0.93, 1.01], tickformat=".3f", gridcolor="#161b2e"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94a3b8"),
        margin=dict(t=10, b=35, l=55, r=10),
    )
    st.plotly_chart(fig_f1, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 3: CNN INTERNALS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔬 CNN Internals":
    st.markdown("# 🔬 CNN Internals")
    st.markdown("<p>Visualize what the CNN actually learns — filters and feature maps.</p>", unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Conv1 Filters
    st.markdown("<div class='section-head'>Conv1 Learned Filters (6 × 5×5)</div>", unsafe_allow_html=True)
    st.markdown("<p>These are the raw weights of the first convolutional layer after training. Each filter detects a specific low-level pattern (edges, curves, strokes).</p>", unsafe_allow_html=True)

    filters = np.array(eval_data["cnn"]["filters"])  # (6, 1, 5, 5)
    fig_filt = make_subplots(rows=1, cols=6,
                              subplot_titles=[f"Filter {i+1}" for i in range(6)],
                              horizontal_spacing=0.04)
    for i in range(6):
        f = filters[i, 0]
        fig_filt.add_trace(go.Heatmap(
            z=f, colorscale="RdBu", showscale=(i==5),
            zmid=0,
        ), row=1, col=i+1)
    fig_filt.update_layout(
        height=180, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080b14",
        font_color="#64748b", margin=dict(t=30, b=10, l=10, r=10),
    )
    fig_filt.update_xaxes(visible=False)
    fig_filt.update_yaxes(visible=False)
    st.plotly_chart(fig_filt, use_container_width=True)

    # Feature maps on a selected sample
    st.markdown("<div class='section-head'>Feature Maps on a Sample Digit</div>", unsafe_allow_html=True)
    st.markdown("<p>What each filter 'sees' when it scans over the input image. Bright areas = strong activation.</p>", unsafe_allow_html=True)

    sample_images = np.array(eval_data["sample_images"])
    sample_labels = np.array(eval_data["sample_labels"])

    col_pick, col_map = st.columns([0.4, 0.6], gap="large")
    with col_pick:
        dig_choice = st.selectbox("Pick digit class", list(range(10)), index=3)
        idxs = [i for i, l in enumerate(sample_labels) if l == dig_choice]
        if idxs:
            chosen = idxs[0]
            img    = sample_images[chosen]
            fig_raw = go.Figure(go.Heatmap(z=img[::-1], colorscale="Greys", showscale=False))
            fig_raw.update_layout(
                height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080b14",
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                margin=dict(t=5, b=5, l=5, r=5),
                title=dict(text=f"Input: digit {dig_choice}", font=dict(color="#94a3b8", size=12))
            )
            st.plotly_chart(fig_raw, use_container_width=True)

    with col_map:
        if idxs:
            # Get feature maps from CNN
            x_tensor = torch.tensor(img.reshape(1, 1, 28, 28).astype(np.float32))
            with torch.no_grad():
                maps = cnn_model.get_feature_maps(x_tensor)

            # Conv1 maps: (1, 6, 24, 24)
            conv1_maps = maps["conv1"][0].numpy()  # (6, 24, 24)
            fig_maps = make_subplots(rows=1, cols=6,
                                      subplot_titles=[f"Map {i+1}" for i in range(6)],
                                      horizontal_spacing=0.04)
            for i in range(6):
                fig_maps.add_trace(go.Heatmap(
                    z=conv1_maps[i][::-1],
                    colorscale=[[0,"#080b14"],[0.5,"#1e2740"],[1,"#38bdf8"]],
                    showscale=False,
                ), row=1, col=i+1)
            fig_maps.update_layout(
                height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080b14",
                font_color="#64748b", margin=dict(t=30, b=10, l=10, r=10),
                title=dict(text="Conv1 Feature Maps (after ReLU)", font=dict(color="#94a3b8", size=12))
            )
            fig_maps.update_xaxes(visible=False)
            fig_maps.update_yaxes(visible=False)
            st.plotly_chart(fig_maps, use_container_width=True)

            # Conv2 maps: (1, 16, 8, 8)
            if "conv2" in maps:
                conv2_maps = maps["conv2"][0].numpy()  # (16, 8, 8)
                fig_maps2 = make_subplots(rows=2, cols=8,
                                           horizontal_spacing=0.03, vertical_spacing=0.08)
                for i in range(16):
                    r, c = divmod(i, 8)
                    fig_maps2.add_trace(go.Heatmap(
                        z=conv2_maps[i][::-1],
                        colorscale=[[0,"#080b14"],[0.5,"#1e2740"],[1,"#c084fc"]],
                        showscale=False,
                    ), row=r+1, col=c+1)
                fig_maps2.update_layout(
                    height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080b14",
                    font_color="#64748b", margin=dict(t=30, b=10, l=10, r=10),
                    title=dict(text="Conv2 Feature Maps — 16 deeper patterns (8×8)", font=dict(color="#94a3b8", size=12))
                )
                fig_maps2.update_xaxes(visible=False)
                fig_maps2.update_yaxes(visible=False)
                st.plotly_chart(fig_maps2, use_container_width=True)

    # Activation distribution
    st.markdown("<div class='section-head'>What the Model Struggles With</div>", unsafe_allow_html=True)
    sample_mlp = np.array(eval_data["sample_mlp_pred"])
    sample_cnn = np.array(eval_data["sample_cnn_pred"])

    mlp_wrong = [(sample_labels[i], sample_mlp[i]) for i in range(len(sample_labels)) if sample_mlp[i] != sample_labels[i]]
    cnn_wrong = [(sample_labels[i], sample_cnn[i]) for i in range(len(sample_labels)) if sample_cnn[i] != sample_labels[i]]

    w1, w2 = st.columns(2, gap="large")
    for col, wrong, name, color in [(w1, mlp_wrong, "MLP", "#818cf8"), (w2, cnn_wrong, "CNN", "#38bdf8")]:
        with col:
            if wrong:
                pairs = [f"{t}→{p}" for t,p in wrong[:12]]
                st.markdown(f"<div style='color:{color};font-size:0.8rem;font-weight:600;margin-bottom:0.5rem;'>{name} Mistakes (true→predicted)</div>", unsafe_allow_html=True)
                st.markdown(" ".join([f"<span class='tag' style='color:{color};'>{p}</span>" for p in pairs]), unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='color:#334155;'>No errors in sample set for {name}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 4: ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📖 Architecture":
    st.markdown("# 📖 Architecture & Math")
    st.markdown("<p>The complete implementation — from partial derivatives to convolutional pooling.</p>", unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='section-head'>MLP from Scratch</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#64748b;line-height:2;font-size:0.88rem;'>
        <b style='color:#e2e8f0;'>Layer sizes:</b> 784 → 256 → 128 → 10<br>
        <b style='color:#e2e8f0;'>Activation:</b> ReLU (hidden), Softmax (output)<br>
        <b style='color:#e2e8f0;'>Optimizer:</b> Adam with bias correction<br>
        <b style='color:#e2e8f0;'>Regularization:</b> L2 + Dropout (keep=0.8)<br>
        <b style='color:#e2e8f0;'>Init:</b> He initialization for ReLU layers<br><br>

        <b style='color:#c084fc;'>Forward pass:</b><br>
        <code style='background:#0d1117;padding:3px 8px;border-radius:4px;color:#818cf8;font-size:0.8rem;'>
        Z = A_prev · W + b<br>
        A = ReLU(Z) = max(0, Z)
        </code><br><br>

        <b style='color:#c084fc;'>Loss:</b><br>
        <code style='background:#0d1117;padding:3px 8px;border-radius:4px;color:#818cf8;font-size:0.8rem;'>
        L = −(1/m) Σ log(ŷ[y])
        </code><br><br>

        <b style='color:#c084fc;'>Backprop (softmax + CE combined):</b><br>
        <code style='background:#0d1117;padding:3px 8px;border-radius:4px;color:#818cf8;font-size:0.8rem;'>
        dZ = ŷ − y_onehot<br>
        dW = (1/m) A_prev.T · dZ + λW<br>
        dA_prev = dZ · W.T
        </code><br><br>

        <b style='color:#c084fc;'>Adam update:</b><br>
        <code style='background:#0d1117;padding:3px 8px;border-radius:4px;color:#818cf8;font-size:0.8rem;'>
        m̂ = m / (1 − β₁ᵗ)<br>
        v̂ = v / (1 − β₂ᵗ)<br>
        W -= α · m̂ / (√v̂ + ε)
        </code>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-head'>CNN (PyTorch LeNet-style)</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#64748b;line-height:2;font-size:0.88rem;'>
        <b style='color:#e2e8f0;'>Architecture:</b><br>
        Conv1(1→6, 5×5) → ReLU → MaxPool(2×2)<br>
        Conv2(6→16, 5×5) → ReLU → MaxPool(2×2)<br>
        FC(256→120) → ReLU → Dropout<br>
        FC(120→84) → ReLU → Dropout<br>
        FC(84→10)<br><br>

        <b style='color:#38bdf8;'>Convolution:</b><br>
        <code style='background:#0d1117;padding:3px 8px;border-radius:4px;color:#38bdf8;font-size:0.8rem;'>
        (X * W)[i,j] = Σ X[i+m, j+n] · W[m,n]
        </code><br><br>

        <b style='color:#38bdf8;'>Max Pooling (2×2):</b><br>
        <code style='background:#0d1117;padding:3px 8px;border-radius:4px;color:#38bdf8;font-size:0.8rem;'>
        P[i,j] = max(X[2i:2i+2, 2j:2j+2])
        </code><br><br>

        <b style='color:#38bdf8;'>Why CNN beats MLP:</b><br>
        <span style='color:#475569;'>
        • Parameter sharing: same filter scans entire image<br>
        • Translation invariance via pooling<br>
        • Local connectivity: detects edges anywhere<br>
        • MLP treats every pixel independently (no spatial info)
        </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-head'>Project Structure</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0d1117;border:1px solid #161b2e;border-radius:10px;
                padding:1.2rem 1.5rem;font-family:"JetBrains Mono",monospace;
                font-size:0.8rem;color:#64748b;line-height:2;'>
    digit_recognizer/<br>
    ├── <span style='color:#818cf8;'>my_library/</span>             <span style='color:#1e2740;'># all core ML from scratch</span><br>
    │   ├── mlp.py              <span style='color:#1e2740;'># MLP: layers, activations, backprop, Adam</span><br>
    │   ├── cnn.py              <span style='color:#1e2740;'># CNN: LeNet architecture + training loop</span><br>
    │   └── metrics.py          <span style='color:#1e2740;'># accuracy, confusion matrix, per-class F1</span><br>
    ├── <span style='color:#38bdf8;'>data/</span>                   <span style='color:#1e2740;'># MNIST downloaded here</span><br>
    ├── <span style='color:#38bdf8;'>models/</span>                 <span style='color:#1e2740;'># .pkl (MLP) + .pth (CNN) saved here</span><br>
    ├── train.py                <span style='color:#1e2740;'># trains both models, saves artifacts</span><br>
    ├── app.py                  <span style='color:#1e2740;'># this Streamlit app</span><br>
    └── requirements.txt<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-head'>CV-worthy Highlights</div>", unsafe_allow_html=True)
    items = [
        ("🧮", "Backpropagation from scratch", "Full chain rule implementation in NumPy — dZ, dW, db for every layer manually computed."),
        ("⚡", "Adam optimizer from scratch",   "Bias-corrected first and second moment estimates — same math as the original 2014 paper."),
        ("🔭", "CNN filter visualization",      "Visualize what the Conv layers actually learn — filters and feature maps per digit."),
        ("📊", "98.24% MLP / 98.73% CNN",       "Strong accuracy on 10,000 MNIST test images with both models."),
        ("🎨", "Interactive draw canvas",       "HTML5 canvas → downscale to 28×28 → live prediction from both models."),
        ("🔬", "MLP vs CNN comparison",         "Side-by-side learning curves, confusion matrices, per-class F1 — shows understanding of both paradigms."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(items):
        with cols[i % 3]:
            st.markdown(f"""
            <div style='background:#0d1117;border:1px solid #1e2740;border-radius:10px;
                        padding:1rem;margin-bottom:0.8rem;'>
              <div style='font-size:1.4rem;'>{icon}</div>
              <div style='font-weight:600;color:#e2e8f0;margin:0.3rem 0;font-size:0.88rem;'>{title}</div>
              <div style='font-size:0.75rem;color:#334155;'>{desc}</div>
            </div>""", unsafe_allow_html=True)
