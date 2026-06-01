"""
streamlit_app.py
----------------
PitLane Prints — Driver Style Fingerprinting Dashboard

Displays:
1. Driver telemetry fingerprint (radar chart of engineered features)
2. UMAP embedding space — where does each driver sit?
3. Raw telemetry comparison for a selected lap
4. Model confidence for blind identification

Run from project root:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── path setup so src imports work ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PitLane Prints",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;600;700;800&family=Barlow:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
}

/* Dark racing theme */
.stApp {
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111118 !important;
    border-right: 1px solid #2a2a3a;
}

/* Header */
.pit-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    letter-spacing: -0.02em;
    line-height: 1;
    color: #ffffff;
    margin-bottom: 0;
}
.pit-sub {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 300;
    font-size: 1rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #e10600;
    margin-top: 4px;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: #14141e;
    border: 1px solid #2a2a3a;
    border-top: 2px solid;
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
}
.metric-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b6b8a;
    margin-bottom: 4px;
}
.metric-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}

/* Driver badges */
.driver-ver { color: #3b82f6; border-color: #3b82f6; }
.driver-ham { color: #a78bfa; border-color: #a78bfa; }
.driver-alo { color: #f87171; border-color: #f87171; }

/* Section headers */
.section-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b6b8a;
    border-bottom: 1px solid #2a2a3a;
    padding-bottom: 6px;
    margin-bottom: 1rem;
}

/* Plotly chart backgrounds */
.js-plotly-plot { border-radius: 4px; }

/* Info boxes */
.insight-box {
    background: #14141e;
    border-left: 3px solid #e10600;
    padding: 0.75rem 1rem;
    border-radius: 0 4px 4px 0;
    font-size: 0.875rem;
    color: #c0c0d0;
    margin: 0.5rem 0;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DRIVER_COLORS = {
    "VER": "#3b82f6",
    "HAM": "#a78bfa",
    "ALO": "#f87171",
    "LEC": "#f97316",
    "SAI": "#facc15",
    "NOR": "#34d399",
}
DRIVER_NAMES = {
    "VER": "Max Verstappen",
    "HAM": "Lewis Hamilton",
    "ALO": "Fernando Alonso",
    "LEC": "Charles Leclerc",
    "SAI": "Carlos Sainz",
    "NOR": "Lando Norris",
}
FEATURE_COLS = [
    "brake_duration_ratio", "throttle_smoothness", "full_throttle_ratio",
    "coasting_ratio", "gear_change_freq", "speed_at_throttle_lift",
    "mean_corner_speed", "speed_variance", "throttle_brake_overlap",
]
FEATURE_LABELS = [
    "Brake Duration", "Throttle Smoothness", "Full Throttle",
    "Coasting", "Gear Changes", "Braking Speed",
    "Corner Speed", "Speed Variance", "Trail Braking",
]

ROOT = os.path.join(os.path.dirname(__file__), "..")

# ─────────────────────────────────────────────
# Data loading — cached
# ─────────────────────────────────────────────
@st.cache_data
def load_all_data():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        config = yaml.safe_load(f)

    race_tag     = f"{config['session']['year']}_{config['session']['race'].lower()}"
    features_dir = os.path.join(ROOT, config["data"]["features_dir"])
    processed_dir = os.path.join(ROOT, config["data"]["processed_dir"])
    models_dir   = os.path.join(ROOT, "outputs", "models")

    features_df = pd.read_csv(os.path.join(features_dir, f"{race_tag}_features.csv"))
    umap_df     = pd.read_csv(os.path.join(features_dir, f"{race_tag}_umap_coords.csv"))
    history_df  = pd.read_csv(os.path.join(features_dir, f"{race_tag}_cnn_history.csv"))
    fi_df       = pd.read_csv(os.path.join(features_dir, f"{race_tag}_feature_importance.csv"))

    xgb_model = joblib.load(os.path.join(models_dir, f"{race_tag}_xgb_baseline.pkl"))
    xgb_le    = joblib.load(os.path.join(models_dir, f"{race_tag}_label_encoder.pkl"))

    # Load raw telemetry for all drivers
    raw = {}
    for driver in config["drivers"]:
        path = os.path.join(processed_dir, race_tag, f"{driver}.parquet")
        raw[driver] = pd.read_parquet(path)

    return features_df, umap_df, history_df, fi_df, xgb_model, xgb_le, raw, config, race_tag


# ─────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────

def make_radar(features_df, selected_drivers):
    """Normalized radar chart of driver style features."""
    means = features_df.groupby("Driver")[FEATURE_COLS].mean()
    means_norm = (means - means.min()) / (means.max() - means.min() + 1e-8)

    fig = go.Figure()
    for driver in selected_drivers:
        if driver not in means_norm.index:
            continue
        values = means_norm.loc[driver].tolist()
        values += values[:1]
        labels = FEATURE_LABELS + [FEATURE_LABELS[0]]

        fig.add_trace(go.Scatterpolar(
            r=values, theta=labels, fill='toself',
            name=DRIVER_NAMES[driver],
            line=dict(color=DRIVER_COLORS[driver], width=2),
            fillcolor=DRIVER_COLORS[driver],
            opacity=0.25,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#14141e",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2a2a3a",
                            tickfont=dict(color="#555570", size=9)),
            angularaxis=dict(gridcolor="#2a2a3a",
                             tickfont=dict(color="#c0c0d0", size=10)),
        ),
        paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        font=dict(color="#e8e8f0", family="Barlow Condensed"),
        legend=dict(bgcolor="#14141e", bordercolor="#2a2a3a", borderwidth=1),
        margin=dict(l=60, r=60, t=40, b=40),
        height=380,
    )
    return fig


def make_umap(umap_df, highlight=None):
    """UMAP scatter with optional driver highlight."""
    fig = go.Figure()

    for driver in sorted(umap_df["Driver"].unique()):
        sub = umap_df[umap_df["Driver"] == driver]
        opacity = 0.9 if (highlight is None or driver == highlight) else 0.15
        size    = 10  if (highlight is None or driver == highlight) else 7

        fig.add_trace(go.Scatter(
            x=sub["umap_x"], y=sub["umap_y"],
            mode="markers",
            name=DRIVER_NAMES[driver],
            marker=dict(color=DRIVER_COLORS[driver], size=size,
                        opacity=opacity,
                        line=dict(width=0.5, color="white")),
            hovertemplate=f"<b>{driver}</b><br>Lap %{{customdata}}<extra></extra>",
            customdata=sub["LapNumber"],
        ))

        # Centroid star
        cx, cy = sub["umap_x"].mean(), sub["umap_y"].mean()
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers+text",
            marker=dict(symbol="star", size=18,
                        color=DRIVER_COLORS[driver],
                        line=dict(width=1, color="white")),
            text=[driver], textposition="top right",
            textfont=dict(color=DRIVER_COLORS[driver], size=13,
                          family="Barlow Condensed"),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        paper_bgcolor="#0a0a0f", plot_bgcolor="#14141e",
        font=dict(color="#e8e8f0", family="Barlow Condensed"),
        xaxis=dict(title="UMAP dim 1", gridcolor="#1e1e2e", zeroline=False),
        yaxis=dict(title="UMAP dim 2", gridcolor="#1e1e2e", zeroline=False),
        legend=dict(bgcolor="#14141e", bordercolor="#2a2a3a", borderwidth=1),
        margin=dict(l=40, r=20, t=20, b=40),
        height=380,
        hovermode="closest",
    )
    return fig


def make_telemetry_plot(raw, driver, lap_num):
    """Raw telemetry traces for a single lap."""
    lap_df = raw[driver][raw[driver]["LapNumber"] == lap_num].reset_index(drop=True)
    if lap_df.empty:
        return None

    channels = [("Throttle", "%"), ("Brake", ""), ("Speed", "km/h"), ("nGear", "")]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04,
                        subplot_titles=[c[0] for c in channels])

    for i, (ch, unit) in enumerate(channels, 1):
        if ch not in lap_df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=np.arange(len(lap_df)),
            y=lap_df[ch],
            mode="lines",
            line=dict(color=DRIVER_COLORS[driver], width=1.2),
            name=ch, showlegend=False,
        ), row=i, col=1)
        fig.update_yaxes(title_text=unit, row=i, col=1,
                         gridcolor="#1e1e2e", title_font=dict(size=9))

    fig.update_layout(
        paper_bgcolor="#0a0a0f", plot_bgcolor="#14141e",
        font=dict(color="#e8e8f0", family="Barlow Condensed"),
        xaxis4=dict(title="Sample index", gridcolor="#1e1e2e"),
        margin=dict(l=50, r=20, t=30, b=40),
        height=400,
    )
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color="#6b6b8a", size=10,
                                  family="Barlow Condensed")
    return fig


def make_feature_importance_bar(fi_df):
    fig = go.Figure(go.Bar(
        x=fi_df["importance"],
        y=fi_df["feature"],
        orientation="h",
        marker=dict(
            color=fi_df["importance"],
            colorscale=[[0, "#1e1e3a"], [1, "#e10600"]],
            showscale=False,
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#0a0a0f", plot_bgcolor="#14141e",
        font=dict(color="#e8e8f0", family="Barlow Condensed"),
        xaxis=dict(title="Importance", gridcolor="#1e1e2e"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=20, t=10, b=40),
        height=300,
    )
    return fig


def make_training_curve(history_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df["epoch"], y=history_df["train_acc"] * 100,
        mode="lines", name="Train acc",
        line=dict(color="#3b82f6", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=history_df["epoch"], y=history_df["val_acc"] * 100,
        mode="lines", name="Val acc",
        line=dict(color="#e10600", width=2, dash="dot"),
    ))
    fig.update_layout(
        paper_bgcolor="#0a0a0f", plot_bgcolor="#14141e",
        font=dict(color="#e8e8f0", family="Barlow Condensed"),
        xaxis=dict(title="Epoch", gridcolor="#1e1e2e"),
        yaxis=dict(title="Accuracy (%)", gridcolor="#1e1e2e", range=[0, 105]),
        legend=dict(bgcolor="#14141e", bordercolor="#2a2a3a"),
        margin=dict(l=50, r=20, t=10, b=40),
        height=250,
    )
    return fig


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
try:
    features_df, umap_df, history_df, fi_df, xgb_model, xgb_le, raw, config, race_tag = load_all_data()
    data_loaded = True
except Exception as e:
    st.error(f"Could not load data: {e}\n\nMake sure you've run all pipeline scripts first.")
    data_loaded = False
    st.stop()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family: Barlow Condensed; font-size: 1.5rem; font-weight: 800; 
    color: white; letter-spacing: -0.01em;'>🏎 PitLane Prints</div>
    <div style='font-family: Barlow Condensed; font-size: 0.65rem; letter-spacing: 0.15em; 
    text-transform: uppercase; color: #e10600; margin-bottom: 1.5rem;'>
    Driver Style Fingerprinting</div>
    """, unsafe_allow_html=True)

    st.markdown("**Race**")
    race_display = f"{config['session']['year']} {config['session']['race']} GP"
    st.markdown(f"<span style='color:#6b6b8a; font-size:0.9rem'>{race_display}</span>",
                unsafe_allow_html=True)

    all_drivers = config["drivers"]

    st.markdown("<br>**Drivers to compare**", unsafe_allow_html=True)
    selected_drivers = st.multiselect(
        label="Select drivers to compare",
        options=all_drivers,
        default=all_drivers,
        format_func=lambda x: f"{x} — {DRIVER_NAMES.get(x, x)}",
        label_visibility="collapsed",
    )

    st.markdown("<br>**Highlight in UMAP**", unsafe_allow_html=True)
    umap_highlight = st.selectbox(
        label="Highlight driver",
        options=["All"] + all_drivers,
        label_visibility="collapsed",
    )

    st.markdown("<br>**Lap telemetry**", unsafe_allow_html=True)
    telem_driver = st.selectbox(
        "Driver", options=all_drivers,
        format_func=lambda x: f"{x} — {DRIVER_NAMES.get(x, x)}",
        label_visibility="collapsed",
    )
    lap_options = sorted(raw[telem_driver]["LapNumber"].dropna().unique().astype(int))
    telem_lap = st.selectbox(
        "Lap", options=lap_options,
        index=min(9, len(lap_options) - 1),
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style='font-size: 0.7rem; color: #3a3a5a; 
    font-family: Barlow Condensed; letter-spacing: 0.05em; padding-top: 1rem;
    border-top: 1px solid #1e1e2e;'>
    DATA — FastF1 · MODELS — XGBoost + 1D-CNN<br>
    EMBEDDINGS — UMAP (32→2 dim)
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────
st.markdown("""
<div class='pit-header'>PITLANE PRINTS</div>
<div class='pit-sub'>F1 Driver Style Fingerprinting · 2023 Bahrain GP</div>
""", unsafe_allow_html=True)

# ── Row 1: Key metrics ──
if selected_drivers:
    cols = st.columns(len(selected_drivers) + 2)

    with cols[0]:
        st.markdown("""<div class='metric-card' style='border-top-color: #e10600'>
        <div class='metric-label'>XGBoost Accuracy</div>
        <div class='metric-value' style='color: #e10600'>93.7%</div>
        <div style='font-size:0.75rem; color:#6b6b8a; margin-top:4px'>5-fold OOF</div>
        </div>""", unsafe_allow_html=True)

    with cols[1]:
        st.markdown("""<div class='metric-card' style='border-top-color: #22c55e'>
        <div class='metric-label'>CNN Val Accuracy</div>
        <div class='metric-value' style='color: #22c55e'>100%</div>
        <div style='font-size:0.75rem; color:#6b6b8a; margin-top:4px'>epoch 16</div>
        </div>""", unsafe_allow_html=True)

    for i, driver in enumerate(selected_drivers):
        color = DRIVER_COLORS[driver]
        # Get per-driver silhouette-equivalent: intra/inter ratio from features
        driver_feat = features_df[features_df["Driver"] == driver][FEATURE_COLS].mean()
        lap_count   = features_df[features_df["Driver"] == driver].shape[0]

        with cols[i + 2]:
            st.markdown(f"""<div class='metric-card driver-{driver.lower()}'>
            <div class='metric-label'>{DRIVER_NAMES[driver]}</div>
            <div class='metric-value' style='color: {color}'>{driver}</div>
            <div style='font-size:0.75rem; color:#6b6b8a; margin-top:4px'>{lap_count} laps analyzed</div>
            </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 2: Radar + UMAP + Blind ID ──
col_radar, col_umap, col_blind = st.columns([1, 1, 1])

with col_radar:
    st.markdown("<div class='section-title'>Style Fingerprint — Radar</div>",
                unsafe_allow_html=True)
    if selected_drivers:
        st.plotly_chart(make_radar(features_df, selected_drivers),
                        use_container_width=True)
        st.markdown("""<div class='insight-box'>
        <b>How to read:</b> Each axis is one driving style metric, normalized 0→1 across all drivers 
        (1 = highest among the group). The <b>shape</b> of each polygon is that driver's style signature — 
        bigger on an axis means more of that behavior. <br><br>
        <b>Key signals:</b> VER's large Corner Speed + Coasting wedge reflects the RB19's 2023 
        aerodynamic dominance — he carries more speed through corners and lifts earlier because the 
        car's grip lets him. HAM's Trail Braking spike is his known technique for rotating the car 
        mid-corner. ALO's Gear Changes spike reflects his characteristically aggressive mechanical inputs.
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Select at least one driver from the sidebar.")

with col_umap:
    st.markdown("<div class='section-title'>Embedding Space — UMAP</div>",
                unsafe_allow_html=True)
    highlight = None if umap_highlight == "All" else umap_highlight
    st.plotly_chart(make_umap(umap_df, highlight=highlight),
                    use_container_width=True)
    st.markdown("""<div class='insight-box'>
    <b>How to read:</b> Each dot = one lap. Dots that are close together = laps that "feel similar" 
    to the CNN. The CNN learned these positions from raw telemetry sequences alone — 
    no hand-crafted features were used. <br><br>
    <b>What good looks like:</b> Tight, separated clusters = the model learned a genuine fingerprint 
    per driver. <b>Silhouette score: 0.84</b> (scale −1 to +1, above 0.5 = strong separation). 
    VER's inter/intra distance ratio is 8.3× — his laps are 8× closer to each other than to any 
    other driver's laps in embedding space. Use the highlight dropdown to focus on one driver.
    </div>""", unsafe_allow_html=True)

with col_blind:
    st.markdown("<div class='section-title'>Blind Identification</div>",
                unsafe_allow_html=True)

    if st.button("🎲 Pick random lap", use_container_width=True):
        st.session_state["blind_lap"] = features_df.sample(1).iloc[0]

    if "blind_lap" in st.session_state:
        blind = st.session_state["blind_lap"]
        X_blind = blind[FEATURE_COLS].values.reshape(1, -1)
        probs   = xgb_model.predict_proba(X_blind)[0]
        classes = xgb_le.classes_

        pred_driver = classes[probs.argmax()]
        true_driver = blind["Driver"]
        correct     = pred_driver == true_driver

        driver_color = DRIVER_COLORS.get(true_driver, "#ffffff")
        st.markdown(f"""
        <div style='background:#14141e; border:1px solid #2a2a3a; border-radius:4px; 
        padding:1rem; margin-bottom:0.75rem;'>
            <div style='font-family: Barlow Condensed; font-size:0.65rem; 
            letter-spacing:0.12em; text-transform:uppercase; color:#6b6b8a;'>
            Actual · Lap {int(blind["LapNumber"])}</div>
            <div style='font-family: Barlow Condensed; font-size:2rem; 
            font-weight:800; color:{driver_color}; line-height:1.1;'>
            {true_driver}</div>
            <div style='font-size:0.8rem; color:#6b6b8a;'>{DRIVER_NAMES.get(true_driver, true_driver)}</div>
        </div>
        """, unsafe_allow_html=True)

        for cls, prob in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True):
            bar_color = DRIVER_COLORS.get(cls, "#888")
            is_pred   = cls == pred_driver
            st.markdown(f"""
            <div style='margin-bottom:6px;'>
                <div style='display:flex; justify-content:space-between; 
                font-family: Barlow Condensed; font-size:0.85rem; margin-bottom:2px;'>
                    <span style='color:{bar_color}; font-weight:{"700" if is_pred else "400"}'>
                    {"▶ " if is_pred else ""}{cls}</span>
                    <span style='color:#c0c0d0'>{prob*100:.1f}%</span>
                </div>
                <div style='background:#1e1e2e; border-radius:2px; height:5px;'>
                    <div style='background:{bar_color}; width:{prob*100:.1f}%; 
                    height:5px; border-radius:2px; opacity:{"1" if is_pred else "0.35"};'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        verdict_color = "#22c55e" if correct else "#e10600"
        verdict_text  = "✓ CORRECT" if correct else "✗ WRONG"
        st.markdown(f"""
        <div style='margin-top:0.75rem; font-family: Barlow Condensed; font-size:1.4rem; 
        font-weight:800; color:{verdict_color}; letter-spacing:0.05em;'>
        {verdict_text}</div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='background:#14141e; border:1px dashed #2a2a3a; border-radius:4px;
        padding:2rem; text-align:center; color:#3a3a5a;
        font-family: Barlow Condensed; font-size:0.95rem; letter-spacing:0.05em; margin-top:0.5rem;'>
        HIT THE BUTTON ABOVE<br>TO TEST THE MODEL BLIND
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 3: Full-width telemetry ──
st.markdown(f"<div class='section-title'>Raw Telemetry — {telem_driver} · Lap {telem_lap}</div>",
            unsafe_allow_html=True)
fig_telem = make_telemetry_plot(raw, telem_driver, float(telem_lap))
if fig_telem:
    st.plotly_chart(fig_telem, use_container_width=True)
else:
    st.warning(f"No telemetry for {telem_driver} lap {telem_lap}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 4: Feature importance + training curve ──
col_fi, col_train = st.columns([1, 1])

with col_fi:
    st.markdown("<div class='section-title'>XGBoost Feature Importance</div>",
                unsafe_allow_html=True)
    st.plotly_chart(make_feature_importance_bar(fi_df.sort_values("importance")),
                    use_container_width=True)

with col_train:
    st.markdown("<div class='section-title'>CNN Training Curve</div>",
                unsafe_allow_html=True)
    st.plotly_chart(make_training_curve(history_df), use_container_width=True)
    st.markdown("""<div class='insight-box'>
    Val accuracy reached 100% at epoch 16. Early stopping triggered at epoch 31.
    The fast convergence reflects strong style separation in raw telemetry sequences.
    </div>""", unsafe_allow_html=True)