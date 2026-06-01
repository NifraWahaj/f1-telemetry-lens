"""
streamlit_app.py
----------------
F1 Telemetry Lens — Driver Style Fingerprinting Dashboard

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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="F1 Telemetry Lens",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;600;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #08080e; color: #dddde8; }

[data-testid="stSidebar"] {
    background-color: #0f0f18 !important;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { font-size: 0.875rem; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; max-width: 1400px; }

/* ── Hero header ── */
.hero { margin-bottom: 2.5rem; }
.hero-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 2.75rem;
    letter-spacing: -0.02em;
    line-height: 1;
    color: #fff;
    margin: 0;
}
.hero-sub {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #e10600;
    margin-top: 6px;
}
.hero-desc {
    font-size: 0.9rem;
    color: #6b6b8a;
    margin-top: 10px;
    max-width: 680px;
    line-height: 1.6;
}

/* ── Stat strip ── */
.stat-strip {
    display: flex;
    gap: 1px;
    background: #1e1e2e;
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 2.5rem;
}
.stat-item {
    flex: 1;
    background: #0f0f18;
    padding: 1rem 1.25rem;
    min-width: 0;
}
.stat-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4a4a6a;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.stat-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1;
    white-space: nowrap;
}
.stat-note {
    font-size: 0.7rem;
    color: #4a4a6a;
    margin-top: 3px;
}

/* ── Section ── */
.section-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 0.7rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #4a4a6a;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 1rem;
}

/* ── Chart card ── */
.chart-card {
    background: #0f0f18;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    padding: 1.25rem 1.25rem 0.75rem;
    height: 100%;
}

/* ── Caption ── */
.chart-caption {
    font-size: 0.78rem;
    color: #4a4a6a;
    line-height: 1.55;
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid #1e1e2e;
}
.chart-caption b { color: #8a8aaa; }

/* ── Blind ID panel ── */
.blind-panel {
    background: #0f0f18;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    padding: 1.25rem;
}
.blind-driver-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.75rem;
    font-weight: 800;
    line-height: 1;
}
.blind-driver-full {
    font-size: 0.8rem;
    color: #4a4a6a;
    margin-top: 2px;
    margin-bottom: 1rem;
}
.prob-row { margin-bottom: 7px; }
.prob-label {
    display: flex;
    justify-content: space-between;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.82rem;
    margin-bottom: 3px;
}
.prob-track { background: #1a1a28; border-radius: 2px; height: 4px; }
.verdict {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    margin-top: 1rem;
}

/* ── Divider ── */
.spacer { height: 2rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DRIVER_COLORS = {
    "VER": "#3b82f6", "HAM": "#a78bfa", "ALO": "#f87171",
    "LEC": "#f97316", "SAI": "#facc15", "NOR": "#34d399",
}
DRIVER_NAMES = {
    "VER": "Max Verstappen", "HAM": "Lewis Hamilton", "ALO": "Fernando Alonso",
    "LEC": "Charles Leclerc", "SAI": "Carlos Sainz",  "NOR": "Lando Norris",
}
FEATURE_COLS = [
    "brake_duration_ratio", "throttle_smoothness", "full_throttle_ratio",
    "coasting_ratio", "gear_change_freq", "speed_at_throttle_lift",
    "mean_corner_speed", "speed_variance", "throttle_brake_overlap",
]
FEATURE_LABELS = [
    "Brake Duration", "Throttle Smoothness", "Full Throttle", "Coasting",
    "Gear Changes", "Braking Speed", "Corner Speed", "Speed Variance", "Trail Braking",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PLOT_BG   = "#0f0f18"
PAPER_BG  = "#08080e"
GRID_COL  = "#1a1a28"
TEXT_COL  = "#dddde8"
FONT      = "Inter"

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
@st.cache_data
def load_all_data():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        config = yaml.safe_load(f)

    race_tag      = f"{config['session']['year']}_{config['session']['race'].lower()}"
    features_dir  = os.path.join(ROOT, config["data"]["features_dir"])
    processed_dir = os.path.join(ROOT, config["data"]["processed_dir"])
    models_dir    = os.path.join(ROOT, "outputs", "models")

    features_df = pd.read_csv(os.path.join(features_dir, f"{race_tag}_features.csv"))
    umap_df     = pd.read_csv(os.path.join(features_dir, f"{race_tag}_umap_coords.csv"))
    history_df  = pd.read_csv(os.path.join(features_dir, f"{race_tag}_cnn_history.csv"))
    fi_df       = pd.read_csv(os.path.join(features_dir, f"{race_tag}_feature_importance.csv"))
    xgb_model   = joblib.load(os.path.join(models_dir,   f"{race_tag}_xgb_baseline.pkl"))
    xgb_le      = joblib.load(os.path.join(models_dir,   f"{race_tag}_label_encoder.pkl"))

    raw = {}
    for driver in config["drivers"]:
        path = os.path.join(processed_dir, race_tag, f"{driver}.parquet")
        try:
            raw[driver] = pd.read_parquet(path)
        except FileNotFoundError:
            raw[driver] = pd.DataFrame()

    return features_df, umap_df, history_df, fi_df, xgb_model, xgb_le, raw, config, race_tag


def base_layout(height=None):
    kwargs = dict(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COL, family=FONT, size=11),
    )
    if height:
        kwargs["height"] = height
    return kwargs


# ─────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────
def make_radar(features_df, selected_drivers):
    means      = features_df.groupby("Driver")[FEATURE_COLS].mean()
    means_norm = (means - means.min()) / (means.max() - means.min() + 1e-8)
    fig = go.Figure()
    for driver in selected_drivers:
        if driver not in means_norm.index:
            continue
        vals = means_norm.loc[driver].tolist()
        vals += vals[:1]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=FEATURE_LABELS + [FEATURE_LABELS[0]],
            fill="toself", name=DRIVER_NAMES[driver],
            line=dict(color=DRIVER_COLORS[driver], width=2),
            fillcolor=DRIVER_COLORS[driver], opacity=0.2,
        ))
    fig.update_layout(
        **base_layout(420),
        polar=dict(
            bgcolor=PLOT_BG,
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID_COL,
                            tickfont=dict(color="#3a3a5a", size=8), tickvals=[0.25,0.5,0.75,1.0]),
            angularaxis=dict(gridcolor=GRID_COL, tickfont=dict(color="#8a8aaa", size=10)),
        ),
        legend=dict(bgcolor=PLOT_BG, bordercolor="#1e1e2e", borderwidth=1,
                    font=dict(size=10), orientation="v", x=1.05, y=1)
                            )
    return fig


def make_umap(umap_df, highlight=None):
    fig = go.Figure()
    for driver in sorted(umap_df["Driver"].unique()):
        sub     = umap_df[umap_df["Driver"] == driver]
        dimmed  = highlight is not None and driver != highlight
        color   = DRIVER_COLORS.get(driver, "#888")
        fig.add_trace(go.Scatter(
            x=sub["umap_x"], y=sub["umap_y"], mode="markers",
            name=DRIVER_NAMES.get(driver, driver),
            marker=dict(color=color, size=8 if not dimmed else 6,
                        opacity=0.85 if not dimmed else 0.12,
                        line=dict(width=0.5, color="white")),
            hovertemplate=f"<b>{driver}</b> · Lap %{{customdata}}<extra></extra>",
            customdata=sub["LapNumber"],
        ))
        cx, cy = sub["umap_x"].mean(), sub["umap_y"].mean()
        if not dimmed:
            fig.add_annotation(x=cx, y=cy, text=driver,
                               font=dict(color=color, size=12, family="Barlow Condensed"),
                               showarrow=False, xshift=14)

    return fig


def make_telemetry(raw, driver, lap_num):
    lap_df = raw[driver][raw[driver]["LapNumber"] == lap_num].reset_index(drop=True)
    if lap_df.empty:
        return None
    channels = [("Throttle", "%"), ("Brake", ""), ("Speed", "km/h"), ("nGear", "")]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.3, 0.15, 0.35, 0.2])
    color = DRIVER_COLORS.get(driver, "#888")
    for i, (ch, unit) in enumerate(channels, 1):
        if ch not in lap_df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=np.arange(len(lap_df)), y=lap_df[ch],
            mode="lines", line=dict(color=color, width=1.2),
            name=ch, showlegend=False,
        ), row=i, col=1)
        fig.update_yaxes(title_text=unit, row=i, col=1,
                         gridcolor=GRID_COL, title_font=dict(size=9),
                         tickfont=dict(size=8))
    fig.update_xaxes(gridcolor=GRID_COL)
    fig.update_layout(**base_layout(380),
                      xaxis4=dict(title_text="Sample index", title_font=dict(size=10)))
    return fig


def make_feature_importance(fi_df):
    df = fi_df.sort_values("importance")
    fig = go.Figure(go.Bar(
        x=df["importance"], y=df["feature"],
        orientation="h",
        marker=dict(color=df["importance"],
                    colorscale=[[0, "#1a1a2e"], [0.4, "#3b3b6e"], [1, "#e10600"]],
                    showscale=False),
        text=[f"{v:.3f}" for v in df["importance"]],
        textposition="outside", textfont=dict(size=9, color="#4a4a6a"),
    ))
    fig.update_layout(
        **base_layout(320),
        xaxis=dict(title_text="Importance", title_font=dict(size=10), gridcolor=GRID_COL),
        yaxis=dict(tickfont=dict(size=9)),
    )
    return fig


def make_training_curve(history_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df["epoch"], y=history_df["train_acc"] * 100,
        mode="lines", name="Train",
        line=dict(color="#3b82f6", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=history_df["epoch"], y=history_df["val_acc"] * 100,
        mode="lines", name="Val",
        line=dict(color="#e10600", width=2, dash="dot"),
    ))
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COL, family=FONT, size=11),
        height=320,
        xaxis=dict(title="Epoch", gridcolor=GRID_COL),
        yaxis=dict(title_text="Accuracy (%)", gridcolor=GRID_COL, range=[0, 105]),
        legend=dict(
            bgcolor=PLOT_BG,
            bordercolor="#1e1e2e",
            font=dict(size=10)
        ),
    )
    return fig


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
try:
    features_df, umap_df, history_df, fi_df, xgb_model, xgb_le, raw, config, race_tag = load_all_data()
except Exception as e:
    st.error(f"Could not load data: {e}\n\nRun the pipeline scripts first.")
    st.stop()

all_drivers = config["drivers"]

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Barlow Condensed;font-size:1.3rem;font-weight:800;
    color:#fff;letter-spacing:-0.01em;margin-bottom:2px;'>F1 Telemetry Lens</div>
    <div style='font-family:Barlow Condensed;font-size:0.6rem;letter-spacing:0.16em;
    text-transform:uppercase;color:#e10600;margin-bottom:1.5rem;'>Driver Style Fingerprinting</div>
    """, unsafe_allow_html=True)

    st.caption(f"📍 {config['session']['year']} {config['session']['race']} Grand Prix")

    st.markdown("---")
    st.markdown("**Radar — drivers to compare**")
    selected_drivers = st.multiselect(
        "drivers", options=all_drivers, default=all_drivers,
        format_func=lambda x: f"{x} — {DRIVER_NAMES.get(x, x)}",
        label_visibility="collapsed",
    )

    st.markdown("**UMAP — highlight driver**")
    umap_highlight = st.selectbox(
        "umap", options=["All"] + all_drivers,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Telemetry viewer**")
    telem_driver = st.selectbox(
        "tdriver", options=all_drivers,
        format_func=lambda x: f"{x} — {DRIVER_NAMES.get(x, x)}",
        label_visibility="collapsed",
    )
    has_telem = not raw[telem_driver].empty
    if has_telem:
        lap_options = sorted(raw[telem_driver]["LapNumber"].dropna().unique().astype(int))
        telem_lap   = st.selectbox("tlap", options=lap_options,
                                   index=min(9, len(lap_options)-1),
                                   label_visibility="collapsed")
    else:
        telem_lap = None
        st.caption("Telemetry unavailable in this environment.")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.65rem;color:#2a2a4a;line-height:1.7;'>
    DATA · FastF1<br>
    ML · XGBoost + 1D-CNN<br>
    VIZ · UMAP (32→2 dim)
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown(f"""
<div class='hero'>
  <div class='hero-title'>F1 Telemetry Lens</div>
  <div class='hero-sub'>{config['session']['year']} {config['session']['race']} Grand Prix · Driver Style Analysis</div>
  <div class='hero-desc'>
    Can a machine learn to recognise a driver's identity purely from how they use the throttle,
    brake, and gear? This pipeline learns a style fingerprint for each driver — no lap times,
    no team data, just raw telemetry sequences.
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Stat strip
# ─────────────────────────────────────────────
stat_items = [
    ("#e10600", "XGBoost OOF", "93.7%", "5-fold cross-val"),
    ("#22c55e", "CNN Val Acc", "100%",  "epoch 16 · 40k params"),
    ("#a78bfa", "Silhouette",  "0.84",  "32-dim embeddings"),
    ("#facc15", "Drivers",     str(len(all_drivers)), "2023 Bahrain GP"),
    ("#3b82f6", "Laps",        str(len(features_df)), "after quality filter"),
]
cols = st.columns(len(stat_items))
for col, (color, label, value, note) in zip(cols, stat_items):
    with col:
        st.markdown(f"""
        <div style='background:#0f0f18;border:1px solid #1e1e2e;border-top:2px solid {color};
        border-radius:6px;padding:1rem 1.1rem;'>
          <div class='stat-label'>{label}</div>
          <div class='stat-value' style='color:{color}'>{value}</div>
          <div class='stat-note'>{note}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Row: Radar + UMAP  (equal halves)
# ─────────────────────────────────────────────
col_r, col_u = st.columns(2, gap="large")

with col_r:
    st.markdown("<div class='section-header'>Style Fingerprint — Radar</div>", unsafe_allow_html=True)
    if selected_drivers:
        st.plotly_chart(make_radar(features_df, selected_drivers), use_container_width=True)
        st.markdown("""<div class='chart-caption'>
        Each axis is one driving style metric, normalized 0→1 across all drivers.
        The <b>shape</b> of each polygon is that driver's signature.
        VER's corner speed + coasting wedge reflects the RB19's 2023 aerodynamic advantage.
        HAM's trail braking spike is his known car-rotation technique.
        ALO's gear change spike reflects characteristically aggressive mechanical inputs.
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Select drivers from the sidebar.")

with col_u:
    st.markdown("<div class='section-header'>Embedding Space — UMAP</div>", unsafe_allow_html=True)
    highlight = None if umap_highlight == "All" else umap_highlight
    st.plotly_chart(make_umap(umap_df, highlight), use_container_width=True)
    st.markdown("""<div class='chart-caption'>
    Each dot = one lap. The 1D-CNN learned these positions from raw telemetry sequences alone —
    no hand-crafted features. Tight, separated clusters mean the model learned a genuine
    fingerprint per driver. <b>Silhouette: 0.84</b> (>0.5 = strong). VER's inter/intra
    distance ratio is 8.3× — his laps are 8× closer to each other than to any other driver.
    </div>""", unsafe_allow_html=True)

st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Row: Blind ID (full width, styled panel)
# ─────────────────────────────────────────────
st.markdown("<div class='section-header'>Blind Identification Challenge</div>", unsafe_allow_html=True)
st.markdown("""<p style='font-size:0.85rem;color:#4a4a6a;margin-bottom:1rem;margin-top:-0.5rem;'>
Pick a random lap from the dataset. The XGBoost model predicts which driver it belongs to
using only the 9 engineered features — no lap number, no context.
</p>""", unsafe_allow_html=True)

bcol_btn, bcol_result, bcol_bars = st.columns([1, 1, 2], gap="large")

with bcol_btn:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    if st.button("🎲  Pick random lap", use_container_width=True, type="primary"):
        st.session_state["blind_lap"] = features_df.sample(1).iloc[0]
    st.markdown("""<div style='font-size:0.75rem;color:#3a3a5a;margin-top:0.75rem;line-height:1.6;'>
    93.7% of the time the model is correct. When it fails it almost always confuses
    ALO ↔ HAM — the two drivers whose style profiles overlap most.
    </div>""", unsafe_allow_html=True)

if "blind_lap" in st.session_state:
    blind       = st.session_state["blind_lap"]
    X_blind     = blind[FEATURE_COLS].values.reshape(1, -1)
    probs       = xgb_model.predict_proba(X_blind)[0]
    classes     = xgb_le.classes_
    pred_driver = classes[probs.argmax()]
    true_driver = blind["Driver"]
    correct     = pred_driver == true_driver
    dcolor      = DRIVER_COLORS.get(true_driver, "#fff")
    verdict_color = "#22c55e" if correct else "#e10600"
    verdict_text  = "✓  CORRECT" if correct else "✗  WRONG"

    with bcol_result:
        st.markdown(f"""
        <div style='background:#0f0f18;border:1px solid #1e1e2e;border-left:3px solid {dcolor};
        border-radius:6px;padding:1.25rem;'>
          <div style='font-family:Barlow Condensed;font-size:0.6rem;letter-spacing:0.14em;
          text-transform:uppercase;color:#3a3a5a;'>Actual · Lap {int(blind["LapNumber"])}</div>
          <div style='font-family:Barlow Condensed;font-size:3rem;font-weight:800;
          color:{dcolor};line-height:1;margin:4px 0 2px;'>{true_driver}</div>
          <div style='font-size:0.8rem;color:#4a4a6a;'>{DRIVER_NAMES.get(true_driver, true_driver)}</div>
          <div style='font-family:Barlow Condensed;font-size:1.1rem;font-weight:700;
          color:{verdict_color};letter-spacing:0.06em;margin-top:1rem;'>{verdict_text}</div>
        </div>
        """, unsafe_allow_html=True)

    with bcol_bars:
        bars_html = ""
        for cls, prob in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True):
            bc      = DRIVER_COLORS.get(cls, "#888")
            is_pred = cls == pred_driver
            weight  = "700" if is_pred else "400"
            prefix  = "▶ " if is_pred else "&nbsp;&nbsp;&nbsp;"
            opacity = "1" if is_pred else "0.3"
            bars_html += f"""
            <div style='margin-bottom:9px;'>
              <div style='display:flex;justify-content:space-between;
              font-family:Barlow Condensed;font-size:0.88rem;margin-bottom:3px;'>
                <span style='color:{bc};font-weight:{weight};'>{prefix}{cls} — {DRIVER_NAMES.get(cls,cls)}</span>
                <span style='color:#8a8aaa;'>{prob*100:.1f}%</span>
              </div>
              <div style='background:#1a1a28;border-radius:3px;height:5px;'>
                <div style='background:{bc};width:{prob*100:.1f}%;height:5px;
                border-radius:3px;opacity:{opacity};transition:width 0.3s;'></div>
              </div>
            </div>"""

        st.markdown("<div style='margin-bottom:10px;font-weight:600;'>Model confidence</div>", unsafe_allow_html=True)

        st.components.v1.html(
            f"""
            <div style='background:#0f0f18;border:1px solid #1e1e2e;border-radius:6px;
            padding:1.25rem;font-family:Inter;color:#dddde8;'>
                {bars_html}
            </div>
            """,
            height=250,
            scrolling=True
        )


else:
    with bcol_result:
        st.markdown("""
        <div style='background:#0f0f18;border:1px dashed #1e1e2e;border-radius:6px;
        padding:2rem;text-align:center;color:#2a2a4a;
        font-family:Barlow Condensed;font-size:0.9rem;letter-spacing:0.06em;'>
        AWAITING SELECTION
        </div>""", unsafe_allow_html=True)

st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Row: Telemetry (full width)
# ─────────────────────────────────────────────
telem_title = f"Raw Telemetry — {telem_driver} · Lap {telem_lap}" if telem_lap else f"Raw Telemetry — {telem_driver}"
st.markdown(f"<div class='section-header'>{telem_title}</div>", unsafe_allow_html=True)

if telem_lap is not None:
    fig_t = make_telemetry(raw, telem_driver, float(telem_lap))
    if fig_t:
        st.plotly_chart(fig_t, use_container_width=True)
    st.markdown("""<div class='chart-caption'>
    <b>Throttle</b> — sharp square drops = aggressive lift-off; gradual ramps = smooth style. &nbsp;
    <b>Brake</b> — binary (0/1); each spike is a braking zone. &nbsp;
    <b>Speed</b> — valleys are corners; a higher valley = more corner speed carried. &nbsp;
    <b>nGear</b> — frequent shifts = aggressive mechanical input.
    Try selecting different drivers on the same lap number — the style differences are visible to the naked eye.
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div style='background:#0f0f18;border:1px dashed #1e1e2e;border-radius:6px;
    padding:1.5rem;font-size:0.85rem;color:#3a3a5a;'>
    Raw telemetry is not available in this deployed environment (files excluded due to size ~200MB).
    Clone the repo and run <code>python src/data/fetch_telemetry.py</code> to enable this panel locally.
    </div>""", unsafe_allow_html=True)

st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Row: Feature importance + Training curve
# ─────────────────────────────────────────────
col_fi, col_tc = st.columns(2, gap="large")

with col_fi:
    st.markdown("<div class='section-header'>XGBoost Feature Importance</div>", unsafe_allow_html=True)
    st.plotly_chart(make_feature_importance(fi_df), use_container_width=True)
    st.markdown("""<div class='chart-caption'>
    Longer bar = the model relied on this feature more across all CV folds.
    <b>mean_corner_speed</b> dominates because VER's car created a ~9 km/h corner speed gap
    no other feature can explain. <b>gear_change_freq</b> and <b>coasting_ratio</b> are the
    most driver-specific — reflecting genuine style choices independent of car performance.
    </div>""", unsafe_allow_html=True)

with col_tc:
    st.markdown("<div class='section-header'>CNN Training Curve</div>", unsafe_allow_html=True)
    st.plotly_chart(make_training_curve(history_df), use_container_width=True)
    st.markdown("""<div class='chart-caption'>
    Blue = train accuracy, red dashed = validation accuracy on unseen laps. A large gap would
    indicate overfitting — the curves track closely here. Val hit 100% at epoch 16; early
    stopping triggered at epoch 31. The XGBoost 5-fold OOF score of <b>93.7%</b> is the more
    conservative generalisation estimate.
    </div>""", unsafe_allow_html=True)