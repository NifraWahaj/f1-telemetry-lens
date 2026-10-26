"""
Regenerate the README result figures from the saved 12-race pipeline artefacts.

Reads only the CSVs written by baseline.py, cnn_encoder.py and
contrastive_encoder.py, so it runs in seconds without retraining anything.

Usage (from project root):
    python src/viz/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

FEATURES_DIR = Path("data/features")
OUT_DIR = Path("outputs/figures")

DRIVERS = ["VER", "NOR", "ALO", "SAI", "LEC", "HAM"]
DRIVER_COLORS = {
    "VER": "#3b82f6", "HAM": "#a78bfa", "ALO": "#f87171",
    "LEC": "#f97316", "SAI": "#facc15", "NOR": "#34d399",
}
BG, PANEL, TEXT, MUTED, GRID, ACCENT = (
    "#08080e", "#0f0f18", "#dddde8", "#8a8aa0", "#22222e", "#e10600",
)

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlecolor": TEXT, "axes.titlepad": 12,
})


def _save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {path}")


def confusion_matrices():
    oof = pd.read_csv(FEATURES_DIR / "all_races_oof_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, col, title in [
        (axes[0], "oof_pred_intrack", "Same-circuit (stratified 5-fold)"),
        (axes[1], "oof_pred_crosscircuit", "Cross-circuit (GroupKFold by race)"),
    ]:
        cm = confusion_matrix(oof["Driver"], oof[col], labels=DRIVERS, normalize="true")
        acc = (oof["Driver"] == oof[col]).mean()
        ax.imshow(cm, cmap="Reds", vmin=0, vmax=1)
        for i in range(len(DRIVERS)):
            for j in range(len(DRIVERS)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > 0.5 else "#1a1a24",
                        fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(len(DRIVERS)), DRIVERS)
        ax.set_yticks(range(len(DRIVERS)), DRIVERS)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(DRIVER_COLORS[lbl.get_text()])
            lbl.set_fontweight("bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{title}  ·  {acc:.1%}")
    fig.suptitle("XGBoost out-of-fold recall: 3,582 laps, 6 drivers, 12 races",
                 color=TEXT, fontsize=14, fontweight="bold", y=1.02)
    _save(fig, "results_confusion_matrix.png")


def umap_embedding():
    coords = pd.read_csv(FEATURES_DIR / "all_races_umap_coords.csv")
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for drv in DRIVERS:
        d = coords[coords["Driver"] == drv]
        ax.scatter(d["umap_x"], d["umap_y"], s=7, alpha=0.65, color=DRIVER_COLORS[drv],
                   linewidths=0, label=f"{drv}  ({len(d)} laps)")
        cx, cy = float(np.median(d["umap_x"])), float(np.median(d["umap_y"]))
        ax.text(cx, cy, drv, ha="center", va="center", fontsize=12, fontweight="bold",
                color="white", bbox=dict(boxstyle="round,pad=0.25", fc=DRIVER_COLORS[drv],
                                         ec="none", alpha=0.9))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("1D-CNN lap embeddings (32-dim → UMAP 2D), all 12 races")
    leg = ax.legend(loc="best", frameon=True, markerscale=3, fontsize=9)
    leg.get_frame().set_facecolor(BG)
    leg.get_frame().set_edgecolor(GRID)
    _save(fig, "results_umap_embeddings.png")


def feature_importance():
    fi = pd.read_csv(FEATURES_DIR / "all_races_feature_importance.csv")
    fi = fi.sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [ACCENT if i >= len(fi) - 3 else "#5a5a70" for i in range(len(fi))]
    ax.barh(fi["feature"], fi["importance"], color=colors, height=0.65)
    for y, v in enumerate(fi["importance"]):
        ax.text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=9, color=MUTED)
    ax.set_xlabel("XGBoost gain importance")
    ax.set_title("Which style metrics identify a driver?")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "results_feature_importance.png")


def model_comparison():
    ce = pd.read_csv(FEATURES_DIR / "all_races_cnn_history.csv")
    sc = pd.read_csv(FEATURES_DIR / "all_races_contrastive_history.csv")
    oof = pd.read_csv(FEATURES_DIR / "all_races_oof_predictions.csv")
    xgb_cross = (oof["Driver"] == oof["oof_pred_crosscircuit"]).mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.plot(ce["epoch"], ce["val_acc"], color="#3b82f6", lw=2, label="CNN, cross-entropy (val acc)")
    ax.plot(sc["epoch"], sc["val_knn_acc"], color="#34d399", lw=2, label="CNN, SupCon (k-NN val acc)")
    ax.axhline(xgb_cross, color="#f97316", ls="--", lw=1.5, label=f"XGBoost cross-circuit ({xgb_cross:.1%})")
    ax.axhline(1 / 6, color=MUTED, ls=":", lw=1.2, label="Random (16.7%)")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy on held-out circuits")
    ax.set_title("Training curves, validated on unseen circuits")
    ax.grid(True, color=GRID, lw=0.6)
    leg = ax.legend(loc="lower right", fontsize=9)
    leg.get_frame().set_facecolor(BG)
    leg.get_frame().set_edgecolor(GRID)

    ax = axes[1]
    labels = ["XGBoost\n9 features", "CNN\ncross-entropy", "CNN\nSupCon"]
    vals = [xgb_cross, ce["val_acc"].max(), sc["val_knn_acc"].max()]
    bars = ax.bar(labels, vals, color=["#f97316", "#3b82f6", "#34d399"], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1%}", ha="center",
                fontweight="bold", color=TEXT)
    ax.axhline(1 / 6, color=MUTED, ls=":", lw=1.2)
    ax.set_ylim(0, 1.1)
    ax.set_title("Cross-circuit accuracy (best checkpoint)")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "results_model_comparison.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating README figures...")
    confusion_matrices()
    umap_embedding()
    feature_importance()
    model_comparison()


if __name__ == "__main__":
    main()
