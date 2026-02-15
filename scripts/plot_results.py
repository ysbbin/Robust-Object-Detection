"""
Visualization of evaluation results: 6 models x 4 test sets.

Generates 5 figures:
  1. mAP@50 grouped bar chart
  2. Degradation from Clean (%) bar chart
  3. Augmentation improvement bar chart
  4. Per-class AP@50 heatmap (Blur condition)
  5. Robustness radar chart

Usage:
    python -m scripts.plot_results
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
RESULTS_PATH = Path("experiments/eval_results.json")
FIG_DIR = Path("experiments/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["Test_Clean", "Test_Noise", "Test_Blur", "Test_LowRes"]
SHORT = {"Test_Clean": "Clean", "Test_Noise": "Noise",
         "Test_Blur": "Blur", "Test_LowRes": "LowRes"}
SHORT_LIST = [SHORT[v] for v in VARIANTS]

MODEL_ORDER = [
    "FasterRCNN", "FasterRCNN_aug",
    "RT-DETR-L", "RT-DETR-L_aug",
    "YOLOv8m", "YOLOv8m_aug",
]
DISPLAY_NAMES = {
    "FasterRCNN": "FRCNN",
    "FasterRCNN_aug": "FRCNN_aug",
    "RT-DETR-L": "RT-DETR",
    "RT-DETR-L_aug": "RT-DETR_aug",
    "YOLOv8m": "YOLOv8m",
    "YOLOv8m_aug": "YOLOv8m_aug",
}
BASELINE_PAIRS = [
    ("FasterRCNN", "FasterRCNN_aug"),
    ("RT-DETR-L", "RT-DETR-L_aug"),
    ("YOLOv8m", "YOLOv8m_aug"),
]
PAIR_SHORT = ["FRCNN", "RT-DETR", "YOLOv8m"]
CLASS_NAMES = ["pedestrian", "car", "van", "truck", "bus", "motor"]

# Colors: light=baseline, dark=augmented
COLORS = {
    "FasterRCNN":     "#93c5fd",  # light blue
    "FasterRCNN_aug": "#2563eb",  # dark blue
    "RT-DETR-L":      "#86efac",  # light green
    "RT-DETR-L_aug":  "#16a34a",  # dark green
    "YOLOv8m":        "#fca5a5",  # light red
    "YOLOv8m_aug":    "#dc2626",  # dark red
}

DPI = 150


def load_results() -> dict:
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# 1. mAP@50 Grouped Bar Chart
# ──────────────────────────────────────────────
def plot_map50_comparison(data: dict):
    fig, ax = plt.subplots(figsize=(12, 6))

    n_models = len(MODEL_ORDER)
    n_variants = len(VARIANTS)
    x = np.arange(n_variants)
    width = 0.12
    offsets = np.arange(n_models) - (n_models - 1) / 2

    for i, model in enumerate(MODEL_ORDER):
        vals = [data[model][v]["mAP50"] for v in VARIANTS]
        bars = ax.bar(x + offsets[i] * width, vals, width * 0.9,
                      label=DISPLAY_NAMES[model], color=COLORS[model],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6.5,
                    fontweight="bold")

    ax.set_xlabel("Test Condition", fontsize=12)
    ax.set_ylabel("mAP@50", fontsize=12)
    ax.set_title("mAP@50 Comparison: Baseline vs Augmented", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LIST, fontsize=11)
    ax.set_ylim(0, 0.82)
    ax.legend(ncol=3, fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "map50_comparison.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# 2. Degradation from Clean (%)
# ──────────────────────────────────────────────
def plot_degradation(data: dict):
    fig, ax = plt.subplots(figsize=(11, 6))

    deg_variants = VARIANTS[1:]  # Noise, Blur, LowRes
    deg_short = [SHORT[v] for v in deg_variants]

    n_models = len(MODEL_ORDER)
    n_vars = len(deg_variants)
    x = np.arange(n_vars)
    width = 0.12
    offsets = np.arange(n_models) - (n_models - 1) / 2

    for i, model in enumerate(MODEL_ORDER):
        clean = data[model]["Test_Clean"]["mAP50"]
        vals = []
        for v in deg_variants:
            cur = data[model][v]["mAP50"]
            pct = (cur - clean) / clean * 100 if clean > 0 else 0.0
            vals.append(pct)
        bars = ax.bar(x + offsets[i] * width, vals, width * 0.9,
                      label=DISPLAY_NAMES[model], color=COLORS[model],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            y_pos = bar.get_height() - 1.5 if val < -5 else bar.get_height() + 0.5
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:.1f}%", ha="center", va="top", fontsize=6,
                    fontweight="bold")

    ax.set_xlabel("Corruption Type", fontsize=12)
    ax.set_ylabel("Degradation from Clean (%)", fontsize=12)
    ax.set_title("Performance Degradation from Clean (mAP@50)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(deg_short, fontsize=11)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.legend(ncol=3, fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "degradation_comparison.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# 3. Augmentation Improvement
# ──────────────────────────────────────────────
def plot_improvement(data: dict):
    fig, ax = plt.subplots(figsize=(10, 6))

    n_pairs = len(BASELINE_PAIRS)
    n_variants = len(VARIANTS)
    x = np.arange(n_pairs)
    width = 0.18

    variant_colors = ["#6366f1", "#f59e0b", "#ef4444", "#10b981"]  # purple, amber, red, green
    offsets = np.arange(n_variants) - (n_variants - 1) / 2

    for j, v in enumerate(VARIANTS):
        vals = []
        for base, aug in BASELINE_PAIRS:
            diff = data[aug][v]["mAP50"] - data[base][v]["mAP50"]
            vals.append(diff)
        bars = ax.bar(x + offsets[j] * width, vals, width * 0.85,
                      label=SHORT[v], color=variant_colors[j],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            y_pos = bar.get_height() + 0.002 if val >= 0 else bar.get_height() - 0.006
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:+.4f}", ha="center", va=va, fontsize=8,
                    fontweight="bold")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("mAP@50 Improvement (Aug - Base)", fontsize=12)
    ax.set_title("Augmentation Effect per Model", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(PAIR_SHORT, fontsize=11)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "augmentation_improvement.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# 4. Per-class AP@50 Heatmap (Blur)
# ──────────────────────────────────────────────
def plot_class_heatmap(data: dict):
    fig, ax = plt.subplots(figsize=(10, 5))

    matrix = []
    row_labels = []
    for model in MODEL_ORDER:
        row = []
        per_class = data[model]["Test_Blur"]["per_class_ap50"]
        for cls in CLASS_NAMES:
            row.append(per_class.get(cls, 0.0))
        matrix.append(row)
        row_labels.append(DISPLAY_NAMES[model])

    matrix = np.array(matrix)

    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=CLASS_NAMES, yticklabels=row_labels,
                vmin=0, vmax=0.9, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "AP@50"}, ax=ax)

    ax.set_title("Per-Class AP@50 under Blur Condition", fontsize=14, fontweight="bold")
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)

    fig.tight_layout()
    out = FIG_DIR / "class_ap50_blur_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# 5. Radar Chart
# ──────────────────────────────────────────────
def plot_radar(data: dict):
    categories = SHORT_LIST
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             subplot_kw=dict(polar=True))

    for idx, (base, aug) in enumerate(BASELINE_PAIRS):
        ax = axes[idx]

        base_vals = [data[base][v]["mAP50"] for v in VARIANTS]
        aug_vals = [data[aug][v]["mAP50"] for v in VARIANTS]
        base_vals += base_vals[:1]
        aug_vals += aug_vals[:1]

        ax.plot(angles, base_vals, "o-", linewidth=2, color=COLORS[base],
                label="Baseline", markersize=6)
        ax.fill(angles, base_vals, alpha=0.15, color=COLORS[base])

        ax.plot(angles, aug_vals, "o-", linewidth=2, color=COLORS[aug],
                label="Augmented", markersize=6)
        ax.fill(angles, aug_vals, alpha=0.15, color=COLORS[aug])

        ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
        ax.set_ylim(0, 0.75)
        ax.set_yticks([0.2, 0.4, 0.6])
        ax.set_yticklabels(["0.2", "0.4", "0.6"], fontsize=7, color="gray")
        ax.set_title(PAIR_SHORT[idx], fontsize=13, fontweight="bold", pad=20)
        ax.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.25, -0.05))

    fig.suptitle("Robustness Profile: Baseline vs Augmented (mAP@50)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "robustness_radar.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    data = load_results()
    print("Generating figures ...\n")

    plot_map50_comparison(data)
    plot_degradation(data)
    plot_improvement(data)
    plot_class_heatmap(data)
    plot_radar(data)

    print(f"\nAll figures saved to: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
