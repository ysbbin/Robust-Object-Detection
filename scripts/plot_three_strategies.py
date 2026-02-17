"""
Visualization: 3-strategy comparison (Baseline vs Augmented vs Restored).

Generates 4 figures comparing all three robustness strategies.

Usage:
    python -m scripts.plot_three_strategies
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
BASELINE_RESULTS = Path("experiments/eval_results.json")
RESTORED_RESULTS = Path("experiments/eval_restored_results.json")
FIG_DIR = Path("experiments/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["Test_Clean", "Test_Noise", "Test_Blur", "Test_LowRes"]
SHORT = {"Test_Clean": "Clean", "Test_Noise": "Noise",
         "Test_Blur": "Blur", "Test_LowRes": "LowRes"}
SHORT_LIST = [SHORT[v] for v in VARIANTS]
CORRUPTION_VARIANTS = VARIANTS[1:]  # Noise, Blur, LowRes

MODELS = ["FasterRCNN", "RT-DETR-L", "YOLOv8m"]
MODEL_DISPLAY = {"FasterRCNN": "FRCNN", "RT-DETR-L": "RT-DETR", "YOLOv8m": "YOLOv8m"}
AUG_KEYS = {"FasterRCNN": "FasterRCNN_aug", "RT-DETR-L": "RT-DETR-L_aug", "YOLOv8m": "YOLOv8m_aug"}

STRATEGIES = ["Baseline", "Augmented", "Restored"]
STRATEGY_COLORS = {
    "Baseline":  "#94a3b8",   # slate gray
    "Augmented": "#3b82f6",   # blue
    "Restored":  "#f97316",   # orange
}
STRATEGY_MARKERS = {"Baseline": "s", "Augmented": "D", "Restored": "o"}

CLASS_NAMES = ["pedestrian", "car", "van", "truck", "bus", "motor"]
DPI = 150


def load_data():
    with open(BASELINE_RESULTS, "r", encoding="utf-8") as f:
        base = json.load(f)
    with open(RESTORED_RESULTS, "r", encoding="utf-8") as f:
        restored = json.load(f)
    return base, restored


def get_map50(base_data, restored_data, model, strategy, variant):
    if strategy == "Baseline":
        return base_data[model][variant]["mAP50"]
    elif strategy == "Augmented":
        return base_data[AUG_KEYS[model]][variant]["mAP50"]
    else:  # Restored
        return restored_data[model][variant]["mAP50"]


# ──────────────────────────────────────────────
# 1. 3-Strategy mAP@50 Comparison (per model)
# ──────────────────────────────────────────────
def plot_strategy_comparison(base_data, restored_data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

    n_strategies = len(STRATEGIES)
    n_variants = len(VARIANTS)
    x = np.arange(n_variants)
    width = 0.25

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        for j, strat in enumerate(STRATEGIES):
            vals = [get_map50(base_data, restored_data, model, strat, v) for v in VARIANTS]
            offset = (j - 1) * width
            bars = ax.bar(x + offset, vals, width * 0.88,
                          label=strat, color=STRATEGY_COLORS[strat],
                          edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                        fontweight="bold")

        ax.set_title(MODEL_DISPLAY[model], fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT_LIST, fontsize=10)
        ax.set_ylim(0, 0.80)
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            ax.set_ylabel("mAP@50", fontsize=12)
        if idx == 1:
            ax.set_xlabel("Test Condition", fontsize=12)
            ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

    fig.suptitle("3-Strategy Comparison: Baseline vs Augmented vs Restored (mAP@50)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "three_strategy_comparison.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# 2. Strategy Effectiveness (Improvement over Baseline)
# ──────────────────────────────────────────────
def plot_strategy_improvement(base_data, restored_data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)

    corruption_short = [SHORT[v] for v in CORRUPTION_VARIANTS]
    x = np.arange(len(CORRUPTION_VARIANTS))
    width = 0.3

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        for j, strat in enumerate(["Augmented", "Restored"]):
            diffs = []
            for v in CORRUPTION_VARIANTS:
                baseline = get_map50(base_data, restored_data, model, "Baseline", v)
                current = get_map50(base_data, restored_data, model, strat, v)
                diffs.append(current - baseline)
            offset = (j - 0.5) * width
            bars = ax.bar(x + offset, diffs, width * 0.85,
                          label=strat, color=STRATEGY_COLORS[strat],
                          edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, diffs):
                y_pos = bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.005
                va = "bottom" if val >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                        f"{val:+.3f}", ha="center", va=va, fontsize=8.5,
                        fontweight="bold")

        ax.set_title(MODEL_DISPLAY[model], fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(corruption_short, fontsize=11)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            ax.set_ylabel("mAP@50 Change from Baseline", fontsize=11)
        if idx == 1:
            ax.set_xlabel("Corruption Type", fontsize=12)
            ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    fig.suptitle("Strategy Effectiveness: Improvement over Baseline (mAP@50)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "strategy_improvement.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# 3. Radar Chart with 3 strategies
# ──────────────────────────────────────────────
def plot_radar_three(base_data, restored_data):
    categories = SHORT_LIST
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5),
                             subplot_kw=dict(polar=True))

    for idx, model in enumerate(MODELS):
        ax = axes[idx]

        for strat in STRATEGIES:
            vals = [get_map50(base_data, restored_data, model, strat, v) for v in VARIANTS]
            vals += vals[:1]
            ax.plot(angles, vals, linestyle="-", linewidth=2, color=STRATEGY_COLORS[strat],
                    label=strat, markersize=5, marker=STRATEGY_MARKERS[strat])
            ax.fill(angles, vals, alpha=0.1, color=STRATEGY_COLORS[strat])

        ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
        ax.set_ylim(0, 0.75)
        ax.set_yticks([0.2, 0.4, 0.6])
        ax.set_yticklabels(["0.2", "0.4", "0.6"], fontsize=7, color="gray")
        ax.set_title(MODEL_DISPLAY[model], fontsize=13, fontweight="bold", pad=20)
        if idx == 1:
            ax.legend(loc="lower center", fontsize=9, bbox_to_anchor=(0.5, -0.25), ncol=3)

    fig.suptitle("Robustness Profile: 3 Strategies (mAP@50)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "three_strategy_radar.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# 4. Best Strategy Summary Heatmap
# ──────────────────────────────────────────────
def plot_best_strategy_heatmap(base_data, restored_data):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Build matrix: rows = models, cols = corruption conditions
    matrix = np.zeros((len(MODELS), len(CORRUPTION_VARIANTS)))
    annot = []

    for i, model in enumerate(MODELS):
        row_annot = []
        for j, v in enumerate(CORRUPTION_VARIANTS):
            vals = {}
            for strat in STRATEGIES:
                vals[strat] = get_map50(base_data, restored_data, model, strat, v)
            best_strat = max(vals, key=vals.get)
            best_val = vals[best_strat]
            baseline_val = vals["Baseline"]
            improvement = best_val - baseline_val
            matrix[i, j] = improvement
            row_annot.append(f"{best_strat[0]}\n+{improvement:.3f}")
        annot.append(row_annot)

    annot = np.array(annot)
    model_labels = [MODEL_DISPLAY[m] for m in MODELS]
    corruption_labels = [SHORT[v] for v in CORRUPTION_VARIANTS]

    sns.heatmap(matrix, annot=annot, fmt="", cmap="YlGn",
                xticklabels=corruption_labels, yticklabels=model_labels,
                vmin=0, vmax=0.25, linewidths=1, linecolor="white",
                cbar_kws={"label": "mAP@50 Improvement over Baseline"}, ax=ax)

    ax.set_title("Best Strategy per Condition\n(A=Augmented, R=Restored)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Corruption Type", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)

    fig.tight_layout()
    out = FIG_DIR / "best_strategy_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    base_data, restored_data = load_data()
    print("Generating 3-strategy comparison figures ...\n")

    plot_strategy_comparison(base_data, restored_data)
    plot_strategy_improvement(base_data, restored_data)
    plot_radar_three(base_data, restored_data)
    plot_best_strategy_heatmap(base_data, restored_data)

    print(f"\nAll figures saved to: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
