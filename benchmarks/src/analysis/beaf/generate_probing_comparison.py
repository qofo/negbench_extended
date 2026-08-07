"""
Generate Probing Model Comparison Horizontal Bar Chart.

Replicates the visual style of probing model accuracy comparisons across:
- Random Chance
- Linear Probes (Default & Tuned)
- MLP Probes (Hidden=8, 32, 64)
- Low-Rank Bilinear Probes (Rank=4, 16, 32)
"""

import os
import json
import argparse
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def render_probing_comparison_bar_chart(
    model_data: Dict[str, float],
    output_dir: str,
    filename: str = "beaf_vision_probing_comparison.png"
):
    """Render horizontal bar chart for probing accuracies matching publication quality."""
    os.makedirs(output_dir, exist_ok=True)

    models = list(model_data.keys())
    accuracies = list(model_data.values())

    # Curated color palette matching reference design
    colors = [
        "#8b979f",  # Random Chance (Gray)
        "#ed8936",  # Linear Default (Light Orange)
        "#dd5816",  # Linear Tuned (Burnt Orange)
        "#44d07b",  # MLP Hidden=8 (Light Emerald)
        "#34b76a",  # MLP Hidden=32 (Medium Emerald)
        "#29824c",  # MLP Hidden=64 (Dark Emerald)
        "#4ba2e3",  # Bilinear Rank=4 (Sky Blue)
        "#3d85b8",  # Bilinear Rank=16 (Steel Blue)
        "#315b7d"   # Bilinear Rank=32 (Dark Navy)
    ]

    # Adjust color palette length if model count varies
    if len(models) != len(colors):
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5.5))

    y_positions = np.arange(len(models))
    bars = ax.barh(y_positions, accuracies, height=0.68, color=colors, edgecolor="#1a1a1a", linewidth=0.7)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(models, fontsize=11, color="#1e293b", fontweight="normal")
    ax.invert_yaxis()  # Top-down order matching reference image

    ax.set_xlim(45, 100)
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.set_xticklabels(["50", "60", "70", "80", "90", "100"], fontsize=10, color="#1e293b")
    ax.set_xlabel("Accuracy (%)", fontsize=11, fontweight="bold", color="#0f172a", labelpad=8)

    # Add exact % labels to the right of each bar
    for bar, val in zip(bars, accuracies):
        x_pos = val + 0.8
        y_pos = bar.get_y() + bar.get_height() / 2.0
        ax.text(x_pos, y_pos, f"{val:.2f}%", va="center", ha="left", fontsize=10, fontweight="bold", color="#1e3a5f")

    # Styling grid lines & border box
    ax.xaxis.grid(True, linestyle="--", color="#d1d5db", alpha=0.7)
    ax.yaxis.grid(True, linestyle="-", color="#e5e7eb", alpha=0.5)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#cccccc")
        spine.set_linewidth(0.8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved probing comparison plot: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate Probing Comparison Bar Chart.")
    parser.add_argument("--output_dir", type=str, default="logs/evaluation/beaf_counterfactual_v5_2", help="Output directory")
    args = parser.parse_args()

    # Exact metrics matching reference design
    default_probing_results = {
        "Random Chance": 50.00,
        "Linear Probe (Default C=1.0)": 56.95,
        "Linear Probe (Tuned C=100)": 70.70,
        "MLP (Hidden=8)": 84.03,
        "MLP (Hidden=32)": 90.38,
        "MLP (Hidden=64)": 90.58,
        "Low-Rank Bilinear (Rank=4)": 92.38,
        "Low-Rank Bilinear (Rank=16)": 95.08,
        "Low-Rank Bilinear (Rank=32)": 95.08,
    }

    render_probing_comparison_bar_chart(default_probing_results, args.output_dir)


if __name__ == "__main__":
    main()
