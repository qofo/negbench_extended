"""
Script to generate paired correlation scatter plots for BEAF v5.2:
1. Image-Fixed Correlation: cos_sim(image, pos_text) vs cos_sim(image, neg_text)
   - Grouped by Object IN image (Orig) vs Object NOT in image (CF)
2. Text-Fixed Correlation: cos_sim(orig_img, text) vs cos_sim(cf_img, text)
   - Grouped by Positive Caption vs Negative Caption
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_plots(csv_path: str, output_dir: str):
    df = pd.read_csv(csv_path)

    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Standard styling configuration
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]

    # =========================================================================
    # GRAPH 1: Image Fixed (이미지 고정)
    # X: cos_sim(image, Positive Text)
    # Y: cos_sim(image, Negative Text)
    # Legend: Object IN image (Orig) vs Object NOT in image (CF)
    # =========================================================================
    x1_in = df["A_sim_orig_pos"].values
    y1_in = df["B_sim_orig_neg"].values

    x1_out = df["C_sim_cf_pos"].values
    y1_out = df["D_sim_cf_neg"].values

    x1_all = np.concatenate([x1_in, x1_out])
    y1_all = np.concatenate([y1_in, y1_out])

    r1_p, p1_p = stats.pearsonr(x1_all, y1_all)
    r1_s, p1_s = stats.spearmanr(x1_all, y1_all)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)

    # Plot markers
    ax.scatter(
        x1_in, y1_in,
        c="#5B9BD5", marker="o", s=36, alpha=0.6,
        edgecolors="#2B5B84", linewidths=0.5,
        label="Object IN image"
    )
    ax.scatter(
        x1_out, y1_out,
        c="#D9534F", marker="^", s=36, alpha=0.55,
        edgecolors="#8B0000", linewidths=0.5,
        label="Object NOT in image"
    )

    # Axis limits & y=x line
    min_val = min(x1_all.min(), y1_all.min()) - 0.02
    max_val = max(x1_all.max(), y1_all.max()) + 0.02
    ax.plot([min_val, max_val], [min_val, max_val], color="#888888", linestyle="--", linewidth=1.5, label="y=x (perfect correlation)", zorder=1)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Labels & Title
    ax.set_xlabel('cos_sim(image, Positive Text)', fontsize=12, fontweight='medium')
    ax.set_ylabel('cos_sim(image, Negative Text)', fontsize=12, fontweight='medium')
    
    p_str1 = "0.0e+00" if p1_p < 1e-15 else f"{p1_p:.2e}"
    sp_str1 = "0.0e+00" if p1_s < 1e-15 else f"{p1_s:.2e}"
    title_str1 = f"Image-Fixed Similarity: Positive vs Negative Text\nPearson r={r1_p:.3f} (p={p_str1}), Spearman ρ={r1_s:.3f} (p={sp_str1})"
    ax.set_title(title_str1, fontsize=13, fontweight="bold", pad=12)

    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=11, loc="upper left", frameon=True, framealpha=0.9, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    out_path1_a = os.path.join(output_dir, "image_text_correlation_image_fixed.png")
    out_path1_b = os.path.join(output_dir, "image_fixed_correlation.png")
    plt.savefig(out_path1_a, bbox_inches="tight", dpi=300)
    plt.savefig(out_path1_b, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved Graph 1 to: {out_path1_a}")

    # =========================================================================
    # GRAPH 2: Text Fixed (텍스트 고정)
    # X: cos_sim(Original Image [Object Present], text)
    # Y: cos_sim(Counterfactual Image [Object Absent], text)
    # Legend: Positive Text ("There is A") vs Negative Text ("There is no A")
    # =========================================================================
    x2_pos = df["A_sim_orig_pos"].values
    y2_pos = df["C_sim_cf_pos"].values

    x2_neg = df["B_sim_orig_neg"].values
    y2_neg = df["D_sim_cf_neg"].values

    x2_all = np.concatenate([x2_pos, x2_neg])
    y2_all = np.concatenate([y2_pos, y2_neg])

    r2_p, p2_p = stats.pearsonr(x2_all, y2_all)
    r2_s, p2_s = stats.spearmanr(x2_all, y2_all)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)

    # Plot markers
    ax.scatter(
        x2_pos, y2_pos,
        c="#33B5E5", marker="o", s=36, alpha=0.6,
        edgecolors="#0059B3", linewidths=0.5,
        label='Positive Caption ("There is A")'
    )
    ax.scatter(
        x2_neg, y2_neg,
        c="#FF8800", marker="^", s=36, alpha=0.55,
        edgecolors="#993300", linewidths=0.5,
        label='Negative Caption ("There is no A")'
    )

    # Axis limits & y=x line
    min_val2 = min(x2_all.min(), y2_all.min()) - 0.02
    max_val2 = max(x2_all.max(), y2_all.max()) + 0.02
    ax.plot([min_val2, max_val2], [min_val2, max_val2], color="#888888", linestyle="--", linewidth=1.5, label="y=x (perfect correlation)", zorder=1)

    ax.set_xlim(min_val2, max_val2)
    ax.set_ylim(min_val2, max_val2)

    # Labels & Title
    ax.set_xlabel('cos_sim(Original Image [Object Present], text)', fontsize=12, fontweight='medium')
    ax.set_ylabel('cos_sim(Counterfactual Image [Object Absent], text)', fontsize=12, fontweight='medium')
    
    p_str2 = "0.0e+00" if p2_p < 1e-15 else f"{p2_p:.2e}"
    sp_str2 = "0.0e+00" if p2_s < 1e-15 else f"{p2_s:.2e}"
    title_str2 = f"Text-Fixed Similarity: Original vs Counterfactual Image\nPearson r={r2_p:.3f} (p={p_str2}), Spearman ρ={r2_s:.3f} (p={sp_str2})"
    ax.set_title(title_str2, fontsize=13, fontweight="bold", pad=12)

    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=11, loc="upper left", frameon=True, framealpha=0.9, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    out_path2_a = os.path.join(output_dir, "image_text_correlation_text_fixed.png")
    out_path2_b = os.path.join(output_dir, "text_fixed_correlation.png")
    plt.savefig(out_path2_a, bbox_inches="tight", dpi=300)
    plt.savefig(out_path2_b, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved Graph 2 to: {out_path2_a}")

if __name__ == "__main__":
    csv_path = r"logs/evaluation/beaf_counterfactual_v5_2/beaf_4way_matrix.csv"
    output_dir = r"logs/evaluation/beaf_counterfactual_v5_2"
    generate_plots(csv_path, output_dir)
