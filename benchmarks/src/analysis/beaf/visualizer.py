"""
BEAF Counterfactual Visualization Module.

Contains rendering functions for scatter plots, histograms, heatmaps, and bar charts.
"""

import os
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.config import batch_cosine_similarity


# =========================================================================== #
# Part A (v1) Renderers: 4-Axis Plots
# =========================================================================== #

def render_image_image_histogram(img_img_df: pd.DataFrame, output_dir: str):
    """Plot distribution and per-object boxplot of sim(orig, cf) — Axis 3."""
    valid = img_img_df.dropna(subset=["sim_img_img"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    ax.hist(valid["sim_img_img"], bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    mean_val = valid["sim_img_img"].mean()
    ax.axvline(mean_val, color="crimson", ls="--", lw=2, label=f"Mean = {mean_val:.4f}")
    ax.set_xlabel("Cosine Similarity  (Original Image <-> Counterfactual Image)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Image Encoder Sensitivity to Object Removal\n(Distribution of Image<->Image Cosine Similarity)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.4)

    ax2 = axes[1]
    top_objects = valid["object_name"].value_counts().head(15).index.tolist()
    plot_data = [valid.loc[valid["object_name"] == obj, "sim_img_img"].values for obj in top_objects]
    bp = ax2.boxplot(plot_data, patch_artist=True, vert=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_objects)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax2.set_xticks(range(1, len(top_objects) + 1))
    ax2.set_xticklabels(top_objects, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Cosine Similarity  (orig <-> cf)", fontsize=11)
    ax2.set_title("Per-Object Visual Change Sensitivity\n(Lower = Image Encoder Detects Object Removal Better)", fontsize=11, fontweight="bold")
    ax2.grid(True, ls="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beaf_image_image_histogram.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_image_image_histogram.png")


def render_4way_heatmap(matrix_df: pd.DataFrame, output_dir: str):
    """Render 2x2 average similarity heatmap (Axis 4)."""
    valid = matrix_df.dropna(subset=["A_sim_orig_pos"])
    grid = np.array([
        [valid["A_sim_orig_pos"].mean(), valid["B_sim_orig_neg"].mean()],
        [valid["C_sim_cf_pos"].mean(),   valid["D_sim_cf_neg"].mean()],
    ])
    vmin, vmax = grid.min() - 0.005, grid.max() + 0.005
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Positive Caption\n(object present in text)", "Negative Caption\n(object absent in text)"], fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Original Image\n(object IN)", "Counterfactual\n(object OUT)"], fontsize=11)
    ax.set_title("CLIP 4-Way Similarity Matrix\n(Image State x Text Polarity)", fontsize=12, fontweight="bold")
    for i in range(2):
        for j in range(2):
            v = grid[i, j]
            tc = "white" if (v < vmin + (vmax - vmin) * 0.25 or v > vmax - (vmax - vmin) * 0.25) else "black"
            ax.text(j, i, f"{'ABCD'[i*2+j]}\n{v:.4f}", ha="center", va="center", fontsize=15, fontweight="bold", color=tc)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beaf_4way_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_4way_heatmap.png")


def render_text_vs_visual_scatter(matrix_df: pd.DataFrame, output_dir: str):
    """Scatter: Text Negation Score (A-B) vs Visual Change Score (A-C) — Axis 4."""
    valid = matrix_df.dropna(subset=["text_negation_score", "visual_change_score", "full_correct"])
    colors = valid["full_correct"].map({True: "seagreen", False: "crimson"})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(valid["text_negation_score"], valid["visual_change_score"], c=colors, alpha=0.55, s=28, edgecolors="none")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Text Negation Score  A - B  (sim_orig_pos - sim_orig_neg)", fontsize=10)
    ax.set_ylabel("Visual Change Score  A - C  (sim_orig_pos - sim_cf_pos)", fontsize=10)
    ax.set_title("CLIP Sensitivity:\nText Negation vs Visual Object Removal", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="seagreen", label="Full Correct (A>B & D>C & A>C & D>B)"), Patch(facecolor="crimson", label="Not Full Correct")], fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beaf_text_vs_visual_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_text_vs_visual_scatter.png")


def render_full_correct_by_object(matrix_df: pd.DataFrame, output_dir: str):
    """Horizontal bar chart of Full Correct Rate per object category — Axis 4."""
    valid = matrix_df.dropna(subset=["full_correct"])
    obj_stats = valid.groupby("object_name")["full_correct"].agg(["sum", "count"]).rename(columns={"sum": "correct", "count": "total"})
    obj_stats["rate_pct"] = obj_stats["correct"] / obj_stats["total"] * 100
    obj_stats = obj_stats.sort_values("rate_pct", ascending=True)
    overall_mean = valid["full_correct"].mean() * 100

    fig, ax = plt.subplots(figsize=(9, max(5, len(obj_stats) * 0.38)))
    bar_colors = [plt.cm.RdYlGn(r / 100) for r in obj_stats["rate_pct"]]
    bars = ax.barh(obj_stats.index, obj_stats["rate_pct"], color=bar_colors, edgecolor="white", height=0.7)
    for bar, (idx, row) in zip(bars, obj_stats.iterrows()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{int(row['correct'])}/{int(row['total'])}", va="center", fontsize=7.5, color="#333333")
    ax.axvline(overall_mean, color="navy", ls="--", lw=1.5, label=f"Overall Mean = {overall_mean:.1f}%")
    ax.set_xlabel("Full Correct Rate (%)", fontsize=11)
    ax.set_title("CLIP Full Correct Rate by Object Category\n(A>B & D>C & A>C & D>B)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 108)
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", alpha=0.4, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beaf_full_correct_rate_by_object.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_full_correct_rate_by_object.png")


# =========================================================================== #
# Part B (v2) Renderers: Vision & Sensitivity Scatter Plots
# =========================================================================== #

def render_scatter_pos_vs_neg(
    all_img_embs: np.ndarray,
    all_pos_embs: np.ndarray,
    all_neg_embs: np.ndarray,
    all_obj_in_img: np.ndarray,
    output_dir: str
):
    """COCO-style Pos vs Neg Cosine Similarity Scatter Plot (N=3,556)."""
    sim_pos = batch_cosine_similarity(all_img_embs, all_pos_embs)
    sim_neg = batch_cosine_similarity(all_img_embs, all_neg_embs)

    r_val, _ = stats.pearsonr(sim_pos, sim_neg)
    rho_val, _ = stats.spearmanr(sim_pos, sim_neg)
    N = len(sim_pos)

    fig, ax = plt.subplots(figsize=(8, 7))

    mask_true = (all_obj_in_img == True)
    mask_false = (all_obj_in_img == False)

    ax.scatter(sim_pos[mask_true], sim_neg[mask_true], c="dodgerblue", alpha=0.35, s=16, label="Object IN image", edgecolors="none")
    ax.scatter(sim_pos[mask_false], sim_neg[mask_false], c="crimson", marker="^", alpha=0.35, s=16, label="Object NOT in image", edgecolors="none")

    min_val = min(sim_pos.min(), sim_neg.min()) - 0.02
    max_val = max(sim_pos.max(), sim_neg.max()) + 0.02
    ax.plot([min_val, max_val], [min_val, max_val], color="gray", ls="--", lw=1.5, label="y=x (perfect correlation)")

    ax.set_xlabel("cos_sim(image, positive_caption)", fontsize=11, fontweight="bold")
    ax.set_ylabel("cos_sim(image, negative_caption)", fontsize=11, fontweight="bold")
    ax.set_title(f"Image-Text Similarity: Positive vs Negative (BEAF Counterfactual)\nPearson r={r_val:.3f}, Spearman rho={rho_val:.3f} (N={N:,})", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(fontsize=10, loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "beaf_scatter_pos_vs_neg.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_scatter_pos_vs_neg.png")


def render_scatter_delta_quadrant(
    sim_orig_pos: np.ndarray,
    sim_orig_neg: np.ndarray,
    sim_cf_pos: np.ndarray,
    output_dir: str
):
    """
    Delta-Delta 4-Quadrant Scatter Plot:
      X = sim(orig, pos) - sim(orig, neg)  [Text Negation Sensitivity]
      Y = sim(orig, pos) - sim(cf, pos)    [Visual Change Sensitivity]
    """
    delta_text = sim_orig_pos - sim_orig_neg
    delta_visual = sim_orig_pos - sim_cf_pos
    N = len(delta_text)

    q1 = np.sum((delta_text > 0) & (delta_visual > 0)) / N * 100
    q2 = np.sum((delta_text <= 0) & (delta_visual > 0)) / N * 100
    q3 = np.sum((delta_text <= 0) & (delta_visual <= 0)) / N * 100
    q4 = np.sum((delta_text > 0) & (delta_visual <= 0)) / N * 100

    fig, ax = plt.subplots(figsize=(8, 7))

    colors = []
    for dt, dv in zip(delta_text, delta_visual):
        if dt > 0 and dv > 0:
            colors.append("seagreen")     # Q1
        elif dt <= 0 and dv > 0:
            colors.append("dodgerblue")   # Q2
        elif dt > 0 and dv <= 0:
            colors.append("crimson")      # Q4 (most common failure)
        else:
            colors.append("gray")         # Q3

    ax.scatter(delta_text, delta_visual, c=colors, alpha=0.45, s=20, edgecolors="none")

    ax.axhline(0, color="black", lw=1.2, ls="--")
    ax.axvline(0, color="black", lw=1.2, ls="--")

    ax.set_xlabel("sim(orig, pos) - sim(orig, neg)", fontsize=11, fontweight="bold")
    ax.set_ylabel("sim(orig, pos) - sim(cf, pos)", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)

    ax.text(0.75, 0.90, f"Q1: Both Sensitive\n({q1:.1f}%)", transform=ax.transAxes, color="seagreen", fontweight="bold", fontsize=10)
    ax.text(0.05, 0.90, f"Q2: Visual Only\n({q2:.1f}%)", transform=ax.transAxes, color="dodgerblue", fontweight="bold", fontsize=10)
    ax.text(0.75, 0.05, f"Q4: Text Only (Failure)\n({q4:.1f}%)", transform=ax.transAxes, color="crimson", fontweight="bold", fontsize=10)
    ax.text(0.05, 0.05, f"Q3: Neither\n({q3:.1f}%)", transform=ax.transAxes, color="gray", fontweight="bold", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "beaf_scatter_delta_text_vs_delta_visual.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_scatter_delta_text_vs_delta_visual.png")


def render_scatter_img_orig_vs_img_cf(
    sim_orig_pos: np.ndarray,
    sim_cf_pos: np.ndarray,
    output_dir: str
):
    """Pure Visual Sensitivity Scatter: X = sim(I_orig, T_pos), Y = sim(I_cf, T_pos)."""
    fig, ax = plt.subplots(figsize=(7, 6.5))

    ax.scatter(sim_orig_pos, sim_cf_pos, c="darkslateblue", alpha=0.45, s=20, edgecolors="none")

    min_v = min(sim_orig_pos.min(), sim_cf_pos.min()) - 0.02
    max_v = max(sim_orig_pos.max(), sim_cf_pos.max()) + 0.02
    ax.plot([min_v, max_v], [min_v, max_v], color="crimson", ls="--", lw=1.5, label="y=x (No Visual Shift)")

    below_cnt = np.sum(sim_cf_pos < sim_orig_pos)
    pct_below = below_cnt / len(sim_orig_pos) * 100

    ax.set_xlabel("sim(Original Image, Positive Caption)", fontsize=10, fontweight="bold")
    ax.set_ylabel("sim(Counterfactual Image, Positive Caption)", fontsize=10, fontweight="bold")
    ax.set_title(f"Pure Visual Sensitivity: Original vs Counterfactual Image\n({pct_below:.1f}% pairs drop similarity when object removed)", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "beaf_scatter_img_orig_vs_img_cf.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_scatter_img_orig_vs_img_cf.png")


def render_scatter_by_object_category(
    df_pairs: pd.DataFrame,
    sim_orig_pos: np.ndarray,
    sim_orig_neg: np.ndarray,
    output_dir: str
):
    """Subplot grid showing Pos vs Neg scatter across Top 6 Object Categories."""
    df_pairs_copy = df_pairs.copy()
    df_pairs_copy["sim_pos"] = sim_orig_pos
    df_pairs_copy["sim_neg"] = sim_orig_neg

    top_objs = df_pairs_copy["object_name"].value_counts().head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, obj in enumerate(top_objs):
        ax = axes[idx]
        sub = df_pairs_copy[df_pairs_copy["object_name"] == obj]

        ax.scatter(sub["sim_pos"], sub["sim_neg"], c="dodgerblue", alpha=0.6, s=24)

        min_v = min(sub["sim_pos"].min(), sub["sim_neg"].min()) - 0.01
        max_v = max(sub["sim_pos"].max(), sub["sim_neg"].max()) + 0.01
        ax.plot([min_v, max_v], [min_v, max_v], color="crimson", ls="--", lw=1.2)

        if len(sub) >= 2:
            r_sub, _ = stats.pearsonr(sub["sim_pos"], sub["sim_neg"])
            r_str = f"Pearson r={r_sub:.3f}"
        else:
            r_str = "n < 2"
        ax.set_title(f"Object: {obj} (n={len(sub)})\n{r_str}", fontsize=11, fontweight="bold")
        ax.set_xlabel("sim(pos)", fontsize=9)
        ax.set_ylabel("sim(neg)", fontsize=9)
        ax.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "beaf_scatter_by_object_category.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_scatter_by_object_category.png")


def render_2x2_factorial_anova_plots(
    anova_df: pd.DataFrame,
    summary_anova: Dict[str, Any],
    output_dir: str
):
    """Render 2x2 Factorial ANOVA Main Effects and Interaction Effect Distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # 1. Text Main Effect
    ax1 = axes[0]
    ax1.hist(anova_df["text_main_effect"], bins=50, color="teal", alpha=0.8, edgecolor="white")
    mean_t = summary_anova["text_main_effect"]["mean"]
    ci_t_low = summary_anova["text_main_effect"]["ci_95_low"]
    ci_t_high = summary_anova["text_main_effect"]["ci_95_high"]
    ax1.axvline(mean_t, color="darkred", ls="--", lw=2, label=f"Mean = {mean_t:.4f}\n(95% CI: [{ci_t_low:.4f}, {ci_t_high:.4f}])")
    ax1.axvline(0, color="gray", ls=":", lw=1.5)
    ax1.set_title("1. Text Main Effect\n[((A-B) + (C-D)) / 2]", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Similarity Difference", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, ls="--", alpha=0.4)

    # 2. Visual Main Effect
    ax2 = axes[1]
    ax2.hist(anova_df["visual_main_effect"], bins=50, color="darkorange", alpha=0.8, edgecolor="white")
    mean_v = summary_anova["visual_main_effect"]["mean"]
    ci_v_low = summary_anova["visual_main_effect"]["ci_95_low"]
    ci_v_high = summary_anova["visual_main_effect"]["ci_95_high"]
    ax2.axvline(mean_v, color="darkred", ls="--", lw=2, label=f"Mean = {mean_v:.4f}\n(95% CI: [{ci_v_low:.4f}, {ci_v_high:.4f}])")
    ax2.axvline(0, color="gray", ls=":", lw=1.5)
    ax2.set_title("2. Visual Main Effect\n[((A-C) + (B-D)) / 2]", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Similarity Difference", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, ls="--", alpha=0.4)

    # 3. Interaction Effect
    ax3 = axes[2]
    ax3.hist(anova_df["interaction_effect"], bins=50, color="crimson", alpha=0.8, edgecolor="white")
    mean_i = summary_anova["interaction_effect"]["mean"]
    ci_i_low = summary_anova["interaction_effect"]["ci_95_low"]
    ci_i_high = summary_anova["interaction_effect"]["ci_95_high"]
    pct_neg = summary_anova["interaction_effect"]["negative_interaction_pct"]
    ax3.axvline(mean_i, color="navy", ls="--", lw=2, label=f"Mean = {mean_i:.4f}\n(95% CI: [{ci_i_low:.4f}, {ci_i_high:.4f}])\nNeg. Inter. = {pct_neg:.1f}%")
    ax3.axvline(0, color="black", ls="-", lw=1.5)
    ax3.set_title("3. Interaction Effect (Core Claim)\n[(A-B) - (C-D)]", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Cross-Modal Interaction", fontsize=10)
    ax3.set_ylabel("Frequency", fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "beaf_2x2_factorial_anova.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_2x2_factorial_anova.png")


def render_2d_margin_state_space(
    sim_orig_pos: np.ndarray,
    sim_orig_neg: np.ndarray,
    sim_cf_pos: np.ndarray,
    sim_cf_neg: np.ndarray,
    output_dir: str
):
    """Render 2D Margin State Space (m_orig vs m_cf) Scatter plot with 4 Task Quadrants and Regression Line."""
    m_orig = sim_orig_pos - sim_orig_neg
    m_cf   = sim_cf_neg - sim_cf_pos

    slope, intercept, r_value, p_value, std_err = stats.linregress(m_orig, m_cf)

    n_total = len(m_orig)
    q1 = (m_orig > 0) & (m_cf > 0)
    q2 = (m_orig <= 0) & (m_cf > 0)
    q3 = (m_orig <= 0) & (m_cf <= 0)
    q4 = (m_orig > 0) & (m_cf <= 0)

    pct_q1 = (q1.sum() / n_total) * 100
    pct_q2 = (q2.sum() / n_total) * 100
    pct_q3 = (q3.sum() / n_total) * 100
    pct_q4 = (q4.sum() / n_total) * 100

    fig, ax = plt.subplots(figsize=(8, 7))

    # Scatter points colored by quadrant
    ax.scatter(m_orig[q4], m_cf[q4], c="crimson", alpha=0.55, s=24)
    ax.scatter(m_orig[q1], m_cf[q1], c="seagreen", alpha=0.7, s=28)
    ax.scatter(m_orig[q2], m_cf[q2], c="dodgerblue", alpha=0.6, s=24)
    ax.scatter(m_orig[q3], m_cf[q3], c="gray", alpha=0.6, s=24)

    # Quadrant lines
    ax.axhline(0, color="black", ls="--", lw=1.2)
    ax.axvline(0, color="black", ls="--", lw=1.2)

    # Regression Line (without legend label)
    x_vals = np.linspace(m_orig.min() - 0.005, m_orig.max() + 0.005, 200)
    y_vals = slope * x_vals + intercept
    ax.plot(x_vals, y_vals, color="black", ls="-", lw=2.0)

    # Axis Labels (pure formulas)
    ax.set_xlabel("sim(orig, pos) - sim(orig, neg)", fontsize=11, fontweight="bold")
    ax.set_ylabel("sim(cf, neg) - sim(cf, pos)", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)

    # Quadrant Percentage Annotations inside the plot
    ax.text(0.70, 0.90, f"Q1: Flip Success\n({pct_q1:.1f}%)", transform=ax.transAxes, color="seagreen", fontweight="bold", fontsize=10)
    ax.text(0.05, 0.90, f"Q2: CF-Only Success\n({pct_q2:.1f}%)", transform=ax.transAxes, color="dodgerblue", fontweight="bold", fontsize=10)
    ax.text(0.70, 0.05, f"Q4: Original-Only Success\n({pct_q4:.1f}%)", transform=ax.transAxes, color="crimson", fontweight="bold", fontsize=10)
    ax.text(0.05, 0.05, f"Q3: Both Fail\n({pct_q3:.1f}%)", transform=ax.transAxes, color="gray", fontweight="bold", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "beaf_2d_margin_state_space.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_2d_margin_state_space.png")


# =========================================================================== #
# Per-Object Layerwise Analysis Renderer
# =========================================================================== #

def render_per_object_layerwise_plot(
    summary: Dict[str, Any],
    output_dir: str,
    raw_df: Optional[pd.DataFrame] = None,
) -> None:
    """Render per-object layerwise cosine similarity and linear probe accuracy.

    Produces a dual-Y-axis line plot where:
      - Left Y-axis (blue): mean linear probe accuracy (%) +- 1 std across objects
      - Right Y-axis (red): mean cosine similarity sim(orig, cf) +- 1 std across objects
      - X-axis: Vision Transformer layers (Embedding, Layer 1..12, Pre-Proj, +Final L2Norm)
    """
    import json

    layer_names = list(summary.keys())

    probe_means = [summary[lk]["probe_acc"]["mean"]   for lk in layer_names]
    probe_stds  = [summary[lk]["probe_acc"]["std"]    for lk in layer_names]
    cos_means   = [summary[lk]["cosine_sim"]["mean"]  for lk in layer_names]
    cos_stds    = [summary[lk]["cosine_sim"]["std"]   for lk in layer_names]

    probe_means = np.array(probe_means, dtype=float)
    probe_stds  = np.array(probe_stds,  dtype=float)
    cos_means   = np.array(cos_means,   dtype=float)
    cos_stds    = np.array(cos_stds,    dtype=float)

    x = list(range(len(layer_names)))

    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax2 = ax1.twinx()

    color_probe = "#1f77b4"
    ax1.plot(x, probe_means, "o-", color=color_probe, lw=2.5, ms=7, label="Linear Probe Acc (%) - mean across objects")
    ax1.fill_between(x, probe_means - probe_stds, probe_means + probe_stds, color=color_probe, alpha=0.15, label="+-1 Std (objects)")
    ax1.set_ylabel("Linear Probe Accuracy (%)", fontsize=12, color=color_probe)
    ax1.tick_params(axis="y", labelcolor=color_probe)
    ax1.set_ylim(max(0, float(np.nanmin(probe_means - probe_stds)) - 5), min(102, float(np.nanmax(probe_means + probe_stds)) + 5))

    color_cos = "#d62728"
    ax2.plot(x, cos_means, "s--", color=color_cos, lw=2.0, ms=6, label="Cosine Sim sim(orig, cf) - mean across objects")
    ax2.fill_between(x, cos_means - cos_stds, cos_means + cos_stds, color=color_cos, alpha=0.12, label="+-1 Std (objects)")
    ax2.set_ylabel("Cosine Similarity sim(orig, cf)", fontsize=12, color=color_cos)
    ax2.tick_params(axis="y", labelcolor=color_cos)

    try:
        pre_proj_idx = layer_names.index("Pre-Projection")
        ax1.axvline(x=pre_proj_idx - 0.5, color="gray", ls=":", lw=1.5, alpha=0.7, label="Post-Transformer Transformations")
    except ValueError:
        pass

    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_names, rotation=35, ha="right", fontsize=9)
    ax1.set_xlabel("Vision Transformer Layer / Pipeline Step", fontsize=12)
    ax1.grid(True, ls="--", alpha=0.4, axis="y")

    n_objs = len(summary[layer_names[0]]["probe_acc"]["per_object"]) if layer_names else 0
    ax1.set_title(f"Per-Object Layerwise: Linear Probe Accuracy & Cosine Similarity\nMean +- Std across {n_objs} objects", fontsize=13, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")

    plt.tight_layout()
    png_path = os.path.join(output_dir, "beaf_per_object_layerwise_analysis.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_per_object_layerwise_analysis.png")

    def _clean(obj):
        if isinstance(obj, float) and (obj != obj):
            return None
        return obj

    json_ready = {
        lk: {
            metric: {
                k: (_clean(v) if not isinstance(v, dict) else {kk: _clean(vv) for kk, vv in v.items()})
                for k, v in metric_data.items()
            }
            for metric, metric_data in layer_data.items()
        }
        for lk, layer_data in summary.items()
    }
    json_path = os.path.join(output_dir, "beaf_per_object_layerwise_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2, ensure_ascii=False)
    print("  Saved: beaf_per_object_layerwise_analysis.json")

    if raw_df is not None and len(raw_df) > 0:
        csv_path = os.path.join(output_dir, "beaf_per_object_layerwise_raw.csv")
        raw_df.to_csv(csv_path, index=False)
        print("  Saved: beaf_per_object_layerwise_raw.csv")
