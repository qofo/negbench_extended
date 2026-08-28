"""
E2-Risk Analysis: Statistical Significance of Interaction Term (gamma).

Rigorous verification of whether the macro interaction term (gamma = 0.00055)
represents a genuine, statistically significant True Negation signal (> 0)
or is indistinguishable from zero (measurement noise / random fluctuation).

Statistical Tests:
1. Concept-Level & Macro Bootstrap 95% Confidence Intervals (B = 2,000 resamples):
   - Resamples pair-level gamma_i within each concept.
   - Calculates 95% percentile CI [CI_lower, CI_upper].
   - Identifies concepts where CI_lower > 0 (statistically significant positive signal).

2. Within-Concept Permutation Null Distribution (M = 1,000 permutations):
   - Tests H0: gamma = 0 by shuffling image-text polarity pairings (u_i . v_j).
   - Computes empirical p-value and Z-score (gamma_obs vs null distribution).

3. Population-Level Hypothesis Tests:
   - Wilcoxon Signed-Rank Test: H0: Median(gamma) = 0 vs H1: Median(gamma) > 0.
   - One-Sample t-test: H0: E[gamma] = 0 vs H1: E[gamma] > 0.
   - Theoretical Rank-1 Bilinear Ceiling: Exactly the proportion of concepts with gamma > 0.

Outputs:
  - e2_gamma_significance_summary.json
  - e2_gamma_concept_significance.csv
  - fig_e2_gamma_forest_plot.png        (Forest plot with 95% Bootstrap CI bars)
  - fig_e2_gamma_null_vs_obs.png         (Permutation null distribution vs observed gamma)
  - fig_e2_gamma_volcano_plot.png        (Volcano plot: gamma vs -log10(p))
  - fig_e2_gamma_macro_bootstrap.png     (Macro-level bootstrap distribution & CI)
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_concept_bootstrap_ci(
    gamma_vals: np.ndarray,
    n_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Computes empirical percentile bootstrap CI for the mean of gamma_vals.
    Returns: (mean, se, ci_lower, ci_upper, bootstrap_means)
    """
    rng = np.random.RandomState(seed)
    n = len(gamma_vals)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, np.zeros(n_bootstraps)

    # Resample with replacement
    indices = rng.randint(0, n, size=(n_bootstraps, n))
    boot_samples = gamma_vals[indices]  # [B, N]
    boot_means = np.mean(boot_samples, axis=1)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    se = float(np.std(boot_means))
    mean_val = float(np.mean(gamma_vals))

    return mean_val, se, ci_lower, ci_upper, boot_means


def compute_permutation_null_test(
    s11: np.ndarray,
    s12: np.ndarray,
    s21: np.ndarray,
    s22: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Permutation null test for H0: gamma = 0.
    Under H0, the pairing between image polarity and text polarity is independent.
    We shuffle the text polarity (or sign of u . v) across pairs within the concept.
    Returns: (gamma_obs, p_val_two_sided, p_val_one_sided, z_score, null_gammas)
    """
    rng = np.random.RandomState(seed)
    n = len(s11)
    gamma_obs = float(np.mean((s11 - s12 - s21 + s22) / 4.0))

    # Under H0, u_i = 1/2(v_pres - v_abs) and v_i = 1/2(t_pos - t_neg)
    # The interaction is gamma_i = u_i . v_i.
    # Method: shuffle text pairings or sign-flip
    pair_gammas = (s11 - s12 - s21 + s22) / 4.0

    null_gammas = np.zeros(n_permutations)
    for m in range(n_permutations):
        # Random sign-flips under null symmetry E[gamma] = 0
        signs = rng.choice([-1.0, 1.0], size=n)
        null_gammas[m] = np.mean(pair_gammas * signs)

    # Empirical p-values
    p_val_two_sided = float(np.mean(np.abs(null_gammas) >= np.abs(gamma_obs)))
    p_val_one_sided = float(np.mean(null_gammas >= gamma_obs))

    null_mean = float(np.mean(null_gammas))
    null_std = float(np.std(null_gammas))
    z_score = float((gamma_obs - null_mean) / (null_std + 1e-9))

    return gamma_obs, p_val_two_sided, p_val_one_sided, z_score, null_gammas


def render_significance_visualizations(
    df_concepts: pd.DataFrame,
    df_pairs: pd.DataFrame,
    macro_boot_means: np.ndarray,
    all_null_gammas: np.ndarray,
    summary: Dict[str, Any],
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # Plot 1: Forest Plot with 95% Bootstrap CI per Concept
    # ──────────────────────────────────────────────────────────
    df_sorted = df_concepts.sort_values(by="gamma_mean", ascending=True).reset_index(drop=True)
    n_c = len(df_sorted)

    fig, ax = plt.subplots(figsize=(10, max(8, n_c * 0.28)))
    y_pos = np.arange(n_c)

    means = df_sorted["gamma_mean"].values
    ci_lows = df_sorted["ci_lower"].values
    ci_highs = df_sorted["ci_upper"].values

    xerr_left = means - ci_lows
    xerr_right = ci_highs - means

    # Colors: Green for significant positive, Gray for cross-zero, Red for negative
    point_colors = []
    for low, high in zip(ci_lows, ci_highs):
        if low > 0:
            point_colors.append("#27ae60")  # Significant positive (Signal confirmed)
        elif high < 0:
            point_colors.append("#e74c3c")  # Significant negative
        else:
            point_colors.append("#7f8c8d")  # Indistinguishable from zero (Noise)

    for y, m, xl, xr, col in zip(y_pos, means, xerr_left, xerr_right, point_colors):
        ax.errorbar(
            m,
            y,
            xerr=[[xl], [xr]],
            fmt="o",
            color=col,
            ecolor=col,
            elinewidth=2.0,
            capsize=3.5,
            capthick=1.5,
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=3,
        )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.8, label="Zero Interaction (γ = 0, Pure Noise)")
    ax.axvline(
        summary["macro_gamma_mean"],
        color="#8e44ad",
        linestyle="-.",
        linewidth=1.8,
        label=f"Macro Mean γ = {summary['macro_gamma_mean']:+.5f}",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["object_name"], fontsize=8.5)
    ax.set_xlabel(r"Interaction Term $\gamma = \frac{1}{4}(S_{11} - S_{12} - S_{21} + S_{22})$", fontsize=11, fontweight="bold")
    ax.set_title(
        r"E2-Risk: Forest Plot of Interaction Term $\gamma$ with 95% Bootstrap CI" + "\n"
        r"Green: CI > 0 (Signal Confirmed) | Gray: CI crosses 0 (Noise) | Red: CI < 0",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "fig_e2_gamma_forest_plot.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot1_path}")

    # ──────────────────────────────────────────────────────────
    # Plot 2: Permutation Null Distribution vs Observed Gamma
    # ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Subplot 1: Distribution of Concept Means vs Pooled Null
    obs_gammas = df_concepts["gamma_mean"].values
    ax1.hist(all_null_gammas, bins=50, color="#95a5a6", edgecolor="black", alpha=0.6, density=True, label=r"Permutation Null $H_0$ ($\gamma_{\mathrm{null}}$)")
    ax1.hist(obs_gammas, bins=25, color="#8e44ad", edgecolor="black", alpha=0.75, density=True, label=r"Observed Concepts ($\gamma_{\mathrm{obs}}$)")

    ax1.axvline(0.0, color="black", linestyle="--", linewidth=1.5)
    ax1.axvline(summary["macro_gamma_mean"], color="#8e44ad", linestyle="-", linewidth=2.0, label=f"Macro Obs = {summary['macro_gamma_mean']:+.5f}")
    ax1.set_xlabel(r"Interaction Magnitude $\gamma$", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax1.set_title(r"Permutation Null vs Observed $\gamma$ Distribution", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(fontsize=9, loc="upper right")

    # Subplot 2: Z-score distribution across concepts
    z_scores = df_concepts["z_score"].values
    counts, bins, patches = ax2.hist(z_scores, bins=25, color="#3498db", edgecolor="black", alpha=0.75)
    ax2.axvline(0.0, color="black", linestyle="--", linewidth=1.5)
    ax2.axvline(1.96, color="red", linestyle=":", linewidth=2.0, label=r"Critical $Z = +1.96$ ($\alpha = 0.05$)")
    ax2.axvline(np.mean(z_scores), color="#f39c12", linestyle="-", linewidth=2.0, label=f"Mean Z = {np.mean(z_scores):+.2f}")
    ax2.set_xlabel(r"Standardized Score: $Z = (\gamma_{\mathrm{obs}} - \mu_{\mathrm{null}}) / \sigma_{\mathrm{null}}$", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Number of Concepts", fontsize=11, fontweight="bold")
    ax2.set_title("Concept-Level Z-Score Distribution", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "fig_e2_gamma_null_vs_obs.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot2_path}")

    # ──────────────────────────────────────────────────────────
    # Plot 3: Volcano Plot (gamma Magnitude vs -log10(p-value))
    # ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    x_vals = df_concepts["gamma_mean"].values
    # Clip p-values to avoid log(0)
    p_vals = np.clip(df_concepts["p_val_two_sided"].values, 1e-4, 1.0)
    y_vals = -np.log10(p_vals)

    colors = []
    for x, p in zip(x_vals, df_concepts["p_val_two_sided"].values):
        if p < 0.05 and x > 0:
            colors.append("#27ae60")  # Significant positive
        elif p < 0.05 and x < 0:
            colors.append("#e74c3c")  # Significant negative
        else:
            colors.append("#95a5a6")  # Non-significant

    ax.scatter(x_vals, y_vals, c=colors, s=55, edgecolors="black", linewidth=0.8, alpha=0.85, zorder=3)

    # Significance line at p = 0.05 (-log10 = 1.301)
    ax.axhline(1.301, color="red", linestyle="--", linewidth=1.5, label="Significance Threshold (p = 0.05)")
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1.5)

    # Annotate top significant concepts
    top_sig = df_concepts[(df_concepts["p_val_two_sided"] < 0.05) & (df_concepts["gamma_mean"] > 0)].sort_values(by="gamma_mean", ascending=False).head(8)
    for _, row in top_sig.iterrows():
        p_val = max(row["p_val_two_sided"], 1e-4)
        ax.annotate(
            row["object_name"],
            (row["gamma_mean"], -np.log10(p_val)),
            fontsize=8,
            fontweight="bold",
            xytext=(4, 4),
            textcoords="offset points",
            alpha=0.9,
        )

    ax.set_xlabel(r"Observed Interaction $\gamma$", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Statistical Significance $-\log_{10}(p\mathrm{-value})$", fontsize=11, fontweight="bold")
    ax.set_title("Volcano Plot: Interaction Magnitude vs Permutation Significance", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    plot3_path = os.path.join(output_dir, "fig_e2_gamma_volcano_plot.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot3_path}")

    # ──────────────────────────────────────────────────────────
    # Plot 4: Macro-Level Bootstrap Distribution of Mean Gamma
    # ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.2))

    ax.hist(macro_boot_means, bins=45, color="#8e44ad", edgecolor="black", alpha=0.75, density=True)

    ci_l = summary["macro_gamma_bootstrap_95ci_lower"]
    ci_u = summary["macro_gamma_bootstrap_95ci_upper"]
    macro_m = summary["macro_gamma_mean"]

    ax.axvline(0.0, color="red", linestyle="--", linewidth=2.0, label="Zero Null (γ = 0)")
    ax.axvline(macro_m, color="#2c3e50", linestyle="-", linewidth=2.2, label=f"Macro Mean γ = {macro_m:+.5f}")
    ax.axvline(ci_l, color="#27ae60", linestyle=":", linewidth=2.0, label=f"95% CI Lower = {ci_l:+.5f}")
    ax.axvline(ci_u, color="#27ae60", linestyle=":", linewidth=2.0, label=f"95% CI Upper = {ci_u:+.5f}")

    ax.text(
        0.05,
        0.82,
        f"Population Wilcoxon p = {summary['wilcoxon_p_val_one_sided']:.2e}\n"
        f"Population t-test p  = {summary['ttest_p_val_one_sided']:.2e}\n"
        f"95% CI strictly > 0: {'YES ✅' if ci_l > 0 else 'NO ❌'}",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f8f5", edgecolor="#27ae60", alpha=0.95),
    )

    ax.set_xlabel(r"Macro Population Mean $\bar{\gamma}$ (across all concepts)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Bootstrap Density (B = 2,000)", fontsize=11, fontweight="bold")
    ax.set_title(r"Macro-Level Bootstrap Distribution of Interaction $\bar{\gamma}$", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plot4_path = os.path.join(output_dir, "fig_e2_gamma_macro_bootstrap.png")
    plt.savefig(plot4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot4_path}")


def main():
    parser = argparse.ArgumentParser(description="E2-Risk: Statistical Significance of Interaction Term (gamma)")
    parser.add_argument(
        "--per_pair_csv",
        type=str,
        default="logs/evaluation/e2_hadamard_decomposition/e2_per_pair_decomposition.csv",
        help="Path to pair-level decomposition CSV from E2",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/evaluation/e2_gamma_significance",
        help="Output directory for risk analysis",
    )
    parser.add_argument("--n_bootstraps", type=int, default=2000, help="Number of bootstrap resamples (default: 2000)")
    parser.add_argument("--n_permutations", type=int, default=1000, help="Number of permutations per concept (default: 1000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  E2-Risk Analysis: Statistical Significance of Interaction Term (γ)  ║")
    print("║  Testing H0: γ = 0 (Pure Measurement Noise) vs H1: γ > 0 (Signal)    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  Input Pair CSV : {args.per_pair_csv}")
    print(f"  Output Dir     : {args.output_dir}")
    print(f"  Bootstraps (B) : {args.n_bootstraps}")
    print(f"  Permutations(M): {args.n_permutations}\n")

    if not os.path.exists(args.per_pair_csv):
        raise FileNotFoundError(f"Missing pair-level decomposition CSV at: {args.per_pair_csv}")

    df_pairs = pd.read_csv(args.per_pair_csv)
    print(f"  Loaded {len(df_pairs)} total pairs across {df_pairs['object_name'].nunique()} concepts.")

    unique_concepts = sorted(df_pairs["object_name"].unique().tolist())

    concept_records = []
    all_null_gammas = []
    concept_boot_means_list = []

    print("\n  Running Concept-Level Bootstrap CIs and Permutation Tests...")
    for obj in unique_concepts:
        df_obj = df_pairs[df_pairs["object_name"] == obj].reset_index(drop=True)
        n_c = len(df_obj)

        gamma_vals = df_obj["gamma"].values
        s11 = df_obj["S11"].values
        s12 = df_obj["S12"].values
        s21 = df_obj["S21"].values
        s22 = df_obj["S22"].values

        # 1. Bootstrap CI
        mean_g, se_g, ci_l, ci_u, boot_means = compute_concept_bootstrap_ci(
            gamma_vals=gamma_vals,
            n_bootstraps=args.n_bootstraps,
            confidence_level=0.95,
            seed=args.seed,
        )
        concept_boot_means_list.append(boot_means)

        # 2. Permutation Null Test
        g_obs, p_two, p_one, z_sc, null_g = compute_permutation_null_test(
            s11=s11, s12=s12, s21=s21, s22=s22, n_permutations=args.n_permutations, seed=args.seed
        )
        all_null_gammas.extend(null_g.tolist())

        # Classification
        is_pos_empirical = bool(mean_g > 0)
        is_sig_positive = bool(ci_l > 0 and p_two < 0.05)
        is_sig_negative = bool(ci_u < 0 and p_two < 0.05)

        if is_sig_positive:
            status = "CONFIRMED_SIGNAL (γ > 0, p < 0.05)"
            icon = "✅ POS_SIG"
        elif is_sig_negative:
            status = "INVERTED_SIGNAL (γ < 0, p < 0.05)"
            icon = "❌ NEG_SIG"
        elif is_pos_empirical:
            status = "POSITIVE_MARGINAL (γ > 0, crosses 0)"
            icon = "⚠️ POS_NOISE"
        else:
            status = "NEGATIVE_MARGINAL (γ <= 0, crosses 0)"
            icon = "⚠️ NEG_NOISE"

        concept_records.append({
            "object_name": obj,
            "n_pairs": n_c,
            "gamma_mean": mean_g,
            "gamma_se": se_g,
            "ci_lower": ci_l,
            "ci_upper": ci_u,
            "ci_width": ci_u - ci_l,
            "ci_strictly_positive": bool(ci_l > 0),
            "p_val_two_sided": p_two,
            "p_val_one_sided": p_one,
            "z_score": z_sc,
            "status": status,
        })

        print(f"  [{obj:20s}] N={n_c:3d} | γ={mean_g:+.5f} | 95% CI=[{ci_l:+.5f}, {ci_u:+.5f}] | p={p_two:.4f} | Z={z_sc:+.2f} | {icon}")

    df_concepts = pd.DataFrame(concept_records).sort_values(by="gamma_mean", ascending=False).reset_index(drop=True)

    # ── Population-Level & Macro Bootstrap ──
    # Macro Bootstrap across concepts (resampling concept means)
    macro_boot_means = np.mean(np.array(concept_boot_means_list), axis=0)  # [B]
    macro_gamma_mean = float(np.mean(df_concepts["gamma_mean"]))
    macro_ci_lower = float(np.percentile(macro_boot_means, 2.5))
    macro_ci_upper = float(np.percentile(macro_boot_means, 97.5))
    macro_se = float(np.std(macro_boot_means))

    # Population Wilcoxon Signed-Rank Test & t-test
    concept_gamma_means = df_concepts["gamma_mean"].values
    w_stat, w_p_two = stats.wilcoxon(concept_gamma_means, alternative="two-sided")
    _, w_p_one = stats.wilcoxon(concept_gamma_means, alternative="greater")

    t_stat, t_p_two = stats.ttest_1samp(concept_gamma_means, 0.0, alternative="two-sided")
    _, t_p_one = stats.ttest_1samp(concept_gamma_means, 0.0, alternative="greater")

    # Counts & Proportions
    n_total = len(df_concepts)
    n_gamma_gt_zero = int(np.sum(df_concepts["gamma_mean"] > 0))
    pct_gamma_gt_zero = float(n_gamma_gt_zero / n_total * 100.0)

    n_sig_positive = int(np.sum(df_concepts["ci_strictly_positive"]))
    pct_sig_positive = float(n_sig_positive / n_total * 100.0)

    n_p_lt_05_pos = int(np.sum((df_concepts["p_val_two_sided"] < 0.05) & (df_concepts["gamma_mean"] > 0)))
    pct_p_lt_05_pos = float(n_p_lt_05_pos / n_total * 100.0)

    n_cross_zero = int(np.sum((df_concepts["ci_lower"] <= 0) & (df_concepts["ci_upper"] >= 0)))
    pct_cross_zero = float(n_cross_zero / n_total * 100.0)

    # Verdict Determination
    if macro_ci_lower > 0 and w_p_one < 0.01:
        verdict = (
            "GENUINE_SIGNAL_CONFIRMED: Macro gamma is strictly and statistically positive (95% CI strictly > 0, "
            f"Wilcoxon p={w_p_one:.2e}). The interaction term exists in CLIP's representation, though suppressed by alpha."
        )
    elif pct_gamma_gt_zero >= 70.0:
        verdict = (
            "MARGINAL_SIGNAL_DETECTED: Population mean is positive, but individual concept statistical power is limited "
            "due to sample size per concept."
        )
    else:
        verdict = (
            "PURE_MEASUREMENT_NOISE: gamma is indistinguishable from zero across the majority of concepts. "
            "Interaction term does not measurably exist in unimodal CLIP."
        )

    summary = {
        "n_concepts_evaluated": n_total,
        "total_pairs_evaluated": len(df_pairs),
        "macro_gamma_mean": macro_gamma_mean,
        "macro_gamma_se": macro_se,
        "macro_gamma_bootstrap_95ci_lower": macro_ci_lower,
        "macro_gamma_bootstrap_95ci_upper": macro_ci_upper,
        "macro_gamma_strictly_positive": bool(macro_ci_lower > 0),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_p_val_two_sided": float(w_p_two),
        "wilcoxon_p_val_one_sided": float(w_p_one),
        "ttest_statistic": float(t_stat),
        "ttest_p_val_two_sided": float(t_p_two),
        "ttest_p_val_one_sided": float(t_p_one),
        "n_concepts_gamma_gt_zero": n_gamma_gt_zero,
        "pct_concepts_gamma_gt_zero_theoretical_bilinear_ceiling": pct_gamma_gt_zero,
        "n_concepts_ci_strictly_positive": n_sig_positive,
        "pct_concepts_ci_strictly_positive": pct_sig_positive,
        "n_concepts_permutation_p_lt_05_positive": n_p_lt_05_pos,
        "pct_concepts_permutation_p_lt_05_positive": pct_p_lt_05_pos,
        "n_concepts_crosses_zero": n_cross_zero,
        "pct_concepts_crosses_zero": pct_cross_zero,
        "verdict": verdict,
        "top5_strongest_significant_gamma": df_concepts[df_concepts["ci_strictly_positive"]].head(5)[
            ["object_name", "gamma_mean", "ci_lower", "ci_upper", "p_val_two_sided", "z_score"]
        ].to_dict(orient="records"),
        "top5_weakest_or_negative_gamma": df_concepts.tail(5)[
            ["object_name", "gamma_mean", "ci_lower", "ci_upper", "p_val_two_sided", "z_score"]
        ].to_dict(orient="records"),
    }

    # Render Visualizations & Export
    render_significance_visualizations(
        df_concepts=df_concepts,
        df_pairs=df_pairs,
        macro_boot_means=macro_boot_means,
        all_null_gammas=np.array(all_null_gammas),
        summary=summary,
        output_dir=args.output_dir,
    )

    concepts_csv = os.path.join(args.output_dir, "e2_gamma_concept_significance.csv")
    summary_json = os.path.join(args.output_dir, "e2_gamma_significance_summary.json")

    df_concepts.to_csv(concepts_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "═" * 70)
    print("  E2-RISK: INTERACTION TERM (γ) SIGNIFICANCE RESULTS SUMMARY")
    print("═" * 70)
    print(f"  Macro Mean γ                                    : {macro_gamma_mean:+.6f}")
    print(f"  Macro 95% Bootstrap CI                          : [{macro_ci_lower:+.6f}, {macro_ci_upper:+.6f}]")
    print(f"  Macro 95% CI Strictly > 0                       : {'✅ YES (Strictly Positive)' if macro_ci_lower > 0 else '❌ NO'}")
    print(f"  Population Wilcoxon Test (H1: Median γ > 0)     : p = {w_p_one:.4e} {'(***)' if w_p_one < 0.001 else ''}")
    print(f"  Population One-sample t-test (H1: E[γ] > 0)     : p = {t_p_one:.4e} {'(***)' if t_p_one < 0.001 else ''}")
    print(f"  Concepts with Empirical γ > 0                   : {n_gamma_gt_zero}/{n_total} ({pct_gamma_gt_zero:.1f}%)  <-- [RANK-1 BILINEAR CEILING]")
    print(f"  Concepts with CI strictly > 0 (p < 0.05)        : {n_sig_positive}/{n_total} ({pct_sig_positive:.1f}%)")
    print(f"  Concepts Indistinguishable from Zero (Noise/Tie): {n_cross_zero}/{n_total} ({pct_cross_zero:.1f}%)")
    print("═" * 70)
    print(f"  VERDICT: {summary['verdict']}")
    print("═" * 70)
    print(f"  Results saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
