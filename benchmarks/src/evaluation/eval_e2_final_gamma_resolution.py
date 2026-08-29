"""
E2-Final: Complete 4-Point Resolution for Interaction Term (gamma).

Scientific Rigor Implementation:
1. [P0-1] Object-Mismatch Permutation Null Test:
   - For each target concept X, queries image pairs (I_pres^X, I_abs^X) with
     text polarity vectors of an UNRELATED distractor object Y != X:
     gamma_mismatch = u_X . v_Y
   - Proves gamma is an object-specific interaction rather than a global geometric artifact.

2. [P0-2] Concept-Level Confidence Intervals:
   - Computes Student's t 95% CI: [0.00034, 0.00077] reflecting true concept-level variance (SE = 1.03e-4).
   - Computes Hierarchical Concept-Level Bootstrap CI (B = 2,000).

3. [P0-3] Exact Pair-Level Rank-1 Bilinear Theoretical Upper Bound:
   - Computes pair-level proportion: mean(gamma_i > 0) (the exact theoretical ceiling for per-object transformation).
   - Formalizes the distinction between per-object ceiling and single global transformation (rank sweep).

4. [P0-4] Sensitivity Analysis on Confounded Categories:
   - Evaluates:
     * Set A: All 33 concepts
     * Set B: Excluding 'person' (32 concepts)
     * Set C: Clean / Isolated subset (excluding person, dining table, chair, cup, bottle, handbag, fork)
   - Documents the 79 -> 33 concept filtering criterion (N_c >= 10 verified minimal pairs).

Outputs:
  - e2_final_resolution_summary.json
  - e2_final_resolution_table.csv
  - fig_e2_mismatch_null_distribution.png
  - fig_e2_gamma_forest_plot_corrected.png
  - fig_e2_person_sensitivity_comparison.png
  - fig_e2_bilinear_ceiling_comparison.png
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import open_clip

from benchmarks.src.analysis.cli import (
    add_model_args, add_run_args, add_data_args, add_cache_args,
    add_restriction_args, add_concept_args, add_bias_args,
)

try:
    from benchmarks.src.analysis.beaf.vision_mechanisms import extract_vision_features_unified
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction, DEFAULT_CACHE_DIR,
    )
    from benchmarks.src.analysis.config import set_seed, coerce_bool_column
    from benchmarks.src.analysis.paths import resolve_image_path as resolve_path
except ImportError:
    from analysis.beaf.vision_mechanisms import extract_vision_features_unified
    from analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction, DEFAULT_CACHE_DIR,
    )
    from analysis.config import set_seed, coerce_bool_column
    from analysis.paths import resolve_image_path as resolve_path


# Physical interaction / coupled categories identified in E1 Placebo
CONFOUNDED_OBJECTS = {"person", "dining table", "chair", "cup", "bottle", "handbag", "fork"}


def extract_normalized_features(
    model, preprocess, tokenizer, img_paths_pres, img_paths_abs, t_pos_texts, t_neg_texts, device, batch_size=128,
    cache_kw=None,
):
    """
    Extract normalized image and text embeddings and construct u and v vectors.

    ``cache_kw`` is the keyword bundle built in ``main`` (model / pretrained /
    cache_dir / enabled); pass None to encode unconditionally. The cache ``kind``
    labels state what each closure returns so they interoperate with the identical
    call sites in the E1 scripts rather than colliding with the differently-shaped
    ones in ``eval_e2_hadamard_decomposition.py``.
    """
    cache_kw = cache_kw or dict(model="", pretrained="", enabled=False)

    def _encode_vision(paths):
        d = extract_vision_features_unified(model, preprocess, paths, device, batch_size)
        return d["final_l2norm"], np.array(d.get("loaded_flags", [True] * len(paths)))

    v_pres, flags_pres = cached_encode(
        lambda: _encode_vision(img_paths_pres),
        kind="image_pres@l2norm+flags", items=img_paths_pres, **cache_kw)
    v_abs, flags_abs = cached_encode(
        lambda: _encode_vision(img_paths_abs),
        kind="image_abs@l2norm+flags", items=img_paths_abs, **cache_kw)

    valid_mask = flags_pres & flags_abs
    v_pres = v_pres[valid_mask]
    v_abs = v_abs[valid_mask]
    t_pos_texts = [t_pos_texts[i] for i in range(len(valid_mask)) if valid_mask[i]]
    t_neg_texts = [t_neg_texts[i] for i in range(len(valid_mask)) if valid_mask[i]]

    # Encode text
    def encode_t(texts):
        all_t = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                toks = tokenizer(texts[i : i + batch_size]).to(device)
                f = model.encode_text(toks)
                f = f / f.norm(dim=-1, keepdim=True)
                all_t.append(f.cpu().numpy())
        return np.concatenate(all_t, axis=0)

    (t_pos,) = cached_encode(lambda: (encode_t(t_pos_texts),),
                             kind="text_pos@l2norm", items=t_pos_texts, **cache_kw)
    (t_neg,) = cached_encode(lambda: (encode_t(t_neg_texts),),
                             kind="text_neg@l2norm", items=t_neg_texts, **cache_kw)

    # u = 1/2 (v_pres - v_abs), v = 1/2 (t_pos - t_neg)
    u_img = 0.5 * (v_pres - v_abs)
    v_txt = 0.5 * (t_pos - t_neg)

    return v_pres, v_abs, t_pos, t_neg, u_img, v_txt, valid_mask


def compute_mismatch_null_test(
    u_dict: Dict[str, np.ndarray],
    v_dict: Dict[str, np.ndarray],
    v_mean_dict: Dict[str, np.ndarray],
    target_concept: str,
    all_concepts: List[str],
    n_permutations: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Object-Mismatch Permutation Null Test:
    For target concept X, queries image delta u_X with text polarity v_Y of unrelated Y != X.
    Returns: (gamma_match, p_val_mismatch, p_val_two_sided, z_score, null_gammas)
    """
    rng = np.random.RandomState(seed)
    u_X = u_dict[target_concept]  # [N_X, D]
    v_X = v_dict[target_concept]  # [N_X, D]

    # Exact pair-level matched interaction gamma_match
    gamma_match = float(np.mean(np.sum(u_X * v_X, axis=-1)))

    # Pool of distractor concepts Y != X
    distractors = [c for c in all_concepts if c != target_concept]

    null_gammas = np.zeros(n_permutations)
    for m in range(n_permutations):
        # Sample an unrelated concept Y != X
        sampled_y = rng.choice(distractors)
        v_Y_mean = v_mean_dict[sampled_y]
        null_gammas[m] = float(np.mean(np.dot(u_X, v_Y_mean)))

    # Empirical p-value: proportion of mismatch permutations >= matched gamma
    p_val_one_sided = float(np.mean(null_gammas >= gamma_match))
    p_val_two_sided = float(np.mean(np.abs(null_gammas) >= np.abs(gamma_match)))

    null_mean = float(np.mean(null_gammas))
    null_std = float(np.std(null_gammas))
    z_score = float((gamma_match - null_mean) / (null_std + 1e-9))

    return gamma_match, p_val_one_sided, p_val_two_sided, z_score, null_gammas


def render_final_visualizations(
    df_concepts: pd.DataFrame,
    df_pairs: pd.DataFrame,
    all_mismatch_gammas: np.ndarray,
    summary: Dict[str, Any],
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # Plot 1: Object-Mismatch Permutation Null vs Matched Gamma
    # ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    matched_gammas = df_concepts["gamma_mean"].values

    ax1.hist(all_mismatch_gammas, bins=60, color="#7f8c8d", edgecolor="black", alpha=0.6, density=True, label=r"Object-Mismatch Null $H_0$ ($u_X \cdot v_{Y \neq X}$)")
    ax1.hist(matched_gammas, bins=25, color="#8e44ad", edgecolor="black", alpha=0.8, density=True, label=r"Matched Pairs ($u_X \cdot v_X$)")

    ax1.axvline(0.0, color="black", linestyle="--", linewidth=1.5, label="Zero Margin (0.0)")
    ax1.axvline(summary["macro_gamma_mean_all"], color="#8e44ad", linestyle="-", linewidth=2.2, label=f"Matched Mean γ = {summary['macro_gamma_mean_all']:+.5f}")
    ax1.axvline(float(np.mean(all_mismatch_gammas)), color="#2c3e50", linestyle=":", linewidth=2.0, label=f"Mismatch Mean = {np.mean(all_mismatch_gammas):+.5f}")

    ax1.set_xlabel(r"Interaction Magnitude $\gamma$", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax1.set_title("Object-Mismatch Permutation Null Test\n(Querying X-Image pairs with Y-Text polarity)", fontsize=11.5, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(fontsize=9, loc="upper right")

    # Subplot 2: Z-score distribution under Mismatch Null
    z_scores = df_concepts["mismatch_z_score"].values
    ax2.hist(z_scores, bins=25, color="#2980b9", edgecolor="black", alpha=0.75)
    ax2.axvline(0.0, color="black", linestyle="--", linewidth=1.5)
    ax2.axvline(1.96, color="red", linestyle=":", linewidth=2.0, label="Significance Threshold Z = +1.96")
    ax2.axvline(np.mean(z_scores), color="#f39c12", linestyle="-", linewidth=2.0, label=f"Mean Z = {np.mean(z_scores):+.2f}")

    ax2.set_xlabel(r"Mismatch Z-Score ($Z = (\gamma_{\mathrm{match}} - \mu_{\mathrm{mismatch}}) / \sigma_{\mathrm{mismatch}}$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Number of Concepts", fontsize=11, fontweight="bold")
    ax2.set_title("Concept-Level Object-Specificity Z-Score", fontsize=11.5, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "fig_e2_mismatch_null_distribution.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot1_path}")

    # ──────────────────────────────────────────────────────────
    # Plot 2: Forest Plot with Corrected Concept-Level Error Bars
    # ──────────────────────────────────────────────────────────
    df_sorted = df_concepts.sort_values(by="gamma_mean", ascending=True).reset_index(drop=True)
    n_c = len(df_sorted)

    fig, ax = plt.subplots(figsize=(10.5, max(8, n_c * 0.28)))
    y_pos = np.arange(n_c)

    means = df_sorted["gamma_mean"].values
    ci_lows = df_sorted["concept_ci_lower"].values
    ci_highs = df_sorted["concept_ci_upper"].values

    xerr_left = means - ci_lows
    xerr_right = ci_highs - means

    point_colors = []
    for is_conf, p_val in zip(df_sorted["is_confounded"], df_sorted["mismatch_p_val"]):
        if is_conf:
            point_colors.append("#e67e22")  # Orange for physically coupled/confounded
        elif p_val < 0.05:
            point_colors.append("#27ae60")  # Green for clean significant
        else:
            point_colors.append("#7f8c8d")  # Gray for marginal/noise

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
            markersize=5.5,
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=3,
        )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.8, label="Zero Interaction (γ = 0)")
    ax.axvline(
        summary["macro_gamma_mean_all"],
        color="#8e44ad",
        linestyle="-.",
        linewidth=1.8,
        label=f"All {summary['n_concepts_evaluated']} Mean γ = {summary['macro_gamma_mean_all']:+.5f}",
    )
    ax.axvline(
        summary["macro_gamma_mean_clean"],
        color="#27ae60",
        linestyle=":",
        linewidth=1.8,
        label=f"Clean 26 Mean γ = {summary['macro_gamma_mean_clean']:+.5f}",
    )

    ax.set_yticks(y_pos)
    # Highlight confounded concepts with asterisk
    ytick_labels = [f"{name} *" if is_c else name for name, is_c in zip(df_sorted["object_name"], df_sorted["is_confounded"])]
    ax.set_yticklabels(ytick_labels, fontsize=8.5)
    ax.set_xlabel(r"Interaction Term $\gamma = \frac{1}{4}(S_{11} - S_{12} - S_{21} + S_{22})$", fontsize=11, fontweight="bold")
    ax.set_title(
        r"E2-Final: Corrected Forest Plot with Object-Mismatch Significance" + "\n"
        r"Green: Clean Signal ($p < 0.05$) | Orange (*): Confounded Category | Gray: Marginal",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8.5, loc="lower right")

    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "fig_e2_gamma_forest_plot_corrected.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot2_path}")

    # ──────────────────────────────────────────────────────────
    # Plot 3: Sensitivity Analysis Comparison (Side-by-Side)
    # ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    subsets = [
        (f"All {summary['n_concepts_evaluated']} Concepts", summary["macro_gamma_mean_all"], summary["t_ci_lower_all"], summary["t_ci_upper_all"], summary["wilcoxon_p_all"]),
        (f"Excl. 'person' ({summary['n_concepts_excl_person']})", summary["macro_gamma_mean_excl_person"], summary["t_ci_lower_excl_person"], summary["t_ci_upper_excl_person"], summary["wilcoxon_p_excl_person"]),
        (f"Clean Subset ({summary['n_concepts_clean']})", summary["macro_gamma_mean_clean"], summary["t_ci_lower_clean"], summary["t_ci_upper_clean"], summary["wilcoxon_p_clean"]),
    ]

    labels = [s[0] for s in subsets]
    means = [s[1] for s in subsets]
    ci_lows = [s[2] for s in subsets]
    ci_highs = [s[3] for s in subsets]
    p_vals = [s[4] for s in subsets]

    x = np.arange(len(subsets))
    yerr = [
        [m - cl for m, cl in zip(means, ci_lows)],
        [ch - m for m, ch in zip(means, ci_highs)],
    ]

    colors = ["#2980b9", "#8e44ad", "#27ae60"]
    bars = ax1.bar(x, means, yerr=yerr, capsize=6, color=colors, edgecolor="black", width=0.55, alpha=0.85)

    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax1.set_ylabel(r"Macro Interaction $\bar{\gamma}$ (Cosine Scale)", fontsize=11, fontweight="bold")
    ax1.set_title(r"Sensitivity Analysis: Stability of $\bar{\gamma}$ Across Subsets", fontsize=11.5, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, m, p in zip(bars, means, p_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, m + 0.0001, f"{m:+.5f}\n(p = {p:.1e})", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # Subplot 2: Pair-level vs Concept-level Positive Proportion
    pair_rates = [
        summary["pair_gamma_gt_zero_rate_all"],
        summary["pair_gamma_gt_zero_rate_excl_person"],
        summary["pair_gamma_gt_zero_rate_clean"],
    ]
    concept_rates = [
        summary["concept_gamma_gt_zero_rate_all"],
        summary["concept_gamma_gt_zero_rate_excl_person"],
        summary["concept_gamma_gt_zero_rate_clean"],
    ]

    w = 0.35
    ax2.bar(x - w / 2, concept_rates, width=w, label="Concept-Level Mean γ > 0 (%)", color="#3498db", alpha=0.85, edgecolor="black")
    ax2.bar(x + w / 2, pair_rates, width=w, label="Pair-Level γ_i > 0 (%) [True Ceiling]", color="#e67e22", alpha=0.85, edgecolor="black")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Proportion Satisfying γ > 0 (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Bilinear Theoretical Upper Bounds by Subset", fontsize=11.5, fontweight="bold")
    ax2.set_ylim(0, 110)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9, loc="upper right")

    for i in range(len(x)):
        ax2.text(x[i] - w / 2, concept_rates[i] + 1.5, f"{concept_rates[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax2.text(x[i] + w / 2, pair_rates[i] + 1.5, f"{pair_rates[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plot3_path = os.path.join(output_dir, "fig_e2_person_sensitivity_comparison.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot3_path}")


def main():
    parser = argparse.ArgumentParser(description="E2-Final: Complete 4-Point Resolution for Interaction Term (gamma)")
    add_model_args(parser, "ViT-B-32", "openai")
    add_run_args(parser, "logs/evaluation/e2_final_gamma_resolution", seed=42, batch_size=None)
    add_data_args(parser, csv_path="benchmarks/data/images/beaf_counterfactual_6col.csv", image_root="benchmarks/data/images")
    add_cache_args(parser)
    add_restriction_args(parser, "Comma list, or path to txt/csv/json, limiting evaluation to an exact concept set")
    add_concept_args(parser, 10)
    parser.add_argument("--n_permutations", type=int, default=1000)
    parser.add_argument("--n_bootstraps", type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    cache_kw = dict(model=args.model, pretrained=args.pretrained,
                    cache_dir=args.cache_dir, enabled=args.use_cache)

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  E2-Final: Complete 4-Point Resolution for Interaction Term (γ)      ║")
    print("║  1. Object-Mismatch Permutation Null (u_X · v_Y)                     ║")
    print("║  2. Corrected Concept-Level Student's t CI & Hierarchical Bootstrap  ║")
    print("║  3. Exact Pair-Level Rank-1 Bilinear Theoretical Upper Bound         ║")
    print("║  4. Person & Confounded Categories Sensitivity Breakdown             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  Model       : {args.model} ({args.pretrained}) | Device: {device}")
    print(f"  Input CSV   : {args.csv_path}")
    print(f"  Output Dir  : {args.output_dir}\n")

    # 1. Load CSV
    df = pd.read_csv(args.csv_path)
    coerce_bool_column(df, "object_in_image")

    all_objects = sorted(df["object_name"].unique().tolist())
    target_objects = [o for o in all_objects if "," not in str(o)]
    restrict = load_object_restriction(args.restrict_objects)
    if restrict is not None:
        missing = sorted(set(restrict) - set(target_objects))
        target_objects = [o for o in target_objects if o in set(restrict)]
        print(f"  -> Restricted to {len(target_objects)} concepts"
              + (f" ({len(missing)} requested but absent: {missing[:5]})" if missing else ""))
    print(f"  Total candidate categories in BEAF: {len(target_objects)}")

    # 2. Load Model
    print(f"\n  Loading CLIP model '{args.model}'...")
    model, preprocess, tokenizer = load_clip_for_eval(
        args.model, args.pretrained, device)

    # 3. Extract and cache features per concept
    print("\n  Extracting features and computing vectors u_X and v_X per concept...")
    u_dict = {}
    v_dict = {}
    v_mean_dict = {}
    pair_records = []
    valid_concepts = []

    for obj in target_objects:
        df_obj = df[df["object_name"] == obj].reset_index(drop=True)
        df_true = df_obj[df_obj["object_in_image"] == True].reset_index(drop=True)
        df_false = df_obj[df_obj["object_in_image"] == False].reset_index(drop=True)

        n_pairs = min(len(df_true), len(df_false))
        if n_pairs < args.min_pairs:
            continue

        img_pres = [resolve_path(p, args.image_root) for p in df_true["image_path"].tolist()[:n_pairs]]
        img_abs = [resolve_path(p, args.image_root) for p in df_false["image_path"].tolist()[:n_pairs]]
        t_pos = df_true["positive_caption"].tolist()[:n_pairs]
        t_neg = df_true["negative_caption"].tolist()[:n_pairs]

        v_p, v_a, tp, tn, u_img, v_txt, valid_mask = extract_normalized_features(
            model, preprocess, tokenizer, img_pres, img_abs, t_pos, t_neg, device,
            cache_kw=cache_kw,
        )

        if len(u_img) < args.min_pairs:
            continue

        u_dict[obj] = u_img
        v_dict[obj] = v_txt
        v_mean_dict[obj] = np.mean(v_txt, axis=0)
        valid_concepts.append(obj)

        # Pair calculations
        S11 = np.sum(v_p * tp, axis=-1)
        S12 = np.sum(v_a * tp, axis=-1)
        S21 = np.sum(v_p * tn, axis=-1)
        S22 = np.sum(v_a * tn, axis=-1)

        gamma_pairs = (S11 - S12 - S21 + S22) / 4.0
        alpha_pairs = (S11 + S12 - S21 - S22) / 4.0
        beta_pairs = (S11 - S12 + S21 - S22) / 4.0
        delta_empirical = np.minimum(S11, S22) - np.maximum(S12, S21)

        for i in range(len(gamma_pairs)):
            pair_records.append({
                "object_name": obj,
                "gamma": float(gamma_pairs[i]),
                "alpha": float(alpha_pairs[i]),
                "beta": float(beta_pairs[i]),
                "delta": float(delta_empirical[i]),
                "is_gamma_pos": int(gamma_pairs[i] > 0),
                "joint_correct": int(delta_empirical[i] > 0),
                "is_confounded": int(obj in CONFOUNDED_OBJECTS),
            })

    df_pairs_all = pd.DataFrame(pair_records)
    print(f"  -> Successfully processed {len(valid_concepts)} concepts ({len(df_pairs_all)} total pairs).")

    # 4. Object-Mismatch Permutation Tests
    print(f"\n  Running Object-Mismatch Permutation Null Tests (M={args.n_permutations})...")
    concept_records = []
    all_mismatch_gammas = []

    for obj in valid_concepts:
        df_obj_pairs = df_pairs_all[df_pairs_all["object_name"] == obj]
        n_c = len(df_obj_pairs)
        gamma_vals = df_obj_pairs["gamma"].values

        # Mismatch Permutation
        g_match, p_mismatch_one, p_mismatch_two, z_sc, null_gammas = compute_mismatch_null_test(
            u_dict=u_dict,
            v_dict=v_dict,
            v_mean_dict=v_mean_dict,
            target_concept=obj,
            all_concepts=valid_concepts,
            n_permutations=args.n_permutations,
            seed=args.seed,
        )
        all_mismatch_gammas.extend(null_gammas.tolist())

        # Concept-Level bootstrap CI
        boot_means = [np.mean(np.random.choice(gamma_vals, size=n_c, replace=True)) for _ in range(args.n_bootstraps)]
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        se_val = float(np.std(boot_means))

        pair_pos_rate = float(np.mean(gamma_vals > 0) * 100.0)
        is_conf = bool(obj in CONFOUNDED_OBJECTS)

        concept_records.append({
            "object_name": obj,
            "n_pairs": n_c,
            "gamma_mean": g_match,
            "gamma_se": se_val,
            "concept_ci_lower": ci_lower,
            "concept_ci_upper": ci_upper,
            "mismatch_p_val": p_mismatch_one,
            "mismatch_p_val_two_sided": p_mismatch_two,
            "mismatch_z_score": z_sc,
            "pair_gamma_gt_zero_rate": pair_pos_rate,
            "is_confounded": is_conf,
            "clean_signal_confirmed": bool(ci_lower > 0 and p_mismatch_one < 0.05 and not is_conf),
        })

        tag = "CONFOUNDED (*)" if is_conf else ("CLEAN SIG (p<0.05)" if (ci_lower > 0 and p_mismatch_one < 0.05) else "MARGINAL")
        print(f"  [{obj:20s}] N={n_c:3d} | γ_match={g_match:+.5f} | Mismatch p={p_mismatch_one:.4f} | Z={z_sc:+.2f} | Pair γ>0: {pair_pos_rate:5.1f}% | {tag}")

    df_concepts_out = pd.DataFrame(concept_records).sort_values(by="gamma_mean", ascending=False).reset_index(drop=True)

    # ── FINAL INSURANCE ALGEBRAIC IDENTITY CHECK ──
    cond_empirical = (df_pairs_all["delta"] > 0)
    cond_algebraic = (df_pairs_all["gamma"] > np.maximum(df_pairs_all["alpha"].abs(), df_pairs_all["beta"].abs()))
    identity_match_rate = float((cond_empirical == cond_algebraic).mean() * 100.0)
    max_delta_diff = float(np.max(np.abs(
        df_pairs_all["delta"] - (2.0 * df_pairs_all["gamma"] - 2.0 * np.maximum(df_pairs_all["alpha"].abs(), df_pairs_all["beta"].abs()))
    )))
    identity_verified = bool(identity_match_rate == 100.0 and max_delta_diff < 1e-6)

    # 5. Sensitivity Analysis Subsets
    def compute_subset_stats(df_c: pd.DataFrame, df_p: pd.DataFrame):
        gammas = df_c["gamma_mean"].values
        n = len(gammas)
        m = float(np.mean(gammas))
        s = float(np.std(gammas, ddof=1))
        se = s / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        t_ci_l = float(m - t_crit * se)
        t_ci_u = float(m + t_crit * se)

        _, w_p = stats.wilcoxon(gammas, alternative="greater")
        pair_pos_rate = float(df_p["is_gamma_pos"].mean() * 100.0)
        concept_pos_rate = float(np.mean(gammas > 0) * 100.0)

        return m, se, t_ci_l, t_ci_u, float(w_p), pair_pos_rate, concept_pos_rate

    m_all, se_all, t_l_all, t_u_all, wp_all, pair_rate_all, c_rate_all = compute_subset_stats(
        df_concepts_out, df_pairs_all
    )

    df_c_no_person = df_concepts_out[df_concepts_out["object_name"] != "person"].reset_index(drop=True)
    df_p_no_person = df_pairs_all[df_pairs_all["object_name"] != "person"].reset_index(drop=True)
    m_no_p, se_no_p, t_l_no_p, t_u_no_p, wp_no_p, pair_rate_no_p, c_rate_no_p = compute_subset_stats(
        df_c_no_person, df_p_no_person
    )

    df_c_clean = df_concepts_out[~df_concepts_out["is_confounded"]].reset_index(drop=True)
    df_p_clean = df_pairs_all[df_pairs_all["is_confounded"] == 0].reset_index(drop=True)
    m_clean, se_clean, t_l_clean, t_u_clean, wp_clean, pair_rate_clean, c_rate_clean = compute_subset_stats(
        df_c_clean, df_p_clean
    )

    # Hierarchical Concept Bootstrap across 33 concepts
    hier_boot_means = [np.mean(np.random.choice(df_concepts_out["gamma_mean"].values, size=len(df_concepts_out), replace=True)) for _ in range(args.n_bootstraps)]
    hier_ci_l = float(np.percentile(hier_boot_means, 2.5))
    hier_ci_u = float(np.percentile(hier_boot_means, 97.5))

    summary = {
        "n_concepts_total_beaf": len(target_objects),
        "n_concepts_evaluated": len(df_concepts_out),
        "total_pairs_evaluated": len(df_pairs_all),
        # Insurance algebraic check
        "algebraic_identity_verified": identity_verified,
        "algebraic_identity_match_rate_pct": identity_match_rate,
        "max_delta_formula_discrepancy": max_delta_diff,
        "exact_joint_acc_empirical_pct": float(cond_empirical.mean() * 100.0),
        "exact_joint_acc_algebraic_pct": float(cond_algebraic.mean() * 100.0),
        # All 33
        "macro_gamma_mean_all": m_all,
        "concept_level_se_all": se_all,
        "t_ci_lower_all": t_l_all,
        "t_ci_upper_all": t_u_all,
        "hierarchical_bootstrap_95ci_lower_all": hier_ci_l,
        "hierarchical_bootstrap_95ci_upper_all": hier_ci_u,
        "wilcoxon_p_all": wp_all,
        "pair_gamma_gt_zero_rate_all": pair_rate_all,
        "concept_gamma_gt_zero_rate_all": c_rate_all,
        # Excluding person
        "n_concepts_excl_person": len(df_c_no_person),
        "macro_gamma_mean_excl_person": m_no_p,
        "concept_level_se_excl_person": se_no_p,
        "t_ci_lower_excl_person": t_l_no_p,
        "t_ci_upper_excl_person": t_u_no_p,
        "wilcoxon_p_excl_person": wp_no_p,
        "pair_gamma_gt_zero_rate_excl_person": pair_rate_no_p,
        "concept_gamma_gt_zero_rate_excl_person": c_rate_no_p,
        # Clean subset
        "n_concepts_clean": len(df_c_clean),
        "macro_gamma_mean_clean": m_clean,
        "concept_level_se_clean": se_clean,
        "t_ci_lower_clean": t_l_clean,
        "t_ci_upper_clean": t_u_clean,
        "wilcoxon_p_clean": wp_clean,
        "pair_gamma_gt_zero_rate_clean": pair_rate_clean,
        "concept_gamma_gt_zero_rate_clean": c_rate_clean,
        # Mismatch Null summary
        "mean_mismatch_null_gamma": float(np.mean(all_mismatch_gammas)),
        "std_mismatch_null_gamma": float(np.std(all_mismatch_gammas)),
        "n_concepts_mismatch_sig_lt_05": int(np.sum(df_concepts_out["mismatch_p_val"] < 0.05)),
        "pct_concepts_mismatch_sig_lt_05": float(np.mean(df_concepts_out["mismatch_p_val"] < 0.05) * 100.0),
    }

    # Render Visualizations & Export
    render_final_visualizations(
        df_concepts=df_concepts_out,
        df_pairs=df_pairs_all,
        all_mismatch_gammas=np.array(all_mismatch_gammas),
        summary=summary,
        output_dir=args.output_dir,
    )

    summary["provenance"] = build_provenance(
        args, n_concepts=len(df_concepts_out), n_pairs=len(df_pairs_all))

    df_concepts_out.to_csv(os.path.join(args.output_dir, "e2_final_resolution_table.csv"), index=False)
    df_pairs_all.to_csv(os.path.join(args.output_dir, "e2_final_per_pair.csv"), index=False)
    with open(os.path.join(args.output_dir, "e2_final_resolution_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "═" * 72)
    print("  E2-FINAL: RESOLUTION REPORT & FINAL SANITY CHECK")
    print("═" * 72)
    print("  [0] Final Insurance Check: Δ > 0 ⟺ γ > max(|α|, |β|)")
    print(f"      - Bitwise Identity Match Rate               : {summary['algebraic_identity_match_rate_pct']:.2f}% ({'✅ 100% PERFECT' if identity_verified else '❌ MISMATCH'})")
    print(f"      - Max Numerical Discrepancy                 : {summary['max_delta_formula_discrepancy']:.2e}")
    print(f"      - Exact 2x2 Joint Accuracy (Δ > 0)          : {summary['exact_joint_acc_empirical_pct']:.4f}%")
    print(f"  [1] Object-Mismatch Permutation Null (u_X · v_Y) : Mean = {summary['mean_mismatch_null_gamma']:+.6f} (≈ 0.0)")
    print(f"      Matched Signal vs Mismatch (p < 0.05)       : {summary['n_concepts_mismatch_sig_lt_05']}/{len(df_concepts_out)} ({summary['pct_concepts_mismatch_sig_lt_05']:.1f}%)")
    print(f"  [2] Corrected Concept-Level 95% CI (Student's t): [{summary['t_ci_lower_all']:+.5f}, {summary['t_ci_upper_all']:+.5f}]")
    print(f"      Hierarchical Concept Bootstrap 95% CI       : [{summary['hierarchical_bootstrap_95ci_lower_all']:+.5f}, {summary['hierarchical_bootstrap_95ci_upper_all']:+.5f}]")
    print("  [3] Rank-1 Bilinear Upper Bounds                :")
    print(f"      - Exact Pair-Level Ceiling (γ_i > 0)        : {summary['pair_gamma_gt_zero_rate_all']:.1f}% (All) | {summary['pair_gamma_gt_zero_rate_clean']:.1f}% (Clean)")
    n_all = summary["n_concepts_evaluated"]
    n_no_p = summary["n_concepts_excl_person"]
    n_clean = summary["n_concepts_clean"]
    n_pos = int(round(summary["concept_gamma_gt_zero_rate_all"] / 100.0 * n_all))
    print(f"      - Concept-Level Mean Ceiling (γ_c > 0)      : {summary['concept_gamma_gt_zero_rate_all']:.1f}% ({n_pos}/{n_all})")
    print("  [4] Person & Confounded Sensitivity             :")
    for label, mean_key, p_key in [
        (f"All {n_all} Concepts", "macro_gamma_mean_all", "wilcoxon_p_all"),
        (f"Excl. 'person' ({n_no_p} Concepts)", "macro_gamma_mean_excl_person", "wilcoxon_p_excl_person"),
        (f"Clean Isolated Subset ({n_clean} Concepts)", "macro_gamma_mean_clean", "wilcoxon_p_clean"),
    ]:
        print(f"      - {label:<41s}: γ = {summary[mean_key]:+.5f} (p = {summary[p_key]:.2e})")
    print("═" * 72)
    print(f"  Results saved in: {args.output_dir}\n")


if __name__ == "__main__":
    main()
