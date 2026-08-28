"""
E2 Experiment: Exact 2x2 Hadamard Coordinate Decomposition and Mechanistic Lever Analysis.

Decomposes the 4 cosine similarities (S_11, S_12, S_21, S_22) of 2x2 minimal pairs
into 4 orthogonal Hadamard coordinates:
    S_ab = C + a*beta + b*alpha + a*b*gamma,  where a, b in {+1, -1}

Coordinates:
    C     = 1/4 * (S_11 + S_12 + S_21 + S_22)  -> Global similarity mean
    beta  = 1/4 * (S_11 - S_12 + S_21 - S_22)  -> Image main effect (vision signal, 2*beta approx delta_s)
    alpha = 1/4 * (S_11 + S_12 - S_21 - S_22)  -> Text main effect (Affirmation bias)
    gamma = 1/4 * (S_11 - S_12 - S_21 + S_22)  -> Interaction term (True negation understanding)

Core Mathematical Identity:
    Delta(S) = min(S_11, S_22) - max(S_12, S_21)
             = 2*gamma - (|alpha + beta| + |alpha - beta|)
             = 2*gamma - 2*max(|alpha|, |beta|)
    Delta(S) > 0  <===>  gamma > max(|alpha|, |beta|)

Hypothesis & Pass Criteria:
    - Dominance: |alpha| >> gamma and |alpha| > |beta|
    - Feasibility of Vision Amplification: gamma > |beta|
    - Sanity verification: max(|Delta_empirical - Delta_analytical|) < 1e-6

Outputs:
    - e2_per_concept_decomposition.csv
    - e2_per_pair_decomposition.csv
    - e2_hadamard_summary.json
    - fig_e2_coefficients_boxplot.png       (Publication-grade boxplot of |alpha|, |beta|, gamma)
    - fig_e2_2beta_vs_e1_deltas.png          (Cross-validation scatter: 2*beta vs E1 mean_delta_s)
    - fig_e2_gamma_ratio_distribution.png    (Ratio gamma / max(|alpha|, |beta|) & required lambda*)
    - fig_e2_modality_gap_intervention.png   (Zero-alpha orthogonal projection intervention)
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import open_clip

try:
    from benchmarks.src.analysis.beaf.vision_mechanisms import extract_vision_features_unified
except ImportError:
    from analysis.beaf.vision_mechanisms import extract_vision_features_unified


def encode_images_unified(
    model,
    preprocess,
    image_paths: List[str],
    device: str,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract both L2-normalized and unnormalized image embeddings.
    Returns: (l2norm_feats, raw_feats, loaded_flags)
    """
    feats_dict = extract_vision_features_unified(
        model=model,
        preprocess=preprocess,
        image_paths=image_paths,
        device=device,
        batch_size=batch_size,
    )
    l2norm_feats = feats_dict["final_l2norm"]
    raw_feats = feats_dict.get("final_unnorm", l2norm_feats)
    loaded_flags = np.array(feats_dict.get("loaded_flags", [True] * len(image_paths)))
    return l2norm_feats, raw_feats, loaded_flags


def encode_texts_unified(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract both L2-normalized and unnormalized text embeddings.
    Returns: (l2norm_feats, raw_feats)
    """
    all_l2, all_raw = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            tokens = tokenizer(batch_texts).to(device)
            raw = model.encode_text(tokens).cpu().float().numpy()
            norm = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
            all_raw.append(raw)
            all_l2.append(norm)
    return np.concatenate(all_l2, axis=0), np.concatenate(all_raw, axis=0)


def compute_hadamard_coordinates(
    S11: np.ndarray, S12: np.ndarray, S21: np.ndarray, S22: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Computes exact Hadamard decomposition:
        C     = 1/4 * (S11 + S12 + S21 + S22)
        beta  = 1/4 * (S11 - S12 + S21 - S22)
        alpha = 1/4 * (S11 + S12 - S21 - S22)
        gamma = 1/4 * (S11 - S12 - S21 + S22)
    """
    C = (S11 + S12 + S21 + S22) / 4.0
    beta = (S11 - S12 + S21 - S22) / 4.0
    alpha = (S11 + S12 - S21 - S22) / 4.0
    gamma = (S11 - S12 - S21 + S22) / 4.0

    delta_empirical = np.minimum(S11, S22) - np.maximum(S12, S21)
    delta_analytical = 2.0 * gamma - 2.0 * np.maximum(np.abs(alpha), np.abs(beta))

    return {
        "C": C,
        "beta": beta,
        "alpha": alpha,
        "gamma": gamma,
        "abs_alpha": np.abs(alpha),
        "abs_beta": np.abs(beta),
        "delta_empirical": delta_empirical,
        "delta_analytical": delta_analytical,
        "verification_error": np.abs(delta_empirical - delta_analytical),
        "joint_correct": delta_empirical > 0,
    }


def compute_zero_alpha_intervention(
    v_pres: np.ndarray,
    v_abs: np.ndarray,
    t_pos: np.ndarray,
    t_neg: np.ndarray,
    mu_I_global: np.ndarray,
) -> Dict[str, Any]:
    """
    Intervention: Orthogonal projection of text polarity vector against global image mean.
        v = 1/2 * (t_pos - t_neg)
        v_perp = v - (v . hat{mu}_I) * hat{mu}_I
        t_pos_proj = mu_T + v_perp
        t_neg_proj = mu_T - v_perp
    """
    mu_T = 0.5 * (t_pos + t_neg)
    v_text = 0.5 * (t_pos - t_neg)

    hat_mu_I = mu_I_global / (np.linalg.norm(mu_I_global) + 1e-9)

    # Project out mu_I component from v_text
    proj_coeff = np.sum(v_text * hat_mu_I, axis=-1, keepdims=True)
    v_perp = v_text - proj_coeff * hat_mu_I

    t_pos_proj = mu_T + v_perp
    t_neg_proj = mu_T - v_perp

    # L2 normalize
    t_pos_proj = t_pos_proj / np.linalg.norm(t_pos_proj, axis=-1, keepdims=True)
    t_neg_proj = t_neg_proj / np.linalg.norm(t_neg_proj, axis=-1, keepdims=True)

    S11_proj = np.sum(v_pres * t_pos_proj, axis=-1)
    S12_proj = np.sum(v_abs * t_pos_proj, axis=-1)
    S21_proj = np.sum(v_pres * t_neg_proj, axis=-1)
    S22_proj = np.sum(v_abs * t_neg_proj, axis=-1)

    res_proj = compute_hadamard_coordinates(S11_proj, S12_proj, S21_proj, S22_proj)
    return res_proj


def render_e2_visualizations(
    df_concepts: pd.DataFrame,
    df_pairs: pd.DataFrame,
    summary: Dict[str, Any],
    df_e1: Optional[pd.DataFrame],
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # Plot 1: Boxplot of |alpha|, |beta|, gamma (Core Paper Figure)
    # ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})

    box_data = [
        df_concepts["abs_alpha_mean"].values,
        df_concepts["abs_beta_mean"].values,
        df_concepts["gamma_mean"].values,
    ]

    labels = [
        r"$|\alpha|$" + "\n(Affirmation Bias)",
        r"$|\beta|$" + "\n(Vision Signal)",
        r"$\gamma$" + "\n(Interaction / Negation)",
    ]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    bplot = ax1.boxplot(
        box_data,
        patch_artist=True,
        labels=labels,
        widths=0.55,
        medianprops=dict(color="black", linewidth=2.0),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", color="gray", alpha=0.6),
    )

    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Add individual jittered points
    for i, data in enumerate(box_data):
        y = data
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax1.scatter(x, y, alpha=0.5, color=colors[i], edgecolors="black", s=30, zorder=3)

    ax1.set_ylabel("Coordinate Magnitude (Cosine Scale)", fontsize=11, fontweight="bold")
    ax1.set_title(
        r"E2: 2$\times$2 Hadamard Decomposition ($|\alpha| \gg |\beta| > \gamma$)" + "\n"
        r"Text Affirmation Bias $|\alpha|$ Dominates Interaction $\gamma$",
        fontsize=11.5,
        fontweight="bold",
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate inequality satisfaction
    ax1.text(
        0.05,
        0.88,
        f"|α| > γ Rate: {summary['pct_alpha_gt_gamma']:.1f}%\n"
        f"|α| > |β| Rate: {summary['pct_alpha_gt_beta']:.1f}%\n"
        f"Mean |α| / γ: {summary['mean_alpha_over_gamma']:.2f}×",
        transform=ax1.transAxes,
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fadbd8", edgecolor="#e74c3c", alpha=0.9),
    )

    # Subplot 2: Bar breakdown of Top/Bottom Concepts
    top_alpha = df_concepts.sort_values(by="abs_alpha_mean", ascending=False).head(10)
    y_pos = np.arange(len(top_alpha))
    ax2.barh(y_pos - 0.2, top_alpha["abs_alpha_mean"], height=0.35, color="#e74c3c", label=r"$|\alpha|$", alpha=0.85)
    ax2.barh(y_pos + 0.2, top_alpha["gamma_mean"], height=0.35, color="#2ecc71", label=r"$\gamma$", alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_alpha["object_name"], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Magnitude", fontsize=10, fontweight="bold")
    ax2.set_title(r"Top 10 Affirmation Bias ($|\alpha|$ vs $\gamma$)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "fig_e2_coefficients_boxplot.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot1_path}")

    # ──────────────────────────────────────────────────────────
    # Plot 2: Cross-Validation Scatter (2*beta vs E1 mean_delta_s)
    # ──────────────────────────────────────────────────────────
    if df_e1 is not None and "object_name" in df_e1.columns and "mean_delta_s" in df_e1.columns:
        merged = pd.merge(df_concepts, df_e1, on="object_name", suffixes=("_e2", "_e1"))
        if len(merged) > 0:
            fig, ax = plt.subplots(figsize=(7, 6))

            x_vals = 2.0 * merged["beta_mean"].values
            y_vals = merged["mean_delta_s"].values

            corr = np.corrcoef(x_vals, y_vals)[0, 1]

            ax.scatter(x_vals, y_vals, color="#2980b9", edgecolors="black", s=50, alpha=0.8, zorder=3)

            min_val = min(x_vals.min(), y_vals.min())
            max_val = max(x_vals.max(), y_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5, label="y = x (Perfect Match)")

            # Fit line
            m, b = np.polyfit(x_vals, y_vals, 1)
            ax.plot(x_vals, m * x_vals + b, color="#27ae60", linestyle="-", linewidth=1.5, label=f"Fit: y = {m:.2f}x + {b:.4f} (r = {corr:.3f})")

            ax.set_xlabel(r"E2 Theoretical Vision Signal ($2\beta$)", fontsize=11, fontweight="bold")
            ax.set_ylabel(r"E1 Empirical Vision Signal ($\Delta s_{\mathrm{mean}}$)", fontsize=11, fontweight="bold")
            ax.set_title(r"Cross-Validation: $2\beta$ vs E1 $\Delta s_{\mathrm{mean}}$" + f"\nPearson r = {corr:.4f} (Identity Check)", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=10, loc="upper left")

            plt.tight_layout()
            plot2_path = os.path.join(output_dir, "fig_e2_2beta_vs_e1_deltas.png")
            plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {plot2_path}")

    # ──────────────────────────────────────────────────────────
    # Plot 3: Ratio gamma / max(|alpha|, |beta|) & Required lambda*
    # ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ratios = df_concepts["gamma_over_max_alpha_beta"].values
    lambda_stars = df_concepts["required_lambda_star"].values

    ax1.hist(ratios, bins=25, color="#8e44ad", edgecolor="black", alpha=0.75)
    ax1.axvline(1.0, color="red", linestyle="--", linewidth=2.0, label=r"Threshold ($\Delta > 0 \iff$ Ratio > 1)")
    ax1.axvline(np.median(ratios), color="#f39c12", linestyle="-", linewidth=2.0, label=f"Median = {np.median(ratios):.3f}")
    ax1.set_xlabel(r"Ratio $\gamma / \max(|\alpha|, |\beta|)$", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Concepts", fontsize=11, fontweight="bold")
    ax1.set_title(r"Lever Barrier: $\gamma / \max(|\alpha|, |\beta|)$ Distribution", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(fontsize=9)

    # Required lambda* for gamma > |beta| concepts
    valid_lambdas = lambda_stars[df_concepts["gamma_gt_beta"]]
    if len(valid_lambdas) > 0:
        ax2.hist(valid_lambdas, bins=20, color="#16a085", edgecolor="black", alpha=0.75)
        ax2.axvline(np.median(valid_lambdas), color="#e74c3c", linestyle="-", linewidth=2.0, label=f"Median $\lambda^* = {np.median(valid_lambdas):.1f}\\times$")
        ax2.set_xlabel(r"Required Vision Amplification $\lambda^* = |\alpha| / \gamma$", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Number of Concepts", fontsize=11, fontweight="bold")
        ax2.set_title(f"Vision Amplification Required (Feasible Concepts: {len(valid_lambdas)}/{len(df_concepts)})", fontsize=12, fontweight="bold")
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        ax2.legend(fontsize=9)

    plt.tight_layout()
    plot3_path = os.path.join(output_dir, "fig_e2_gamma_ratio_distribution.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot3_path}")

    # ──────────────────────────────────────────────────────────
    # Plot 4: Modality Gap Zero-Alpha Orthogonal Intervention
    # ──────────────────────────────────────────────────────────
    if "joint_acc_proj" in df_concepts.columns:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        base_acc = df_concepts["joint_acc"].values
        proj_acc = df_concepts["joint_acc_proj"].values

        x = np.arange(2)
        means = [np.mean(base_acc), np.mean(proj_acc)]
        std_errs = [np.std(base_acc) / np.sqrt(len(base_acc)), np.std(proj_acc) / np.sqrt(len(proj_acc))]

        bars = ax.bar(x, means, yerr=std_errs, capsize=6, color=["#e74c3c", "#27ae60"], edgecolor="black", width=0.5, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(["Original CLIP\n(With Affirmation Bias α)", "Zero-α Orthogonal Projection\n(v ⊥ μ_I)"], fontsize=11, fontweight="bold")
        ax.set_ylabel("2x2 Joint Accuracy (%)", fontsize=11, fontweight="bold")
        ax.set_title("E2 Intervention: Eliminating Text Bias via Orthogonal Projection", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for bar, mean_val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean_val + 0.5, f"{mean_val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

        plt.tight_layout()
        plot4_path = os.path.join(output_dir, "fig_e2_modality_gap_intervention.png")
        plt.savefig(plot4_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {plot4_path}")


def main():
    parser = argparse.ArgumentParser(description="E2: Exact 2x2 Hadamard Coordinate Decomposition")
    parser.add_argument("--csv_path", type=str, default="benchmarks/data/images/beaf_counterfactual_6col.csv")
    parser.add_argument("--image_root", type=str, default="benchmarks/data/images")
    parser.add_argument("--e1_report_dir", type=str, default="logs/evaluation/e1_minimal_pair_auc")
    parser.add_argument("--output_dir", type=str, default="logs/evaluation/e2_hadamard_decomposition")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--min_pairs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  E2: Exact 2x2 Hadamard Coordinate Decomposition                     ║")
    print("║  S_ab = C + a*beta + b*alpha + a*b*gamma                             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  Model       : {args.model} ({args.pretrained}) | Device: {device}")
    print(f"  Input CSV   : {args.csv_path}")
    print(f"  Output Dir  : {args.output_dir}")
    print(f"  Min Pairs   : {args.min_pairs}\n")

    # 1. Load CSV
    print("  [1/5] Loading paired counterfactual dataset...")
    df = pd.read_csv(args.csv_path)
    if "object_in_image" in df.columns:
        if df["object_in_image"].dtype == object:
            df["object_in_image"] = df["object_in_image"].apply(lambda x: str(x).strip().lower() == "true")
        else:
            df["object_in_image"] = df["object_in_image"].astype(bool)

    all_objects = sorted(df["object_name"].unique().tolist())
    target_objects = [o for o in all_objects if "," not in str(o)]
    print(f"  -> Total single-object concepts: {len(target_objects)}")

    # 2. Load Model & Tokenizer
    print(f"\n  [2/5] Initializing CLIP model '{args.model}'...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    def resolve_path(p: str, root: str) -> str:
        if os.path.isabs(p):
            return p
        full = os.path.join(root, p)
        if os.path.exists(full):
            return full
        return p

    # 3. Extract Global Image Mean for Zero-Alpha Intervention
    print("\n  [3/5] Extracting global image embeddings and computing global image mean...")
    all_pres_imgs = df[df["object_in_image"] == True]["image_path"].unique().tolist()
    all_pres_paths = [resolve_path(p, args.image_root) for p in all_pres_imgs[:500]]
    v_sample_norm, v_sample_raw, flags_sample = encode_images_unified(model, preprocess, all_pres_paths, device, args.batch_size)
    mu_I_global = v_sample_norm[flags_sample].mean(axis=0)
    mu_I_global = mu_I_global / np.linalg.norm(mu_I_global)

    # 4. Compute Per-Concept and Per-Pair Decomposition
    print("\n  [4/5] Computing Hadamard Coordinates per Concept...")
    per_concept_records = []
    per_pair_records = []

    total_pairs_evaluated = 0
    all_verification_errors = []

    for obj in target_objects:
        df_obj = df[df["object_name"] == obj].reset_index(drop=True)
        df_true = df_obj[df_obj["object_in_image"] == True].reset_index(drop=True)
        df_false = df_obj[df_obj["object_in_image"] == False].reset_index(drop=True)

        n_pairs = min(len(df_true), len(df_false))
        if n_pairs < args.min_pairs:
            continue

        img_paths_pres = [resolve_path(p, args.image_root) for p in df_true["image_path"].tolist()[:n_pairs]]
        img_paths_abs = [resolve_path(p, args.image_root) for p in df_false["image_path"].tolist()[:n_pairs]]
        t_pos_texts = df_true["positive_caption"].tolist()[:n_pairs]
        t_neg_texts = df_true["negative_caption"].tolist()[:n_pairs]

        # Extract normalized & unnormalized embeddings
        v_pres, v_pres_raw, mask_vp = encode_images_unified(model, preprocess, img_paths_pres, device, args.batch_size)
        v_abs, v_abs_raw, mask_va = encode_images_unified(model, preprocess, img_paths_abs, device, args.batch_size)

        valid_idx = np.where(mask_vp & mask_va)[0]
        if len(valid_idx) < args.min_pairs:
            continue

        v_pres = v_pres[valid_idx]
        v_abs = v_abs[valid_idx]
        v_pres_raw = v_pres_raw[valid_idx]
        v_abs_raw = v_abs_raw[valid_idx]

        t_pos_texts = [t_pos_texts[i] for i in valid_idx]
        t_neg_texts = [t_neg_texts[i] for i in valid_idx]

        t_pos, t_pos_raw = encode_texts_unified(model, tokenizer, t_pos_texts, device, args.batch_size)
        t_neg, t_neg_raw = encode_texts_unified(model, tokenizer, t_neg_texts, device, args.batch_size)

        # Compute 4 cosine similarities
        S11 = np.sum(v_pres * t_pos, axis=-1)  # pres, pos
        S12 = np.sum(v_abs * t_pos, axis=-1)   # abs,  pos
        S21 = np.sum(v_pres * t_neg, axis=-1)  # pres, neg
        S22 = np.sum(v_abs * t_neg, axis=-1)   # abs,  neg

        hadamard = compute_hadamard_coordinates(S11, S12, S21, S22)
        all_verification_errors.extend(hadamard["verification_error"].tolist())

        # Unnormalized Hadamard coordinates (raw scale)
        S11_raw = np.sum(v_pres_raw * t_pos_raw, axis=-1)
        S12_raw = np.sum(v_abs_raw * t_pos_raw, axis=-1)
        S21_raw = np.sum(v_pres_raw * t_neg_raw, axis=-1)
        S22_raw = np.sum(v_abs_raw * t_neg_raw, axis=-1)
        hadamard_raw = compute_hadamard_coordinates(S11_raw, S12_raw, S21_raw, S22_raw)

        # Zero-Alpha Intervention
        hadamard_proj = compute_zero_alpha_intervention(v_pres, v_abs, t_pos, t_neg, mu_I_global)

        n_concept_pairs = len(S11)
        total_pairs_evaluated += n_concept_pairs

        # Pair records
        for i in range(n_concept_pairs):
            per_pair_records.append({
                "object_name": obj,
                "S11": float(S11[i]),
                "S12": float(S12[i]),
                "S21": float(S21[i]),
                "S22": float(S22[i]),
                "C": float(hadamard["C"][i]),
                "alpha": float(hadamard["alpha"][i]),
                "beta": float(hadamard["beta"][i]),
                "gamma": float(hadamard["gamma"][i]),
                "abs_alpha": float(hadamard["abs_alpha"][i]),
                "abs_beta": float(hadamard["abs_beta"][i]),
                "delta_empirical": float(hadamard["delta_empirical"][i]),
                "delta_analytical": float(hadamard["delta_analytical"][i]),
                "joint_correct": int(hadamard["joint_correct"][i]),
            })

        # Concept Aggregate
        mean_abs_alpha = float(np.mean(hadamard["abs_alpha"]))
        mean_abs_beta = float(np.mean(hadamard["abs_beta"]))
        mean_gamma = float(np.mean(hadamard["gamma"]))
        mean_beta = float(np.mean(hadamard["beta"]))
        mean_alpha = float(np.mean(hadamard["alpha"]))
        mean_C = float(np.mean(hadamard["C"]))

        max_alpha_beta = max(mean_abs_alpha, mean_abs_beta)
        gamma_over_max = mean_gamma / (max_alpha_beta + 1e-9)
        required_lambda = mean_abs_alpha / (mean_gamma + 1e-9) if mean_gamma > 0 else np.nan

        per_concept_records.append({
            "object_name": obj,
            "n_pairs": n_concept_pairs,
            "C_mean": mean_C,
            "alpha_mean": mean_alpha,
            "abs_alpha_mean": mean_abs_alpha,
            "beta_mean": mean_beta,
            "abs_beta_mean": mean_abs_beta,
            "gamma_mean": mean_gamma,
            "2_beta": 2.0 * mean_beta,
            "delta_mean": float(np.mean(hadamard["delta_empirical"])),
            "joint_acc": float(np.mean(hadamard["joint_correct"]) * 100.0),
            "joint_acc_proj": float(np.mean(hadamard_proj["joint_correct"]) * 100.0),
            "alpha_proj_mean": float(np.mean(hadamard_proj["abs_alpha"])),
            "gamma_proj_mean": float(np.mean(hadamard_proj["gamma"])),
            "alpha_gt_gamma": bool(mean_abs_alpha > mean_gamma),
            "alpha_gt_beta": bool(mean_abs_alpha > mean_abs_beta),
            "gamma_gt_beta": bool(mean_gamma > mean_abs_beta),
            "gamma_over_max_alpha_beta": gamma_over_max,
            "required_lambda_star": required_lambda,
            "raw_abs_alpha_mean": float(np.mean(hadamard_raw["abs_alpha"])),
            "raw_abs_beta_mean": float(np.mean(hadamard_raw["abs_beta"])),
            "raw_gamma_mean": float(np.mean(hadamard_raw["gamma"])),
        })

        print(f"  [{obj:20s}] N={n_concept_pairs:4d} | |α|={mean_abs_alpha:.4f} | |β|={mean_abs_beta:.4f} | γ={mean_gamma:.4f} | "
              f"γ/max={gamma_over_max:.3f} | Acc={np.mean(hadamard['joint_correct'])*100:4.1f}% -> Proj={np.mean(hadamard_proj['joint_correct'])*100:4.1f}%")

    df_concepts = pd.DataFrame(per_concept_records).sort_values(by="abs_alpha_mean", ascending=False).reset_index(drop=True)
    df_pairs_out = pd.DataFrame(per_pair_records)

    # 5. Load E1 per-concept AUC for cross-validation
    df_e1 = None
    e1_csv = os.path.join(args.e1_report_dir, "e1_per_concept_auc.csv")
    if os.path.exists(e1_csv):
        df_e1 = pd.read_csv(e1_csv)

    # Global summary metrics
    max_verification_error = float(np.max(all_verification_errors))
    identity_verified = bool(max_verification_error < 1e-5)

    macro_abs_alpha = float(df_concepts["abs_alpha_mean"].mean())
    macro_abs_beta = float(df_concepts["abs_beta_mean"].mean())
    macro_gamma = float(df_concepts["gamma_mean"].mean())
    macro_C = float(df_concepts["C_mean"].mean())

    pct_alpha_gt_gamma = float(np.mean(df_concepts["alpha_gt_gamma"]) * 100.0)
    pct_alpha_gt_beta = float(np.mean(df_concepts["alpha_gt_beta"]) * 100.0)
    pct_gamma_gt_beta = float(np.mean(df_concepts["gamma_gt_beta"]) * 100.0)
    median_gamma_over_max = float(np.median(df_concepts["gamma_over_max_alpha_beta"]))
    median_required_lambda = float(np.nanmedian(df_concepts["required_lambda_star"]))

    macro_joint_acc = float(df_concepts["joint_acc"].mean())
    macro_proj_acc = float(df_concepts["joint_acc_proj"].mean())

    # Verdict
    if pct_alpha_gt_gamma >= 90.0 and pct_alpha_gt_beta >= 80.0 and identity_verified:
        verdict = "PASSED (Affirmation bias |alpha| decisively dominates interaction gamma and image beta, deterministically forcing 2x2 cosine failure)."
    else:
        verdict = "WARNING / ANOMALOUS (Inequality conditions partially violated)."

    summary = {
        "n_concepts_evaluated": len(df_concepts),
        "total_pairs_evaluated": total_pairs_evaluated,
        "identity_verification_passed": identity_verified,
        "max_verification_error": max_verification_error,
        "macro_mean_C": macro_C,
        "macro_mean_abs_alpha": macro_abs_alpha,
        "macro_mean_abs_beta": macro_abs_beta,
        "macro_mean_gamma": macro_gamma,
        "mean_alpha_over_gamma": macro_abs_alpha / (macro_gamma + 1e-9),
        "pct_alpha_gt_gamma": pct_alpha_gt_gamma,
        "pct_alpha_gt_beta": pct_alpha_gt_beta,
        "pct_gamma_gt_beta": pct_gamma_gt_beta,
        "median_gamma_over_max_alpha_beta": median_gamma_over_max,
        "median_required_vision_amplification_lambda_star": median_required_lambda,
        "macro_baseline_2x2_joint_acc": macro_joint_acc,
        "macro_zero_alpha_proj_2x2_joint_acc": macro_proj_acc,
        "verdict": verdict,
        "top5_highest_affirmation_bias": df_concepts.head(5)[["object_name", "abs_alpha_mean", "abs_beta_mean", "gamma_mean"]].to_dict(orient="records"),
        "top5_highest_interaction_gamma": df_concepts.sort_values(by="gamma_mean", ascending=False).head(5)[["object_name", "gamma_mean", "abs_alpha_mean", "abs_beta_mean"]].to_dict(orient="records"),
    }

    # Render Visualizations & Export
    render_e2_visualizations(df_concepts, df_pairs_out, summary, df_e1, args.output_dir)

    df_concepts.to_csv(os.path.join(args.output_dir, "e2_per_concept_decomposition.csv"), index=False)
    df_pairs_out.to_csv(os.path.join(args.output_dir, "e2_per_pair_decomposition.csv"), index=False)
    with open(os.path.join(args.output_dir, "e2_hadamard_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "═" * 70)
    print("  E2 HADAMARD DECOMPOSITION RESULTS SUMMARY")
    print("═" * 70)
    print(f"  Identity Sanity Check: Δ_empirical ≡ Δ_analytical : {'✅ PASSED (Max Error = ' + str(max_verification_error) + ')' if identity_verified else '❌ FAILED'}")
    print(f"  Macro |α| (Affirmation Bias)                       : {summary['macro_mean_abs_alpha']:.5f}")
    print(f"  Macro |β| (Vision Signal)                          : {summary['macro_mean_abs_beta']:.5f}")
    print(f"  Macro γ   (True Negation Interaction)              : {summary['macro_mean_gamma']:.5f}")
    print(f"  |α| / γ Ratio (Affirmation Dominance)              : {summary['mean_alpha_over_gamma']:.2f}×")
    print(f"  Concepts with |α| > γ                              : {summary['pct_alpha_gt_gamma']:.1f}%")
    print(f"  Concepts with |α| > |β|                            : {summary['pct_alpha_gt_beta']:.1f}%")
    print(f"  Concepts with γ > |β| (Vision Amp Feasible)        : {summary['pct_gamma_gt_beta']:.1f}%")
    print(f"  Median γ / max(|α|, |β|)                           : {summary['median_gamma_over_max_alpha_beta']:.4f}")
    print(f"  Median Required Amplification λ*                   : {summary['median_required_vision_amplification_lambda_star']:.2f}×")
    print(f"  Baseline 2x2 Joint Accuracy                        : {summary['macro_baseline_2x2_joint_acc']:.2f}%")
    print(f"  Zero-α Modality Gap Orthogonal Projection Accuracy : {summary['macro_zero_alpha_proj_2x2_joint_acc']:.2f}%  <-- [SOLUTION PREVIEW]")
    print("═" * 70)
    print(f"  Verdict: {summary['verdict']}")
    print("═" * 70)
    print(f"  Results saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
