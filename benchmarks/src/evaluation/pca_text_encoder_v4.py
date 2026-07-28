"""
CLIP Negation Information Degradation Mechanism & Representation Analysis Module (Refined 4th Edition).

Stage 1 Experiments:
  1-A. Unified Single-Pass Feature Extraction & Multi-Metric Breakdown (Cosine, Unnormalized Dot Product, L2 Distance)
  1-B. Direction Preservation Analysis (Distance Ratio: Negation vs Control with Deranged Permutation)
  1-C. Linear Probe & Template Shortcut Analysis (Stratified K-Fold CV & source_template Breakdown)
  1-D. Intrinsic Dimensionality & PCA Spectrum (Effective Rank, Participation Ratio, Negation Subspace Geometry)

Stage 2 & 3 Experiments:
  2. Micro-Batched Image-Text Retrieval Metrics & Ranking Flip Rate (Accuracy, Pearson r, Flip Rate)
  3. Projection Matrix SVD & Negation Direction Alignment Analysis (Singular Value Truncation & Alignment)
  4. Layer-wise PCA Grid Visualization (2D Scatter Grid & Centroid Distance Tracking)
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any, Union
from PIL import Image

import open_clip


# ==============================================================================
# 1. Unified Single-Pass Feature Extraction Engine
# ==============================================================================

def extract_all_features_unified(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: str = "cpu",
    target_token: str = "eot",
    batch_size: int = 256,
    custom_projection: Optional[Union[np.ndarray, str]] = None,
) -> Dict[str, Any]:
    """
    Extract ALL layer hidden states (Layer 0 to Layer L) and 5 granular pipeline steps
    in a SINGLE unified forward pass to eliminate redundant compute.

    Returns:
        dict with keys:
          - 'layers': Dict[str, np.ndarray] -> "Embedding", "Layer 1", ..., "Layer L"
          - 'pipeline': Dict[str, np.ndarray] -> Step0 to Step4
          - 'final_l2norm': np.ndarray -> Final normalized embedding
    """
    model.eval()
    all_tokens = tokenizer(texts).to(device)

    text_tower = getattr(model, 'text', model)
    token_embedding = text_tower.token_embedding
    positional_embedding = text_tower.positional_embedding
    transformer = text_tower.transformer
    ln_final = text_tower.ln_final
    text_projection = getattr(text_tower, 'text_projection', None)
    attn_mask = getattr(text_tower, 'attn_mask', None)

    resblocks = transformer.resblocks
    num_layers = 1 + len(resblocks)

    layer_batches = [[] for _ in range(num_layers)]
    pipeline_batches = {
        "Step0_Embedding": [],
        "Step1_Layer12_Raw": [],
        "Step2_Layer12_LN": [],
        "Step3_Projected_Unnorm": [],
        "Step4_Final_L2Norm": []
    }

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_tokens = all_tokens[start:end]

        with torch.no_grad():
            cast_dtype = transformer.get_cast_dtype()
            eot_indices = batch_tokens.argmax(dim=-1).cpu()
            batch_idx = torch.arange(batch_tokens.shape[0])

            # Layer 0: Embedding
            x = token_embedding(batch_tokens).to(cast_dtype)
            seq_len = batch_tokens.shape[1]
            x = x + positional_embedding[:seq_len].to(cast_dtype)

            hidden_states = [x]

            # Pass through Transformer resblocks
            x_perm = x.permute(1, 0, 2)
            for block in resblocks:
                x_perm = block(x_perm, attn_mask=attn_mask)
                hidden_states.append(x_perm.permute(1, 0, 2))

            # Store Layer-wise features according to target_token strategy
            for l_idx, hs in enumerate(hidden_states):
                hs_cpu = hs.float().cpu()
                if target_token == "eot":
                    feat = hs_cpu[batch_idx, eot_indices].numpy()
                elif target_token == "mean":
                    feat = hs_cpu.mean(dim=1).numpy()
                elif target_token == "all":
                    feat = hs_cpu.reshape(-1, hs_cpu.shape[-1]).numpy()
                else:
                    feat = hs_cpu[batch_idx, eot_indices].numpy()
                layer_batches[l_idx].append(feat)

            # Extract 5 Pipeline steps (using target_token strategy)
            def extract_step_token(tensor_b_l_d):
                t_cpu = tensor_b_l_d.float().cpu()
                if target_token == "eot":
                    return t_cpu[batch_idx, eot_indices]
                elif target_token == "mean":
                    return t_cpu.mean(dim=1)
                else:
                    return t_cpu[batch_idx, eot_indices]

            step0 = extract_step_token(hidden_states[0])
            step1 = extract_step_token(hidden_states[-1])

            x_ln = ln_final(hidden_states[-1])
            step2 = extract_step_token(x_ln)

            # Step 3: Projection
            if custom_projection is not None:
                if isinstance(custom_projection, str) and custom_projection == "identity":
                    step3 = step2.clone()
                else:
                    W_custom = torch.from_numpy(custom_projection).float().cpu()
                    step3 = step2 @ W_custom
            elif text_projection is not None:
                if isinstance(text_projection, nn.Linear):
                    step3 = text_projection(step2.to(text_projection.weight.dtype)).float().cpu()
                else:
                    step3 = (step2.to(text_projection.dtype) @ text_projection).float().cpu()
            else:
                step3 = step2.clone()

            # Step 4: Final L2 Normalization
            step4 = F.normalize(step3, dim=-1).cpu()

            pipeline_batches["Step0_Embedding"].append(step0.numpy())
            pipeline_batches["Step1_Layer12_Raw"].append(step1.numpy())
            pipeline_batches["Step2_Layer12_LN"].append(step2.numpy())
            pipeline_batches["Step3_Projected_Unnorm"].append(step3.numpy())
            pipeline_batches["Step4_Final_L2Norm"].append(step4.numpy())

    # Format output dictionaries
    layer_dict = {}
    for l_idx, feats in enumerate(layer_batches):
        name = "Embedding" if l_idx == 0 else f"Layer {l_idx}"
        layer_dict[name] = np.concatenate(feats, axis=0)

    pipeline_dict = {k: np.concatenate(v, axis=0) for k, v in pipeline_batches.items()}

    return {
        "layers": layer_dict,
        "pipeline": pipeline_dict,
        "final_l2norm": pipeline_dict["Step4_Final_L2Norm"]
    }


# ==============================================================================
# 2. Stage 1-A: Multi-Metric & Layer Breakdown Analysis
# ==============================================================================

def analyze_pipeline_and_layer_breakdown(
    pos_features: Dict[str, Any],
    neg_features: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Evaluate Cosine Similarity, Unnormalized Dot Product, and L2 Distance
    across pipeline steps and Transformer layers to isolate L2 Norm metric artifacts.
    """
    print("\n" + "="*60)
    print("Stage 1-A: Multi-Metric Pipeline & Layer-wise Breakdown Analysis")
    print("="*60)

    # 1. Pipeline Steps Breakdown
    step_names = [
        "Step0_Embedding",
        "Step1_Layer12_Raw",
        "Step2_Layer12_LN",
        "Step3_Projected_Unnorm",
        "Step4_Final_L2Norm"
    ]
    labels_map = {
        "Step0_Embedding": "Step 0: Embedding",
        "Step1_Layer12_Raw": "Step 1: Layer12 Raw",
        "Step2_Layer12_LN": "Step 2: Layer12+LN",
        "Step3_Projected_Unnorm": "Step 3: +Projection",
        "Step4_Final_L2Norm": "Step 4: +L2 Norm (Final)"
    }

    pipeline_results = []
    for idx, sname in enumerate(step_names):
        pos_f = pos_features["pipeline"][sname]
        neg_f = neg_features["pipeline"][sname]

        # Metric 1: Cosine Similarity
        pos_norm = pos_f / (np.linalg.norm(pos_f, axis=1, keepdims=True) + 1e-8)
        neg_norm = neg_f / (np.linalg.norm(neg_f, axis=1, keepdims=True) + 1e-8)
        cosine_sims = np.sum(pos_norm * neg_norm, axis=1)

        # Metric 2: Unnormalized Dot Product
        dot_prods = np.sum(pos_f * neg_f, axis=1)

        # Metric 3: L2 Distance
        l2_dists = np.linalg.norm(pos_f - neg_f, axis=1)

        pipeline_results.append({
            "step_id": idx,
            "step_key": sname,
            "step_name": labels_map[sname],
            "mean_cosine_sim": float(np.mean(cosine_sims)),
            "std_cosine_sim": float(np.std(cosine_sims)),
            "mean_dot_product": float(np.mean(dot_prods)),
            "mean_l2_distance": float(np.mean(l2_dists)),
        })
        print(f"  [{labels_map[sname]:25s}] Cosine Sim: {np.mean(cosine_sims):.4f} | Dot Prod: {np.mean(dot_prods):.2f} | L2 Dist: {np.mean(l2_dists):.4f}")

    df_pipeline = pd.DataFrame(pipeline_results)
    df_pipeline.to_csv(os.path.join(output_dir, "pipeline_step_breakdown.csv"), index=False)

    # 2. Individual Transformer Layer Breakdown
    layer_names = list(pos_features["layers"].keys())
    layer_results = []

    for l_name in layer_names:
        pos_f = pos_features["layers"][l_name]
        neg_f = neg_features["layers"][l_name]

        pos_norm = pos_f / (np.linalg.norm(pos_f, axis=1, keepdims=True) + 1e-8)
        neg_norm = neg_f / (np.linalg.norm(neg_f, axis=1, keepdims=True) + 1e-8)
        sims = np.sum(pos_norm * neg_norm, axis=1)

        layer_results.append({
            "layer": l_name,
            "mean_cosine_sim": float(np.mean(sims)),
            "std_cosine_sim": float(np.std(sims)),
            "median_cosine_sim": float(np.median(sims)),
        })

    df_layer = pd.DataFrame(layer_results)
    df_layer.to_csv(os.path.join(output_dir, "layerwise_cosine_breakdown.csv"), index=False)

    # Plot Pipeline Line Plot
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x_labels = [labels_map[k] for k in step_names]
    means_cos = df_pipeline["mean_cosine_sim"].values

    ax1.plot(x_labels, means_cos, "o-", color="crimson", lw=2.5, ms=8, label="Mean Cosine Sim")
    ax1.set_ylabel("Cosine Similarity", color="crimson", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="crimson")
    ax1.set_title("Pipeline Breakdown: Representation Geometry Shift Across Steps", fontsize=12, fontweight="bold")
    ax1.grid(True, ls="--", alpha=0.5)

    ax2 = ax1.twinx()
    means_l2 = df_pipeline["mean_l2_distance"].values
    ax2.plot(x_labels, means_l2, "s--", color="dodgerblue", lw=2, ms=7, label="Mean L2 Distance")
    ax2.set_ylabel("L2 Distance", color="dodgerblue", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="dodgerblue")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pipeline_step_lineplot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {"pipeline": pipeline_results, "layers": layer_results}


# ==============================================================================
# 3. Stage 1-B: Direction Preservation Analysis (Deranged Permutation)
# ==============================================================================

def analyze_direction_preservation(
    pos_features: Dict[str, Any],
    neg_features: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Compare Distance Ratio (||pos - neg||_post / ||pos - neg||_pre) for Negation vs Random Control.
    Uses strict deranged permutation for control pairs to prevent self-matching artifacts.
    """
    print("\n" + "="*60)
    print("Stage 1-B: Direction Preservation Analysis (Negation vs Random Control)")
    print("="*60)

    pos_pre = pos_features["pipeline"]["Step2_Layer12_LN"]
    neg_pre = neg_features["pipeline"]["Step2_Layer12_LN"]

    pos_post = pos_features["pipeline"]["Step4_Final_L2Norm"]
    neg_post = neg_features["pipeline"]["Step4_Final_L2Norm"]

    # Negation Pairs Distance Ratio
    dist_pre_neg = np.linalg.norm(pos_pre - neg_pre, axis=1)
    dist_post_neg = np.linalg.norm(pos_post - neg_post, axis=1)
    ratio_neg = dist_post_neg / (dist_pre_neg + 1e-8)

    # Deranged Random Permutation for Control (guarantees rand_idx[i] != i)
    N = len(pos_pre)
    np.random.seed(42)
    rand_idx = (np.arange(N) + np.random.randint(1, N, size=N)) % N

    rand_pre = pos_pre[rand_idx]
    rand_post = pos_post[rand_idx]

    dist_pre_ctrl = np.linalg.norm(pos_pre - rand_pre, axis=1)
    dist_post_ctrl = np.linalg.norm(pos_post - rand_post, axis=1)
    ratio_ctrl = dist_post_ctrl / (dist_pre_ctrl + 1e-8)

    # Two-sample Welch's t-test
    t_stat, p_val = stats.ttest_ind(ratio_neg, ratio_ctrl, equal_var=False)

    print(f"Negation Pairs  : Pre Dist={np.mean(dist_pre_neg):.4f} -> Post={np.mean(dist_post_neg):.4f} (Ratio={np.mean(ratio_neg):.4f})")
    print(f"Control Pairs   : Pre Dist={np.mean(dist_pre_ctrl):.4f} -> Post={np.mean(dist_post_ctrl):.4f} (Ratio={np.mean(ratio_ctrl):.4f})")
    print(f"Welch's T-test  : t={t_stat:.4f}, p-value={p_val:.2e}")

    # Plot Comparison Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratio_neg, bins=35, alpha=0.6, color="crimson", edgecolor="black", label=f"Negation Pairs (Mean: {np.mean(ratio_neg):.4f})")
    ax.hist(ratio_ctrl, bins=35, alpha=0.6, color="gray", edgecolor="black", label=f"Control Random Pairs (Mean: {np.mean(ratio_ctrl):.4f})")
    ax.set_title(f"Direction Preservation: Negation vs Control Pairs (p={p_val:.1e})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Distance Ratio (Post-Proj Dist / Pre-Proj Dist)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "direction_preservation_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    report = {
        "negation_mean_dist_pre": float(np.mean(dist_pre_neg)),
        "negation_mean_dist_post": float(np.mean(dist_post_neg)),
        "negation_mean_ratio": float(np.mean(ratio_neg)),
        "control_mean_dist_pre": float(np.mean(dist_pre_ctrl)),
        "control_mean_dist_post": float(np.mean(dist_post_ctrl)),
        "control_mean_ratio": float(np.mean(ratio_ctrl)),
        "ttest_t_stat": float(t_stat),
        "ttest_p_value": float(p_val)
    }

    rpt_path = os.path.join(output_dir, "direction_preservation_report.json")
    with open(rpt_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# ==============================================================================
# 4. Stage 1-C: Linear Probe & Template Shortcut Analysis
# ==============================================================================

def analyze_linear_probe_and_subsets(
    pos_features: Dict[str, Any],
    neg_features: Dict[str, Any],
    pair_metadata: List[dict],
    output_dir: str
) -> Dict[str, Any]:
    """
    Train Logistic Regression linear probe (5-Fold CV) and break down metrics
    by source_template to verify whether 99% probe accuracy is shortcut learning.
    """
    print("\n" + "="*60)
    print("Stage 1-C: Linear Probe & Sub-Dataset Template Shortcut Analysis")
    print("="*60)

    n_pos = len(pair_metadata)
    n_neg = len(pair_metadata)
    y = np.array([1] * n_pos + [0] * n_neg)

    # 1. Overall Linear Probe across Steps
    probe_results = {}
    step_keys = ["Step0_Embedding", "Step2_Layer12_LN", "Step4_Final_L2Norm"]
    step_labels = ["Step 0 (Embed)", "Step 2 (Layer12+LN)", "Step 4 (Final L2Norm)"]

    for skey, slabel in zip(step_keys, step_labels):
        X_pos = pos_features["pipeline"][skey]
        X_neg = neg_features["pipeline"][skey]
        X = np.vstack([X_pos, X_neg])
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_norm, y, cv=cv, scoring="accuracy")

        mean_acc = float(np.mean(scores)) * 100
        std_acc = float(np.std(scores)) * 100
        probe_results[slabel] = {"mean_accuracy": mean_acc, "std_accuracy": std_acc}
        print(f"  [{slabel:22s}] Linear Probe Accuracy: {mean_acc:.2f}% (±{std_acc:.2f}%)")

    # 2. Template Sub-dataset Breakdown
    df_meta = pd.DataFrame(pair_metadata)
    template_report = {}

    if "source_template" in df_meta.columns:
        unique_templates = df_meta["source_template"].unique()
        print("\n  --- Sub-dataset Breakdown by source_template ---")

        for tmpl in unique_templates:
            mask = (df_meta["source_template"] == tmpl).values
            n_sub = np.sum(mask)
            if n_sub < 10:
                continue

            y_sub = np.array([1] * n_sub + [0] * n_sub)
            X_pos_sub = pos_features["pipeline"]["Step4_Final_L2Norm"][mask]
            X_neg_sub = neg_features["pipeline"]["Step4_Final_L2Norm"][mask]
            X_sub = np.vstack([X_pos_sub, X_neg_sub])
            X_sub_norm = X_sub / (np.linalg.norm(X_sub, axis=1, keepdims=True) + 1e-8)

            clf_sub = LogisticRegression(max_iter=1000, random_state=42)
            cv_sub = StratifiedKFold(n_splits=min(5, n_sub // 2), shuffle=True, random_state=42)
            scores_sub = cross_val_score(clf_sub, X_sub_norm, y_sub, cv=cv_sub, scoring="accuracy")

            # Cosine similarity for this subset
            pos_n = X_pos_sub / (np.linalg.norm(X_pos_sub, axis=1, keepdims=True) + 1e-8)
            neg_n = X_neg_sub / (np.linalg.norm(X_neg_sub, axis=1, keepdims=True) + 1e-8)
            sims_sub = np.sum(pos_n * neg_n, axis=1)

            mean_acc_sub = float(np.mean(scores_sub)) * 100
            mean_sim_sub = float(np.mean(sims_sub))

            template_report[str(tmpl)] = {
                "sample_count": int(n_sub),
                "linear_probe_accuracy_pct": mean_acc_sub,
                "mean_cosine_sim": mean_sim_sub
            }
            print(f"    [{str(tmpl):25s}] (N={n_sub:5d}) Probe Acc: {mean_acc_sub:6.2f}% | Cosine Sim: {mean_sim_sub:.4f}")

    # Save outputs
    report_path = os.path.join(output_dir, "linear_probe_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"overall_probe": probe_results, "template_breakdown": template_report}, f, indent=2)

    # Plot Overall Linear Probe Accuracy
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(probe_results.keys(), [v["mean_accuracy"] for v in probe_results.values()],
                  color=["gray", "seagreen", "crimson"], alpha=0.85, edgecolor="black")
    ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=11)
    ax.set_title("Linear Probe: Separability Pre vs Post Projection", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", ls="--", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "linear_probe_accuracy.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {"overall": probe_results, "templates": template_report}


# ==============================================================================
# 5. Stage 1-D: Intrinsic Dimensionality & Negation Subspace Analysis
# ==============================================================================

def compute_intrinsic_dimensionality(X: np.ndarray) -> Tuple[float, float]:
    """
    Compute Effective Rank (r_eff via spectral entropy) and Participation Ratio (PR).
    """
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    cov = (X_centered.T @ X_centered) / X_centered.shape[0]
    eigenvals = np.linalg.eigvalsh(cov)
    eigenvals = np.sort(np.maximum(eigenvals, 1e-12))[::-1]

    total_val = np.sum(eigenvals)
    p = eigenvals / total_val

    entropy = -np.sum(p * np.log(p + 1e-12))
    eff_rank = float(np.exp(entropy))
    pr = float((np.sum(eigenvals) ** 2) / np.sum(eigenvals ** 2))

    return eff_rank, pr


def analyze_pca_spectrum_compression(
    pos_features: Dict[str, Any],
    neg_features: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Compare Intrinsic Dimensionality (Pre vs Post) and compute Effective Rank
    specifically for the Negation Difference Vectors (pos - neg).
    """
    print("\n" + "="*60)
    print("Stage 1-D: Intrinsic Dimensionality & Negation Subspace Geometry")
    print("="*60)

    X_pre = np.vstack([pos_features["pipeline"]["Step2_Layer12_LN"], neg_features["pipeline"]["Step2_Layer12_LN"]])
    X_post = np.vstack([pos_features["pipeline"]["Step4_Final_L2Norm"], neg_features["pipeline"]["Step4_Final_L2Norm"]])

    # Intrinsic Dimensionality of Full Representation Space
    eff_rank_pre, pr_pre = compute_intrinsic_dimensionality(X_pre)
    eff_rank_post, pr_post = compute_intrinsic_dimensionality(X_post)

    # Intrinsic Dimensionality of Negation Difference Vector Subspace (pos - neg)
    diff_pre = pos_features["pipeline"]["Step2_Layer12_LN"] - neg_features["pipeline"]["Step2_Layer12_LN"]
    diff_post = pos_features["pipeline"]["Step4_Final_L2Norm"] - neg_features["pipeline"]["Step4_Final_L2Norm"]
    eff_rank_diff_pre, pr_diff_pre = compute_intrinsic_dimensionality(diff_pre)
    eff_rank_diff_post, pr_diff_post = compute_intrinsic_dimensionality(diff_post)

    # PCA Variance Spectrum
    n_comp = min(10, X_pre.shape[1], X_post.shape[1])
    pca_pre = PCA(n_components=n_comp).fit(X_pre)
    pca_post = PCA(n_components=n_comp).fit(X_post)

    var_pre = pca_pre.explained_variance_ratio_
    var_post = pca_post.explained_variance_ratio_

    print(f"Full Space Pre-Proj (Layer12+LN) : Eff Rank={eff_rank_pre:.2f}, PR={pr_pre:.2f}, PC1={var_pre[0]*100:.2f}%")
    print(f"Full Space Post-Proj (Final L2)  : Eff Rank={eff_rank_post:.2f}, PR={pr_post:.2f}, PC1={var_post[0]*100:.2f}%")
    print(f"Negation Diff Subspace Pre-Proj  : Eff Rank={eff_rank_diff_pre:.2f}, PR={pr_diff_pre:.2f}")
    print(f"Negation Diff Subspace Post-Proj : Eff Rank={eff_rank_diff_post:.2f}, PR={pr_diff_post:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    indices = np.arange(1, n_comp + 1)
    ax.plot(indices, var_pre * 100, "o-", color="seagreen", lw=2, label=f"Pre-Projection (r_eff={eff_rank_pre:.1f}, PR={pr_pre:.1f})")
    ax.plot(indices, var_post * 100, "s-", color="crimson", lw=2, label=f"Post-Projection (r_eff={eff_rank_post:.1f}, PR={pr_post:.1f})")
    ax.set_xlabel("Principal Component Index", fontsize=11)
    ax.set_ylabel("Explained Variance Ratio (%)", fontsize=11)
    ax.set_title("PCA Variance Spectrum & Intrinsic Dimensionality", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "pca_spectrum_compression.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    report = {
        "pre_effective_rank": eff_rank_pre,
        "pre_participation_ratio": pr_pre,
        "post_effective_rank": eff_rank_post,
        "post_participation_ratio": pr_post,
        "diff_subspace_pre_effective_rank": eff_rank_diff_pre,
        "diff_subspace_post_effective_rank": eff_rank_diff_post,
    }

    rpt_path = os.path.join(output_dir, "pca_spectrum_report.json")
    with open(rpt_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# ==============================================================================
# 6. Stage 3: Projection Matrix SVD & Negation Direction Alignment
# ==============================================================================

def analyze_projection_svd_ablation(
    model: nn.Module,
    tokenizer: Any,
    pos_texts: List[str],
    neg_texts: List[str],
    output_dir: str,
    device: str = "cpu",
    batch_size: int = 256
) -> Dict[str, Any]:
    """
    Perform SVD on W_proj = U S V^T and measure Cosine Alignment between
    mean negation direction d_neg and right singular vectors V.
    Also tests singular value truncation.
    """
    print("\n" + "="*60)
    print("Stage 3: Projection SVD & Negation Direction Alignment Analysis")
    print("="*60)

    text_tower = getattr(model, 'text', model)
    text_projection = getattr(text_tower, 'text_projection', None)

    if text_projection is None:
        print("Model does not have a text_projection matrix. Skipping SVD Ablation.")
        return {}

    if isinstance(text_projection, nn.Linear):
        W_orig = text_projection.weight.T.detach().cpu().numpy()
    else:
        W_orig = text_projection.detach().cpu().numpy()

    D_in, D_out = W_orig.shape

    # Compute SVD: W = U S V^T
    U, S, Vh = np.linalg.svd(W_orig, full_matrices=False)

    # Calculate Mean Negation Difference Vector at Pre-Projection (Step 2: Layer12+LN)
    pos_feats = extract_all_features_unified(model, tokenizer, pos_texts, device, batch_size=batch_size)
    neg_feats = extract_all_features_unified(model, tokenizer, neg_texts, device, batch_size=batch_size)

    pos_step2 = pos_feats["pipeline"]["Step2_Layer12_LN"]
    neg_step2 = neg_feats["pipeline"]["Step2_Layer12_LN"]

    diff_vecs = pos_step2 - neg_step2
    mean_d_neg = np.mean(diff_vecs, axis=0)
    norm_d_neg = mean_d_neg / (np.linalg.norm(mean_d_neg) + 1e-8)

    # Compute Cosine Alignment of d_neg with right singular vectors (rows of Vh)
    alignments = np.abs(np.dot(Vh, norm_d_neg))
    top_aligned_idx = np.argsort(alignments)[::-1]

    print(f"  SVD Singular Values S (Top 5): {S[:5].round(3)}")
    print(f"  Negation Direction Alignment with Top Singular Vector V1: {alignments[0]:.4f}")
    print(f"  Max Alignment: {np.max(alignments):.4f} (with Singular Vector #{top_aligned_idx[0]+1})")

    # Singular Value Truncation Experiment (Top-k vs Bottom-k singular values)
    k_keep = max(1, len(S) // 2)

    # Keep Top-k
    S_top = S.copy()
    S_top[k_keep:] = 0.0
    W_top = U @ np.diag(S_top) @ Vh

    # Keep Bottom-k
    S_bot = S.copy()
    S_bot[:k_keep] = 0.0
    W_bot = U @ np.diag(S_bot) @ Vh

    pos_top = extract_all_features_unified(model, tokenizer, pos_texts, device, batch_size=batch_size, custom_projection=W_top)
    neg_top = extract_all_features_unified(model, tokenizer, neg_texts, device, batch_size=batch_size, custom_projection=W_top)

    pos_bot = extract_all_features_unified(model, tokenizer, pos_texts, device, batch_size=batch_size, custom_projection=W_bot)
    neg_bot = extract_all_features_unified(model, tokenizer, neg_texts, device, batch_size=batch_size, custom_projection=W_bot)

    def get_cosine_sim(p_dict, n_dict):
        p = p_dict["final_l2norm"]
        n = n_dict["final_l2norm"]
        return float(np.mean(np.sum(p * n, axis=1)))

    sim_orig = get_cosine_sim(pos_feats, neg_feats)
    sim_top = get_cosine_sim(pos_top, neg_top)
    sim_bot = get_cosine_sim(pos_bot, neg_bot)

    print(f"  Original W_proj Cosine Sim : {sim_orig:.4f}")
    print(f"  Top-{k_keep} Singular Values Cosine Sim: {sim_top:.4f}")
    print(f"  Bottom-{k_keep} Singular Values Cosine Sim: {sim_bot:.4f}")

    svd_report = {
        "singular_values_top10": S[:10].tolist(),
        "top1_alignment": float(alignments[0]),
        "max_alignment": float(np.max(alignments)),
        "max_alignment_singular_vector_idx": int(top_aligned_idx[0]),
        "cosine_sim_original": sim_orig,
        "cosine_sim_top_half_svd": sim_top,
        "cosine_sim_bottom_half_svd": sim_bot,
    }

    svd_path = os.path.join(output_dir, "projection_svd_report.json")
    with open(svd_path, "w", encoding="utf-8") as f:
        json.dump(svd_report, f, indent=2)

    return svd_report


# ==============================================================================
# 7. Stage 2: Micro-Batched Image-Text Retrieval Metrics
# ==============================================================================

def analyze_image_text_retrieval_metrics(
    model: nn.Module,
    tokenizer: Any,
    preprocess: Any,
    pair_metadata: List[dict],
    pos_texts: List[str],
    neg_texts: List[str],
    image_root: str,
    output_dir: str,
    device: str = "cpu",
    batch_size: int = 256,
):
    """
    Compute Binary MCQ Accuracy, Ranking Flip Rate, and Pearson correlation
    using efficient micro-batched text and image encoding.
    """
    print("\n" + "="*60)
    print("Stage 2: Micro-Batched Image-Text Retrieval & Flip Rate Analysis")
    print("="*60)

    model.eval()

    # Pre-encode all positive and negative captions in batches
    print(f"Pre-encoding {len(pos_texts)} caption pairs in batches of {batch_size}...")
    pos_embs = []
    neg_embs = []

    for start in range(0, len(pos_texts), batch_size):
        end = min(start + batch_size, len(pos_texts))
        pos_tok = tokenizer(pos_texts[start:end]).to(device)
        neg_tok = tokenizer(neg_texts[start:end]).to(device)

        with torch.no_grad():
            p_emb = model.encode_text(pos_tok, normalize=True).float().cpu()
            n_emb = model.encode_text(neg_tok, normalize=True).float().cpu()

        pos_embs.append(p_emb)
        neg_embs.append(n_emb)

    pos_embs = torch.cat(pos_embs, dim=0)
    neg_embs = torch.cat(neg_embs, dim=0)

    # Group pairs by unique image path
    image_pairs = {}
    for i, meta in enumerate(pair_metadata):
        img_path = os.path.join(image_root, meta["image_path"])
        if img_path not in image_pairs:
            image_pairs[img_path] = []
        image_pairs[img_path].append(i)

    print(f"Processing {len(image_pairs)} unique images...")
    results = []
    skipped = 0

    for img_path, indices in image_pairs.items():
        if not os.path.exists(img_path):
            skipped += len(indices)
            continue

        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        except Exception:
            skipped += len(indices)
            continue

        with torch.no_grad():
            img_emb = model.encode_image(image, normalize=True).float().cpu()

        for idx in indices:
            p_emb = pos_embs[idx : idx + 1]
            n_emb = neg_embs[idx : idx + 1]

            sim_pos = float(F.cosine_similarity(img_emb, p_emb).item())
            sim_neg = float(F.cosine_similarity(img_emb, n_emb).item())

            obj_in_img = pair_metadata[idx].get("object_in_image", None)
            if isinstance(obj_in_img, str):
                obj_in_img = obj_in_img.strip().lower() == "true"

            results.append({
                "image_path": pair_metadata[idx]["image_path"],
                "object_name": pair_metadata[idx].get("object_name", ""),
                "object_in_image": obj_in_img,
                "source_template": pair_metadata[idx].get("source_template", ""),
                "positive_caption": pos_texts[idx],
                "negative_caption": neg_texts[idx],
                "sim_image_pos": sim_pos,
                "sim_image_neg": sim_neg,
                "sim_diff": sim_pos - sim_neg,
            })

    if len(results) == 0:
        print("No valid images found. Skipping Retrieval Metrics.")
        return

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(output_dir, "image_text_similarity.csv"), index=False)

    sim_pos_arr = res_df["sim_image_pos"].values
    sim_neg_arr = res_df["sim_image_neg"].values
    sim_diff_arr = res_df["sim_diff"].values

    pearson_r = float(np.corrcoef(sim_pos_arr, sim_neg_arr)[0, 1])

    in_img_mask = res_df["object_in_image"] == True
    if np.sum(in_img_mask) > 0:
        flip_rate_in_img = float(np.mean(sim_neg_arr[in_img_mask] > sim_pos_arr[in_img_mask])) * 100
        accuracy_in_img = float(np.mean(sim_pos_arr[in_img_mask] > sim_neg_arr[in_img_mask])) * 100
    else:
        flip_rate_in_img = 0.0
        accuracy_in_img = 0.0

    retrieval_summary = {
        "total_pairs_evaluated": len(results),
        "pearson_r": pearson_r,
        "mean_sim_pos": float(np.mean(sim_pos_arr)),
        "mean_sim_neg": float(np.mean(sim_neg_arr)),
        "mean_sim_diff": float(np.mean(sim_diff_arr)),
        "binary_mcq_accuracy_pct": accuracy_in_img,
        "ranking_flip_rate_pct": flip_rate_in_img,
    }

    sum_path = os.path.join(output_dir, "retrieval_metrics_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_summary, f, indent=2)

    print("\n=== Retrieval Metrics Summary ===")
    print(f"  Pearson r                 : {pearson_r:.4f}")
    print(f"  Binary MCQ Accuracy       : {accuracy_in_img:.1f}%")
    print(f"  Ranking Flip Rate         : {flip_rate_in_img:.1f}%")


# ==============================================================================
# 8. Layer-wise PCA Grid Visualization
# ==============================================================================

def analyze_and_plot_pca(
    pos_features: Dict[str, Any],
    neg_features: Dict[str, Any],
    output_dir: str,
    target_token: str = "eot"
) -> Dict[str, Any]:
    """
    Generate single-canvas comparative PCA scatter grid (pca_grid_{target_token}.png)
    reusing already extracted layer features.
    """
    print("\n" + "="*60)
    print(f"Layer-wise PCA Grid Visualization (Target Token: '{target_token}')")
    print("="*60)

    layer_names = list(pos_features["layers"].keys())
    num_layers = len(layer_names)
    n_pos = len(pos_features["layers"][layer_names[0]])

    cols = min(4, num_layers)
    rows = (num_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if num_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    analysis_report = [
        "=== CLIP Text Encoder Layer-wise PCA Analysis Report ===",
        f"Target Token Strategy: {target_token}",
        f"Total Layers analyzed: {num_layers}\n"
    ]

    for l_idx, l_name in enumerate(layer_names):
        pos_f = pos_features["layers"][l_name]
        neg_f = neg_features["layers"][l_name]

        combined = np.vstack([pos_f, neg_f])
        pca = PCA(n_components=min(combined.shape[0], combined.shape[1], 2))
        combined_pca = pca.fit_transform(combined)

        pos_pca = combined_pca[:n_pos]
        neg_pca = combined_pca[n_pos:]

        var_ratio = pca.explained_variance_ratio_
        total_var_2d = float(np.sum(var_ratio[:2])) if len(var_ratio) >= 2 else float(np.sum(var_ratio))

        pos_mean_orig = pos_f.mean(axis=0)
        neg_mean_orig = neg_f.mean(axis=0)
        centroid_dist_orig = float(np.linalg.norm(pos_mean_orig - neg_mean_orig))

        pos_mean_pca = pos_pca.mean(axis=0)
        neg_mean_pca = neg_pca.mean(axis=0)

        report_str = (f"[{l_name}] 2D Explained Variance: {total_var_2d*100:.2f}% "
                      f"(PC1: {var_ratio[0]*100:.1f}%, PC2: {var_ratio[1]*100:.1f}%) | "
                      f"Group Centroid Dist (Orig Dim): {centroid_dist_orig:.4f}")
        analysis_report.append(report_str)

        ax = axes[l_idx]
        ax.scatter(pos_pca[:, 0], pos_pca[:, 1], c="dodgerblue", label="Positive", alpha=0.75, edgecolors="k", linewidth=0.5, s=40)
        ax.scatter(neg_pca[:, 0], neg_pca[:, 1], c="crimson", label="Negative", alpha=0.75, edgecolors="k", linewidth=0.5, marker="^", s=40)
        ax.scatter(pos_mean_pca[0], pos_mean_pca[1], c="navy", s=120, marker="X", label="Pos Centroid", edgecolors="w")
        ax.scatter(neg_mean_pca[0], neg_mean_pca[1], c="darkred", s=120, marker="X", label="Neg Centroid", edgecolors="w")

        ax.set_title(f"{l_name}\n(Var: {total_var_2d*100:.1f}%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("PC 1", fontsize=9)
        ax.set_ylabel("PC 2", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

        if l_idx == 0:
            ax.legend(fontsize=8, loc="best")

    for l_idx in range(num_layers, len(axes)):
        fig.delaxes(axes[l_idx])

    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"pca_grid_{target_token}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()

    report_filename = os.path.join(output_dir, f"pca_report_{target_token}.txt")
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(analysis_report))

    return {"plot_filename": plot_filename, "report_filename": report_filename}


# ==============================================================================
# Main Execution Entrypoint
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Negation Mechanism Analysis Refined 4th Edition")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--target_token", type=str, default="eot", choices=["eot", "mean", "all"])
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="logs/pipeline_breakdown/openai_vit_b32")
    parser.add_argument("--max_samples", type=int, default=60000)
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    pos_texts = []
    neg_texts = []
    pair_metadata = []

    if args.csv_path and os.path.exists(args.csv_path):
        df = pd.read_csv(args.csv_path).head(args.max_samples)
        pos_texts = df["positive_caption"].astype(str).tolist()
        neg_texts = df["negative_caption"].astype(str).tolist()
        for _, row in df.iterrows():
            meta = {
                "image_path": str(row.get("image_path", "")),
                "object_name": str(row.get("object_name", "")),
                "object_in_image": row.get("object_in_image", None),
                "source_template": str(row.get("source_template", ""))
            }
            if isinstance(meta["object_in_image"], str):
                meta["object_in_image"] = meta["object_in_image"].strip().lower() == "true"
            pair_metadata.append(meta)

    # Load model & tokenizer
    print(f"Loading model {args.model} ({args.pretrained})...")
    model, preprocess, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    # 1. Single Unified Feature Extraction Pass
    print("\nExecuting Single-Pass Unified Feature Extraction...")
    pos_features = extract_all_features_unified(model, tokenizer, pos_texts, device, args.target_token, args.batch_size)
    neg_features = extract_all_features_unified(model, tokenizer, neg_texts, device, args.target_token, args.batch_size)

    # 2. Stage 1-A. Multi-Metric Pipeline & Layer Breakdown
    analyze_pipeline_and_layer_breakdown(pos_features, neg_features, args.output_dir)

    # 3. Stage 1-B. Direction Preservation Analysis
    analyze_direction_preservation(pos_features, neg_features, args.output_dir)

    # 4. Stage 1-C. Linear Probe & Sub-dataset Template Analysis
    analyze_linear_probe_and_subsets(pos_features, neg_features, pair_metadata, args.output_dir)

    # 5. Stage 1-D. Intrinsic Dimensionality & Negation Subspace
    analyze_pca_spectrum_compression(pos_features, neg_features, args.output_dir)

    # 6. Layer-wise PCA Scatter Grid Plot
    analyze_and_plot_pca(pos_features, neg_features, args.output_dir, args.target_token)

    # 7. Stage 3. Projection SVD & Negation Direction Alignment
    analyze_projection_svd_ablation(model, tokenizer, pos_texts, neg_texts, args.output_dir, device, args.batch_size)

    # 8. Stage 2. Micro-Batched Retrieval Metrics (if image_root provided)
    if args.image_root:
        analyze_image_text_retrieval_metrics(model, tokenizer, preprocess, pair_metadata, pos_texts, neg_texts, args.image_root, args.output_dir, device, args.batch_size)

    print(f"\n✅ Refined 4th Edition Pipeline Analysis Complete! All results saved in: {args.output_dir}")
