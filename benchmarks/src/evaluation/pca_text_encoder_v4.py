"""
CLIP Negation Information Degradation Mechanism & Representation Analysis Module (Refined 4th Edition - Clean Code Refactored).

Architecture & Design Patterns:
  - Separation of Concerns (Pure Calculation vs Plotting & Serialization)
  - Type Safety (Enums for Steps and Metadata Keys, Dataclass for Configuration)
  - DRY Math Helpers (l2_normalize, batch_cosine_similarity, batch_l2_distance)
  - Pure Local RNG (np.random.default_rng to avoid global state side-effects)
"""

import os
import argparse
import json
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Union
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
from PIL import Image

import open_clip


# ==============================================================================
# 0. Domain Enums, Config Dataclasses, and Math Helpers
# ==============================================================================

class PipelineStep(str, Enum):
    EMBEDDING = "Step0_Embedding"
    LAYER12_RAW = "Step1_Layer12_Raw"
    LAYER12_LN = "Step2_Layer12_LN"
    PROJECTED_UNNORM = "Step3_Projected_Unnorm"
    FINAL_L2NORM = "Step4_Final_L2Norm"


class MetadataKey(str, Enum):
    IMAGE_PATH = "image_path"
    OBJECT_NAME = "object_name"
    OBJECT_IN_IMAGE = "object_in_image"
    SOURCE_TEMPLATE = "source_template"


@dataclass
class RetrievalConfig:
    image_root: str
    output_dir: str
    device: str = "cpu"
    batch_size: int = 256
    image_batch_size: int = 64


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize vectors along the last dimension using L2 norm."""
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def batch_cosine_similarity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between row vectors u and v."""
    return np.sum(l2_normalize(u) * l2_normalize(v), axis=-1)


def batch_dot_product(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute pairwise unnormalized dot product between row vectors u and v."""
    return np.sum(u * v, axis=-1)


def batch_l2_distance(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute pairwise L2 distance between row vectors u and v."""
    return np.linalg.norm(u - v, axis=-1)


# ==============================================================================
# 1. Unified Single-Pass Feature Extraction Engine & Assertion Check
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
        PipelineStep.EMBEDDING.value: [],
        PipelineStep.LAYER12_RAW.value: [],
        PipelineStep.LAYER12_LN.value: [],
        PipelineStep.PROJECTED_UNNORM.value: [],
        PipelineStep.FINAL_L2NORM.value: []
    }
    inter_layer_batches = {f"Layer{i}": [] for i in range(1, len(resblocks))}

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

                if 1 <= l_idx < len(resblocks):
                    inter_layer_batches[f"Layer{l_idx}"].append(feat)

            # Extract 5 Pipeline steps
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

            pipeline_batches[PipelineStep.EMBEDDING.value].append(step0.numpy())
            pipeline_batches[PipelineStep.LAYER12_RAW.value].append(step1.numpy())
            pipeline_batches[PipelineStep.LAYER12_LN.value].append(step2.numpy())
            pipeline_batches[PipelineStep.PROJECTED_UNNORM.value].append(step3.numpy())
            pipeline_batches[PipelineStep.FINAL_L2NORM.value].append(step4.numpy())

    layer_dict = {}
    for l_idx, feats in enumerate(layer_batches):
        name = "Embedding" if l_idx == 0 else f"Layer {l_idx}"
        layer_dict[name] = np.concatenate(feats, axis=0)

    pipeline_dict = {k: np.concatenate(v, axis=0) for k, v in pipeline_batches.items()}
    for k, v in inter_layer_batches.items():
        pipeline_dict[k] = np.concatenate(v, axis=0)

    return {
        "layers": layer_dict,
        "pipeline": pipeline_dict,
        "final_l2norm": pipeline_dict[PipelineStep.FINAL_L2NORM.value]
    }


def assert_embedding_consistency(
    model: nn.Module,
    tokenizer: Any,
    sample_texts: List[str],
    extracted_final_embs: np.ndarray,
    device: str = "cpu"
):
    """Cross-validate manual forward pass outputs against official model.encode_text()."""
    if len(sample_texts) == 0:
        return

    n_check = min(10, len(sample_texts))
    check_tokens = tokenizer(sample_texts[:n_check]).to(device)

    with torch.no_grad():
        official_embs = model.encode_text(check_tokens, normalize=True).float().cpu().numpy()

    diff = np.abs(extracted_final_embs[:n_check] - official_embs)
    max_diff = float(np.max(diff))

    assert max_diff < 1e-3, f"Embedding consistency assertion failed! Max diff: {max_diff:.6f}"
    print(f"✅ Embedding Consistency Verified! (Max diff vs model.encode_text: {max_diff:.6e})")


# ==============================================================================
# 2. Stage 1-A: Multi-Metric Pipeline & Layer Breakdown Analysis
# ==============================================================================

def compute_pipeline_and_layer_breakdown(
    pos_features: Dict[str, Any],
    neg_features: Dict[str, Any]
) -> Dict[str, Any]:
    """Pure calculation of multi-metric breakdown across all sequence steps and layers."""
    full_step_keys = [PipelineStep.EMBEDDING.value] + [f"Layer{i}" for i in range(1, 12)] + [
        PipelineStep.LAYER12_RAW.value,
        PipelineStep.LAYER12_LN.value,
        PipelineStep.PROJECTED_UNNORM.value,
        PipelineStep.FINAL_L2NORM.value
    ]

    labels_map = {PipelineStep.EMBEDDING.value: "Step 0: Embed"}
    for i in range(1, 12):
        labels_map[f"Layer{i}"] = f"Layer {i}"
    labels_map[PipelineStep.LAYER12_RAW.value] = "Layer 12 Raw"
    labels_map[PipelineStep.LAYER12_LN.value] = "Layer 12+LN"
    labels_map[PipelineStep.PROJECTED_UNNORM.value] = "+Projection"
    labels_map[PipelineStep.FINAL_L2NORM.value] = "+Final L2Norm"

    pipeline_results = []
    for idx, sname in enumerate(full_step_keys):
        pos_f = pos_features["pipeline"][sname]
        neg_f = neg_features["pipeline"][sname]

        cosine_sims = batch_cosine_similarity(pos_f, neg_f)
        dot_prods = batch_dot_product(pos_f, neg_f)
        l2_dists = batch_l2_distance(pos_f, neg_f)

        pipeline_results.append({
            "step_id": idx,
            "step_key": sname,
            "step_name": labels_map[sname],
            "mean_cosine_sim": float(np.mean(cosine_sims)),
            "std_cosine_sim": float(np.std(cosine_sims)),
            "mean_dot_product": float(np.mean(dot_prods)),
            "mean_l2_distance": float(np.mean(l2_dists)),
        })

    layer_results = []
    for l_name, pos_f in pos_features["layers"].items():
        neg_f = neg_features["layers"][l_name]
        sims = batch_cosine_similarity(pos_f, neg_f)

        layer_results.append({
            "layer": l_name,
            "mean_cosine_sim": float(np.mean(sims)),
            "std_cosine_sim": float(np.std(sims)),
            "median_cosine_sim": float(np.median(sims)),
        })

    return {"pipeline": pipeline_results, "layers": layer_results}


def plot_and_save_pipeline_breakdown(data: Dict[str, Any], output_dir: str):
    """Side-effect function: Save CSV reports and render dual-axis line plot."""
    print("\n" + "="*60)
    print("Stage 1-A: Full 16-Step Multi-Metric Pipeline & Layer Breakdown Analysis")
    print("="*60)

    for row in data["pipeline"]:
        print(f"  [{row['step_name']:20s}] Cosine Sim: {row['mean_cosine_sim']:.4f} | Dot Prod: {row['mean_dot_product']:.2f} | L2 Dist: {row['mean_l2_distance']:.4f}")

    df_pipeline = pd.DataFrame(data["pipeline"])
    df_pipeline.to_csv(os.path.join(output_dir, "full_pipeline_step_breakdown.csv"), index=False)

    df_layer = pd.DataFrame(data["layers"])
    df_layer.to_csv(os.path.join(output_dir, "layerwise_cosine_breakdown.csv"), index=False)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    x_labels = df_pipeline["step_name"].values
    means_cos = df_pipeline["mean_cosine_sim"].values

    ax1.plot(x_labels, means_cos, "o-", color="crimson", lw=2.5, ms=7, label="Mean Cosine Sim")
    ax1.set_ylabel("Cosine Similarity", color="crimson", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="crimson")
    ax1.set_title("Full Sequence Pipeline Breakdown: Representation Geometry Shift Across All Layers & Projection", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax1.grid(True, ls="--", alpha=0.5)

    ax2 = ax1.twinx()
    means_l2 = df_pipeline["mean_l2_distance"].values
    ax2.plot(x_labels, means_l2, "s--", color="dodgerblue", lw=2, ms=6, label="Mean L2 Distance")
    ax2.set_ylabel("L2 Distance", color="dodgerblue", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="dodgerblue")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pipeline_step_lineplot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 3. Stage 1-B: Direction Preservation Analysis (Deranged Permutation)
# ==============================================================================

def compute_direction_preservation(
    pos_features: Dict[str, Any],
    neg_features: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Pure calculation of distance ratios using local RNG for control derangement."""
    pos_pre = pos_features["pipeline"][PipelineStep.LAYER12_LN.value]
    neg_pre = neg_features["pipeline"][PipelineStep.LAYER12_LN.value]

    pos_post = pos_features["pipeline"][PipelineStep.FINAL_L2NORM.value]
    neg_post = neg_features["pipeline"][PipelineStep.FINAL_L2NORM.value]

    dist_pre_neg = batch_l2_distance(pos_pre, neg_pre)
    dist_post_neg = batch_l2_distance(pos_post, neg_post)
    ratio_neg = dist_post_neg / (dist_pre_neg + 1e-8)

    # Pure local RNG to avoid global state mutation
    N = len(pos_pre)
    rng = np.random.default_rng(seed=seed)
    rand_idx = (np.arange(N) + rng.integers(1, N, size=N)) % N

    rand_pre = pos_pre[rand_idx]
    rand_post = pos_post[rand_idx]

    dist_pre_ctrl = batch_l2_distance(pos_pre, rand_pre)
    dist_post_ctrl = batch_l2_distance(pos_post, rand_post)
    ratio_ctrl = dist_post_ctrl / (dist_pre_ctrl + 1e-8)

    t_stat, p_val = stats.ttest_ind(ratio_neg, ratio_ctrl, equal_var=False)

    return {
        "negation_mean_dist_pre": float(np.mean(dist_pre_neg)),
        "negation_mean_dist_post": float(np.mean(dist_post_neg)),
        "negation_mean_ratio": float(np.mean(ratio_neg)),
        "control_mean_dist_pre": float(np.mean(dist_pre_ctrl)),
        "control_mean_dist_post": float(np.mean(dist_post_ctrl)),
        "control_mean_ratio": float(np.mean(ratio_ctrl)),
        "ttest_t_stat": float(t_stat),
        "ttest_p_value": float(p_val),
        "ratio_neg": ratio_neg,
        "ratio_ctrl": ratio_ctrl
    }


def plot_and_save_direction_preservation(report: Dict[str, Any], output_dir: str):
    """Side-effect function: Save JSON report and render histogram plot."""
    print("\n" + "="*60)
    print("Stage 1-B: Direction Preservation Analysis (Negation vs Random Control)")
    print("="*60)

    print(f"Negation Pairs  : Pre Dist={report['negation_mean_dist_pre']:.4f} -> Post={report['negation_mean_dist_post']:.4f} (Ratio={report['negation_mean_ratio']:.4f})")
    print(f"Control Pairs   : Pre Dist={report['control_mean_dist_pre']:.4f} -> Post={report['control_mean_dist_post']:.4f} (Ratio={report['control_mean_ratio']:.4f})")
    print(f"Welch's T-test  : t={report['ttest_t_stat']:.4f}, p-value={report['ttest_p_value']:.2e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(report["ratio_neg"], bins=35, alpha=0.6, color="crimson", edgecolor="black", label=f"Negation Pairs (Mean: {report['negation_mean_ratio']:.4f})")
    ax.hist(report["ratio_ctrl"], bins=35, alpha=0.6, color="gray", edgecolor="black", label=f"Control Random Pairs (Mean: {report['control_mean_ratio']:.4f})")
    ax.set_title(f"Direction Preservation: Negation vs Control Pairs (p={report['ttest_p_value']:.1e})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Distance Ratio (Post-Proj Dist / Pre-Proj Dist)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "direction_preservation_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    serializable_report = {k: v for k, v in report.items() if k not in ["ratio_neg", "ratio_ctrl"]}
    rpt_path = os.path.join(output_dir, "direction_preservation_report.json")
    with open(rpt_path, "w", encoding="utf-8") as f:
        json.dump(serializable_report, f, indent=2)


# ==============================================================================
# 4. Stage 1-C: Linear Probe & Template Shortcut Analysis
# ==============================================================================

def compute_linear_probe_and_subsets(
    pos_features: Dict[str, Any],
    neg_features: Dict[str, Any],
    pair_metadata: List[dict]
) -> Dict[str, Any]:
    """Pure calculation of Linear Probe 5-Fold Cross-Validation & Template Subsets."""
    n_pos = len(pair_metadata)
    n_neg = len(pair_metadata)
    y = np.array([1] * n_pos + [0] * n_neg)

    probe_results = {}
    step_keys = [PipelineStep.EMBEDDING.value, PipelineStep.LAYER12_LN.value, PipelineStep.FINAL_L2NORM.value]
    step_labels = ["Step 0 (Embed)", "Step 2 (Layer12+LN)", "Step 4 (Final L2Norm)"]

    for skey, slabel in zip(step_keys, step_labels):
        X_pos = pos_features["pipeline"][skey]
        X_neg = neg_features["pipeline"][skey]
        X = np.vstack([X_pos, X_neg])
        X_norm = l2_normalize(X)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_norm, y, cv=cv, scoring="accuracy")

        probe_results[slabel] = {
            "mean_accuracy": float(np.mean(scores)) * 100,
            "std_accuracy": float(np.std(scores)) * 100
        }

    df_meta = pd.DataFrame(pair_metadata)
    template_report = {}

    tmpl_key = MetadataKey.SOURCE_TEMPLATE.value
    if tmpl_key in df_meta.columns:
        unique_templates = df_meta[tmpl_key].unique()

        for tmpl in unique_templates:
            mask = (df_meta[tmpl_key] == tmpl).values
            n_sub = np.sum(mask)
            if n_sub < 10:
                continue

            y_sub = np.array([1] * n_sub + [0] * n_sub)
            X_pos_sub = pos_features["pipeline"][PipelineStep.FINAL_L2NORM.value][mask]
            X_neg_sub = neg_features["pipeline"][PipelineStep.FINAL_L2NORM.value][mask]
            X_sub = np.vstack([X_pos_sub, X_neg_sub])
            X_sub_norm = l2_normalize(X_sub)

            clf_sub = LogisticRegression(max_iter=1000, random_state=42)
            cv_sub = StratifiedKFold(n_splits=min(5, n_sub // 2), shuffle=True, random_state=42)
            scores_sub = cross_val_score(clf_sub, X_sub_norm, y_sub, cv=cv_sub, scoring="accuracy")

            sims_sub = batch_cosine_similarity(X_pos_sub, X_neg_sub)

            template_report[str(tmpl)] = {
                "sample_count": int(n_sub),
                "linear_probe_accuracy_pct": float(np.mean(scores_sub)) * 100,
                "linear_probe_accuracy_std_pct": float(np.std(scores_sub)) * 100,
                "mean_cosine_sim": float(np.mean(sims_sub))
            }

    return {"overall_probe": probe_results, "template_breakdown": template_report}


def plot_and_save_linear_probe(results: Dict[str, Any], output_dir: str):
    """Side-effect function: Save Linear Probe JSON report and render bar plot."""
    print("\n" + "="*60)
    print("Stage 1-C: Linear Probe & Sub-Dataset Template Shortcut Analysis")
    print("="*60)

    for slabel, info in results["overall_probe"].items():
        print(f"  [{slabel:22s}] Linear Probe Accuracy: {info['mean_accuracy']:.2f}% (±{info['std_accuracy']:.2f}%)")

    if results["template_breakdown"]:
        print("\n  --- Sub-dataset Breakdown by source_template ---")
        for tmpl, info in results["template_breakdown"].items():
            print(f"    [{tmpl:25s}] (N={info['sample_count']:5d}) Probe Acc: {info['linear_probe_accuracy_pct']:6.2f}% (±{info['linear_probe_accuracy_std_pct']:.1f}%) | Cosine Sim: {info['mean_cosine_sim']:.4f}")

    report_path = os.path.join(output_dir, "linear_probe_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(results["overall_probe"].keys(), [v["mean_accuracy"] for v in results["overall_probe"].values()],
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


# ==============================================================================
# 5. Stage 1-D: Intrinsic Dimensionality & Negation Subspace Analysis
# ==============================================================================

def compute_intrinsic_dimensionality(X: np.ndarray) -> Tuple[float, float]:
    """Compute Effective Rank (r_eff via spectral entropy) and Participation Ratio (PR)."""
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


def compute_pca_spectrum_compression(
    pos_features: Dict[str, Any],
    neg_features: Dict[str, Any]
) -> Dict[str, Any]:
    """Pure calculation of Intrinsic Dimensionality for full space and negation difference subspace."""
    X_pre = np.vstack([pos_features["pipeline"][PipelineStep.LAYER12_LN.value], neg_features["pipeline"][PipelineStep.LAYER12_LN.value]])
    X_post = np.vstack([pos_features["pipeline"][PipelineStep.FINAL_L2NORM.value], neg_features["pipeline"][PipelineStep.FINAL_L2NORM.value]])

    eff_rank_pre, pr_pre = compute_intrinsic_dimensionality(X_pre)
    eff_rank_post, pr_post = compute_intrinsic_dimensionality(X_post)

    diff_pre = pos_features["pipeline"][PipelineStep.LAYER12_LN.value] - neg_features["pipeline"][PipelineStep.LAYER12_LN.value]
    diff_post = pos_features["pipeline"][PipelineStep.FINAL_L2NORM.value] - neg_features["pipeline"][PipelineStep.FINAL_L2NORM.value]
    eff_rank_diff_pre, pr_diff_pre = compute_intrinsic_dimensionality(diff_pre)
    eff_rank_diff_post, pr_diff_post = compute_intrinsic_dimensionality(diff_post)

    n_comp = min(10, X_pre.shape[1], X_post.shape[1])
    pca_pre = PCA(n_components=n_comp).fit(X_pre)
    pca_post = PCA(n_components=n_comp).fit(X_post)

    return {
        "pre_effective_rank": eff_rank_pre,
        "pre_participation_ratio": pr_pre,
        "post_effective_rank": eff_rank_post,
        "post_participation_ratio": pr_post,
        "diff_subspace_pre_effective_rank": eff_rank_diff_pre,
        "diff_subspace_post_effective_rank": eff_rank_diff_post,
        "var_pre": pca_pre.explained_variance_ratio_.tolist(),
        "var_post": pca_post.explained_variance_ratio_.tolist(),
    }


def plot_and_save_pca_spectrum(report: Dict[str, Any], output_dir: str):
    """Side-effect function: Save PCA spectrum JSON report and render plot."""
    print("\n" + "="*60)
    print("Stage 1-D: Intrinsic Dimensionality & Negation Subspace Geometry")
    print("="*60)

    print(f"Full Space Pre-Proj (Layer12+LN) : Eff Rank={report['pre_effective_rank']:.2f}, PR={report['pre_participation_ratio']:.2f}, PC1={report['var_pre'][0]*100:.2f}%")
    print(f"Full Space Post-Proj (Final L2)  : Eff Rank={report['post_effective_rank']:.2f}, PR={report['post_participation_ratio']:.2f}, PC1={report['var_post'][0]*100:.2f}%")
    print(f"Negation Diff Subspace Pre-Proj  : Eff Rank={report['diff_subspace_pre_effective_rank']:.2f}")
    print(f"Negation Diff Subspace Post-Proj : Eff Rank={report['diff_subspace_post_effective_rank']:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    n_comp = len(report["var_pre"])
    indices = np.arange(1, n_comp + 1)
    ax.plot(indices, np.array(report["var_pre"]) * 100, "o-", color="seagreen", lw=2, label=f"Pre-Projection (r_eff={report['pre_effective_rank']:.1f}, PR={report['pre_participation_ratio']:.1f})")
    ax.plot(indices, np.array(report["var_post"]) * 100, "s-", color="crimson", lw=2, label=f"Post-Projection (r_eff={report['post_effective_rank']:.1f}, PR={report['post_participation_ratio']:.1f})")
    ax.set_xlabel("Principal Component Index", fontsize=11)
    ax.set_ylabel("Explained Variance Ratio (%)", fontsize=11)
    ax.set_title("PCA Variance Spectrum & Intrinsic Dimensionality", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "pca_spectrum_compression.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    serializable = {k: v for k, v in report.items() if k not in ["var_pre", "var_post"]}
    rpt_path = os.path.join(output_dir, "pca_spectrum_report.json")
    with open(rpt_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


# ==============================================================================
# 6. Stage 3: Projection Matrix SVD, Alignment & 10%-90% Spectrum Sweep
# ==============================================================================

def compute_projection_svd_ablation(
    model: nn.Module,
    pos_features: Dict[str, Any],
    neg_features: Dict[str, Any],
    target_token: str = "eot",
) -> Dict[str, Any]:
    """Pure calculation of SVD Singular Vector Alignment & 10%-90% Truncation Sweep."""
    text_tower = getattr(model, 'text', model)
    text_projection = getattr(text_tower, 'text_projection', None)

    if text_projection is None:
        return {}

    if isinstance(text_projection, nn.Linear):
        W_orig = text_projection.weight.T.detach().cpu().numpy()
    else:
        W_orig = text_projection.detach().cpu().numpy()

    U, S, Vh = np.linalg.svd(W_orig, full_matrices=False)

    pos_step2 = pos_features["pipeline"][PipelineStep.LAYER12_LN.value]
    neg_step2 = neg_features["pipeline"][PipelineStep.LAYER12_LN.value]

    diff_vecs = pos_step2 - neg_step2
    mean_d_neg = np.mean(diff_vecs, axis=0)
    norm_d_neg = mean_d_neg / (np.linalg.norm(mean_d_neg) + 1e-8)

    alignments = np.abs(np.dot(Vh, norm_d_neg))
    top_aligned_idx = np.argsort(alignments)[::-1]

    def compute_projected_similarity(pos_s2: np.ndarray, neg_s2: np.ndarray, W_proj: np.ndarray) -> float:
        p_proj = pos_s2 @ W_proj
        n_proj = neg_s2 @ W_proj
        return float(np.mean(batch_cosine_similarity(p_proj, n_proj)))

    pos_orig_norm = pos_features["final_l2norm"]
    neg_orig_norm = neg_features["final_l2norm"]
    sim_orig = float(np.mean(np.sum(pos_orig_norm * neg_orig_norm, axis=1)))

    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sweep_results = []

    for r in ratios:
        k_keep = max(1, int(round(len(S) * r)))

        S_top = S.copy()
        S_top[k_keep:] = 0.0
        W_top = U @ np.diag(S_top) @ Vh
        sim_top = compute_projected_similarity(pos_step2, neg_step2, W_top)

        S_bot = S.copy()
        S_bot[: (len(S) - k_keep)] = 0.0
        W_bot = U @ np.diag(S_bot) @ Vh
        sim_bot = compute_projected_similarity(pos_step2, neg_step2, W_bot)

        sweep_results.append({
            "keep_ratio": r,
            "k_singular_values": k_keep,
            "cosine_sim_top_k": sim_top,
            "cosine_sim_bottom_k": sim_bot
        })

    return {
        "target_token": target_token,
        "singular_values_top10": S[:10].tolist(),
        "top1_alignment": float(alignments[0]),
        "max_alignment": float(np.max(alignments)),
        "max_alignment_singular_vector_idx": int(top_aligned_idx[0]),
        "cosine_sim_original": sim_orig,
        "spectrum_sweep": sweep_results,
    }


def plot_and_save_svd_ablation(svd_report: Dict[str, Any], output_dir: str):
    """Side-effect function: Save SVD JSON report and render truncation sweep curve."""
    if not svd_report:
        print("Model does not have a text_projection matrix. Skipping SVD Ablation.")
        return

    print("\n" + "="*60)
    print(f"Stage 3: Projection SVD & Singular Value Spectrum Sweep (Target Token: '{svd_report['target_token']}')")
    print("="*60)

    print(f"  SVD Singular Values S (Top 5): {np.array(svd_report['singular_values_top10'][:5]).round(3)}")
    print(f"  Negation Direction Alignment with Top Singular Vector V1: {svd_report['top1_alignment']:.4f}")
    print(f"  Max Alignment: {svd_report['max_alignment']:.4f} (with Singular Vector #{svd_report['max_alignment_singular_vector_idx']+1})")

    print("\n  --- SVD Spectrum Sweep Results ---")
    for sr in svd_report["spectrum_sweep"]:
        print(f"    Keep Ratio {sr['keep_ratio']*100:2.0f}% ({sr['k_singular_values']:3d} SVs) | Top-k Sim: {sr['cosine_sim_top_k']:.4f} | Bottom-k Sim: {sr['cosine_sim_bottom_k']:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    pcts = [sr["keep_ratio"] * 100 for sr in svd_report["spectrum_sweep"]]
    sim_tops = [sr["cosine_sim_top_k"] for sr in svd_report["spectrum_sweep"]]
    sim_bots = [sr["cosine_sim_bottom_k"] for sr in svd_report["spectrum_sweep"]]

    ax.plot(pcts, sim_tops, "o-", color="crimson", lw=2, label="Keep Top-k Singular Values")
    ax.plot(pcts, sim_bots, "s--", color="dodgerblue", lw=2, label="Keep Bottom-k Singular Values")
    ax.axhline(svd_report["cosine_sim_original"], color="black", ls=":", label=f"Original W_proj Sim ({svd_report['cosine_sim_original']:.4f})")
    ax.set_xlabel("Singular Values Retained (%)", fontsize=11)
    ax.set_ylabel("Final Cosine Similarity", fontsize=11)
    ax.set_title("SVD Spectrum Sweep: Top-k vs Bottom-k Singular Value Truncation", fontsize=12, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()

    sweep_plot_path = os.path.join(output_dir, "projection_svd_spectrum_sweep.png")
    plt.savefig(sweep_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    svd_path = os.path.join(output_dir, "projection_svd_report.json")
    with open(svd_path, "w", encoding="utf-8") as f:
        json.dump(svd_report, f, indent=2)


# ==============================================================================
# 7. Stage 2: Micro-Batched Image-Text Retrieval & Symmetric Absence Evaluation
# ==============================================================================

def analyze_image_text_retrieval_metrics(
    model: nn.Module,
    tokenizer: Any,
    preprocess: Any,
    pair_metadata: List[dict],
    pos_texts: List[str],
    neg_texts: List[str],
    config: RetrievalConfig
):
    """Compute Binary MCQ Accuracy, Ranking Flip Rate, Tie Rate, and Pearson correlation."""
    print("\n" + "="*60)
    print("Stage 2: Micro-Batched Image-Text Retrieval & Symmetric Absence Evaluation")
    print("="*60)

    model.eval()

    print(f"Pre-encoding {len(pos_texts)} caption pairs in batches of {config.batch_size}...")
    pos_embs = []
    neg_embs = []

    for start in range(0, len(pos_texts), config.batch_size):
        end = min(start + config.batch_size, len(pos_texts))
        pos_tok = tokenizer(pos_texts[start:end]).to(config.device)
        neg_tok = tokenizer(neg_texts[start:end]).to(config.device)

        with torch.no_grad():
            p_emb = model.encode_text(pos_tok, normalize=True).float().cpu()
            n_emb = model.encode_text(neg_tok, normalize=True).float().cpu()

        pos_embs.append(p_emb)
        neg_embs.append(n_emb)

    pos_embs = torch.cat(pos_embs, dim=0)
    neg_embs = torch.cat(neg_embs, dim=0)

    image_to_indices = {}
    path_key = MetadataKey.IMAGE_PATH.value
    obj_key = MetadataKey.OBJECT_IN_IMAGE.value

    for i, meta in enumerate(pair_metadata):
        img_path = os.path.join(config.image_root, meta[path_key])
        if img_path not in image_to_indices:
            image_to_indices[img_path] = []
        image_to_indices[img_path].append(i)

    unique_paths = list(image_to_indices.keys())
    print(f"Processing {len(unique_paths)} unique images in micro-batches of {config.image_batch_size}...")

    results = []
    skipped_count = 0

    for b_start in range(0, len(unique_paths), config.image_batch_size):
        b_paths = unique_paths[b_start : b_start + config.image_batch_size]
        batch_tensors = []
        valid_paths = []

        for img_path in b_paths:
            if not os.path.exists(img_path):
                skipped_count += len(image_to_indices[img_path])
                continue

            try:
                img_t = preprocess(Image.open(img_path).convert("RGB"))
                batch_tensors.append(img_t)
                valid_paths.append(img_path)
            except Exception as e:
                skipped_count += len(image_to_indices[img_path])
                print(f"  [Warning] Failed to load image: {img_path} ({e})")
                continue

        if len(batch_tensors) == 0:
            continue

        imgs_stacked = torch.stack(batch_tensors, dim=0).to(config.device)
        with torch.no_grad():
            img_embs = model.encode_image(imgs_stacked, normalize=True).float().cpu()

        for idx_in_batch, img_path in enumerate(valid_paths):
            img_emb = img_embs[idx_in_batch : idx_in_batch + 1]
            pair_indices = image_to_indices[img_path]

            for idx in pair_indices:
                p_emb = pos_embs[idx : idx + 1]
                n_emb = neg_embs[idx : idx + 1]

                sim_pos = float(F.cosine_similarity(img_emb, p_emb).item())
                sim_neg = float(F.cosine_similarity(img_emb, n_emb).item())

                obj_in_img = pair_metadata[idx].get(obj_key, None)
                if isinstance(obj_in_img, str):
                    obj_in_img = obj_in_img.strip().lower() == "true"

                results.append({
                    MetadataKey.IMAGE_PATH.value: pair_metadata[idx][path_key],
                    MetadataKey.OBJECT_NAME.value: pair_metadata[idx].get(MetadataKey.OBJECT_NAME.value, ""),
                    MetadataKey.OBJECT_IN_IMAGE.value: obj_in_img,
                    MetadataKey.SOURCE_TEMPLATE.value: pair_metadata[idx].get(MetadataKey.SOURCE_TEMPLATE.value, ""),
                    "positive_caption": pos_texts[idx],
                    "negative_caption": neg_texts[idx],
                    "sim_image_pos": sim_pos,
                    "sim_image_neg": sim_neg,
                    "sim_diff": sim_pos - sim_neg,
                })

    if skipped_count > 0:
        print(f"  ⚠️ Warning: Skipped {skipped_count} pairs due to missing/corrupt image files.")

    if len(results) == 0:
        print("No valid images found. Skipping Retrieval Metrics.")
        return

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(config.output_dir, "image_text_similarity.csv"), index=False)

    sim_pos_arr = res_df["sim_image_pos"].values
    sim_neg_arr = res_df["sim_image_neg"].values
    sim_diff_arr = res_df["sim_diff"].values

    pearson_r = float(np.corrcoef(sim_pos_arr, sim_neg_arr)[0, 1])

    # 1. Object Present in Image
    in_img_mask = res_df[obj_key] == True
    if np.sum(in_img_mask) > 0:
        accuracy_in_img = float(np.mean(sim_pos_arr[in_img_mask] > sim_neg_arr[in_img_mask])) * 100
        flip_rate_in_img = float(np.mean(sim_neg_arr[in_img_mask] > sim_pos_arr[in_img_mask])) * 100
        tie_rate_in_img = float(np.mean(np.isclose(sim_pos_arr[in_img_mask], sim_neg_arr[in_img_mask], atol=1e-6))) * 100
    else:
        accuracy_in_img, flip_rate_in_img, tie_rate_in_img = 0.0, 0.0, 0.0

    # 2. Object Absent in Image
    out_img_mask = res_df[obj_key] == False
    if np.sum(out_img_mask) > 0:
        neg_pref_acc_out_img = float(np.mean(sim_neg_arr[out_img_mask] > sim_pos_arr[out_img_mask])) * 100
        pos_pref_rate_out_img = float(np.mean(sim_pos_arr[out_img_mask] > sim_neg_arr[out_img_mask])) * 100
        tie_rate_out_img = float(np.mean(np.isclose(sim_pos_arr[out_img_mask], sim_neg_arr[out_img_mask], atol=1e-6))) * 100
    else:
        neg_pref_acc_out_img, pos_pref_rate_out_img, tie_rate_out_img = 0.0, 0.0, 0.0

    retrieval_summary = {
        "total_pairs_evaluated": len(results),
        "total_skipped_pairs": skipped_count,
        "pearson_r": pearson_r,
        "mean_sim_pos": float(np.mean(sim_pos_arr)),
        "mean_sim_neg": float(np.mean(sim_neg_arr)),
        "mean_sim_diff": float(np.mean(sim_diff_arr)),
        "object_present_subgroup": {
            "count": int(np.sum(in_img_mask)),
            "positive_caption_accuracy_pct": accuracy_in_img,
            "ranking_flip_rate_pct": flip_rate_in_img,
            "tie_rate_pct": tie_rate_in_img,
        },
        "object_absent_subgroup": {
            "count": int(np.sum(out_img_mask)),
            "negative_caption_accuracy_pct": neg_pref_acc_out_img,
            "positive_caption_flip_rate_pct": pos_pref_rate_out_img,
            "tie_rate_pct": tie_rate_out_img,
        }
    }

    sum_path = os.path.join(config.output_dir, "retrieval_metrics_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_summary, f, indent=2)

    print("\n=== Retrieval Metrics Summary ===")
    print(f"  Evaluated Pairs                     : {len(results)} (Skipped: {skipped_count})")
    print(f"  Pearson r                           : {pearson_r:.4f}")
    print(f"  [Object Present] Pos Caption Acc    : {accuracy_in_img:.1f}% (Flip: {flip_rate_in_img:.1f}%, Tie: {tie_rate_in_img:.1f}%)")
    print(f"  [Object Absent ] Neg Caption Acc    : {neg_pref_acc_out_img:.1f}% (Pos Preference: {pos_pref_rate_out_img:.1f}%, Tie: {tie_rate_out_img:.1f}%)")


# ==============================================================================
# 8. Layer-wise PCA Grid Visualization
# ==============================================================================

def plot_and_save_layerwise_pca_grid(
    pos_features: Dict[str, Any],
    neg_features: Dict[str, Any],
    output_dir: str,
    target_token: str = "eot"
):
    """Generate single-canvas comparative PCA scatter grid (pca_grid_{target_token}.png)."""
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


# ==============================================================================
# Main Execution Entrypoint
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Negation Mechanism Analysis Refined 4th Edition (Clean Code)")
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

    path_k = MetadataKey.IMAGE_PATH.value
    obj_name_k = MetadataKey.OBJECT_NAME.value
    obj_in_img_k = MetadataKey.OBJECT_IN_IMAGE.value
    tmpl_k = MetadataKey.SOURCE_TEMPLATE.value

    if args.csv_path and os.path.exists(args.csv_path):
        df = pd.read_csv(args.csv_path).head(args.max_samples)
        pos_texts = df["positive_caption"].astype(str).tolist()
        neg_texts = df["negative_caption"].astype(str).tolist()
        for _, row in df.iterrows():
            meta = {
                path_k: str(row.get(path_k, "")),
                obj_name_k: str(row.get(obj_name_k, "")),
                obj_in_img_k: row.get(obj_in_img_k, None),
                tmpl_k: str(row.get(tmpl_k, ""))
            }
            if isinstance(meta[obj_in_img_k], str):
                meta[obj_in_img_k] = meta[obj_in_img_k].strip().lower() == "true"
            pair_metadata.append(meta)

    # Early Validation Check
    assert len(pos_texts) > 0, f"No valid caption pairs loaded! Check CSV path: {args.csv_path}"
    print(f"Loaded {len(pos_texts)} paired captions from: {args.csv_path}")

    # Load model & tokenizer
    print(f"Loading model {args.model} ({args.pretrained})...")
    model, preprocess, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    # 1. Single Unified Feature Extraction Pass
    print("\nExecuting Single-Pass Unified Feature Extraction...")
    pos_features = extract_all_features_unified(model, tokenizer, pos_texts, device, args.target_token, args.batch_size)
    neg_features = extract_all_features_unified(model, tokenizer, neg_texts, device, args.target_token, args.batch_size)

    # Verify Embedding Consistency against official encode_text
    assert_embedding_consistency(model, tokenizer, pos_texts, pos_features["final_l2norm"], device)

    # 2. Stage 1-A. Multi-Metric Pipeline & Layer Breakdown
    pipeline_data = compute_pipeline_and_layer_breakdown(pos_features, neg_features)
    plot_and_save_pipeline_breakdown(pipeline_data, args.output_dir)

    # 3. Stage 1-B. Direction Preservation Analysis
    dir_pres_report = compute_direction_preservation(pos_features, neg_features, seed=42)
    plot_and_save_direction_preservation(dir_pres_report, args.output_dir)

    # 4. Stage 1-C. Linear Probe & Sub-dataset Template Analysis
    probe_results = compute_linear_probe_and_subsets(pos_features, neg_features, pair_metadata)
    plot_and_save_linear_probe(probe_results, args.output_dir)

    # 5. Stage 1-D. Intrinsic Dimensionality & Negation Subspace
    pca_spec_report = compute_pca_spectrum_compression(pos_features, neg_features)
    plot_and_save_pca_spectrum(pca_spec_report, args.output_dir)

    # 6. Layer-wise PCA Scatter Grid Plot
    plot_and_save_layerwise_pca_grid(pos_features, neg_features, args.output_dir, args.target_token)

    # 7. Stage 3. Projection SVD & 10%-90% Spectrum Sweep
    svd_report = compute_projection_svd_ablation(model, pos_features, neg_features, target_token=args.target_token)
    plot_and_save_svd_ablation(svd_report, args.output_dir)

    # 8. Stage 2. Micro-Batched Retrieval Metrics & Symmetric Absence Evaluation
    if args.image_root:
        retrieval_cfg = RetrievalConfig(
            image_root=args.image_root,
            output_dir=args.output_dir,
            device=device,
            batch_size=args.batch_size,
            image_batch_size=64
        )
        analyze_image_text_retrieval_metrics(model, tokenizer, preprocess, pair_metadata, pos_texts, neg_texts, retrieval_cfg)

    print(f"\n✅ Refined 4th Edition Pipeline Analysis Complete! All results saved in: {args.output_dir}")
