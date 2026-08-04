"""
BEAF Counterfactual v2 Comprehensive Analysis Script (1,778 Full Pairs).

Analyzes CLIP visual and text encoder mechanisms using paired counterfactual image data
(row 2n = original image with object vs row 2n+1 = edited image without object):

  1. Scatter Plots (4 types):
     - beaf_scatter_pos_vs_neg.png               COCO-style Pos vs Neg Cosine Sim (N=3,556)
     - beaf_scatter_delta_text_vs_delta_visual.png Delta-Delta 4-Quadrant Sensitivity Analysis
     - beaf_scatter_img_orig_vs_img_cf.png       Pure Visual Sensitivity Plot (I_orig vs I_cf)
     - beaf_scatter_by_object_category.png       Per-Object Category Stratified Scatter Subplots

  2. Vision Encoder Mechanism Analyses:
     - beaf_vision_pipeline_breakdown.csv & .png Layer-wise ViT Visual Feature Shift
     - beaf_vision_svd_sweep.png & .json          Visual Projection Matrix SVD & Alignment
     - beaf_vision_linear_probe.json             5-fold CV Linear Probing for Object Presence
     - beaf_vision_direction_preservation.json   Pre/Post Projection Distance Ratio & t-test
     - beaf_v2_summary_report.json               Overall Comprehensive Quantitative Metrics

Usage:
  python -m benchmarks.src.analysis.run_beaf_analysis_v2 \\
      --csv_path beaf_counterfactual_6col.csv \\
      --image_root "" \\
      --output_dir logs/evaluation/beaf_counterfactual_v2/openai_vit_b32 \\
      --model ViT-B-32 \\
      --pretrained openai
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from PIL import Image

# --------------------------------------------------------------------------- #
# Path bootstrap
# --------------------------------------------------------------------------- #
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS_SRC = os.path.abspath(os.path.join(_FILE_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_FILE_DIR, "..", "..", ".."))
for _p in [_BENCHMARKS_SRC, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import open_clip
from analysis.config import (
    l2_normalize,
    batch_cosine_similarity,
    batch_l2_distance,
    batch_dot_product,
)
from analysis.extractor import extract_all_features_unified


# =========================================================================== #
# Data Loader: 2n & 2n+1 Pairing (1,778 Full Pairs)
# =========================================================================== #

def load_beaf_paired_dataset(csv_path: str, image_root: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load beaf_counterfactual_6col.csv and construct 1,778 exact pairs
    by pairing row 2n (object_in_image=True) and row 2n+1 (object_in_image=False).

    Returns:
        df_raw   : full raw dataframe with resolved image paths (3,556 rows)
        df_pairs : paired dataframe with 1,778 rows containing:
                   pair_id, object_name, orig_path, cf_path,
                   positive_caption, negative_caption, source_template
    """
    df = pd.read_csv(csv_path)

    # Resolve image paths
    if image_root:
        df["abs_image_path"] = df["image_path"].apply(lambda p: os.path.join(image_root, p))
    else:
        df["abs_image_path"] = df["image_path"]

    def _to_bool(v) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return False

    df["object_in_image"] = df["object_in_image"].apply(_to_bool)

    pairs = []
    num_rows = len(df)
    for i in range(0, num_rows - 1, 2):
        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]

        # Determine which row is orig (True) vs cf (False)
        if row1["object_in_image"] and not row2["object_in_image"]:
            orig_row, cf_row = row1, row2
        elif not row1["object_in_image"] and row2["object_in_image"]:
            orig_row, cf_row = row2, row1
        else:
            # Fallback if both True or both False
            orig_row, cf_row = row1, row2

        pairs.append({
            "pair_id":           i // 2,
            "object_name":       str(orig_row.get("object_name", "")),
            "orig_path":         orig_row["abs_image_path"],
            "cf_path":           cf_row["abs_image_path"],
            "positive_caption":  str(orig_row["positive_caption"]),
            "negative_caption":  str(orig_row["negative_caption"]),
            "source_template":   str(orig_row.get("source_template", "")),
        })

    df_pairs = pd.DataFrame(pairs)
    return df, df_pairs


# =========================================================================== #
# Vision Feature Extraction Engine
# =========================================================================== #

def extract_vision_features_unified(
    model: nn.Module,
    preprocess: Any,
    image_paths: List[str],
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Extract intermediate Vision Transformer layer representations and pipeline steps.

    Returns:
        Dict containing:
          - "layers": Dict[str, np.ndarray] layer0 (embed) to layer12
          - "pre_proj": np.ndarray (before visual projection)
          - "final_l2norm": np.ndarray (final L2-normalized image embedding)
          - "loaded_flags": List[bool]
    """
    model.eval()
    visual = getattr(model, "visual", model)

    # Dissect OpenCLIP visual architecture
    conv1 = getattr(visual, "conv1", None)
    class_embedding = getattr(visual, "class_embedding", None)
    positional_embedding = getattr(visual, "positional_embedding", None)
    ln_pre = getattr(visual, "ln_pre", None)
    transformer = getattr(visual, "transformer", None)
    ln_post = getattr(visual, "ln_post", None)
    proj = getattr(visual, "proj", None)

    resblocks = transformer.resblocks if transformer is not None else []
    num_layers = 1 + len(resblocks)

    layer_batches = [[] for _ in range(num_layers)]
    pre_proj_batches = []
    final_l2_batches = []
    loaded_flags = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors = []
        valid_idx = []

        for j, p in enumerate(batch_paths):
            if not os.path.exists(p):
                loaded_flags.append(False)
                continue
            try:
                t = preprocess(Image.open(p).convert("RGB"))
                tensors.append(t)
                valid_idx.append(j)
                loaded_flags.append(True)
            except Exception as ex:
                loaded_flags.append(False)

        if len(tensors) == 0:
            # All images in batch failed
            dummy_dim = proj.shape[1] if proj is not None else 512
            for l_idx in range(num_layers):
                layer_batches[l_idx].append(np.zeros((len(batch_paths), dummy_dim)))
            pre_proj_batches.append(np.zeros((len(batch_paths), dummy_dim)))
            final_l2_batches.append(np.zeros((len(batch_paths), dummy_dim)))
            continue

        stacked = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            cast_dtype = transformer.get_cast_dtype() if hasattr(transformer, "get_cast_dtype") else stacked.dtype

            # ViT Stem forward
            if conv1 is not None:
                x = conv1(stacked)  # [B, C, H, W]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, N_grid]
                x = x.permute(0, 2, 1)  # [B, N_grid, C]
            else:
                x = stacked

            if class_embedding is not None:
                class_emb = class_embedding.to(x.dtype)
                if class_emb.ndim == 1:
                    class_emb = class_emb.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
                x = torch.cat([class_emb, x], dim=1)

            if positional_embedding is not None:
                x = x + positional_embedding.to(x.dtype)

            if ln_pre is not None:
                x = ln_pre(x)

            hidden_states = [x]

            # ViT Blocks forward
            x_perm = x.permute(1, 0, 2)
            for block in resblocks:
                x_perm = block(x_perm)
                hidden_states.append(x_perm.permute(1, 0, 2))

            # Layer pooling (extract CLS token = index 0)
            pooled_layers = []
            for hs in hidden_states:
                cls_feat = hs[:, 0, :].float().cpu().numpy()
                pooled_layers.append(cls_feat)

            # Post LN & Proj
            x_post = hidden_states[-1][:, 0, :]
            if ln_post is not None:
                x_post = ln_post(x_post)
            pre_proj_feat = x_post.float().cpu().numpy()

            if proj is not None:
                if isinstance(proj, torch.Tensor):
                    x_proj = x_post.to(proj.dtype) @ proj
                else:
                    x_proj = proj(x_post)
            else:
                x_proj = x_post

            final_l2_feat = F.normalize(x_proj.float(), dim=-1).cpu().numpy()

            # Align batch indexing
            embed_dim = final_l2_feat.shape[1]
            pre_dim = pre_proj_feat.shape[1]

            for l_idx in range(num_layers):
                l_arr = np.zeros((len(batch_paths), pooled_layers[l_idx].shape[1]))
                vi = 0
                for j in range(len(batch_paths)):
                    if j in valid_idx:
                        l_arr[j] = pooled_layers[l_idx][vi]
                        if l_idx == num_layers - 1:
                            vi += 1
                layer_batches[l_idx].append(l_arr)

            pre_arr = np.zeros((len(batch_paths), pre_dim))
            post_arr = np.zeros((len(batch_paths), embed_dim))
            vi = 0
            for j in range(len(batch_paths)):
                if j in valid_idx:
                    pre_arr[j] = pre_proj_feat[vi]
                    post_arr[j] = final_l2_feat[vi]
                    vi += 1

            pre_proj_batches.append(pre_arr)
            final_l2_batches.append(post_arr)

    layer_dict = {}
    for l_idx, feats in enumerate(layer_batches):
        name = "Embedding" if l_idx == 0 else f"Layer {l_idx}"
        layer_dict[name] = np.concatenate(feats, axis=0)

    return {
        "layers":        layer_dict,
        "pre_proj":      np.concatenate(pre_proj_batches, axis=0),
        "final_l2norm":  np.concatenate(final_l2_batches, axis=0),
        "loaded_flags":  loaded_flags,
    }


# =========================================================================== #
# Scatter Plot Visualizations
# =========================================================================== #

def render_scatter_pos_vs_neg(
    all_img_embs: np.ndarray,
    all_pos_embs: np.ndarray,
    all_neg_embs: np.ndarray,
    all_obj_in_img: np.ndarray,
    output_dir: str
):
    """
    COCO-style Pos vs Neg Cosine Similarity Scatter Plot (N=3,556).
    """
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

    # y=x line
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

    # Color code by quadrant
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

    ax.set_xlabel("Text Negation Score  sim(orig, pos) - sim(orig, neg)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Visual Change Score  sim(orig, pos) - sim(cf, pos)", fontsize=10, fontweight="bold")
    ax.set_title("Delta-Delta Quadrant Scatter: Text Negation vs Visual Change Sensitivity", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)

    # Annotate Quadrants
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
    """
    Pure Visual Sensitivity Scatter:
      X = sim(I_orig, T_pos)
      Y = sim(I_cf, T_pos)
    """
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
    """
    Subplot grid showing Pos vs Neg scatter across Top 6 Object Categories.
    """
    df_pairs["sim_pos"] = sim_orig_pos
    df_pairs["sim_neg"] = sim_orig_neg

    top_objs = df_pairs["object_name"].value_counts().head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, obj in enumerate(top_objs):
        ax = axes[idx]
        sub = df_pairs[df_pairs["object_name"] == obj]

        ax.scatter(sub["sim_pos"], sub["sim_neg"], c="dodgerblue", alpha=0.6, s=24)

        min_v = min(sub["sim_pos"].min(), sub["sim_neg"].min()) - 0.01
        max_v = max(sub["sim_pos"].max(), sub["sim_neg"].max()) + 0.01
        ax.plot([min_v, max_v], [min_v, max_v], color="crimson", ls="--", lw=1.2)

        r_sub, _ = stats.pearsonr(sub["sim_pos"], sub["sim_neg"])
        ax.set_title(f"Object: {obj} (n={len(sub)})\nPearson r={r_sub:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("sim(pos)", fontsize=9)
        ax.set_ylabel("sim(neg)", fontsize=9)
        ax.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "beaf_scatter_by_object_category.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_scatter_by_object_category.png")


# =========================================================================== #
# Vision Encoder Mechanism Analyses
# =========================================================================== #

def compute_vision_pipeline_breakdown(
    vis_orig: Dict[str, Any],
    vis_cf: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Track layer-wise visual feature similarity & L2 distance shift between orig and cf images.
    """
    layer_results = []
    layer_names = list(vis_orig["layers"].keys())

    for l_name in layer_names:
        f_orig = vis_orig["layers"][l_name]
        f_cf   = vis_cf["layers"][l_name]

        cos_sims = batch_cosine_similarity(f_orig, f_cf)
        l2_dists = batch_l2_distance(f_orig, f_cf)
        dot_prods = batch_dot_product(f_orig, f_cf)

        layer_results.append({
            "layer": l_name,
            "mean_cosine_sim": float(np.mean(cos_sims)),
            "std_cosine_sim": float(np.std(cos_sims)),
            "mean_l2_distance": float(np.mean(l2_dists)),
            "mean_dot_product": float(np.mean(dot_prods))
        })

    # Add Pre-proj and Final L2Norm
    cos_pre = batch_cosine_similarity(vis_orig["pre_proj"], vis_cf["pre_proj"])
    l2_pre  = batch_l2_distance(vis_orig["pre_proj"], vis_cf["pre_proj"])
    layer_results.append({
        "layer": "Pre-Projection (LN)",
        "mean_cosine_sim": float(np.mean(cos_pre)),
        "std_cosine_sim": float(np.std(cos_pre)),
        "mean_l2_distance": float(np.mean(l2_pre)),
        "mean_dot_product": float(np.mean(batch_dot_product(vis_orig["pre_proj"], vis_cf["pre_proj"])))
    })

    cos_final = batch_cosine_similarity(vis_orig["final_l2norm"], vis_cf["final_l2norm"])
    l2_final  = batch_l2_distance(vis_orig["final_l2norm"], vis_cf["final_l2norm"])
    layer_results.append({
        "layer": "+Final L2Norm",
        "mean_cosine_sim": float(np.mean(cos_final)),
        "std_cosine_sim": float(np.std(cos_final)),
        "mean_l2_distance": float(np.mean(l2_final)),
        "mean_dot_product": float(np.mean(batch_dot_product(vis_orig["final_l2norm"], vis_cf["final_l2norm"])))
    })

    df_res = pd.DataFrame(layer_results)
    df_res.to_csv(os.path.join(output_dir, "beaf_vision_pipeline_breakdown.csv"), index=False)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x_labs = df_res["layer"].values
    c_vals = df_res["mean_cosine_sim"].values

    ax1.plot(x_labs, c_vals, "o-", color="darkgreen", lw=2.5, ms=7, label="Mean Cosine Sim")
    ax1.set_ylabel("Cosine Similarity (orig ↔ cf image)", color="darkgreen", fontsize=10, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="darkgreen")
    ax1.set_title("Vision Encoder Transformer Layer Breakdown: Visual Feature Shift on Object Removal", fontsize=11, fontweight="bold")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    ax1.grid(True, ls="--", alpha=0.5)

    ax2 = ax1.twinx()
    l2_vals = df_res["mean_l2_distance"].values
    ax2.plot(x_labs, l2_vals, "s--", color="darkorange", lw=2, ms=6, label="Mean L2 Distance")
    ax2.set_ylabel("L2 Distance", color="darkorange", fontsize=10, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beaf_vision_pipeline_breakdown.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("  Saved: beaf_vision_pipeline_breakdown.csv & .png")
    return {"breakdown": layer_results}


def compute_vision_svd_sweep(
    model: nn.Module,
    vis_orig: Dict[str, Any],
    vis_cf: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Perform SVD on Visual Projection Matrix (W_vis) and evaluate Truncation Sweep.
    """
    visual = getattr(model, "visual", model)
    proj = getattr(visual, "proj", None)

    if proj is None:
        print("  [Notice] Vision tower has no explicit proj tensor. Skipping Vision SVD Sweep.")
        return {}

    if isinstance(proj, nn.Linear):
        W = proj.weight.detach().cpu().numpy().T
    elif isinstance(proj, torch.Tensor):
        W = proj.detach().cpu().numpy()
    else:
        return {}

    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    f_pre_orig = vis_orig["pre_proj"]
    f_pre_cf   = vis_cf["pre_proj"]
    diff_pre   = f_pre_orig - f_pre_cf
    diff_norm  = l2_normalize(diff_pre)

    alignments = np.abs(diff_norm @ U)
    mean_alignments = np.mean(alignments, axis=0)

    sim_orig = batch_cosine_similarity(vis_orig["final_l2norm"], vis_cf["final_l2norm"]).mean()

    sweep_results = []
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    d_in = W.shape[0]
    for r in ratios:
        k = max(1, int(d_in * r))

        # Top-k
        W_top = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        p_orig_top = l2_normalize(f_pre_orig @ W_top)
        p_cf_top   = l2_normalize(f_pre_cf @ W_top)
        sim_top    = float(batch_cosine_similarity(p_orig_top, p_cf_top).mean())

        # Bottom-k
        W_bot = U[:, -k:] @ np.diag(S[-k:]) @ Vt[-k:, :]
        p_orig_bot = l2_normalize(f_pre_orig @ W_bot)
        p_cf_bot   = l2_normalize(f_pre_cf @ W_bot)
        sim_bot    = float(batch_cosine_similarity(p_orig_bot, p_cf_bot).mean())

        sweep_results.append({
            "keep_ratio": r,
            "k_singular_values": k,
            "cosine_sim_top_k": sim_top,
            "cosine_sim_bottom_k": sim_bot
        })

    report = {
        "singular_values_top10": S[:10].tolist(),
        "mean_alignment_with_top1_singular_vector": float(mean_alignments[0]),
        "max_alignment_singular_vector_idx": int(np.argmax(mean_alignments)),
        "cosine_sim_original": float(sim_orig),
        "spectrum_sweep": sweep_results
    }

    with open(os.path.join(output_dir, "beaf_vision_svd_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Plot Sweep
    fig, ax = plt.subplots(figsize=(8, 5))
    pcts = [sr["keep_ratio"] * 100 for sr in sweep_results]
    tops = [sr["cosine_sim_top_k"] for sr in sweep_results]
    bots = [sr["cosine_sim_bottom_k"] for sr in sweep_results]

    ax.plot(pcts, tops, "o-", color="purple", lw=2, label="Keep Top-k Singular Values")
    ax.plot(pcts, bots, "s--", color="teal", lw=2, label="Keep Bottom-k Singular Values")
    ax.axhline(sim_orig, color="black", ls=":", label=f"Original W_vis Sim ({sim_orig:.4f})")

    ax.set_xlabel("Singular Values Retained (%)", fontsize=11)
    ax.set_ylabel("Final Vision Cosine Similarity", fontsize=11)
    ax.set_title("Visual Projection SVD Spectrum Sweep: Top-k vs Bottom-k Truncation", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beaf_vision_svd_sweep.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("  Saved: beaf_vision_svd_sweep.png & .json")
    return report


def compute_vision_linear_probe(
    vis_orig: Dict[str, Any],
    vis_cf: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Train 5-fold cross-validated Linear Probe on Vision Transformer features
    to classify object_in_image (True vs False).
    """
    n_orig = len(vis_orig["pre_proj"])
    n_cf   = len(vis_cf["pre_proj"])
    y = np.array([1] * n_orig + [0] * n_cf)

    probe_results = {}

    for l_name in list(vis_orig["layers"].keys()) + ["Pre-Projection", "+Final L2Norm"]:
        if l_name in vis_orig["layers"]:
            X_orig = vis_orig["layers"][l_name]
            X_cf   = vis_cf["layers"][l_name]
        elif l_name == "Pre-Projection":
            X_orig = vis_orig["pre_proj"]
            X_cf   = vis_cf["pre_proj"]
        else:
            X_orig = vis_orig["final_l2norm"]
            X_cf   = vis_cf["final_l2norm"]

        X = np.vstack([X_orig, X_cf])
        X_norm = l2_normalize(X)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_norm, y, cv=cv, scoring="accuracy")

        probe_results[l_name] = {
            "mean_accuracy_pct": float(np.mean(scores) * 100),
            "std_accuracy_pct":  float(np.std(scores) * 100),
        }

    with open(os.path.join(output_dir, "beaf_vision_linear_probe.json"), "w", encoding="utf-8") as f:
        json.dump(probe_results, f, indent=2)

    print("  Saved: beaf_vision_linear_probe.json")
    return probe_results


def compute_vision_direction_preservation(
    vis_orig: Dict[str, Any],
    vis_cf: Dict[str, Any],
    output_dir: str,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Compute pre/post projection visual distance compression ratio
    and run Welch's t-test against random control image pairs.
    """
    orig_pre  = vis_orig["pre_proj"]
    cf_pre    = vis_cf["pre_proj"]
    orig_post = vis_orig["final_l2norm"]
    cf_post   = vis_cf["final_l2norm"]

    dist_pre_pair  = batch_l2_distance(orig_pre, cf_pre)
    dist_post_pair = batch_l2_distance(orig_post, cf_post)
    ratio_pair     = dist_post_pair / (dist_pre_pair + 1e-8)

    N = len(orig_pre)
    rng = np.random.default_rng(seed=seed)
    rand_idx = (np.arange(N) + rng.integers(1, N, size=N)) % N

    rand_pre  = orig_pre[rand_idx]
    rand_post = orig_post[rand_idx]

    dist_pre_ctrl  = batch_l2_distance(orig_pre, rand_pre)
    dist_post_ctrl = batch_l2_distance(orig_post, rand_post)
    ratio_ctrl     = dist_post_ctrl / (dist_pre_ctrl + 1e-8)

    t_stat, p_val = stats.ttest_ind(ratio_pair, ratio_ctrl, equal_var=False)

    report = {
        "counterfactual_pair_mean_ratio": float(np.mean(ratio_pair)),
        "control_pair_mean_ratio":        float(np.mean(ratio_ctrl)),
        "welch_ttest_t_stat":             float(t_stat),
        "welch_ttest_p_value":            float(p_val)
    }

    with open(os.path.join(output_dir, "beaf_vision_direction_preservation.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("  Saved: beaf_vision_direction_preservation.json")
    return report


# =========================================================================== #
# Main Pipeline
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="BEAF Counterfactual v2 Analysis Pipeline (1,778 Full Pairs)"
    )
    parser.add_argument("--csv_path",   type=str, required=True)
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="logs/evaluation/beaf_counterfactual_v2/openai_vit_b32")
    parser.add_argument("--model",      type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--img_batch",  type=int, default=64)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  BEAF Counterfactual v2 Comprehensive Analysis Pipeline")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Model      : {args.model} ({args.pretrained})")
    print(f"  CSV        : {args.csv_path}")
    print(f"  Output dir : {args.output_dir}")
    print("=" * 60)

    # 1. Load Data
    print("\n[Step 1] Loading BEAF CSV & Constructing 1,778 Pairs ...")
    df_raw, df_pairs = load_beaf_paired_dataset(args.csv_path, args.image_root)
    n_pairs = len(df_pairs)
    print(f"  Raw rows : {len(df_raw)}")
    print(f"  Exact Counterfactual Pairs (orig ↔ cf) : {n_pairs}")

    # 2. Load Model
    print("\n[Step 2] Loading OpenCLIP Model ...")
    model, preprocess, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    # 3. Extract Text Embeddings
    print("\n[Step 3] Extracting Text Features ...")
    pos_texts = df_pairs["positive_caption"].tolist()
    neg_texts = df_pairs["negative_caption"].tolist()

    pos_feat_dict = extract_all_features_unified(model, tokenizer, pos_texts, device, "eot", args.batch_size)
    neg_feat_dict = extract_all_features_unified(model, tokenizer, neg_texts, device, "eot", args.batch_size)

    pos_embs = pos_feat_dict["final_l2norm"]
    neg_embs = neg_feat_dict["final_l2norm"]

    # 4. Extract Vision Features
    print("\n[Step 4] Extracting Vision Features for Original & Counterfactual Images ...")
    orig_paths = df_pairs["orig_path"].tolist()
    cf_paths   = df_pairs["cf_path"].tolist()

    vis_orig = extract_vision_features_unified(model, preprocess, orig_paths, device, args.img_batch)
    vis_cf   = extract_vision_features_unified(model, preprocess, cf_paths, device, args.img_batch)

    orig_embs = vis_orig["final_l2norm"]
    cf_embs   = vis_cf["final_l2norm"]

    # Combine all image embeddings for N=3,556 scatter plot
    all_img_embs = np.vstack([orig_embs, cf_embs])
    all_pos_embs = np.vstack([pos_embs, pos_embs])
    all_neg_embs = np.vstack([neg_embs, neg_embs])
    all_obj_flags = np.array([True] * n_pairs + [False] * n_pairs)

    # 5. Render Scatter Plots
    print("\n[Step 5] Rendering Scatter Plots ...")
    render_scatter_pos_vs_neg(all_img_embs, all_pos_embs, all_neg_embs, all_obj_flags, args.output_dir)

    sim_orig_pos = batch_cosine_similarity(orig_embs, pos_embs)
    sim_orig_neg = batch_cosine_similarity(orig_embs, neg_embs)
    sim_cf_pos   = batch_cosine_similarity(cf_embs, pos_embs)

    render_scatter_delta_quadrant(sim_orig_pos, sim_orig_neg, sim_cf_pos, args.output_dir)
    render_scatter_img_orig_vs_img_cf(sim_orig_pos, sim_cf_pos, args.output_dir)
    render_scatter_by_object_category(df_pairs, sim_orig_pos, sim_orig_neg, args.output_dir)

    # 6. Vision Encoder Mechanism Analyses
    print("\n[Step 6] Executing Vision Encoder Mechanism Analyses ...")
    vis_breakdown = compute_vision_pipeline_breakdown(vis_orig, vis_cf, args.output_dir)
    vis_svd       = compute_vision_svd_sweep(model, vis_orig, vis_cf, args.output_dir)
    vis_probe     = compute_vision_linear_probe(vis_orig, vis_cf, args.output_dir)
    vis_dir_pres  = compute_vision_direction_preservation(vis_orig, vis_cf, args.output_dir, seed=args.seed)

    # 7. Comprehensive JSON Summary Report
    print("\n[Step 7] Writing Comprehensive Summary Report ...")
    r_val, _ = stats.pearsonr(batch_cosine_similarity(all_img_embs, all_pos_embs), batch_cosine_similarity(all_img_embs, all_neg_embs))
    rho_val, _ = stats.spearmanr(batch_cosine_similarity(all_img_embs, all_pos_embs), batch_cosine_similarity(all_img_embs, all_neg_embs))

    summary_v2 = {
        "model":             args.model,
        "pretrained":        args.pretrained,
        "n_raw_rows":        len(df_raw),
        "n_exact_pairs":     n_pairs,
        "scatter_pos_vs_neg": {
            "n_total":          len(all_img_embs),
            "pearson_r":        round(float(r_val), 6),
            "spearman_rho":     round(float(rho_val), 6),
        },
        "delta_delta_quadrants": {
            "q1_both_sensitive_pct": round(float(np.sum((sim_orig_pos - sim_orig_neg > 0) & (sim_orig_pos - sim_cf_pos > 0)) / n_pairs * 100), 2),
            "q2_visual_only_pct":    round(float(np.sum((sim_orig_pos - sim_orig_neg <= 0) & (sim_orig_pos - sim_cf_pos > 0)) / n_pairs * 100), 2),
            "q3_neither_pct":        round(float(np.sum((sim_orig_pos - sim_orig_neg <= 0) & (sim_orig_pos - sim_cf_pos <= 0)) / n_pairs * 100), 2),
            "q4_text_only_failure_pct": round(float(np.sum((sim_orig_pos - sim_orig_neg > 0) & (sim_orig_pos - sim_cf_pos <= 0)) / n_pairs * 100), 2),
        },
        "pure_visual_sensitivity": {
            "pct_pairs_similarity_dropped": round(float(np.sum(sim_cf_pos < sim_orig_pos) / n_pairs * 100), 2),
            "mean_sim_orig_pos":            round(float(sim_orig_pos.mean()), 6),
            "mean_sim_cf_pos":              round(float(sim_cf_pos.mean()), 6),
        },
        "vision_direction_preservation": vis_dir_pres,
        "vision_linear_probe":           vis_probe,
    }

    with open(os.path.join(args.output_dir, "beaf_v2_summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary_v2, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  BEAF v2 Analysis Complete! All artifacts saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
