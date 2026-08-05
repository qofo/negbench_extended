"""
BEAF Vision Encoder Mechanism Analysis Module.

Contains feature extraction and quantitative mechanism analyses:
- Vision Transformer layer breakdown
- SVD Projection Matrix spectrum sweep
- Stratified 5-Fold Linear Probing for object presence
- Direction preservation ratio & Welch's t-test
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.config import (
    l2_normalize,
    batch_cosine_similarity,
    batch_l2_distance,
    batch_dot_product,
)


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
            dummy_dim = proj.shape[1] if proj is not None else 512
            for l_idx in range(num_layers):
                layer_batches[l_idx].append(np.zeros((len(batch_paths), dummy_dim)))
            pre_proj_batches.append(np.zeros((len(batch_paths), dummy_dim)))
            final_l2_batches.append(np.zeros((len(batch_paths), dummy_dim)))
            continue

        stacked = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            cast_dtype = transformer.get_cast_dtype() if hasattr(transformer, "get_cast_dtype") else stacked.dtype

            if conv1 is not None:
                x = conv1(stacked)
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x = x.permute(0, 2, 1)
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

            x_perm = x.permute(1, 0, 2)
            for block in resblocks:
                x_perm = block(x_perm)
                hidden_states.append(x_perm.permute(1, 0, 2))

            pooled_layers = []
            for hs in hidden_states:
                cls_feat = hs[:, 0, :].float().cpu().numpy()
                pooled_layers.append(cls_feat)

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
        "layers":       layer_dict,
        "pre_proj":     np.concatenate(pre_proj_batches, axis=0),
        "final_l2norm": np.concatenate(final_l2_batches, axis=0),
        "loaded_flags": loaded_flags,
    }


def compute_vision_pipeline_breakdown(
    vis_orig: Dict[str, Any],
    vis_cf: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """Track layer-wise visual feature similarity & L2 distance shift between orig and cf images."""
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
    """Perform SVD on Visual Projection Matrix (W_vis) and evaluate Truncation Sweep."""
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

        W_top = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        p_orig_top = l2_normalize(f_pre_orig @ W_top)
        p_cf_top   = l2_normalize(f_pre_cf @ W_top)
        sim_top    = float(batch_cosine_similarity(p_orig_top, p_cf_top).mean())

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
    """Train 5-fold cross-validated Linear Probe on Vision Transformer features to classify object_in_image."""
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
    """Compute pre/post projection visual distance compression ratio and run Welch's t-test."""
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
