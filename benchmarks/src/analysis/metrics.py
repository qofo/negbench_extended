"""
Representation Geometry & Empirical Evaluation Metrics Engine.

This module provides pure functional metric computation for analyzing
CLIP's negation processing mechanisms across 6 analytical dimensions:
1. Multi-metric sequence breakdown (Cosine, Dot Product, L2 Distance)
2. Direction preservation & Welch's two-sample t-test with deranged control pairs
3. Stratified k-fold linear probing & template shortcut estimation
4. Intrinsic dimensionality estimation (Spectral Entropy Effective Rank & Participation Ratio)
5. Projection matrix Singular Value Decomposition (SVD) & 10%-90% spectrum truncation sweep
6. Micro-batched cross-modal retrieval & symmetric object absence evaluation
"""

import os
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from PIL import Image

from .config import (
    PipelineStep,
    MetadataKey,
    RetrievalConfig,
    l2_normalize,
    batch_cosine_similarity,
    batch_dot_product,
    batch_l2_distance
)


# ==============================================================================
# Stage 1-A: Multi-Metric Pipeline & Layer Breakdown
# ==============================================================================

def compute_pipeline_and_layer_breakdown(
    pos_features: Dict[str, Any],
    neg_features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Quantify geometric shift metrics across 16 sequential transformation steps.

    Args:
        pos_features (Dict[str, Any]): Feature dictionary for positive captions.
        neg_features (Dict[str, Any]): Feature dictionary for negative captions.

    Returns:
        Dict[str, Any]: Step-wise multi-metric and layer-wise similarity statistics.
    """
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


# ==============================================================================
# Stage 1-B: Direction Preservation Analysis
# ==============================================================================

def compute_direction_preservation(
    pos_features: Dict[str, Any],
    neg_features: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluate directional distance preservation ratio against a deranged random control baseline.

    Args:
        pos_features (Dict[str, Any]): Positive caption features.
        neg_features (Dict[str, Any]): Negative caption features.
        seed (int): Local seed for deranged permutation generation.

    Returns:
        Dict[str, Any]: Distance compression ratios and Welch's two-sample t-test statistics.
    """
    pos_pre = pos_features["pipeline"][PipelineStep.LAYER12_LN.value]
    neg_pre = neg_features["pipeline"][PipelineStep.LAYER12_LN.value]

    pos_post = pos_features["pipeline"][PipelineStep.FINAL_L2NORM.value]
    neg_post = neg_features["pipeline"][PipelineStep.FINAL_L2NORM.value]

    dist_pre_neg = batch_l2_distance(pos_pre, neg_pre)
    dist_post_neg = batch_l2_distance(pos_post, neg_post)
    ratio_neg = dist_post_neg / (dist_pre_neg + 1e-8)

    # Deranged permutation generation via local RNG
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


# ==============================================================================
# Stage 1-C: Linear Probe & Template Shortcut Analysis
# ==============================================================================

def compute_linear_probe_and_subsets(
    pos_features: Dict[str, Any],
    neg_features: Dict[str, Any],
    pair_metadata: List[dict],
    fit_intercept: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate linear separability of negation representations using 5-fold cross-validated logistic regression.

    Args:
        pos_features (Dict[str, Any]): Positive caption features.
        neg_features (Dict[str, Any]): Negative caption features.
        pair_metadata (List[dict]): Metadata dictionary for sub-dataset template decomposition.
        fit_intercept (bool): Whether to include bias / intercept term in LogisticRegression (default: True).

    Returns:
        Dict[str, Any]: Cross-validated probing accuracies and template-specific probe breakdowns.
    """
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

        clf = LogisticRegression(max_iter=1000, random_state=42, fit_intercept=fit_intercept)
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

            clf_sub = LogisticRegression(max_iter=1000, random_state=42, fit_intercept=fit_intercept)
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


# ==============================================================================
# Stage 1-D: Intrinsic Dimensionality & Negation Subspace Analysis
# ==============================================================================

def compute_intrinsic_dimensionality(X: np.ndarray) -> Tuple[float, float]:
    """
    Compute Effective Rank (r_eff via spectral entropy) and Participation Ratio (PR).

    Args:
        X (np.ndarray): Feature matrix of shape (N, D).

    Returns:
        Tuple[float, float]: (Effective Rank, Participation Ratio).
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


def compute_pca_spectrum_compression(
    pos_features: Dict[str, Any],
    neg_features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Estimate intrinsic dimensionality for full representation space and negation difference subspace.

    Args:
        pos_features (Dict[str, Any]): Positive caption features.
        neg_features (Dict[str, Any]): Negative caption features.

    Returns:
        Dict[str, Any]: Effective rank, participation ratio, and PCA variance spectrum ratios.
    """
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


# ==============================================================================
# Stage 3: Projection Matrix SVD, Alignment & Spectrum Sweep
# ==============================================================================

def compute_projection_svd_ablation(
    model: nn.Module,
    pos_features: Dict[str, Any],
    neg_features: Dict[str, Any],
    target_token: str = "eot",
) -> Dict[str, Any]:
    """
    Perform Singular Value Decomposition (SVD) on text_projection matrix W_proj and spectrum sweep.

    Args:
        model (nn.Module): Pre-trained CLIP model.
        pos_features (Dict[str, Any]): Positive caption features.
        neg_features (Dict[str, Any]): Negative caption features.
        target_token (str): Target token pooling strategy.

    Returns:
        Dict[str, Any]: Singular values, directional alignment scores, and 10%-90% truncation sweep curve.
    """
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


# ==============================================================================
# Stage 2: Micro-Batched Image-Text Retrieval & Symmetric Absence Evaluation
# ==============================================================================

def compute_image_text_retrieval_metrics(
    model: nn.Module,
    tokenizer: Any,
    preprocess: Any,
    pair_metadata: List[dict],
    pos_texts: List[str],
    neg_texts: List[str],
    config: RetrievalConfig
) -> Dict[str, Any]:
    """
    Evaluate cross-modal image-text retrieval metrics and symmetric object presence/absence accuracy.

    Args:
        model (nn.Module): Pre-trained CLIP model.
        tokenizer (Any): Model tokenization engine.
        preprocess (Any): Image pre-processing transform.
        pair_metadata (List[dict]): Metadata dictionary for object presence tags.
        pos_texts (List[str]): Positive text captions.
        neg_texts (List[str]): Negative text captions.
        config (RetrievalConfig): Processing batch size and path parameters.

    Returns:
        Dict[str, Any]: Summary dictionary containing Pearson r, subgroup accuracies, and pandas DataFrame.
    """
    model.eval()

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

    if len(results) == 0:
        return {"skipped_count": skipped_count, "results_df": None}

    res_df = pd.DataFrame(results)

    sim_pos_arr = res_df["sim_image_pos"].values
    sim_neg_arr = res_df["sim_image_neg"].values
    sim_diff_arr = res_df["sim_diff"].values

    pearson_r = float(np.corrcoef(sim_pos_arr, sim_neg_arr)[0, 1])

    # Subgroup 1: Object Present in Image
    in_img_mask = res_df[obj_key] == True
    if np.sum(in_img_mask) > 0:
        accuracy_in_img = float(np.mean(sim_pos_arr[in_img_mask] > sim_neg_arr[in_img_mask])) * 100
        flip_rate_in_img = float(np.mean(sim_neg_arr[in_img_mask] > sim_pos_arr[in_img_mask])) * 100
        tie_rate_in_img = float(np.mean(np.isclose(sim_pos_arr[in_img_mask], sim_neg_arr[in_img_mask], atol=1e-6))) * 100
    else:
        accuracy_in_img, flip_rate_in_img, tie_rate_in_img = 0.0, 0.0, 0.0

    # Subgroup 2: Object Absent in Image
    out_img_mask = res_df[obj_key] == False
    if np.sum(out_img_mask) > 0:
        neg_pref_acc_out_img = float(np.mean(sim_neg_arr[out_img_mask] > sim_pos_arr[out_img_mask])) * 100
        pos_pref_rate_out_img = float(np.mean(sim_pos_arr[out_img_mask] > sim_neg_arr[out_img_mask])) * 100
        tie_rate_out_img = float(np.mean(np.isclose(sim_pos_arr[out_img_mask], sim_neg_arr[out_img_mask], atol=1e-6))) * 100
    else:
        neg_pref_acc_out_img, pos_pref_rate_out_img, tie_rate_out_img = 0.0, 0.0, 0.0

    summary = {
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

    return {"summary": summary, "results_df": res_df, "skipped_count": skipped_count}
