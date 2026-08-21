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
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
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
from analysis.beaf.probe_factory import (
    ElementWiseNonLinearPyTorch,
    LowRankBilinearPyTorch,
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

    # Check how many images exist on disk before processing
    missing_paths = [p for p in image_paths if not os.path.exists(p)]
    if len(missing_paths) > 0:
        print(f"[CRITICAL ERROR] {len(missing_paths)}/{len(image_paths)} image files DO NOT EXIST on disk!")
        print(f"Example missing path: '{missing_paths[0]}'")
        print(f"Please check --image_root path or CSV image_path values!")
    else:
        print(f"All {len(image_paths)} image files found successfully on disk.")

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

            # make a patches
            if conv1 is not None:
                x = conv1(stacked)
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x = x.permute(0, 2, 1)
            else:
                x = stacked
    
            # add CLS token
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
            # transformer blocks
            for block in resblocks:
                x_perm = block(x_perm)
                hidden_states.append(x_perm.permute(1, 0, 2))

            pooled_layers = []
            for hs in hidden_states:
                # Use only CLS token
                cls_feat = hs[:, 0, :].float().cpu().numpy()
                pooled_layers.append(cls_feat)


            # take after 12nd layer, after LN, after projection, after L2 nomalization respectively
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

    prev_delta = None
    for l_idx, l_name in enumerate(layer_names):
        f_orig = vis_orig["layers"][l_name]
        f_cf   = vis_cf["layers"][l_name]

        cos_sims = batch_cosine_similarity(f_orig, f_cf)
        l2_dists = batch_l2_distance(f_orig, f_cf)
        dot_prods = batch_dot_product(f_orig, f_cf)

        delta_v = f_orig - f_cf
        delta_norm = float(np.mean(np.linalg.norm(delta_v, axis=1)))

        if prev_delta is not None:
            dir_sim = float(batch_cosine_similarity(delta_v, prev_delta).mean())
        else:
            dir_sim = 1.0
        prev_delta = delta_v

        layer_results.append({
            "layer": l_name,
            "mean_cosine_sim": float(np.mean(cos_sims)),
            "std_cosine_sim": float(np.std(cos_sims)),
            "mean_l2_distance": float(np.mean(l2_dists)),
            "mean_dot_product": float(np.mean(dot_prods)),
            "delta_vector_norm": delta_norm,
            "layer_to_layer_direction_sim": dir_sim,
        })

    cos_pre = batch_cosine_similarity(vis_orig["pre_proj"], vis_cf["pre_proj"])
    l2_pre  = batch_l2_distance(vis_orig["pre_proj"], vis_cf["pre_proj"])
    delta_pre = vis_orig["pre_proj"] - vis_cf["pre_proj"]
    dir_sim_pre = float(batch_cosine_similarity(delta_pre, prev_delta).mean()) if prev_delta is not None else 1.0
    layer_results.append({
        "layer": "Pre-Projection (LN)",
        "mean_cosine_sim": float(np.mean(cos_pre)),
        "std_cosine_sim": float(np.std(cos_pre)),
        "mean_l2_distance": float(np.mean(l2_pre)),
        "mean_dot_product": float(np.mean(batch_dot_product(vis_orig["pre_proj"], vis_cf["pre_proj"]))),
        "delta_vector_norm": float(np.mean(np.linalg.norm(delta_pre, axis=1))),
        "layer_to_layer_direction_sim": dir_sim_pre,
    })

    cos_final = batch_cosine_similarity(vis_orig["final_l2norm"], vis_cf["final_l2norm"])
    l2_final  = batch_l2_distance(vis_orig["final_l2norm"], vis_cf["final_l2norm"])
    delta_final = vis_orig["final_l2norm"] - vis_cf["final_l2norm"]
    layer_results.append({
        "layer": "+Final L2Norm",
        "mean_cosine_sim": float(np.mean(cos_final)),
        "std_cosine_sim": float(np.std(cos_final)),
        "mean_l2_distance": float(np.mean(l2_final)),
        "mean_dot_product": float(np.mean(batch_dot_product(vis_orig["final_l2norm"], vis_cf["final_l2norm"]))),
        "delta_vector_norm": float(np.mean(np.linalg.norm(delta_final, axis=1))),
        "layer_to_layer_direction_sim": 1.0,
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

    f_pre_orig = vis_orig["pre_proj"]
    f_pre_cf   = vis_cf["pre_proj"]
    diff_pre   = f_pre_orig - f_pre_cf
    diff_norm  = l2_normalize(diff_pre)

    if isinstance(proj, nn.Linear):
        W = proj.weight.detach().cpu().numpy().T
    elif hasattr(proj, "detach"):
        W = proj.detach().cpu().numpy()
    else:
        return {}

    if W.ndim == 2 and f_pre_orig.ndim == 2 and W.shape[0] != f_pre_orig.shape[1]:
        W = W.T

    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    alignments = np.abs(diff_norm @ U)
    mean_alignments = np.mean(alignments, axis=0)

    sim_orig = batch_cosine_similarity(vis_orig["final_l2norm"], vis_cf["final_l2norm"]).mean()

    sweep_results = []
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    num_sv = len(S)
    for r in ratios:
        k = max(1, min(num_sv, int(num_sv * r)))

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
    output_dir: str,
    object_names: Optional[np.ndarray] = None,
    pair_ids: Optional[np.ndarray] = None,
    seed: int = 42,
    fit_intercept: bool = True,
) -> Dict[str, Any]:
    """Train 5-fold cross-validated Linear Probe on Vision Transformer features to classify object_in_image.

    Fold strategy:
    - If `pair_ids` is provided: GroupKFold by pair_id, ensuring orig and cf from the
      same edited pair always land in the same fold (prevents visual similarity leakage).
    - If only `object_names` is provided: GroupKFold by object_name for Unseen Object
      generalization testing. NOTE: this variant risks within-pair leakage when the
      same pair's orig/cf images appear in different folds.
    - Otherwise: StratifiedKFold (5-fold, random).
    """
    n_orig = len(vis_orig["pre_proj"])
    n_cf   = len(vis_cf["pre_proj"])
    # Label 1 = original (object present), 0 = counterfactual (object absent)
    y = np.array([1] * n_orig + [0] * n_cf)

    if pair_ids is not None and len(pair_ids) == n_orig:
        # ✅ Correct: same pair_id guarantees orig and cf are in the same fold
        groups = np.concatenate([pair_ids, pair_ids])
        gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        cv_splits = list(gkf.split(X=np.zeros(len(y)), y=y, groups=groups))
        print(f"\n  ✅ [Linear Probe - GroupKFold by pair_id] {len(np.unique(groups))} unique pairs, "
              f"no within-pair data leakage.")
    elif object_names is not None and len(object_names) == n_orig:
        # ⚠️ Unseen-object generalization split (object_name level)
        # Warning: orig/cf may end up in different folds if object has many pairs
        groups = np.concatenate([object_names, object_names])
        gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        cv_splits = list(gkf.split(X=np.zeros(len(y)), y=y, groups=groups))
        print(f"\n  ⚠️  [Linear Probe - GroupKFold by object_name] {len(np.unique(groups))} unique objects. "
              f"Use pair_ids for stricter leakage control.")
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_splits = list(skf.split(X=np.zeros(len(y)), y=y))
        print(f"\n  🔍 [Linear Probe - StratifiedKFold] Samples: {n_orig} orig + {n_cf} cf = {len(y)} total")

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

        clf = LogisticRegression(max_iter=1000, random_state=42, fit_intercept=fit_intercept)
        scores = cross_val_score(clf, X_norm, y, cv=cv_splits, scoring="accuracy")

        mean_acc = float(np.mean(scores) * 100)
        std_acc  = float(np.std(scores) * 100)

        probe_results[l_name] = {
            "mean_accuracy_pct": mean_acc,
            "std_accuracy_pct":  std_acc,
        }
        print(f"     - [{l_name:16s}] Accuracy: {mean_acc:6.2f}% ± {std_acc:4.2f}%")

    with open(os.path.join(output_dir, "beaf_vision_linear_probe.json"), "w", encoding="utf-8") as f:
        json.dump(probe_results, f, indent=2)

    # Plot Layer-wise & Pipeline-step Linear Probe Accuracy Line Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    layers = list(probe_results.keys())
    accs = [item["mean_accuracy_pct"] for item in probe_results.values()]
    stds = [item["std_accuracy_pct"] for item in probe_results.values()]

    x_coords = list(range(len(layers)))
    ax.plot(x_coords, accs, "o-", color="#1f77b4", lw=2.5, ms=7, label="5-Fold CV Accuracy (%)")

    if "Layer 12" in layers:
        l12_idx = layers.index("Layer 12")
        ax.axvline(x=l12_idx + 0.5, color="crimson", ls="--", alpha=0.7, label="Post-Layer 12 Transformations")

    ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=12)
    ax.set_xlabel("Transformer Layer / Pipeline Step", fontsize=12)
    ax.set_title("CLIP Vision Encoder Layer-wise & Pipeline-step Linear Probe Accuracy", fontsize=13, fontweight="bold")
    ax.set_xticks(x_coords)
    ax.set_xticklabels(layers, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(min(accs) - 5, min(100, max(accs) + 5))
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "beaf_vision_linear_probe.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("  Saved: beaf_vision_linear_probe.json & .png")
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


# ElementWiseNonLinearPyTorch and LowRankBilinearPyTorch are imported from
# analysis.beaf.probe_factory (Single Source of Truth for all PyTorch probe models).


def train_eval_element_wise_gelu(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    seed: int = 42, epochs: int = 300, lr: float = 1e-2
) -> float:
    """Train PyTorch Element-wise Non-linear GELU Probe (Feature-wise, 0% Dimension Mixing Control)."""
    torch.manual_seed(seed)
    d_in = X_train.shape[1]
    model = ElementWiseNonLinearPyTorch(d_in)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_te)) >= 0.5).float()
        acc = float((preds == y_te).float().mean().item() * 100)
    return acc


def train_eval_low_rank_bilinear(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    rank: int, seed: int = 42, epochs: int = 300, lr: float = 1e-2
) -> float:
    """Train PyTorch real Low-Rank Bilinear Probe: f(x) = x^T U V^T x + w_0^T x + b."""
    torch.manual_seed(seed)
    d_in = X_train.shape[1]
    model = LowRankBilinearPyTorch(d_in, rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_te)) >= 0.5).float()
        acc = float((preds == y_te).float().mean().item() * 100)
    return acc


def compute_vision_non_linear_probe(
    vis_orig: Dict[str, Any],
    vis_cf: Dict[str, Any],
    output_dir: str,
    seed: int = 42,
    object_names: Optional[np.ndarray] = None,
    eval_raw_images: bool = False,
) -> Dict[str, Any]:
    """
    Train 5-fold cross-validated Probes on Visual Features.
    If eval_raw_images is True, probes directly on raw single image vectors X_orig (+1) vs X_cf (0).
    Otherwise, probes on Visual Edit Difference Vectors (Real Shift vs Norm-Matched Random Noise).
    """
    n_orig = len(vis_orig["pre_proj"])
    rng = np.random.default_rng(seed=seed)
    rand_idx = (np.arange(n_orig) + rng.integers(1, n_orig, size=n_orig)) % n_orig

    stages = ["Pre-Projection", "+Final L2Norm"]

    report = {
        "elementwise_gelu": {},
        "linear_tuned": {},
        "polynomial_kernel": {},
        "low_rank_bilinear": {},
        "rbf_kernel": {},
        "mlp_capacity": {}
    }

    mode_str = "Raw Image Vectors (Single X_orig vs X_cf)" if eval_raw_images else "Visual Edit Difference Vectors (Real Shift vs Random Dir)"
    print(f"\n  🔍 [Debug: Non-Linear Probe ({mode_str})] Samples: {n_orig * 2} total vectors")

    for stage_name in stages:
        print(f"\n  === Stage: {stage_name} ===")
        if stage_name == "Pre-Projection":
            f_orig = vis_orig["pre_proj"]
            f_cf   = vis_cf["pre_proj"]
        else:
            f_orig = vis_orig["final_l2norm"]
            f_cf   = vis_cf["final_l2norm"]

        if eval_raw_images:
            X_data = np.vstack([l2_normalize(f_orig), l2_normalize(f_cf)])
            y_data = np.array([1] * n_orig + [0] * n_orig)
            if object_names is not None and len(object_names) == n_orig:
                groups = np.concatenate([object_names, object_names])
                gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
                cv_splits = list(gkf.split(X_data, y_data, groups=groups))
                print(f"     🔍 [GroupKFold Enabled] Probing raw single images grouped by {len(np.unique(groups))} unique object_names.")
            else:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                cv_splits = list(skf.split(X_data, y_data))
        else:
            diff_real = f_orig - f_cf
            d_in = diff_real.shape[1]
            rng_ctrl = np.random.default_rng(seed=seed)
            rand_dirs = rng_ctrl.normal(size=(n_orig, d_in))
            rand_dirs = rand_dirs / np.linalg.norm(rand_dirs, axis=1, keepdims=True)
            norms_real = np.linalg.norm(diff_real, axis=1, keepdims=True)
            diff_ctrl = rand_dirs * norms_real

            norm_real = np.linalg.norm(diff_real, axis=1).mean()
            norm_ctrl = np.linalg.norm(diff_ctrl, axis=1).mean()
            print(f"     - Mean L2 Norm: Real removal shift = {norm_real:.4f} | Control (norm-matched random dir) = {norm_ctrl:.4f}")

            X_data = np.vstack([l2_normalize(diff_real), l2_normalize(diff_ctrl)])
            y_data = np.array([1] * n_orig + [0] * n_orig)

            if object_names is not None and len(object_names) == n_orig:
                groups = np.concatenate([object_names, object_names[rand_idx]])
                gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
                cv_splits = list(gkf.split(X_data, y_data, groups=groups))
                print(f"     🔍 [GroupKFold Enabled] Non-linear Probe grouped by {len(np.unique(groups))} unique object_names.")
            else:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                cv_splits = list(skf.split(X_data, y_data))

        # Stage 0: Element-wise Non-linear GELU Probe (Feature-wise, 0% Dimension Mixing Control)
        print("     [Stage 0/5] Element-wise Non-linear GELU Probe (0% Dimension Mixing):")
        gelu_results = {}
        fold_accs = []
        for train_idx, test_idx in cv_splits:
            acc = train_eval_element_wise_gelu(
                X_data[train_idx], y_data[train_idx],
                X_data[test_idx], y_data[test_idx],
                seed=seed
            )
            fold_accs.append(acc)
        mean_acc = float(np.mean(fold_accs))
        std_acc  = float(np.std(fold_accs))
        gelu_results["gelu_elementwise"] = {"mean_acc": mean_acc, "std_acc": std_acc}
        print(f"       * GELU Feature-wise (No Mixing): {mean_acc:6.2f}% ± {std_acc:4.2f}%")
        report["elementwise_gelu"][stage_name] = gelu_results

        # Stage 1: Linear Probe Baseline (C Sweep)
        print("     [Stage 1/5] Linear Baseline (C Sweep 1e-3..100):")
        lin_results = {}
        for c in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
            clf = LogisticRegression(C=c, max_iter=1000, random_state=seed)
            scores = cross_val_score(clf, X_data, y_data, cv=cv_splits, scoring="accuracy")
            mean_acc = float(np.mean(scores) * 100)
            std_acc  = float(np.std(scores) * 100)
            lin_results[f"C_{c}"] = {"mean_acc": mean_acc, "std_acc": std_acc}
            print(f"       * C = {c:6.3f}: {mean_acc:6.2f}% ± {std_acc:4.2f}%")
        report["linear_tuned"][stage_name] = lin_results

        # Stage 2: Polynomial Kernel Probe (Degree 2, 3)
        print("     [Stage 2/5] Polynomial Kernel (Degree 2, 3):")
        poly_results = {}
        for deg in [2, 3]:
            clf = SVC(kernel="poly", degree=deg, coef0=1.0, C=1.0, random_state=seed)
            scores = cross_val_score(clf, X_data, y_data, cv=cv_splits, scoring="accuracy")
            mean_acc = float(np.mean(scores) * 100)
            std_acc  = float(np.std(scores) * 100)
            poly_results[f"degree_{deg}"] = {"mean_acc": mean_acc, "std_acc": std_acc}
            print(f"       * Degree {deg}: {mean_acc:6.2f}% ± {std_acc:4.2f}%")
        report["polynomial_kernel"][stage_name] = poly_results

        # Stage 3: Real Low-Rank Bilinear Probe: f(x) = x^T U V^T x + w_0^T x + b
        print("     [Stage 3/5] Real Low-Rank Bilinear f(x) = x^T U V^T x + w_0^T x + b (Rank 4, 8, 16, 32, 64):")
        bilinear_results = {}
        r_list = [1,2,3,4,8,16,512]
        for r in r_list:
            fold_accs = []
            for train_idx, test_idx in cv_splits:
                acc = train_eval_low_rank_bilinear(
                    X_data[train_idx], y_data[train_idx],
                    X_data[test_idx], y_data[test_idx],
                    rank=r, seed=seed
                )
                fold_accs.append(acc)
            mean_acc = float(np.mean(fold_accs))
            std_acc  = float(np.std(fold_accs))
            bilinear_results[f"rank_{r}"] = {"mean_acc": mean_acc, "std_acc": std_acc}
            print(f"       * Bilinear Rank {r:2d}: {mean_acc:6.2f}% ± {std_acc:4.2f}%")
        report["low_rank_bilinear"][stage_name] = bilinear_results

        # Stage 4: RBF Kernel SVM Probe
        print("     [Stage 4/5] RBF Kernel SVM Probe (Gamma 1e-4..1.0):")
        rbf_results = {}
        for g in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            clf = SVC(kernel="rbf", gamma=g, C=1.0, random_state=seed)
            scores = cross_val_score(clf, X_data, y_data, cv=cv_splits, scoring="accuracy")
            mean_acc = float(np.mean(scores) * 100)
            std_acc  = float(np.std(scores) * 100)
            rbf_results[f"gamma_{g}"] = {"mean_acc": mean_acc, "std_acc": std_acc}
            print(f"       * Gamma {g:6.4f}: {mean_acc:6.2f}% ± {std_acc:4.2f}%")
        report["rbf_kernel"][stage_name] = rbf_results

        # Stage 5: MLP Capacity Probe (Hidden 8, 16, 32, 64)
        print("     [Stage 5/5] MLP Capacity Probe (Hidden 1..64):")
        mlp_results = {}
        for h in [1, 2, 4, 8, 16, 32, 64]:
            clf = MLPClassifier(hidden_layer_sizes=(h,), activation="relu", max_iter=1000, random_state=seed)
            scores = cross_val_score(clf, X_data, y_data, cv=cv_splits, scoring="accuracy")
            mean_acc = float(np.mean(scores) * 100)
            std_acc  = float(np.std(scores) * 100)
            mlp_results[f"hidden_{h}"] = {"mean_acc": mean_acc, "std_acc": std_acc}
            print(f"       * Hidden {h:2d}: {mean_acc:6.2f}% ± {std_acc:4.2f}%")
        report["mlp_capacity"][stage_name] = mlp_results

    prefix = "beaf_vision_raw_non_linear_probe" if eval_raw_images else "beaf_vision_non_linear_probe"
    # Save JSON
    json_path = os.path.join(output_dir, f"{prefix}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

        json.dump(report, f, indent=2)

    # Plot 2x3 Subplots for 5 Stages
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    # Stage 1: Linear C Sweep
    ax = axes[0, 0]
    for stage_name in stages:
        cs = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
        accs = [report["linear_tuned"][stage_name][f"C_{c}"]["mean_acc"] for c in cs]
        ax.plot([str(c) for c in cs], accs, marker="o", lw=2, label=stage_name)
    ax.set_title("1. Linear Baseline (C Sweep)", fontweight="bold")
    ax.set_xlabel("Logistic Regression C Parameter")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend()

    # Stage 2: Poly Kernel (Degree 2, 3)
    ax = axes[0, 1]
    for stage_name in stages:
        degs = [2, 3]
        accs = [report["polynomial_kernel"][stage_name][f"degree_{d}"]["mean_acc"] for d in degs]
        ax.plot([str(d) for d in degs], accs, marker="s", lw=2, label=stage_name)
    ax.set_title("2. Quadratic Polynomial (Degree 2, 3)", fontweight="bold")
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend()

    # Stage 3: Real Low-Rank Bilinear
    ax = axes[0, 2]
    for stage_name in stages:
        ranks = [1,2,3,4, 8, 16,512]
        accs = [report["low_rank_bilinear"][stage_name][f"rank_{r}"]["mean_acc"] for r in ranks]
        ax.plot([str(r) for r in ranks], accs, marker="d", color="crimson" if stage_name=="+Final L2Norm" else "navy", lw=2, label=stage_name)
    ax.set_title("3. Low-Rank Bilinear (W = U V^T)", fontweight="bold")
    ax.set_xlabel("Bilinear Subspace Rank r")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend()

    # Stage 4: RBF Kernel
    ax = axes[1, 0]
    for stage_name in stages:
        gammas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        accs = [report["rbf_kernel"][stage_name][f"gamma_{g}"]["mean_acc"] for g in gammas]
        ax.plot([str(g) for g in gammas], accs, marker="v", lw=2, label=stage_name)
    ax.set_title("4. RBF Kernel SVM Probe", fontweight="bold")
    ax.set_xlabel("RBF Gamma Parameter")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend()

    # Stage 5: MLP Capacity
    ax = axes[1, 1]
    for stage_name in stages:
        hdims = [1, 2, 4, 8, 16, 32, 64]
        accs = [report["mlp_capacity"][stage_name][f"hidden_{h}"]["mean_acc"] for h in hdims]
        ax.plot([str(h) for h in hdims], accs, marker="^", lw=2, label=stage_name)
    ax.set_title("5. MLP Capacity Probe (Hidden Dim)", fontweight="bold")
    ax.set_xlabel("Hidden Layer Units (h)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend()

    # Hide 6th empty subplot
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {prefix}.json & .png")
    return report


