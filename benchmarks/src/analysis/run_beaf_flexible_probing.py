"""
BEAF Flexible Multi-Classifier Layerwise Probing Script.

Evaluates CLIP vision features across layers using various linear, non-linear (MLP, RBF-SVM),
and Bilinear (Full, Low-Rank) probing algorithms per object and per layer.
"""

import os
import json
import argparse
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from analysis.plotting import plt, render_top_objects_grid as _render_top_objects_grid

import open_clip
from analysis.config import get_layer_features as _get_feats
from analysis.beaf.beaf_loader import load_and_verify_counterfactual_pairs
from analysis.beaf.vision_mechanisms import extract_vision_features_unified
from analysis.beaf.probe_factory import (
    SUPPORTED_PROBES,
    get_param_candidates,
    create_probe_classifier,
    format_params,
)


def compute_flexible_per_object_stats(
    vis_orig: Dict[str, Any],
    vis_cf: Dict[str, Any],
    df_pairs: pd.DataFrame,
    probe_type: str = "logistic",
    seed: int = 42,
    use_bias: bool = True,
) -> pd.DataFrame:
    """Compute per-object layerwise Train vs Val probing accuracy with inner CV hyperparameter tuning."""
    layer_keys = list(vis_orig["layers"].keys())
    all_keys   = layer_keys + ["Pre-Projection", "+Final L2Norm"]

    object_names   = df_pairs["object_name"].values
    pair_ids       = df_pairs["pair_id"].values if "pair_id" in df_pairs.columns else np.arange(len(df_pairs))
    unique_objects = sorted(df_pairs["object_name"].unique().tolist())

    param_candidates = get_param_candidates(probe_type)
    raw_records = []

    print(f"\n  [Flexible Probing: {probe_type.upper()}] Processing {len(unique_objects)} objects across {len(all_keys)} layers...")
    print(f"  [Tuning Grid] {len(param_candidates)} hyperparameter combinations per fold | Use Bias: {use_bias}")

    for obj in unique_objects:
        mask      = (object_names == obj)
        n_obj     = int(np.sum(mask))
        obj_pairs = pair_ids[mask]

        if n_obj < 2:
            continue

        groups_all = np.concatenate([obj_pairs, obj_pairs])
        n_unique_groups = len(np.unique(obj_pairs))
        n_folds = min(5, n_unique_groups)

        for lk in all_keys:
            X_o = _get_feats(vis_orig, lk)[mask]
            X_c = _get_feats(vis_cf,   lk)[mask]

            X_o_n = X_o / (np.linalg.norm(X_o, axis=1, keepdims=True) + 1e-8)
            X_c_n = X_c / (np.linalg.norm(X_c, axis=1, keepdims=True) + 1e-8)

            X_all = np.vstack([X_o_n, X_c_n])
            y_all = np.array([1] * n_obj + [0] * n_obj)

            if n_folds < 2:
                train_acc = 50.0
                val_acc   = 50.0
                best_param_str = format_params(param_candidates[0])
            else:
                gkf = GroupKFold(n_splits=n_folds)
                cv_splits = list(gkf.split(X_all, y_all, groups=groups_all))

                train_scores = []
                val_scores = []
                best_param_list = []

                for tr_idx, val_idx in cv_splits:
                    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
                    groups_tr = groups_all[tr_idx]

                    unique_tr_groups = len(np.unique(groups_tr))
                    if unique_tr_groups >= 2:
                        n_inner = min(3, unique_tr_groups)
                        inner_gkf = GroupKFold(n_splits=n_inner)
                        inner_cv = list(inner_gkf.split(X_tr, y_tr, groups=groups_tr))

                        best_p_fold = param_candidates[0]
                        best_inner_score = -1.0

                        for p_dict in param_candidates:
                            inner_scores = []
                            for in_tr, in_val in inner_cv:
                                clf_in = create_probe_classifier(probe_type, seed=seed, fit_intercept=use_bias, use_bias=use_bias, **p_dict)
                                clf_in.fit(X_tr[in_tr], y_tr[in_tr])
                                inner_scores.append(clf_in.score(X_tr[in_val], y_tr[in_val]))
                            mean_in = float(np.mean(inner_scores))
                            if mean_in > best_inner_score:
                                best_inner_score = mean_in
                                best_p_fold = p_dict
                    else:
                        best_p_fold = param_candidates[0]

                    clf = create_probe_classifier(probe_type, seed=seed, fit_intercept=use_bias, use_bias=use_bias, **best_p_fold)
                    clf.fit(X_tr, y_tr)
                    train_scores.append(clf.score(X_tr, y_tr))
                    val_scores.append(clf.score(X_all[val_idx], y_all[val_idx]))
                    best_param_list.append(best_p_fold)

                train_acc = float(np.mean(train_scores) * 100)
                val_acc   = float(np.mean(val_scores)   * 100)
                best_param_str = format_params(best_param_list[0]) if best_param_list else ""

            raw_records.append({
                "probe_type":    probe_type,
                "object_name":   obj,
                "layer_name":    lk,
                "n_pairs":       n_obj,
                "train_acc_pct": train_acc,
                "val_acc_pct":   val_acc,
                "gap_pct":       train_acc - val_acc,
                "best_params":   best_param_str,
            })

    return pd.DataFrame(raw_records)


def render_probe_summary_plot(raw_df: pd.DataFrame, probe_type: str, output_dir: str) -> None:
    """Render overall layerwise mean Train Acc vs Val Acc plot for the given probe_type."""
    layer_names = raw_df["layer_name"].unique().tolist()

    train_means, train_stds = [], []
    val_means, val_stds     = [], []

    for lk in layer_names:
        sub = raw_df[raw_df["layer_name"] == lk]
        train_means.append(sub["train_acc_pct"].mean())
        train_stds.append(sub["train_acc_pct"].std())
        val_means.append(sub["val_acc_pct"].mean())
        val_stds.append(sub["val_acc_pct"].std())

    x = np.arange(len(layer_names))
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(x, train_means, "o-", color="#1f77b4", lw=2.5, ms=7, label="Train Acc (%) [Mean]")
    ax.fill_between(x, np.array(train_means) - np.array(train_stds), np.array(train_means) + np.array(train_stds), color="#1f77b4", alpha=0.15)

    ax.plot(x, val_means, "s--", color="#d62728", lw=2.5, ms=7, label="Val Acc (%) [Mean CV]")
    ax.fill_between(x, np.array(val_means) - np.array(val_stds), np.array(val_means) + np.array(val_stds), color="#d62728", alpha=0.15)

    if "Pre-Projection" in layer_names:
        idx = layer_names.index("Pre-Projection")
        ax.axvline(x=idx - 0.5, color="gray", ls=":", lw=1.5, alpha=0.7, label="Post-Transformer Transformations")

    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Transformer Layer / Pipeline Step", fontsize=12)
    n_objs = len(raw_df["object_name"].unique())
    ax.set_title(f"Per-Object Layerwise Probing: Train vs Val Acc ({probe_type.upper()}) [N={n_objs} Objects]", fontsize=13, fontweight="bold")
    ax.set_ylim(40, 105)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"beaf_{probe_type}_train_val_summary.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def render_top_objects_grid(raw_df: pd.DataFrame, probe_type: str, output_dir: str, top_k: int = 16) -> None:
    """Render per-layer train/val accuracy for the top-k best-sampled objects."""
    _render_top_objects_grid(
        raw_df,
        os.path.join(output_dir, f"beaf_{probe_type}_train_val_top_grid.png"),
        f"Per-Object Layerwise Probing: Train vs Val Accuracy ({probe_type.upper()})",
        top_k=top_k,
    )


def main():
    parser = argparse.ArgumentParser(description="BEAF Flexible Multi-Classifier Layerwise Probing")
    parser.add_argument("--csv_path", type=str, default="benchmarks/data/images/beaf_counterfactual_6col.csv")
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="logs/evaluation/beaf_flexible_probe")
    parser.add_argument("--probe_type", type=str, default="logistic", choices=SUPPORTED_PROBES, help="Classifier algorithm")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_bias", "--no-bias", action="store_true", default=False,
                        help="Disable bias/intercept in probing classifiers (default: bias enabled)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  BEAF Flexible Multi-Classifier Layerwise Probing")
    print("=" * 60)
    print(f"  Probe Type : {args.probe_type.upper()}")
    print(f"  Model      : {args.model} ({args.pretrained})")
    print(f"  CSV        : {args.csv_path}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Use Bias   : {not args.no_bias}")
    print("=" * 60)

    # 1. Load Data
    df_raw, df_pairs, pair_metadata = load_and_verify_counterfactual_pairs(args.csv_path, args.image_root)
    if args.max_samples > 0:
        df_pairs = df_pairs.head(args.max_samples).copy()

    # 2. Load Model
    # create_model_and_transforms returns (model, preprocess_train, preprocess_val).
    # Feature extraction must take the *val* transform: the train one is
    # RandomResizedCrop + augmentation, which makes embeddings non-deterministic
    # and not comparable with the evaluation/ scripts.
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device)

    # 3. Extract Vision Features
    orig_paths = df_pairs["orig_path"].tolist()
    cf_paths   = df_pairs["cf_path"].tolist()

    vis_orig = extract_vision_features_unified(model, preprocess, orig_paths, device=device, batch_size=args.batch_size)
    vis_cf   = extract_vision_features_unified(model, preprocess, cf_paths,   device=device, batch_size=args.batch_size)

    # Filter out missing images
    n_pairs = len(df_pairs)
    flags_orig = np.array(vis_orig.get("loaded_flags", [True] * n_pairs))
    flags_cf   = np.array(vis_cf.get("loaded_flags",   [True] * n_pairs))
    valid_mask = flags_orig & flags_cf

    df_pairs = df_pairs[valid_mask].reset_index(drop=True)
    for k in vis_orig["layers"]:
        vis_orig["layers"][k] = vis_orig["layers"][k][valid_mask]
        vis_cf["layers"][k]   = vis_cf["layers"][k][valid_mask]
    vis_orig["pre_proj"]     = vis_orig["pre_proj"][valid_mask]
    vis_cf["pre_proj"]       = vis_cf["pre_proj"][valid_mask]
    vis_orig["final_l2norm"] = vis_orig["final_l2norm"][valid_mask]
    vis_cf["final_l2norm"]   = vis_cf["final_l2norm"][valid_mask]

    # 4. Compute Flexible Probing Statistics
    raw_df = compute_flexible_per_object_stats(vis_orig, vis_cf, df_pairs, probe_type=args.probe_type, seed=args.seed, use_bias=not args.no_bias)

    # Save CSV & Summary JSON
    csv_path = os.path.join(args.output_dir, f"beaf_{args.probe_type}_layerwise.csv")
    raw_df.to_csv(csv_path, index=False)

    summary_dict = {}
    for lk in raw_df["layer_name"].unique():
        sub = raw_df[raw_df["layer_name"] == lk]
        summary_dict[lk] = {
            "train_acc_mean": float(sub["train_acc_pct"].mean()),
            "train_acc_std":  float(sub["train_acc_pct"].std()),
            "val_acc_mean":   float(sub["val_acc_pct"].mean()),
            "val_acc_std":    float(sub["val_acc_pct"].std()),
            "gap_mean":       float(sub["gap_pct"].mean()),
        }

    json_path = os.path.join(args.output_dir, f"beaf_{args.probe_type}_layerwise.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    # 5. Render Plots
    render_probe_summary_plot(raw_df, args.probe_type, args.output_dir)
    render_top_objects_grid(raw_df, args.probe_type, args.output_dir, top_k=16)

    print(f"\n  ✅ Flexible Probing [{args.probe_type.upper()}] Finished! Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()

