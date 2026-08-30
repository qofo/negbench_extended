"""
BEAF Per-Object Layerwise Linear Probe Train vs Val Comparison Script.

Extracts CLIP vision features across layers and computes 5-Fold GroupKFold Linear Probe
Train Accuracy vs Validation Accuracy per object and per layer.
"""

import os
import json
import argparse
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from analysis.plotting import plt, render_top_objects_grid as _render_top_objects_grid

import open_clip
from analysis.config import get_layer_features as _get_feats
from analysis.beaf.beaf_loader import load_and_verify_counterfactual_pairs
from analysis.beaf.vision_mechanisms import extract_vision_features_unified


def compute_per_object_train_val_stats(
    vis_orig: Dict[str, Any],
    vis_cf: Dict[str, Any],
    df_pairs: pd.DataFrame,
    seed: int = 42,
    fit_intercept: bool = True,
) -> pd.DataFrame:
    """Compute per-object layerwise Train vs Val Linear Probe accuracy using GroupKFold."""
    layer_keys = list(vis_orig["layers"].keys())
    all_keys   = layer_keys + ["Pre-Projection", "+Final L2Norm"]

    object_names   = df_pairs["object_name"].values
    pair_ids       = df_pairs["pair_id"].values if "pair_id" in df_pairs.columns else np.arange(len(df_pairs))
    unique_objects = sorted(df_pairs["object_name"].unique().tolist())

    raw_records = []

    print(f"\n  [Train vs Val Linear Probe] Processing {len(unique_objects)} objects across {len(all_keys)} layers | Fit Intercept: {fit_intercept}...")

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
                best_c    = 0.1
            else:
                gkf = GroupKFold(n_splits=n_folds)
                cv_splits = list(gkf.split(X_all, y_all, groups=groups_all))

                c_candidates = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

                train_scores = []
                val_scores = []
                best_c_list = []

                for tr_idx, val_idx in cv_splits:
                    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
                    groups_tr = groups_all[tr_idx]

                    unique_tr_groups = len(np.unique(groups_tr))
                    if unique_tr_groups >= 2:
                        n_inner = min(3, unique_tr_groups)
                        inner_gkf = GroupKFold(n_splits=n_inner)
                        inner_cv = list(inner_gkf.split(X_tr, y_tr, groups=groups_tr))

                        best_c_fold = c_candidates[0]
                        best_inner_score = -1.0

                        for c in c_candidates:
                            inner_scores = []
                            for in_tr, in_val in inner_cv:
                                clf_in = LogisticRegression(C=c, max_iter=1000, random_state=seed, fit_intercept=fit_intercept)
                                clf_in.fit(X_tr[in_tr], y_tr[in_tr])
                                inner_scores.append(clf_in.score(X_tr[in_val], y_tr[in_val]))
                            mean_in = float(np.mean(inner_scores))
                            if mean_in > best_inner_score:
                                best_inner_score = mean_in
                                best_c_fold = c
                    else:
                        best_c_fold = 0.1

                    clf = LogisticRegression(C=best_c_fold, max_iter=1000, random_state=seed, fit_intercept=fit_intercept)
                    clf.fit(X_tr, y_tr)
                    train_scores.append(clf.score(X_tr, y_tr))
                    val_scores.append(clf.score(X_all[val_idx], y_all[val_idx]))
                    best_c_list.append(best_c_fold)

                train_acc = float(np.mean(train_scores) * 100)
                val_acc   = float(np.mean(val_scores)   * 100)
                best_c    = float(np.median(best_c_list))

            raw_records.append({
                "object_name":   obj,
                "layer_name":    lk,
                "n_pairs":       n_obj,
                "train_acc_pct": train_acc,
                "val_acc_pct":   val_acc,
                "gap_pct":       train_acc - val_acc,
                "best_c":        best_c,
            })

    return pd.DataFrame(raw_records)


def render_train_val_summary_plot(raw_df: pd.DataFrame, output_dir: str) -> None:
    """Render overall layerwise mean Train Acc vs Val Acc plot across all objects."""
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

    ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=12)
    ax.set_xlabel("Vision Transformer Layer / Pipeline Step", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(45, 105)
    ax.grid(True, ls="--", alpha=0.4)

    n_objs = len(raw_df["object_name"].unique())
    ax.set_title(f"Layerwise Linear Probe: Train vs Val Accuracy (Mean ± Std across {n_objs} Objects)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "beaf_per_object_train_val_summary.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def render_top_objects_grid(raw_df: pd.DataFrame, output_dir: str, top_k: int = 16) -> None:
    """Render per-layer train/val accuracy for the top-k best-sampled objects."""
    _render_top_objects_grid(
        raw_df,
        os.path.join(output_dir, "beaf_per_object_train_val_top_grid.png"),
        "Per-Object Layerwise Linear Probe: Train vs Val Accuracy (Top Objects)",
        top_k=top_k,
    )


def main():
    parser = argparse.ArgumentParser(description="BEAF Per-Object Train vs Val Linear Probe Analysis")
    parser.add_argument("--csv_path", type=str, default="benchmarks/data/images/beaf_counterfactual_6col.csv")
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="logs/evaluation/beaf_per_obj_test2")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--img_batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_bias", "--no-bias", action="store_true", default=False,
                        help="Disable bias/intercept in linear probes (default: bias enabled)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  BEAF Per-Object Train vs Val Linear Probe Analysis")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Model      : {args.model} ({args.pretrained})")
    print(f"  CSV        : {args.csv_path}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Use Bias   : {not args.no_bias}")
    print("=" * 60)

    # 1. Load Data
    df_raw, df_pairs, _ = load_and_verify_counterfactual_pairs(args.csv_path, args.image_root)
    n_pairs = len(df_pairs)
    print(f"  Loaded {n_pairs} verified pairs ({len(df_raw)} raw rows)")

    # 2. Load Model
    # create_model_and_transforms returns (model, preprocess_train, preprocess_val).
    # Feature extraction must take the *val* transform: the train one is
    # RandomResizedCrop + augmentation, which makes embeddings non-deterministic
    # and not comparable with the evaluation/ scripts.
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()

    # 3. Extract Vision Features
    print("\nExtracting Vision Features...")
    vis_orig = extract_vision_features_unified(model, preprocess, df_pairs["orig_path"].tolist(), device, args.img_batch)
    vis_cf   = extract_vision_features_unified(model, preprocess, df_pairs["cf_path"].tolist(),   device, args.img_batch)

    # Filter out missing images
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

    # 4. Compute Train vs Val Stats
    raw_df = compute_per_object_train_val_stats(vis_orig, vis_cf, df_pairs, seed=args.seed, fit_intercept=not args.no_bias)

    # Save CSV
    csv_path = os.path.join(args.output_dir, "beaf_per_object_train_val_layerwise.csv")
    raw_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Save JSON summary
    summary_dict = {}
    for lk in raw_df["layer_name"].unique():
        sub = raw_df[raw_df["layer_name"] == lk]
        summary_dict[lk] = {
            "train_acc_mean": float(sub["train_acc_pct"].mean()),
            "train_acc_std":  float(sub["train_acc_pct"].std()),
            "val_acc_mean":   float(sub["val_acc_pct"].mean()),
            "val_acc_std":    float(sub["val_acc_pct"].std()),
            "gap_mean":       float(sub["gap_pct"].mean()),
            "best_c_median":  float(sub["best_c"].median()),
        }
    json_path = os.path.join(args.output_dir, "beaf_per_object_train_val_layerwise.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"  Saved: {json_path}")

    # 5. Render Plots
    render_train_val_summary_plot(raw_df, args.output_dir)
    render_top_objects_grid(raw_df, args.output_dir, top_k=16)

    print("\nExecution complete!")


if __name__ == "__main__":
    main()
