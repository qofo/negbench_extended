"""
Stage 1: Global Negation Subspace Analysis & Cross-Category Transfer Probe Engine.

This module extracts negation difference vectors across diverse object categories/templates,
evaluates the Effective Rank and PCA spectrum of the difference matrix D, conducts
Cross-Category Transfer Probing (training on subset of categories, testing on unseen categories),
and exports the global negation basis matrix U_neg and linear probe weight vector w for downstream evaluation.
"""

import os
import json
import argparse
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy import stats
import open_clip

from .config import AnalysisConfig, MetadataKey, l2_normalize
from .extractor import extract_all_features_unified


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Global Negation Subspace Analysis")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model architecture")
    parser.add_argument("--pretrained", type=str, default="openai", help="Pretrained weights tag")
    parser.add_argument("--csv_path", type=str, default="benchmarks/data/images/COCO_val_full_paired.csv", help="Path to paired caption CSV")
    parser.add_argument("--output_dir", type=str, default="logs/subspace_analysis", help="Output directory for saved basis and reports")
    parser.add_argument("--max_samples", type=int, default=60000, help="Maximum number of caption pairs")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def compute_subspace_spectrum(diff_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Compute Effective Rank, Participation Ratio, and SVD spectrum on negation difference matrix.
    """
    N, D = diff_matrix.shape
    diff_centered = diff_matrix - np.mean(diff_matrix, axis=0, keepdims=True)
    cov = (diff_centered.T @ diff_centered) / N
    eigenvals = np.linalg.eigvalsh(cov)
    eigenvals = np.sort(np.maximum(eigenvals, 1e-12))[::-1]

    total_val = np.sum(eigenvals)
    p = eigenvals / total_val

    entropy = -np.sum(p * np.log(p + 1e-12))
    eff_rank = float(np.exp(entropy))
    pr = float((np.sum(eigenvals) ** 2) / np.sum(eigenvals ** 2))

    U, S, Vh = np.linalg.svd(diff_centered, full_matrices=False)
    explained_var = (S ** 2) / np.sum(S ** 2)

    return {
        "num_samples": int(N),
        "feature_dim": int(D),
        "effective_rank": eff_rank,
        "participation_ratio": pr,
        "top10_singular_values": S[:10].tolist(),
        "top10_explained_var_pct": (explained_var[:10] * 100).tolist(),
        "cumulative_var_top5_pct": float(np.sum(explained_var[:5])) * 100,
        "cumulative_var_top10_pct": float(np.sum(explained_var[:10])) * 100,
        "singular_vectors_Vh": Vh
    }


def evaluate_cross_category_transfer(
    pos_features: np.ndarray,
    neg_features: np.ndarray,
    pair_metadata: List[dict]
) -> Dict[str, Any]:
    """
    Train Linear Probe on 80% of categories/templates, evaluate accuracy on 20% unseen categories.
    """
    df_meta = pd.DataFrame(pair_metadata)
    tmpl_key = MetadataKey.SOURCE_TEMPLATE.value
    obj_key = MetadataKey.OBJECT_NAME.value

    # Primary grouping key
    group_col = tmpl_key if tmpl_key in df_meta.columns else obj_key
    if group_col not in df_meta.columns:
        return {"error": "No category grouping metadata found"}

    groups = df_meta[group_col].values
    unique_groups = np.unique(groups)

    if len(unique_groups) < 2:
        return {"error": f"Not enough distinct categories for transfer probe (found {len(unique_groups)})"}

    np.random.seed(42)
    shuffled_groups = np.random.permutation(unique_groups)
    split_idx = max(1, int(len(shuffled_groups) * 0.8))

    train_groups = set(shuffled_groups[:split_idx])
    test_groups = set(shuffled_groups[split_idx:])

    train_mask = np.array([g in train_groups for g in groups])
    test_mask = np.array([g in test_groups for g in groups])

    n_train = np.sum(train_mask)
    n_test = np.sum(test_mask)

    if n_train == 0 or n_test == 0:
        return {"error": "Train or test split empty after grouping"}

    X_train_pos = pos_features[train_mask]
    X_train_neg = neg_features[train_mask]
    X_train = l2_normalize(np.vstack([X_train_pos, X_train_neg]))
    y_train = np.array([1] * n_train + [0] * n_train)

    X_test_pos = pos_features[test_mask]
    X_test_neg = neg_features[test_mask]
    X_test = l2_normalize(np.vstack([X_test_pos, X_test_neg]))
    y_test = np.array([1] * n_test + [0] * n_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    train_acc = float(clf.score(X_train, y_train)) * 100
    test_acc = float(clf.score(X_test, y_test)) * 100

    probe_weight = clf.coef_[0] # Shape (D,)
    probe_bias = float(clf.intercept_[0])

    return {
        "train_categories_count": len(train_groups),
        "test_categories_count": len(test_groups),
        "train_samples_count": int(n_train * 2),
        "test_samples_count": int(n_test * 2),
        "train_accuracy_pct": train_acc,
        "unseen_test_accuracy_pct": test_acc,
        "probe_weight": probe_weight,
        "probe_bias": probe_bias
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("==========================================================")
    print("  Executing Stage 1: Global Negation Subspace Analysis")
    print(f"  Model      : {args.model} ({args.pretrained})")
    print(f"  CSV Path   : {args.csv_path}")
    print(f"  Output Dir : {args.output_dir}")
    print("==========================================================")

    if not os.path.exists(args.csv_path):
        # Fallback search
        alt_path = "COCO_val_full_paired.csv"
        if os.path.exists(alt_path):
            args.csv_path = alt_path

    assert os.path.exists(args.csv_path), f"CSV path does not exist: {args.csv_path}"

    df = pd.read_csv(args.csv_path).head(args.max_samples)
    pos_texts = df["positive_caption"].astype(str).tolist()
    neg_texts = df["negative_caption"].astype(str).tolist()

    path_k = MetadataKey.IMAGE_PATH.value
    obj_name_k = MetadataKey.OBJECT_NAME.value
    obj_in_img_k = MetadataKey.OBJECT_IN_IMAGE.value
    tmpl_k = MetadataKey.SOURCE_TEMPLATE.value

    pair_metadata = []
    for _, row in df.iterrows():
        meta = {
            path_k: str(row.get(path_k, "")),
            obj_name_k: str(row.get(obj_name_k, "")),
            obj_in_img_k: row.get(obj_in_img_k, None),
            tmpl_k: str(row.get(tmpl_k, ""))
        }
        pair_metadata.append(meta)

    print(f"Loading OpenCLIP model {args.model} ({args.pretrained})...")
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    print("\nExtracting representations across layers...")
    pos_features = extract_all_features_unified(model, tokenizer, pos_texts, device, "eot", args.batch_size)
    neg_features = extract_all_features_unified(model, tokenizer, neg_texts, device, "eot", args.batch_size)

    pos_final = pos_features["final_l2norm"]
    neg_final = neg_features["final_l2norm"]

    # 1. Difference Subspace SVD & Effective Rank
    diff_matrix = pos_final - neg_final
    spectrum_report = compute_subspace_spectrum(diff_matrix)

    Vh = spectrum_report.pop("singular_vectors_Vh")
    U_neg_top5 = Vh[:5, :] # Top 5 singular vectors as negation basis matrix (5, D)

    # 2. Cross-Category Transfer Probe
    transfer_report = evaluate_cross_category_transfer(pos_final, neg_final, pair_metadata)
    probe_weight = transfer_report.pop("probe_weight", None)
    probe_bias = transfer_report.pop("probe_bias", None)

    # Save Basis & Probe Weights
    basis_path = os.path.join(args.output_dir, "negation_subspace_basis_top5.npy")
    np.save(basis_path, U_neg_top5)
    print(f"✅ Saved Top-5 Negation Subspace Basis to: {basis_path}")

    if probe_weight is not None:
        probe_path = os.path.join(args.output_dir, "linear_probe_weights.npz")
        np.savez(probe_path, weight=probe_weight, bias=probe_bias)
        print(f"✅ Saved Linear Probe Weights to: {probe_path}")

    # Export Summary JSON
    summary = {
        "subspace_spectrum": spectrum_report,
        "cross_category_transfer_probe": transfer_report
    }
    json_path = os.path.join(args.output_dir, "subspace_analysis_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Stage 1 Subspace Analysis Summary ===")
    print(f"  Effective Rank of Diff Subspace : {spectrum_report['effective_rank']:.2f}")
    print(f"  Top-5 Cumulative Explained Var  : {spectrum_report['cumulative_var_top5_pct']:.2f}%")
    print(f"  Train Categories Count          : {transfer_report.get('train_categories_count', 'N/A')}")
    print(f"  Unseen Category Probe Accuracy  : {transfer_report.get('unseen_test_accuracy_pct', 0.0):.2f}%")
    print(f"\n✅ All Stage 1 outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
