"""
CLIP Text Encoder Layer-wise Linear Probe Analysis Module.

Evaluates linear separability of positive vs negative caption representations
across ALL individual layers (Layer 0 to Layer 12) plus Final Projected Embedding
using LogisticRegression with 5-Fold Stratified Cross-Validation.

Reuses the unified feature extraction engine from `benchmarks.src.analysis.extractor`.
"""

import os
import argparse
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import open_clip

from benchmarks.src.analysis.extractor import extract_all_features_unified


def extract_layerwise_feature_dict(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: str = "cpu",
    target_token: str = "eot",
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    """
    Extract layer-wise and pipeline step feature matrices using unified extractor.

    Returns:
        feature_dict (Dict[str, np.ndarray]): Mapping from layer/step name to (N, D) features.
    """
    res = extract_all_features_unified(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        target_token=target_token,
        batch_size=batch_size,
    )

    feature_dict = {}
    for name, feats in res["layers"].items():
        feature_dict[name] = feats

    # Post-Layer 12 pipeline transformation steps
    pipeline_dict = res["pipeline"]
    feature_dict["Layer 12 + LN"] = pipeline_dict["Layer 12 + LN"]
    feature_dict["Projected (Unnorm)"] = pipeline_dict["Projected (Unnorm)"]
    feature_dict["Final (L2 Normed)"] = pipeline_dict["Final (L2 Normed)"]

    return feature_dict


def evaluate_layerwise_linear_probe(
    pos_layer_dict: Dict[str, np.ndarray],
    neg_layer_dict: Dict[str, np.ndarray],
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Run Stratified 5-Fold Cross-Validation Logistic Regression Linear Probe for every layer/step.

    Args:
        pos_layer_dict (Dict[str, np.ndarray]): Layer features for positive captions.
        neg_layer_dict (Dict[str, np.ndarray]): Layer features for negative captions.
        n_splits (int): Number of folds for StratifiedKFold.

    Returns:
        df_res (pd.DataFrame): DataFrame containing accuracy stats per layer.
    """
    n_pos = len(next(iter(pos_layer_dict.values())))
    n_neg = len(next(iter(neg_layer_dict.values())))
    y = np.array([1] * n_pos + [0] * n_neg)

    results = []
    layer_names = list(pos_layer_dict.keys())

    for l_name in layer_names:
        X_pos = pos_layer_dict[l_name]
        X_neg = neg_layer_dict[l_name]
        X = np.vstack([X_pos, X_neg])

        # Standardize features (L2 normalization per sample)
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_norm, y, cv=cv, scoring="accuracy")

        mean_acc = float(np.mean(scores)) * 100
        std_acc = float(np.std(scores)) * 100
        min_acc = float(np.min(scores)) * 100
        max_acc = float(np.max(scores)) * 100

        results.append({
            "layer": l_name,
            "mean_accuracy_pct": mean_acc,
            "std_accuracy_pct": std_acc,
            "min_accuracy_pct": min_acc,
            "max_accuracy_pct": max_acc,
            "feature_dim": X.shape[1],
        })

        print(f"  [{l_name:22s}] Acc: {mean_acc:6.2f}% (±{std_acc:4.2f}%) [Dim: {X.shape[1]}]")

    return pd.DataFrame(results)


def plot_and_export_results(df_res: pd.DataFrame, output_dir: str) -> Tuple[str, str, str]:
    """
    Save evaluation results to CSV, JSON, and PNG plot files.

    Returns:
        Tuple[str, str, str]: Paths to (csv_path, json_path, plot_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    results = df_res.to_dict(orient="records")

    csv_path = os.path.join(output_dir, "layerwise_linear_probe.csv")
    df_res.to_csv(csv_path, index=False)

    json_path = os.path.join(output_dir, "layerwise_linear_probe.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Visualization plot
    fig, ax = plt.subplots(figsize=(12, 6))
    layers = df_res["layer"].values
    accs = df_res["mean_accuracy_pct"].values
    stds = df_res["std_accuracy_pct"].values

    x_coords = list(range(len(layers)))
    ax.plot(x_coords, accs, "o-", color="#1f77b4", lw=2.5, ms=7, label="5-Fold CV Accuracy (%)")
    ax.fill_between(x_coords, accs - stds, accs + stds, color="#1f77b4", alpha=0.15)

    if "Layer 12" in layers:
        l12_idx = list(layers).index("Layer 12")
        ax.axvline(x=l12_idx + 0.5, color="crimson", ls="--", alpha=0.7, label="Post-Layer 12 Transformations")

    ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=12)
    ax.set_xlabel("Transformer Layer / Pipeline Step", fontsize=12)
    ax.set_title("CLIP Text Encoder Layer-wise & Pipeline-step Linear Probe Accuracy", fontsize=13, fontweight="bold")
    ax.set_xticks(x_coords)
    ax.set_xticklabels(layers, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(min(accs) - 5, min(100, max(accs) + 5))
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "layerwise_linear_probe.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nArtifacts saved successfully:")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  Plot: {plot_path}")

    return csv_path, json_path, plot_path


def run_layerwise_linear_probe_pipeline(
    model: nn.Module,
    tokenizer: Any,
    pos_texts: List[str],
    neg_texts: List[str],
    output_dir: str,
    device: str = "cpu",
    target_token: str = "eot",
    batch_size: int = 256,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Main orchestration function for layer-wise linear probing.
    """
    print("=" * 70)
    print(f"Executing Layer-wise & Pipeline-step Linear Probe Analysis ({n_splits}-Fold CV)")
    print("=" * 70)

    print("Extracting features for positive captions...")
    pos_layer_dict = extract_layerwise_feature_dict(model, tokenizer, pos_texts, device, target_token, batch_size)

    print("Extracting features for negative captions...")
    neg_layer_dict = extract_layerwise_feature_dict(model, tokenizer, neg_texts, device, target_token, batch_size)

    df_res = evaluate_layerwise_linear_probe(pos_layer_dict, neg_layer_dict, n_splits=n_splits)
    plot_and_export_results(df_res, output_dir)

    return df_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Layer-wise Linear Probe Analysis")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--csv_path", type=str, default="COCO_val_full_paired.csv", help="Path to Paired CSV")
    parser.add_argument("--output_dir", type=str, default="logs/evaluation/linear_probe_layerwise")
    parser.add_argument("--target_token", type=str, default="eot", choices=["eot", "mean", "all"])
    parser.add_argument("--max_samples", type=int, default=60000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if not os.path.exists(args.csv_path):
        if os.path.exists("COCO_val_mcq_top100_paired.csv"):
            args.csv_path = "COCO_val_mcq_top100_paired.csv"

    print(f"Loading paired dataset from: {args.csv_path}")
    df = pd.read_csv(args.csv_path).head(args.max_samples)
    pos_texts = df["positive_caption"].astype(str).tolist()
    neg_texts = df["negative_caption"].astype(str).tolist()
    print(f"Total positive/negative caption pairs: {len(pos_texts)}")

    print(f"Loading model {args.model} ({args.pretrained})...")
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    run_layerwise_linear_probe_pipeline(
        model=model,
        tokenizer=tokenizer,
        pos_texts=pos_texts,
        neg_texts=neg_texts,
        output_dir=args.output_dir,
        device=device,
        target_token=args.target_token,
        batch_size=args.batch_size,
        n_splits=args.n_splits,
    )
