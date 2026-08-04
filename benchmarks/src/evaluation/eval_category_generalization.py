"""
Appended Module: Category-Level Cross-Generalization Evaluator for NegBench.

This script extends the evaluation suite WITHOUT modifying any existing code files.
It evaluates 6 Scoring Heads (Cosine, Weighted Cosine, Bilinear, Logistic Regression,
Shallow MLP, Deep MLP) under a strict 100% Unseen Category Split (GroupKFold by object_name).
This directly tests whether Bilinear / MLP learn a universal 'matching function' (Similarity Bottleneck)
or memorize object-specific patterns.
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import open_clip

# Import existing modules
from evaluation.scoring_heads import (
    BaseScorer,
    build_scorer
)
from evaluation.eval_scoring_heads import (
    set_seed,
    train_and_eval_fold,
    compute_mcq_accuracy_breakdown
)


def extract_mcq_embeddings_with_objects(
    model: nn.Module,
    tokenizer: Any,
    preprocess: Any,
    csv_file: str,
    device: str = "cuda",
    batch_size: int = 64,
    image_root: str = "",
    group_col: str = "object_name"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]:
    """
    Extract frozen CLIP embeddings and object/category metadata for Category Cross-Generalization.
    """
    model.eval()

    df = pd.read_csv(csv_file, sep=",")
    caption_cols = [c for c in df.columns if c.startswith("caption_")]
    num_answers = len(caption_cols) if caption_cols else 4

    path_col = "image_path"
    if "image_path" not in df.columns:
        if "filepath" in df.columns:
            path_col = "filepath"
        else:
            path_col = df.columns[0]

    img_embed_list = []
    text_embed_list = []
    target_list = []
    q_type_list = []
    object_name_list = []

    print(f"\nExtracting CLIP embeddings & category attributes for {len(df)} samples from {csv_file}...")
    dataset_dir = os.path.dirname(os.path.abspath(csv_file))

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Caching Features & Objects"):
        rel_img_path = str(row[path_col])
        full_img_path = rel_img_path
        if not os.path.exists(full_img_path):
            candidates = [
                os.path.join(image_root, rel_img_path) if image_root else "",
                os.path.join(dataset_dir, rel_img_path),
                os.path.join(dataset_dir, "images", rel_img_path),
                os.path.join(os.path.dirname(dataset_dir), "images", rel_img_path),
                os.path.join(os.path.dirname(dataset_dir), rel_img_path)
            ]
            for cand in candidates:
                if cand and os.path.exists(cand):
                    full_img_path = cand
                    break

        if not os.path.exists(full_img_path):
            continue

        try:
            img = Image.open(full_img_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
        except Exception as e:
            continue

        captions = [str(row[f"caption_{i}"]) for i in range(num_answers)]
        correct_answer = int(row["correct_answer"]) if "correct_answer" in row else 0
        q_type = str(row["correct_answer_template"]) if "correct_answer_template" in row else "mcq"

        # Determine object / category attribute for grouping
        if group_col in row and pd.notna(row[group_col]):
            obj_name = str(row[group_col])
        elif "object_name" in row and pd.notna(row["object_name"]):
            obj_name = str(row["object_name"])
        elif "category" in row and pd.notna(row["category"]):
            obj_name = str(row["category"])
        elif "source_template" in row and pd.notna(row["source_template"]):
            obj_name = str(row["source_template"])
        else:
            obj_name = os.path.basename(rel_img_path).split("_")[0]

        with torch.no_grad():
            img_feat = F.normalize(model.encode_image(img_tensor), dim=-1).cpu()
            tokens = tokenizer(captions).to(device)
            text_feat = F.normalize(model.encode_text(tokens), dim=-1).cpu()

        img_embed_list.append(img_feat)
        text_embed_list.append(text_feat.unsqueeze(0))
        target_list.append(correct_answer)
        q_type_list.append(q_type)
        object_name_list.append(obj_name)

    all_img_embeds = torch.cat(img_embed_list, dim=0)
    all_text_embeds = torch.cat(text_embed_list, dim=0)
    all_targets = torch.tensor(target_list, dtype=torch.long)

    unique_objs = len(set(object_name_list))
    print(f"✅ Successfully cached embeddings for {all_img_embeds.shape[0]} samples across {unique_objs} unique categories.")
    return all_img_embeds, all_text_embeds, all_targets, q_type_list, object_name_list


def evaluate_category_generalization(
    img_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    targets: torch.Tensor,
    question_types: List[str],
    object_names: List[str],
    device: str = "cuda",
    n_splits: int = 5,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Run Group Cross-Validation across 6 scoring models on 100% Unseen Category Splits.
    """
    feature_dim = img_embeds.shape[1]
    N = len(targets)
    targets_np = targets.numpy()

    scoring_models = [
        ("Cosine", "Very Low", "CLIP Default"),
        ("Weighted Cosine", "Low", "Is feature weighting sufficient?"),
        ("Bilinear", "Medium", "Are dim interactions required?"),
        ("Logistic Regression", "Medium", "Is linear boundary sufficient?"),
        ("Shallow MLP", "High", "Is non-linearity required?"),
        ("Deep MLP", "Very High", "Is expressiveness lacking?")
    ]

    unique_qtypes, qtype_indices = np.unique(question_types, return_inverse=True)
    if len(unique_qtypes) < 2:
        qtype_indices = targets_np

    unique_objs, group_indices = np.unique(object_names, return_inverse=True)
    print(f"\n⚡ Executing Category Cross-Generalization Split across {len(unique_objs)} unique categories.")

    if StratifiedGroupKFold is not None and len(unique_objs) >= n_splits:
        splitter = StratifiedGroupKFold(n_splits=n_splits)
        splits_generator = list(splitter.split(np.zeros(N), qtype_indices, groups=group_indices))
    else:
        splitter = GroupKFold(n_splits=n_splits)
        splits_generator = list(splitter.split(np.zeros(N), qtype_indices, groups=group_indices))

    all_results = {}

    for model_name, expr_level, hypothesis in scoring_models:
        print(f"\n" + "="*75)
        print(f"Evaluating Model : {model_name:20s} | Expressiveness: {expr_level:10s}")
        print(f"Evaluation Mode : Category Cross-Generalization (100% Unseen Categories)")
        print("="*75)

        oof_predictions = np.zeros(N, dtype=int)

        for fold, (train_idx, val_idx) in enumerate(splits_generator):
            train_imgs, train_texts, train_y = img_embeds[train_idx], text_embeds[train_idx], targets[train_idx]
            val_imgs, val_texts, val_y = img_embeds[val_idx], text_embeds[val_idx], targets[val_idx]

            train_ds = TensorDataset(train_imgs, train_texts, train_y)
            val_ds = TensorDataset(val_imgs, val_texts, val_y)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            scorer = build_scorer(model_name, feature_dim)
            _, fold_preds = train_and_eval_fold(
                scorer, train_loader, val_loader, device=device, epochs=epochs, lr=lr
            )

            oof_predictions[val_idx] = fold_preds

        metrics = compute_mcq_accuracy_breakdown(oof_predictions, targets_np, question_types)
        metrics["expressiveness"] = expr_level
        metrics["hypothesis"] = hypothesis
        metrics["evaluation_mode"] = "category_cross_generalization"

        all_results[model_name] = metrics
        print(f"  --> Unseen Objects OOF Accuracy: Total={metrics['total_accuracy']:.2f}% | Pos={metrics['positive_accuracy']:.2f}% | Neg={metrics['negative_accuracy']:.2f}% | Hyb={metrics['hybrid_accuracy']:.2f}%")

    return all_results


def plot_category_generalization_comparison(results: Dict[str, Dict[str, Any]], output_dir: str):
    """Render bar chart comparing accuracy on 100% unseen categories."""
    models = list(results.keys())
    total_accs = [results[m]["total_accuracy"] for m in models]
    pos_accs = [results[m]["positive_accuracy"] for m in models]
    neg_accs = [results[m]["negative_accuracy"] for m in models]
    hyb_accs = [results[m]["hybrid_accuracy"] for m in models]

    df_plot = pd.DataFrame({
        "Model": models * 4,
        "Accuracy (%)": total_accs + pos_accs + neg_accs + hyb_accs,
        "Metric": ["Total"] * len(models) + ["Positive"] * len(models) + ["Negative"] * len(models) + ["Hybrid"] * len(models)
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = {"Total": "#1f77b4", "Positive": "#2ca02c", "Negative": "#d62728", "Hybrid": "#ff7f0e"}
    sns.barplot(data=df_plot, x="Model", y="Accuracy (%)", hue="Metric", palette=palette, ax=ax)

    ax.set_title("Category Cross-Generalization (Unseen Objects Split)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Scoring Function Model (Increasing Expressiveness ->)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Out-of-Fold Accuracy on Unseen Objects (%)", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", ls="--", alpha=0.5)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    ax.legend(title="MCQ Question Type", fontsize=10)

    for bar, acc in zip(ax.patches[:len(models)], total_accs):
        h = bar.get_height()
        ax.annotate(f"{acc:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "scoring_head_category_generalization.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved category cross-generalization bar plot to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Category-Level Cross-Generalization Evaluator")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model architecture")
    parser.add_argument("--pretrained", type=str, default="openai", help="Pretrained weights checkpoint tag")
    parser.add_argument("--coco-mcq", type=str, default="COCO_val_mcq_llama3.1_rephrased.csv", help="Path to MCQ CSV file")
    parser.add_argument("--image-root", type=str, default="", help="Root directory containing images")
    parser.add_argument("--output-dir", type=str, default="logs/evaluation/category_generalization_experiments", help="Output directory")
    parser.add_argument("--group-col", type=str, default="object_name", help="Column used for Category Cross-Generalization grouping")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of Cross-Validation folds")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per fold")
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Category Cross-Generalization Evaluation on device: {device}")

    if not os.path.exists(args.coco_mcq):
        alt_paths = [
            "COCO_val_mcq_llama3.1_rephrased.csv",
            "benchmarks/data/images/COCO_val_mcq_llama3.1_rephrased.csv",
            "data/images/COCO_val_mcq_llama3.1_rephrased.csv"
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                args.coco_mcq = alt
                break

    assert os.path.exists(args.coco_mcq), f"Could not locate MCQ CSV file at: {args.coco_mcq}"

    print(f"Loading OpenCLIP model {args.model} ({args.pretrained})...")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    img_embeds, text_embeds, targets, question_types, object_names = extract_mcq_embeddings_with_objects(
        model, tokenizer, preprocess_val, args.coco_mcq, device=device, batch_size=args.batch_size, image_root=args.image_root, group_col=args.group_col
    )

    results = evaluate_category_generalization(
        img_embeds, text_embeds, targets, question_types, object_names,
        device=device, n_splits=args.n_splits, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed
    )

    rows = []
    print("\n" + "="*85)
    print("  FINAL CATEGORY CROSS-GENERALIZATION SUMMARY (100% UNSEEN OBJECTS)")
    print("="*85)
    print(f"{'Model':22s} | {'Expressiveness':14s} | {'Total Acc':9s} | {'Pos Acc':8s} | {'Neg Acc':8s} | {'Hyb Acc':8s}")
    print("-" * 85)

    for mname, mdata in results.items():
        print(f"{mname:22s} | {mdata['expressiveness']:14s} | {mdata['total_accuracy']:8.2f}% | {mdata['positive_accuracy']:7.2f}% | {mdata['negative_accuracy']:7.2f}% | {mdata['hybrid_accuracy']:7.2f}%")
        rows.append({
            "Model": mname,
            "Expressiveness": mdata["expressiveness"],
            "Hypothesis": mdata["hypothesis"],
            "Evaluation_Mode": "category_cross_generalization",
            "Total_Accuracy_Pct": mdata["total_accuracy"],
            "Positive_Accuracy_Pct": mdata["positive_accuracy"],
            "Negative_Accuracy_Pct": mdata["negative_accuracy"],
            "Hybrid_Accuracy_Pct": mdata["hybrid_accuracy"],
            "Total_Samples": mdata["total_samples"]
        })
    print("="*85)

    csv_path = os.path.join(args.output_dir, "category_generalization_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    json_path = os.path.join(args.output_dir, "category_generalization_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_category_generalization_comparison(results, args.output_dir)
    print(f"\n✅ All category cross-generalization results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
