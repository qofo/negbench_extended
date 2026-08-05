"""
Evaluation Script for Expressive Scoring Heads on MCQ Benchmarks in NegBench.

Compares 6 scoring functions of varying expressiveness (Cosine, Weighted Cosine,
Bilinear, Logistic Regression, Shallow MLP, Deep MLP) using frozen CLIP embeddings.
Uses 5-Fold Cross Validation to ensure leak-free out-of-fold (OOF) evaluation.
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
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import open_clip

# Import scoring heads
from evaluation.scoring_heads import (
    BaseScorer,
    CosineScorer,
    WeightedCosineScorer,
    BilinearScorer,
    LogisticRegressionScorer,
    ShallowMLPScorer,
    DeepMLPScorer,
    build_scorer
)
from training.data import CsvMCQDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_mcq_embeddings(
    model: nn.Module,
    tokenizer: Any,
    preprocess: Any,
    csv_file: str,
    device: str = "cuda",
    batch_size: int = 64,
    image_root: str = ""
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[List[str]]]:
    """
    Extract and cache frozen CLIP image and candidate text embeddings for MCQ dataset.

    Returns:
        img_embeds: (N, D)
        text_embeds: (N, K, D)
        targets: (N,)
        question_types: List of str (length N)
        caption_types: List of List of str (N, K)
    """
    model.eval()

    # Load dataset via CsvMCQDataset or direct pandas
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
    cap_type_list = []

    print(f"\nExtracting CLIP embeddings for {len(df)} samples from {csv_file}...")

    # Canonical caption types if missing from row
    default_caption_types = ['gt', 'hybrid', 'positive', 'negative'][:num_answers]

    dataset_dir = os.path.dirname(os.path.abspath(csv_file))

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Caching Features"):
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
            # Skip missing images gracefully
            continue

        try:
            img = Image.open(full_img_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
        except Exception as e:
            continue

        # Auto-detect caption column format (MCQ caption_0..N vs Paired positive_caption/negative_caption)
        if caption_cols:
            captions = [str(row[c]) for c in caption_cols]
            correct_answer = int(row["correct_answer"]) if "correct_answer" in row else 0
            q_type = str(row["correct_answer_template"]) if "correct_answer_template" in row else "mcq"
        elif "positive_caption" in row and "negative_caption" in row:
            captions = [str(row["positive_caption"]), str(row["negative_caption"])]
            # If object_in_image is True, ground truth is positive (0), else negative (1)
            obj_in_img = str(row.get("object_in_image", "True")).strip().lower() in ["true", "1", "t", "yes"]
            correct_answer = 0 if obj_in_img else 1
            q_type = "positive" if obj_in_img else "negative"
        else:
            # Fallback for caption_i
            captions = [str(row[f"caption_{i}"]) for i in range(num_answers)]
            correct_answer = int(row["correct_answer"]) if "correct_answer" in row else 0
            q_type = str(row["correct_answer_template"]) if "correct_answer_template" in row else "mcq"

        # Encode image
        with torch.no_grad():
            img_feat = F.normalize(model.encode_image(img_tensor), dim=-1).cpu()

            # Encode text options
            tokens = tokenizer(captions).to(device)
            text_feat = F.normalize(model.encode_text(tokens), dim=-1).cpu()  # (K, D)

        img_embed_list.append(img_feat)  # (1, D)
        text_embed_list.append(text_feat.unsqueeze(0))  # (1, K, D)
        target_list.append(correct_answer)
        q_type_list.append(q_type)
        cap_type_list.append(default_caption_types)

    all_img_embeds = torch.cat(img_embed_list, dim=0)  # (N, D)
    all_text_embeds = torch.cat(text_embed_list, dim=0)  # (N, K, D)
    all_targets = torch.tensor(target_list, dtype=torch.long)  # (N,)

    print(f"✅ Successfully cached embeddings for {all_img_embeds.shape[0]} valid samples (Dim: {all_img_embeds.shape[1]}, Options: {all_text_embeds.shape[1]})")
    return all_img_embeds, all_text_embeds, all_targets, q_type_list, cap_type_list


def train_and_eval_fold(
    scorer: BaseScorer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4
) -> Tuple[BaseScorer, np.ndarray]:
    """Train scoring model on train_loader and predict out-of-fold scores on val_loader."""
    scorer = scorer.to(device)
    
    # Cosine scorer or any zero-parameter scorer requires no training
    if isinstance(scorer, CosineScorer) or len(list(scorer.parameters())) == 0:
        scorer.eval()
        oof_preds = []
        with torch.no_grad():
            for imgs, texts, _ in val_loader:
                imgs, texts = imgs.to(device), texts.to(device)
                scores = scorer(imgs, texts)
                preds = torch.argmax(scores, dim=1).cpu().numpy()
                oof_preds.append(preds)
        return scorer, np.concatenate(oof_preds)

    optimizer = torch.optim.AdamW(scorer.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        scorer.train()
        for imgs, texts, targets in train_loader:
            imgs, texts, targets = imgs.to(device), texts.to(device), targets.to(device)
            optimizer.zero_grad()
            scores = scorer(imgs, texts)  # (B, K)
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()

    # Predict on validation split
    scorer.eval()
    oof_preds = []
    with torch.no_grad():
        for imgs, texts, _ in val_loader:
            imgs, texts = imgs.to(device), texts.to(device)
            scores = scorer(imgs, texts)
            preds = torch.argmax(scores, dim=1).cpu().numpy()
            oof_preds.append(preds)

    return scorer, np.concatenate(oof_preds)


def compute_mcq_accuracy_breakdown(
    oof_preds: np.ndarray,
    targets: np.ndarray,
    question_types: List[str]
) -> Dict[str, Any]:
    """Compute overall, positive, negative, and hybrid MCQ accuracy breakdown."""
    total_samples = len(targets)
    correct_mask = (oof_preds == targets)
    total_acc = float(np.mean(correct_mask)) * 100.0

    q_types_np = np.array(question_types)
    type_accs = {}

    for qt in ["positive", "negative", "hybrid"]:
        mask = (q_types_np == qt)
        if np.sum(mask) > 0:
            acc = float(np.mean(correct_mask[mask])) * 100.0
            type_accs[f"{qt}_accuracy"] = acc
            type_accs[f"{qt}_count"] = int(np.sum(mask))
        else:
            type_accs[f"{qt}_accuracy"] = 0.0
            type_accs[f"{qt}_count"] = 0

    return {
        "total_accuracy": total_acc,
        "positive_accuracy": type_accs.get("positive_accuracy", 0.0),
        "negative_accuracy": type_accs.get("negative_accuracy", 0.0),
        "hybrid_accuracy": type_accs.get("hybrid_accuracy", 0.0),
        "total_samples": total_samples,
    }


def evaluate_all_scoring_heads(
    img_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    targets: torch.Tensor,
    question_types: List[str],
    device: str = "cuda",
    n_splits: int = 5,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """Run 5-Fold Stratified CV across 6 scoring models and collect OOF metrics."""
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

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Convert question_types to category integers for StratifiedKFold
    unique_qtypes, qtype_indices = np.unique(question_types, return_inverse=True)
    if len(unique_qtypes) < 2:
        qtype_indices = targets_np

    all_results = {}

    for model_name, expr_level, hypothesis in scoring_models:
        print(f"\n" + "="*70)
        print(f"Evaluating Model: {model_name:20s} | Expressiveness: {expr_level:10s}")
        print(f"Hypothesis       : {hypothesis}")
        print("="*70)

        oof_predictions = np.zeros(N, dtype=int)

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(N), qtype_indices)):
            train_imgs, train_texts, train_y = img_embeds[train_idx], text_embeds[train_idx], targets[train_idx]
            val_imgs, val_texts, val_y = img_embeds[val_idx], text_embeds[val_idx], targets[val_idx]

            train_ds = TensorDataset(train_imgs, train_texts, train_y)
            val_ds = TensorDataset(val_imgs, val_texts, val_y)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            # Build fresh scorer for fold
            scorer = build_scorer(model_name, feature_dim)
            _, fold_preds = train_and_eval_fold(
                scorer, train_loader, val_loader, device=device, epochs=epochs, lr=lr
            )

            oof_predictions[val_idx] = fold_preds

        # Compute accuracy breakdown on out-of-fold predictions
        metrics = compute_mcq_accuracy_breakdown(oof_predictions, targets_np, question_types)
        metrics["expressiveness"] = expr_level
        metrics["hypothesis"] = hypothesis

        all_results[model_name] = metrics
        print(f"  --> OOF Total Accuracy: {metrics['total_accuracy']:.2f}% | Pos: {metrics['positive_accuracy']:.2f}% | Neg: {metrics['negative_accuracy']:.2f}% | Hyb: {metrics['hybrid_accuracy']:.2f}%")

    return all_results


def plot_scoring_head_comparison(results: Dict[str, Dict[str, Any]], output_dir: str):
    """Render comprehensive comparative bar chart of accuracy vs expressiveness."""
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

    ax.set_title("NegBench CLIP MCQ Evaluation Across Decision Boundaries & Expressiveness", fontsize=13, fontweight="bold")
    ax.set_xlabel("Scoring Function Model (Increasing Expressiveness ->)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Out-of-Fold Accuracy (%)", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", ls="--", alpha=0.5)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    ax.legend(title="MCQ Question Type", fontsize=10)

    # Annotate bars with total accuracy
    for bar, acc in zip(ax.patches[:len(models)], total_accs):
        h = bar.get_height()
        ax.annotate(f"{acc:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "scoring_head_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved comparison bar plot to: {plot_path}")


def train_and_save_full_scorer(
    model_name: str,
    img_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    targets: torch.Tensor,
    save_path: str,
    device: str = "cuda",
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 64
):
    """Train a full-dataset scorer model and export state_dict checkpoint."""
    feature_dim = img_embeds.shape[1]
    scorer = build_scorer(model_name, feature_dim).to(device)
    if isinstance(scorer, CosineScorer):
        print("CosineScorer has no trainable parameters to save.")
        return

    train_ds = TensorDataset(img_embeds, text_embeds, targets)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining full-dataset '{model_name}' scorer for checkpoint export...")
    for epoch in range(epochs):
        scorer.train()
        for imgs, texts, y in train_loader:
            imgs, texts, y = imgs.to(device), texts.to(device), y.to(device)
            optimizer.zero_grad()
            scores = scorer(imgs, texts)
            loss = criterion(scores, y)
            loss.backward()
            optimizer.step()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({
        "state_dict": scorer.state_dict(),
        "model_name": model_name,
        "feature_dim": feature_dim
    }, save_path)
    print(f"✅ Saved trained '{model_name}' scorer checkpoint to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Expressive Scoring Heads on CLIP MCQ Evaluation")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model architecture")
    parser.add_argument("--pretrained", type=str, default="openai", help="Pretrained weights checkpoint tag")
    parser.add_argument("--coco-mcq", type=str, default="COCO_val_mcq_llama3.1_rephrased.csv", help="Path to MCQ CSV file")
    parser.add_argument("--image-root", type=str, default="", help="Root directory containing images")
    parser.add_argument("--output-dir", type=str, default="logs/evaluation/scoring_head_experiments", help="Output directory")
    parser.add_argument("--save-scorer-path", type=str, default=None, help="Path to save trained scorer checkpoint (.pt)")
    parser.add_argument("--save-scorer-model", type=str, default="Deep MLP", help="Scorer model architecture to save")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of Cross-Validation folds")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per fold")
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on device: {device}")

    # Fallback CSV path resolution
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

    # Load OpenCLIP model
    print(f"Loading OpenCLIP model {args.model} ({args.pretrained})...")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    # Step 1: Cache Image & Text Embeddings
    img_embeds, text_embeds, targets, question_types, caption_types = extract_mcq_embeddings(
        model, tokenizer, preprocess_val, args.coco_mcq, device=device, batch_size=args.batch_size, image_root=args.image_root
    )

    # Step 2: Evaluate 6 Scoring Heads via 5-Fold Stratified Cross Validation
    results = evaluate_all_scoring_heads(
        img_embeds, text_embeds, targets, question_types,
        device=device, n_splits=args.n_splits, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed
    )

    # Step 3: Export Summary Table & JSON
    rows = []
    print("\n" + "="*85)
    print("  FINAL MCQ EVALUATION SUMMARY ACROSS DECISION BOUNDARIES & EXPRESSIVENESS")
    print("="*85)
    print(f"{'Model':22s} | {'Expressiveness':14s} | {'Total Acc':9s} | {'Pos Acc':8s} | {'Neg Acc':8s} | {'Hyb Acc':8s}")
    print("-" * 85)

    for mname, mdata in results.items():
        print(f"{mname:22s} | {mdata['expressiveness']:14s} | {mdata['total_accuracy']:8.2f}% | {mdata['positive_accuracy']:7.2f}% | {mdata['negative_accuracy']:7.2f}% | {mdata['hybrid_accuracy']:7.2f}%")
        rows.append({
            "Model": mname,
            "Expressiveness": mdata["expressiveness"],
            "Hypothesis": mdata["hypothesis"],
            "Total_Accuracy_Pct": mdata["total_accuracy"],
            "Positive_Accuracy_Pct": mdata["positive_accuracy"],
            "Negative_Accuracy_Pct": mdata["negative_accuracy"],
            "Hybrid_Accuracy_Pct": mdata["hybrid_accuracy"],
            "Total_Samples": mdata["total_samples"]
        })
    print("="*85)

    df_summary = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "scoring_head_summary.csv")
    df_summary.to_csv(csv_path, index=False)

    json_path = os.path.join(args.output_dir, "scoring_head_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Step 4: Generate Plot
    plot_scoring_head_comparison(results, args.output_dir)

    # Step 5: Save trained scorer checkpoint if requested
    save_checkpoint_path = args.save_scorer_path
    if not save_checkpoint_path:
        save_checkpoint_path = os.path.join(args.output_dir, "checkpoints", "deep_mlp_scorer.pt")

    train_and_save_full_scorer(
        args.save_scorer_model, img_embeds, text_embeds, targets, save_checkpoint_path,
        device=device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size
    )

    print(f"\n✅ All results, comparison plots, and scorer checkpoints successfully saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

