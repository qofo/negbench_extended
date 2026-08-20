"""
Vision Ablation Shortcut Diagnostic for Trained Scoring Heads.

Critique #3: Do the trained scorers (Bilinear 74.8%, Deep MLP 86.7%) actually
use vision information, or are they exploiting text-only shortcuts?

Methodology:
  Condition A (Original):  Normal image embeddings (reference baseline).
  Condition B (Zero):      Replace ALL image embeddings with zero vectors.
  Condition C (Shuffle):   Randomly permute image embeddings across samples.
  Condition D (Gaussian):  Replace image embeddings with random Gaussian vectors.

If Condition B/C/D accuracy ≈ Condition A, the scorer has learned a text-only shortcut.
The negation matching improvement is NOT due to better image-text interaction.

Breakdown: positive / negative / hybrid question types.

Usage:
    python -m benchmarks.src.evaluation.eval_vision_ablation_shortcut \\
        --model ViT-B-32 --pretrained openai \\
        --coco-mcq COCO_val_mcq_llama3.1_rephrased.csv \\
        --output-dir logs/evaluation/vision_ablation_shortcut
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

import open_clip

from src.evaluation.scoring_heads import (
    BaseScorer,
    CosineScorer,
    WeightedCosineScorer,
    BilinearScorer,
    LogisticRegressionScorer,
    ShallowMLPScorer,
    DeepMLPScorer,
    build_scorer,
)
from src.evaluation.eval_scoring_heads import (
    extract_mcq_embeddings,
    compute_mcq_accuracy_breakdown,
)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

from benchmarks.src.analysis.config import set_seed  # noqa: E402 — centralized seed control


def compute_extended_breakdown(
    oof_preds: np.ndarray,
    targets: np.ndarray,
    question_types: List[str],
) -> Dict[str, Any]:
    """Compute accuracy breakdown by question type with counts."""
    total_samples = len(targets)
    correct_mask = (oof_preds == targets)
    total_acc = float(np.mean(correct_mask)) * 100.0

    q_types_np = np.array(question_types)
    result = {
        "total_accuracy": total_acc,
        "total_samples": total_samples,
        "total_correct": int(np.sum(correct_mask)),
    }

    for qt in sorted(set(question_types)):
        mask = (q_types_np == qt)
        n_qt = int(np.sum(mask))
        if n_qt > 0:
            acc = float(np.mean(correct_mask[mask])) * 100.0
            n_correct = int(np.sum(correct_mask[mask]))
        else:
            acc = 0.0
            n_correct = 0
        result[f"{qt}_accuracy"] = acc
        result[f"{qt}_count"] = n_qt
        result[f"{qt}_correct"] = n_correct

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Train + Ablated Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def train_fold_return_scorer(
    scorer: BaseScorer,
    train_loader: DataLoader,
    device: str = "cuda",
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> BaseScorer:
    """Train scorer on train_loader and return trained model (no eval)."""
    scorer = scorer.to(device)

    if isinstance(scorer, CosineScorer) or len(list(scorer.parameters())) == 0:
        return scorer

    optimizer = torch.optim.AdamW(scorer.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        scorer.train()
        for imgs, texts, targets in train_loader:
            imgs, texts, targets = imgs.to(device), texts.to(device), targets.to(device)
            optimizer.zero_grad()
            scores = scorer(imgs, texts)
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()

    return scorer


def eval_scorer_on_loader(
    scorer: BaseScorer,
    val_loader: DataLoader,
    device: str = "cuda",
) -> np.ndarray:
    """Evaluate trained scorer on val_loader and return predictions."""
    scorer.eval()
    oof_preds = []
    with torch.no_grad():
        for imgs, texts, _ in val_loader:
            imgs, texts = imgs.to(device), texts.to(device)
            scores = scorer(imgs, texts)
            preds = torch.argmax(scores, dim=1).cpu().numpy()
            oof_preds.append(preds)
    return np.concatenate(oof_preds)


def create_ablated_loader(
    img_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    targets: torch.Tensor,
    indices: np.ndarray,
    ablation_mode: str,
    batch_size: int = 64,
    seed: int = 42,
) -> DataLoader:
    """
    Create a DataLoader with vision embeddings ablated according to mode.

    Modes:
      "original":  No modification.
      "zero":      Replace all image embeddings with zero vectors.
      "shuffle":   Randomly permute image embeddings across samples.
      "gaussian":  Replace with random Gaussian vectors (same norm distribution).
    """
    sel_imgs = img_embeds[indices].clone()
    sel_texts = text_embeds[indices]
    sel_y = targets[indices]

    if ablation_mode == "zero":
        sel_imgs = torch.zeros_like(sel_imgs)
    elif ablation_mode == "shuffle":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(sel_imgs))
        sel_imgs = sel_imgs[perm]
    elif ablation_mode == "gaussian":
        # Match mean norm of original embeddings
        orig_norms = sel_imgs.norm(dim=-1, keepdim=True)
        mean_norm = orig_norms.mean().item()
        sel_imgs = torch.randn_like(sel_imgs)
        sel_imgs = F.normalize(sel_imgs, dim=-1) * mean_norm
    elif ablation_mode == "original":
        pass
    else:
        raise ValueError(f"Unknown ablation_mode: {ablation_mode}")

    ds = TensorDataset(sel_imgs, sel_texts, sel_y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ──────────────────────────────────────────────────────────────────────────────
# Main Experiment
# ──────────────────────────────────────────────────────────────────────────────

def run_vision_ablation_diagnostic(
    img_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    targets: torch.Tensor,
    question_types: List[str],
    device: str = "cuda",
    n_splits: int = 5,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Run vision ablation shortcut diagnostic across all scorer types.

    For each scorer type:
      1. Train on ORIGINAL embeddings (normal 5-Fold CV).
      2. At evaluation time, test with 4 conditions:
         (a) Original vision  (b) Zero vision  (c) Shuffled vision  (d) Gaussian vision
      3. Report accuracy breakdown per condition per question type.

    Returns:
        { scorer_name: { condition: { accuracy breakdown dict } } }
    """
    feature_dim = img_embeds.shape[1]
    N = len(targets)
    targets_np = targets.numpy()

    scoring_models = [
        ("Cosine",              "cosine",              "Very Low"),
        ("Weighted Cosine",     "weighted_cosine",     "Low"),
        ("Bilinear",            "bilinear",            "Medium"),
        ("Logistic Regression", "logistic_regression", "Medium"),
        ("Shallow MLP",         "shallow_mlp",         "High"),
        ("Deep MLP",            "deep_mlp",            "Very High"),
    ]

    ablation_modes = ["original", "zero", "shuffle", "gaussian"]

    unique_qtypes, qtype_indices = np.unique(question_types, return_inverse=True)
    if len(unique_qtypes) < 2:
        qtype_indices = targets_np

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    all_results = {}

    for model_name, model_type, expr_level in scoring_models:
        print(f"\n{'='*80}")
        print(f"  Scorer: {model_name:22s} | Expressiveness: {expr_level}")
        print(f"{'='*80}")

        # Store predictions per condition
        condition_preds = {mode: np.zeros(N, dtype=int) for mode in ablation_modes}

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(N), qtype_indices)):
            print(f"\n  Fold {fold+1}/{n_splits}  (train={len(train_idx)}, val={len(val_idx)})")

            # ── Train on ORIGINAL embeddings ──
            train_ds = TensorDataset(
                img_embeds[train_idx], text_embeds[train_idx], targets[train_idx]
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            scorer = build_scorer(model_type, feature_dim)
            scorer = train_fold_return_scorer(
                scorer, train_loader, device=device, epochs=epochs, lr=lr
            )

            # ── Evaluate under each ablation condition ──
            for mode in ablation_modes:
                val_loader = create_ablated_loader(
                    img_embeds, text_embeds, targets,
                    val_idx, ablation_mode=mode,
                    batch_size=batch_size, seed=seed + fold,
                )
                fold_preds = eval_scorer_on_loader(scorer, val_loader, device=device)
                condition_preds[mode][val_idx] = fold_preds

        # ── Compute accuracy breakdown per condition ──
        model_results = {}
        for mode in ablation_modes:
            metrics = compute_extended_breakdown(
                condition_preds[mode], targets_np, question_types
            )
            model_results[mode] = metrics

            print(f"\n    [{mode.upper():10s}] Total: {metrics['total_accuracy']:.2f}%", end="")
            for qt in ["positive", "negative", "hybrid"]:
                k = f"{qt}_accuracy"
                if k in metrics:
                    print(f" | {qt.capitalize()[:3]}: {metrics[k]:.2f}%", end="")
            print()

        # ── Compute delta (drop from original) ──
        orig_total = model_results["original"]["total_accuracy"]
        for mode in ["zero", "shuffle", "gaussian"]:
            delta = model_results[mode]["total_accuracy"] - orig_total
            model_results[mode]["delta_from_original"] = delta
            print(f"    delta({mode:8s} - original) = {delta:+.2f}%")

        all_results[model_name] = model_results

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_vision_ablation_results(
    results: Dict[str, Dict[str, Dict]],
    output_dir: str,
):
    """Generate grouped bar chart: Original vs Zero vs Shuffle vs Gaussian per scorer."""
    models = list(results.keys())
    conditions = ["original", "zero", "shuffle", "gaussian"]
    condition_labels = ["Original", "Zero Vision", "Shuffle Vision", "Gaussian Vision"]

    # ── Plot 1: Total Accuracy ──
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(models))
    width = 0.2

    colors = ["#1f77b4", "#d62728", "#ff7f0e", "#9467bd"]
    hatches = ["", "//", "\\\\", "xx"]

    for i, (cond, label, color, hatch) in enumerate(
        zip(conditions, condition_labels, colors, hatches)
    ):
        accs = [results[m][cond]["total_accuracy"] for m in models]
        bars = ax.bar(
            x + (i - 1.5) * width, accs, width,
            label=label, color=color, alpha=0.85, hatch=hatch, edgecolor="white"
        )
        for bar, acc in zip(bars, accs):
            h = bar.get_height()
            ax.annotate(
                f"{acc:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, fontweight="bold"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=11)
    ax.set_ylabel("Out-of-Fold Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Vision Ablation Shortcut Diagnostic\n"
        "If ablated approx original -> scorer uses text-only shortcut",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "vision_ablation_total_accuracy.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_path}")

    # ── Plot 2: Per Question Type Breakdown (Negative vs Positive) ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle(
        "Vision Ablation by Question Type: Positive / Negative / Hybrid",
        fontsize=14, fontweight="bold"
    )

    for ax, qt in zip(axes, ["positive", "negative", "hybrid"]):
        key = f"{qt}_accuracy"
        for i, (cond, label, color, hatch) in enumerate(
            zip(conditions, condition_labels, colors, hatches)
        ):
            accs = [
                results[m][cond].get(key, 0.0) for m in models
            ]
            bars = ax.bar(
                x + (i - 1.5) * width, accs, width,
                label=label, color=color, alpha=0.85, hatch=hatch, edgecolor="white"
            )
            for bar, acc in zip(bars, accs):
                h = bar.get_height()
                ax.annotate(
                    f"{acc:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6
                )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{qt.capitalize()} Questions", fontsize=12, fontweight="bold")
        ax.set_ylabel("Accuracy (%)" if qt == "positive" else "", fontsize=11)
        ax.grid(True, axis="y", ls="--", alpha=0.4)
        ax.set_ylim(0, 110)

    axes[0].legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "vision_ablation_per_question_type.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    # ── Plot 3: Delta (Accuracy Drop) Heatmap ──
    delta_data = []
    for m in models:
        row = []
        for cond in ["zero", "shuffle", "gaussian"]:
            delta = results[m][cond]["total_accuracy"] - results[m]["original"]["total_accuracy"]
            row.append(delta)
        delta_data.append(row)

    delta_arr = np.array(delta_data)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(delta_arr, cmap="RdYlGn", aspect="auto", vmin=-80, vmax=10)

    ax.set_xticks(range(3))
    ax.set_xticklabels(["Zero Vision", "Shuffle Vision", "Gaussian Vision"], fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)

    for i in range(len(models)):
        for j in range(3):
            val = delta_arr[i, j]
            color = "white" if abs(val) > 30 else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax.set_title(
        "Accuracy Drop from Vision Ablation (delta %)\n"
        "Green approx 0 -> TEXT SHORTCUT  |  Red < 0 -> Vision is used",
        fontsize=12, fontweight="bold"
    )
    plt.colorbar(im, ax=ax, label="delta Accuracy (%)")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "vision_ablation_delta_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary Table
# ──────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: Dict[str, Dict[str, Dict]]):
    """Print comprehensive summary table."""
    print("\n" + "=" * 120)
    print("  VISION ABLATION SHORTCUT DIAGNOSTIC — COMPREHENSIVE SUMMARY")
    print("=" * 120)

    header = f"{'Scorer':22s} | {'Condition':12s} | {'Total':7s} | {'Pos':7s} | {'Neg':7s} | {'Hyb':7s} | {'D Total':8s} | {'Shortcut?':10s}"
    print(header)
    print("-" * 120)

    for model_name, model_results in results.items():
        orig_total = model_results["original"]["total_accuracy"]

        for cond in ["original", "zero", "shuffle", "gaussian"]:
            metrics = model_results[cond]
            total = metrics["total_accuracy"]
            pos = metrics.get("positive_accuracy", 0.0)
            neg = metrics.get("negative_accuracy", 0.0)
            hyb = metrics.get("hybrid_accuracy", 0.0)
            delta = total - orig_total if cond != "original" else 0.0

            # Determine shortcut verdict
            if cond == "original":
                verdict = "---"
            elif abs(delta) < 5.0:
                verdict = "!! YES"
            elif abs(delta) < 15.0:
                verdict = "! PARTIAL"
            else:
                verdict = "OK NO"

            print(
                f"  {model_name:20s} | {cond:12s} | {total:6.2f}% | {pos:6.2f}% | "
                f"{neg:6.2f}% | {hyb:6.2f}% | {delta:+7.2f}% | {verdict}"
            )
        print("-" * 120)

    print("=" * 120)
    print("  Interpretation Guide:")
    print("    !! YES     = Accuracy drop < 5%  -> scorer mostly uses TEXT-ONLY shortcut")
    print("    !  PARTIAL = Accuracy drop 5-15% -> scorer partially relies on vision")
    print("    OK NO      = Accuracy drop > 15% -> scorer genuinely uses vision information")
    print("=" * 120)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vision Ablation Shortcut Diagnostic for Trained Scoring Heads"
    )
    parser.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP architecture")
    parser.add_argument("--pretrained", type=str, default="openai", help="Pretrained weights")
    parser.add_argument("--coco-mcq", type=str, default="COCO_val_mcq_llama3.1_rephrased.csv",
                        help="Path to MCQ CSV file")
    parser.add_argument("--image-root", type=str, default="", help="Root directory for images")
    parser.add_argument("--output-dir", type=str,
                        default="logs/evaluation/vision_ablation_shortcut",
                        help="Output directory")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per fold")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # CSV path resolution
    if not os.path.exists(args.coco_mcq):
        for alt in [
            "COCO_val_mcq_llama3.1_rephrased.csv",
            "benchmarks/data/images/COCO_val_mcq_llama3.1_rephrased.csv",
        ]:
            if os.path.exists(alt):
                args.coco_mcq = alt
                break
    assert os.path.exists(args.coco_mcq), f"MCQ CSV not found: {args.coco_mcq}"

    # Load CLIP
    print(f"\nLoading OpenCLIP {args.model} ({args.pretrained})...")
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    # Cache embeddings
    img_embeds, text_embeds, targets, question_types, _ = extract_mcq_embeddings(
        model, tokenizer, preprocess_val,
        args.coco_mcq, device=device,
        batch_size=args.batch_size, image_root=args.image_root,
    )

    # Run diagnostic
    results = run_vision_ablation_diagnostic(
        img_embeds, text_embeds, targets, question_types,
        device=device, n_splits=args.n_splits,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, seed=args.seed,
    )

    # Print summary
    print_summary_table(results)

    # Save JSON
    json_path = os.path.join(args.output_dir, "vision_ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved JSON: {json_path}")

    # Save CSV summary
    rows = []
    for model_name, model_results in results.items():
        orig_total = model_results["original"]["total_accuracy"]
        for cond, metrics in model_results.items():
            rows.append({
                "Scorer": model_name,
                "Condition": cond,
                "Total_Accuracy": metrics["total_accuracy"],
                "Positive_Accuracy": metrics.get("positive_accuracy", 0.0),
                "Negative_Accuracy": metrics.get("negative_accuracy", 0.0),
                "Hybrid_Accuracy": metrics.get("hybrid_accuracy", 0.0),
                "Delta_from_Original": metrics["total_accuracy"] - orig_total,
            })
    csv_path = os.path.join(args.output_dir, "vision_ablation_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    # Generate plots
    plot_vision_ablation_results(results, args.output_dir)

    print(f"\n  Vision Ablation Shortcut Diagnostic complete! Results: {args.output_dir}")


if __name__ == "__main__":
    main()
