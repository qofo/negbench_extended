"""
Concept Vector Subtraction (Negation Ablation) Experiment.

Research Question (RQ2): Is the negation hyperplane direction in text embeddings
a *necessary* channel for negation matching? If we project it out, do all
scorers degrade to near-random performance?

Methodology:
  1. Fit a Linear SVM / LogReg probe on text embeddings to classify
     positive vs. negative captions.
  2. Extract the learned hyperplane normal vector w_neg (unit norm).
  3. Ablate every text embedding:
         t_ablated = t - (t . w_neg) w_neg
     This removes the negation direction while preserving all orthogonal info.
  4. Re-evaluate all Scorer types (Cosine, Bilinear, MLP, NonLinear Bi-Encoder)
     on the ablated embeddings using the pre-trained scorer checkpoints.
  5. Compare ablated vs. original accuracy -> if ablated ≈ Cosine baseline,
     w_neg is a *necessary* channel (Proof of Necessity).

Usage:
    python -m benchmarks.src.evaluation.eval_concept_ablation \
        --model ViT-B-32 --pretrained openai \
        --coco-mcq <path/to/mcq.csv> --image-root <img_dir> \
        --output-dir logs/evaluation/scoring_head_experiments
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import open_clip

from src.evaluation.scoring_heads import (
    BaseScorer,
    CosineScorer,
    BilinearScorer,
    DeepMLPScorer,
    NonLinearBiEncoderScorer,
    build_scorer,
)
from src.evaluation.eval_scoring_heads import (
    extract_mcq_embeddings,
    train_and_eval_fold,
    compute_mcq_accuracy_breakdown,
)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

from benchmarks.src.analysis.config import set_seed  # noqa: E402 — centralized seed control


# ──────────────────────────────────────────────────────────────────────────────
# Negation Vector Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_negation_direction(
    text_embeds: torch.Tensor,
    question_types: List[str],
) -> torch.Tensor:
    """
    Fit a logistic regression probe on text embeddings to find the negation
    hyperplane normal vector w_neg.

    Positive label (1): question_type in ['negative']   (negation captions)
    Negative label (0): question_type in ['positive']   (affirmative captions)

    Returns:
        w_neg: unit-norm direction vector (D,) in the original embedding space.
    """
    q_types = np.array(question_types)

    # Build per-caption binary labels from MCQ question types:
    # For each MCQ item, we use the question_type to determine if the CORRECT
    # caption is a negation-type caption. We use ALL text embedding slots.
    # For simplicity: treat the correct-answer embeddings with type 'negative'
    # as the positive class, and 'positive' type as the negative class.
    # text_embeds: (N, K, D) -> we use caption slot 0 (correct answer) direction
    N, K, D = text_embeds.shape

    # Heuristic: extract features from the first caption slot for each sample
    # and label by question_type. 'negative' -> class 1, 'positive' -> class 0
    pos_mask = q_types == "positive"
    neg_mask = q_types == "negative"

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        raise ValueError(
            "Need both 'positive' and 'negative' question_type samples to "
            "fit the negation probe. Check the MCQ dataset."
        )

    # Use caption slot 0 (correct answer text embedding)
    X_pos = text_embeds[pos_mask, 0, :].numpy()   # (n_pos, D)
    X_neg = text_embeds[neg_mask, 0, :].numpy()   # (n_neg, D)

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([
        np.zeros(len(X_pos), dtype=int),
        np.ones(len(X_neg),  dtype=int),
    ])

    print(f"\nFitting negation probe:  n_positive={len(X_pos)}, n_negative={len(X_neg)}")

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf.fit(X, y)
    acc_train = clf.score(X, y)
    print(f"  Probe train accuracy: {acc_train*100:.2f}%")

    # Extract and normalise the hyperplane normal
    w_neg_raw = clf.coef_[0]                   # (D,)
    w_neg     = w_neg_raw / (np.linalg.norm(w_neg_raw) + 1e-12)
    w_neg_t   = torch.tensor(w_neg, dtype=torch.float32)

    return w_neg_t


# ──────────────────────────────────────────────────────────────────────────────
# Ablation
# ──────────────────────────────────────────────────────────────────────────────

def ablate_negation_direction(
    text_embeds: torch.Tensor,
    w_neg: torch.Tensor,
) -> torch.Tensor:
    """
    Project out the negation direction from all text embeddings.

    t_ablated = t - (t . w_neg) * w_neg

    Args:
        text_embeds: (N, K, D) tensor of text embeddings.
        w_neg:       (D,) unit-norm negation direction vector.
    Returns:
        ablated:     (N, K, D) tensor with negation direction removed.
    """
    w = w_neg.to(text_embeds.device)  # (D,)

    # Compute projection: (N, K, D) @ (D,) -> (N, K)
    proj_scalar = torch.einsum("nkd,d->nk", text_embeds, w)  # (N, K)

    # Subtract projected component: broadcast (N, K) * (D,) -> (N, K, D)
    ablated = text_embeds - proj_scalar.unsqueeze(-1) * w.unsqueeze(0).unsqueeze(0)

    return ablated


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_scorers_on_embeds(
    img_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    targets: torch.Tensor,
    question_types: List[str],
    scorer_configs: List[Tuple[str, str, int]],   # (display_name, model_type, rank)
    device: str = "cuda",
    n_splits: int = 5,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
) -> Dict[str, Dict]:
    """Run 5-Fold CV for each scorer on provided embeddings."""
    feature_dim = img_embeds.shape[1]
    N = len(targets)
    targets_np = targets.numpy()

    unique_qtypes, qtype_indices = np.unique(question_types, return_inverse=True)
    if len(unique_qtypes) < 2:
        qtype_indices = targets_np

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_results = {}

    for display_name, model_type, rank in scorer_configs:
        print(f"\n  Evaluating: {display_name}")
        oof_preds = np.zeros(N, dtype=int)

        for fold_i, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(N), qtype_indices)
        ):
            train_imgs  = img_embeds[train_idx]
            train_texts = text_embeds[train_idx]
            train_y     = targets[train_idx]
            val_imgs    = img_embeds[val_idx]
            val_texts   = text_embeds[val_idx]
            val_y       = targets[val_idx]

            train_ds = TensorDataset(train_imgs, train_texts, train_y)
            val_ds   = TensorDataset(val_imgs,   val_texts,   val_y)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

            scorer = build_scorer(model_type, feature_dim, rank=rank)
            _, fold_preds = train_and_eval_fold(
                scorer, train_loader, val_loader,
                device=device, epochs=epochs, lr=lr,
            )
            oof_preds[val_idx] = fold_preds

        metrics = compute_mcq_accuracy_breakdown(oof_preds, targets_np, question_types)
        all_results[display_name] = metrics
        print(
            f"    Total: {metrics['total_accuracy']:.2f}% | "
            f"Pos: {metrics['positive_accuracy']:.2f}% | "
            f"Neg: {metrics['negative_accuracy']:.2f}%"
        )

    return all_results


def evaluate_intervention_ablation(
    img_embeds: torch.Tensor,
    text_embeds_orig: torch.Tensor,
    text_embeds_ablated: torch.Tensor,
    targets: torch.Tensor,
    question_types: List[str],
    scorer_configs: List[Tuple[str, str, int]],
    device: str = "cuda",
    n_splits: int = 5,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Mode A (Intervention Ablation):
    Train Scorer on ORIGINAL text embeddings, but evaluate on ABLATED text embeddings.
    Measures direct feature dependency of the pre-trained scorer on w_neg.
    """
    feature_dim = img_embeds.shape[1]
    N = len(targets)
    targets_np = targets.numpy()

    unique_qtypes, qtype_indices = np.unique(question_types, return_inverse=True)
    if len(unique_qtypes) < 2:
        qtype_indices = targets_np

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_results = {}

    for display_name, model_type, rank in scorer_configs:
        print(f"\n  Evaluating Intervention Mode A: {display_name}")
        oof_preds = np.zeros(N, dtype=int)

        for fold_i, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(N), qtype_indices)
        ):
            # Train on ORIGINAL text embeddings
            train_imgs  = img_embeds[train_idx]
            train_texts = text_embeds_orig[train_idx]
            train_y     = targets[train_idx]

            # Validate on ABLATED text embeddings (Intervention)
            val_imgs    = img_embeds[val_idx]
            val_texts   = text_embeds_ablated[val_idx]
            val_y       = targets[val_idx]

            train_ds = TensorDataset(train_imgs, train_texts, train_y)
            val_ds   = TensorDataset(val_imgs,   val_texts,   val_y)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

            scorer = build_scorer(model_type, feature_dim, rank=rank)
            trained_scorer, fold_preds = train_and_eval_fold(
                scorer, train_loader, val_loader,
                device=device, epochs=epochs, lr=lr,
            )
            oof_preds[val_idx] = fold_preds

        metrics = compute_mcq_accuracy_breakdown(oof_preds, targets_np, question_types)
        all_results[display_name] = metrics
        print(
            f"    Intervention Total: {metrics['total_accuracy']:.2f}% | "
            f"Pos: {metrics['positive_accuracy']:.2f}% | "
            f"Neg: {metrics['negative_accuracy']:.2f}%"
        )

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_ablation_comparison(
    original_results: Dict[str, Dict],
    ablated_results:  Dict[str, Dict],
    output_dir: str,
):
    """Side-by-side grouped bar chart: original vs negation-ablated text embeddings."""
    models = list(original_results.keys())

    orig_total  = [original_results[m]["total_accuracy"]    for m in models]
    orig_neg    = [original_results[m]["negative_accuracy"] for m in models]
    abl_total   = [ablated_results[m]["total_accuracy"]     for m in models]
    abl_neg     = [ablated_results[m]["negative_accuracy"]  for m in models]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - 1.5*width, orig_total,  width, label="Original – Total",    color="#1f77b4", alpha=0.9)
    bars2 = ax.bar(x - 0.5*width, orig_neg,    width, label="Original – Neg",      color="#d62728", alpha=0.9)
    bars3 = ax.bar(x + 0.5*width, abl_total,   width, label="Ablated  – Total",    color="#1f77b4", alpha=0.35, hatch="//")
    bars4 = ax.bar(x + 1.5*width, abl_neg,     width, label="Ablated  – Neg",      color="#d62728", alpha=0.35, hatch="//")

    for bar in [*bars1, *bars2, *bars3, *bars4]:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Out-of-Fold Accuracy (%)", fontsize=11)
    ax.set_title(
        "Concept Vector Subtraction: Effect of Removing Negation Direction $w_{\\mathrm{neg}}$\n"
        "from Text Embeddings (Solid = Original, Hatched = Ablated)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "concept_ablation_results.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved ablation comparison plot: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Concept Vector Subtraction (Negation Ablation) Experiment on NegBench MCQ."
    )
    parser.add_argument("--model",       type=str, default="ViT-B-32", help="OpenCLIP architecture")
    parser.add_argument("--pretrained",  type=str, default="openai",   help="Pretrained weights tag")
    parser.add_argument("--coco-mcq",   type=str, required=True,      help="Path to MCQ CSV file")
    parser.add_argument("--image-root", type=str, default="",         help="Root directory for images")
    parser.add_argument("--output-dir", type=str,
                        default="logs/evaluation/scoring_head_experiments",
                        help="Output directory for results")
    parser.add_argument("--best-rank",  type=int,  default=32,   help="Rank k to use for Bi-Encoder scorers")
    parser.add_argument("--n-splits",   type=int,  default=5,    help="Number of CV folds")
    parser.add_argument("--epochs",     type=int,  default=15,   help="Training epochs per fold")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int,  default=64,   help="Batch size")
    parser.add_argument("--seed",       type=int,  default=42,   help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Best rank k = {args.best_rank}")

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

    # ── Step 1: Extract negation direction w_neg ────────────────────────────
    w_neg = extract_negation_direction(text_embeds, question_types)
    print(f"  w_neg norm check: {w_neg.norm().item():.4f}  (should be ≈1.0)")

    # ── Step 2: Ablate text embeddings ─────────────────────────────────────
    print("\nAblating negation direction from text embeddings...")
    text_embeds_ablated = ablate_negation_direction(text_embeds, w_neg)
    print(f"  Original shape: {text_embeds.shape} | Ablated shape: {text_embeds_ablated.shape}")

    # ── Step 3: Define scorer configs ───────────────────────────────────────
    k = args.best_rank
    scorer_configs = [
        ("Cosine",             "cosine",             k),
        ("Full Bilinear",      "bilinear",            k),
        ("Deep MLP",           "deep_mlp",            k),
        (f"LR Bilinear (k={k})", "low_rank_bilinear",  k),
        (f"NL Bi-Enc (k={k})",  "nonlinear_biencoder", k),
    ]

    # ── Step 4: Evaluate on ORIGINAL embeddings ─────────────────────────────
    print("\n" + "="*70)
    print("EVALUATING: Original Text Embeddings")
    print("="*70)
    original_results = evaluate_scorers_on_embeds(
        img_embeds, text_embeds, targets, question_types,
        scorer_configs,
        device=device, n_splits=args.n_splits,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, seed=args.seed,
    )

    # ── Step 5: Evaluate Mode A (Intervention: Train Original -> Test Ablated)
    print("\n" + "="*70)
    print("EVALUATING Mode A: Intervention Ablation (Train Original -> Test Ablated)")
    print("="*70)
    intervention_results = evaluate_intervention_ablation(
        img_embeds, text_embeds, text_embeds_ablated, targets, question_types,
        scorer_configs,
        device=device, n_splits=args.n_splits,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, seed=args.seed,
    )

    # ── Step 6: Evaluate Mode B (Retrained: Train Ablated -> Test Ablated) ──
    print("\n" + "="*70)
    print("EVALUATING Mode B: Retrained Ablation (Train Ablated -> Test Ablated)")
    print("="*70)
    retrained_results = evaluate_scorers_on_embeds(
        img_embeds, text_embeds_ablated, targets, question_types,
        scorer_configs,
        device=device, n_splits=args.n_splits,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, seed=args.seed,
    )

    # ── Step 7: Print Summary Comparison Table ──────────────────────────────
    print("\n" + "="*110)
    print("CONCEPT ABLATION COMPREHENSIVE SUMMARY (Original vs. Intervention Mode A vs. Retrained Mode B)")
    print("="*110)
    print(f"{'Scorer':25s} | {'Original':9s} | {'Mode A (Interv)':15s} | {'Mode B (Retrain)':16s} | {'Interv Δ':9s} | {'Retrain Δ':9s}")
    print("-" * 110)

    summary_rows = []
    for name in [cfg[0] for cfg in scorer_configs]:
        orig = original_results[name]
        interv = intervention_results[name]
        retrain = retrained_results[name]

        d_interv = interv["total_accuracy"] - orig["total_accuracy"]
        d_retrain = retrain["total_accuracy"] - orig["total_accuracy"]

        print(
            f"{name:25s} | {orig['total_accuracy']:8.2f}% | {interv['total_accuracy']:14.2f}% | "
            f"{retrain['total_accuracy']:15.2f}% | {d_interv:+8.2f}% | {d_retrain:+8.2f}%"
        )
        summary_rows.append({
            "Scorer": name,
            "Original_Total": orig["total_accuracy"],
            "Intervention_ModeA_Total": interv["total_accuracy"],
            "Retrained_ModeB_Total": retrain["total_accuracy"],
            "Intervention_Delta": d_interv,
            "Retrained_Delta": d_retrain,
            "Original_Negative": orig["negative_accuracy"],
            "Intervention_Negative": interv["negative_accuracy"],
            "Retrained_Negative": retrain["negative_accuracy"],
        })
    print("="*110)

    # ── Step 8: Save results ─────────────────────────────────────────────────
    out_json = os.path.join(args.output_dir, "concept_ablation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "w_neg_extraction": {
                "method": "LogisticRegression on positive vs negative question_type captions",
                "w_neg_norm": float(w_neg.norm().item()),
            },
            "original": original_results,
            "intervention_mode_a": intervention_results,
            "retrained_mode_b": retrained_results,
            "summary": summary_rows,
        }, f, indent=2)
    print(f"\n✅ Saved ablation JSON: {out_json}")

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.output_dir, "concept_ablation_summary.csv"), index=False
    )

    plot_ablation_comparison(original_results, retrained_results, args.output_dir)
    print(f"\n✅ Concept ablation experiment complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
