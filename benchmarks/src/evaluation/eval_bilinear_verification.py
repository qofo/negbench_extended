"""
Verification Script for Bilinear Scorer vs. Low-Rank (k=512) Bilinear.

This script executes 3 rigorous verification experiments to disentangle
expressiveness vs. optimization landscape differences:

Experiment 1: Full Bilinear Initialization Impact (Identity vs. Random Normal(0.02))
Experiment 2: Epoch Convergence Sweep (Epochs 15, 30, 50, 100)
Experiment 3: Mathematical Equivalence Verification (W = A^T B Weight Transfer)
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
    BilinearScorer,
    LowRankBilinearScorer,
    build_scorer,
)
from src.evaluation.eval_scoring_heads import (
    extract_mcq_embeddings,
    compute_mcq_accuracy_breakdown,
)


from benchmarks.src.analysis.config import set_seed  # noqa: E402 — centralized seed control


class RandomBilinearScorer(BaseScorer):
    """Full Bilinear Scorer with Random Normal(0.02) Initialization instead of Identity."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.W = nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        if img_emb.dim() == 2:
            img_emb = img_emb.unsqueeze(1)
        v_norm = F.normalize(img_emb, dim=-1)
        t_norm = F.normalize(text_emb, dim=-1)
        v_W = torch.matmul(v_norm, self.W)
        scores = torch.sum(v_W * t_norm, dim=-1) + self.bias
        return scores


def train_and_eval_fold(
    scorer: BaseScorer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4
) -> Tuple[BaseScorer, np.ndarray]:
    """Train scoring model and return trained model along with out-of-fold predictions."""
    scorer = scorer.to(device)
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

    scorer.eval()
    oof_preds = []
    with torch.no_grad():
        for imgs, texts, _ in val_loader:
            imgs, texts = imgs.to(device), texts.to(device)
            scores = scorer(imgs, texts)
            preds = torch.argmax(scores, dim=1).cpu().numpy()
            oof_preds.append(preds)

    return scorer, np.concatenate(oof_preds)


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 1 & 2: Epoch & Initialization Convergence Sweep
# ──────────────────────────────────────────────────────────────────────────────

def run_epochs_and_init_sweep(
    img_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    targets: torch.Tensor,
    question_types: List[str],
    epoch_list: List[int],
    device: str = "cuda",
    n_splits: int = 5,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Evaluate Full Bilinear (Identity vs Random) and Low-Rank (k=512) across Epochs."""
    feature_dim = img_embeds.shape[1]
    N = len(targets)
    targets_np = targets.numpy()

    unique_qtypes, qtype_indices = np.unique(question_types, return_inverse=True)
    if len(unique_qtypes) < 2:
        qtype_indices = targets_np

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    models_to_test = [
        ("Full_Bilinear_Identity", lambda: BilinearScorer(feature_dim)),
        ("Full_Bilinear_Random", lambda: RandomBilinearScorer(feature_dim)),
        ("LowRank_Bilinear_k512", lambda: LowRankBilinearScorer(feature_dim, rank=512)),
    ]

    sweep_results = {m[0]: {} for m in models_to_test}

    for model_name, model_fn in models_to_test:
        print(f"\n{'='*70}\nRunning Sweep for: {model_name}\n{'='*70}")
        for ep in epoch_list:
            print(f"  --> Training Epochs = {ep:3d}")
            oof_preds = np.zeros(N, dtype=int)

            for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(N), qtype_indices)):
                train_ds = TensorDataset(img_embeds[train_idx], text_embeds[train_idx], targets[train_idx])
                val_ds = TensorDataset(img_embeds[val_idx], text_embeds[val_idx], targets[val_idx])

                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

                scorer = model_fn()
                _, fold_preds = train_and_eval_fold(scorer, train_loader, val_loader, device=device, epochs=ep, lr=lr)
                oof_preds[val_idx] = fold_preds

            metrics = compute_mcq_accuracy_breakdown(oof_preds, targets_np, question_types)
            sweep_results[model_name][ep] = metrics
            print(f"      Total Acc: {metrics['total_accuracy']:.2f}% | Pos: {metrics['positive_accuracy']:.2f}% | Neg: {metrics['negative_accuracy']:.2f}%")

    return sweep_results


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 3: Mathematical Equivalence & Weight Transfer (W = A^T B)
# ──────────────────────────────────────────────────────────────────────────────

def run_weight_transfer_verification(
    img_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    targets: torch.Tensor,
    device: str = "cuda",
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 64
) -> Dict[str, Any]:
    """Train LowRank (k=512), transfer W_eff = A^T B to Full Bilinear, and check exact predictions."""
    feature_dim = img_embeds.shape[1]
    print(f"\n{'='*70}\nRunning Experiment 3: Weight Transfer (W = A^T B) Equivalence Check\n{'='*70}")

    train_ds = TensorDataset(img_embeds, text_embeds, targets)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    # 1. Train Low-Rank (k=512)
    lr_scorer = LowRankBilinearScorer(feature_dim, rank=512).to(device)
    optimizer = torch.optim.AdamW(lr_scorer.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Training LowRankBilinearScorer (k=512)...")
    for epoch in range(epochs):
        lr_scorer.train()
        for imgs, texts, y in train_loader:
            imgs, texts, y = imgs.to(device), texts.to(device), y.to(device)
            optimizer.zero_grad()
            scores = lr_scorer(imgs, texts)
            loss = criterion(scores, y)
            loss.backward()
            optimizer.step()

    lr_scorer.eval()

    # 2. Extract W_eff = A^T B and bias
    with torch.no_grad():
        A = lr_scorer.proj_v.weight  # (512, 512)
        B = lr_scorer.proj_t.weight  # (512, 512)
        # s(v, t) = (A v) . (B t) = v^T (A^T B) t
        W_eff = torch.matmul(A.T, B) # (512, 512)
        bias_eff = lr_scorer.bias.clone()

    # 3. Construct Full Bilinear Scorer and transfer W_eff
    full_scorer = BilinearScorer(feature_dim).to(device)
    with torch.no_grad():
        full_scorer.W.copy_(W_eff)
        full_scorer.bias.copy_(bias_eff)
    full_scorer.eval()

    # 4. Predict and measure absolute score difference
    lr_scores_list = []
    full_scores_list = []

    with torch.no_grad():
        for imgs, texts, _ in val_loader:
            imgs, texts = imgs.to(device), texts.to(device)
            s_lr = lr_scorer(imgs, texts)
            s_full = full_scorer(imgs, texts)

            lr_scores_list.append(s_lr)
            full_scores_list.append(s_full)

    lr_scores = torch.cat(lr_scores_list, dim=0)
    full_scores = torch.cat(full_scores_list, dim=0)

    max_diff = torch.max(torch.abs(lr_scores - full_scores)).item()
    mean_diff = torch.mean(torch.abs(lr_scores - full_scores)).item()
    lr_preds = torch.argmax(lr_scores, dim=1)
    full_preds = torch.argmax(full_scores, dim=1)
    pred_matches = torch.sum(lr_preds == full_preds).item()
    match_pct = (pred_matches / len(targets)) * 100.0

    print("\n📊 Weight Transfer Verification Summary:")
    print(f"   - Max Absolute Score Difference:  {max_diff:.8e}")
    print(f"   - Mean Absolute Score Difference: {mean_diff:.8e}")
    print(f"   - Prediction Match Percentage:    {match_pct:.2f}% ({pred_matches}/{len(targets)})")

    return {
        "max_absolute_score_difference": max_diff,
        "mean_absolute_score_difference": mean_diff,
        "prediction_match_percentage": match_pct,
        "is_mathematically_identical": max_diff < 1e-4
    }


def plot_epoch_sweep(sweep_results: Dict[str, Dict[int, Dict[str, float]]], output_dir: str):
    """Plot Epoch vs Accuracy for Identity Init, Random Init, and Low-Rank k=512."""
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = {
        "Full_Bilinear_Identity": ("o-", "#d62728", "Full Bilinear (Identity Init)"),
        "Full_Bilinear_Random": ("s--", "#ff7f0e", "Full Bilinear (Random Init)"),
        "LowRank_Bilinear_k512": ("^-", "#2ca02c", "Low-Rank Bilinear (k=512, Random Init)"),
    }

    for model_name, epoch_data in sweep_results.items():
        epochs = sorted(epoch_data.keys())
        accs = [epoch_data[ep]["total_accuracy"] for ep in epochs]
        marker_style, color, label = styles[model_name]
        ax.plot(epochs, accs, marker_style, color=color, label=label, lw=2.5, ms=7)

    ax.set_xlabel("Training Epochs", fontsize=12, fontweight="bold")
    ax.set_ylabel("Out-of-Fold Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Optimization Dynamics: Full Bilinear vs. Low-Rank (k=512)", fontsize=13, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "bilinear_verification_convergence.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved convergence plot to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Verification of Bilinear Scorer Optimization & Parameterization Effects")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model architecture")
    parser.add_argument("--pretrained", type=str, default="openai", help="Pretrained weights checkpoint tag")
    parser.add_argument("--coco-mcq", type=str, default="COCO_val_mcq_llama3.1_rephrased.csv", help="Path to MCQ CSV file")
    parser.add_argument("--image-root", type=str, default="", help="Root directory containing images")
    parser.add_argument("--output-dir", type=str, default="logs/evaluation/scoring_head_experiments", help="Output directory")
    parser.add_argument("--epochs-list", type=int, nargs="+", default=[15, 30, 50, 100], help="Epoch values to sweep")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of Cross-Validation folds")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running verification experiments on device: {device}")

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

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    # 1. Extract Embeddings
    img_embeds, text_embeds, targets, question_types, _ = extract_mcq_embeddings(
        model, tokenizer, preprocess_val, args.coco_mcq, device=device, batch_size=args.batch_size, image_root=args.image_root
    )

    # 2. Run Epoch & Initialization Convergence Sweep (Experiments 1 & 2)
    sweep_results = run_epochs_and_init_sweep(
        img_embeds, text_embeds, targets, question_types,
        epoch_list=args.epochs_list, device=device, n_splits=args.n_splits, lr=args.lr, batch_size=args.batch_size, seed=args.seed
    )

    # 3. Run Weight Transfer Equivalence Verification (Experiment 3)
    transfer_results = run_weight_transfer_verification(
        img_embeds, text_embeds, targets, device=device, epochs=15, lr=args.lr, batch_size=args.batch_size
    )

    # 4. Save Json & Plot
    combined_results = {
        "sweep_results": sweep_results,
        "weight_transfer_verification": transfer_results
    }

    json_path = os.path.join(args.output_dir, "bilinear_verification_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)

    plot_epoch_sweep(sweep_results, args.output_dir)

    print(f"\n✅ All verification experiments completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
