"""
Rank-k Sweep Experiment for Low-Rank Bi-Encoder Scorers.

Research Question (RQ3): What is the minimal rank k at which negation matching
accuracy saturates? This identifies the dimensionality of the 'negation
interaction subspace' within CLIP's joint embedding space.

Two models are swept:
  - LowRankBilinearScorer:    s(v,t) = (Av).(Bt)            [linear, cacheable]
  - NonLinearBiEncoderScorer: s(v,t) = GELU(Av).GELU(Bt)   [non-linear, cacheable]

Both preserve O(1) offline Bi-Encoder retrieval caching.

Usage:
    python -m benchmarks.src.evaluation.eval_rank_sweep \
        --model ViT-B-32 --pretrained openai \
        --coco-mcq <path/to/mcq.csv> --image-root <img_dir> \
        --output-dir logs/evaluation/scoring_head_experiments \
        --ranks 1 2 4 8 16 32 64 128 256 512
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
from PIL import Image

import open_clip

from src.evaluation.scoring_heads import (
    BaseScorer,
    CosineScorer,
    LowRankBilinearScorer,
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
# Sweep Core
# ──────────────────────────────────────────────────────────────────────────────

def run_rank_sweep(
    img_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    targets: torch.Tensor,
    question_types: List[str],
    ranks: List[int],
    device: str = "cuda",
    n_splits: int = 5,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Run 5-Fold Stratified CV for each rank k for LowRankBilinear and NonLinearBiEncoder.

    Returns:
        results: {
            "LowRankBilinear": { k: {"total_accuracy": ..., "negative_accuracy": ..., ...} },
            "NonLinearBiEncoder": { k: {...} }
        }
    """
    feature_dim = img_embeds.shape[1]
    N = len(targets)
    targets_np = targets.numpy()

    unique_qtypes, qtype_indices = np.unique(question_types, return_inverse=True)
    if len(unique_qtypes) < 2:
        qtype_indices = targets_np

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results: Dict[str, Dict[int, Dict]] = {
        "LowRankBilinear": {},
        "NonLinearBiEncoder": {},
    }

    model_configs = [
        ("LowRankBilinear", "low_rank_bilinear"),
        ("NonLinearBiEncoder", "nonlinear_biencoder"),
    ]

    for model_key, model_name in model_configs:
        print(f"\n{'='*70}")
        print(f"Sweeping Model: {model_key}")
        print(f"{'='*70}")

        for rank in ranks:
            print(f"\n  -- Rank k = {rank:4d} --")
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

                scorer = build_scorer(model_name, feature_dim, rank=rank)
                _, fold_preds = train_and_eval_fold(
                    scorer, train_loader, val_loader,
                    device=device, epochs=epochs, lr=lr,
                )
                oof_preds[val_idx] = fold_preds

            metrics = compute_mcq_accuracy_breakdown(oof_preds, targets_np, question_types)
            results[model_key][rank] = metrics
            print(
                f"    Total: {metrics['total_accuracy']:.2f}% | "
                f"Pos: {metrics['positive_accuracy']:.2f}% | "
                f"Neg: {metrics['negative_accuracy']:.2f}%"
            )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_rank_sweep(
    results: Dict[str, Dict[int, Dict]],
    cosine_baseline: float,
    full_bilinear: float,
    deep_mlp: float,
    output_dir: str,
):
    """Render rank-k vs accuracy sweep curves with baselines annotated."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle(
        "Low-Rank Bi-Encoder: Rank $k$ Sweep\n"
        "(Negation Interaction Subspace Dimension Diagnostic)",
        fontsize=14, fontweight="bold"
    )

    colors = {
        "Total Acc":    "#1f77b4",
        "Neg Acc":      "#d62728",
        "Pos Acc":      "#2ca02c",
    }

    titles = {
        "LowRankBilinear":   "Linear Low-Rank Bilinear  $s(v,t)=(Av)\\cdot(Bt)$",
        "NonLinearBiEncoder": "Non-Linear Bi-Encoder  $s(v,t)=\\mathrm{GELU}(Av)\\cdot\\mathrm{GELU}(Bt)$",
    }

    for ax, (model_key, model_results) in zip(axes, results.items()):
        ranks = sorted(model_results.keys())
        total_accs = [model_results[k]["total_accuracy"]   for k in ranks]
        neg_accs   = [model_results[k]["negative_accuracy"] for k in ranks]
        pos_accs   = [model_results[k]["positive_accuracy"] for k in ranks]

        ax.plot(ranks, total_accs, "o-", color=colors["Total Acc"],  label="Total Acc",    lw=2.5)
        ax.plot(ranks, neg_accs,   "s--", color=colors["Neg Acc"],   label="Neg Acc",     lw=2)
        ax.plot(ranks, pos_accs,   "^:",  color=colors["Pos Acc"],   label="Pos Acc",     lw=2)

        # Baselines
        ax.axhline(cosine_baseline, ls="--", color="gray",   alpha=0.7, lw=1.5, label=f"Cosine ({cosine_baseline:.1f}%)")
        ax.axhline(full_bilinear,   ls="-.", color="purple", alpha=0.7, lw=1.5, label=f"Full Bilinear ({full_bilinear:.1f}%)")
        ax.axhline(deep_mlp,        ls=":",  color="brown",  alpha=0.7, lw=1.5, label=f"Deep MLP ({deep_mlp:.1f}%)")

        ax.set_xscale("log", base=2)
        ax.set_xticks(ranks)
        ax.set_xticklabels([str(k) for k in ranks], rotation=45, ha="right")
        ax.set_xlabel("Rank $k$", fontsize=12)
        ax.set_ylabel("Out-of-Fold Accuracy (%)", fontsize=12)
        ax.set_title(titles[model_key], fontsize=11)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_ylim(0, 105)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "rank_sweep_performance.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved rank sweep plot: {out_path}")


def save_results(results: Dict, output_dir: str):
    """Serialise results dict to JSON (converting int keys to str)."""
    serialisable = {
        model_key: {str(k): v for k, v in rank_results.items()}
        for model_key, rank_results in results.items()
    }
    out_path = os.path.join(output_dir, "rank_sweep_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    print(f"✅ Saved rank sweep results: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rank-k Sweep Experiment for Low-Rank Bi-Encoder Scorers on NegBench MCQ."
    )
    parser.add_argument("--model",       type=str, default="ViT-B-32",  help="OpenCLIP architecture")
    parser.add_argument("--pretrained",  type=str, default="openai",    help="Pretrained weights tag or checkpoint path")
    parser.add_argument("--coco-mcq",   type=str, required=True,       help="Path to MCQ CSV file")
    parser.add_argument("--image-root", type=str, default="",          help="Root directory for images")
    parser.add_argument("--output-dir", type=str,
                        default="logs/evaluation/scoring_head_experiments",
                        help="Output directory for results")
    parser.add_argument("--ranks", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                        help="List of rank values to sweep")
    parser.add_argument("--n-splits",   type=int,   default=5,    help="Number of CV folds")
    parser.add_argument("--epochs",     type=int,   default=15,   help="Training epochs per fold")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int,   default=64,   help="Batch size")
    parser.add_argument("--seed",       type=int,   default=42,   help="Random seed")
    # Known baselines for annotation on the plot
    parser.add_argument("--cosine-baseline",  type=float, default=39.30, help="Known Cosine accuracy (%)")
    parser.add_argument("--full-bilinear",    type=float, default=74.79, help="Known Full Bilinear accuracy (%)")
    parser.add_argument("--deep-mlp",         type=float, default=86.69, help="Known Deep MLP accuracy (%)")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Ranks to sweep: {args.ranks}")

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

    # Run sweep
    results = run_rank_sweep(
        img_embeds, text_embeds, targets, question_types,
        ranks=args.ranks,
        device=device,
        n_splits=args.n_splits,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Save JSON
    save_results(results, args.output_dir)

    # Plot
    plot_rank_sweep(
        results,
        cosine_baseline=args.cosine_baseline,
        full_bilinear=args.full_bilinear,
        deep_mlp=args.deep_mlp,
        output_dir=args.output_dir,
    )

    print(f"\n✅ Rank sweep complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
