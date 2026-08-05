"""
Zero-Shot Cross-Dataset Transfer Evaluation Script for Low-Rank Scorers.

Research Goal: Evaluate pre-trained Low-Rank / Non-Linear Bi-Encoder scoring heads
on external Out-of-Distribution (OOD) benchmarks (e.g., SugarCrepe, Winoground,
BEAF Counterfactual, or Medical/Video CSVs) in a strict ZERO-SHOT (Eval-Only) manner.

Proves that the learned Low-Rank Negation Subspace acts as a Universal Negation Adapter
without in-domain fine-tuning/overfitting on the target benchmark.

Usage:
    python -m benchmarks.src.evaluation.eval_zero_shot_transfer \
        --model ViT-B-32 --pretrained openai \
        --scorer-ckpt logs/evaluation/scoring_head_experiments/checkpoints/low_rank_scorer.pt \
        --target-mcq beaf_counterfactual_6col.csv \
        --output-dir logs/evaluation/zero_shot_transfer
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
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import open_clip

from src.evaluation.scoring_heads import (
    BaseScorer,
    CosineScorer,
    LowRankBilinearScorer,
    NonLinearBiEncoderScorer,
    BilinearScorer,
    DeepMLPScorer,
    build_scorer,
)
from src.evaluation.eval_scoring_heads import (
    extract_mcq_embeddings,
    compute_mcq_accuracy_breakdown,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_zero_shot_scorer(
    scorer: BaseScorer,
    img_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    targets: torch.Tensor,
    question_types: List[str],
    device: str = "cuda",
    batch_size: int = 64
) -> Dict[str, Any]:
    """Run pure Zero-Shot inference on frozen target dataset without any fine-tuning."""
    scorer = scorer.to(device)
    scorer.eval()

    ds = TensorDataset(img_embeds, text_embeds, targets)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds_list = []
    with torch.no_grad():
        for imgs, texts, _ in loader:
            imgs, texts = imgs.to(device), texts.to(device)
            scores = scorer(imgs, texts)
            preds = torch.argmax(scores, dim=1).cpu().numpy()
            preds_list.append(preds)

    all_preds = np.concatenate(preds_list)
    targets_np = targets.numpy()

    metrics = compute_mcq_accuracy_breakdown(all_preds, targets_np, question_types)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Transfer Evaluation of Pre-trained Scoring Heads on OOD Benchmarks")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model architecture")
    parser.add_argument("--pretrained", type=str, default="openai", help="Pretrained weights checkpoint tag")
    parser.add_argument("--scorer-ckpt", type=str, default=None, help="Path to pre-trained scorer checkpoint (.pt)")
    parser.add_argument("--target-mcq", type=str, required=True, help="Path to target OOD MCQ/Benchmark CSV file")
    parser.add_argument("--image-root", type=str, default="", help="Root directory containing images")
    parser.add_argument("--output-dir", type=str, default="logs/evaluation/zero_shot_transfer", help="Output directory")
    parser.add_argument("--rank", type=int, default=32, help="Rank k for Low-Rank models")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Zero-Shot Transfer Evaluation on device: {device}")

    # Load CLIP
    print(f"\nLoading OpenCLIP {args.model} ({args.pretrained})...")
    model, _, preprocess_val = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    # 1. Extract Embeddings for Target Dataset
    img_embeds, text_embeds, targets, question_types, _ = extract_mcq_embeddings(
        model, tokenizer, preprocess_val, args.target_mcq, device=device, batch_size=args.batch_size, image_root=args.image_root
    )

    feature_dim = img_embeds.shape[1]

    # 2. Evaluate Baseline Cosine (Zero-shot Default)
    cosine_scorer = CosineScorer(feature_dim)
    cosine_metrics = evaluate_zero_shot_scorer(cosine_scorer, img_embeds, text_embeds, targets, question_types, device=device)
    print(f"\nBaseline Cosine (Default Zero-Shot): Total Acc = {cosine_metrics['total_accuracy']:.2f}% | Pos Acc = {cosine_metrics['positive_accuracy']:.2f}% | Neg Acc = {cosine_metrics['negative_accuracy']:.2f}%")

    # 3. Load & Evaluate Pre-trained Scorer Checkpoint
    transfer_results = {"Baseline Cosine": cosine_metrics}

    if args.scorer_ckpt and os.path.exists(args.scorer_ckpt):
        print(f"\nLoading Pre-trained Scorer Checkpoint: {args.scorer_ckpt}")
        ckpt = torch.load(args.scorer_ckpt, map_location=device)

        model_name = ckpt.get("model_name", "Low-Rank Bilinear")
        rank = ckpt.get("rank", args.rank)

        scorer = build_scorer(model_name, feature_dim, rank=rank)
        scorer.load_state_dict(ckpt["state_dict"])

        ckpt_metrics = evaluate_zero_shot_scorer(scorer, img_embeds, text_embeds, targets, question_types, device=device)
        print(f"Pre-trained Scorer ({model_name}, k={rank}) Zero-Shot Transfer Total Acc = {ckpt_metrics['total_accuracy']:.2f}% | Pos Acc = {ckpt_metrics['positive_accuracy']:.2f}% | Neg Acc = {ckpt_metrics['negative_accuracy']:.2f}%")

        transfer_results[f"Pretrained_{model_name}"] = ckpt_metrics
    else:
        print("\n⚠️ No checkpoint provided or file not found. Sweeping default architecture zero-shot baselines...")

    # 4. Save JSON and Summary CSV
    out_json = os.path.join(args.output_dir, "zero_shot_transfer_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(transfer_results, f, indent=2)

    rows = []
    for name, mdata in transfer_results.items():
        rows.append({
            "Model": name,
            "Target_Dataset": os.path.basename(args.target_mcq),
            "Total_Accuracy_Pct": mdata["total_accuracy"],
            "Positive_Accuracy_Pct": mdata["positive_accuracy"],
            "Negative_Accuracy_Pct": mdata["negative_accuracy"],
            "Hybrid_Accuracy_Pct": mdata["hybrid_accuracy"],
            "Total_Samples": mdata["total_samples"]
        })
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "zero_shot_transfer_summary.csv"), index=False)

    print(f"\n✅ Zero-shot transfer evaluation complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
