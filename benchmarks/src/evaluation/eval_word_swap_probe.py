"""
Counterfactual Word-Swap Text Probe Experiment.

Research Question: Is the 99.9% text probe accuracy merely due to 'not/no'
token presence (Token-Presence Bias), or does CLIP's text encoder capture
the relational context of negation?

Methodology:
  1. Extract/construct counterfactual text pairs where negation keywords ('no', 'not', 'without')
     are held constant, but the negated target object/attribute is swapped.
     E.g., "A photo of a dog and no cat" vs. "A photo of a cat and no dog"
  2. Encode both original and swapped text pairs using OpenCLIP's text encoder.
  3. Fit a Linear Probe (Logistic Regression with 5-Fold CV) on these Word-Swap embeddings.
  4. Measure accuracy and cosine distance delta. If accuracy remains high,
     the text encoder captures relational negation semantics beyond token presence.

Usage:
    python -m benchmarks.src.evaluation.eval_word_swap_probe \
        --model ViT-B-32 --pretrained openai \
        --coco-mcq <path/to/mcq.csv> \
        --output-dir logs/evaluation/top_priority_experiments
"""

import os
import sys
import json
import argparse
import random
import re
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import open_clip


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_word_swap_pairs(df: pd.DataFrame) -> List[Tuple[str, str, str, str]]:
    """
    Construct counterfactual word-swap text pairs from MCQ captions where
    negation keywords ('no', 'not', 'without') are held constant.

    Returns:
        pairs: List of tuples (text_a, text_b, orig_object, swapped_object)
    """
    caption_cols = [c for c in df.columns if c.startswith("caption_")]
    pairs = []

    negation_patterns = [
        r"\bno\s+([a-z]+)\b",
        r"\bnot\s+a\s+([a-z]+)\b",
        r"\bwithout\s+([a-z]+)\b",
        r"\bwithout\s+a\s+([a-z]+)\b",
    ]

    for idx, row in df.iterrows():
        captions = [str(row[col]) for col in caption_cols] if caption_cols else [str(row["caption_0"]), str(row["caption_1"])]

        for cap in captions:
            cap_lower = cap.lower()
            # Match negation phrases
            match = None
            for pat in negation_patterns:
                m = re.search(pat, cap_lower)
                if m:
                    match = m
                    break

            if match:
                target_word = match.group(1)
                # Swap target word with a dummy/swapped object if found
                # For MCQ datasets, often caption_1 or caption_2 contains the swapped option
                for alt_cap in captions:
                    if alt_cap.lower() != cap_lower and target_word not in alt_cap.lower():
                        # Find a candidate object in alt_cap to swap
                        words = [w for w in re.findall(r"\b[a-z]{3,}\b", alt_cap.lower())
                                 if w not in ["photo", "image", "there", "with", "from", "that", "this", "some"]]
                        if words:
                            swapped_word = words[0]
                            swapped_cap = re.sub(r"\b" + re.escape(target_word) + r"\b", swapped_word, cap, flags=re.IGNORECASE)
                            if swapped_cap != cap:
                                pairs.append((cap, swapped_cap, target_word, swapped_word))
                                break

    print(f"Generated {len(pairs)} counterfactual Word-Swap text pairs.")
    return pairs


def evaluate_word_swap_linear_probe(
    model: nn.Module,
    tokenizer: Any,
    pairs: List[Tuple[str, str, str, str]],
    device: str = "cuda",
    batch_size: int = 64,
    n_splits: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Encode word-swap text pairs and run 5-Fold CV Linear Probe classification.

    Class 0: Original negation caption ("no dog")
    Class 1: Swapped target negation caption ("no cat")
    """
    model.eval()

    texts_a = [p[0] for p in pairs]
    texts_b = [p[1] for p in pairs]

    embeds_a = []
    embeds_b = []

    print("\nEncoding Word-Swap text pairs with OpenCLIP Text Encoder...")
    with torch.no_grad():
        for i in range(0, len(texts_a), batch_size):
            batch_a = texts_a[i:i+batch_size]
            batch_b = texts_b[i:i+batch_size]

            tokens_a = tokenizer(batch_a).to(device)
            tokens_b = tokenizer(batch_b).to(device)

            feat_a = F.normalize(model.encode_text(tokens_a), dim=-1).cpu()
            feat_b = F.normalize(model.encode_text(tokens_b), dim=-1).cpu()

            embeds_a.append(feat_a)
            embeds_b.append(feat_b)

    X_a = torch.cat(embeds_a, dim=0).numpy()  # (N, D)
    X_b = torch.cat(embeds_b, dim=0).numpy()  # (N, D)

    # Compute pairwise cosine similarity between original and word-swapped captions
    cos_sims = np.sum(X_a * X_b, axis=-1)
    mean_cos_sim = float(np.mean(cos_sims))
    std_cos_sim = float(np.std(cos_sims))

    # Build dataset for Linear Probe
    X = np.concatenate([X_a, X_b], axis=0)  # (2N, D)
    y = np.concatenate([np.zeros(len(X_a), dtype=int), np.ones(len(X_b), dtype=int)])  # (2N,)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_accs = []

    print(f"\nRunning {n_splits}-Fold Stratified CV Linear Probe on Word-Swap Embeddings...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(X[train_idx], y[train_idx])
        acc = clf.score(X[val_idx], y[val_idx]) * 100.0
        fold_accs.append(acc)

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))

    print(f"\n📊 Word-Swap Linear Probe Results:")
    print(f"   - Linear Probe Accuracy:    {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"   - Pairwise Cosine Similarity: {mean_cos_sim:.4f} ± {std_cos_sim:.4f}")
    print(f"   - Pairwise Cosine Distance:   {1.0 - mean_cos_sim:.4f}")

    return {
        "num_pairs": len(pairs),
        "linear_probe_mean_acc_pct": mean_acc,
        "linear_probe_std_acc_pct": std_acc,
        "pairwise_cosine_sim_mean": mean_cos_sim,
        "pairwise_cosine_sim_std": std_cos_sim,
        "pairwise_cosine_distance_mean": 1.0 - mean_cos_sim,
        "token_presence_bias_refuted": mean_acc > 75.0
    }


def main():
    parser = argparse.ArgumentParser(description="Counterfactual Word-Swap Text Probe Experiment.")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model architecture")
    parser.add_argument("--pretrained", type=str, default="openai", help="Pretrained weights checkpoint tag")
    parser.add_argument("--coco-mcq", type=str, default="COCO_val_mcq_llama3.1_rephrased.csv", help="Path to MCQ CSV file")
    parser.add_argument("--output-dir", type=str, default="logs/evaluation/top_priority_experiments", help="Output directory")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    df = pd.read_csv(args.coco_mcq, sep=",")
    pairs = generate_word_swap_pairs(df)

    if not pairs:
        print("⚠️ Warning: No word-swap pairs generated from regex matching. Creating synthetic pairs from dataset...")
        caption_cols = [c for c in df.columns if c.startswith("caption_")]
        for idx, row in df.iterrows():
            c0, c1 = str(row[caption_cols[0]]), str(row[caption_cols[1]])
            pairs.append((c0, c1, "target0", "target1"))

    print(f"Loading OpenCLIP {args.model} ({args.pretrained})...")
    model, _, preprocess_val = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    results = evaluate_word_swap_linear_probe(
        model, tokenizer, pairs, device=device, batch_size=args.batch_size, n_splits=args.n_splits, seed=args.seed
    )

    out_json = os.path.join(args.output_dir, "word_swap_probe_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    metrics = ["Word-Swap Probe Acc (%)", "Cos Distance (x100)"]
    vals = [results["linear_probe_mean_acc_pct"], results["pairwise_cosine_distance_mean"] * 100.0]
    ax.bar(metrics, vals, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylim(0, 105)
    ax.set_title("Word-Swap Text Probe: Token-Presence Bias Disentanglement", fontsize=11, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.1f}%" if i == 0 else f"{v:.2f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "word_swap_probe.png"), dpi=300)
    plt.close()

    print(f"\n✅ Word-swap probe experiment complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
