"""
Text Ablation Shortcut Diagnostic for Trained Scoring Heads.

Symmetric counterpart to eval_vision_ablation_shortcut.py.
Now we keep the IMAGE fixed and ablate the TEXT embeddings.

Three ablation conditions:

  Condition A (original):
    Normal text embeddings. Baseline reference.

  Condition B (shuffle_across):
    For each option slot k, shuffle that slot's text embeddings across
    all N samples. e.g. question 1's caption_0 embedding gets question 7's
    caption_0 embedding.
    -> Destroys WITHIN-QUESTION semantic contrast while preserving the
       distributional statistics of each slot.
    -> If MLP still scores high: it relies on the absolute embedding style
       of caption_0 (the "true caption distribution") rather than
       relative comparison of options within a question.
    -> If MLP drops to ~25%: it was doing genuine semantic matching.

  Condition C (shuffle_options):
    Within each question, randomly permute the K=4 option text embeddings
    AND update the target label to follow the correct caption.
    -> Directly tests position/index bias.
    -> If accuracy drops significantly: the scorer used option ORDER (index)
       as a shortcut (possible if architecture has positional asymmetry).
    -> For our [img, text_k] independent scorers, expected: NO drop.
       Any drop would be very surprising and indicate hidden positional bias.

  Condition D (gaussian):
    Replace ALL text embeddings with random Gaussian vectors (matched norm).
    -> Destroys all semantic AND distributional structure.
    -> Should collapse to ~25% (pure random) for all scorers.
    -> Ground truth baseline for "what does random look like?".

Key insight:
  - Vision ablation showed MLP ignores image (Shuffle ~= Original).
  - Text ablation will show WHICH PART of text MLP relies on:
    (a) shuffle_across ~= original  → relies on ABSOLUTE STYLE of caption_0
    (b) shuffle_options ~= original → no positional bias (expected)
    (c) gaussian = ~25%             → relies on text embedding structure

Usage (on remote server):
    export PYTHONPATH="$(pwd)/benchmarks:$(pwd):$PYTHONPATH"
    python -m benchmarks.src.evaluation.eval_text_ablation_shortcut \\
        --model ViT-B-32 --pretrained openai \\
        --coco-mcq COCO_val_mcq_llama3.1_rephrased.csv \\
        --output-dir logs/evaluation/text_ablation_shortcut
"""

import os
import json
import argparse
import random
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import open_clip

from src.evaluation.scoring_heads import (
    BaseScorer,
    CosineScorer,
    build_scorer,
)
from src.evaluation.eval_scoring_heads import extract_mcq_embeddings


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_extended_breakdown(
    oof_preds: np.ndarray,
    targets: np.ndarray,
    question_types: List[str],
) -> Dict[str, Any]:
    """Accuracy breakdown by question type."""
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
        else:
            acc = 0.0
        result[f"{qt}_accuracy"] = acc
        result[f"{qt}_count"] = n_qt
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Text Ablation DataLoader Factory
# ──────────────────────────────────────────────────────────────────────────────

def create_text_ablated_loader(
    img_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    targets: torch.Tensor,
    indices: np.ndarray,
    ablation_mode: str,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple:
    """
    Create a DataLoader with TEXT embeddings ablated.

    text_embeds shape: (N, K, D)  where K = number of options (e.g. 4)

    Returns (DataLoader, ablated_targets_for_indices)
    - ablated_targets is usually identical to original targets,
      EXCEPT for shuffle_options where the correct index changes.
    """
    N_sub = len(indices)
    rng = np.random.default_rng(seed)

    sel_imgs = img_embeds[indices].clone()          # (N_sub, D)
    sel_texts = text_embeds[indices].clone()        # (N_sub, K, D)
    sel_y = targets[indices].clone()               # (N_sub,)

    if ablation_mode == "original":
        # No modification
        pass

    elif ablation_mode == "shuffle_across":
        # For each option slot k, shuffle that slot's embeddings
        # across the N_sub samples independently.
        # This breaks the BETWEEN-QUESTION semantic alignment:
        # question i's slot-k text now gets question j's slot-k text.
        # Within-question structure (4 options belonging together) is PRESERVED
        # as each slot is shuffled by its own permutation.
        K = sel_texts.shape[1]
        for k in range(K):
            perm = rng.permutation(N_sub)
            sel_texts[:, k, :] = sel_texts[perm, k, :]
        # targets unchanged: labels still reference the correct option by index,
        # but now the text at each position is from a different question.
        # Semantic alignment destroyed; distributional style of each slot preserved.

    elif ablation_mode == "shuffle_options":
        # Within each question, randomly permute the K options.
        # The TARGET LABEL is updated to track where caption_0 moved.
        # This directly tests positional / index bias.
        K = sel_texts.shape[1]
        new_targets = sel_y.clone()
        for i in range(N_sub):
            perm = rng.permutation(K)
            sel_texts[i] = sel_texts[i][perm]
            # correct answer was at position sel_y[i]; find where it went
            orig_correct = sel_y[i].item()
            new_pos = int(np.where(perm == orig_correct)[0][0])
            new_targets[i] = new_pos
        sel_y = new_targets

    elif ablation_mode == "gaussian":
        # Replace all text embeddings with random Gaussian vectors.
        # Match the mean L2-norm of the original text embeddings.
        K = sel_texts.shape[1]
        orig_norms = sel_texts.norm(dim=-1, keepdim=True)      # (N_sub, K, 1)
        mean_norm = orig_norms.mean().item()
        rand_texts = torch.randn_like(sel_texts)
        sel_texts = F.normalize(rand_texts, dim=-1) * mean_norm
        # targets unchanged

    else:
        raise ValueError(f"Unknown text ablation_mode: {ablation_mode}")

    ds = TensorDataset(sel_imgs, sel_texts, sel_y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader, sel_y.numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Training Utilities
# ──────────────────────────────────────────────────────────────────────────────

def train_fold_return_scorer(
    scorer: BaseScorer,
    train_loader: DataLoader,
    device: str = "cuda",
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> BaseScorer:
    """Train scorer on original train_loader, return trained model."""
    scorer = scorer.to(device)

    if isinstance(scorer, CosineScorer) or len(list(scorer.parameters())) == 0:
        return scorer

    optimizer = torch.optim.AdamW(scorer.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        scorer.train()
        for imgs, texts, tgts in train_loader:
            imgs, texts, tgts = imgs.to(device), texts.to(device), tgts.to(device)
            optimizer.zero_grad()
            loss = criterion(scorer(imgs, texts), tgts)
            loss.backward()
            optimizer.step()

    return scorer


def eval_scorer_on_loader(
    scorer: BaseScorer,
    val_loader: DataLoader,
    device: str = "cuda",
) -> np.ndarray:
    """Evaluate and return predictions."""
    scorer.eval()
    preds = []
    with torch.no_grad():
        for imgs, texts, _ in val_loader:
            imgs, texts = imgs.to(device), texts.to(device)
            p = torch.argmax(scorer(imgs, texts), dim=1).cpu().numpy()
            preds.append(p)
    return np.concatenate(preds)


# ──────────────────────────────────────────────────────────────────────────────
# Main Experiment
# ──────────────────────────────────────────────────────────────────────────────

def run_text_ablation_diagnostic(
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
    Train each scorer on ORIGINAL (img, text), then evaluate under 4 text
    ablation conditions. Compare accuracy breakdown per condition × question type.
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

    ablation_modes = ["original", "shuffle_across", "shuffle_options", "gaussian"]

    unique_qtypes, qtype_indices = np.unique(question_types, return_inverse=True)
    if len(unique_qtypes) < 2:
        qtype_indices = targets_np

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_results = {}

    for model_name, model_type, expr_level in scoring_models:
        print(f"\n{'='*80}")
        print(f"  Scorer: {model_name:22s} | Expressiveness: {expr_level}")
        print(f"{'='*80}")

        # Collect OOF predictions per ablation condition.
        # For shuffle_options, targets differ per fold/condition, so we store
        # both the predictions AND the adjusted targets.
        condition_preds   = {m: np.zeros(N, dtype=int) for m in ablation_modes}
        condition_targets = {m: targets_np.copy()       for m in ablation_modes}

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(N), qtype_indices)):
            print(f"  Fold {fold+1}/{n_splits} ...", end=" ")

            # ── Train on ORIGINAL embeddings ──
            train_ds = TensorDataset(
                img_embeds[train_idx], text_embeds[train_idx], targets[train_idx]
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            scorer = build_scorer(model_type, feature_dim)
            scorer = train_fold_return_scorer(
                scorer, train_loader, device=device, epochs=epochs, lr=lr
            )

            # ── Evaluate under each text ablation ──
            for mode in ablation_modes:
                val_loader, adj_targets = create_text_ablated_loader(
                    img_embeds, text_embeds, targets,
                    val_idx, ablation_mode=mode,
                    batch_size=batch_size, seed=seed + fold * 10,
                )
                fold_preds = eval_scorer_on_loader(scorer, val_loader, device=device)
                condition_preds[mode][val_idx]   = fold_preds
                condition_targets[mode][val_idx] = adj_targets  # updated for shuffle_options

            print("done")

        # ── Compute accuracy breakdown per condition ──
        model_results = {}
        for mode in ablation_modes:
            metrics = compute_extended_breakdown(
                condition_preds[mode],
                condition_targets[mode],
                question_types,
            )
            model_results[mode] = metrics

            total = metrics["total_accuracy"]
            orig  = model_results["original"]["total_accuracy"] if mode != "original" else total
            delta = total - model_results["original"]["total_accuracy"]

            print(f"    [{mode:16s}] Total: {total:6.2f}%", end="")
            for qt in ["positive", "negative", "hybrid"]:
                k = f"{qt}_accuracy"
                if k in metrics:
                    print(f" | {qt[:3].capitalize()}: {metrics[k]:.1f}%", end="")
            if mode != "original":
                print(f" | delta: {delta:+.2f}%")
            else:
                print()

        # delta from original
        orig_total = model_results["original"]["total_accuracy"]
        for mode in ["shuffle_across", "shuffle_options", "gaussian"]:
            model_results[mode]["delta_from_original"] = \
                model_results[mode]["total_accuracy"] - orig_total

        all_results[model_name] = model_results

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_text_ablation_results(
    results: Dict[str, Dict[str, Dict]],
    output_dir: str,
):
    models = list(results.keys())
    conditions = ["original", "shuffle_across", "shuffle_options", "gaussian"]
    condition_labels = [
        "Original",
        "Shuffle Across\n(cross-question)",
        "Shuffle Options\n(within-question)",
        "Gaussian\n(random noise)",
    ]
    colors  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    hatches = ["",         "\\\\",    "//",       "xx"]

    x     = np.arange(len(models))
    width = 0.2

    # ── Plot 1: Total Accuracy ──
    fig, ax = plt.subplots(figsize=(16, 7))

    for i, (cond, label, color, hatch) in enumerate(
        zip(conditions, condition_labels, colors, hatches)
    ):
        accs = [results[m][cond]["total_accuracy"] for m in models]
        bars = ax.bar(
            x + (i - 1.5) * width, accs, width,
            label=label.replace("\n", " "), color=color, alpha=0.85,
            hatch=hatch, edgecolor="white"
        )
        for bar, acc in zip(bars, accs):
            ax.annotate(
                f"{acc:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

    ax.axhline(25, color="red", ls="--", lw=1.5, alpha=0.7, label="25% (random baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=11)
    ax.set_ylabel("Out-of-Fold Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Text Ablation Shortcut Diagnostic\n"
        "Symmetric counterpart to Vision Ablation — IMAGE is fixed, TEXT is ablated",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.set_ylim(0, 108)

    plt.tight_layout()
    out1 = os.path.join(output_dir, "text_ablation_total_accuracy.png")
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out1}")

    # ── Plot 2: Per Question Type (Positive / Negative / Hybrid) ──
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=False)
    fig.suptitle(
        "Text Ablation by Question Type: Positive / Negative / Hybrid",
        fontsize=14, fontweight="bold"
    )

    for ax, qt in zip(axes, ["positive", "negative", "hybrid"]):
        key = f"{qt}_accuracy"
        for i, (cond, label, color, hatch) in enumerate(
            zip(conditions, condition_labels, colors, hatches)
        ):
            accs = [results[m][cond].get(key, 0.0) for m in models]
            bars = ax.bar(
                x + (i - 1.5) * width, accs, width,
                label=label.replace("\n", " "), color=color, alpha=0.85,
                hatch=hatch, edgecolor="white"
            )
            for bar, acc in zip(bars, accs):
                ax.annotate(
                    f"{acc:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6
                )
        ax.axhline(25, color="red", ls="--", lw=1.5, alpha=0.6, label="25% random")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{qt.capitalize()} Questions", fontsize=12, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.grid(True, axis="y", ls="--", alpha=0.4)
        ax.set_ylim(0, 112)

    axes[0].legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    out2 = os.path.join(output_dir, "text_ablation_per_question_type.png")
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out2}")

    # ── Plot 3: Delta Heatmap ──
    ablated_conds = ["shuffle_across", "shuffle_options", "gaussian"]
    ablated_labels = ["Shuffle Across", "Shuffle Options", "Gaussian"]

    delta_data = []
    for m in models:
        row = [
            results[m][c]["total_accuracy"] - results[m]["original"]["total_accuracy"]
            for c in ablated_conds
        ]
        delta_data.append(row)

    delta_arr = np.array(delta_data)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(delta_arr, cmap="RdYlGn", aspect="auto", vmin=-70, vmax=5)

    ax.set_xticks(range(len(ablated_conds)))
    ax.set_xticklabels(ablated_labels, fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)

    for i in range(len(models)):
        for j in range(len(ablated_conds)):
            val = delta_arr[i, j]
            color = "white" if abs(val) > 30 else "black"
            ax.text(j, i, f"{val:+.1f}%",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    ax.set_title(
        "Text Ablation: Accuracy Drop (delta %)\n"
        "Green approx 0 -> Shortcut (text style)  |  Red < 0 -> Genuine semantic use",
        fontsize=12, fontweight="bold"
    )
    plt.colorbar(im, ax=ax, label="delta Accuracy (%)")
    plt.tight_layout()
    out3 = os.path.join(output_dir, "text_ablation_delta_heatmap.png")
    plt.savefig(out3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out3}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary Table
# ──────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: Dict[str, Dict[str, Dict]]):
    modes = ["original", "shuffle_across", "shuffle_options", "gaussian"]
    print("\n" + "=" * 130)
    print("  TEXT ABLATION SHORTCUT DIAGNOSTIC — COMPREHENSIVE SUMMARY")
    print("=" * 130)
    header = (
        f"{'Scorer':22s} | {'Condition':18s} | {'Total':7s} | "
        f"{'Pos':7s} | {'Neg':7s} | {'Hyb':7s} | {'delta':8s} | {'Interpretation':20s}"
    )
    print(header)
    print("-" * 130)

    for model_name, model_results in results.items():
        orig_total = model_results["original"]["total_accuracy"]
        for mode in modes:
            m = model_results[mode]
            total = m["total_accuracy"]
            pos   = m.get("positive_accuracy", 0.0)
            neg   = m.get("negative_accuracy", 0.0)
            hyb   = m.get("hybrid_accuracy", 0.0)
            delta = total - orig_total if mode != "original" else 0.0

            if mode == "original":
                interp = "---"
            elif abs(delta) < 3.0:
                interp = "!! STYLE SHORTCUT"
            elif abs(delta) < 15.0:
                interp = "! PARTIAL SEMANTIC"
            else:
                interp = "OK GENUINE USE"

            print(
                f"  {model_name:20s} | {mode:18s} | {total:6.2f}% | "
                f"{pos:6.2f}% | {neg:6.2f}% | {hyb:6.2f}% | "
                f"{delta:+7.2f}% | {interp}"
            )
        print("-" * 130)

    print("=" * 130)
    print("  Interpretation:")
    print("    !! STYLE SHORTCUT  = delta < 3%  -> Model uses caption style, not content")
    print("    !  PARTIAL SEMANTIC = 3-15% drop -> Model partially uses text meaning")
    print("    OK GENUINE USE      = >15% drop  -> Model relies on true semantic content")
    print("=" * 130)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Text Ablation Shortcut Diagnostic for Trained Scoring Heads"
    )
    parser.add_argument("--model",      type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--coco-mcq",  type=str,
                        default="COCO_val_mcq_llama3.1_rephrased.csv",
                        help="Path to MCQ CSV")
    parser.add_argument("--image-root", type=str, default="")
    parser.add_argument("--output-dir", type=str,
                        default="logs/evaluation/text_ablation_shortcut")
    parser.add_argument("--n-splits",   type=int, default=5)
    parser.add_argument("--epochs",     type=int, default=15)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Resolve CSV path
    csv_path = args.coco_mcq
    if not os.path.exists(csv_path):
        for alt in [
            "COCO_val_mcq_llama3.1_rephrased.csv",
            "benchmarks/data/images/COCO_val_mcq_llama3.1_rephrased.csv",
        ]:
            if os.path.exists(alt):
                csv_path = alt
                break
    assert os.path.exists(csv_path), f"MCQ CSV not found: {csv_path}"

    print(f"\nLoading OpenCLIP {args.model} ({args.pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    print("\nExtracting / loading cached embeddings...")
    img_embeds, text_embeds, targets, question_types, _ = extract_mcq_embeddings(
        model, tokenizer, preprocess,
        csv_path, device=device,
        batch_size=args.batch_size, image_root=args.image_root,
    )

    print(f"\nimg_embeds : {tuple(img_embeds.shape)}")
    print(f"text_embeds: {tuple(text_embeds.shape)}")
    print(f"N samples  : {len(targets)}")
    print(f"Q-types    : {dict(zip(*np.unique(question_types, return_counts=True)))}")

    results = run_text_ablation_diagnostic(
        img_embeds, text_embeds, targets, question_types,
        device=device, n_splits=args.n_splits,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, seed=args.seed,
    )

    print_summary_table(results)

    json_path = os.path.join(args.output_dir, "text_ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved JSON: {json_path}")

    rows = []
    for model_name, model_results in results.items():
        orig_total = model_results["original"]["total_accuracy"]
        for cond, metrics in model_results.items():
            rows.append({
                "Scorer":             model_name,
                "Condition":          cond,
                "Total_Accuracy":     metrics["total_accuracy"],
                "Positive_Accuracy":  metrics.get("positive_accuracy", 0.0),
                "Negative_Accuracy":  metrics.get("negative_accuracy", 0.0),
                "Hybrid_Accuracy":    metrics.get("hybrid_accuracy", 0.0),
                "Delta_from_Original": metrics["total_accuracy"] - orig_total,
            })
    csv_out = os.path.join(args.output_dir, "text_ablation_summary.csv")
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f"  Saved CSV: {csv_out}")

    plot_text_ablation_results(results, args.output_dir)
    print(f"\n  Text Ablation Diagnostic complete! -> {args.output_dir}")


if __name__ == "__main__":
    main()
