"""
BEAF A/B Swap Compositional Negation Diagnostic.

Three experiments using beaf_counterfactual_ab_swap.csv:

1. Text-only Sanity Probe:
   Layer-wise linear probe on T_XY ("A but no B") vs T_YX ("B but no A").
   Expected: ~50% across all layers (proves no lexical shortcut).

2. Unary vs Compound Negation:
   Compare atomic / unary-negation / compound-negation accuracy to diagnose
   whether CLIP fails at compositional binding or just object detection.

3. ΔS Margin Analysis:
   Histogram of score differences S(I, T_correct) - S(I, T_incorrect)
   for cosine similarity, diagnosing how close the decision boundary is.

Usage:
    python -m benchmarks.src.evaluation.eval_ab_swap_negation_diagnostic \
        --csv_path benchmarks/data/images/beaf_counterfactual_ab_swap.csv \
        --output_dir logs/evaluation/ab_swap_diagnostic \
        --model ViT-B-32 --pretrained openai
"""

import os
import argparse
import json
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import open_clip

from benchmarks.src.analysis.config import coerce_bool_column

from benchmarks.src.evaluation.eval_layerwise_linear_probe import (
    extract_layerwise_feature_dict,
    evaluate_layerwise_linear_probe,
)


# ============================================================
# Noun formatting (consistent with generate_beaf_ab_swap_dataset.py)
# ============================================================
NO_ARTICLE_NOUNS = {"broccoli", "scissors", "skis"}


def format_a_noun(obj: str) -> str:
    """Return 'a/an obj' or bare noun for uncountable/plural nouns."""
    obj_clean = obj.lower().strip()
    if obj_clean in NO_ARTICLE_NOUNS:
        return obj_clean
    elif obj_clean[0] in "aeiou":
        return f"an {obj_clean}"
    else:
        return f"a {obj_clean}"


# ============================================================
# Experiment 1: Text-only Layer-wise Sanity Probe
# ============================================================
def run_text_sanity_probe(
    model,
    tokenizer,
    pos_texts: List[str],
    neg_texts: List[str],
    device: str,
    output_dir: str,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """
    Layer-wise linear probe on T_XY vs T_YX.

    Both caption types contain identical negation tokens and object names,
    differing only in which object occupies the positive vs negated slot.
    A well-controlled dataset should yield ~50% (chance) across all layers.
    """
    print("=" * 60)
    print("Experiment 1: Text-only Layer-wise Sanity Probe")
    print(f"  Positive captions (T_XY): {len(pos_texts)}")
    print(f"  Negative captions (T_YX): {len(neg_texts)}")
    print("  Expected: ~50% (Chance Level) across all layers")
    print("=" * 60)

    print("\nExtracting T_XY features (layer-wise)...")
    pos_features = extract_layerwise_feature_dict(
        model, tokenizer, pos_texts, device, "eot", batch_size
    )
    print("Extracting T_YX features (layer-wise)...")
    neg_features = extract_layerwise_feature_dict(
        model, tokenizer, neg_texts, device, "eot", batch_size
    )

    df_res = evaluate_layerwise_linear_probe(pos_features, neg_features, n_splits=5)

    # Save CSV
    csv_path = os.path.join(output_dir, "exp1_layerwise_sanity_probe.csv")
    df_res.to_csv(csv_path, index=False)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, 6))
    layers = df_res["layer"].values
    accs = df_res["mean_accuracy_pct"].values
    stds = df_res["std_accuracy_pct"].values
    x = list(range(len(layers)))

    ax.plot(x, accs, "o-", color="#e74c3c", lw=2.5, ms=7, label="5-Fold CV Accuracy (%)")
    ax.fill_between(x, accs - stds, accs + stds, color="#e74c3c", alpha=0.15)
    ax.axhline(y=50.0, color="#2ecc71", ls="--", lw=2, alpha=0.8, label="Chance Level (50%)")

    ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=12)
    ax.set_xlabel("Transformer Layer / Pipeline Step", fontsize=12)
    ax.set_title(
        "Exp 1: Text Sanity Probe — A/B Swap (Expect ~50%)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(40, 60)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "exp1_layerwise_sanity_probe.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Summary
    max_acc = float(df_res["mean_accuracy_pct"].max())
    mean_acc = float(df_res["mean_accuracy_pct"].mean())
    verdict = "PASS (no shortcut)" if max_acc < 55.0 else "WARNING (possible shortcut)"

    print(f"\n  Verdict : {verdict}")
    print(f"  Max layer accuracy  : {max_acc:.2f}%")
    print(f"  Mean layer accuracy : {mean_acc:.2f}%")
    print(f"  Saved: {csv_path}")
    print(f"  Saved: {plot_path}")

    return {
        "verdict": verdict,
        "max_accuracy_pct": max_acc,
        "mean_accuracy_pct": mean_acc,
        "per_layer": df_res.to_dict(orient="records"),
    }


# ============================================================
# Experiment 2: Unary vs Compound Negation
# ============================================================
def _encode_texts_batched(model, tokenizer, texts, device, batch_size=256):
    """Encode a list of texts in batches, return L2-normalized embeddings on CPU."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            embs = model.encode_text(tokens, normalize=True).float().cpu()
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)


def _encode_images_one_by_one(model, preprocess, paths, device, embed_dim):
    """Encode images from paths. Returns (embeddings, valid_mask)."""
    from PIL import Image

    embs = []
    valid_mask = []
    for p in paths:
        resolved = p
        if not os.path.exists(resolved):
            resolved = os.path.join("data/coco/images/val2014", os.path.basename(p))
        if os.path.exists(resolved):
            try:
                img = preprocess(Image.open(resolved).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.encode_image(img, normalize=True).float().cpu()
                embs.append(emb)
                valid_mask.append(True)
                continue
            except Exception:
                pass
        embs.append(torch.zeros(1, embed_dim))
        valid_mask.append(False)
    return torch.cat(embs, dim=0), valid_mask


def run_unary_vs_compound(
    model,
    tokenizer,
    preprocess,
    df: pd.DataFrame,
    device: str,
    output_dir: str,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """
    Compare three levels of negation understanding:

    1. Atomic presence:   S(I, "a A") vs S(I, "a B")
    2. Unary negation:    S(I, "a A") vs S(I, "no A")  (present object)
                          S(I, "no B") vs S(I, "a B")  (absent object)
    3. Compound negation: S(I, "A but no B") vs S(I, "B but no A")
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Unary vs Compound Negation Analysis")
    print("=" * 60)

    # Separate True/False rows (pairs are consecutive)
    df_true = df[df["object_in_image"] == True].reset_index(drop=True)
    df_false = df[df["object_in_image"] == False].reset_index(drop=True)
    n_pairs = len(df_true)

    # Parse A, B from object_name
    objects_a = df_true["object_name"].apply(lambda x: str(x).split(",")[0].strip()).tolist()
    objects_b = df_true["object_name"].apply(lambda x: str(x).split(",")[1].strip()).tolist()

    # Build 6 query types
    q_atomic_a = [format_a_noun(a) for a in objects_a]
    q_atomic_b = [format_a_noun(b) for b in objects_b]
    q_neg_a = [f"no {a}" for a in objects_a]
    q_neg_b = [f"no {b}" for b in objects_b]
    q_compound_pos = df_true["positive_caption"].tolist()
    q_compound_neg = df_true["negative_caption"].tolist()

    print(f"  Total pairs: {n_pairs}")
    print(f"  Example A/B: {objects_a[0]} / {objects_b[0]}")
    print(f"    Atomic:   '{q_atomic_a[0]}' vs '{q_atomic_b[0]}'")
    print(f"    Negated:  '{q_neg_a[0]}' vs '{q_neg_b[0]}'")
    print(f"    Compound: '{q_compound_pos[0]}' vs '{q_compound_neg[0]}'")

    # Encode all text queries
    print("\n  Encoding 6 query types...")
    t_atomic_a = _encode_texts_batched(model, tokenizer, q_atomic_a, device, batch_size)
    t_atomic_b = _encode_texts_batched(model, tokenizer, q_atomic_b, device, batch_size)
    t_neg_a = _encode_texts_batched(model, tokenizer, q_neg_a, device, batch_size)
    t_neg_b = _encode_texts_batched(model, tokenizer, q_neg_b, device, batch_size)
    t_compound_pos = _encode_texts_batched(model, tokenizer, q_compound_pos, device, batch_size)
    t_compound_neg = _encode_texts_batched(model, tokenizer, q_compound_neg, device, batch_size)

    # ── Text-side statistics (always available) ──
    cos_atomic = F.cosine_similarity(t_atomic_a, t_atomic_b, dim=-1)
    cos_neg = F.cosine_similarity(t_neg_a, t_neg_b, dim=-1)
    cos_compound = F.cosine_similarity(t_compound_pos, t_compound_neg, dim=-1)

    text_stats = {
        "cos_atomic_a_vs_b_mean": float(cos_atomic.mean()),
        "cos_atomic_a_vs_b_std": float(cos_atomic.std()),
        "cos_negated_a_vs_b_mean": float(cos_neg.mean()),
        "cos_negated_a_vs_b_std": float(cos_neg.std()),
        "cos_compound_pos_vs_neg_mean": float(cos_compound.mean()),
        "cos_compound_pos_vs_neg_std": float(cos_compound.std()),
    }

    print("\n  Text-side cosine similarities (query separability):")
    print(f"    cos('a A', 'a B'):              {text_stats['cos_atomic_a_vs_b_mean']:.4f} ± {text_stats['cos_atomic_a_vs_b_std']:.4f}")
    print(f"    cos('no A', 'no B'):            {text_stats['cos_negated_a_vs_b_mean']:.4f} ± {text_stats['cos_negated_a_vs_b_std']:.4f}")
    print(f"    cos(compound_pos, compound_neg):{text_stats['cos_compound_pos_vs_neg_mean']:.4f} ± {text_stats['cos_compound_pos_vs_neg_std']:.4f}")

    # ── Image-based scoring (if images are available) ──
    has_images = _check_images_exist(df_true["image_path"].tolist())

    image_results = None
    if has_images:
        print("\n  Images found. Computing image-text scores...")
        embed_dim = t_atomic_a.shape[-1]
        image_results = _compute_image_scores(
            model, preprocess, device, embed_dim,
            df_true["image_path"].tolist(),
            df_false["image_path"].tolist(),
            t_atomic_a, t_atomic_b,
            t_neg_a, t_neg_b,
            t_compound_pos, t_compound_neg,
            objects_a, objects_b,
            output_dir,
        )
    else:
        print("\n  [Note] Images not found locally. Text-side statistics only.")

    # ── Plot ──
    _plot_exp2(text_stats, image_results, output_dir)

    # ── Save summary ──
    summary = {"text_statistics": text_stats, "n_pairs": n_pairs}
    if image_results is not None:
        summary["image_accuracy"] = image_results["accuracy"]
        summary["image_margin"] = image_results["margin"]
        summary["valid_image_pairs"] = image_results["valid_pairs"]

    json_path = os.path.join(output_dir, "exp2_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {json_path}")

    return summary


def _check_images_exist(paths: List[str], sample_n: int = 5) -> bool:
    """Check whether any of the first sample_n image paths are accessible."""
    for p in paths[:sample_n]:
        if os.path.exists(p):
            return True
        alt = os.path.join("data/coco/images/val2014", os.path.basename(p))
        if os.path.exists(alt):
            return True
    return False


def _compute_image_scores(
    model, preprocess, device, embed_dim,
    img_paths_xy, img_paths_yx,
    t_atomic_a, t_atomic_b, t_neg_a, t_neg_b,
    t_compound_pos, t_compound_neg,
    objects_a, objects_b,
    output_dir,
) -> Dict[str, Any]:
    """Compute 6 cosine scores per image pair and derive accuracy metrics."""
    print("    Encoding I_XY images (A present, B absent)...")
    img_xy, mask_xy = _encode_images_one_by_one(model, preprocess, img_paths_xy, device, embed_dim)
    print("    Encoding I_YX images (B present, A absent)...")
    img_yx, mask_yx = _encode_images_one_by_one(model, preprocess, img_paths_yx, device, embed_dim)

    # Filter to valid pairs where both images exist
    valid_idx = [i for i in range(len(mask_xy)) if mask_xy[i] and mask_yx[i]]
    if len(valid_idx) == 0:
        print("    No valid image pairs found!")
        return None

    idx = torch.tensor(valid_idx, dtype=torch.long)
    i_xy = img_xy[idx]
    i_yx = img_yx[idx]
    ta = t_atomic_a[idx]
    tb = t_atomic_b[idx]
    tna = t_neg_a[idx]
    tnb = t_neg_b[idx]
    tcp = t_compound_pos[idx]
    tcn = t_compound_neg[idx]

    print(f"    Valid image pairs: {len(valid_idx)}")

    cos = lambda u, v: (u * v).sum(dim=-1)

    # ── 6 scores for I_XY (A present, B absent) ──
    s_xy = {
        "s_a": cos(i_xy, ta), "s_b": cos(i_xy, tb),
        "s_na": cos(i_xy, tna), "s_nb": cos(i_xy, tnb),
        "s_cp": cos(i_xy, tcp), "s_cn": cos(i_xy, tcn),
    }
    # ── 6 scores for I_YX (B present, A absent) ──
    s_yx = {
        "s_a": cos(i_yx, ta), "s_b": cos(i_yx, tb),
        "s_na": cos(i_yx, tna), "s_nb": cos(i_yx, tnb),
        "s_cp": cos(i_yx, tcp), "s_cn": cos(i_yx, tcn),
    }

    # ── Accuracy Metrics ──

    # Atomic: does model detect which object is present?
    atomic_xy = (s_xy["s_a"] > s_xy["s_b"]).float()     # I_XY has A → s_A should win
    atomic_yx = (s_yx["s_b"] > s_yx["s_a"]).float()     # I_YX has B → s_B should win
    atomic_acc = float(torch.cat([atomic_xy, atomic_yx]).mean() * 100)

    # Unary negation (present object): "a A" should beat "no A" when A IS present
    unary_pres_xy = (s_xy["s_a"] > s_xy["s_na"]).float()
    unary_pres_yx = (s_yx["s_b"] > s_yx["s_nb"]).float()
    unary_present_acc = float(torch.cat([unary_pres_xy, unary_pres_yx]).mean() * 100)

    # Unary negation (absent object): "no B" should beat "a B" when B is NOT present
    unary_abs_xy = (s_xy["s_nb"] > s_xy["s_b"]).float()
    unary_abs_yx = (s_yx["s_na"] > s_yx["s_a"]).float()
    unary_absent_acc = float(torch.cat([unary_abs_xy, unary_abs_yx]).mean() * 100)

    # Compound: "A but no B" should beat "B but no A" for I_XY
    compound_xy = (s_xy["s_cp"] > s_xy["s_cn"]).float()
    compound_yx = (s_yx["s_cn"] > s_yx["s_cp"]).float()
    compound_acc = float(torch.cat([compound_xy, compound_yx]).mean() * 100)

    # ── Margins ──
    margin_xy = s_xy["s_cp"] - s_xy["s_cn"]
    margin_yx = s_yx["s_cn"] - s_yx["s_cp"]
    margins = torch.cat([margin_xy, margin_yx])

    print("\n    ┌─────────────────────────────────────────────┐")
    print(f"    │ Accuracy Results ({len(valid_idx)} valid pairs)       │")
    print("    ├─────────────────────────────────────────────┤")
    print(f"    │ Atomic Object Presence     : {atomic_acc:6.2f}%        │")
    print(f"    │ Unary Negation (present)   : {unary_present_acc:6.2f}%        │")
    print(f"    │ Unary Negation (absent)    : {unary_absent_acc:6.2f}%        │")
    print(f"    │ Compound Negation (A∧¬B)   : {compound_acc:6.2f}%        │")
    print("    └─────────────────────────────────────────────┘")
    print(f"    Mean ΔS margin (compound): {float(margins.mean()):.4f} ± {float(margins.std()):.4f}")

    # ── Save 6-score CSV ──
    rows = []
    for j, vi in enumerate(valid_idx):
        for label, scores, img_type in [("XY", s_xy, "A_present"), ("YX", s_yx, "B_present")]:
            rows.append({
                "pair_idx": vi,
                "object_a": objects_a[vi],
                "object_b": objects_b[vi],
                "image_type": img_type,
                "s_atomic_a": float(scores["s_a"][j]),
                "s_atomic_b": float(scores["s_b"][j]),
                "s_neg_a": float(scores["s_na"][j]),
                "s_neg_b": float(scores["s_nb"][j]),
                "s_compound_pos": float(scores["s_cp"][j]),
                "s_compound_neg": float(scores["s_cn"][j]),
            })
    df_scores = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "exp2_6score_matrix.csv")
    df_scores.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}")

    return {
        "valid_pairs": len(valid_idx),
        "accuracy": {
            "atomic_presence_pct": atomic_acc,
            "unary_negation_present_pct": unary_present_acc,
            "unary_negation_absent_pct": unary_absent_acc,
            "compound_negation_pct": compound_acc,
        },
        "margin": {
            "compound_mean": float(margins.mean()),
            "compound_std": float(margins.std()),
            "compound_median": float(margins.median()),
        },
        "margins_tensor": margins,
    }


def _plot_exp2(text_stats, image_results, output_dir):
    """Generate Experiment 2 visualizations."""

    if image_results is not None:
        # ── Bar chart: 4 accuracy levels ──
        fig, ax = plt.subplots(figsize=(8, 5))
        acc = image_results["accuracy"]
        names = [
            "Atomic\n(Object ID)",
            "Unary Neg.\n(Present Obj)",
            "Unary Neg.\n(Absent Obj)",
            "Compound\n(A ∧ ¬B)",
        ]
        vals = [
            acc["atomic_presence_pct"],
            acc["unary_negation_present_pct"],
            acc["unary_negation_absent_pct"],
            acc["compound_negation_pct"],
        ]
        colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]
        bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=1.5, width=0.6)

        ax.axhline(y=50.0, color="gray", ls="--", lw=1.5, alpha=0.7, label="Chance (50%)")

        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(
            "Exp 2: Atomic → Unary → Compound Negation Accuracy",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_ylim(0, max(vals) + 10)
        ax.legend(fontsize=11)
        ax.grid(axis="y", ls="--", alpha=0.4)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "exp2_unary_vs_compound.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {plot_path}")

    else:
        # Text-only mode: plot query separability
        fig, ax = plt.subplots(figsize=(7, 5))
        names = ["Atomic\n(A vs B)", "Negated\n(¬A vs ¬B)", "Compound\n(A∧¬B vs B∧¬A)"]
        means = [
            text_stats["cos_atomic_a_vs_b_mean"],
            text_stats["cos_negated_a_vs_b_mean"],
            text_stats["cos_compound_pos_vs_neg_mean"],
        ]
        stds = [
            text_stats["cos_atomic_a_vs_b_std"],
            text_stats["cos_negated_a_vs_b_std"],
            text_stats["cos_compound_pos_vs_neg_std"],
        ]
        colors = ["#3498db", "#e67e22", "#e74c3c"]

        bars = ax.bar(names, means, yerr=stds, color=colors, edgecolor="white",
                      linewidth=1.5, width=0.5, capsize=5)
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{m:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

        ax.set_ylabel("Cosine Similarity (text ↔ text)", fontsize=12)
        ax.set_title(
            "Exp 2: Text Query Separability (no images)",
            fontsize=13, fontweight="bold",
        )
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", ls="--", alpha=0.4)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "exp2_text_separability.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {plot_path}")


# ============================================================
# Experiment 3: ΔS Margin Analysis
# ============================================================
def run_margin_analysis(margins: torch.Tensor, output_dir: str) -> Dict[str, Any]:
    """Plot ΔS histogram and compute summary statistics."""
    print("\n" + "=" * 60)
    print("Experiment 3: ΔS Margin Analysis (Compound Negation)")
    print("=" * 60)

    m = margins.numpy()
    mean_m = float(np.mean(m))
    std_m = float(np.std(m))
    median_m = float(np.median(m))
    pct_positive = float(np.mean(m > 0) * 100)

    print(f"  Mean ΔS   : {mean_m:.4f}")
    print(f"  Std  ΔS   : {std_m:.4f}")
    print(f"  Median ΔS : {median_m:.4f}")
    print(f"  P(ΔS > 0) : {pct_positive:.2f}% (= Accuracy)")

    # ── Histogram ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(m, bins=80, color="#3498db", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(x=0, color="gray", ls="-", lw=1.5, alpha=0.8, label="Decision Boundary (ΔS=0)")
    ax.axvline(x=mean_m, color="#e74c3c", ls="--", lw=2, alpha=0.9,
               label=f"Mean ΔS = {mean_m:.4f}")

    # Shade correct/incorrect regions
    ax.axvspan(0, max(m) * 1.1, alpha=0.05, color="green")
    ax.axvspan(min(m) * 1.1, 0, alpha=0.05, color="red")
    ax.text(max(m) * 0.6, ax.get_ylim()[1] * 0.85, "Correct", fontsize=12,
            color="green", fontweight="bold", alpha=0.7)
    ax.text(min(m) * 0.6, ax.get_ylim()[1] * 0.85, "Incorrect", fontsize=12,
            color="red", fontweight="bold", alpha=0.7)

    ax.set_xlabel("ΔS = S(I, T_correct) − S(I, T_incorrect)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Exp 3: Cosine Score Margin Distribution (Acc={pct_positive:.1f}%)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "exp3_margin_histogram.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_path}")

    summary = {
        "mean_delta_s": mean_m,
        "std_delta_s": std_m,
        "median_delta_s": median_m,
        "accuracy_pct": pct_positive,
        "n_samples": len(m),
    }

    json_path = os.path.join(output_dir, "exp3_margin_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    return summary


# ============================================================
# Main Orchestration
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="BEAF A/B Swap Compositional Negation Diagnostic"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="benchmarks/data/images/beaf_counterfactual_ab_swap.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/evaluation/ab_swap_diagnostic",
    )
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("╔══════════════════════════════════════════════════════╗")
    print("║  BEAF A/B Swap Compositional Negation Diagnostic    ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Model  : {args.model} ({args.pretrained})")
    print(f"  Device : {device}")
    print(f"  CSV    : {args.csv_path}")
    print(f"  Output : {args.output_dir}\n")

    # Load data
    df = pd.read_csv(args.csv_path)
    # Normalize object_in_image to bool
    coerce_bool_column(df, "object_in_image")

    n_pairs = len(df) // 2
    print(f"  Loaded {len(df)} rows ({n_pairs} counterfactual pairs)\n")

    # Load model
    print(f"Loading CLIP model '{args.model}' ({args.pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    # ── Deduplicate texts for Experiment 1 ──
    # Use one caption per source_template (unique A/B pair)
    df_unique = df[df["object_in_image"] == True].drop_duplicates(
        subset=["positive_caption", "negative_caption"]
    ).reset_index(drop=True)
    pos_texts = df_unique["positive_caption"].tolist()
    neg_texts = df_unique["negative_caption"].tolist()

    full_report = {}

    # Experiment 1
    exp1_results = run_text_sanity_probe(
        model, tokenizer, pos_texts, neg_texts,
        device, args.output_dir, args.batch_size,
    )
    full_report["experiment_1_text_sanity_probe"] = exp1_results

    # Experiment 2
    exp2_results = run_unary_vs_compound(
        model, tokenizer, preprocess, df,
        device, args.output_dir, args.batch_size,
    )
    full_report["experiment_2_unary_vs_compound"] = exp2_results

    # Experiment 3 (only if images were available and produced margins)
    if (
        exp2_results is not None
        and "image_accuracy" in exp2_results
    ):
        # Reload margins from exp2 — they were computed during exp2
        # Re-derive from the 6-score CSV
        df_scores = pd.read_csv(os.path.join(args.output_dir, "exp2_6score_matrix.csv"))
        margins_correct = []
        for _, row in df_scores.iterrows():
            if row["image_type"] == "A_present":
                margins_correct.append(row["s_compound_pos"] - row["s_compound_neg"])
            else:  # B_present
                margins_correct.append(row["s_compound_neg"] - row["s_compound_pos"])
        margins_tensor = torch.tensor(margins_correct, dtype=torch.float32)

        exp3_results = run_margin_analysis(margins_tensor, args.output_dir)
        full_report["experiment_3_margin_analysis"] = exp3_results
    else:
        print("\n  [Exp 3 skipped] No image data available for margin analysis.")

    # Save full report
    # Remove non-serializable items
    report_clean = json.loads(json.dumps(full_report, default=str))
    report_path = os.path.join(args.output_dir, "full_diagnostic_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_clean, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("All experiments complete!")
    print(f"Full report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
