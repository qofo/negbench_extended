"""
4-Condition Inequality Decomposition for CLIP Cosine Negation Matching.

Decomposes the 2×2 correct matching criterion:
    min(S₁₁, S₂₂) > max(S₁₂, S₂₁)    [P(all 4 satisfied) = 1.27%]

into 4 individual independently verifiable conditions:

  (C1) Text-side, present image: m(I_pres) = S₁₁ - S₁₂ > 0
       → "Positive caption preferred when object IS present"

  (C2) Text-side, absent image:  m(I_abs)  = S₂₂ - S₂₁ > 0   (equivalently S₂₁ < S₂₂)
       → "Negative caption preferred when object is ABSENT"

  (C3) Image-side, positive text: n(t_pos) = S₁₁ - S₂₁ > 0
       → "Present image preferred for positive caption"

  (C4) Image-side, negative text: n(t_neg) = S₂₂ - S₁₂ > 0   (equivalently S₁₂ < S₂₂)
       → "Absent image preferred for negative caption"

The joint condition holds iff (C1) ∧ (C2) ∧ (C3) ∧ (C4).

This table localizes: "cosine fails" → "THIS inequality breaks N% of the time".

Output:
  - 4condition_decomposition_table.csv       (per-object breakdown)
  - 4condition_decomposition_summary.json    (macro-averaged Table 1 data)
  - fig_4condition_bars.png                  (visualization)

Usage:
    python -m benchmarks.src.evaluation.eval_4condition_decomposition \\
        --output_dir logs/evaluation/4condition_decomposition \\
        --model ViT-B-32 --pretrained openai
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import open_clip

from benchmarks.src.evaluation.eval_unary_mechanistic_analysis import (
    encode_images_safely,
    encode_texts_safely,
    resolve_image_path,
)


# ============================================================
# Core: 4-Condition Decomposition
# ============================================================
def compute_4condition_decomposition(
    v_pos: torch.Tensor,    # (N, D) — images WITH object present
    v_neg: torch.Tensor,    # (N, D) — images WITHOUT object (counterfactual)
    t_pos: torch.Tensor,    # (N, D) — positive captions ("a photo of a dog")
    t_neg: torch.Tensor,    # (N, D) — negative captions ("a photo without a dog")
) -> Dict[str, Any]:
    """
    Compute the 4 independent inequality conditions and their satisfaction rates.
    
    Score matrix (per sample i):
        S₁₁[i] = cos(v_pos[i], t_pos[i])     S₁₂[i] = cos(v_pos[i], t_neg[i])
        S₂₁[i] = cos(v_neg[i], t_pos[i])     S₂₂[i] = cos(v_neg[i], t_neg[i])
    """
    cos = lambda u, v: F.cosine_similarity(u, v, dim=-1)

    s_11 = cos(v_pos, t_pos).numpy()   # S₁₁: present-image × positive-text (correct)
    s_12 = cos(v_pos, t_neg).numpy()   # S₁₂: present-image × negative-text (wrong)
    s_21 = cos(v_neg, t_pos).numpy()   # S₂₁: absent-image  × positive-text (wrong)
    s_22 = cos(v_neg, t_neg).numpy()   # S₂₂: absent-image  × negative-text (correct)

    n = len(s_11)

    # ── 4 Individual Conditions ──
    # (C1) Text discrimination for present image: m(I_pres) = S₁₁ - S₁₂ > 0
    c1_margins = s_11 - s_12   # positive text should score higher than negative text
    c1_satisfied = c1_margins > 0

    # (C2) Text discrimination for absent image: m(I_abs) = S₂₂ - S₂₁ > 0
    c2_margins = s_22 - s_21   # negative text should score higher than positive text
    c2_satisfied = c2_margins > 0

    # (C3) Image discrimination for positive text: n(t_pos) = S₁₁ - S₂₁ > 0
    c3_margins = s_11 - s_21   # present image should score higher than absent
    c3_satisfied = c3_margins > 0

    # (C4) Image discrimination for negative text: n(t_neg) = S₂₂ - S₁₂ > 0
    c4_margins = s_22 - s_12   # absent image should score higher for negative text
    c4_satisfied = c4_margins > 0

    # Joint condition: ALL 4 must hold
    all_satisfied = c1_satisfied & c2_satisfied & c3_satisfied & c4_satisfied

    # Original 2x2 margin: min(S₁₁, S₂₂) - max(S₁₂, S₂₁)
    correct_min = np.minimum(s_11, s_22)
    wrong_max = np.maximum(s_12, s_21)
    joint_margin = correct_min - wrong_max
    joint_correct = joint_margin > 0

    return {
        # Satisfaction rates (percentage)
        "P_C1_text_pres":  float(np.mean(c1_satisfied) * 100),
        "P_C2_text_abs":   float(np.mean(c2_satisfied) * 100),
        "P_C3_img_pos":    float(np.mean(c3_satisfied) * 100),
        "P_C4_img_neg":    float(np.mean(c4_satisfied) * 100),
        "P_all_4":         float(np.mean(all_satisfied) * 100),
        "P_joint_2x2":     float(np.mean(joint_correct) * 100),

        # Mean margins (signed)
        "margin_C1_mean":  float(np.mean(c1_margins)),
        "margin_C2_mean":  float(np.mean(c2_margins)),
        "margin_C3_mean":  float(np.mean(c3_margins)),
        "margin_C4_mean":  float(np.mean(c4_margins)),
        "margin_joint_mean": float(np.mean(joint_margin)),

        # Margin std
        "margin_C1_std":   float(np.std(c1_margins)),
        "margin_C2_std":   float(np.std(c2_margins)),
        "margin_C3_std":   float(np.std(c3_margins)),
        "margin_C4_std":   float(np.std(c4_margins)),

        # Mean raw scores
        "S11_mean": float(np.mean(s_11)),
        "S12_mean": float(np.mean(s_12)),
        "S21_mean": float(np.mean(s_21)),
        "S22_mean": float(np.mean(s_22)),

        "n_samples": n,
    }


# ============================================================
# Visualization
# ============================================================
def render_4condition_figure(summary: Dict[str, float], output_dir: str):
    """Bar chart showing 4 individual condition satisfaction rates vs joint."""
    labels = [
        "C1: m(I_pres)>0\nPos text preferred\nwhen obj present",
        "C2: m(I_abs)<0\nNeg text preferred\nwhen obj absent",
        "C3: n(t_pos)>0\nPres img preferred\nfor pos text",
        "C4: n(t_neg)<0\nAbs img preferred\nfor neg text",
        "All 4\nsatisfied",
        "Random\nbaseline\n(1/6)",
    ]

    values = [
        summary["P_C1_text_pres"],
        summary["P_C2_text_abs"],
        summary["P_C3_img_pos"],
        summary["P_C4_img_neg"],
        summary["P_all_4"],
        100.0 / 6.0,  # random 2x2 baseline ≈ 16.7%
    ]

    colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6", "#e67e22", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", width=0.6)

    for bar, val in zip(bars, values):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_ylabel("Satisfaction Rate (%)", fontsize=13)
    ax.set_title("4-Condition Decomposition of CLIP Cosine Negation Matching\n"
                 "(Why does joint matching = 1.27%?)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, ha="center")
    ax.set_ylim(0, 110)
    ax.axhline(50, color="gray", ls="--", lw=1, alpha=0.5, label="Chance (50%)")
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig_4condition_bars.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def render_4condition_margins_figure(summary: Dict[str, float], output_dir: str):
    """Bar chart showing mean margins for each condition (signed)."""
    labels = [
        "C1: m(I_pres)\nS₁₁−S₁₂",
        "C2: m(I_abs)\nS₂₂−S₂₁",
        "C3: n(t_pos)\nS₁₁−S₂₁",
        "C4: n(t_neg)\nS₂₂−S₁₂",
        "Joint\nmin−max",
    ]

    means = [
        summary["margin_C1_mean"],
        summary["margin_C2_mean"],
        summary["margin_C3_mean"],
        summary["margin_C4_mean"],
        summary["margin_joint_mean"],
    ]

    stds = [
        summary.get("margin_C1_std", 0),
        summary.get("margin_C2_std", 0),
        summary.get("margin_C3_std", 0),
        summary.get("margin_C4_std", 0),
        0,  # joint std not tracked in summary
    ]

    colors = ["#2ecc71" if m > 0 else "#e74c3c" for m in means]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors, edgecolor="black", width=0.55,
                  yerr=stds, capsize=5, error_kw={"lw": 1.5})

    for bar, val in zip(bars, means):
        yval = bar.get_height()
        offset = 0.001 if val >= 0 else -0.003
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + offset,
                f"{val:+.4f}", ha="center", va="bottom" if val >= 0 else "top",
                fontweight="bold", fontsize=11)

    ax.axhline(0, color="black", lw=1.5)
    ax.set_ylabel("Mean Margin (cosine units)", fontsize=12)
    ax.set_title("Mean Signed Margins per Condition (Macro-averaged over objects)\n"
                 "Positive = correct ordering; Negative = systematically wrong",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis="y", ls="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig_4condition_margins.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def render_score_matrix_heatmap(summary: Dict[str, float], output_dir: str):
    """2×2 mean score matrix heatmap."""
    matrix = np.array([
        [summary["S11_mean"], summary["S12_mean"]],
        [summary["S21_mean"], summary["S22_mean"]],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(matrix, cmap="RdYlGn", vmin=matrix.min() - 0.005, vmax=matrix.max() + 0.005)
    fig.colorbar(cax, ax=ax, label="Mean Cosine Similarity")

    for i in range(2):
        for j in range(2):
            color = "white" if (i == j) else "black"
            label = "✓" if (i == j) else "✗"
            ax.text(j, i, f"{matrix[i, j]:.4f}\n{label}",
                    ha="center", va="center", fontsize=14, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["t_pos\n(positive text)", "t_neg\n(negative text)"], fontsize=11)
    ax.set_yticklabels(["I_pres\n(object present)", "I_abs\n(object absent)"], fontsize=11)
    ax.set_title("Mean 2×2 Score Matrix S_ij = cos(I_i, t_j)\n(Macro-averaged over objects)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig_score_matrix_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# Main Orchestrator
# ============================================================
def run_4condition_decomposition(
    csv_path: str = "benchmarks/data/images/beaf_counterfactual_6col.csv",
    output_dir: str = "logs/evaluation/4condition_decomposition",
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    min_pairs_per_obj: int = 6,
    batch_size: int = 128,
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  4-Condition Inequality Decomposition (Table 1)           ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"  Model       : {model_name} ({pretrained}) | Device: {device}")
    print(f"  Input CSV   : {csv_path}")
    print(f"  Output Dir  : {output_dir}")
    print(f"  Min Pairs   : {min_pairs_per_obj}\n")

    # Load CSV
    df = pd.read_csv(csv_path)
    if "object_in_image" in df.columns:
        if df["object_in_image"].dtype == object:
            df["object_in_image"] = df["object_in_image"].apply(
                lambda x: str(x).strip().lower() == "true"
            )
        else:
            df["object_in_image"] = df["object_in_image"].astype(bool)

    # Single-object targets only
    all_objects = df["object_name"].unique().tolist()
    target_objects = [o for o in all_objects if "," not in str(o)]
    print(f"  Total single-object candidates: {len(target_objects)}")

    # Load CLIP model
    print(f"\n  Loading CLIP model '{model_name}' ({pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    embed_dim = 512

    per_object_records = []
    analyzed_objects = []

    for obj in target_objects:
        df_obj = df[df["object_name"] == obj].reset_index(drop=True)
        df_true = df_obj[df_obj["object_in_image"] == True].reset_index(drop=True)
        df_false = df_obj[df_obj["object_in_image"] == False].reset_index(drop=True)

        n_pairs = min(len(df_true), len(df_false))
        if n_pairs < min_pairs_per_obj:
            continue

        img_paths_pos = df_true["image_path"].tolist()[:n_pairs]
        img_paths_neg = df_false["image_path"].tolist()[:n_pairs]
        t_pos_texts = df_true["positive_caption"].tolist()[:n_pairs]
        t_neg_texts = df_true["negative_caption"].tolist()[:n_pairs]

        # Encode
        v_pos, mask_vp = encode_images_safely(model, preprocess, img_paths_pos, device, embed_dim, batch_size)
        v_neg, mask_vn = encode_images_safely(model, preprocess, img_paths_neg, device, embed_dim, batch_size)

        valid_idx = [i for i in range(n_pairs) if mask_vp[i] and mask_vn[i]]
        if len(valid_idx) < min_pairs_per_obj:
            continue

        v_pos = v_pos[valid_idx]
        v_neg = v_neg[valid_idx]
        t_pos_texts = [t_pos_texts[i] for i in valid_idx]
        t_neg_texts = [t_neg_texts[i] for i in valid_idx]

        t_pos = encode_texts_safely(model, tokenizer, t_pos_texts, device, batch_size)
        t_neg = encode_texts_safely(model, tokenizer, t_neg_texts, device, batch_size)

        # Compute decomposition
        res = compute_4condition_decomposition(v_pos, v_neg, t_pos, t_neg)
        res["object_name"] = obj
        per_object_records.append(res)
        analyzed_objects.append(obj)

        print(f"  [{obj:20s}] N={res['n_samples']:4d} | "
              f"C1={res['P_C1_text_pres']:5.1f}% C2={res['P_C2_text_abs']:5.1f}% "
              f"C3={res['P_C3_img_pos']:5.1f}% C4={res['P_C4_img_neg']:5.1f}% "
              f"| Joint={res['P_all_4']:5.1f}%")

    if not per_object_records:
        print("\n  No valid objects found. Exiting.")
        return

    # ── Aggregate Macro-Averaged Results ──
    df_results = pd.DataFrame(per_object_records)
    csv_out = os.path.join(output_dir, "4condition_decomposition_table.csv")
    df_results.to_csv(csv_out, index=False)
    print(f"\n  Saved per-object table: {csv_out}")

    # Macro-average (per-object mean)
    summary = {
        "P_C1_text_pres":   float(df_results["P_C1_text_pres"].mean()),
        "P_C2_text_abs":    float(df_results["P_C2_text_abs"].mean()),
        "P_C3_img_pos":     float(df_results["P_C3_img_pos"].mean()),
        "P_C4_img_neg":     float(df_results["P_C4_img_neg"].mean()),
        "P_all_4":          float(df_results["P_all_4"].mean()),
        "P_joint_2x2":      float(df_results["P_joint_2x2"].mean()),

        "margin_C1_mean":   float(df_results["margin_C1_mean"].mean()),
        "margin_C2_mean":   float(df_results["margin_C2_mean"].mean()),
        "margin_C3_mean":   float(df_results["margin_C3_mean"].mean()),
        "margin_C4_mean":   float(df_results["margin_C4_mean"].mean()),
        "margin_joint_mean": float(df_results["margin_joint_mean"].mean()),

        "margin_C1_std":    float(df_results["margin_C1_mean"].std()),
        "margin_C2_std":    float(df_results["margin_C2_mean"].std()),
        "margin_C3_std":    float(df_results["margin_C3_mean"].std()),
        "margin_C4_std":    float(df_results["margin_C4_mean"].std()),

        "S11_mean":         float(df_results["S11_mean"].mean()),
        "S12_mean":         float(df_results["S12_mean"].mean()),
        "S21_mean":         float(df_results["S21_mean"].mean()),
        "S22_mean":         float(df_results["S22_mean"].mean()),

        "n_objects_analyzed": len(analyzed_objects),
        "total_samples":     int(df_results["n_samples"].sum()),
    }

    json_out = os.path.join(output_dir, "4condition_decomposition_summary.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary JSON: {json_out}")

    # ── Print Table 1 ──
    print("\n" + "=" * 70)
    print("  ╔════════════════════════════════════════════════════════╗")
    print("  ║              TABLE 1: 4-Condition Decomposition       ║")
    print("  ╠════════════════════════════════════════════════════════╣")
    print(f"  ║  C1: P(m(I_pres) > 0)  = {summary['P_C1_text_pres']:6.2f}%   [pos text wins on I+]  ║")
    print(f"  ║  C2: P(m(I_abs)  < 0)  = {summary['P_C2_text_abs']:6.2f}%   [neg text wins on I−]  ║")
    print(f"  ║  C3: P(n(t_pos)  > 0)  = {summary['P_C3_img_pos']:6.2f}%   [I+ wins for t_pos]    ║")
    print(f"  ║  C4: P(n(t_neg)  < 0)  = {summary['P_C4_img_neg']:6.2f}%   [I− wins for t_neg]    ║")
    print(f"  ║                                                      ║")
    print(f"  ║  P(all 4 satisfied)     = {summary['P_all_4']:6.2f}%                      ║")
    print(f"  ║  Random baseline (1/6)  = 16.67%                     ║")
    print("  ╠════════════════════════════════════════════════════════╣")
    print("  ║  Mean Score Matrix (macro-averaged):                  ║")
    print(f"  ║    S₁₁ = {summary['S11_mean']:+.4f}   S₁₂ = {summary['S12_mean']:+.4f}               ║")
    print(f"  ║    S₂₁ = {summary['S21_mean']:+.4f}   S₂₂ = {summary['S22_mean']:+.4f}               ║")
    print("  ╠════════════════════════════════════════════════════════╣")
    print("  ║  Mean Signed Margins:                                 ║")
    print(f"  ║    C1 margin (S₁₁−S₁₂) = {summary['margin_C1_mean']:+.5f}                   ║")
    print(f"  ║    C2 margin (S₂₂−S₂₁) = {summary['margin_C2_mean']:+.5f}                   ║")
    print(f"  ║    C3 margin (S₁₁−S₂₁) = {summary['margin_C3_mean']:+.5f}                   ║")
    print(f"  ║    C4 margin (S₂₂−S₁₂) = {summary['margin_C4_mean']:+.5f}                   ║")
    print("  ╚════════════════════════════════════════════════════════╝")
    print("=" * 70)

    # ── Render Figures ──
    render_4condition_figure(summary, output_dir)
    render_4condition_margins_figure(summary, output_dir)
    render_score_matrix_heatmap(summary, output_dir)

    print(f"\n  4-Condition Decomposition Complete!")
    print(f"  Objects analyzed: {len(analyzed_objects)}")
    print(f"  Total samples:    {summary['total_samples']}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="4-Condition Inequality Decomposition for CLIP Cosine Negation Matching (Table 1)"
    )
    parser.add_argument("--csv_path", type=str,
                        default="benchmarks/data/images/beaf_counterfactual_6col.csv")
    parser.add_argument("--output_dir", type=str,
                        default="logs/evaluation/4condition_decomposition")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--min_pairs", type=int, default=6,
                        help="Minimum counterfactual pairs per object (default: 6)")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    run_4condition_decomposition(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        model_name=args.model,
        pretrained=args.pretrained,
        min_pairs_per_obj=args.min_pairs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
