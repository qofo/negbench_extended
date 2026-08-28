"""
E1 Experiment: Atomic Concept Detection AUC on Counterfactual Minimal Pairs.

Evaluates the core foundational assumption of Alshehri et al. (ICML 2026, LCSE):
    "CLIP reliably detects whether concept X is present (AUC = 0.88)."

Tests whether this atomic concept presence score p(X) holds when scene context
is held constant using BEAF 1:1 counterfactual minimal pairs (I_pres vs I_abs)
with purely atomic affirmative prompts (e.g., "a photo of a {concept}").

Outputs:
  - e1_per_pair_scores.csv (Pair-level s_pres, s_abs, delta_s, correctness)
  - e1_per_concept_auc.csv (Per-concept AUC, win rate, delta_s statistics)
  - e1_summary_report.json (Macro AUC, Natural vs CF comparison, Alshehri gap)
  - fig_e1_concept_auc_distribution.png (Concept-level AUC distribution plot)
  - fig_e1_score_delta_distribution.png (Delta s sensitivity distribution plot)
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import open_clip

# Robust imports for both module and standalone execution
try:
    from benchmarks.src.analysis.beaf.beaf_loader import load_and_verify_counterfactual_pairs
    from benchmarks.src.analysis.beaf.vision_mechanisms import extract_vision_features_unified
except ImportError:
    from analysis.beaf.beaf_loader import load_and_verify_counterfactual_pairs
    from analysis.beaf.vision_mechanisms import extract_vision_features_unified


DEFAULT_TEMPLATES = [
    "a photo of a {}",
    "a {}",
    "there is a {} in the image",
    "an image of a {}",
]


def extract_normalized_image_features(
    model,
    preprocess,
    image_paths: List[str],
    device: str,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract final L2-normalized image embeddings using single-pass unified extractor."""
    feats_dict = extract_vision_features_unified(
        model=model,
        preprocess=preprocess,
        image_paths=image_paths,
        device=device,
        batch_size=batch_size,
    )
    final_feats = feats_dict["final_l2norm"]
    loaded_flags = np.array(feats_dict.get("loaded_flags", [True] * len(image_paths)))
    return final_feats, loaded_flags


def extract_normalized_text_features(
    model,
    tokenizer,
    concepts: List[str],
    device: str,
    prompt_template: str = "a photo of a {}",
    ensemble_prompts: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Extract final L2-normalized text embedding for each concept.
    Supports single atomic prompt or prompt template ensembling.
    """
    concept_embeddings = {}
    with torch.no_grad():
        for c in concepts:
            c_clean = c.replace("_", " ").strip()
            if ensemble_prompts:
                prompts = [tpl.format(c_clean) for tpl in DEFAULT_TEMPLATES]
            else:
                prompts = [prompt_template.format(c_clean)]

            tokens = tokenizer(prompts).to(device)
            text_feats = model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            mean_feat = text_feats.mean(dim=0, keepdim=True)
            mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
            concept_embeddings[c] = mean_feat.cpu().numpy().squeeze(0)

    return concept_embeddings


def compute_e1_minimal_pair_auc(
    df_pairs: pd.DataFrame,
    feats_pres: np.ndarray,
    feats_abs: np.ndarray,
    concept_text_feats: Dict[str, np.ndarray],
    natural_neg_samples: int = 10,
    min_pairs: int = 20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Computes pairwise cosine similarity, delta_s, and ROC-AUC per concept
    for both Counterfactual Minimal Pairs (I_pres vs I_abs) and Natural Heterogeneous pairs.
    Filters concepts with fewer than min_pairs counterfactual pairs.
    """
    np.random.seed(seed)
    unique_concepts = sorted(df_pairs["object_name"].unique().tolist())

    pair_records = []
    concept_records = []

    all_cf_y_true = []
    all_cf_y_scores = []
    all_nat_y_true = []
    all_nat_y_scores = []

    for c in unique_concepts:
        mask = (df_pairs["object_name"] == c).values
        n_c = int(np.sum(mask))
        if n_c < min_pairs:
            continue

        c_indices = np.where(mask)[0]
        t_vec = concept_text_feats[c]  # [D]

        # Normalized feature vectors for present and absent images
        emb_pres = feats_pres[c_indices]  # [N_c, D]
        emb_abs = feats_abs[c_indices]    # [N_c, D]

        # Cosine similarities
        s_pres = np.dot(emb_pres, t_vec)
        s_abs = np.dot(emb_abs, t_vec)
        delta_s = s_pres - s_abs

        # Pairwise wins: pres > abs (1.0), tie (0.5), pres < abs (0.0)
        wins = (s_pres > s_abs).astype(float) + 0.5 * (s_pres == s_abs).astype(float)
        cf_pairwise_auc = float(np.mean(wins))

        # Pooled ROC-AUC for this concept (treating pres=1, abs=0)
        y_true_c = np.concatenate([np.ones(n_c), np.zeros(n_c)])
        y_score_c = np.concatenate([s_pres, s_abs])
        try:
            cf_roc_auc = float(roc_auc_score(y_true_c, y_score_c))
        except ValueError:
            cf_roc_auc = cf_pairwise_auc

        all_cf_y_true.extend(y_true_c.tolist())
        all_cf_y_scores.extend(y_score_c.tolist())

        # Natural Between-Image Discrimination Baseline:
        # Compare I_pres with random images from OTHER concepts that do NOT contain concept c
        other_indices = np.where(~mask)[0]
        if len(other_indices) >= n_c:
            sampled_other_idx = np.random.choice(other_indices, size=n_c, replace=False)
            emb_nat_abs = feats_pres[sampled_other_idx]
            s_nat_abs = np.dot(emb_nat_abs, t_vec)
            y_true_nat = np.concatenate([np.ones(n_c), np.zeros(n_c)])
            y_score_nat = np.concatenate([s_pres, s_nat_abs])
            try:
                nat_roc_auc = float(roc_auc_score(y_true_nat, y_score_nat))
            except ValueError:
                nat_roc_auc = float(np.mean(s_pres > s_nat_abs))
            all_nat_y_true.extend(y_true_nat.tolist())
            all_nat_y_scores.extend(y_score_nat.tolist())
        else:
            nat_roc_auc = np.nan

        # Save pair-level records
        for i, idx in enumerate(c_indices):
            pair_row = df_pairs.iloc[idx]
            pair_records.append({
                "pair_id": pair_row.get("pair_id", idx),
                "object_name": c,
                "source_template": pair_row.get("source_template", ""),
                "orig_path": pair_row.get("orig_path", ""),
                "cf_path": pair_row.get("cf_path", ""),
                "score_pres": float(s_pres[i]),
                "score_abs": float(s_abs[i]),
                "delta_s": float(delta_s[i]),
                "pres_gt_abs": int(s_pres[i] > s_abs[i]),
                "is_tie": int(s_pres[i] == s_abs[i]),
            })

        concept_records.append({
            "object_name": c,
            "n_pairs": n_c,
            "cf_pairwise_auc": cf_pairwise_auc,
            "cf_roc_auc": cf_roc_auc,
            "natural_roc_auc": nat_roc_auc,
            "auc_gap_nat_minus_cf": nat_roc_auc - cf_pairwise_auc if not np.isnan(nat_roc_auc) else np.nan,
            "win_rate_pct": cf_pairwise_auc * 100.0,
            "mean_delta_s": float(np.mean(delta_s)),
            "median_delta_s": float(np.median(delta_s)),
            "std_delta_s": float(np.std(delta_s)),
            "pct_delta_le_zero": float(np.mean(delta_s <= 0) * 100.0),
        })

    df_pairs_out = pd.DataFrame(pair_records)
    df_concepts_out = pd.DataFrame(concept_records).sort_values(by="cf_pairwise_auc", ascending=False).reset_index(drop=True)

    # Macro & Overall Summary
    macro_cf_auc = float(df_concepts_out["cf_pairwise_auc"].mean())
    macro_nat_auc = float(df_concepts_out["natural_roc_auc"].dropna().mean()) if "natural_roc_auc" in df_concepts_out.columns else np.nan
    pooled_cf_auc = float(roc_auc_score(all_cf_y_true, all_cf_y_scores)) if len(all_cf_y_true) > 0 else macro_cf_auc
    pooled_nat_auc = float(roc_auc_score(all_nat_y_true, all_nat_y_scores)) if len(all_nat_y_true) > 0 else np.nan
    overall_win_rate = float(df_pairs_out["pres_gt_abs"].mean() * 100.0)
    pct_delta_le_zero_overall = float((df_pairs_out["delta_s"] <= 0).mean() * 100.0)

    # Determine Verdict based on Alshehri 0.88 benchmark
    alshehri_baseline = 0.88
    if macro_cf_auc <= 0.65:
        verdict = "STRONG_HEADLINE_DISCOVERY (0.88 was an artifact of scene context; vision evidence collapses on minimal pairs)"
    elif macro_cf_auc <= 0.78:
        verdict = "BOUNDARY_CONDITION_ESTABLISHED (Vision evidence substantially degraded under counterfactual control)"
    else:
        verdict = "VISION_EVIDENCE_ROBUST (Atomic presence detection holds even under strict minimal pair control)"

    summary_report = {
        "alshehri_reported_auc": alshehri_baseline,
        "min_pairs_threshold": min_pairs,
        "counterfactual_macro_auc": macro_cf_auc,
        "counterfactual_pooled_auc": pooled_cf_auc,
        "natural_between_image_macro_auc": macro_nat_auc,
        "natural_between_image_pooled_auc": pooled_nat_auc,
        "auc_drop_under_counterfactual_control": float(alshehri_baseline - macro_cf_auc),
        "overall_pairwise_win_rate_pct": overall_win_rate,
        "overall_pct_delta_le_zero": pct_delta_le_zero_overall,
        "total_pairs_evaluated": len(df_pairs_out),
        "total_concepts_evaluated": len(df_concepts_out),
        "verdict": verdict,
        "top5_strongest_concepts": df_concepts_out.head(5)[["object_name", "cf_pairwise_auc", "mean_delta_s"]].to_dict(orient="records"),
        "top5_weakest_concepts": df_concepts_out.tail(5)[["object_name", "cf_pairwise_auc", "mean_delta_s"]].to_dict(orient="records"),
    }

    return df_pairs_out, df_concepts_out, summary_report


def render_e1_visualizations(
    df_concepts: pd.DataFrame,
    df_pairs: pd.DataFrame,
    summary: Dict[str, Any],
    output_dir: str,
):
    """Render publication-grade visualization artifacts for E1."""
    os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # Plot 1: Concept-Level AUC Distribution & Alshehri Baseline
    # ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))

    concepts = df_concepts["object_name"].values
    cf_aucs = df_concepts["cf_pairwise_auc"].values
    nat_aucs = df_concepts["natural_roc_auc"].values if "natural_roc_auc" in df_concepts.columns else None

    x = np.arange(len(concepts))
    width = 0.4

    if nat_aucs is not None and not np.isnan(nat_aucs).all():
        ax.bar(x - width/2, nat_aucs, width, label="Natural (Between-Image AUC)", color="#3498db", alpha=0.85)
        ax.bar(x + width/2, cf_aucs, width, label="BEAF Counterfactual (Within-Scene Minimal Pair AUC)", color="#e74c3c", alpha=0.9)
    else:
        ax.bar(x, cf_aucs, width * 1.5, label="BEAF Counterfactual Minimal Pair AUC", color="#e74c3c", alpha=0.9)

    # Baseline reference lines
    ax.axhline(0.88, color="#2ecc71", linestyle="--", linewidth=2.0, label="Alshehri et al. Reported Atomic AUC (0.88)")
    ax.axhline(summary["counterfactual_macro_auc"], color="#c0392b", linestyle="-.", linewidth=2.0, label=f"E1 Counterfactual Macro AUC ({summary['counterfactual_macro_auc']:.3f})")
    ax.axhline(0.50, color="gray", linestyle=":", linewidth=1.5, label="Random Chance (0.50)")

    ax.set_ylabel("Atomic Concept Detection AUC", fontsize=12, fontweight="bold")
    ax.set_xlabel("Object Concept (BEAF Benchmark)", fontsize=12, fontweight="bold")
    ax.set_title("E1: Atomic Concept Detection AUC — Counterfactual Minimal Pair vs Natural Baseline", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(concepts, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0.35, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "fig_e1_concept_auc_distribution.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot1_path}")

    # ──────────────────────────────────────────────────────────
    # Plot 2: Sensitivity Delta_s Distribution (s_pres - s_abs)
    # ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    deltas = df_pairs["delta_s"].values

    n_bins = 50
    counts, bins, patches = ax.hist(deltas, bins=n_bins, color="#2980b9", edgecolor="black", alpha=0.75, density=True)

    # Highlight delta <= 0 (blind / reverse zone)
    for count, b_left, patch in zip(counts, bins[:-1], patches):
        if b_left < 0:
            patch.set_facecolor("#e74c3c")
            patch.set_alpha(0.85)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=2.0, label="Zero Margin (s_pres = s_abs)")
    ax.axvline(np.mean(deltas), color="#27ae60", linestyle="-", linewidth=2.0, label=f"Mean Δs = {np.mean(deltas):+.4f}")
    ax.axvline(np.median(deltas), color="#f39c12", linestyle=":", linewidth=2.0, label=f"Median Δs = {np.median(deltas):+.4f}")

    pct_fail = (deltas <= 0).mean() * 100.0
    ax.text(0.05, 0.85, f"Insensitive / Inverted (Δs ≤ 0): {pct_fail:.1f}%\nTotal Pairs: {len(deltas)}",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fadbd8", edgecolor="#e74c3c", alpha=0.9))

    ax.set_xlabel("Similarity Margin: Δs = CosSim(I_pres, T_c) - CosSim(I_abs, T_c)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title("E1: Embedding Sensitivity Distribution under Object Erasure", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "fig_e1_score_delta_distribution.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot2_path}")


def main():
    parser = argparse.ArgumentParser(description="E1: Minimal-Pair Atomic Concept Detection AUC Evaluator")
    parser.add_argument("--csv_path", type=str, default="benchmarks/data/images/beaf_counterfactual_6col.csv")
    parser.add_argument("--image_root", type=str, default="benchmarks/data/images")
    parser.add_argument("--output_dir", type=str, default="logs/evaluation/e1_minimal_pair_auc")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--prompt_template", type=str, default="a photo of a {}")
    parser.add_argument("--ensemble_prompts", action="store_true", help="Ensemble multiple atomic prompt templates")
    parser.add_argument("--min_pairs", type=int, default=20, help="Minimum counterfactual pairs per concept (default: 20)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  E1: Atomic Concept Detection AUC on BEAF Minimal Pairs              ║")
    print("║  Testing Alshehri et al. (ICML 2026) AUC = 0.88 Assumption           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  Model       : {args.model} ({args.pretrained or 'Random Init'}) | Device: {device}")
    print(f"  Dataset CSV : {args.csv_path}")
    print(f"  Image Root  : {args.image_root}")
    print(f"  Output Dir  : {args.output_dir}")
    print(f"  Min Pairs   : {args.min_pairs}")
    print(f"  Prompt      : {args.prompt_template} (Ensemble: {args.ensemble_prompts})\n")

    # 1. Load Counterfactual Minimal Pairs
    print("  [1/4] Loading and verifying 1:1 counterfactual image pairs...")
    df_raw, df_pairs, _ = load_and_verify_counterfactual_pairs(args.csv_path, args.image_root)
    n_pairs = len(df_pairs)
    unique_concepts = sorted(df_pairs["object_name"].unique().tolist())
    print(f"  -> Verified {n_pairs} pairs across {len(unique_concepts)} distinct object concepts.")

    # 2. Load Model & Tokenizer
    print(f"\n  [2/4] Initializing CLIP model '{args.model}'...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    # 3. Extract Features
    print("\n  [3/4] Extracting vision embeddings for present and absent images...")
    feats_pres, flags_pres = extract_normalized_image_features(model, preprocess, df_pairs["orig_path"].tolist(), device, args.batch_size)
    feats_abs, flags_abs = extract_normalized_image_features(model, preprocess, df_pairs["cf_path"].tolist(), device, args.batch_size)

    # Filter any missing image entries
    valid_mask = flags_pres & flags_abs
    if not np.all(valid_mask):
        df_pairs = df_pairs[valid_mask].reset_index(drop=True)
        feats_pres = feats_pres[valid_mask]
        feats_abs = feats_abs[valid_mask]
        print(f"  -> Filtered {np.sum(~valid_mask)} unreadable pairs. Active pairs: {len(df_pairs)}")

    print(f"  Extracting atomic text embeddings for {len(unique_concepts)} concepts...")
    concept_text_feats = extract_normalized_text_features(
        model=model,
        tokenizer=tokenizer,
        concepts=unique_concepts,
        device=device,
        prompt_template=args.prompt_template,
        ensemble_prompts=args.ensemble_prompts,
    )

    # 4. Compute E1 Metrics
    print("\n  [4/4] Computing Pairwise Similarity, Margins (Δs), and Concept ROC-AUC...")
    df_pairs_out, df_concepts_out, summary = compute_e1_minimal_pair_auc(
        df_pairs=df_pairs,
        feats_pres=feats_pres,
        feats_abs=feats_abs,
        concept_text_feats=concept_text_feats,
        min_pairs=args.min_pairs,
        seed=args.seed,
    )

    # Render Visualizations & Export
    render_e1_visualizations(df_concepts_out, df_pairs_out, summary, args.output_dir)

    pairs_csv = os.path.join(args.output_dir, "e1_per_pair_scores.csv")
    concepts_csv = os.path.join(args.output_dir, "e1_per_concept_auc.csv")
    report_json = os.path.join(args.output_dir, "e1_summary_report.json")

    df_pairs_out.to_csv(pairs_csv, index=False)
    df_concepts_out.to_csv(concepts_csv, index=False)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Terminal Summary Table
    print("\n" + "═" * 70)
    print("  E1 EXPERIMENT RESULTS SUMMARY")
    print("═" * 70)
    print(f"  Alshehri et al. Reported Atomic AUC : {summary['alshehri_reported_auc']:.4f}")
    print(f"  Natural Baseline Between-Image AUC : {summary['natural_between_image_macro_auc']:.4f}")
    print(f"  BEAF Counterfactual Minimal Pair AUC: {summary['counterfactual_macro_auc']:.4f}  <-- [CRITICAL HEADLINE]")
    print(f"  AUC Gap (Alshehri vs Counterfactual): {summary['auc_drop_under_counterfactual_control']:+.4f}")
    print(f"  Overall Pairwise Win Rate (s_p > s_a): {summary['overall_pairwise_win_rate_pct']:.2f}%")
    print(f"  Percentage of Failure/Ties (Δs <= 0) : {summary['overall_pct_delta_le_zero']:.2f}%")
    print(f"  Evaluated Concepts (N >= {args.min_pairs})     : {len(df_concepts_out)} objects (Total {len(df_pairs_out)} pairs)")
    print(f"  Verdict                             : {summary['verdict']}")
    print("═" * 70)
    print(f"  Results saved in: {args.output_dir}")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
