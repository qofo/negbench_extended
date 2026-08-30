"""
E1 Placebo Test: Unrelated-Object AUC on X-Deletion Minimal Pairs.

SCIENTIFIC QUESTION:
    The E1 experiment shows AUC = 0.80 when querying concept X on pairs
    (I_pres^X vs I_abs^X) where X was removed. Is this signal X-specific,
    or does it reflect inpainting artifacts (texture, brightness, fill-in
    patterns) that inflate AUC for ANY query?

DESIGN:
    For each target object X and its deletion pair (I_pres^X, I_abs^X):
      - Find unrelated distractor object Y that:
          * Y != X
          * Y co-occurs in the SAME original image as X (present in I_pres^X)
          * Y is NOT the deleted object -> Y should appear in BOTH images
      - Compute: AUC(I_pres^X, I_abs^X | query=T_Y)

    Prediction:
      - If AUC(Y) ~= 0.50 -> the 0.80 is X-specific vision signal  PASS
      - If AUC(Y) >= 0.60 -> inpainting artifacts contaminate signal FAIL

    Since Y is present in BOTH I_pres^X and I_abs^X (only X changed),
    a well-functioning system should produce AUC = 0.50 for Y.

IMPLEMENTATION STRATEGY:
    1. Load beaf_counterfactual_6col.csv -> pairs indexed by (image_id, object_name)
    2. For each image_id, find ALL objects annotated in that image
    3. For each target object X on image_id, pick a random Y != X that:
       - also appears (object_in_image=True) for the same original image_id
    4. Use the (I_pres^X, I_abs^X) pair but query with T_Y
    5. Compute per-pair delta_s_Y = cos(I_pres^X, T_Y) - cos(I_abs^X, T_Y)
    6. Compute AUC_Y over all pairs
    7. Compare AUC_X (from E1) vs AUC_Y (placebo)

Outputs:
  - placebo_per_pair_scores.csv
  - placebo_per_concept_auc.csv
  - placebo_summary.json
  - fig_placebo_auc_comparison.png   (AUC_X vs AUC_Y side-by-side per concept)
  - fig_placebo_delta_distribution.png

Usage:
    python -m benchmarks.src.evaluation.eval_e1_placebo_test

Pass/Fail verdict:
    AUC_Y (macro) < 0.55  -> PASS  (signal is X-specific)
    AUC_Y (macro) >= 0.60 -> FAIL  (artifact contamination suspected)
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import open_clip

try:
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.cli import (
    add_model_args, add_run_args, add_data_args, add_cache_args,
    add_restriction_args, add_concept_args, add_bias_args,
    )
    from benchmarks.src.analysis.beaf.beaf_loader import load_and_verify_counterfactual_pairs
    from benchmarks.src.analysis.beaf.vision_mechanisms import extract_vision_features_unified
    from benchmarks.src.analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction, DEFAULT_CACHE_DIR,
    )
    from benchmarks.src.analysis.config import set_seed, coerce_bool_column
    from benchmarks.src.analysis.paths import resolve_image_path as resolve_path
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.model_loader import load_clip_for_eval
    from analysis.cli import (
    add_model_args, add_run_args, add_data_args, add_cache_args,
    add_restriction_args, add_concept_args, add_bias_args,
    )
    from analysis.beaf.beaf_loader import load_and_verify_counterfactual_pairs
    from analysis.beaf.vision_mechanisms import extract_vision_features_unified
    from analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction, DEFAULT_CACHE_DIR,
    )
    from analysis.config import set_seed, coerce_bool_column
    from analysis.paths import resolve_image_path as resolve_path


DEFAULT_TEMPLATES = [
    "a photo of a {}",
    "a {}",
    "there is a {} in the image",
    "an image of a {}",
]


# ============================================================
# Feature Extraction Helpers
# ============================================================
def extract_normalized_image_features(
    model, preprocess, image_paths: List[str], device: str, batch_size: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    feats_dict = extract_vision_features_unified(
        model=model, preprocess=preprocess,
        image_paths=image_paths, device=device, batch_size=batch_size,
    )
    final_feats = feats_dict["final_l2norm"]
    loaded_flags = np.array(feats_dict.get("loaded_flags", [True] * len(image_paths)))
    return final_feats, loaded_flags


def extract_normalized_text_features(
    model, tokenizer, concepts: List[str], device: str,
    prompt_template: str = "a photo of a {}", ensemble_prompts: bool = False,
) -> Dict[str, np.ndarray]:
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


# ============================================================
# Core: Build Placebo Pairing
# ============================================================
def build_placebo_assignments(
    df: pd.DataFrame, seed: int = 42
) -> pd.DataFrame:
    """
    For each row in df where object_in_image=True (the "X present" image),
    find a distractor object Y != X that:
      - Also appears (object_in_image=True) for the SAME original image path
      - Has its own counterfactual pair in df

    Returns a DataFrame of "present" rows with columns added:
      distractor_object: Y (or None)
      distractor_available: bool
    """
    rng = np.random.RandomState(seed)

    present_rows = df[df["object_in_image"] == True][["image_path", "object_name"]].copy()
    img_to_objects: Dict[str, List[str]] = {}
    for _, row in present_rows.iterrows():
        img = row["image_path"]
        obj = row["object_name"]
        img_to_objects.setdefault(img, []).append(obj)

    df_pres = df[df["object_in_image"] == True].copy()
    distractor_objects = []

    for _, row in df_pres.iterrows():
        orig_img = row["image_path"]
        target_x = row["object_name"]
        co_objects = img_to_objects.get(orig_img, [])
        candidates = [o for o in co_objects if o != target_x]
        if candidates:
            chosen = rng.choice(candidates)
            distractor_objects.append(chosen)
        else:
            distractor_objects.append(None)

    df_pres = df_pres.copy()
    df_pres["distractor_object"] = distractor_objects
    df_pres["distractor_available"] = df_pres["distractor_object"].notna()
    return df_pres


# ============================================================
# Core: Compute Placebo AUC
# ============================================================
def compute_placebo_auc(
    df_pres: pd.DataFrame,
    feats_pres_all: np.ndarray,
    feats_abs_all: np.ndarray,
    concept_text_feats: Dict[str, np.ndarray],
    min_pairs: int = 20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    For each target concept X, compute:
      - AUC_X: standard E1 signal (query=T_X on X-deletion pair)
      - AUC_Y: placebo signal (query=T_Y on the SAME pair, Y!=X, Y co-occurs)
    """
    all_concepts = sorted(df_pres["object_name"].unique().tolist())

    pair_records = []
    concept_records = []
    all_x_y_true, all_x_y_scores = [], []
    all_placebo_y_true, all_placebo_y_scores = [], []

    for x_concept in all_concepts:
        x_mask = (df_pres["object_name"] == x_concept).values
        n_x = int(np.sum(x_mask))
        if n_x < min_pairs:
            continue

        x_subset = df_pres[x_mask].reset_index(drop=True)
        has_distractor = x_subset["distractor_available"].values
        n_valid = int(np.sum(has_distractor))

        if n_valid < min_pairs:
            print(f"  [SKIP] {x_concept}: only {n_valid}/{n_x} pairs have a valid distractor (need {min_pairs})")
            continue

        x_subset_valid = x_subset[has_distractor].reset_index(drop=True)
        global_x_indices = np.where(x_mask)[0][has_distractor]
        emb_pres = feats_pres_all[global_x_indices]
        emb_abs = feats_abs_all[global_x_indices]
        n_valid = len(emb_pres)

        t_x = concept_text_feats.get(x_concept)
        if t_x is None:
            print(f"  [SKIP] {x_concept}: no text embedding found")
            continue

        # AUC_X: standard E1 signal
        s_x_pres = np.dot(emb_pres, t_x)
        s_x_abs = np.dot(emb_abs, t_x)
        delta_x = s_x_pres - s_x_abs
        wins_x = (s_x_pres > s_x_abs).astype(float) + 0.5 * (s_x_pres == s_x_abs).astype(float)
        auc_x = float(np.mean(wins_x))

        y_true_x = np.concatenate([np.ones(n_valid), np.zeros(n_valid)])
        y_score_x = np.concatenate([s_x_pres, s_x_abs])
        try:
            roc_auc_x = float(roc_auc_score(y_true_x, y_score_x))
        except ValueError:
            roc_auc_x = auc_x
        all_x_y_true.extend(y_true_x.tolist())
        all_x_y_scores.extend(y_score_x.tolist())

        # AUC_Y: placebo signal
        placebo_deltas = []
        placebo_wins = []

        for i in range(n_valid):
            y_concept = x_subset_valid.iloc[i]["distractor_object"]
            t_y = concept_text_feats.get(y_concept)
            if t_y is None:
                placebo_deltas.append(0.0)
                placebo_wins.append(0.5)
                continue

            s_y_pres_i = float(np.dot(emb_pres[i], t_y))
            s_y_abs_i = float(np.dot(emb_abs[i], t_y))
            delta_y_i = s_y_pres_i - s_y_abs_i
            win_y_i = (1.0 if delta_y_i > 0 else 0.5 if delta_y_i == 0 else 0.0)
            placebo_deltas.append(delta_y_i)
            placebo_wins.append(win_y_i)
            all_placebo_y_true.extend([1.0, 0.0])
            all_placebo_y_scores.extend([s_y_pres_i, s_y_abs_i])

            pair_records.append({
                "target_object_X": x_concept,
                "distractor_object_Y": y_concept,
                "image_path_orig": x_subset_valid.iloc[i]["image_path"],
                "s_X_pres": float(s_x_pres[i]),
                "s_X_abs": float(s_x_abs[i]),
                "delta_X": float(delta_x[i]),
                "win_X": int(s_x_pres[i] > s_x_abs[i]),
                "s_Y_pres": s_y_pres_i,
                "s_Y_abs": s_y_abs_i,
                "delta_Y": delta_y_i,
                "win_Y": int(win_y_i > 0.5),
            })

        placebo_deltas = np.array(placebo_deltas)
        placebo_wins = np.array(placebo_wins)
        auc_y = float(np.mean(placebo_wins))
        verdict_str = "PASS" if auc_y < 0.55 else ("BORDERLINE" if auc_y < 0.60 else "FAIL")
        concept_records.append({
            "object_X": x_concept,
            "n_pairs": n_valid,
            "AUC_X": auc_x,
            "ROC_AUC_X": roc_auc_x,
            "mean_delta_X": float(np.mean(delta_x)),
            "AUC_Y_placebo": auc_y,
            "mean_delta_Y": float(np.mean(placebo_deltas)),
            "std_delta_Y": float(np.std(placebo_deltas)),
            "AUC_gap_X_minus_Y": auc_x - auc_y,
            "pct_delta_Y_le_zero": float(np.mean(placebo_deltas <= 0) * 100.0),
            "verdict": verdict_str,
        })

        icon = "PASS" if verdict_str == "PASS" else ("BORDERLINE" if verdict_str == "BORDERLINE" else "FAIL")
        print(f"  [{x_concept:20s}] N={n_valid:4d} | AUC_X={auc_x:.3f} | AUC_Y={auc_y:.3f} "
              f"| dX={np.mean(delta_x):+.4f} | dY={np.mean(placebo_deltas):+.4f} | {icon}")

    df_pairs_out = pd.DataFrame(pair_records)
    df_concepts_out = (pd.DataFrame(concept_records)
                       .sort_values(by="AUC_Y_placebo", ascending=False)
                       .reset_index(drop=True))

    macro_auc_x = float(df_concepts_out["AUC_X"].mean())
    macro_auc_y = float(df_concepts_out["AUC_Y_placebo"].mean())
    try:
        pooled_auc_x = float(roc_auc_score(all_x_y_true, all_x_y_scores))
    except ValueError:
        pooled_auc_x = macro_auc_x
    try:
        pooled_auc_y = float(roc_auc_score(all_placebo_y_true, all_placebo_y_scores))
    except ValueError:
        pooled_auc_y = macro_auc_y

    n_pass = int((df_concepts_out["AUC_Y_placebo"] < 0.55).sum())
    n_border = int(((df_concepts_out["AUC_Y_placebo"] >= 0.55) & (df_concepts_out["AUC_Y_placebo"] < 0.60)).sum())
    n_fail = int((df_concepts_out["AUC_Y_placebo"] >= 0.60).sum())
    n_total = len(df_concepts_out)

    if macro_auc_y < 0.55:
        verdict = "PLACEBO PASS: AUC_Y is near chance. The 0.80 is X-specific vision signal, not an inpainting artifact."
    elif macro_auc_y < 0.60:
        verdict = "BORDERLINE: Some artifact contamination possible. Per-concept investigation recommended."
    else:
        verdict = "PLACEBO FAIL: AUC_Y >= 0.60. Inpainting artifacts likely inflate E1 AUC. Claims in Section 2 need revision."

    summary = {
        "n_concepts_evaluated": n_total,
        "total_pairs_evaluated": len(df_pairs_out),
        "macro_AUC_X": macro_auc_x,
        "macro_AUC_Y_placebo": macro_auc_y,
        "pooled_AUC_X": pooled_auc_x,
        "pooled_AUC_Y_placebo": pooled_auc_y,
        "AUC_X_minus_Y_macro": macro_auc_x - macro_auc_y,
        "n_concepts_PASS": n_pass,
        "n_concepts_BORDERLINE": n_border,
        "n_concepts_FAIL": n_fail,
        "pct_PASS": round(100.0 * n_pass / max(n_total, 1), 1),
        "pct_FAIL": round(100.0 * n_fail / max(n_total, 1), 1),
        "mean_delta_Y_global": float(df_pairs_out["delta_Y"].mean()) if len(df_pairs_out) > 0 else 0.0,
        "mean_delta_X_global": float(df_pairs_out["delta_X"].mean()) if len(df_pairs_out) > 0 else 0.0,
        "pass_threshold_AUC_Y": 0.55,
        "fail_threshold_AUC_Y": 0.60,
        "verdict": verdict,
        "top5_worst_placebo": df_concepts_out.head(5)[
            ["object_X", "AUC_Y_placebo", "AUC_X", "AUC_gap_X_minus_Y"]
        ].to_dict(orient="records"),
        "top5_best_placebo": df_concepts_out.tail(5)[
            ["object_X", "AUC_Y_placebo", "AUC_X", "AUC_gap_X_minus_Y"]
        ].to_dict(orient="records"),
    }

    return df_pairs_out, df_concepts_out, summary


# ============================================================
# Visualization
# ============================================================
def render_placebo_visualizations(
    df_concepts: pd.DataFrame,
    df_pairs: pd.DataFrame,
    summary: Dict[str, Any],
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: AUC_X vs AUC_Y per concept bar chart
    fig, ax = plt.subplots(figsize=(max(14, len(df_concepts) * 0.3), 6))
    x = np.arange(len(df_concepts))
    width = 0.38
    concepts = df_concepts["object_X"].values
    auc_x_vals = df_concepts["AUC_X"].values
    auc_y_vals = df_concepts["AUC_Y_placebo"].values

    ax.bar(x - width / 2, auc_x_vals, width,
           label="AUC_X (target: T_X query on X-deletion pair)", color="#2980b9", alpha=0.88, edgecolor="black")
    ax.bar(x + width / 2, auc_y_vals, width,
           label="AUC_Y (placebo: T_Y query on X-deletion pair)", color="#e74c3c", alpha=0.88, edgecolor="black")

    ax.axhline(0.50, color="black", ls=":", lw=2.0, label="Random Chance (0.50)")
    ax.axhline(0.55, color="orange", ls="--", lw=1.5, alpha=0.8, label="PASS threshold (0.55)")
    ax.axhline(0.60, color="#c0392b", ls="--", lw=1.5, alpha=0.8, label="FAIL threshold (0.60)")
    ax.axhline(summary["macro_AUC_Y_placebo"], color="#c0392b", ls="-.", lw=2.0,
               label=f"Macro AUC_Y = {summary['macro_AUC_Y_placebo']:.3f}")

    ax.set_ylabel("Pairwise AUC (I_pres^X > I_abs^X)", fontsize=12)
    ax.set_title(
        "E1 Placebo Test: X-Specific Signal vs Inpainting Artifact Check\n"
        "T_Y query on X-deletion pairs — AUC_Y should be ~0.50 if signal is artifact-free",
        fontsize=12, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(concepts, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylim(0.35, 1.05)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")

    for tick, auc_y in zip(ax.get_xticklabels(), auc_y_vals):
        if auc_y >= 0.60:
            tick.set_color("#e74c3c")
        elif auc_y >= 0.55:
            tick.set_color("orange")
        else:
            tick.set_color("#27ae60")

    plt.tight_layout()
    out1 = os.path.join(output_dir, "fig_placebo_auc_comparison.png")
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out1}")

    # Figure 2: delta_X vs delta_Y distributions
    if len(df_pairs) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        delta_x_vals = df_pairs["delta_X"].values
        delta_y_vals = df_pairs["delta_Y"].values

        for ax, deltas, color, label, auc_label in [
            (ax1, delta_x_vals, "#2980b9",
             "ds_X = cos(I_pres^X, T_X) - cos(I_abs^X, T_X)",
             f"AUC_X = {summary['macro_AUC_X']:.3f}"),
            (ax2, delta_y_vals, "#e74c3c",
             "ds_Y = cos(I_pres^X, T_Y) - cos(I_abs^X, T_Y)",
             f"AUC_Y = {summary['macro_AUC_Y_placebo']:.3f} (placebo)"),
        ]:
            counts, bins, patches = ax.hist(
                deltas, bins=50, color=color, edgecolor="black", alpha=0.75, density=True
            )
            for count, b_left, patch in zip(counts, bins[:-1], patches):
                if b_left < 0:
                    patch.set_facecolor("#c0392b")
                    patch.set_alpha(0.9)
            ax.axvline(0.0, color="black", ls="--", lw=2.0, label="Zero margin")
            ax.axvline(float(np.mean(deltas)), color="#f1c40f", ls="-", lw=2.0,
                       label=f"Mean = {np.mean(deltas):+.4f}")
            pct_fail = (deltas <= 0).mean() * 100.0
            ax.text(0.04, 0.92, f"Delta <= 0: {pct_fail:.1f}%\n{auc_label}",
                    transform=ax.transAxes, fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.9))
            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel("Density", fontsize=11)
            ax.grid(axis="y", ls="--", alpha=0.4)
            ax.legend(fontsize=9)

        ax1.set_title("Target Signal: ds_X (should be positive)", fontsize=12, fontweight="bold")
        ax2.set_title("Placebo Signal: ds_Y (should be ~0, symmetric)", fontsize=12, fontweight="bold", color="#c0392b")
        plt.suptitle("E1 Placebo Test - Similarity Margin Distributions", fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        out2 = os.path.join(output_dir, "fig_placebo_delta_distribution.png")
        plt.savefig(out2, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out2}")

    # Figure 3: scatter AUC_X vs AUC_Y
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        df_concepts["AUC_X"], df_concepts["AUC_Y_placebo"],
        c=df_concepts["AUC_Y_placebo"], cmap="RdYlGn_r",
        vmin=0.40, vmax=0.75, s=80, edgecolors="black", alpha=0.9
    )
    plt.colorbar(sc, ax=ax, label="AUC_Y (placebo)")
    for _, row in df_concepts.iterrows():
        ax.annotate(row["object_X"], (row["AUC_X"], row["AUC_Y_placebo"]),
                    fontsize=7, alpha=0.75, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0.50, color="black", ls=":", lw=1.5, label="Chance (0.50)")
    ax.axhline(0.55, color="orange", ls="--", lw=1.2, alpha=0.8, label="PASS threshold")
    ax.axhline(0.60, color="red", ls="--", lw=1.2, alpha=0.8, label="FAIL threshold")
    ax.set_xlabel("AUC_X (target E1 signal)", fontsize=12)
    ax.set_ylabel("AUC_Y (placebo -- inpainting artifact check)", fontsize=12)
    ax.set_title("Placebo Test: AUC_X vs AUC_Y per Concept", fontsize=12, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out3 = os.path.join(output_dir, "fig_placebo_scatter.png")
    plt.savefig(out3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out3}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="E1 Placebo Test: Unrelated-Object AUC on X-Deletion Minimal Pairs"
    )
    add_model_args(parser, "ViT-B-32", "openai")
    add_run_args(parser, "logs/evaluation/e1_placebo_test", seed=42, batch_size=128)
    add_data_args(parser, csv_path="benchmarks/data/images/beaf_counterfactual_6col.csv", image_root="benchmarks/data/images")
    add_cache_args(parser)
    add_restriction_args(parser, "Comma list, or path to txt/csv/json, limiting the evaluated target concepts X. " "Distractors Y are still drawn from the full concept pool, so the placebo " "assignment is unchanged by this flag.")
    add_concept_args(parser, help_text="Min pairs (with valid distractor) per concept (default: 20)")
    parser.add_argument("--prompt_template", type=str, default="a photo of a {}")
    parser.add_argument("--ensemble_prompts", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_kw = dict(model=args.model, pretrained=args.pretrained,
                    cache_dir=args.cache_dir, enabled=args.use_cache)

    print("=" * 68)
    print("  E1 Placebo Test: Unrelated-Object AUC")
    print("  H0: AUC_Y ~= 0.50  -> signal is X-specific, not artifact")
    print("=" * 68)
    print(f"  Model    : {args.model} ({args.pretrained}) | Device: {device}")
    print(f"  CSV      : {args.csv_path}")
    print(f"  Out Dir  : {args.output_dir}")
    print(f"  Min Pairs: {args.min_pairs}\n")

    # 1. Load raw CSV
    print("  [1/5] Loading CSV...")
    df_full = pd.read_csv(args.csv_path)
    coerce_bool_column(df_full, "object_in_image")
    print(f"  -> {len(df_full)} rows, {df_full['object_name'].nunique()} concepts")

    # 2. Build placebo assignments on present rows
    print("  [2/5] Building distractor (Y != X) assignments...")
    df_pres_assigned = build_placebo_assignments(df_full, seed=args.seed)
    n_with_distractor = int(df_pres_assigned["distractor_available"].sum())
    print(f"  -> {n_with_distractor}/{len(df_pres_assigned)} present-rows have a valid Y")

    # 3. Resolve counterfactual (absent) image paths
    print("  [3/5] Resolving counterfactual image paths...")
    df_abs_full = df_full[df_full["object_in_image"] == False].copy()
    has_source_template = "source_template" in df_full.columns
    abs_lookup: Dict[Tuple, str] = {}
    for _, row in df_abs_full.iterrows():
        key = (row["object_name"], row.get("source_template", "") if has_source_template else "")
        abs_lookup[key] = row["image_path"]

    def get_cf_path(row):
        key = (row["object_name"], row.get("source_template", "") if has_source_template else "")
        return abs_lookup.get(key, None)

    df_pres_assigned["cf_path"] = df_pres_assigned.apply(get_cf_path, axis=1)
    df_pres_assigned = df_pres_assigned[df_pres_assigned["cf_path"].notna()].reset_index(drop=True)
    print(f"  -> {len(df_pres_assigned)} rows with resolved CF paths")

    # 4. Load model and extract features
    print(f"\n  [4/5] Loading CLIP model '{args.model}' and extracting features...")
    model, preprocess, tokenizer = load_clip_for_eval(
        args.model, args.pretrained, device)

    # Text features cover every concept, including distractors outside any restriction.
    all_concepts = sorted(df_full["object_name"].unique().tolist())
    print(f"  Extracting text embeddings for {len(all_concepts)} concepts...")

    def _encode_concept_prompts():
        d = extract_normalized_text_features(
            model, tokenizer, all_concepts, device,
            prompt_template=args.prompt_template,
            ensemble_prompts=args.ensemble_prompts,
        )
        return (np.stack([d[c] for c in all_concepts]),)

    (concept_matrix,) = cached_encode(
        _encode_concept_prompts,
        kind=f"text_concept@l2norm|{args.prompt_template}|ens={int(args.ensemble_prompts)}",
        items=all_concepts, **cache_kw)
    concept_text_feats = {c: concept_matrix[i] for i, c in enumerate(all_concepts)}

    pres_paths = [resolve_path(p, args.image_root) for p in df_pres_assigned["image_path"].tolist()]
    abs_paths = [resolve_path(p, args.image_root) for p in df_pres_assigned["cf_path"].tolist()]

    feats_pres, flags_pres = cached_encode(
        lambda: extract_normalized_image_features(model, preprocess, pres_paths, device, args.batch_size),
        kind="image_pres@l2norm+flags", items=pres_paths, **cache_kw)
    feats_abs, flags_abs = cached_encode(
        lambda: extract_normalized_image_features(model, preprocess, abs_paths, device, args.batch_size),
        kind="image_abs@l2norm+flags", items=abs_paths, **cache_kw)

    valid_mask = flags_pres & flags_abs
    df_pres_assigned = df_pres_assigned[valid_mask].reset_index(drop=True)
    feats_pres = feats_pres[valid_mask]
    feats_abs = feats_abs[valid_mask]
    print(f"  -> {int(np.sum(valid_mask))}/{len(valid_mask)} pairs successfully loaded")

    # Restriction is applied here, after the distractors were drawn, so limiting the
    # reported concepts never changes which Y each remaining pair was assigned.
    restrict = load_object_restriction(args.restrict_objects)
    if restrict is not None:
        keep_mask = df_pres_assigned["object_name"].isin(set(restrict)).to_numpy()
        missing = sorted(set(restrict) - set(df_pres_assigned["object_name"].unique().tolist()))
        df_pres_assigned = df_pres_assigned[keep_mask].reset_index(drop=True)
        feats_pres = feats_pres[keep_mask]
        feats_abs = feats_abs[keep_mask]
        print(f"  -> Restricted to {df_pres_assigned['object_name'].nunique()} target concepts"
              + (f" ({len(missing)} requested but absent: {missing[:5]})" if missing else ""))

    # 5. Compute placebo AUC
    print("\n  [5/5] Computing AUC_X and AUC_Y (placebo)...")
    df_pairs_out, df_concepts_out, summary = compute_placebo_auc(
        df_pres=df_pres_assigned,
        feats_pres_all=feats_pres,
        feats_abs_all=feats_abs,
        concept_text_feats=concept_text_feats,
        min_pairs=args.min_pairs,
        seed=args.seed,
    )

    # Render and save
    render_placebo_visualizations(df_concepts_out, df_pairs_out, summary, args.output_dir)

    summary["provenance"] = build_provenance(
        args, n_concepts=len(df_concepts_out), n_pairs=len(df_pairs_out),
        prompt_template=args.prompt_template, ensemble_prompts=bool(args.ensemble_prompts))

    df_pairs_out.to_csv(os.path.join(args.output_dir, "placebo_per_pair_scores.csv"), index=False)
    df_concepts_out.to_csv(os.path.join(args.output_dir, "placebo_per_concept_auc.csv"), index=False)
    with open(os.path.join(args.output_dir, "placebo_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Terminal Report
    print("\n" + "=" * 70)
    print("  E1 PLACEBO TEST RESULTS")
    print("=" * 70)
    print(f"  AUC_X (target, macro)   : {summary['macro_AUC_X']:.4f}  <- original E1 signal")
    print(f"  AUC_Y (placebo, macro)  : {summary['macro_AUC_Y_placebo']:.4f}  <- should be ~0.50")
    print(f"  Delta (AUC_X - AUC_Y)   : {summary['AUC_X_minus_Y_macro']:+.4f}")
    print(f"  Concepts PASS (<0.55)   : {summary['n_concepts_PASS']}/{summary['n_concepts_evaluated']} ({summary['pct_PASS']:.1f}%)")
    print(f"  Concepts FAIL (>=0.60)  : {summary['n_concepts_FAIL']}/{summary['n_concepts_evaluated']} ({summary['pct_FAIL']:.1f}%)")
    print(f"  Mean delta_X (target)   : {summary['mean_delta_X_global']:+.5f}")
    print(f"  Mean delta_Y (placebo)  : {summary['mean_delta_Y_global']:+.5f}")
    print("=" * 70)
    print(f"  VERDICT: {summary['verdict']}")
    print("=" * 70)
    print(f"  Results saved to: {args.output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
