"""
Negation Information Existence Probe in CLIP Text Embeddings.

Research Question:
    Does the CLIP text encoder's embedding vector actually encode polarity
    information — i.e., does it distinguish "X exists" from "X does not exist"
    even when the sentence contains identical words?

Two experiments:

Experiment A — Layer-wise Pairwise Cosine Distance:
    For each counterfactual pair (T_XY, T_YX) with identical word sets,
    measure cos(g_l(T_XY), g_l(T_YX)) at every layer l.
    If negation binding is being built by Self-Attention:
        → cos should DECREASE as layers deepen (vectors diverge).
    If CLIP is purely Bag-of-Words:
        → cos ≈ 1.0 at all layers (vectors stay identical).

Experiment B — Per-Object Polarity Probe:
    For each object O, collect:
        Class 1 (Affirmed): sentences where O IS the present entity.
        Class 0 (Negated):  sentences where O is the ABSENT/negated entity.
    Both classes contain the word O equally → no lexical shortcut possible.
    Fit a layer-wise logistic regression; high accuracy (>65%) proves a
    linear polarity direction exists in embedding space.

Usage:
    python -m benchmarks.src.evaluation.eval_negation_existence_probe \\
        --csv_path benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv \\
        --output_dir logs/evaluation/negation_existence_probe \\
        --model ViT-B-32 --pretrained openai
"""

import os
import argparse
import json
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import open_clip

from benchmarks.src.evaluation.eval_layerwise_linear_probe import (
    extract_layerwise_feature_dict,
)


# ============================================================
# Experiment A: Layer-wise Pairwise Cosine Distance
# ============================================================
def run_pairwise_distance_analysis(
    model,
    tokenizer,
    pos_texts: List[str],
    neg_texts: List[str],
    device: str,
    output_dir: str,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """
    For every counterfactual pair (T_XY, T_YX), compute the cosine similarity
    between the two embeddings at each Transformer layer.

    Falling cosine similarity as layers deepen = negation binding is being built.
    Cosine ≈ 1.0 everywhere = pure Bag-of-Words, no negation encoding.
    """
    print("=" * 65)
    print("Experiment A: Layer-wise Pairwise Cosine Distance")
    print(f"  N pairs : {len(pos_texts)}")
    print("  Hypothesis: cos(T_XY, T_YX) should DECREASE as layers deepen")
    print("              if Self-Attention encodes negation binding.")
    print("=" * 65)

    assert len(pos_texts) == len(neg_texts), \
        "pos_texts and neg_texts must have the same length (paired)"

    print("\n  Extracting T_XY features (layer-wise)...")
    pos_features = extract_layerwise_feature_dict(
        model, tokenizer, pos_texts, device, "eot", batch_size
    )
    print("  Extracting T_YX features (layer-wise)...")
    neg_features = extract_layerwise_feature_dict(
        model, tokenizer, neg_texts, device, "eot", batch_size
    )

    layer_names = list(pos_features.keys())
    results = []

    print("\n  Computing pairwise cosine similarities per layer:")
    for l_name in layer_names:
        v_pos = torch.tensor(pos_features[l_name], dtype=torch.float32)
        v_neg = torch.tensor(neg_features[l_name], dtype=torch.float32)

        # Pairwise cos similarity for matched pairs
        cos_vals = F.cosine_similarity(v_pos, v_neg, dim=-1).numpy()  # (N,)

        mean_cos = float(np.mean(cos_vals))
        std_cos = float(np.std(cos_vals))
        mean_dist = float(np.mean(1.0 - cos_vals))   # cosine distance

        results.append({
            "layer": l_name,
            "mean_cosine_similarity": mean_cos,
            "std_cosine_similarity": std_cos,
            "mean_cosine_distance": mean_dist,
        })
        print(f"  [{l_name:25s}] cos = {mean_cos:.4f} ± {std_cos:.4f}  "
              f"(dist = {mean_dist:.4f})")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "expA_layerwise_pairwise_cosine_distance.csv")
    df.to_csv(csv_path, index=False)

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = list(range(len(layer_names)))
    cos_vals_mean = df["mean_cosine_similarity"].values
    cos_vals_std = df["std_cosine_similarity"].values
    dist_vals = df["mean_cosine_distance"].values

    # Left: Cosine Similarity (should fall)
    ax = axes[0]
    ax.plot(x, cos_vals_mean, "o-", color="#e74c3c", lw=2.5, ms=7)
    ax.fill_between(x, cos_vals_mean - cos_vals_std, cos_vals_mean + cos_vals_std,
                    color="#e74c3c", alpha=0.15)
    ax.axhline(y=1.0, color="gray", ls="--", lw=1.5, alpha=0.5, label="BoW Ceiling (cos=1.0)")
    ax.set_ylabel("Cosine Similarity (T_XY, T_YX)", fontsize=12)
    ax.set_xlabel("Transformer Layer / Pipeline Step", fontsize=12)
    ax.set_title("Exp A: Pairwise Cosine Similarity\n(↓ means negation binding is built)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(max(0, min(cos_vals_mean) - 0.05), 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.4)

    # Right: Cosine Distance (should rise)
    ax = axes[1]
    ax.plot(x, dist_vals, "o-", color="#2980b9", lw=2.5, ms=7)
    ax.axhline(y=0.0, color="gray", ls="--", lw=1.5, alpha=0.5, label="BoW Floor (dist=0.0)")
    ax.set_ylabel("Cosine Distance = 1 − cos(T_XY, T_YX)", fontsize=12)
    ax.set_xlabel("Transformer Layer / Pipeline Step", fontsize=12)
    ax.set_title("Exp A: Pairwise Cosine Distance\n(↑ means negation binding is built)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(-0.01, max(dist_vals) + 0.05)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.4)

    plt.suptitle(
        "CLIP Text Encoder: Does Negation Binding Emerge Across Layers?\n"
        f"(N={len(pos_texts)} counterfactual pairs, identical word sets)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "expA_layerwise_pairwise_cosine_distance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ── Interpretation ──
    embed_cos = float(df[df["layer"] == "Embedding"]["mean_cosine_similarity"].values[0]) \
        if "Embedding" in df["layer"].values else cos_vals_mean[0]
    final_cos = cos_vals_mean[-1]
    drop = embed_cos - final_cos

    verdict = "BINDING_FOUND" if drop > 0.02 else "NO_BINDING"
    print(f"\n  Embedding cos     : {embed_cos:.4f}")
    print(f"  Final layer cos   : {final_cos:.4f}")
    print(f"  Total drop        : {drop:.4f}")
    print(f"  Verdict           : {verdict}")
    print(f"  Saved: {csv_path}")
    print(f"  Saved: {plot_path}")

    return {
        "verdict": verdict,
        "embedding_cosine_similarity": embed_cos,
        "final_cosine_similarity": final_cos,
        "total_cosine_drop": drop,
        "per_layer": df.to_dict(orient="records"),
    }


# ============================================================
# Experiment B: Per-Object Polarity Probe
# ============================================================
def run_per_object_polarity_probe(
    model,
    tokenizer,
    df_unique: pd.DataFrame,
    device: str,
    output_dir: str,
    min_samples_per_class: int = 10,
    n_splits: int = 5,
    batch_size: int = 256,
    fit_intercept: bool = True,
) -> Dict[str, Any]:
    """
    For each object O, split all sentences containing O into:
        Class 1 (y=1, Affirmed):  sentences where O is the PRESENT entity.
        Class 0 (y=0, Negated):   sentences where O is the ABSENT entity.

    Both classes contain the word O → zero lexical shortcut.
    High linear probe accuracy → a polarity direction exists in embedding space.
    """
    print("\n" + "=" * 65)
    print("Experiment B: Per-Object Polarity Probe (Macro-Average)")
    print(f"  Min samples per class : {min_samples_per_class}")
    print(f"  Fit Intercept / Bias  : {fit_intercept}")
    print("  Hypothesis: Accuracy >> 50% → polarity direction exists in embedding space.")
    print("=" * 65)

    # ── Collect sentences per object ──
    obj_affirmed: Dict[str, List[str]] = {}
    obj_negated: Dict[str, List[str]] = {}

    for _, row in df_unique.iterrows():
        # Support both object_a/b columns and comma-split object_name
        if "object_a" in row.index and pd.notna(row.get("object_a")):
            a = str(row["object_a"]).strip()
            b = str(row["object_b"]).strip()
        else:
            parts = [s.strip() for s in str(row["object_name"]).split(",")]
            a, b = parts[0], parts[1]

        pos_cap = str(row["positive_caption"]).strip()
        neg_cap = str(row["negative_caption"]).strip()

        # A is affirmed in pos_cap, negated in neg_cap
        obj_affirmed.setdefault(a, []).append(pos_cap)
        obj_negated.setdefault(a, []).append(neg_cap)
        # B is negated in pos_cap, affirmed in neg_cap
        obj_affirmed.setdefault(b, []).append(neg_cap)
        obj_negated.setdefault(b, []).append(pos_cap)

    valid_objects = sorted([
        obj for obj in obj_affirmed
        if len(obj_affirmed[obj]) >= min_samples_per_class
        and len(obj_negated[obj]) >= min_samples_per_class
    ])

    print(f"\n  Total unique objects  : {len(obj_affirmed)}")
    print(f"  Valid objects (>={min_samples_per_class}): {len(valid_objects)}")
    print(f"  Object list           : {valid_objects}\n")

    if not valid_objects:
        print("  [Warning] No objects met the min_samples threshold. Skipping Exp B.")
        return {"error": "no_valid_objects"}

    # ── Extract features for all unique sentences ──
    all_sents: List[str] = []
    seen = set()
    for obj in valid_objects:
        for s in obj_affirmed[obj] + obj_negated[obj]:
            if s not in seen:
                all_sents.append(s)
                seen.add(s)
    sent_idx = {s: i for i, s in enumerate(all_sents)}

    print(f"  Extracting layer-wise features for {len(all_sents)} unique sentences...")
    global_feats = extract_layerwise_feature_dict(
        model, tokenizer, all_sents, device, "eot", batch_size
    )
    layer_names = list(global_feats.keys())
    print(f"  Layers extracted: {layer_names}\n")

    # ── Per-object probe (Train vs Val across layers) ──
    raw_records = []
    per_obj_rows = []
    layer_macro_train: Dict[str, List[float]] = {l: [] for l in layer_names}
    layer_macro_val: Dict[str, List[float]] = {l: [] for l in layer_names}

    print("  Running per-object logistic regression probes (Train vs Val)...")
    for obj in valid_objects:
        aff_sents = obj_affirmed[obj]
        neg_sents = obj_negated[obj]
        n = min(len(aff_sents), len(neg_sents))
        aff_idx = [sent_idx[s] for s in aff_sents[:n]]
        neg_idx = [sent_idx[s] for s in neg_sents[:n]]
        y = np.array([1] * n + [0] * n)

        row = {"object": obj, "n_per_class": n, "total": n * 2}
        eff_splits = max(2, min(n_splits, n))

        for l_name in layer_names:
            X = np.vstack([
                global_feats[l_name][aff_idx],
                global_feats[l_name][neg_idx],
            ])
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

            cv = StratifiedKFold(n_splits=eff_splits, shuffle=True, random_state=42)
            train_fold_scores = []
            val_fold_scores = []

            for tr_idx, val_idx in cv.split(X_norm, y):
                X_tr, y_tr = X_norm[tr_idx], y[tr_idx]
                X_val, y_val = X_norm[val_idx], y[val_idx]

                clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, fit_intercept=fit_intercept)
                clf.fit(X_tr, y_tr)

                train_fold_scores.append(float(clf.score(X_tr, y_tr)) * 100.0)
                val_fold_scores.append(float(clf.score(X_val, y_val)) * 100.0)

            train_acc = float(np.mean(train_fold_scores))
            val_acc = float(np.mean(val_fold_scores))

            row[f"{l_name}_train"] = train_acc
            row[f"{l_name}_val"] = val_acc
            row[l_name] = val_acc

            layer_macro_train[l_name].append(train_acc)
            layer_macro_val[l_name].append(val_acc)

            raw_records.append({
                "object_name": obj,
                "layer_name": l_name,
                "n_pairs": n,
                "train_acc_pct": train_acc,
                "val_acc_pct": val_acc,
                "gap_pct": train_acc - val_acc,
            })

        final_key = "Final (L2 Normed)" if "Final (L2 Normed)" in layer_names else layer_names[-1]
        print(f"    [{obj:15s}] n={n:3d}  Final Train Acc: {row[f'{final_key}_train']:.1f}% | Val Acc: {row[f'{final_key}_val']:.1f}%")
        per_obj_rows.append(row)

    df_raw = pd.DataFrame(raw_records)
    raw_csv = os.path.join(output_dir, "expB_per_object_train_val_records.csv")
    df_raw.to_csv(raw_csv, index=False)

    df_per_obj = pd.DataFrame(per_obj_rows)
    per_obj_csv = os.path.join(output_dir, "expB_per_object_accuracy_breakdown.csv")
    df_per_obj.to_csv(per_obj_csv, index=False)

    # ── Macro-Average per layer ──
    macro_rows = []
    for l_name in layer_names:
        tr_scores = layer_macro_train[l_name]
        val_scores = layer_macro_val[l_name]
        macro_rows.append({
            "layer": l_name,
            "train_macro_mean_pct": float(np.mean(tr_scores)),
            "train_macro_std_pct": float(np.std(tr_scores)),
            "val_macro_mean_pct": float(np.mean(val_scores)),
            "val_macro_std_pct": float(np.std(val_scores)),
            "gap_pct": float(np.mean(tr_scores) - np.mean(val_scores)),
            "min_val_object_pct": float(np.min(val_scores)),
            "max_val_object_pct": float(np.max(val_scores)),
            "n_objects": len(val_scores),
        })

    df_macro = pd.DataFrame(macro_rows)
    macro_csv = os.path.join(output_dir, "expB_layerwise_macro_accuracy.csv")
    df_macro.to_csv(macro_csv, index=False)

    # ── Render Exactly Matching Train vs Val Summary Plot ──
    render_train_val_summary_plot(df_macro, df_raw, output_dir)
    render_top_objects_grid(df_raw, output_dir, top_k=16)

    # ── Summary ──
    final_col = "Final (L2 Normed)" if "Final (L2 Normed)" in df_per_obj.columns else df_per_obj.columns[-1]
    final_macro_val = float(df_macro.iloc[-1]["val_macro_mean_pct"])
    final_macro_train = float(df_macro.iloc[-1]["train_macro_mean_pct"])
    max_macro_val = float(df_macro["val_macro_mean_pct"].max())
    argmax_layer = df_macro.loc[df_macro["val_macro_mean_pct"].idxmax(), "layer"]
    verdict = "POLARITY_EXISTS" if max_macro_val > 65.0 else "WEAK_OR_ABSENT"

    print(f"\n  Best layer (Val)      : {argmax_layer} ({max_macro_val:.2f}%)")
    print(f"  Final layer Val acc   : {final_macro_val:.2f}% (Train: {final_macro_train:.2f}%)")
    print(f"  Verdict               : {verdict}")
    print(f"  Saved: {per_obj_csv}")
    print(f"  Saved: {macro_csv}")

    df_sorted = df_per_obj.sort_values(by=final_col, ascending=True)
    return {
        "verdict": verdict,
        "n_valid_objects": len(valid_objects),
        "best_layer": argmax_layer,
        "best_layer_macro_accuracy_pct": max_macro_val,
        "final_layer_val_macro_accuracy_pct": final_macro_val,
        "final_layer_train_macro_accuracy_pct": final_macro_train,
        "per_layer_macro": macro_rows,
        "top5_objects": df_sorted.tail(5)[["object", final_col]].to_dict(orient="records"),
        "bottom5_objects": df_sorted.head(5)[["object", final_col]].to_dict(orient="records"),
    }


def render_train_val_summary_plot(df_macro: pd.DataFrame, raw_df: pd.DataFrame, output_dir: str) -> None:
    """
    Render layerwise Train vs Val accuracy plot with exact matching aesthetics to
    beaf_per_object_train_val_summary.png.
    """
    layer_names = df_macro["layer"].tolist()
    train_means = df_macro["train_macro_mean_pct"].values
    train_stds = df_macro["train_macro_std_pct"].values
    val_means = df_macro["val_macro_mean_pct"].values
    val_stds = df_macro["val_macro_std_pct"].values

    x = np.arange(len(layer_names))
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(x, train_means, "o-", color="#1f77b4", lw=2.5, ms=7, label="Train Acc (%) [Mean]")
    ax.fill_between(x, train_means - train_stds, train_means + train_stds, color="#1f77b4", alpha=0.15)

    ax.plot(x, val_means, "s--", color="#d62728", lw=2.5, ms=7, label="Val Acc (%) [Mean CV]")
    ax.fill_between(x, val_means - val_stds, val_means + val_stds, color="#d62728", alpha=0.15)

    # Post-Transformer vertical line
    post_keys = ["Layer 12 + LN", "Projected (Unnorm)", "Pre-Projection"]
    for pk in post_keys:
        if pk in layer_names:
            idx = layer_names.index(pk)
            ax.axvline(x=idx - 0.5, color="gray", ls=":", lw=1.5, alpha=0.7, label="Post-Transformer Transformations")
            break

    ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=12)
    ax.set_xlabel("Text Transformer Layer / Pipeline Step", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(45, 105)
    ax.grid(True, ls="--", alpha=0.4)

    n_objs = len(raw_df["object_name"].unique())
    ax.set_title(
        f"Layerwise Linear Probe: Train vs Val Accuracy (Mean ± Std across {n_objs} Objects)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "beaf_per_object_train_val_summary.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    # Also save as layerwise_per_object_polarity_train_val.png for clarity
    alt_path = os.path.join(output_dir, "layerwise_per_object_train_val_summary.png")
    plt.savefig(alt_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def render_top_objects_grid(raw_df: pd.DataFrame, output_dir: str, top_k: int = 16) -> None:
    """Render a grid of subplots showing Train vs Val Accuracy per layer for top-k objects by sample count."""
    import math
    obj_counts = raw_df.groupby("object_name")["n_pairs"].first().sort_values(ascending=False)
    top_objects = obj_counts.head(top_k).index.tolist()

    cols = 4
    rows = math.ceil(len(top_objects) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.5 * rows), sharex=True, sharey=True)
    axes = axes.flatten()

    layer_names = raw_df["layer_name"].unique().tolist()
    x = np.arange(len(layer_names))

    for i, obj in enumerate(top_objects):
        ax = axes[i]
        sub = raw_df[raw_df["object_name"] == obj].set_index("layer_name").reindex(layer_names)
        n_pairs = int(sub["n_pairs"].iloc[0])

        ax.plot(x, sub["train_acc_pct"], "o-", color="#1f77b4", lw=2, ms=5, label="Train Acc")
        ax.plot(x, sub["val_acc_pct"], "s--", color="#d62728", lw=2, ms=5, label="Val Acc")

        ax.set_title(f"{obj} (N={n_pairs} pairs)", fontsize=11, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.3)
        ax.set_ylim(35, 105)

        if i % cols == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=10)
        if i >= (rows - 1) * cols:
            ax.set_xticks(x)
            ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=8)

    for j in range(len(top_objects), len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=11)
    plt.suptitle(f"Top-{len(top_objects)} Objects: Layerwise Train vs Val Accuracy", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_grid = os.path.join(output_dir, "beaf_top_objects_train_val_grid.png")
    plt.savefig(out_grid, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_grid}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="CLIP Text Embedding Negation Existence Probe"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/evaluation/negation_existence_probe",
    )
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--min_samples", type=int, default=10,
                        help="Min sentences per class for Experiment B")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--no_bias", "--no-bias", action="store_true", default=False,
                        help="Disable bias/intercept in linear probes (default: bias enabled)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║   CLIP Text Embedding Negation Existence Probe            ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"  Model      : {args.model} ({args.pretrained}) | Device: {device}")
    print(f"  CSV        : {args.csv_path}")
    print(f"  Output     : {args.output_dir}")
    print(f"  Use Bias   : {not args.no_bias}\n")

    # ── Load Data ──
    df = pd.read_csv(args.csv_path)
    if df["object_in_image"].dtype == object:
        df["object_in_image"] = df["object_in_image"].apply(
            lambda x: str(x).strip().lower() == "true"
        )
    else:
        df["object_in_image"] = df["object_in_image"].astype(bool)

    # Unique caption pairs (deduplicated)
    df_unique = (
        df[df["object_in_image"] == True]
        .drop_duplicates(subset=["positive_caption", "negative_caption"])
        .reset_index(drop=True)
    )
    pos_texts = df_unique["positive_caption"].tolist()
    neg_texts = df_unique["negative_caption"].tolist()
    print(f"  Loaded {len(df_unique)} unique counterfactual caption pairs.\n")

    # ── Load Model ──
    print(f"Loading CLIP '{args.model}' ({args.pretrained})...")
    model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    full_report: Dict[str, Any] = {
        "model": args.model,
        "pretrained": args.pretrained,
        "csv_path": args.csv_path,
        "n_pairs": len(df_unique),
        "use_bias": not args.no_bias,
    }

    # ── Experiment A ──
    exp_a = run_pairwise_distance_analysis(
        model, tokenizer, pos_texts, neg_texts,
        device, args.output_dir, args.batch_size,
    )
    full_report["experiment_A_pairwise_distance"] = exp_a

    # ── Experiment B ──
    exp_b = run_per_object_polarity_probe(
        model, tokenizer, df_unique,
        device, args.output_dir,
        min_samples_per_class=args.min_samples,
        batch_size=args.batch_size,
        fit_intercept=not args.no_bias,
    )
    full_report["experiment_B_per_object_polarity"] = exp_b

    # ── Combined Verdict ──
    print("\n" + "=" * 65)
    print("  FINAL VERDICT")
    print("=" * 65)
    a_pass = exp_a.get("total_cosine_drop", 0) > 0.02
    b_pass = exp_b.get("best_layer_macro_accuracy_pct", 0) > 65.0

    if a_pass and b_pass:
        verdict = "STRONG: Negation information IS encoded in CLIP text embeddings."
    elif a_pass or b_pass:
        verdict = "WEAK: Partial evidence of negation encoding. See individual results."
    else:
        verdict = "NEGATIVE: No clear negation encoding found in text embeddings."

    full_report["combined_verdict"] = verdict
    print(f"  Exp A (Cosine Drop > 0.02) : {'PASS ✓' if a_pass else 'FAIL ✗'}")
    print(f"  Exp B (Max Acc > 65%)      : {'PASS ✓' if b_pass else 'FAIL ✗'}")
    print(f"  Combined                   : {verdict}")

    # ── Save Report ──
    report_path = os.path.join(args.output_dir, "full_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Full report : {report_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
