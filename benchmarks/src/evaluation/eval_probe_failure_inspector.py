"""
Vision and Text Linear Probe Failure Inspector Module.

Executes exact validated probing pipelines matching:
  - Vision: beaf_per_obj_test4 (~70.5% Val Acc via GroupKFold on 1:1 counterfactuals)
  - Text: negation_existence_probe2 (75.35% Val Acc via StratifiedKFold on diverse counterfactuals)

Collects and exports all Out-of-Fold (OOF) failure cases (misclassified samples):
  - vision_probing_failures.csv (misclassified images, true label, predicted prob, loss margin)
  - text_probing_failures.csv (misclassified captions, polarity, predicted prob, negation cue)
  - top_failed_objects_breakdown.csv (per-object vision vs text error rate comparison)
  - 1_vision_train_val_summary.png (reproduced vision curve)
  - 1_text_train_val_summary.png (reproduced text curve)
  - fig_probe_failures_by_object.png (error rate by object)
  - fig_text_failure_patterns_by_cue.png (error rate by negation syntax pattern)

Usage:
    python -m benchmarks.src.evaluation.eval_probe_failure_inspector \\
        --output_dir logs/evaluation/probe_failure_inspection \\
        --model ViT-B-32 --pretrained openai
"""

import os
import json
import argparse
import math
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import open_clip

# Reuse existing verified infrastructure
from benchmarks.src.analysis.config import get_layer_features as _get_feats, set_seed, coerce_bool_column
from benchmarks.src.analysis.feature_cache import build_provenance, load_object_restriction
from benchmarks.src.analysis.model_loader import load_clip_for_eval
from benchmarks.src.analysis.beaf.beaf_loader import load_and_verify_counterfactual_pairs
from benchmarks.src.analysis.beaf.vision_mechanisms import extract_vision_features_unified
from benchmarks.src.evaluation.eval_layerwise_linear_probe import extract_layerwise_feature_dict


# ============================================================
# 1. Vision Probing Pipeline & Failure Collector
# ============================================================
def run_vision_probing_and_inspect_failures(
    model,
    preprocess,
    csv_path: str,
    image_root: str,
    device: str,
    output_dir: str,
    min_pairs: int = 20,
    batch_size: int = 64,
    seed: int = 42,
    restrict_objects: List[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes exact GroupKFold linear probing on 1:1 counterfactual image pairs
    and extracts Out-of-Fold (OOF) misclassified images.
    Filters objects with fewer than min_pairs counterfactual pairs.

    ``restrict_objects`` additionally pins the evaluated concepts to an explicit
    list. A threshold alone cannot reproduce another experiment's concept set here,
    because the vision and text sides read different CSVs and apply different
    criteria (``min_pairs`` on image pairs vs ``min_samples`` per caption class),
    so the two sides land on different objects for the same threshold.
    """
    print("\n" + "=" * 65)
    print("  [Vision Probing & Failure Inspection]")
    print(f"  CSV        : {csv_path}")
    print(f"  Image Root : {image_root}")
    print(f"  Min Pairs  : {min_pairs}")
    print("=" * 65)

    df_raw, df_pairs, pair_metadata = load_and_verify_counterfactual_pairs(csv_path, image_root)
    if restrict_objects is not None:
        keep = set(restrict_objects)
        missing = sorted(keep - set(df_pairs["object_name"].unique().tolist()))
        df_pairs = df_pairs[df_pairs["object_name"].isin(keep)].reset_index(drop=True)
        print(f"  Restricted to {df_pairs['object_name'].nunique()} concepts"
              + (f" ({len(missing)} requested but absent: {missing[:5]})" if missing else ""))
    n_pairs = len(df_pairs)
    print(f"  Loaded and verified {n_pairs} counterfactual image pairs.")

    print("  Extracting layerwise vision features...")
    vis_orig = extract_vision_features_unified(model, preprocess, df_pairs["orig_path"].tolist(), device, batch_size)
    vis_cf = extract_vision_features_unified(model, preprocess, df_pairs["cf_path"].tolist(), device, batch_size)

    # Filter out missing images if any
    flags_orig = np.array(vis_orig.get("loaded_flags", [True] * n_pairs))
    flags_cf = np.array(vis_cf.get("loaded_flags", [True] * n_pairs))
    valid_mask = flags_orig & flags_cf

    df_pairs = df_pairs[valid_mask].reset_index(drop=True)
    for k in vis_orig["layers"]:
        vis_orig["layers"][k] = vis_orig["layers"][k][valid_mask]
        vis_cf["layers"][k] = vis_cf["layers"][k][valid_mask]
    vis_orig["pre_proj"] = vis_orig["pre_proj"][valid_mask]
    vis_cf["pre_proj"] = vis_cf["pre_proj"][valid_mask]
    vis_orig["final_l2norm"] = vis_orig["final_l2norm"][valid_mask]
    vis_cf["final_l2norm"] = vis_cf["final_l2norm"][valid_mask]

    layer_keys = list(vis_orig["layers"].keys())
    all_keys = layer_keys + ["Pre-Projection", "+Final L2Norm"]

    object_names = df_pairs["object_name"].values
    pair_ids = df_pairs["pair_id"].values if "pair_id" in df_pairs.columns else np.arange(len(df_pairs))
    unique_objects = sorted(df_pairs["object_name"].unique().tolist())

    raw_records = []
    failure_records = []

    print(f"  Processing objects with >= {min_pairs} pairs across {len(all_keys)} vision layers...")

    for obj in unique_objects:
        mask = (object_names == obj)
        n_obj = int(np.sum(mask))
        obj_pairs = pair_ids[mask]
        sub_df = df_pairs[mask].reset_index(drop=True)

        if n_obj < min_pairs:
            continue

        groups_all = np.concatenate([obj_pairs, obj_pairs])
        n_unique_groups = len(np.unique(obj_pairs))
        n_folds = min(5, n_unique_groups)

        for lk in all_keys:
            X_o = _get_feats(vis_orig, lk)[mask]
            X_c = _get_feats(vis_cf, lk)[mask]

            X_o_n = X_o / (np.linalg.norm(X_o, axis=1, keepdims=True) + 1e-8)
            X_c_n = X_c / (np.linalg.norm(X_c, axis=1, keepdims=True) + 1e-8)

            X_all = np.vstack([X_o_n, X_c_n])
            y_all = np.array([1] * n_obj + [0] * n_obj)

            # Metadata for failure tracking
            sample_paths = sub_df["orig_path"].tolist() + sub_df["cf_path"].tolist()
            sample_types = ["orig_present"] * n_obj + ["cf_absent"] * n_obj
            sample_pair_ids = obj_pairs.tolist() + obj_pairs.tolist()

            if n_folds < 2:
                train_acc = 50.0
                val_acc = 50.0
                best_c = 0.1
            else:
                gkf = GroupKFold(n_splits=n_folds)
                cv_splits = list(gkf.split(X_all, y_all, groups=groups_all))
                c_candidates = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

                train_scores = []
                val_scores = []
                best_c_list = []

                for tr_idx, val_idx in cv_splits:
                    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
                    groups_tr = groups_all[tr_idx]

                    # Inner CV for C selection
                    unique_tr_groups = len(np.unique(groups_tr))
                    if unique_tr_groups >= 2:
                        n_inner = min(3, unique_tr_groups)
                        inner_gkf = GroupKFold(n_splits=n_inner)
                        inner_cv = list(inner_gkf.split(X_tr, y_tr, groups=groups_tr))

                        best_c_fold = c_candidates[0]
                        best_inner_score = -1.0

                        for c in c_candidates:
                            inner_scores = []
                            for in_tr, in_val in inner_cv:
                                clf_in = LogisticRegression(C=c, max_iter=1000, random_state=seed)
                                clf_in.fit(X_tr[in_tr], y_tr[in_tr])
                                inner_scores.append(clf_in.score(X_tr[in_val], y_tr[in_val]))
                            mean_in = float(np.mean(inner_scores))
                            if mean_in > best_inner_score:
                                best_inner_score = mean_in
                                best_c_fold = c
                    else:
                        best_c_fold = 0.1

                    clf = LogisticRegression(C=best_c_fold, max_iter=1000, random_state=seed)
                    clf.fit(X_tr, y_tr)
                    train_scores.append(clf.score(X_tr, y_tr))
                    val_scores.append(clf.score(X_all[val_idx], y_all[val_idx]))
                    best_c_list.append(best_c_fold)

                    # ── Record Out-of-Fold Failures for Final L2Norm Layer ──
                    if lk == "+Final L2Norm":
                        val_probs = clf.predict_proba(X_all[val_idx])[:, 1]
                        val_preds = clf.predict(X_all[val_idx])
                        for idx_in_val, orig_idx in enumerate(val_idx):
                            y_true = y_all[orig_idx]
                            y_pred = val_preds[idx_in_val]
                            p_present = val_probs[idx_in_val]
                            if y_true != y_pred:
                                loss_margin = abs(p_present - y_true)
                                failure_records.append({
                                    "object_name": obj,
                                    "pair_id": sample_pair_ids[orig_idx],
                                    "image_type": sample_types[orig_idx],
                                    "image_path": sample_paths[orig_idx],
                                    "true_label": int(y_true),
                                    "pred_label": int(y_pred),
                                    "error_type": "False_Negative" if y_true == 1 else "False_Positive",
                                    "prob_present": float(p_present),
                                    "loss_margin": float(loss_margin),
                                    "best_c": float(best_c_fold),
                                })

                train_acc = float(np.mean(train_scores) * 100)
                val_acc = float(np.mean(val_scores) * 100)
                best_c = float(np.median(best_c_list))

            raw_records.append({
                "object_name": obj,
                "layer_name": lk,
                "n_pairs": n_obj,
                "train_acc_pct": train_acc,
                "val_acc_pct": val_acc,
                "gap_pct": train_acc - val_acc,
                "best_c": best_c,
            })

    df_vision_stats = pd.DataFrame(raw_records)
    df_vision_failures = pd.DataFrame(failure_records)

    # Sort failures by loss margin descending (worst mistakes first)
    if not df_vision_failures.empty:
        df_vision_failures = df_vision_failures.sort_values(by="loss_margin", ascending=False).reset_index(drop=True)

    # Render Vision Summary Plot (Exact match to test4)
    _render_vision_summary_plot(df_vision_stats, output_dir)

    # Save CSVs
    vis_csv = os.path.join(output_dir, "beaf_vision_per_object_layerwise.csv")
    df_vision_stats.to_csv(vis_csv, index=False)
    fail_csv = os.path.join(output_dir, "vision_probing_failures.csv")
    df_vision_failures.to_csv(fail_csv, index=False)

    print(f"  [Vision Complete] Total failures captured: {len(df_vision_failures)} samples.")
    print(f"  Saved: {vis_csv}")
    print(f"  Saved: {fail_csv}")
    return df_vision_stats, df_vision_failures


def _render_vision_summary_plot(raw_df: pd.DataFrame, output_dir: str):
    """Render overall layerwise mean Train Acc vs Val Acc plot across all objects."""
    layer_names = raw_df["layer_name"].unique().tolist()
    train_means, train_stds = [], []
    val_means, val_stds = [], []

    for lk in layer_names:
        sub = raw_df[raw_df["layer_name"] == lk]
        train_means.append(sub["train_acc_pct"].mean())
        train_stds.append(sub["train_acc_pct"].std())
        val_means.append(sub["val_acc_pct"].mean())
        val_stds.append(sub["val_acc_pct"].std())

    x = np.arange(len(layer_names))
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(x, train_means, "o-", color="#1f77b4", lw=2.5, ms=7, label="Train Acc (%) [Mean]")
    ax.fill_between(x, np.array(train_means) - np.array(train_stds), np.array(train_means) + np.array(train_stds), color="#1f77b4", alpha=0.15)

    ax.plot(x, val_means, "s--", color="#d62728", lw=2.5, ms=7, label="Val Acc (%) [Mean CV]")
    ax.fill_between(x, np.array(val_means) - np.array(val_stds), np.array(val_means) + np.array(val_stds), color="#d62728", alpha=0.15)

    if "Pre-Projection" in layer_names:
        idx = layer_names.index("Pre-Projection")
        ax.axvline(x=idx - 0.5, color="gray", ls=":", lw=1.5, alpha=0.7, label="Post-Transformer Transformations")

    ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=12)
    ax.set_xlabel("Vision Transformer Layer / Pipeline Step", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(45, 105)
    ax.grid(True, ls="--", alpha=0.4)

    n_objs = len(raw_df["object_name"].unique())
    ax.set_title(f"Layerwise Linear Probe: Train vs Val Accuracy (Mean ± Std across {n_objs} Objects)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "1_vision_train_val_summary.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Plot: {out_path}")


# ============================================================
# 2. Text Probing Pipeline & Failure Collector
# ============================================================
def run_text_probing_and_inspect_failures(
    model,
    tokenizer,
    csv_path: str,
    device: str,
    output_dir: str,
    min_samples_per_class: int = 20,
    n_splits: int = 5,
    batch_size: int = 256,
    seed: int = 42,
    restrict_objects: List[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes StratifiedKFold linear probing on diverse counterfactual caption pairs
    and extracts Out-of-Fold (OOF) misclassified captions.
    """
    print("\n" + "=" * 65)
    print("  [Text Probing & Failure Inspection]")
    print(f"  CSV : {csv_path}")
    print("=" * 65)

    df = pd.read_csv(csv_path)
    coerce_bool_column(df, "object_in_image")

    df_unique = df[df["object_in_image"] == True].drop_duplicates(
        subset=["positive_caption", "negative_caption"]
    ).reset_index(drop=True)
    print(f"  Loaded {len(df_unique)} unique diverse caption pairs.")

    obj_affirmed: Dict[str, List[Tuple[str, str]]] = {}
    obj_negated: Dict[str, List[Tuple[str, str]]] = {}

    for _, row in df_unique.iterrows():
        a = str(row["object_a"]).strip() if "object_a" in row.index else str(row["object_name"]).split(",")[0].strip()
        b = str(row["object_b"]).strip() if "object_b" in row.index else str(row["object_name"]).split(",")[1].strip()
        pos_cap = str(row["positive_caption"]).strip()
        neg_cap = str(row["negative_caption"]).strip()
        tpl = str(row.get("source_template", "unknown"))

        obj_affirmed.setdefault(a, []).append((pos_cap, tpl))
        obj_negated.setdefault(a, []).append((neg_cap, tpl))
        obj_affirmed.setdefault(b, []).append((neg_cap, tpl))
        obj_negated.setdefault(b, []).append((pos_cap, tpl))

    valid_objects = sorted([
        obj for obj in obj_affirmed
        if len(obj_affirmed[obj]) >= min_samples_per_class and len(obj_negated[obj]) >= min_samples_per_class
    ])
    print(f"  Valid objects (>={min_samples_per_class}): {len(valid_objects)}")
    if restrict_objects is not None:
        keep = set(restrict_objects)
        missing = sorted(keep - set(valid_objects))
        valid_objects = [o for o in valid_objects if o in keep]
        print(f"  Restricted to {len(valid_objects)} concepts"
              + (f" ({len(missing)} requested but absent or below threshold: {missing[:5]})" if missing else ""))

    all_sents = []
    seen = set()
    for obj in valid_objects:
        for cap, _ in obj_affirmed[obj] + obj_negated[obj]:
            if cap not in seen:
                all_sents.append(cap)
                seen.add(cap)
    sent_idx = {s: i for i, s in enumerate(all_sents)}

    print(f"  Extracting layerwise features for {len(all_sents)} unique sentences...")
    global_feats = extract_layerwise_feature_dict(model, tokenizer, all_sents, device, "eot", batch_size)
    layer_names = list(global_feats.keys())

    raw_records = []
    failure_records = []

    for obj in valid_objects:
        aff_items = obj_affirmed[obj]
        neg_items = obj_negated[obj]
        n = min(len(aff_items), len(neg_items))
        aff_items = aff_items[:n]
        neg_items = neg_items[:n]

        aff_sents = [c for c, _ in aff_items]
        neg_sents = [c for c, _ in neg_items]
        aff_tpls = [t for _, t in aff_items]
        neg_tpls = [t for _, t in neg_items]

        aff_idx = [sent_idx[s] for s in aff_sents]
        neg_idx = [sent_idx[s] for s in neg_sents]
        y = np.array([1] * n + [0] * n)
        all_sents_obj = aff_sents + neg_sents
        all_tpls_obj = aff_tpls + neg_tpls

        eff_splits = max(2, min(n_splits, n))

        for l_name in layer_names:
            X = np.vstack([global_feats[l_name][aff_idx], global_feats[l_name][neg_idx]])
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

            cv = StratifiedKFold(n_splits=eff_splits, shuffle=True, random_state=seed)
            c_candidates = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

            train_fold_scores = []
            val_fold_scores = []
            best_c_list = []

            for tr_idx, val_idx in cv.split(X_norm, y):
                X_tr, y_tr = X_norm[tr_idx], y[tr_idx]
                X_val, y_val = X_norm[val_idx], y[val_idx]

                # Inner CV for optimal C selection in text probe
                min_class_count = min(np.sum(y_tr == 1), np.sum(y_tr == 0))
                if min_class_count >= 2:
                    n_inner = min(3, min_class_count)
                    inner_cv = list(StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed).split(X_tr, y_tr))
                    best_c_fold = c_candidates[0]
                    best_inner_score = -1.0

                    for c in c_candidates:
                        inner_scores = []
                        for in_tr, in_val in inner_cv:
                            clf_in = LogisticRegression(C=c, max_iter=1000, random_state=seed)
                            clf_in.fit(X_tr[in_tr], y_tr[in_tr])
                            inner_scores.append(clf_in.score(X_tr[in_val], y_tr[in_val]))
                        mean_in = float(np.mean(inner_scores))
                        if mean_in > best_inner_score:
                            best_inner_score = mean_in
                            best_c_fold = c
                else:
                    best_c_fold = 1.0

                clf = LogisticRegression(C=best_c_fold, max_iter=1000, random_state=seed)
                clf.fit(X_tr, y_tr)

                train_fold_scores.append(float(clf.score(X_tr, y_tr)) * 100.0)
                val_fold_scores.append(float(clf.score(X_val, y_val)) * 100.0)
                best_c_list.append(best_c_fold)

                # ── Record Out-of-Fold Failures for Final Layer ──
                if l_name == "Final (L2 Normed)":
                    val_probs = clf.predict_proba(X_val)[:, 1]
                    val_preds = clf.predict(X_val)
                    for idx_in_val, orig_idx in enumerate(val_idx):
                        y_true = y[orig_idx]
                        y_pred = val_preds[idx_in_val]
                        p_affirmed = val_probs[idx_in_val]
                        if y_true != y_pred:
                            caption_str = all_sents_obj[orig_idx]
                            # Identify negation cue
                            neg_cue = "none"
                            for cue in ["no", "not", "without", "lacking", "lacks", "absent", "free of", "free-of", "does not"]:
                                if cue in caption_str.lower():
                                    neg_cue = cue
                                    break

                            loss_margin = abs(p_affirmed - y_true)
                            failure_records.append({
                                "object_name": obj,
                                "caption_text": caption_str,
                                "template": all_tpls_obj[orig_idx],
                                "true_polarity": int(y_true),
                                "pred_polarity": int(y_pred),
                                "error_type": "False_Negated" if y_true == 1 else "False_Affirmed",
                                "prob_affirmed": float(p_affirmed),
                                "loss_margin": float(loss_margin),
                                "negation_cue": neg_cue,
                                "best_c": float(best_c_fold),
                            })

            train_acc = float(np.mean(train_fold_scores))
            val_acc = float(np.mean(val_fold_scores))
            best_c = float(np.median(best_c_list))

            raw_records.append({
                "object_name": obj,
                "layer_name": l_name,
                "n_pairs": n,
                "train_acc_pct": train_acc,
                "val_acc_pct": val_acc,
                "gap_pct": train_acc - val_acc,
                "best_c": best_c,
            })

    df_text_stats = pd.DataFrame(raw_records)
    df_text_failures = pd.DataFrame(failure_records)

    if not df_text_failures.empty:
        df_text_failures = df_text_failures.sort_values(by="loss_margin", ascending=False).reset_index(drop=True)

    # Render Text Summary Plot (Exact match to probe2)
    _render_text_summary_plot(df_text_stats, output_dir)

    # Save CSVs
    txt_csv = os.path.join(output_dir, "beaf_text_per_object_layerwise.csv")
    df_text_stats.to_csv(txt_csv, index=False)
    fail_csv = os.path.join(output_dir, "text_probing_failures.csv")
    df_text_failures.to_csv(fail_csv, index=False)

    print(f"  [Text Complete] Total failures captured: {len(df_text_failures)} samples.")
    print(f"  Saved: {txt_csv}")
    print(f"  Saved: {fail_csv}")
    return df_text_stats, df_text_failures


def _render_text_summary_plot(raw_df: pd.DataFrame, output_dir: str):
    """Render overall layerwise mean Train Acc vs Val Acc plot across all objects."""
    layer_names = raw_df["layer_name"].unique().tolist()
    train_means, train_stds = [], []
    val_means, val_stds = [], []

    for lk in layer_names:
        sub = raw_df[raw_df["layer_name"] == lk]
        train_means.append(sub["train_acc_pct"].mean())
        train_stds.append(sub["train_acc_pct"].std())
        val_means.append(sub["val_acc_pct"].mean())
        val_stds.append(sub["val_acc_pct"].std())

    x = np.arange(len(layer_names))
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(x, train_means, "o-", color="#1f77b4", lw=2.5, ms=7, label="Train Acc (%) [Mean]")
    ax.fill_between(x, np.array(train_means) - np.array(train_stds), np.array(train_means) + np.array(train_stds), color="#1f77b4", alpha=0.15)

    ax.plot(x, val_means, "s--", color="#d62728", lw=2.5, ms=7, label="Val Acc (%) [Mean CV]")
    ax.fill_between(x, np.array(val_means) - np.array(val_stds), np.array(val_means) + np.array(val_stds), color="#d62728", alpha=0.15)

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
    ax.set_title(f"Layerwise Linear Probe: Train vs Val Accuracy (Mean ± Std across {n_objs} Objects)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "1_text_train_val_summary.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Plot: {out_path}")


# ============================================================
# 3. Failure Analytics & Visualizations
# ============================================================
def generate_failure_analytics(
    df_vis_stats: pd.DataFrame,
    df_txt_stats: pd.DataFrame,
    df_vis_fail: pd.DataFrame,
    df_txt_fail: pd.DataFrame,
    output_dir: str,
    provenance: Dict[str, Any] = None,
):
    """Generate joint breakdown tables and failure distribution plots."""
    print("\n" + "=" * 65)
    print("  [Generating Failure Analytics & Comparative Plots]")
    print("=" * 65)

    # Final Layer Val Accuracy per object
    v_final = df_vis_stats[df_vis_stats["layer_name"] == "+Final L2Norm"].set_index("object_name")["val_acc_pct"]
    t_final = df_txt_stats[df_txt_stats["layer_name"] == "Final (L2 Normed)"].set_index("object_name")["val_acc_pct"]

    common_objects = sorted(list(set(v_final.index).intersection(set(t_final.index))))

    comparison_rows = []
    for obj in common_objects:
        v_acc = float(v_final.get(obj, np.nan))
        t_acc = float(t_final.get(obj, np.nan))
        comparison_rows.append({
            "object_name": obj,
            "vision_val_acc_pct": v_acc,
            "text_val_acc_pct": t_acc,
            "vision_error_rate_pct": 100.0 - v_acc,
            "text_error_rate_pct": 100.0 - t_acc,
            "n_vision_failures": len(df_vis_fail[df_vis_fail["object_name"] == obj]) if not df_vis_fail.empty else 0,
            "n_text_failures": len(df_txt_fail[df_txt_fail["object_name"] == obj]) if not df_txt_fail.empty else 0,
        })

    df_comp = pd.DataFrame(comparison_rows).sort_values(by="vision_error_rate_pct", ascending=False)
    comp_csv = os.path.join(output_dir, "top_failed_objects_breakdown.csv")
    df_comp.to_csv(comp_csv, index=False)
    print(f"  Saved Breakdown Table: {comp_csv}")

    # ── Figure: Top-20 Failed Objects Comparison ──
    top20 = df_comp.head(20)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(top20))
    width = 0.35

    ax.bar(x - width/2, top20["vision_error_rate_pct"], width, label="Vision Error Rate (%)", color="#e74c3c")
    ax.bar(x + width/2, top20["text_error_rate_pct"], width, label="Text Error Rate (%)", color="#3498db")

    ax.set_ylabel("Error Rate (%) [Higher = More Failures]", fontsize=12)
    ax.set_title("Top-20 Challenging Objects: Vision vs Text Probing Error Rates", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(top20["object_name"], rotation=40, ha="right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.legend(fontsize=11)
    plt.tight_layout()

    out_fig1 = os.path.join(output_dir, "fig_probe_failures_by_object.png")
    plt.savefig(out_fig1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_fig1}")

    # ── Figure: Text Failures by Negation Cue ──
    if not df_txt_fail.empty:
        cue_counts = df_txt_fail["negation_cue"].value_counts()
        fig, ax = plt.subplots(figsize=(9, 5))
        cue_counts.plot(kind="bar", color="#9b59b6", edgecolor="black", ax=ax)
        ax.set_ylabel("Number of Text Classification Failures", fontsize=12)
        ax.set_xlabel("Negation Cue / Syntax Pattern in Caption", fontsize=12)
        ax.set_title("Distribution of Text Probing Failures across Negation Cues", fontsize=13, fontweight="bold")
        ax.set_xticklabels(cue_counts.index, rotation=30, ha="right", fontsize=10)
        ax.grid(axis="y", ls="--", alpha=0.4)
        for i, v in enumerate(cue_counts):
            ax.text(i, v + 1, str(v), ha="center", fontweight="bold", fontsize=10)
        plt.tight_layout()

        out_fig2 = os.path.join(output_dir, "fig_text_failure_patterns_by_cue.png")
        plt.savefig(out_fig2, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_fig2}")

    # ── Full Summary JSON Report ──
    # Each modality's own mean is taken over its own object set (they differ in size),
    # so a table that puts the two side by side needs the common-set mean as well.
    v_common = v_final.loc[common_objects] if common_objects else v_final.iloc[:0]
    t_common = t_final.loc[common_objects] if common_objects else t_final.iloc[:0]
    vis_n = df_vis_stats[df_vis_stats["layer_name"] == "+Final L2Norm"]["n_pairs"]
    txt_n = df_txt_stats[df_txt_stats["layer_name"] == "Final (L2 Normed)"]["n_pairs"]

    summary_report = {
        "concept_sets": {
            "n_vision_objects": int(v_final.shape[0]),
            "n_text_objects": int(t_final.shape[0]),
            "n_common_objects": len(common_objects),
            "median_pairs_per_vision_object": float(vis_n.median()) if len(vis_n) else None,
            "median_samples_per_text_object": float(txt_n.median()) if len(txt_n) else None,
            "common_objects": common_objects,
        },
        "harmonized_common_set": {
            "vision_mean_val_accuracy_pct": float(v_common.mean()) if len(v_common) else None,
            "text_mean_val_accuracy_pct": float(t_common.mean()) if len(t_common) else None,
        },
        "vision_summary": {
            "total_failures": len(df_vis_fail),
            "mean_val_accuracy_pct": float(df_vis_stats[df_vis_stats["layer_name"] == "+Final L2Norm"]["val_acc_pct"].mean()),
            "false_positive_count": int(len(df_vis_fail[df_vis_fail["error_type"] == "False_Positive"])) if not df_vis_fail.empty else 0,
            "false_negative_count": int(len(df_vis_fail[df_vis_fail["error_type"] == "False_Negative"])) if not df_vis_fail.empty else 0,
            "worst_failed_objects": df_comp.head(5)[["object_name", "vision_val_acc_pct"]].to_dict(orient="records"),
        },
        "text_summary": {
            "total_failures": len(df_txt_fail),
            "mean_val_accuracy_pct": float(df_txt_stats[df_txt_stats["layer_name"] == "Final (L2 Normed)"]["val_acc_pct"].mean()),
            "false_affirmed_count": int(len(df_txt_fail[df_txt_fail["error_type"] == "False_Affirmed"])) if not df_txt_fail.empty else 0,
            "false_negated_count": int(len(df_txt_fail[df_txt_fail["error_type"] == "False_Negated"])) if not df_txt_fail.empty else 0,
            "worst_failed_objects": df_comp.sort_values(by="text_error_rate_pct", ascending=False).head(5)[["object_name", "text_val_acc_pct"]].to_dict(orient="records"),
        },
        "provenance": provenance,
    }
    report_json = os.path.join(output_dir, "probe_failure_comprehensive_report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"  Saved Full Summary JSON: {report_json}")


# ============================================================
# Main CLI Orchestration
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Vision and Text Linear Probe Failure Inspector")
    parser.add_argument("--vision_csv", type=str, default="benchmarks/data/images/beaf_counterfactual_6col.csv")
    parser.add_argument("--text_csv", type=str, default="benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv")
    parser.add_argument("--image_root", type=str, default="benchmarks/data/images")
    parser.add_argument("--output_dir", type=str, default="logs/evaluation/probe_failure_inspection")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--min_pairs", type=int, default=20, help="Minimum counterfactual pairs per object for vision probing (default: 20)")
    parser.add_argument("--min_samples", type=int, default=20, help="Minimum samples per class for text probing (default: 20)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restrict_objects", type=str, default=None,
                        help="Comma list, or path to txt/csv/json, pinning both probes to an exact "
                             "concept set (e.g. E2's 33-concept table, so probe accuracies and "
                             "decomposition coefficients describe the same population)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    restrict = load_object_restriction(args.restrict_objects)

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  Vision & Text Linear Probe Failure Inspector             ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"  Model       : {args.model} ({args.pretrained}) | Device: {device}")
    print(f"  Output Dir  : {args.output_dir}")
    print(f"  Min Pairs   : {args.min_pairs}")
    print(f"  Min Samples : {args.min_samples}\n")

    # Load Model
    print(f"Loading CLIP '{args.model}' ({args.pretrained})...")
    model, preprocess, tokenizer = load_clip_for_eval(
        args.model, args.pretrained, device)

    # 1. Vision Probing & Failures
    df_vis_stats, df_vis_fail = run_vision_probing_and_inspect_failures(
        model, preprocess, args.vision_csv, args.image_root, device, args.output_dir,
        min_pairs=args.min_pairs, batch_size=args.batch_size, seed=args.seed,
        restrict_objects=restrict,
    )

    # 2. Text Probing & Failures
    df_txt_stats, df_txt_fail = run_text_probing_and_inspect_failures(
        model, tokenizer, args.text_csv, device, args.output_dir,
        min_samples_per_class=args.min_samples, batch_size=args.batch_size, seed=args.seed,
        restrict_objects=restrict,
    )

    # 3. Failure Analytics & Plots
    generate_failure_analytics(
        df_vis_stats, df_txt_stats, df_vis_fail, df_txt_fail, args.output_dir,
        provenance=build_provenance(args),
    )

    print("\n" + "=" * 65)
    print("  Failure Inspection Complete!")
    print(f"  Inspect Failures CSV : {args.output_dir}/vision_probing_failures.csv")
    print(f"  Inspect Failures CSV : {args.output_dir}/text_probing_failures.csv")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
