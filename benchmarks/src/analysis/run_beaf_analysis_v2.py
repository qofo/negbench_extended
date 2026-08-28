"""
BEAF Counterfactual Unified Analysis Script (v1+v2 Unified CLI Entrypoint).

Analyzes CLIP visual and text encoder mechanisms using paired counterfactual image data.
This script coordinates:
  Part A — 4-Axis BEAF Analysis (Text-Text, Image-Text, Image-Image, 4-Way Cross)
  Part B — Vision Encoder Mechanism Analyses (Layer breakdown, SVD sweep, Linear probe, Direction preservation)

Usage:
  python -m analysis.run_beaf_analysis_v2 \\
      --csv_path beaf_counterfactual_6col.csv \\
      --image_root "" \\
      --output_dir logs/evaluation/beaf_counterfactual_v2/openai_vit_b32 \\
      --model ViT-B-32 \\
      --pretrained openai
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from PIL import Image

import open_clip
from analysis.config import (
    MetadataKey,
    RetrievalConfig,
    l2_normalize,
    batch_cosine_similarity,
    filter_vision_dict,
)
from analysis.extractor import extract_all_features_unified
from analysis.metrics import (
    compute_pipeline_and_layer_breakdown,
    compute_image_text_retrieval_metrics,
)
from analysis.beaf import (
    render_image_image_histogram,
    render_4way_heatmap,
    render_text_vs_visual_scatter,
    render_full_correct_by_object,
    render_scatter_pos_vs_neg,
    render_scatter_delta_quadrant,
    render_scatter_img_orig_vs_img_cf,
    render_scatter_by_object_category,
    render_2x2_factorial_anova_plots,
    render_2d_margin_state_space,
    extract_vision_features_unified,
    compute_vision_pipeline_breakdown,
    compute_vision_svd_sweep,
    compute_vision_linear_probe,
    compute_vision_direction_preservation,
    compute_2x2_factorial_anova,
    compute_quadrant_bootstrap_ci,
    load_and_verify_counterfactual_pairs,
    compute_per_object_layerwise_stats,
    render_per_object_layerwise_plot,
)



# =========================================================================== #
# Image Encoding Helper
# =========================================================================== #



def _encode_image_paths(
    paths: List[str],
    model,
    preprocess,
    device: str,
    batch_size: int,
    fallback_dim: int = 512,
) -> Tuple[np.ndarray, List[bool]]:
    """Encode image file paths with CLIP visual encoder."""
    model.eval()
    all_embs: List[np.ndarray] = []
    loaded_flags: List[bool] = []

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        tensors: List[torch.Tensor] = []
        valid_positions: List[int] = []

        for j, p in enumerate(batch_paths):
            if not os.path.exists(p):
                loaded_flags.append(False)
                continue
            try:
                t = preprocess(Image.open(p).convert("RGB"))
                tensors.append(t)
                valid_positions.append(j)
                loaded_flags.append(True)
            except Exception as ex:
                print(f"  [Warning] Cannot load {p}: {ex}")
                loaded_flags.append(False)

        if tensors:
            stacked = torch.stack(tensors, dim=0).to(device)
            with torch.no_grad():
                batch_embs = model.encode_image(stacked, normalize=True).float().cpu()
            embed_dim = batch_embs.shape[1]
            placeholder = torch.zeros(len(batch_paths), embed_dim)
            for vp_idx, pos in enumerate(valid_positions):
                placeholder[pos] = batch_embs[vp_idx]
            all_embs.append(placeholder.numpy())
        else:
            all_embs.append(np.zeros((len(batch_paths), fallback_dim), dtype=np.float32))

    return np.concatenate(all_embs, axis=0), loaded_flags


def _classify_failure_mode(A: float, B: float, C: float, D: float) -> str:
    """Classify a 2x2 similarity quad into mutually exclusive outcome categories based on a 3-bit diagnostic truth table.

    2x2 Matrix:
        caption=pos  caption=neg
    image=orig   A            B
    image=cf     C            D

    3 Diagnostic Conditions:
        text_ok   : A > B  - model scores orig image higher with positive caption than negative
        visual_ok : A > C  - model distinguishes object-present from object-absent image
        cf_coh_ok : D > C  - counterfactual image scores higher with negative caption

    Mutually Exclusive Categories (3-bit Truth Table):
        PASS                 : text_ok=T, visual_ok=T, cf_coh_ok=T (all 3 pass)
        FAIL_CF_COHERENCE    : text_ok=T, visual_ok=T, cf_coh_ok=F (only counterfactual coherence fails)
        FAIL_VISUAL_ONLY     : text_ok=T, visual_ok=F, cf_coh_ok=T (only visual discrimination fails)
        FAIL_TEXT_ONLY       : text_ok=F, visual_ok=T, cf_coh_ok=T (only text negation fails)
        FAIL_TEXT_AND_CF     : text_ok=F, visual_ok=T, cf_coh_ok=F (both text and counterfactual fail)
        FAIL_VISUAL_AND_CF   : text_ok=T, visual_ok=F, cf_coh_ok=F (both visual and counterfactual fail)
        FAIL_BOTH            : text_ok=F, visual_ok=F, cf_coh_ok=T (both text and visual fail, cf passes)
        FAIL_ALL             : text_ok=F, visual_ok=F, cf_coh_ok=F (all 3 fail)
    """
    text_ok   = bool(A > B)
    visual_ok = bool(A > C)
    cf_coh_ok = bool(D > C)

    if text_ok and visual_ok and cf_coh_ok:
        return "PASS"
    elif text_ok and visual_ok and not cf_coh_ok:
        return "FAIL_CF_COHERENCE"
    elif text_ok and not visual_ok and cf_coh_ok:
        return "FAIL_VISUAL_ONLY"
    elif not text_ok and visual_ok and cf_coh_ok:
        return "FAIL_TEXT_ONLY"
    elif not text_ok and visual_ok and not cf_coh_ok:
        return "FAIL_TEXT_AND_CF"
    elif text_ok and not visual_ok and not cf_coh_ok:
        return "FAIL_VISUAL_AND_CF"
    elif not text_ok and not visual_ok and cf_coh_ok:
        return "FAIL_BOTH"
    else:  # not text_ok and not visual_ok and not cf_coh_ok
        return "FAIL_ALL"


def compute_image_image_cosine(
    cf_pairs: pd.DataFrame,
    model,
    preprocess,
    device: str,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Compute sim(encode_image(orig), encode_image(cf)) for each counterfactual pair (Axis 3)."""
    model.eval()
    orig_embs, orig_loaded = _encode_image_paths(cf_pairs["orig_path"].tolist(), model, preprocess, device, batch_size)
    cf_embs, cf_loaded     = _encode_image_paths(cf_pairs["cf_path"].tolist(),   model, preprocess, device, batch_size)

    sims = []
    for i in range(len(cf_pairs)):
        if orig_loaded[i] and cf_loaded[i]:
            o = torch.from_numpy(orig_embs[i : i + 1])
            c = torch.from_numpy(cf_embs[i : i + 1])
            sims.append(float(F.cosine_similarity(o, c).item()))
        else:
            sims.append(float("nan"))

    result_df = cf_pairs.copy()
    result_df["sim_img_img"] = sims
    result_df["orig_loaded"] = orig_loaded
    result_df["cf_loaded"]   = cf_loaded
    return result_df


def compute_4way_matrix(
    cf_pairs: pd.DataFrame,
    model,
    tokenizer,
    preprocess,
    device: str,
    text_batch_size: int = 256,
    img_batch_size: int = 32,
) -> pd.DataFrame:
    """Compute four cross-modal similarities per quad (Axis 4)."""
    model.eval()

    def _encode_texts(texts: List[str]) -> torch.Tensor:
        embs = []
        for start in range(0, len(texts), text_batch_size):
            tokens = tokenizer(texts[start : start + text_batch_size]).to(device)
            with torch.no_grad():
                embs.append(model.encode_text(tokens, normalize=True).float().cpu())
        return torch.cat(embs, dim=0)

    pos_embs   = _encode_texts(cf_pairs["positive_caption"].tolist())
    neg_embs   = _encode_texts(cf_pairs["negative_caption"].tolist())
    orig_np, orig_loaded = _encode_image_paths(cf_pairs["orig_path"].tolist(), model, preprocess, device, img_batch_size)
    cf_np, cf_loaded     = _encode_image_paths(cf_pairs["cf_path"].tolist(),   model, preprocess, device, img_batch_size)

    A_list, B_list, C_list, D_list = [], [], [], []
    text_neg_correct, visual_cf_correct, cf_text_correct, full_correct, failure_mode = [], [], [], [], []

    for i in range(len(cf_pairs)):
        if orig_loaded[i] and cf_loaded[i]:
            o  = torch.from_numpy(orig_np[i : i + 1])
            c  = torch.from_numpy(cf_np[i : i + 1])
            pt = pos_embs[i : i + 1]
            nt = neg_embs[i : i + 1]
            A = float(F.cosine_similarity(o, pt).item())
            B = float(F.cosine_similarity(o, nt).item())
            C = float(F.cosine_similarity(c, pt).item())
            D = float(F.cosine_similarity(c, nt).item())
        else:
            A = B = C = D = float("nan")
        A_list.append(A); B_list.append(B); C_list.append(C); D_list.append(D)
        if not any(np.isnan([A, B, C, D])):
            text_neg_correct.append(bool(A > B))
            visual_cf_correct.append(bool(A > C))
            cf_text_correct.append(bool(D > C))
            full_correct.append(bool(A > B and D > C and A > C and D > B))
            failure_mode.append(_classify_failure_mode(A, B, C, D))
        else:
            text_neg_correct.append(None)
            visual_cf_correct.append(None)
            cf_text_correct.append(None)
            full_correct.append(None)
            failure_mode.append("LOAD_ERROR")

    result = cf_pairs.copy()
    result["A_sim_orig_pos"]       = A_list
    result["B_sim_orig_neg"]       = B_list
    result["C_sim_cf_pos"]         = C_list
    result["D_sim_cf_neg"]         = D_list
    result["text_negation_score"]  = [a - b if not np.isnan(a) else float("nan") for a, b in zip(A_list, B_list)]
    result["visual_change_score"]  = [a - c if not np.isnan(a) else float("nan") for a, c in zip(A_list, C_list)]
    result["cf_coherence_score"]   = [d - c if not np.isnan(d) else float("nan") for d, c in zip(D_list, C_list)]
    result["text_neg_correct"]     = text_neg_correct
    result["visual_cf_correct"]    = visual_cf_correct
    result["cf_text_correct"]      = cf_text_correct
    result["full_correct"]         = full_correct
    result["failure_mode"]         = failure_mode
    return result


# =========================================================================== #
# Main Pipeline Entrypoint
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="BEAF Counterfactual Unified Analysis Pipeline (v1 4-Axis + v2 Vision Deep-dive)"
    )
    parser.add_argument("--csv_path",    type=str, required=True)
    parser.add_argument("--image_root",  type=str, default="")
    parser.add_argument("--output_dir",  type=str, default="logs/evaluation/beaf_counterfactual_v2/openai_vit_b32")
    parser.add_argument("--model",       type=str, default="ViT-B-32")
    parser.add_argument("--pretrained",  type=str, default="openai")
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--img_batch",   type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0, help="Cap number of CSV rows loaded (0 = no cap)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--run_full_4axis", action="store_true", default=False,
                        help="Enable full 4-Axis analyses (Direction Preservation, Image-Image Cosine, 4-Way Matrix). Default: ANOVA-only fast run.")
    parser.add_argument("--no_bias", "--no-bias", action="store_true", default=False,
                        help="Disable bias/intercept in linear probes (default: bias enabled)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  BEAF Counterfactual Unified Analysis Pipeline (v1+v2)")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Model      : {args.model} ({args.pretrained})")
    print(f"  CSV        : {args.csv_path}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Use Bias   : {not args.no_bias}")
    print("=" * 60)

    # 1. Load Data
    print("\n[Step 1] Loading BEAF CSV & Verifying Pair Integrity ...")
    df_raw, df_pairs, pair_metadata = load_and_verify_counterfactual_pairs(args.csv_path, args.image_root)
    if args.max_samples > 0:
        df_pairs      = df_pairs.head(args.max_samples).copy()
        pair_metadata = pair_metadata[:args.max_samples * 2]
    n_pairs = len(df_pairs)
    print(f"  Raw rows                                 : {len(df_raw)}")
    print(f"  Unified Verified Pairs (orig <-> cf)    : {n_pairs}")

    # 2. Load Model
    print("\n[Step 2] Loading OpenCLIP Model ...")
    # create_model_and_transforms returns (model, preprocess_train, preprocess_val).
    # Feature extraction must take the *val* transform: the train one is
    # RandomResizedCrop + augmentation, which makes embeddings non-deterministic
    # and not comparable with the evaluation/ scripts.
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)
    model.eval()
    print("  Model loaded successfully.")

    # 3. Extract Text Features
    print("\n[Step 3] Extracting Text Features (Part B) ...")
    pos_texts = df_pairs["positive_caption"].tolist()
    neg_texts = df_pairs["negative_caption"].tolist()

    pos_feat_dict = extract_all_features_unified(model, tokenizer, pos_texts, device, "eot", args.batch_size)
    neg_feat_dict = extract_all_features_unified(model, tokenizer, neg_texts, device, "eot", args.batch_size)

    pos_embs = pos_feat_dict["final_l2norm"]
    neg_embs = neg_feat_dict["final_l2norm"]

    # 4. Extract Vision Features
    print("\n[Step 4] Extracting Vision Features & Filtering Missing Images ...")
    vis_orig = extract_vision_features_unified(model, preprocess, df_pairs["orig_path"].tolist(), device, args.img_batch)
    vis_cf   = extract_vision_features_unified(model, preprocess, df_pairs["cf_path"].tolist(),   device, args.img_batch)

    # Filter out missing images using loaded_flags (#2)
    flags_orig = np.array(vis_orig.get("loaded_flags", [True] * n_pairs))
    flags_cf   = np.array(vis_cf.get("loaded_flags",   [True] * n_pairs))
    valid_mask = flags_orig & flags_cf

    n_valid = int(np.sum(valid_mask))
    n_dropped = n_pairs - n_valid
    if n_dropped > 0:
        print(f"  ⚠️ [Data Integrity] Dropped {n_dropped}/{n_pairs} pairs due to missing/corrupted image files!")
        print(f"     Retained {n_valid} valid image pairs for all downstream analyses.")
    else:
        print(f"  ✅ [Data Integrity] 100% ({n_valid}/{n_pairs}) image pairs loaded successfully with 0 missing files.")

    df_pairs = df_pairs[valid_mask].reset_index(drop=True)
    pos_embs = pos_embs[valid_mask]
    neg_embs = neg_embs[valid_mask]

    vis_orig = filter_vision_dict(vis_orig, valid_mask)
    vis_cf   = filter_vision_dict(vis_cf, valid_mask)

    n_pairs = n_valid
    orig_embs = vis_orig["final_l2norm"]
    cf_embs   = vis_cf["final_l2norm"]

    all_img_embs  = np.vstack([orig_embs, cf_embs])
    all_pos_embs  = np.vstack([pos_embs, pos_embs])
    all_neg_embs  = np.vstack([neg_embs, neg_embs])
    all_obj_flags = np.array([True] * n_pairs + [False] * n_pairs)

    # 5. Render Part B Scatter Plots & Compute 2x2 Factorial ANOVA
    print("\n[Step 5] Part B — Rendering Scatter Plots & 2x2 Factorial ANOVA ...")
    render_scatter_pos_vs_neg(all_img_embs, all_pos_embs, all_neg_embs, all_obj_flags, args.output_dir)

    sim_orig_pos = batch_cosine_similarity(orig_embs, pos_embs)
    sim_orig_neg = batch_cosine_similarity(orig_embs, neg_embs)
    sim_cf_pos   = batch_cosine_similarity(cf_embs,   pos_embs)
    sim_cf_neg   = batch_cosine_similarity(cf_embs,   neg_embs)

    # 2x2 Factorial ANOVA Main Effects & Interaction Effect
    anova_df, summary_anova = compute_2x2_factorial_anova(sim_orig_pos, sim_orig_neg, sim_cf_pos, sim_cf_neg, seed=args.seed)
    anova_df.to_csv(os.path.join(args.output_dir, "beaf_2x2_factorial_anova.csv"), index=False)
    render_2x2_factorial_anova_plots(anova_df, summary_anova, args.output_dir)
    render_2d_margin_state_space(sim_orig_pos, sim_orig_neg, sim_cf_pos, sim_cf_neg, args.output_dir)

    print("     [2x2 Factorial ANOVA 95% Bootstrap CI]:")
    print(f"       - Text Main Effect   : {summary_anova['text_main_effect']['mean']:6.4f} (95% CI: [{summary_anova['text_main_effect']['ci_95_low']:6.4f}, {summary_anova['text_main_effect']['ci_95_high']:6.4f}])")
    print(f"       - Visual Main Effect : {summary_anova['visual_main_effect']['mean']:6.4f} (95% CI: [{summary_anova['visual_main_effect']['ci_95_low']:6.4f}, {summary_anova['visual_main_effect']['ci_95_high']:6.4f}])")
    print(f"       - Interaction Effect : {summary_anova['interaction_effect']['mean']:6.4f} (95% CI: [{summary_anova['interaction_effect']['ci_95_low']:6.4f}, {summary_anova['interaction_effect']['ci_95_high']:6.4f}]) | Negative Inter. = {summary_anova['interaction_effect']['negative_interaction_pct']:.1f}%")

    quad_bootstrap_ci = compute_quadrant_bootstrap_ci(sim_orig_pos, sim_orig_neg, sim_cf_pos, margin=0.01, n_bootstraps=1000, seed=args.seed)
    render_scatter_delta_quadrant(sim_orig_pos, sim_orig_neg, sim_cf_pos, args.output_dir)
    render_scatter_img_orig_vs_img_cf(sim_orig_pos, sim_cf_pos, args.output_dir)
    render_scatter_by_object_category(df_pairs, sim_orig_pos, sim_orig_neg, args.output_dir)

    # [Steps 6-10 Commented out for 2x2 ANOVA focus]
    object_names = df_pairs["object_name"].values if "object_name" in df_pairs.columns else None
    pair_ids     = df_pairs["pair_id"].values     if "pair_id"     in df_pairs.columns else None
    vis_probe = compute_vision_linear_probe(
        vis_orig, vis_cf, args.output_dir,
        object_names=object_names,
        pair_ids=pair_ids,
        seed=args.seed,
        fit_intercept=not args.no_bias,
    )

    # Per-object layerwise analysis & visualization (mean + std shaded area)
    raw_obj_df, obj_layer_summary = compute_per_object_layerwise_stats(vis_orig, vis_cf, df_pairs, seed=args.seed)
    render_per_object_layerwise_plot(obj_layer_summary, args.output_dir, raw_df=raw_obj_df)


    # Steps 6-10: Full 4-Axis analyses (optional, gated by --run_full_4axis)
    if args.run_full_4axis:
        print("\n[Steps 6-10] Running full 4-Axis analyses (Direction Preservation, Image-Image Cosine, 4-Way Matrix) ...")
        vis_dir_pres  = compute_vision_direction_preservation(vis_orig, vis_cf, args.output_dir, seed=args.seed)
        img_img_df = compute_image_image_cosine(df_pairs, model, preprocess, device, args.img_batch)
        matrix_df  = compute_4way_matrix(df_pairs, model, tokenizer, preprocess, device, args.batch_size, args.img_batch)
    else:
        print("\n[Steps 6-10] Skipped (use --run_full_4axis to enable Direction Preservation, Image-Image Cosine, 4-Way Matrix).")

    # 11. Write Comprehensive Summary Report (2x2 Factorial ANOVA Focus)
    print("\n[Step 11] Writing 2x2 Factorial ANOVA Summary Report ...")
    r_val, _   = stats.pearsonr(batch_cosine_similarity(all_img_embs, all_pos_embs),
                                batch_cosine_similarity(all_img_embs, all_neg_embs))
    rho_val, _ = stats.spearmanr(batch_cosine_similarity(all_img_embs, all_pos_embs),
                                 batch_cosine_similarity(all_img_embs, all_neg_embs))

    comprehensive_summary = {
        "model":         args.model,
        "pretrained":    args.pretrained,
        "use_bias":      not args.no_bias,
        "n_raw_rows":    len(df_raw),
        "n_exact_pairs": n_pairs,
        "part_b_scatter_pos_vs_neg": {
            "n_total":      len(all_img_embs),
            "pearson_r":    round(float(r_val), 6),
            "spearman_rho": round(float(rho_val), 6),
        },
        "part_b_delta_delta_quadrants": quad_bootstrap_ci,
        "part_b_2x2_factorial_anova":  summary_anova,
    }

    summary_path = os.path.join(args.output_dir, "beaf_comprehensive_summary_report.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(comprehensive_summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("  BEAF 2x2 Factorial ANOVA Fast Run Complete!")
    print(f"  Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
