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
    compute_vision_non_linear_probe,
    compute_vision_direction_preservation,
)


# =========================================================================== #
# Data Loaders & Data Computation Helpers
# =========================================================================== #

def load_beaf_csv(csv_path: str, image_root: str) -> Tuple[pd.DataFrame, List[dict]]:
    """Load beaf_counterfactual_6col.csv and resolve absolute image paths for Axis 1-4."""
    df = pd.read_csv(csv_path)
    if image_root:
        df["abs_image_path"] = df["image_path"].apply(lambda p: os.path.join(image_root, p))
    else:
        df["abs_image_path"] = df["image_path"]

    def _to_bool(v) -> Optional[bool]:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return None

    df["object_in_image"] = df["object_in_image"].apply(_to_bool)

    pair_metadata = []
    for _, row in df.iterrows():
        pair_metadata.append({
            MetadataKey.IMAGE_PATH.value:      row["image_path"],
            MetadataKey.OBJECT_NAME.value:     str(row.get("object_name", "")),
            MetadataKey.OBJECT_IN_IMAGE.value: row["object_in_image"],
            MetadataKey.SOURCE_TEMPLATE.value: str(row.get("source_template", "")),
        })

    return df, pair_metadata


def load_and_verify_counterfactual_pairs(csv_path: str, image_root: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
    """Load beaf_counterfactual_6col.csv, group by source_template, and enforce strict pairing integrity.
    Returns:
        (df_raw, df_pairs, pair_metadata)
    """
    df = pd.read_csv(csv_path)

    def _resolve_path(p: str, root: str) -> str:
        p_str = str(p).strip()
        if os.path.exists(p_str):
            return p_str
        if root:
            candidate1 = os.path.join(root, p_str)
            if os.path.exists(candidate1):
                return candidate1
            if p_str.startswith("data/images/"):
                candidate2 = os.path.join(root, p_str[len("data/images/"):])
                if os.path.exists(candidate2):
                    return candidate2
            if p_str.startswith("data/"):
                candidate3 = os.path.join(root, p_str[len("data/"):])
                if os.path.exists(candidate3):
                    return candidate3
        return os.path.join(root, p_str) if root else p_str

    df["abs_image_path"] = df["image_path"].apply(lambda p: _resolve_path(p, image_root))

    def _to_bool(v) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return False

    df["object_in_image"] = df["object_in_image"].apply(_to_bool)

    pair_metadata = []
    for _, row in df.iterrows():
        pair_metadata.append({
            MetadataKey.IMAGE_PATH.value:      row["image_path"],
            MetadataKey.OBJECT_NAME.value:     str(row.get("object_name", "")),
            MetadataKey.OBJECT_IN_IMAGE.value: row["object_in_image"],
            MetadataKey.SOURCE_TEMPLATE.value: str(row.get("source_template", "")),
        })

    pairs = []
    num_pairs = len(df) // 2
    for i in range(num_pairs):
        row1 = df.iloc[2 * i]
        row2 = df.iloc[2 * i + 1]

        b1 = row1["object_in_image"]
        b2 = row2["object_in_image"]

        assert (b1 and not b2) or (not b1 and b2), f"Row pair {i} object_in_image mismatch: {b1}, {b2}"

        orig_row = row1 if b1 else row2
        cf_row   = row2 if b1 else row1

        # Strict Assertion Checks (#1)
        assert orig_row["object_in_image"] == True, f"Orig row for pair {i} must have object_in_image == True"
        assert cf_row["object_in_image"] == False, f"CF row for pair {i} must have object_in_image == False"
        assert str(orig_row.get("object_name")) == str(cf_row.get("object_name")), f"Object name mismatch in pair {i}"
        assert str(orig_row.get("source_template")) == str(cf_row.get("source_template")), f"Source template mismatch in pair {i}"

        pairs.append({
            "pair_id":          i,
            "source_template":  str(orig_row.get("source_template", "")),
            "object_name":      str(orig_row.get("object_name", "")),
            "orig_path":        orig_row["abs_image_path"],
            "cf_path":          cf_row["abs_image_path"],
            "positive_caption": str(orig_row["positive_caption"]),
            "negative_caption": str(orig_row["negative_caption"]),
        })

    df_pairs = pd.DataFrame(pairs)
    print(f"  ✅ [Unified Pairing Verified] Extracted all {len(df_pairs)} exact counterfactual pairs with 100% strict assertion checks.")
    return df, df_pairs, pair_metadata


def compute_quadrant_bootstrap_ci(
    sim_orig_pos: np.ndarray,
    sim_orig_neg: np.ndarray,
    sim_cf_pos: np.ndarray,
    margin: float = 0.01,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Calculate quadrant proportions with noise margin and 95% Bootstrap Confidence Intervals (#3)."""
    delta_text = sim_orig_pos - sim_orig_neg
    delta_vis  = sim_orig_pos - sim_cf_pos
    n = len(delta_text)

    def _get_quadrants(dt, dv):
        q1 = (dt > margin) & (dv > margin)
        q2 = (dt <= margin) & (dv > margin)
        q3 = (dt <= margin) & (dv <= margin)
        q4 = (dt > margin) & (dv <= margin)
        q_near_zero = (np.abs(dt) <= margin) | (np.abs(dv) <= margin)
        return {
            "q1_both_sensitive_pct": float(np.mean(q1) * 100),
            "q2_visual_only_pct":    float(np.mean(q2) * 100),
            "q3_neither_pct":        float(np.mean(q3) * 100),
            "q4_text_only_pct":      float(np.mean(q4) * 100),
            "near_zero_margin_pct":  float(np.mean(q_near_zero) * 100),
        }

    point_estimates = _get_quadrants(delta_text, delta_vis)

    rng = np.random.default_rng(seed=seed)
    boot_dist = {k: [] for k in point_estimates}
    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        q_boot = _get_quadrants(delta_text[boot_idx], delta_vis[boot_idx])
        for k, v in q_boot.items():
            boot_dist[k].append(v)

    summary_ci = {}
    for k, v in point_estimates.items():
        low = float(np.percentile(boot_dist[k], 2.5))
        high = float(np.percentile(boot_dist[k], 97.5))
        summary_ci[k] = {
            "mean_pct": round(v, 2),
            "ci_95_low": round(low, 2),
            "ci_95_high": round(high, 2),
        }
    return summary_ci


def compute_2x2_factorial_anova(
    sim_orig_pos: np.ndarray,
    sim_orig_neg: np.ndarray,
    sim_cf_pos: np.ndarray,
    sim_cf_neg: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute 2x2 Factorial ANOVA Main Effects & Interaction Effect per sample pair with 95% Bootstrap CI.

    Raw 2x2 Similarity Matrix per pair:
                caption=pos   caption=neg
    image=orig      A (orig_pos)  B (orig_neg)
    image=cf        C (cf_pos)    D (cf_neg)

    3 Orthogonal Derived Metrics:
    - Text Main Effect    = ((A - B) + (C - D)) / 2
    - Visual Main Effect  = ((A - C) + (B - D)) / 2
    - Interaction Effect = (A - B) - (C - D) == (A - C) - (B - D)
    """
    A = sim_orig_pos
    B = sim_orig_neg
    C = sim_cf_pos
    D = sim_cf_neg

    text_main_effect   = ((A - B) + (C - D)) / 2.0
    visual_main_effect = ((A - C) + (B - D)) / 2.0
    interaction_effect = (A - B) - (C - D)

    anova_df = pd.DataFrame({
        "sim_A_orig_pos":       A,
        "sim_B_orig_neg":       B,
        "sim_C_cf_pos":         C,
        "sim_D_cf_neg":         D,
        "text_main_effect":     text_main_effect,
        "visual_main_effect":   visual_main_effect,
        "interaction_effect":   interaction_effect,
    })

    n = len(A)
    rng = np.random.default_rng(seed=seed)

    boot_t, boot_v, boot_i = [], [], []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        boot_t.append(np.mean(text_main_effect[idx]))
        boot_v.append(np.mean(visual_main_effect[idx]))
        boot_i.append(np.mean(interaction_effect[idx]))

    summary_anova = {
        "text_main_effect": {
            "mean": round(float(np.mean(text_main_effect)), 6),
            "std":  round(float(np.std(text_main_effect)), 6),
            "ci_95_low": round(float(np.percentile(boot_t, 2.5)), 6),
            "ci_95_high": round(float(np.percentile(boot_t, 97.5)), 6),
        },
        "visual_main_effect": {
            "mean": round(float(np.mean(visual_main_effect)), 6),
            "std":  round(float(np.std(visual_main_effect)), 6),
            "ci_95_low": round(float(np.percentile(boot_v, 2.5)), 6),
            "ci_95_high": round(float(np.percentile(boot_v, 97.5)), 6),
        },
        "interaction_effect": {
            "mean": round(float(np.mean(interaction_effect)), 6),
            "std":  round(float(np.std(interaction_effect)), 6),
            "ci_95_low": round(float(np.percentile(boot_i, 2.5)), 6),
            "ci_95_high": round(float(np.percentile(boot_i, 97.5)), 6),
            "negative_interaction_pct": round(float(np.mean(interaction_effect < 0) * 100), 2),
        },
    }

    return anova_df, summary_anova


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
    text_neg_correct, visual_cf_correct, cf_text_correct, full_correct = [], [], [], []

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
        else:
            text_neg_correct.append(None)
            visual_cf_correct.append(None)
            cf_text_correct.append(None)
            full_correct.append(None)

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

    pos_texts_v1 = df_raw[df_raw["object_in_image"] == True]["positive_caption"].astype(str).tolist()
    neg_texts_v1 = df_raw[df_raw["object_in_image"] == True]["negative_caption"].astype(str).tolist()

    # 2. Load Model
    print("\n[Step 2] Loading OpenCLIP Model ...")
    model, preprocess, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
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

    for k in vis_orig["layers"]:
        vis_orig["layers"][k] = vis_orig["layers"][k][valid_mask]
        vis_cf["layers"][k]   = vis_cf["layers"][k][valid_mask]
    vis_orig["pre_proj"]     = vis_orig["pre_proj"][valid_mask]
    vis_cf["pre_proj"]       = vis_cf["pre_proj"][valid_mask]
    vis_orig["final_l2norm"] = vis_orig["final_l2norm"][valid_mask]
    vis_cf["final_l2norm"]   = vis_cf["final_l2norm"][valid_mask]

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
    # object_names = df_pairs["object_name"].values if "object_name" in df_pairs.columns else None
    # vis_breakdown = compute_vision_pipeline_breakdown(vis_orig, vis_cf, args.output_dir)
    # vis_svd       = compute_vision_svd_sweep(model, vis_orig, vis_cf, args.output_dir)
    # vis_probe     = compute_vision_linear_probe(vis_orig, vis_cf, args.output_dir, object_names=object_names)
    # vis_nl_probe  = compute_vision_non_linear_probe(vis_orig, vis_cf, args.output_dir, seed=args.seed, object_names=object_names)
    # vis_dir_pres  = compute_vision_direction_preservation(vis_orig, vis_cf, args.output_dir, seed=args.seed)
    # pos_features_v1 = extract_all_features_unified(model, tokenizer, pos_texts_v1, device, "eot", args.batch_size)
    # neg_features_v1 = extract_all_features_unified(model, tokenizer, neg_texts_v1, device, "eot", args.batch_size)
    # pipeline_data = compute_pipeline_and_layer_breakdown(pos_features_v1, neg_features_v1)
    # retrieval_data = compute_image_text_retrieval_metrics(model, tokenizer, preprocess, pair_metadata, pos_texts_v1, neg_texts_v1, retrieval_cfg)
    # img_img_df = compute_image_image_cosine(df_pairs, model, preprocess, device, args.img_batch)
    # matrix_df  = compute_4way_matrix(df_pairs, model, tokenizer, preprocess, device, args.batch_size, args.img_batch)

    # 11. Write Comprehensive Summary Report (2x2 Factorial ANOVA Focus)
    print("\n[Step 11] Writing 2x2 Factorial ANOVA Summary Report ...")
    r_val, _   = stats.pearsonr(batch_cosine_similarity(all_img_embs, all_pos_embs),
                                batch_cosine_similarity(all_img_embs, all_neg_embs))
    rho_val, _ = stats.spearmanr(batch_cosine_similarity(all_img_embs, all_pos_embs),
                                 batch_cosine_similarity(all_img_embs, all_neg_embs))

    comprehensive_summary = {
        "model":         args.model,
        "pretrained":    args.pretrained,
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
    print(f"  BEAF 2x2 Factorial ANOVA Fast Run Complete!")
    print(f"  Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
