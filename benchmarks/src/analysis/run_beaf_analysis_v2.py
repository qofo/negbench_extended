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


def build_counterfactual_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Group rows by source_template and extract (original, counterfactual) image pairs."""
    pairs = []
    for tmpl, grp in df.groupby("source_template"):
        orig_rows = grp[grp["object_in_image"] == True]
        cf_rows   = grp[grp["object_in_image"] == False]
        if orig_rows.empty or cf_rows.empty:
            continue
        orig_row = orig_rows.iloc[0]
        cf_row   = cf_rows.iloc[0]
        pairs.append({
            "source_template":  tmpl,
            "object_name":      str(orig_row.get("object_name", "")),
            "orig_path":        orig_row["abs_image_path"],
            "cf_path":          cf_row["abs_image_path"],
            "positive_caption": str(orig_row["positive_caption"]),
            "negative_caption": str(orig_row["negative_caption"]),
        })
    return pd.DataFrame(pairs)


def load_beaf_paired_dataset(csv_path: str, image_root: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load beaf_counterfactual_6col.csv and construct exact 2n/2n+1 pairs."""
    df = pd.read_csv(csv_path)
    if image_root:
        df["abs_image_path"] = df["image_path"].apply(lambda p: os.path.join(image_root, p))
    else:
        df["abs_image_path"] = df["image_path"]

    def _to_bool(v) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return False

    df["object_in_image"] = df["object_in_image"].apply(_to_bool)

    pairs = []
    num_rows = len(df)
    for i in range(0, num_rows - 1, 2):
        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]

        if row1["object_in_image"] and not row2["object_in_image"]:
            orig_row, cf_row = row1, row2
        elif not row1["object_in_image"] and row2["object_in_image"]:
            orig_row, cf_row = row2, row1
        else:
            orig_row, cf_row = row1, row2

        pairs.append({
            "pair_id":          i // 2,
            "object_name":      str(orig_row.get("object_name", "")),
            "orig_path":        orig_row["abs_image_path"],
            "cf_path":          cf_row["abs_image_path"],
            "positive_caption": str(orig_row["positive_caption"]),
            "negative_caption": str(orig_row["negative_caption"]),
            "source_template":  str(orig_row.get("source_template", "")),
        })

    return df, pd.DataFrame(pairs)


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
    print("\n[Step 1] Loading BEAF CSV ...")
    df_raw, df_pairs = load_beaf_paired_dataset(args.csv_path, args.image_root)
    n_pairs = len(df_pairs)
    print(f"  Raw rows                                 : {len(df_raw)}")
    print(f"  Exact Counterfactual Pairs (orig <-> cf) : {n_pairs}")

    df_v1, pair_metadata = load_beaf_csv(args.csv_path, args.image_root)
    if args.max_samples > 0:
        df_v1         = df_v1.head(args.max_samples).copy()
        pair_metadata = pair_metadata[:args.max_samples]
    cf_pairs_v1  = build_counterfactual_pairs(df_v1)
    pos_texts_v1 = df_v1["positive_caption"].astype(str).tolist()
    neg_texts_v1 = df_v1["negative_caption"].astype(str).tolist()
    print(f"  Axis 1-4 pairs (source_template based)  : {len(cf_pairs_v1)}")

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
    print("\n[Step 4] Extracting Vision Features (Part B) ...")
    vis_orig = extract_vision_features_unified(model, preprocess, df_pairs["orig_path"].tolist(), device, args.img_batch)
    vis_cf   = extract_vision_features_unified(model, preprocess, df_pairs["cf_path"].tolist(),   device, args.img_batch)

    orig_embs = vis_orig["final_l2norm"]
    cf_embs   = vis_cf["final_l2norm"]

    all_img_embs  = np.vstack([orig_embs, cf_embs])
    all_pos_embs  = np.vstack([pos_embs, pos_embs])
    all_neg_embs  = np.vstack([neg_embs, neg_embs])
    all_obj_flags = np.array([True] * n_pairs + [False] * n_pairs)

    # 5. Render Part B Scatter Plots
    print("\n[Step 5] Part B — Rendering Scatter Plots ...")
    render_scatter_pos_vs_neg(all_img_embs, all_pos_embs, all_neg_embs, all_obj_flags, args.output_dir)

    sim_orig_pos = batch_cosine_similarity(orig_embs, pos_embs)
    sim_orig_neg = batch_cosine_similarity(orig_embs, neg_embs)
    sim_cf_pos   = batch_cosine_similarity(cf_embs,   pos_embs)

    render_scatter_delta_quadrant(sim_orig_pos, sim_orig_neg, sim_cf_pos, args.output_dir)
    render_scatter_img_orig_vs_img_cf(sim_orig_pos, sim_cf_pos, args.output_dir)
    render_scatter_by_object_category(df_pairs, sim_orig_pos, sim_orig_neg, args.output_dir)

    print("\n[Step 6] Part B — Vision Encoder Mechanism Analyses ...")
    vis_breakdown = compute_vision_pipeline_breakdown(vis_orig, vis_cf, args.output_dir)
    vis_svd       = compute_vision_svd_sweep(model, vis_orig, vis_cf, args.output_dir)
    vis_probe     = compute_vision_linear_probe(vis_orig, vis_cf, args.output_dir)
    vis_nl_probe  = compute_vision_non_linear_probe(vis_orig, vis_cf, args.output_dir, seed=args.seed)
    vis_dir_pres  = compute_vision_direction_preservation(vis_orig, vis_cf, args.output_dir, seed=args.seed)

    # 7. Part A — Axis 1: Text <-> Text
    print("\n" + "=" * 60)
    print("[Step 7] Part A — Axis 1: Text <-> Text Pipeline Breakdown")
    print("=" * 60)

    pos_features_v1 = extract_all_features_unified(model, tokenizer, pos_texts_v1, device, "eot", args.batch_size)
    neg_features_v1 = extract_all_features_unified(model, tokenizer, neg_texts_v1, device, "eot", args.batch_size)
    pipeline_data = compute_pipeline_and_layer_breakdown(pos_features_v1, neg_features_v1)

    pd.DataFrame(pipeline_data["pipeline"]).to_csv(
        os.path.join(args.output_dir, "beaf_text_text_pipeline.csv"), index=False
    )
    pd.DataFrame(pipeline_data["layers"]).to_csv(
        os.path.join(args.output_dir, "beaf_text_text_cosine.csv"), index=False
    )
    for row in pipeline_data["pipeline"]:
        print(f"    [{row['step_name']:22s}] Cosine: {row['mean_cosine_sim']:.4f}  L2: {row['mean_l2_distance']:.4f}")
    print("  Saved: beaf_text_text_pipeline.csv, beaf_text_text_cosine.csv")

    final_text_cosine = float(next(
        r["mean_cosine_sim"] for r in reversed(pipeline_data["pipeline"])
    ))

    # 8. Part A — Axis 2: Image <-> Text Retrieval
    print("\n" + "=" * 60)
    print("[Step 8] Part A — Axis 2: Image <-> Text Retrieval")
    print("=" * 60)

    retrieval_cfg = RetrievalConfig(
        image_root=args.image_root,
        output_dir=args.output_dir,
        device=device,
        batch_size=args.batch_size,
        image_batch_size=args.img_batch,
    )
    retrieval_data = compute_image_text_retrieval_metrics(
        model, tokenizer, preprocess,
        pair_metadata, pos_texts_v1, neg_texts_v1,
        retrieval_cfg,
    )

    summary_axis2: Dict[str, Any] = {}
    if retrieval_data.get("results_df") is not None:
        retrieval_data["results_df"].to_csv(
            os.path.join(args.output_dir, "beaf_image_text_similarity.csv"), index=False
        )
        summary_axis2 = retrieval_data["summary"]
        with open(os.path.join(args.output_dir, "beaf_image_text_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_axis2, f, indent=2)
        print(f"  Pearson r: {summary_axis2.get('pearson_r', 'N/A'):.4f}")
        print("  Saved: beaf_image_text_similarity.csv, beaf_image_text_summary.json")
    else:
        print("  [Warning] No images processed for Axis 2. Check image_root / file paths.")

    # 9. Part A — Axis 3: Image <-> Image Cosine
    print("\n" + "=" * 60)
    print("[Step 9] Part A — Axis 3: Image <-> Image Visual Sensitivity")
    print("=" * 60)

    img_img_df = compute_image_image_cosine(cf_pairs_v1, model, preprocess, device, args.img_batch)
    valid_img  = img_img_df.dropna(subset=["sim_img_img"])
    summary_axis3: Dict[str, Any] = {}

    if len(valid_img) > 0:
        mean_sim = valid_img["sim_img_img"].mean()
        print(f"  Pairs computed       : {len(valid_img)} / {len(cf_pairs_v1)}")
        print(f"  Mean sim(orig, cf)   : {mean_sim:.4f}")
        img_img_df.to_csv(os.path.join(args.output_dir, "beaf_image_image_cosine.csv"), index=False)
        render_image_image_histogram(img_img_df, args.output_dir)
        summary_axis3 = {
            "n_pairs":  int(len(valid_img)),
            "mean_sim": round(float(mean_sim), 6),
            "std_sim":  round(float(valid_img["sim_img_img"].std()), 6),
        }
    else:
        print("  [Warning] No valid image pairs for Axis 3.")

    # 10. Part A — Axis 4: 4-Way Cross Similarity
    print("\n" + "=" * 60)
    print("[Step 10] Part A — Axis 4: 4-Way Cross Similarity Matrix")
    print("=" * 60)

    matrix_df  = compute_4way_matrix(cf_pairs_v1, model, tokenizer, preprocess, device, args.batch_size, args.img_batch)
    valid_4way = matrix_df.dropna(subset=["full_correct"])
    summary_axis4: Dict[str, Any] = {}

    if len(valid_4way) > 0:
        fcr = float(pd.Series(valid_4way["full_correct"]).mean()) * 100
        print(f"  Full Correct Rate: {fcr:.1f}%")
        matrix_df.to_csv(os.path.join(args.output_dir, "beaf_4way_matrix.csv"), index=False)
        render_4way_heatmap(matrix_df, args.output_dir)
        render_text_vs_visual_scatter(matrix_df, args.output_dir)
        render_full_correct_by_object(matrix_df, args.output_dir)
        summary_axis4 = {
            "n_pairs":               int(len(valid_4way)),
            "full_correct_rate_pct": round(fcr, 2),
            "mean_A_sim_orig_pos":   round(float(valid_4way["A_sim_orig_pos"].mean()), 6),
            "mean_B_sim_orig_neg":   round(float(valid_4way["B_sim_orig_neg"].mean()), 6),
            "mean_C_sim_cf_pos":     round(float(valid_4way["C_sim_cf_pos"].mean()), 6),
            "mean_D_sim_cf_neg":     round(float(valid_4way["D_sim_cf_neg"].mean()), 6),
        }
    else:
        print("  [Warning] No valid pairs for Axis 4.")

    # 11. Write Comprehensive Summary Report
    print("\n[Step 11] Writing Comprehensive Summary Report ...")
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
        "part_b_delta_delta_quadrants": {
            "q1_both_sensitive_pct":    round(float(np.sum((sim_orig_pos - sim_orig_neg > 0) & (sim_orig_pos - sim_cf_pos > 0)) / n_pairs * 100), 2),
            "q2_visual_only_pct":       round(float(np.sum((sim_orig_pos - sim_orig_neg <= 0) & (sim_orig_pos - sim_cf_pos > 0)) / n_pairs * 100), 2),
            "q3_neither_pct":           round(float(np.sum((sim_orig_pos - sim_orig_neg <= 0) & (sim_orig_pos - sim_cf_pos <= 0)) / n_pairs * 100), 2),
            "q4_text_only_failure_pct": round(float(np.sum((sim_orig_pos - sim_orig_neg > 0) & (sim_orig_pos - sim_cf_pos <= 0)) / n_pairs * 100), 2),
        },
        "part_b_vision_direction_preservation": vis_dir_pres,
        "part_b_vision_linear_probe":           vis_probe,
        "axis1_text_text": {
            "final_l2norm_cosine_sim": round(final_text_cosine, 6),
            "pipeline_steps": [
                {"step": r["step_name"], "mean_cosine_sim": round(r["mean_cosine_sim"], 6)}
                for r in pipeline_data["pipeline"]
            ],
        },
        "axis2_image_text":  summary_axis2 if summary_axis2 else None,
        "axis3_image_image": summary_axis3 if summary_axis3 else None,
        "axis4_4way":        summary_axis4 if summary_axis4 else None,
    }

    summary_path = os.path.join(args.output_dir, "beaf_comprehensive_summary_report.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(comprehensive_summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  BEAF Unified Analysis Complete!")
    print(f"  All artifacts saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
