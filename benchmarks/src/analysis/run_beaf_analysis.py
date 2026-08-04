"""
BEAF Counterfactual Analysis Script.

Analyzes CLIP visual and text encoder similarity across 4 axes using
paired counterfactual image data (original image with object vs. edited image without object):

  Axis 1 — Text <-> Text:   sim(pos_text, neg_text) across all transformer layers
  Axis 2 — Image <-> Text:  sim(image, pos_text) vs sim(image, neg_text) grouped by object_in_image
  Axis 3 — Image <-> Image: sim(img_original, img_counterfactual) - visual encoder sensitivity
  Axis 4 — 4-way Cross:     A=sim(img_orig, pos), B=sim(img_orig, neg),
                             C=sim(img_cf,   pos), D=sim(img_cf,   neg)
                             -> Full Correct Rate: A>B & D>C & A>C & D>B

Outputs to --output_dir:
  beaf_text_text_cosine.csv             Axis 1 layer-wise mean cosine similarities
  beaf_text_text_pipeline.csv           Axis 1 full pipeline step breakdown
  beaf_image_text_similarity.csv        Axis 2 per-pair image-text similarities
  beaf_image_text_summary.json          Axis 2 summary (accuracy by object_in_image group)
  beaf_image_image_cosine.csv           Axis 3 per-pair original<->counterfactual image similarity
  beaf_image_image_histogram.png        Axis 3 distribution plot + per-object boxplot
  beaf_4way_matrix.csv                  Axis 4 per-pair A, B, C, D values + correctness flags
  beaf_4way_heatmap.png                 Axis 4 average similarity heatmap (2x2)
  beaf_text_vs_visual_scatter.png       Axis 4 Text Negation Score vs Visual Change Score scatter
  beaf_full_correct_rate_by_object.png  Axis 4 Full Correct Rate per object category
  beaf_summary_report.json              Overall summary metrics for all 4 axes

Usage:
  python -m benchmarks.src.analysis.run_beaf_analysis \\
      --csv_path beaf_counterfactual_6col.csv \\
      --image_root "" \\
      --output_dir logs/evaluation/beaf_counterfactual/openai_vit_b32 \\
      --model ViT-B-32 \\
      --pretrained openai
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------------------------------- #
# Path bootstrap (works both as a script and as a module)
# --------------------------------------------------------------------------- #
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS_SRC = os.path.abspath(os.path.join(_FILE_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_FILE_DIR, "..", "..", ".."))
for _p in [_BENCHMARKS_SRC, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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


# =========================================================================== #
# Helper: load and pair BEAF CSV
# =========================================================================== #

def load_beaf_csv(csv_path: str, image_root: str) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Load beaf_counterfactual_6col.csv and resolve absolute image paths.

    CSV columns expected:
        image_path, object_name, positive_caption, negative_caption,
        object_in_image, source_template

    Returns:
        df            : raw DataFrame with resolved 'abs_image_path' column added
        pair_metadata : list of dicts compatible with MetadataKey enum
    """
    df = pd.read_csv(csv_path)

    # Resolve image paths (csv path is used as-is when image_root is empty)
    if image_root:
        df["abs_image_path"] = df["image_path"].apply(
            lambda p: os.path.join(image_root, p)
        )
    else:
        df["abs_image_path"] = df["image_path"]

    # Parse object_in_image as bool
    def _to_bool(v) -> Optional[bool]:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return None

    df["object_in_image"] = df["object_in_image"].apply(_to_bool)

    # Build metadata list (compatible with compute_image_text_retrieval_metrics)
    pair_metadata = []
    for _, row in df.iterrows():
        pair_metadata.append({
            MetadataKey.IMAGE_PATH.value:     row["image_path"],       # relative - used by retrieval fn
            MetadataKey.OBJECT_NAME.value:    str(row.get("object_name", "")),
            MetadataKey.OBJECT_IN_IMAGE.value: row["object_in_image"],
            MetadataKey.SOURCE_TEMPLATE.value: str(row.get("source_template", "")),
        })

    return df, pair_metadata


def build_counterfactual_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group rows by source_template (which uniquely identifies one (image, object) experiment)
    and extract (original_image, counterfactual_image) pairs.

    Each source_template has exactly two rows:
      - object_in_image == True  -> original image (object present)
      - object_in_image == False -> counterfactual image (object removed)

    Returns a DataFrame with columns:
        source_template, object_name,
        orig_path, cf_path,
        positive_caption, negative_caption
    """
    pairs = []
    for tmpl, grp in df.groupby("source_template"):
        orig_rows = grp[grp["object_in_image"] == True]
        cf_rows   = grp[grp["object_in_image"] == False]

        if orig_rows.empty or cf_rows.empty:
            continue  # skip incomplete template groups

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


# =========================================================================== #
# Shared image-batch encoder
# =========================================================================== #

def _encode_image_paths(
    paths: List[str],
    model,
    preprocess,
    device: str,
    batch_size: int,
    fallback_dim: int = 512,
) -> Tuple[np.ndarray, List[bool]]:
    """
    Encode a list of image file paths with the CLIP visual encoder.

    Returns:
        embs         : float32 numpy array of shape (N, D); zero-vector for failed images
        loaded_flags : bool list of length N indicating which images loaded successfully
    """
    all_embs: List[np.ndarray] = []
    loaded_flags: List[bool] = []

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        tensors: List[torch.Tensor] = []
        valid_positions: List[int] = []     # positions within this batch that loaded

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

        # Infer embedding dim from first successful batch or use fallback
        if tensors:
            stacked = torch.stack(tensors, dim=0).to(device)
            with torch.no_grad():
                batch_embs = model.encode_image(stacked, normalize=True).float().cpu()
            embed_dim = batch_embs.shape[1]

            # Place embeddings at their original positions; zeros for failures
            placeholder = torch.zeros(len(batch_paths), embed_dim)
            for vp_idx, pos in enumerate(valid_positions):
                placeholder[pos] = batch_embs[vp_idx]
            all_embs.append(placeholder.numpy())
        else:
            # Entire batch failed - emit zeros with fallback dim
            all_embs.append(np.zeros((len(batch_paths), fallback_dim), dtype=np.float32))

    return np.concatenate(all_embs, axis=0), loaded_flags


# =========================================================================== #
# Axis 3: Image <-> Image cosine similarity
# =========================================================================== #

def compute_image_image_cosine(
    cf_pairs: pd.DataFrame,
    model,
    preprocess,
    device: str,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    For each (orig_path, cf_path) pair, compute:
        sim(encode_image(orig), encode_image(cf))

    This measures how differently the CLIP visual encoder represents
    the original image (object present) vs. the counterfactual (object removed).

    Returns a DataFrame extending cf_pairs with additional columns:
        sim_img_img (float), orig_loaded (bool), cf_loaded (bool)
    """
    model.eval()

    print(f"  Encoding {len(cf_pairs)} original images ...")
    orig_embs, orig_loaded = _encode_image_paths(
        cf_pairs["orig_path"].tolist(), model, preprocess, device, batch_size
    )

    print(f"  Encoding {len(cf_pairs)} counterfactual images ...")
    cf_embs, cf_loaded = _encode_image_paths(
        cf_pairs["cf_path"].tolist(), model, preprocess, device, batch_size
    )

    sims = []
    for i in range(len(cf_pairs)):
        if orig_loaded[i] and cf_loaded[i]:
            o = torch.from_numpy(orig_embs[i : i + 1])
            c = torch.from_numpy(cf_embs[i : i + 1])
            s = float(F.cosine_similarity(o, c).item())
        else:
            s = float("nan")
        sims.append(s)

    result_df = cf_pairs.copy()
    result_df["sim_img_img"] = sims
    result_df["orig_loaded"] = orig_loaded
    result_df["cf_loaded"]   = cf_loaded
    return result_df


# =========================================================================== #
# Axis 4: 4-way cross similarity matrix
# =========================================================================== #

def compute_4way_matrix(
    cf_pairs: pd.DataFrame,
    model,
    tokenizer,
    preprocess,
    device: str,
    text_batch_size: int = 256,
    img_batch_size: int = 32,
) -> pd.DataFrame:
    """
    Compute four cross-modal similarity scores per (orig_image, cf_image, pos_text, neg_text) quad:

        A = sim(img_orig,  pos_text)   [should be HIGH - object present, text affirms]
        B = sim(img_orig,  neg_text)   [should be LOW  - object present, text denies]
        C = sim(img_cf,    pos_text)   [should be LOW  - object absent,  text affirms]
        D = sim(img_cf,    neg_text)   [should be HIGH - object absent,  text denies]

    Derived scores:
        text_negation_score  = A - B  (how much CLIP prefers positive text on original image)
        visual_change_score  = A - C  (how much CLIP detects object removal on positive text)
        cf_coherence_score   = D - C  (how much CLIP prefers negative text on counterfactual)

    Correctness flags (all should be True for a well-calibrated model):
        text_neg_correct  : A > B
        visual_cf_correct : A > C
        cf_text_correct   : D > C
        full_correct      : A>B & D>C & A>C & D>B (all four constraints satisfied)

    Returns wide DataFrame with one row per source_template pair.
    """
    model.eval()

    pos_texts = cf_pairs["positive_caption"].tolist()
    neg_texts = cf_pairs["negative_caption"].tolist()

    # --- Encode texts ---
    def _encode_texts(texts: List[str]) -> torch.Tensor:
        embs = []
        for start in range(0, len(texts), text_batch_size):
            end = min(start + text_batch_size, len(texts))
            tokens = tokenizer(texts[start:end]).to(device)
            with torch.no_grad():
                e = model.encode_text(tokens, normalize=True).float().cpu()
            embs.append(e)
        return torch.cat(embs, dim=0)

    print("  Encoding positive texts ...")
    pos_embs = _encode_texts(pos_texts)  # (N, D)
    print("  Encoding negative texts ...")
    neg_embs = _encode_texts(neg_texts)  # (N, D)

    # --- Encode images (reuse shared helper) ---
    print("  Encoding original images ...")
    orig_embs_np, orig_loaded = _encode_image_paths(
        cf_pairs["orig_path"].tolist(), model, preprocess, device, img_batch_size
    )
    print("  Encoding counterfactual images ...")
    cf_embs_np, cf_loaded = _encode_image_paths(
        cf_pairs["cf_path"].tolist(), model, preprocess, device, img_batch_size
    )

    # --- Compute 4 similarity scores per row ---
    A_list, B_list, C_list, D_list = [], [], [], []
    text_neg_correct, visual_cf_correct, cf_text_correct, full_correct = [], [], [], []

    for i in range(len(cf_pairs)):
        if orig_loaded[i] and cf_loaded[i]:
            o  = torch.from_numpy(orig_embs_np[i : i + 1])
            c  = torch.from_numpy(cf_embs_np[i : i + 1])
            pt = pos_embs[i : i + 1]
            nt = neg_embs[i : i + 1]

            A = float(F.cosine_similarity(o,  pt).item())
            B = float(F.cosine_similarity(o,  nt).item())
            C = float(F.cosine_similarity(c,  pt).item())
            D = float(F.cosine_similarity(c,  nt).item())
        else:
            A = B = C = D = float("nan")

        A_list.append(A)
        B_list.append(B)
        C_list.append(C)
        D_list.append(D)

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
    result["A_sim_orig_pos"]        = A_list
    result["B_sim_orig_neg"]        = B_list
    result["C_sim_cf_pos"]          = C_list
    result["D_sim_cf_neg"]          = D_list
    result["text_negation_score"]   = [
        a - b if not np.isnan(a) else float("nan") for a, b in zip(A_list, B_list)
    ]
    result["visual_change_score"]   = [
        a - c if not np.isnan(a) else float("nan") for a, c in zip(A_list, C_list)
    ]
    result["cf_coherence_score"]    = [
        d - c if not np.isnan(d) else float("nan") for d, c in zip(D_list, C_list)
    ]
    result["text_neg_correct"]      = text_neg_correct
    result["visual_cf_correct"]     = visual_cf_correct
    result["cf_text_correct"]       = cf_text_correct
    result["full_correct"]          = full_correct
    return result


# =========================================================================== #
# Visualization helpers
# =========================================================================== #

def render_image_image_histogram(img_img_df: pd.DataFrame, output_dir: str):
    """
    Plot:
      Left  - histogram of sim(orig, cf) across all pairs
      Right - boxplot of sim(orig, cf) per object category (top-15 by count)
    """
    valid = img_img_df.dropna(subset=["sim_img_img"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: distribution histogram
    ax = axes[0]
    ax.hist(valid["sim_img_img"], bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    mean_val = valid["sim_img_img"].mean()
    ax.axvline(mean_val, color="crimson", ls="--", lw=2,
               label=f"Mean = {mean_val:.4f}")
    ax.set_xlabel("Cosine Similarity  (Original Image <-> Counterfactual Image)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        "Image Encoder Sensitivity to Object Removal\n"
        "(Distribution of Image<->Image Cosine Similarity)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.4)

    # Right: per-object boxplot
    ax2 = axes[1]
    obj_counts  = valid["object_name"].value_counts()
    top_objects = obj_counts.head(15).index.tolist()
    plot_data   = [
        valid.loc[valid["object_name"] == obj, "sim_img_img"].values
        for obj in top_objects
    ]

    bp = ax2.boxplot(plot_data, patch_artist=True, vert=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_objects)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax2.set_xticks(range(1, len(top_objects) + 1))
    ax2.set_xticklabels(top_objects, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Cosine Similarity  (orig <-> cf)", fontsize=11)
    ax2.set_title(
        "Per-Object Visual Change Sensitivity\n"
        "(Lower = Image Encoder Detects Object Removal Better)",
        fontsize=11, fontweight="bold"
    )
    ax2.grid(True, ls="--", alpha=0.4, axis="y")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "beaf_image_image_histogram.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: beaf_image_image_histogram.png")


def render_4way_heatmap(matrix_df: pd.DataFrame, output_dir: str):
    """
    Render 2x2 heatmap of average A, B, C, D similarities.
      Rows = Image state (Original with object / Counterfactual without object)
      Cols = Text type  (Positive / Negative)
    """
    valid = matrix_df.dropna(subset=["A_sim_orig_pos"])

    grid = np.array([
        [valid["A_sim_orig_pos"].mean(), valid["B_sim_orig_neg"].mean()],
        [valid["C_sim_cf_pos"].mean(),   valid["D_sim_cf_neg"].mean()],
    ])

    vmin = grid.min() - 0.005
    vmax = grid.max() + 0.005

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=vmin, vmax=vmax)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Positive Caption\n(object present in text)", "Negative Caption\n(object absent in text)"],
        fontsize=11
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(
        ["Original Image\n(object IN)", "Counterfactual\n(object OUT)"],
        fontsize=11
    )
    ax.set_title(
        "CLIP 4-Way Similarity Matrix\n(Image State x Text Polarity)",
        fontsize=12, fontweight="bold"
    )

    cell_labels = [["A", "B"], ["C", "D"]]
    for i in range(2):
        for j in range(2):
            v = grid[i, j]
            label = f"{cell_labels[i][j]}\n{v:.4f}"
            text_color = "white" if (v < vmin + (vmax - vmin) * 0.25 or v > vmax - (vmax - vmin) * 0.25) else "black"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=15, fontweight="bold", color=text_color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "beaf_4way_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: beaf_4way_heatmap.png")


def render_text_vs_visual_scatter(matrix_df: pd.DataFrame, output_dir: str):
    """
    Scatter plot:
      X-axis = Text Negation Score  (A - B): how well CLIP distinguishes pos vs neg text on orig image
      Y-axis = Visual Change Score  (A - C): how well CLIP detects object removal on pos text
      Color  = Full Correct (green) / Not Full Correct (red)
    """
    valid = matrix_df.dropna(subset=["text_negation_score", "visual_change_score", "full_correct"])

    colors = valid["full_correct"].map({True: "seagreen", False: "crimson"})

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        valid["text_negation_score"],
        valid["visual_change_score"],
        c=colors,
        alpha=0.55,
        s=28,
        edgecolors="none",
    )

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Text Negation Score  A - B  (sim_orig_pos - sim_orig_neg)", fontsize=10)
    ax.set_ylabel("Visual Change Score  A - C  (sim_orig_pos - sim_cf_pos)", fontsize=10)
    ax.set_title(
        "CLIP Sensitivity:\nText Negation vs Visual Object Removal",
        fontsize=11, fontweight="bold"
    )
    ax.grid(True, ls="--", alpha=0.35)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="seagreen", label="Full Correct (A>B & D>C & A>C & D>B)"),
        Patch(facecolor="crimson",  label="Not Full Correct"),
    ]
    ax.legend(handles=legend_els, fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "beaf_text_vs_visual_scatter.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: beaf_text_vs_visual_scatter.png")


def render_full_correct_by_object(matrix_df: pd.DataFrame, output_dir: str):
    """
    Horizontal bar chart showing Full Correct Rate (%) per object category,
    sorted ascending so worst-performing objects appear at the top.
    """
    valid = matrix_df.dropna(subset=["full_correct"])

    obj_stats = (
        valid.groupby("object_name")["full_correct"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "correct", "count": "total"})
    )
    obj_stats["rate_pct"] = obj_stats["correct"] / obj_stats["total"] * 100
    obj_stats = obj_stats.sort_values("rate_pct", ascending=True)

    overall_mean = valid["full_correct"].mean() * 100

    fig_height = max(5, len(obj_stats) * 0.38)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    bar_colors = [plt.cm.RdYlGn(r / 100) for r in obj_stats["rate_pct"]]
    bars = ax.barh(
        obj_stats.index, obj_stats["rate_pct"],
        color=bar_colors, edgecolor="white", height=0.7
    )

    # Annotate correct/total counts
    for bar, (idx, row) in zip(bars, obj_stats.iterrows()):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['correct'])}/{int(row['total'])}",
            va="center", fontsize=7.5, color="#333333"
        )

    ax.axvline(
        overall_mean, color="navy", ls="--", lw=1.5,
        label=f"Overall Mean = {overall_mean:.1f}%"
    )
    ax.set_xlabel("Full Correct Rate (%)", fontsize=11)
    ax.set_title(
        "CLIP Full Correct Rate by Object Category\n(A>B & D>C & A>C & D>B)",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlim(0, 108)
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", alpha=0.4, axis="x")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "beaf_full_correct_rate_by_object.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: beaf_full_correct_rate_by_object.png")


# =========================================================================== #
# Main pipeline
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="BEAF Counterfactual Analysis: Visual & Text Encoder Cosine Similarity"
    )
    parser.add_argument(
        "--csv_path", type=str, required=True,
        help="Path to beaf_counterfactual_6col.csv"
    )
    parser.add_argument(
        "--image_root", type=str, default="",
        help="Root directory prepended to image_path column. "
             "Leave empty ('') to use image_path column as-is."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="logs/evaluation/beaf_counterfactual/openai_vit_b32",
        help="Directory for all output artifacts"
    )
    parser.add_argument(
        "--model", type=str, default="ViT-B-32",
        help="OpenCLIP model architecture (e.g. ViT-B-32, ViT-L-14)"
    )
    parser.add_argument(
        "--pretrained", type=str, default="openai",
        help="OpenCLIP pretrained weights tag"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Text encoding batch size"
    )
    parser.add_argument(
        "--img_batch", type=int, default=32,
        help="Image encoding batch size"
    )
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="Cap number of CSV rows loaded (0 = no cap; useful for quick tests)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (reserved for future stochastic components)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  BEAF Counterfactual Analysis Pipeline")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Model      : {args.model} ({args.pretrained})")
    print(f"  CSV        : {args.csv_path}")
    print(f"  Image root : {args.image_root or '<use csv path as-is>'}")
    print(f"  Output dir : {args.output_dir}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Step 0: Load CSV & build counterfactual pairs
    # ------------------------------------------------------------------ #
    print("\n[Step 0] Loading BEAF CSV ...")
    df, pair_metadata = load_beaf_csv(args.csv_path, args.image_root)

    if args.max_samples > 0:
        df            = df.head(args.max_samples).copy()
        pair_metadata = pair_metadata[:args.max_samples]

    pos_texts = df["positive_caption"].astype(str).tolist()
    neg_texts = df["negative_caption"].astype(str).tolist()
    n_templates = df["source_template"].nunique()
    print(f"  Rows loaded          : {len(df)}")
    print(f"  Unique templates     : {n_templates}")
    print(f"  Unique object names  : {df['object_name'].nunique()}")

    cf_pairs = build_counterfactual_pairs(df)
    print(f"  Counterfactual pairs : {len(cf_pairs)}")

    # ------------------------------------------------------------------ #
    # Step 1: Load CLIP model
    # ------------------------------------------------------------------ #
    print("\n[Step 1] Loading CLIP model ...")
    model, preprocess, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)
    model.eval()
    print("  Model loaded successfully.")

    # ================================================================== #
    # Axis 1: Text <-> Text (reuse existing extractor + pipeline breakdown)
    # ================================================================== #
    print("\n" + "=" * 60)
    print("Axis 1: Text <-> Text — Layer-wise & Pipeline Cosine Similarity")
    print("=" * 60)

    pos_features = extract_all_features_unified(
        model, tokenizer, pos_texts, device, "eot", args.batch_size
    )
    neg_features = extract_all_features_unified(
        model, tokenizer, neg_texts, device, "eot", args.batch_size
    )

    pipeline_data = compute_pipeline_and_layer_breakdown(pos_features, neg_features)

    df_pipeline = pd.DataFrame(pipeline_data["pipeline"])
    df_pipeline.to_csv(
        os.path.join(args.output_dir, "beaf_text_text_pipeline.csv"), index=False
    )
    df_layer = pd.DataFrame(pipeline_data["layers"])
    df_layer.to_csv(
        os.path.join(args.output_dir, "beaf_text_text_cosine.csv"), index=False
    )

    print("  Pipeline step-wise cosine similarity (pos <-> neg text):")
    for row in pipeline_data["pipeline"]:
        print(f"    [{row['step_name']:22s}] Cosine: {row['mean_cosine_sim']:.4f}  "
              f"L2: {row['mean_l2_distance']:.4f}")
    print("  Saved: beaf_text_text_pipeline.csv, beaf_text_text_cosine.csv")

    # Retrieve the final layer cosine for summary
    final_text_cosine = float(
        next(
            r["mean_cosine_sim"] for r in reversed(pipeline_data["pipeline"])
        )
    )

    # ================================================================== #
    # Axis 2: Image <-> Text (reuse compute_image_text_retrieval_metrics)
    # ================================================================== #
    print("\n" + "=" * 60)
    print("Axis 2: Image <-> Text — Retrieval Accuracy by object_in_image")
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
        pair_metadata, pos_texts, neg_texts,
        retrieval_cfg,
    )

    summary2: Dict[str, Any] = {}
    if retrieval_data.get("results_df") is not None:
        retrieval_data["results_df"].to_csv(
            os.path.join(args.output_dir, "beaf_image_text_similarity.csv"), index=False
        )
        summary2 = retrieval_data["summary"]
        with open(
            os.path.join(args.output_dir, "beaf_image_text_summary.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(summary2, f, indent=2)

        print(f"  Evaluated pairs          : {summary2['total_pairs_evaluated']}"
              f"  (skipped: {retrieval_data['skipped_count']})")
        print(f"  Pearson r                : {summary2['pearson_r']:.4f}")
        og = summary2["object_present_subgroup"]
        ab = summary2["object_absent_subgroup"]
        print(f"  [Obj Present] Pos Acc    : {og['positive_caption_accuracy_pct']:.1f}%"
              f"  (Flip: {og['ranking_flip_rate_pct']:.1f}%)")
        print(f"  [Obj Absent ] Neg Acc    : {ab['negative_caption_accuracy_pct']:.1f}%"
              f"  (Pos Pref: {ab['positive_caption_flip_rate_pct']:.1f}%)")
        print("  Saved: beaf_image_text_similarity.csv, beaf_image_text_summary.json")
    else:
        print("  [Warning] No images processed for Axis 2. Check image_root / file paths.")

    # ================================================================== #
    # Axis 3: Image <-> Image — Visual encoder sensitivity to object removal
    # ================================================================== #
    print("\n" + "=" * 60)
    print("Axis 3: Image <-> Image — Visual Encoder Object Removal Sensitivity")
    print("=" * 60)

    img_img_df = compute_image_image_cosine(
        cf_pairs, model, preprocess, device, args.img_batch
    )

    valid_img = img_img_df.dropna(subset=["sim_img_img"])
    axis3_summary: Dict[str, Any] = {}

    if len(valid_img) > 0:
        mean_sim = valid_img["sim_img_img"].mean()
        std_sim  = valid_img["sim_img_img"].std()
        min_sim  = valid_img["sim_img_img"].min()
        max_sim  = valid_img["sim_img_img"].max()

        print(f"  Pairs computed           : {len(valid_img)} / {len(cf_pairs)}")
        print(f"  Mean sim(orig, cf)       : {mean_sim:.4f}")
        print(f"  Std                      : {std_sim:.4f}")
        print(f"  Min / Max                : {min_sim:.4f} / {max_sim:.4f}")

        # Per-object statistics
        per_obj = (
            valid_img.groupby("object_name")["sim_img_img"]
            .agg(["mean", "std", "count"])
            .sort_values("mean")
        )
        print("\n  Per-object sim(orig, cf) (top-5 most sensitive):")
        for obj, row in per_obj.head(5).iterrows():
            print(f"    {obj:20s}  mean={row['mean']:.4f}  std={row['std']:.4f}  n={int(row['count'])}")

        img_img_df.to_csv(
            os.path.join(args.output_dir, "beaf_image_image_cosine.csv"), index=False
        )
        print("  Saved: beaf_image_image_cosine.csv")
        render_image_image_histogram(img_img_df, args.output_dir)

        axis3_summary = {
            "n_pairs":   int(len(valid_img)),
            "mean_sim":  round(float(mean_sim), 6),
            "std_sim":   round(float(std_sim),  6),
            "min_sim":   round(float(min_sim),  6),
            "max_sim":   round(float(max_sim),  6),
            "per_object": {
                obj: {
                    "mean": round(float(r["mean"]), 4),
                    "std":  round(float(r["std"]),  4),
                    "count": int(r["count"]),
                }
                for obj, r in per_obj.iterrows()
            },
        }
    else:
        print("  [Warning] No valid image pairs for Axis 3. Check image file paths.")

    # ================================================================== #
    # Axis 4: 4-way cross similarity matrix
    # ================================================================== #
    print("\n" + "=" * 60)
    print("Axis 4: 4-Way Cross Similarity Matrix")
    print("=" * 60)

    matrix_df = compute_4way_matrix(
        cf_pairs, model, tokenizer, preprocess,
        device, args.batch_size, args.img_batch,
    )

    valid_4way = matrix_df.dropna(subset=["full_correct"])
    axis4_summary: Dict[str, Any] = {}

    if len(valid_4way) > 0:
        fcr  = float(pd.Series(valid_4way["full_correct"]).mean())   * 100
        tnr  = float(pd.Series(valid_4way["text_neg_correct"]).mean()) * 100
        vcr  = float(pd.Series(valid_4way["visual_cf_correct"]).mean()) * 100
        ctcr = float(pd.Series(valid_4way["cf_text_correct"]).mean()) * 100

        mean_A = valid_4way["A_sim_orig_pos"].mean()
        mean_B = valid_4way["B_sim_orig_neg"].mean()
        mean_C = valid_4way["C_sim_cf_pos"].mean()
        mean_D = valid_4way["D_sim_cf_neg"].mean()

        print(f"  Pairs evaluated                      : {len(valid_4way)}")
        print(f"  Full Correct Rate (all 4 constraints): {fcr:.1f}%")
        print(f"    Text Negation Correct  (A > B)     : {tnr:.1f}%")
        print(f"    Visual Change Correct  (A > C)     : {vcr:.1f}%")
        print(f"    CF Text Correct        (D > C)     : {ctcr:.1f}%")
        print(f"  Mean similarities:")
        print(f"    A = sim(orig, pos) = {mean_A:.4f}")
        print(f"    B = sim(orig, neg) = {mean_B:.4f}")
        print(f"    C = sim(cf,   pos) = {mean_C:.4f}")
        print(f"    D = sim(cf,   neg) = {mean_D:.4f}")
        print(f"  Mean Text Negation Score (A-B) : {valid_4way['text_negation_score'].mean():.4f}")
        print(f"  Mean Visual Change Score (A-C)  : {valid_4way['visual_change_score'].mean():.4f}")

        matrix_df.to_csv(
            os.path.join(args.output_dir, "beaf_4way_matrix.csv"), index=False
        )
        print("  Saved: beaf_4way_matrix.csv")

        render_4way_heatmap(matrix_df, args.output_dir)
        render_text_vs_visual_scatter(matrix_df, args.output_dir)
        render_full_correct_by_object(matrix_df, args.output_dir)

        axis4_summary = {
            "n_pairs":                    int(len(valid_4way)),
            "full_correct_rate_pct":      round(fcr, 2),
            "text_negation_correct_pct":  round(tnr, 2),
            "visual_change_correct_pct":  round(vcr, 2),
            "cf_text_correct_pct":        round(ctcr, 2),
            "mean_A_sim_orig_pos":        round(float(mean_A), 6),
            "mean_B_sim_orig_neg":        round(float(mean_B), 6),
            "mean_C_sim_cf_pos":          round(float(mean_C), 6),
            "mean_D_sim_cf_neg":          round(float(mean_D), 6),
            "mean_text_negation_score":   round(float(valid_4way["text_negation_score"].mean()), 6),
            "mean_visual_change_score":   round(float(valid_4way["visual_change_score"].mean()),  6),
            "mean_cf_coherence_score":    round(float(valid_4way["cf_coherence_score"].mean()),   6),
        }
    else:
        print("  [Warning] No valid pairs for Axis 4. Check image file paths.")

    # ================================================================== #
    # Overall summary report
    # ================================================================== #
    print("\n" + "=" * 60)
    print("Writing overall summary report ...")
    final_summary = {
        "model":      args.model,
        "pretrained": args.pretrained,
        "csv_path":   args.csv_path,
        "n_rows":     int(len(df)),
        "n_cf_pairs": int(len(cf_pairs)),
        "axis1_text_text": {
            "final_l2norm_cosine_sim": round(final_text_cosine, 6),
            "pipeline_steps": [
                {"step": r["step_name"], "mean_cosine_sim": round(r["mean_cosine_sim"], 6)}
                for r in pipeline_data["pipeline"]
            ],
        },
        "axis2_image_text": summary2 if summary2 else None,
        "axis3_image_image": axis3_summary if axis3_summary else None,
        "axis4_4way": axis4_summary if axis4_summary else None,
    }

    summary_path = os.path.join(args.output_dir, "beaf_summary_report.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  BEAF Analysis Complete!")
    print(f"  All artifacts saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
