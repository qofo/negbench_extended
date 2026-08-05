"""
BEAF Counterfactual Comprehensive Analysis Script (Unified v1+v2).

Analyzes CLIP visual and text encoder mechanisms using paired counterfactual image data.
This script merges the original 4-Axis analysis (v1) with the Vision Encoder deep-dive (v2)
into a single unified pipeline.

  Part A — 4-Axis BEAF Analysis (from v1):
    Axis 1 — Text <-> Text:   layer-wise & pipeline cosine similarity (pos vs neg text)
    Axis 2 — Image <-> Text:  retrieval accuracy grouped by object_in_image
    Axis 3 — Image <-> Image: visual encoder sensitivity to object removal
    Axis 4 — 4-Way Cross:     A=sim(orig,pos), B=sim(orig,neg), C=sim(cf,pos), D=sim(cf,neg)
                               Full Correct Rate: A>B & D>C & A>C & D>B

  Part B — Vision Encoder Mechanism Analyses (from v2):
    - Scatter Plots (4 types): pos_vs_neg, delta_quadrant, img_orig_vs_cf, by_object
    - beaf_vision_pipeline_breakdown.csv & .png  Layer-wise ViT Visual Feature Shift
    - beaf_vision_svd_sweep.png & .json           Visual Projection Matrix SVD & Alignment
    - beaf_vision_linear_probe.json              5-fold CV Linear Probing for Object Presence
    - beaf_vision_direction_preservation.json    Pre/Post Projection Distance Ratio & t-test
    - beaf_comprehensive_summary_report.json     Combined summary of all analyses

Usage:
  python -m analysis.run_beaf_analysis_v2 \\
      --csv_path beaf_counterfactual_6col.csv \\
      --image_root "" \\
      --output_dir logs/evaluation/beaf_counterfactual_v2/openai_vit_b32 \\
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
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from PIL import Image

import open_clip
from analysis.config import (
    MetadataKey,
    RetrievalConfig,
    l2_normalize,
    batch_cosine_similarity,
    batch_l2_distance,
    batch_dot_product,
)
from analysis.extractor import extract_all_features_unified
from analysis.metrics import (
    compute_pipeline_and_layer_breakdown,
    compute_image_text_retrieval_metrics,
)


# =========================================================================== #
# Part A — v1 Helpers: 4-Axis BEAF Analysis
# =========================================================================== #

def load_beaf_csv(csv_path: str, image_root: str) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Load beaf_counterfactual_6col.csv and resolve absolute image paths.
    Returns (df, pair_metadata) compatible with compute_image_text_retrieval_metrics.
    """
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
    """
    Group rows by source_template and extract (original, counterfactual) image pairs.
    Each source_template has two rows: object_in_image==True (orig) and ==False (cf).
    """
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


def _encode_image_paths(
    paths: List[str],
    model,
    preprocess,
    device: str,
    batch_size: int,
    fallback_dim: int = 512,
) -> Tuple[np.ndarray, List[bool]]:
    """Encode image file paths with CLIP visual encoder. Returns (embeddings, loaded_flags)."""
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
    """
    Compute four cross-modal similarities per (orig_image, cf_image, pos_text, neg_text) quad (Axis 4):
      A = sim(img_orig, pos_text), B = sim(img_orig, neg_text),
      C = sim(img_cf,   pos_text), D = sim(img_cf,   neg_text)
    """
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


def render_image_image_histogram(img_img_df: pd.DataFrame, output_dir: str):
    """Plot distribution and per-object boxplot of sim(orig, cf) — Axis 3."""
    valid = img_img_df.dropna(subset=["sim_img_img"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    ax.hist(valid["sim_img_img"], bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    mean_val = valid["sim_img_img"].mean()
    ax.axvline(mean_val, color="crimson", ls="--", lw=2, label=f"Mean = {mean_val:.4f}")
    ax.set_xlabel("Cosine Similarity  (Original Image <-> Counterfactual Image)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Image Encoder Sensitivity to Object Removal\n(Distribution of Image<->Image Cosine Similarity)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.4)

    ax2 = axes[1]
    top_objects = valid["object_name"].value_counts().head(15).index.tolist()
    plot_data = [valid.loc[valid["object_name"] == obj, "sim_img_img"].values for obj in top_objects]
    bp = ax2.boxplot(plot_data, patch_artist=True, vert=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_objects)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax2.set_xticks(range(1, len(top_objects) + 1))
    ax2.set_xticklabels(top_objects, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Cosine Similarity  (orig <-> cf)", fontsize=11)
    ax2.set_title("Per-Object Visual Change Sensitivity\n(Lower = Image Encoder Detects Object Removal Better)", fontsize=11, fontweight="bold")
    ax2.grid(True, ls="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beaf_image_image_histogram.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_image_image_histogram.png")


def render_4way_heatmap(matrix_df: pd.DataFrame, output_dir: str):
    """Render 2x2 average similarity heatmap (Axis 4)."""
    valid = matrix_df.dropna(subset=["A_sim_orig_pos"])
    grid = np.array([
        [valid["A_sim_orig_pos"].mean(), valid["B_sim_orig_neg"].mean()],
        [valid["C_sim_cf_pos"].mean(),   valid["D_sim_cf_neg"].mean()],
    ])
    vmin, vmax = grid.min() - 0.005, grid.max() + 0.005
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Positive Caption\n(object present in text)", "Negative Caption\n(object absent in text)"], fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Original Image\n(object IN)", "Counterfactual\n(object OUT)"], fontsize=11)
    ax.set_title("CLIP 4-Way Similarity Matrix\n(Image State x Text Polarity)", fontsize=12, fontweight="bold")
    for i in range(2):
        for j in range(2):
            v = grid[i, j]
            tc = "white" if (v < vmin + (vmax - vmin) * 0.25 or v > vmax - (vmax - vmin) * 0.25) else "black"
            ax.text(j, i, f"{'ABCD'[i*2+j]}\n{v:.4f}", ha="center", va="center", fontsize=15, fontweight="bold", color=tc)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beaf_4way_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_4way_heatmap.png")


def render_text_vs_visual_scatter(matrix_df: pd.DataFrame, output_dir: str):
    """Scatter: Text Negation Score (A-B) vs Visual Change Score (A-C) — Axis 4."""
    valid = matrix_df.dropna(subset=["text_negation_score", "visual_change_score", "full_correct"])
    colors = valid["full_correct"].map({True: "seagreen", False: "crimson"})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(valid["text_negation_score"], valid["visual_change_score"], c=colors, alpha=0.55, s=28, edgecolors="none")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Text Negation Score  A - B  (sim_orig_pos - sim_orig_neg)", fontsize=10)
    ax.set_ylabel("Visual Change Score  A - C  (sim_orig_pos - sim_cf_pos)", fontsize=10)
    ax.set_title("CLIP Sensitivity:\nText Negation vs Visual Object Removal", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="seagreen", label="Full Correct (A>B & D>C & A>C & D>B)"), Patch(facecolor="crimson", label="Not Full Correct")], fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beaf_text_vs_visual_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_text_vs_visual_scatter.png")


def render_full_correct_by_object(matrix_df: pd.DataFrame, output_dir: str):
    """Horizontal bar chart of Full Correct Rate per object category — Axis 4."""
    valid = matrix_df.dropna(subset=["full_correct"])
    obj_stats = valid.groupby("object_name")["full_correct"].agg(["sum", "count"]).rename(columns={"sum": "correct", "count": "total"})
    obj_stats["rate_pct"] = obj_stats["correct"] / obj_stats["total"] * 100
    obj_stats = obj_stats.sort_values("rate_pct", ascending=True)
    overall_mean = valid["full_correct"].mean() * 100

    fig, ax = plt.subplots(figsize=(9, max(5, len(obj_stats) * 0.38)))
    bar_colors = [plt.cm.RdYlGn(r / 100) for r in obj_stats["rate_pct"]]
    bars = ax.barh(obj_stats.index, obj_stats["rate_pct"], color=bar_colors, edgecolor="white", height=0.7)
    for bar, (idx, row) in zip(bars, obj_stats.iterrows()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{int(row['correct'])}/{int(row['total'])}", va="center", fontsize=7.5, color="#333333")
    ax.axvline(overall_mean, color="navy", ls="--", lw=1.5, label=f"Overall Mean = {overall_mean:.1f}%")
    ax.set_xlabel("Full Correct Rate (%)", fontsize=11)
    ax.set_title("CLIP Full Correct Rate by Object Category\n(A>B & D>C & A>C & D>B)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 108)
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", alpha=0.4, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beaf_full_correct_rate_by_object.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_full_correct_rate_by_object.png")


# =========================================================================== #
# Part B — v2 Data Loader: 2n & 2n+1 Pairing (1,778 Full Pairs)
# =========================================================================== #


def load_beaf_paired_dataset(csv_path: str, image_root: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load beaf_counterfactual_6col.csv and construct 1,778 exact pairs
    by pairing row 2n (object_in_image=True) and row 2n+1 (object_in_image=False).

    Returns:
        df_raw   : full raw dataframe with resolved image paths (3,556 rows)
        df_pairs : paired dataframe with 1,778 rows containing:
                   pair_id, object_name, orig_path, cf_path,
                   positive_caption, negative_caption, source_template
    """
    df = pd.read_csv(csv_path)

    # Resolve image paths
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

        # Determine which row is orig (True) vs cf (False)
        if row1["object_in_image"] and not row2["object_in_image"]:
            orig_row, cf_row = row1, row2
        elif not row1["object_in_image"] and row2["object_in_image"]:
            orig_row, cf_row = row2, row1
        else:
            # Fallback if both True or both False
            orig_row, cf_row = row1, row2

        pairs.append({
            "pair_id":           i // 2,
            "object_name":       str(orig_row.get("object_name", "")),
            "orig_path":         orig_row["abs_image_path"],
            "cf_path":           cf_row["abs_image_path"],
            "positive_caption":  str(orig_row["positive_caption"]),
            "negative_caption":  str(orig_row["negative_caption"]),
            "source_template":   str(orig_row.get("source_template", "")),
        })

    df_pairs = pd.DataFrame(pairs)
    return df, df_pairs


# =========================================================================== #
# Vision Feature Extraction Engine
# =========================================================================== #

def extract_vision_features_unified(
    model: nn.Module,
    preprocess: Any,
    image_paths: List[str],
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Extract intermediate Vision Transformer layer representations and pipeline steps.

    Returns:
        Dict containing:
          - "layers": Dict[str, np.ndarray] layer0 (embed) to layer12
          - "pre_proj": np.ndarray (before visual projection)
          - "final_l2norm": np.ndarray (final L2-normalized image embedding)
          - "loaded_flags": List[bool]
    """
    model.eval()
    visual = getattr(model, "visual", model)

    # Dissect OpenCLIP visual architecture
    conv1 = getattr(visual, "conv1", None)
    class_embedding = getattr(visual, "class_embedding", None)
    positional_embedding = getattr(visual, "positional_embedding", None)
    ln_pre = getattr(visual, "ln_pre", None)
    transformer = getattr(visual, "transformer", None)
    ln_post = getattr(visual, "ln_post", None)
    proj = getattr(visual, "proj", None)

    resblocks = transformer.resblocks if transformer is not None else []
    num_layers = 1 + len(resblocks)

    layer_batches = [[] for _ in range(num_layers)]
    pre_proj_batches = []
    final_l2_batches = []
    loaded_flags = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors = []
        valid_idx = []

        for j, p in enumerate(batch_paths):
            if not os.path.exists(p):
                loaded_flags.append(False)
                continue
            try:
                t = preprocess(Image.open(p).convert("RGB"))
                tensors.append(t)
                valid_idx.append(j)
                loaded_flags.append(True)
            except Exception as ex:
                loaded_flags.append(False)

        if len(tensors) == 0:
            # All images in batch failed
            dummy_dim = proj.shape[1] if proj is not None else 512
            for l_idx in range(num_layers):
                layer_batches[l_idx].append(np.zeros((len(batch_paths), dummy_dim)))
            pre_proj_batches.append(np.zeros((len(batch_paths), dummy_dim)))
            final_l2_batches.append(np.zeros((len(batch_paths), dummy_dim)))
            continue

        stacked = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            cast_dtype = transformer.get_cast_dtype() if hasattr(transformer, "get_cast_dtype") else stacked.dtype

            # ViT Stem forward
            if conv1 is not None:
                x = conv1(stacked)  # [B, C, H, W]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, N_grid]
                x = x.permute(0, 2, 1)  # [B, N_grid, C]
            else:
                x = stacked

            if class_embedding is not None:
                class_emb = class_embedding.to(x.dtype)
                if class_emb.ndim == 1:
                    class_emb = class_emb.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
                x = torch.cat([class_emb, x], dim=1)

            if positional_embedding is not None:
                x = x + positional_embedding.to(x.dtype)

            if ln_pre is not None:
                x = ln_pre(x)

            hidden_states = [x]

            # ViT Blocks forward
            x_perm = x.permute(1, 0, 2)
            for block in resblocks:
                x_perm = block(x_perm)
                hidden_states.append(x_perm.permute(1, 0, 2))

            # Layer pooling (extract CLS token = index 0)
            pooled_layers = []
            for hs in hidden_states:
                cls_feat = hs[:, 0, :].float().cpu().numpy()
                pooled_layers.append(cls_feat)

            # Post LN & Proj
            x_post = hidden_states[-1][:, 0, :]
            if ln_post is not None:
                x_post = ln_post(x_post)
            pre_proj_feat = x_post.float().cpu().numpy()

            if proj is not None:
                if isinstance(proj, torch.Tensor):
                    x_proj = x_post.to(proj.dtype) @ proj
                else:
                    x_proj = proj(x_post)
            else:
                x_proj = x_post

            final_l2_feat = F.normalize(x_proj.float(), dim=-1).cpu().numpy()

            # Align batch indexing
            embed_dim = final_l2_feat.shape[1]
            pre_dim = pre_proj_feat.shape[1]

            for l_idx in range(num_layers):
                l_arr = np.zeros((len(batch_paths), pooled_layers[l_idx].shape[1]))
                vi = 0
                for j in range(len(batch_paths)):
                    if j in valid_idx:
                        l_arr[j] = pooled_layers[l_idx][vi]
                        if l_idx == num_layers - 1:
                            vi += 1
                layer_batches[l_idx].append(l_arr)

            pre_arr = np.zeros((len(batch_paths), pre_dim))
            post_arr = np.zeros((len(batch_paths), embed_dim))
            vi = 0
            for j in range(len(batch_paths)):
                if j in valid_idx:
                    pre_arr[j] = pre_proj_feat[vi]
                    post_arr[j] = final_l2_feat[vi]
                    vi += 1

            pre_proj_batches.append(pre_arr)
            final_l2_batches.append(post_arr)

    layer_dict = {}
    for l_idx, feats in enumerate(layer_batches):
        name = "Embedding" if l_idx == 0 else f"Layer {l_idx}"
        layer_dict[name] = np.concatenate(feats, axis=0)

    return {
        "layers":        layer_dict,
        "pre_proj":      np.concatenate(pre_proj_batches, axis=0),
        "final_l2norm":  np.concatenate(final_l2_batches, axis=0),
        "loaded_flags":  loaded_flags,
    }


# =========================================================================== #
# Scatter Plot Visualizations
# =========================================================================== #

def render_scatter_pos_vs_neg(
    all_img_embs: np.ndarray,
    all_pos_embs: np.ndarray,
    all_neg_embs: np.ndarray,
    all_obj_in_img: np.ndarray,
    output_dir: str
):
    """
    COCO-style Pos vs Neg Cosine Similarity Scatter Plot (N=3,556).
    """
    sim_pos = batch_cosine_similarity(all_img_embs, all_pos_embs)
    sim_neg = batch_cosine_similarity(all_img_embs, all_neg_embs)

    r_val, _ = stats.pearsonr(sim_pos, sim_neg)
    rho_val, _ = stats.spearmanr(sim_pos, sim_neg)
    N = len(sim_pos)

    fig, ax = plt.subplots(figsize=(8, 7))

    mask_true = (all_obj_in_img == True)
    mask_false = (all_obj_in_img == False)

    ax.scatter(sim_pos[mask_true], sim_neg[mask_true], c="dodgerblue", alpha=0.35, s=16, label="Object IN image", edgecolors="none")
    ax.scatter(sim_pos[mask_false], sim_neg[mask_false], c="crimson", marker="^", alpha=0.35, s=16, label="Object NOT in image", edgecolors="none")

    # y=x line
    min_val = min(sim_pos.min(), sim_neg.min()) - 0.02
    max_val = max(sim_pos.max(), sim_neg.max()) + 0.02
    ax.plot([min_val, max_val], [min_val, max_val], color="gray", ls="--", lw=1.5, label="y=x (perfect correlation)")

    ax.set_xlabel("cos_sim(image, positive_caption)", fontsize=11, fontweight="bold")
    ax.set_ylabel("cos_sim(image, negative_caption)", fontsize=11, fontweight="bold")
    ax.set_title(f"Image-Text Similarity: Positive vs Negative (BEAF Counterfactual)\nPearson r={r_val:.3f}, Spearman rho={rho_val:.3f} (N={N:,})", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(fontsize=10, loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "beaf_scatter_pos_vs_neg.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_scatter_pos_vs_neg.png")


def render_scatter_delta_quadrant(
    sim_orig_pos: np.ndarray,
    sim_orig_neg: np.ndarray,
    sim_cf_pos: np.ndarray,
    output_dir: str
):
    """
    Delta-Delta 4-Quadrant Scatter Plot:
      X = sim(orig, pos) - sim(orig, neg)  [Text Negation Sensitivity]
      Y = sim(orig, pos) - sim(cf, pos)    [Visual Change Sensitivity]
    """
    delta_text = sim_orig_pos - sim_orig_neg
    delta_visual = sim_orig_pos - sim_cf_pos
    N = len(delta_text)

    q1 = np.sum((delta_text > 0) & (delta_visual > 0)) / N * 100
    q2 = np.sum((delta_text <= 0) & (delta_visual > 0)) / N * 100
    q3 = np.sum((delta_text <= 0) & (delta_visual <= 0)) / N * 100
    q4 = np.sum((delta_text > 0) & (delta_visual <= 0)) / N * 100

    fig, ax = plt.subplots(figsize=(8, 7))

    # Color code by quadrant
    colors = []
    for dt, dv in zip(delta_text, delta_visual):
        if dt > 0 and dv > 0:
            colors.append("seagreen")     # Q1
        elif dt <= 0 and dv > 0:
            colors.append("dodgerblue")   # Q2
        elif dt > 0 and dv <= 0:
            colors.append("crimson")      # Q4 (most common failure)
        else:
            colors.append("gray")         # Q3

    ax.scatter(delta_text, delta_visual, c=colors, alpha=0.45, s=20, edgecolors="none")

    ax.axhline(0, color="black", lw=1.2, ls="--")
    ax.axvline(0, color="black", lw=1.2, ls="--")

    ax.set_xlabel("Text Negation Score  sim(orig, pos) - sim(orig, neg)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Visual Change Score  sim(orig, pos) - sim(cf, pos)", fontsize=10, fontweight="bold")
    ax.set_title("Delta-Delta Quadrant Scatter: Text Negation vs Visual Change Sensitivity", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)

    # Annotate Quadrants
    ax.text(0.75, 0.90, f"Q1: Both Sensitive\n({q1:.1f}%)", transform=ax.transAxes, color="seagreen", fontweight="bold", fontsize=10)
    ax.text(0.05, 0.90, f"Q2: Visual Only\n({q2:.1f}%)", transform=ax.transAxes, color="dodgerblue", fontweight="bold", fontsize=10)
    ax.text(0.75, 0.05, f"Q4: Text Only (Failure)\n({q4:.1f}%)", transform=ax.transAxes, color="crimson", fontweight="bold", fontsize=10)
    ax.text(0.05, 0.05, f"Q3: Neither\n({q3:.1f}%)", transform=ax.transAxes, color="gray", fontweight="bold", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "beaf_scatter_delta_text_vs_delta_visual.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_scatter_delta_text_vs_delta_visual.png")


def render_scatter_img_orig_vs_img_cf(
    sim_orig_pos: np.ndarray,
    sim_cf_pos: np.ndarray,
    output_dir: str
):
    """
    Pure Visual Sensitivity Scatter:
      X = sim(I_orig, T_pos)
      Y = sim(I_cf, T_pos)
    """
    fig, ax = plt.subplots(figsize=(7, 6.5))

    ax.scatter(sim_orig_pos, sim_cf_pos, c="darkslateblue", alpha=0.45, s=20, edgecolors="none")

    min_v = min(sim_orig_pos.min(), sim_cf_pos.min()) - 0.02
    max_v = max(sim_orig_pos.max(), sim_cf_pos.max()) + 0.02
    ax.plot([min_v, max_v], [min_v, max_v], color="crimson", ls="--", lw=1.5, label="y=x (No Visual Shift)")

    below_cnt = np.sum(sim_cf_pos < sim_orig_pos)
    pct_below = below_cnt / len(sim_orig_pos) * 100

    ax.set_xlabel("sim(Original Image, Positive Caption)", fontsize=10, fontweight="bold")
    ax.set_ylabel("sim(Counterfactual Image, Positive Caption)", fontsize=10, fontweight="bold")
    ax.set_title(f"Pure Visual Sensitivity: Original vs Counterfactual Image\n({pct_below:.1f}% pairs drop similarity when object removed)", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "beaf_scatter_img_orig_vs_img_cf.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_scatter_img_orig_vs_img_cf.png")


def render_scatter_by_object_category(
    df_pairs: pd.DataFrame,
    sim_orig_pos: np.ndarray,
    sim_orig_neg: np.ndarray,
    output_dir: str
):
    """
    Subplot grid showing Pos vs Neg scatter across Top 6 Object Categories.
    """
    df_pairs["sim_pos"] = sim_orig_pos
    df_pairs["sim_neg"] = sim_orig_neg

    top_objs = df_pairs["object_name"].value_counts().head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, obj in enumerate(top_objs):
        ax = axes[idx]
        sub = df_pairs[df_pairs["object_name"] == obj]

        ax.scatter(sub["sim_pos"], sub["sim_neg"], c="dodgerblue", alpha=0.6, s=24)

        min_v = min(sub["sim_pos"].min(), sub["sim_neg"].min()) - 0.01
        max_v = max(sub["sim_pos"].max(), sub["sim_neg"].max()) + 0.01
        ax.plot([min_v, max_v], [min_v, max_v], color="crimson", ls="--", lw=1.2)

        r_sub, _ = stats.pearsonr(sub["sim_pos"], sub["sim_neg"])
        ax.set_title(f"Object: {obj} (n={len(sub)})\nPearson r={r_sub:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("sim(pos)", fontsize=9)
        ax.set_ylabel("sim(neg)", fontsize=9)
        ax.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "beaf_scatter_by_object_category.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: beaf_scatter_by_object_category.png")


# =========================================================================== #
# Vision Encoder Mechanism Analyses
# =========================================================================== #

def compute_vision_pipeline_breakdown(
    vis_orig: Dict[str, Any],
    vis_cf: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Track layer-wise visual feature similarity & L2 distance shift between orig and cf images.
    """
    layer_results = []
    layer_names = list(vis_orig["layers"].keys())

    for l_name in layer_names:
        f_orig = vis_orig["layers"][l_name]
        f_cf   = vis_cf["layers"][l_name]

        cos_sims = batch_cosine_similarity(f_orig, f_cf)
        l2_dists = batch_l2_distance(f_orig, f_cf)
        dot_prods = batch_dot_product(f_orig, f_cf)

        layer_results.append({
            "layer": l_name,
            "mean_cosine_sim": float(np.mean(cos_sims)),
            "std_cosine_sim": float(np.std(cos_sims)),
            "mean_l2_distance": float(np.mean(l2_dists)),
            "mean_dot_product": float(np.mean(dot_prods))
        })

    # Add Pre-proj and Final L2Norm
    cos_pre = batch_cosine_similarity(vis_orig["pre_proj"], vis_cf["pre_proj"])
    l2_pre  = batch_l2_distance(vis_orig["pre_proj"], vis_cf["pre_proj"])
    layer_results.append({
        "layer": "Pre-Projection (LN)",
        "mean_cosine_sim": float(np.mean(cos_pre)),
        "std_cosine_sim": float(np.std(cos_pre)),
        "mean_l2_distance": float(np.mean(l2_pre)),
        "mean_dot_product": float(np.mean(batch_dot_product(vis_orig["pre_proj"], vis_cf["pre_proj"])))
    })

    cos_final = batch_cosine_similarity(vis_orig["final_l2norm"], vis_cf["final_l2norm"])
    l2_final  = batch_l2_distance(vis_orig["final_l2norm"], vis_cf["final_l2norm"])
    layer_results.append({
        "layer": "+Final L2Norm",
        "mean_cosine_sim": float(np.mean(cos_final)),
        "std_cosine_sim": float(np.std(cos_final)),
        "mean_l2_distance": float(np.mean(l2_final)),
        "mean_dot_product": float(np.mean(batch_dot_product(vis_orig["final_l2norm"], vis_cf["final_l2norm"])))
    })

    df_res = pd.DataFrame(layer_results)
    df_res.to_csv(os.path.join(output_dir, "beaf_vision_pipeline_breakdown.csv"), index=False)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x_labs = df_res["layer"].values
    c_vals = df_res["mean_cosine_sim"].values

    ax1.plot(x_labs, c_vals, "o-", color="darkgreen", lw=2.5, ms=7, label="Mean Cosine Sim")
    ax1.set_ylabel("Cosine Similarity (orig ↔ cf image)", color="darkgreen", fontsize=10, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="darkgreen")
    ax1.set_title("Vision Encoder Transformer Layer Breakdown: Visual Feature Shift on Object Removal", fontsize=11, fontweight="bold")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    ax1.grid(True, ls="--", alpha=0.5)

    ax2 = ax1.twinx()
    l2_vals = df_res["mean_l2_distance"].values
    ax2.plot(x_labs, l2_vals, "s--", color="darkorange", lw=2, ms=6, label="Mean L2 Distance")
    ax2.set_ylabel("L2 Distance", color="darkorange", fontsize=10, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beaf_vision_pipeline_breakdown.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("  Saved: beaf_vision_pipeline_breakdown.csv & .png")
    return {"breakdown": layer_results}


def compute_vision_svd_sweep(
    model: nn.Module,
    vis_orig: Dict[str, Any],
    vis_cf: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Perform SVD on Visual Projection Matrix (W_vis) and evaluate Truncation Sweep.
    """
    visual = getattr(model, "visual", model)
    proj = getattr(visual, "proj", None)

    if proj is None:
        print("  [Notice] Vision tower has no explicit proj tensor. Skipping Vision SVD Sweep.")
        return {}

    if isinstance(proj, nn.Linear):
        W = proj.weight.detach().cpu().numpy().T
    elif isinstance(proj, torch.Tensor):
        W = proj.detach().cpu().numpy()
    else:
        return {}

    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    f_pre_orig = vis_orig["pre_proj"]
    f_pre_cf   = vis_cf["pre_proj"]
    diff_pre   = f_pre_orig - f_pre_cf
    diff_norm  = l2_normalize(diff_pre)

    alignments = np.abs(diff_norm @ U)
    mean_alignments = np.mean(alignments, axis=0)

    sim_orig = batch_cosine_similarity(vis_orig["final_l2norm"], vis_cf["final_l2norm"]).mean()

    sweep_results = []
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    d_in = W.shape[0]
    for r in ratios:
        k = max(1, int(d_in * r))

        # Top-k
        W_top = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        p_orig_top = l2_normalize(f_pre_orig @ W_top)
        p_cf_top   = l2_normalize(f_pre_cf @ W_top)
        sim_top    = float(batch_cosine_similarity(p_orig_top, p_cf_top).mean())

        # Bottom-k
        W_bot = U[:, -k:] @ np.diag(S[-k:]) @ Vt[-k:, :]
        p_orig_bot = l2_normalize(f_pre_orig @ W_bot)
        p_cf_bot   = l2_normalize(f_pre_cf @ W_bot)
        sim_bot    = float(batch_cosine_similarity(p_orig_bot, p_cf_bot).mean())

        sweep_results.append({
            "keep_ratio": r,
            "k_singular_values": k,
            "cosine_sim_top_k": sim_top,
            "cosine_sim_bottom_k": sim_bot
        })

    report = {
        "singular_values_top10": S[:10].tolist(),
        "mean_alignment_with_top1_singular_vector": float(mean_alignments[0]),
        "max_alignment_singular_vector_idx": int(np.argmax(mean_alignments)),
        "cosine_sim_original": float(sim_orig),
        "spectrum_sweep": sweep_results
    }

    with open(os.path.join(output_dir, "beaf_vision_svd_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Plot Sweep
    fig, ax = plt.subplots(figsize=(8, 5))
    pcts = [sr["keep_ratio"] * 100 for sr in sweep_results]
    tops = [sr["cosine_sim_top_k"] for sr in sweep_results]
    bots = [sr["cosine_sim_bottom_k"] for sr in sweep_results]

    ax.plot(pcts, tops, "o-", color="purple", lw=2, label="Keep Top-k Singular Values")
    ax.plot(pcts, bots, "s--", color="teal", lw=2, label="Keep Bottom-k Singular Values")
    ax.axhline(sim_orig, color="black", ls=":", label=f"Original W_vis Sim ({sim_orig:.4f})")

    ax.set_xlabel("Singular Values Retained (%)", fontsize=11)
    ax.set_ylabel("Final Vision Cosine Similarity", fontsize=11)
    ax.set_title("Visual Projection SVD Spectrum Sweep: Top-k vs Bottom-k Truncation", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "beaf_vision_svd_sweep.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("  Saved: beaf_vision_svd_sweep.png & .json")
    return report


def compute_vision_linear_probe(
    vis_orig: Dict[str, Any],
    vis_cf: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Train 5-fold cross-validated Linear Probe on Vision Transformer features
    to classify object_in_image (True vs False).
    """
    n_orig = len(vis_orig["pre_proj"])
    n_cf   = len(vis_cf["pre_proj"])
    y = np.array([1] * n_orig + [0] * n_cf)

    probe_results = {}

    for l_name in list(vis_orig["layers"].keys()) + ["Pre-Projection", "+Final L2Norm"]:
        if l_name in vis_orig["layers"]:
            X_orig = vis_orig["layers"][l_name]
            X_cf   = vis_cf["layers"][l_name]
        elif l_name == "Pre-Projection":
            X_orig = vis_orig["pre_proj"]
            X_cf   = vis_cf["pre_proj"]
        else:
            X_orig = vis_orig["final_l2norm"]
            X_cf   = vis_cf["final_l2norm"]

        X = np.vstack([X_orig, X_cf])
        X_norm = l2_normalize(X)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_norm, y, cv=cv, scoring="accuracy")

        probe_results[l_name] = {
            "mean_accuracy_pct": float(np.mean(scores) * 100),
            "std_accuracy_pct":  float(np.std(scores) * 100),
        }

    with open(os.path.join(output_dir, "beaf_vision_linear_probe.json"), "w", encoding="utf-8") as f:
        json.dump(probe_results, f, indent=2)

    print("  Saved: beaf_vision_linear_probe.json")
    return probe_results


def compute_vision_direction_preservation(
    vis_orig: Dict[str, Any],
    vis_cf: Dict[str, Any],
    output_dir: str,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Compute pre/post projection visual distance compression ratio
    and run Welch's t-test against random control image pairs.
    """
    orig_pre  = vis_orig["pre_proj"]
    cf_pre    = vis_cf["pre_proj"]
    orig_post = vis_orig["final_l2norm"]
    cf_post   = vis_cf["final_l2norm"]

    dist_pre_pair  = batch_l2_distance(orig_pre, cf_pre)
    dist_post_pair = batch_l2_distance(orig_post, cf_post)
    ratio_pair     = dist_post_pair / (dist_pre_pair + 1e-8)

    N = len(orig_pre)
    rng = np.random.default_rng(seed=seed)
    rand_idx = (np.arange(N) + rng.integers(1, N, size=N)) % N

    rand_pre  = orig_pre[rand_idx]
    rand_post = orig_post[rand_idx]

    dist_pre_ctrl  = batch_l2_distance(orig_pre, rand_pre)
    dist_post_ctrl = batch_l2_distance(orig_post, rand_post)
    ratio_ctrl     = dist_post_ctrl / (dist_pre_ctrl + 1e-8)

    t_stat, p_val = stats.ttest_ind(ratio_pair, ratio_ctrl, equal_var=False)

    report = {
        "counterfactual_pair_mean_ratio": float(np.mean(ratio_pair)),
        "control_pair_mean_ratio":        float(np.mean(ratio_ctrl)),
        "welch_ttest_t_stat":             float(t_stat),
        "welch_ttest_p_value":            float(p_val)
    }

    with open(os.path.join(output_dir, "beaf_vision_direction_preservation.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("  Saved: beaf_vision_direction_preservation.json")
    return report


# =========================================================================== #
# Main Pipeline
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="BEAF Counterfactual Unified Analysis Pipeline (v1 4-Axis + v2 Vision Deep-dive)"
    )
    parser.add_argument("--csv_path",   type=str, required=True)
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="logs/evaluation/beaf_counterfactual_v2/openai_vit_b32")
    parser.add_argument("--model",      type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--img_batch",  type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Cap number of CSV rows loaded (0 = no cap)")
    parser.add_argument("--seed",       type=int, default=42)
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

    # ------------------------------------------------------------------ #
    # Step 1: Load Data
    # ------------------------------------------------------------------ #
    print("\n[Step 1] Loading BEAF CSV ...")

    # Part B loader: strict 2n/2n+1 row pairing
    df_raw, df_pairs = load_beaf_paired_dataset(args.csv_path, args.image_root)
    n_pairs = len(df_pairs)
    print(f"  Raw rows                                 : {len(df_raw)}")
    print(f"  Exact Counterfactual Pairs (orig <-> cf) : {n_pairs}")

    # Part A loader: source_template groupby pairing + pair_metadata for retrieval
    df_v1, pair_metadata = load_beaf_csv(args.csv_path, args.image_root)
    if args.max_samples > 0:
        df_v1         = df_v1.head(args.max_samples).copy()
        pair_metadata = pair_metadata[:args.max_samples]
    cf_pairs_v1  = build_counterfactual_pairs(df_v1)
    pos_texts_v1 = df_v1["positive_caption"].astype(str).tolist()
    neg_texts_v1 = df_v1["negative_caption"].astype(str).tolist()
    print(f"  Axis 1-4 pairs (source_template based)  : {len(cf_pairs_v1)}")

    # ------------------------------------------------------------------ #
    # Step 2: Load Model
    # ------------------------------------------------------------------ #
    print("\n[Step 2] Loading OpenCLIP Model ...")
    model, preprocess, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)
    model.eval()
    print("  Model loaded successfully.")

    # ------------------------------------------------------------------ #
    # Step 3: Extract Text Features (Part B)
    # ------------------------------------------------------------------ #
    print("\n[Step 3] Extracting Text Features (Part B) ...")
    pos_texts = df_pairs["positive_caption"].tolist()
    neg_texts = df_pairs["negative_caption"].tolist()

    pos_feat_dict = extract_all_features_unified(model, tokenizer, pos_texts, device, "eot", args.batch_size)
    neg_feat_dict = extract_all_features_unified(model, tokenizer, neg_texts, device, "eot", args.batch_size)

    pos_embs = pos_feat_dict["final_l2norm"]
    neg_embs = neg_feat_dict["final_l2norm"]

    # ------------------------------------------------------------------ #
    # Step 4: Extract Vision Features (Part B)
    # ------------------------------------------------------------------ #
    print("\n[Step 4] Extracting Vision Features (Part B) ...")
    vis_orig = extract_vision_features_unified(model, preprocess, df_pairs["orig_path"].tolist(), device, args.img_batch)
    vis_cf   = extract_vision_features_unified(model, preprocess, df_pairs["cf_path"].tolist(),   device, args.img_batch)

    orig_embs = vis_orig["final_l2norm"]
    cf_embs   = vis_cf["final_l2norm"]

    all_img_embs  = np.vstack([orig_embs, cf_embs])
    all_pos_embs  = np.vstack([pos_embs, pos_embs])
    all_neg_embs  = np.vstack([neg_embs, neg_embs])
    all_obj_flags = np.array([True] * n_pairs + [False] * n_pairs)

    # ------------------------------------------------------------------ #
    # Step 5: Part B — Scatter Plots
    # ------------------------------------------------------------------ #
    print("\n[Step 5] Part B — Rendering Scatter Plots ...")
    render_scatter_pos_vs_neg(all_img_embs, all_pos_embs, all_neg_embs, all_obj_flags, args.output_dir)

    sim_orig_pos = batch_cosine_similarity(orig_embs, pos_embs)
    sim_orig_neg = batch_cosine_similarity(orig_embs, neg_embs)
    sim_cf_pos   = batch_cosine_similarity(cf_embs,   pos_embs)

    render_scatter_delta_quadrant(sim_orig_pos, sim_orig_neg, sim_cf_pos, args.output_dir)
    render_scatter_img_orig_vs_img_cf(sim_orig_pos, sim_cf_pos, args.output_dir)
    render_scatter_by_object_category(df_pairs, sim_orig_pos, sim_orig_neg, args.output_dir)

    # ------------------------------------------------------------------ #
    # Step 6: Part B — Vision Encoder Mechanism Analyses
    # ------------------------------------------------------------------ #
    print("\n[Step 6] Part B — Vision Encoder Mechanism Analyses ...")
    vis_breakdown = compute_vision_pipeline_breakdown(vis_orig, vis_cf, args.output_dir)
    vis_svd       = compute_vision_svd_sweep(model, vis_orig, vis_cf, args.output_dir)
    vis_probe     = compute_vision_linear_probe(vis_orig, vis_cf, args.output_dir)
    vis_dir_pres  = compute_vision_direction_preservation(vis_orig, vis_cf, args.output_dir, seed=args.seed)

    # ------------------------------------------------------------------ #
    # Step 7: Part A — Axis 1: Text <-> Text
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Step 8: Part A — Axis 2: Image <-> Text Retrieval
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Step 9: Part A — Axis 3: Image <-> Image Cosine
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Step 10: Part A — Axis 4: 4-Way Cross Similarity
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Step 11: Write Comprehensive Summary Report
    # ------------------------------------------------------------------ #
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
        # Part B
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
        # Part A
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

    with open(os.path.join(args.output_dir, "beaf_comprehensive_summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(comprehensive_summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  BEAF Unified Analysis Complete!")
    print(f"  All artifacts saved to: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
