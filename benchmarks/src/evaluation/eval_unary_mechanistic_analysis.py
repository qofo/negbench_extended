"""
4-Stage Unary Mechanistic Analysis for Multimodal Negation (E1 ~ E4).

Investigates the core research questions:
- E1: Do image and text representations contain linear signals for presence/polarity?
- E2: Are visual absence and textual negation semantically aligned (Probe Normal & Centroid Shift)?
- E3: Does probe alignment predict cosine correctness margin M = min(S_++, S_--) - max(S_+-, S_-+)?
- E4: Why does cosine fail, and what does Bilinear recover (W = D + O: Diagonal vs Off-Diagonal Ablation)?

Outputs:
  - fig1_representation_probing.png      (E1: Vision & Text Layerwise Probe Acc)
  - fig2_alignment_distribution.png       (E2: A_normal, A_centroid vs Random Null Distribution)
  - fig3_alignment_vs_margin_scatter.png (E3: Alignment vs Cosine Margin M Scatter & Pearson/Spearman)
  - fig4_bilinear_ablation_bar.png       (E4: Cosine vs Diagonal vs Off-Diagonal vs Full Bilinear)
  - e1_to_e4_summary_table.csv
  - full_mechanistic_report.json

Usage:
    python -m benchmarks.src.evaluation.eval_unary_mechanistic_analysis \\
        --csv_path benchmarks/data/images/beaf_counterfactual_6col.csv \\
        --output_dir logs/evaluation/unary_mechanistic_analysis \\
        --model ViT-B-32 --pretrained openai
"""

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import open_clip

from benchmarks.src.evaluation.eval_layerwise_linear_probe import (
    extract_layerwise_feature_dict,
)


# ============================================================
# Helpers & Image Loader
# ============================================================
def resolve_image_path(path: str) -> Optional[str]:
    """Resolve image path across local directory structures."""
    if os.path.exists(path):
        return path
    alt = os.path.join("data/coco/images/val2014", os.path.basename(path))
    if os.path.exists(alt):
        return alt
    alt2 = os.path.join("benchmarks/data/images", os.path.basename(path))
    if os.path.exists(alt2):
        return alt2
    return None


def encode_images_safely(
    model: nn.Module,
    preprocess: Any,
    image_paths: List[str],
    device: str,
    embed_dim: int,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, List[bool]]:
    """Batch encode images with existence tracking."""
    embs = []
    valid_mask = []
    model.eval()

    for p in image_paths:
        resolved = resolve_image_path(p)
        if resolved is not None:
            try:
                img = Image.open(resolved).convert("RGB")
                tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.encode_image(tensor, normalize=True).float().cpu()
                embs.append(emb)
                valid_mask.append(True)
                continue
            except Exception:
                pass
        embs.append(torch.zeros(1, embed_dim))
        valid_mask.append(False)

    return torch.cat(embs, dim=0), valid_mask


def encode_texts_safely(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    """Batch encode text strings into L2-normalized vectors."""
    all_embs = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens, normalize=True).float().cpu()
        all_embs.append(emb)
    return torch.cat(all_embs, dim=0)


# ============================================================
# Stage E1: Modality Information Probing
# ============================================================
def fit_linear_probe(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    fit_intercept: bool = True,
) -> Tuple[float, float, np.ndarray, float]:
    """
    Fit Logistic Regression linear probe on pos vs neg features.
    Returns (mean_accuracy_pct, std_accuracy_pct, unit_normal_vector_w, bias_b).
    """
    n_pos, n_neg = len(X_pos), len(X_neg)
    n = min(n_pos, n_neg)
    X = np.vstack([X_pos[:n], X_neg[:n]])
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    y = np.array([1] * n + [0] * n)

    eff_splits = max(2, min(n_splits, n))
    cv = StratifiedKFold(n_splits=eff_splits, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed, fit_intercept=fit_intercept)
    scores = cross_val_score(clf, X_norm, y, cv=cv, scoring="accuracy")

    # Fit on all data for normal vector
    clf.fit(X_norm, y)
    w = clf.coef_[0]
    w_unit = w / (np.linalg.norm(w) + 1e-8)
    b = float(clf.intercept_[0]) if fit_intercept else 0.0

    return float(np.mean(scores)) * 100.0, float(np.std(scores)) * 100.0, w_unit, b


# ============================================================
# Stage E2: Cross-Modal Alignment Metrics
# ============================================================
def compute_cross_modal_alignment(
    w_I: np.ndarray,
    w_T: np.ndarray,
    d_I: np.ndarray,
    d_T: np.ndarray,
    n_null_samples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute Probe Normal Alignment and Centroid Shift Alignment with Random Null Test.
    """
    # Unit normalization
    w_I_unit = w_I / (np.linalg.norm(w_I) + 1e-8)
    w_T_unit = w_T / (np.linalg.norm(w_T) + 1e-8)
    d_I_unit = d_I / (np.linalg.norm(d_I) + 1e-8)
    d_T_unit = d_T / (np.linalg.norm(d_T) + 1e-8)

    a_normal = float(np.dot(w_I_unit, w_T_unit))
    a_centroid = float(np.dot(d_I_unit, d_T_unit))

    # Random Null Test: sample random unit vectors on S^(D-1)
    np.random.seed(seed)
    dim = len(w_I_unit)
    random_vectors = np.random.randn(n_null_samples, dim)
    random_vectors /= (np.linalg.norm(random_vectors, axis=1, keepdims=True) + 1e-8)

    null_cos_w = np.dot(random_vectors, w_I_unit)
    null_cos_d = np.dot(random_vectors, d_I_unit)

    p_val_normal = float(np.mean(null_cos_w >= a_normal))
    p_val_centroid = float(np.mean(null_cos_d >= a_centroid))

    return {
        "a_normal": a_normal,
        "a_centroid": a_centroid,
        "p_val_normal": p_val_normal,
        "p_val_centroid": p_val_centroid,
        "null_cos_w_mean": float(np.mean(null_cos_w)),
        "null_cos_w_std": float(np.std(null_cos_w)),
        "null_cos_d_mean": float(np.mean(null_cos_d)),
        "null_cos_d_std": float(np.std(null_cos_d)),
    }


# ============================================================
# Stage E3: 2x2 Score Matrix and Margin (M)
# ============================================================
def compute_2x2_margin(
    v_pos: torch.Tensor,
    v_neg: torch.Tensor,
    t_pos: torch.Tensor,
    t_neg: torch.Tensor,
) -> Dict[str, Any]:
    """
    Compute 2x2 score matrix:
        S = [[cos(v+, t+), cos(v+, t-)],
             [cos(v-, t+), cos(v-, t-)]]
    and Correctness Margin M = min(S++, S--) - max(S+-, S-+).
    """
    cos = lambda u, v: F.cosine_similarity(u, v, dim=-1)

    s_pp = cos(v_pos, t_pos)  # S++ (correct)
    s_pm = cos(v_pos, t_neg)  # S+- (incorrect)
    s_mp = cos(v_neg, t_pos)  # S-+ (incorrect)
    s_mm = cos(v_neg, t_neg)  # S-- (correct)

    correct_min = torch.minimum(s_pp, s_mm)
    wrong_max = torch.maximum(s_pm, s_mp)
    m = (correct_min - wrong_max).numpy()

    acc = float(np.mean(m > 0) * 100.0)
    return {
        "margin_mean": float(np.mean(m)),
        "margin_std": float(np.std(m)),
        "margin_median": float(np.median(m)),
        "accuracy_pct": acc,
        "margins": m,
        "s_pp_mean": float(s_pp.mean()),
        "s_pm_mean": float(s_pm.mean()),
        "s_mp_mean": float(s_mp.mean()),
        "s_mm_mean": float(s_mm.mean()),
    }


# ============================================================
# Stage E4: Bilinear Scorer, LABCLIP & Diagonal vs Off-Diagonal Ablation
# ============================================================
class BilinearMatcher(nn.Module):
    """Bilinear Interaction Head: s(v, t) = v^T W t + b."""
    def __init__(self, dim: int, use_bias: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim))
        self.bias = nn.Parameter(torch.zeros(1)) if use_bias else None

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # v: (B, D), t: (B, D)
        # s(v, t) = sum_i sum_j v_i W_ij t_j
        vW = torch.matmul(v, self.W)
        scores = torch.sum(vW * t, dim=-1)
        if self.bias is not None:
            scores = scores + self.bias
        return scores


class LABCLIPMatcher(nn.Module):
    """
    LABCLIP Linear Alignment Head: s(v, t) = v^T normalize(W t) + b.
    Applies a linear alignment transformation to the text embedding followed by L2 normalization,
    matching the architecture from Koishigarina et al. (2025).
    """
    def __init__(self, dim: int, use_bias: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim))
        self.bias = nn.Parameter(torch.zeros(1)) if use_bias else None

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # v: (B, D), t: (B, D)
        t_aligned = torch.matmul(t, self.W.T)
        t_norm = F.normalize(t_aligned, dim=-1)
        v_norm = F.normalize(v, dim=-1)
        scores = torch.sum(v_norm * t_norm, dim=-1)
        if self.bias is not None:
            scores = scores + self.bias
        return scores


def train_bilinear_matcher(
    v_pos: torch.Tensor,
    v_neg: torch.Tensor,
    t_pos: torch.Tensor,
    t_neg: torch.Tensor,
    epochs: int = 60,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    seed: int = 42,
    use_bias: bool = True,
) -> np.ndarray:
    """
    Train Bilinear Scorer using Margin Ranking Loss on 2x2 pairs.
    Returns learned W matrix as numpy array of shape (D, D).
    """
    torch.manual_seed(seed)
    dim = v_pos.shape[-1]
    model = BilinearMatcher(dim, use_bias=use_bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MarginRankingLoss(margin=0.1)

    n = len(v_pos)
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Positive pairs: (v+, t+), (v-, t-)
        s_pp = model(v_pos, t_pos)
        s_mm = model(v_neg, t_neg)

        # Negative pairs: (v+, t-), (v-, t+)
        s_pm = model(v_pos, t_neg)
        s_mp = model(v_neg, t_pos)

        target = torch.ones(n)
        loss = (criterion(s_pp, s_pm, target) + criterion(s_mm, s_mp, target) +
                criterion(s_pp, s_mp, target) + criterion(s_mm, s_pm, target)) / 4.0

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        W_learned = model.W.detach().cpu().numpy()
    return W_learned


def train_labclip_matcher(
    v_pos: torch.Tensor,
    v_neg: torch.Tensor,
    t_pos: torch.Tensor,
    t_neg: torch.Tensor,
    epochs: int = 60,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    seed: int = 42,
    use_bias: bool = True,
) -> np.ndarray:
    """
    Train LABCLIP Alignment Scorer using the exact same Margin Ranking Loss on 2x2 pairs.
    Returns learned W matrix as numpy array of shape (D, D).
    """
    torch.manual_seed(seed)
    dim = v_pos.shape[-1]
    model = LABCLIPMatcher(dim, use_bias=use_bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MarginRankingLoss(margin=0.1)

    n = len(v_pos)
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Positive pairs: (v+, t+), (v-, t-)
        s_pp = model(v_pos, t_pos)
        s_mm = model(v_neg, t_neg)

        # Negative pairs: (v+, t-), (v-, t+)
        s_pm = model(v_pos, t_neg)
        s_mp = model(v_neg, t_pos)

        target = torch.ones(n)
        loss = (criterion(s_pp, s_pm, target) + criterion(s_mm, s_mp, target) +
                criterion(s_pp, s_mp, target) + criterion(s_mm, s_pm, target)) / 4.0

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        W_learned = model.W.detach().cpu().numpy()
    return W_learned


def evaluate_bilinear_ablation(
    v_pos: torch.Tensor,
    v_neg: torch.Tensor,
    t_pos: torch.Tensor,
    t_neg: torch.Tensor,
    W_full: np.ndarray,
    W_labclip: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate scoring heads:
        1. Cosine:                   s = v^T t
        2. LABCLIP (Linear Align):   s = v^T normalize(W_lab t)
        3. Diagonal (Reweighting):   s = v^T D t   (D = diag(W))
        4. Off-diagonal (Cross-Int): s = v^T O t   (O = W - D)
        5. Full Bilinear (Joint):    s = v^T W t
    """
    D = np.diag(np.diag(W_full))
    O = W_full - D

    v_p, v_m = v_pos.numpy(), v_neg.numpy()
    t_p, t_m = t_pos.numpy(), t_neg.numpy()

    scorers = {
        "Cosine": lambda v, t: np.sum(v * t, axis=-1),
    }

    if W_labclip is not None:
        def labclip_score(v, t):
            t_proj = t @ W_labclip.T
            t_norm = t_proj / (np.linalg.norm(t_proj, axis=-1, keepdims=True) + 1e-12)
            v_norm = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
            return np.sum(v_norm * t_norm, axis=-1)
        scorers["LABCLIP (Linear Alignment)"] = labclip_score

    scorers["Diagonal (Reweighting)"] = lambda v, t: np.sum((v @ D) * t, axis=-1)
    scorers["Off-Diagonal (Cross-Inter)"] = lambda v, t: np.sum((v @ O) * t, axis=-1)
    scorers["Full Bilinear (Joint)"] = lambda v, t: np.sum((v @ W_full) * t, axis=-1)

    results = {}
    for name, score_fn in scorers.items():
        s_pp = score_fn(v_p, t_p)
        s_pm = score_fn(v_p, t_m)
        s_mp = score_fn(v_m, t_p)
        s_mm = score_fn(v_m, t_m)

        correct_min = np.minimum(s_pp, s_mm)
        wrong_max = np.maximum(s_pm, s_mp)
        m = correct_min - wrong_max

        acc = float(np.mean(m > 0) * 100.0)
        results[name] = {
            "accuracy_pct": acc,
            "margin_mean": float(np.mean(m)),
            "margin_std": float(np.std(m)),
            "margin_median": float(np.median(m)),
        }

    return results


# ============================================================
# Main Mechanistic Analysis Orchestrator
# ============================================================
def run_unary_mechanistic_analysis(
    csv_path: str = "benchmarks/data/images/beaf_counterfactual_6col.csv",
    output_dir: str = "logs/evaluation/unary_mechanistic_analysis",
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    target_objects: Optional[List[str]] = None,
    min_pairs_per_obj: int = 8,
    batch_size: int = 256,
    use_bias: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  4-Stage Unary Mechanistic Analysis (E1 ~ E4)            ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"  Model       : {model_name} ({pretrained}) | Device: {device}")
    print(f"  Input CSV   : {csv_path}")
    print(f"  Output Dir  : {output_dir}")
    print(f"  Use Bias    : {use_bias}\n")

    # Load dataset
    df = pd.read_csv(csv_path)
    if "object_in_image" in df.columns:
        if df["object_in_image"].dtype == object:
            df["object_in_image"] = df["object_in_image"].apply(
                lambda x: str(x).strip().lower() == "true"
            )
        else:
            df["object_in_image"] = df["object_in_image"].astype(bool)

    # Filter single-object unary pairs
    # Group by object_name and pair consecutive True/False rows
    objects_in_data = df["object_name"].unique().tolist()
    if target_objects is None:
        target_objects = [o for o in objects_in_data if "," not in str(o)]

    print(f"Total candidate single objects in CSV: {len(target_objects)}")

    # Load CLIP model
    print(f"Loading CLIP model '{model_name}' ({pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    embed_dim = 512

    # Container for all object results
    all_obj_records = []
    scatter_data_w = []
    scatter_data_d = []
    scatter_margins = []
    scatter_objects = []

    ablation_accs_summary = {
        "Cosine": [],
        "LABCLIP (Linear Alignment)": [],
        "Diagonal (Reweighting)": [],
        "Off-Diagonal (Cross-Inter)": [],
        "Full Bilinear (Joint)": [],
    }

    analyzed_objects = []

    for obj in target_objects:
        df_obj = df[df["object_name"] == obj].reset_index(drop=True)
        df_true = df_obj[df_obj["object_in_image"] == True].reset_index(drop=True)
        df_false = df_obj[df_obj["object_in_image"] == False].reset_index(drop=True)

        n_pairs = min(len(df_true), len(df_false))
        if n_pairs < min_pairs_per_obj:
            continue

        print(f"\n──────────────────────────────────────────────────────────")
        print(f"  Analyzing Object: [{obj}] (N = {n_pairs} counterfactual pairs)")
        print(f"──────────────────────────────────────────────────────────")

        img_paths_pos = df_true["image_path"].tolist()[:n_pairs]
        img_paths_neg = df_false["image_path"].tolist()[:n_pairs]
        t_pos_texts = df_true["positive_caption"].tolist()[:n_pairs]
        t_neg_texts = df_true["negative_caption"].tolist()[:n_pairs]

        # 1. Encode Images & Texts
        v_pos, mask_vp = encode_images_safely(model, preprocess, img_paths_pos, device, embed_dim)
        v_neg, mask_vn = encode_images_safely(model, preprocess, img_paths_neg, device, embed_dim)

        valid_idx = [i for i in range(n_pairs) if mask_vp[i] and mask_vn[i]]
        if len(valid_idx) < min_pairs_per_obj:
            print(f"  [Skip] Valid image pairs insufficient ({len(valid_idx)} < {min_pairs_per_obj})")
            continue

        v_pos = v_pos[valid_idx]
        v_neg = v_neg[valid_idx]
        t_pos_texts = [t_pos_texts[i] for i in valid_idx]
        t_neg_texts = [t_neg_texts[i] for i in valid_idx]

        t_pos = encode_texts_safely(model, tokenizer, t_pos_texts, device, batch_size)
        t_neg = encode_texts_safely(model, tokenizer, t_neg_texts, device, batch_size)

        analyzed_objects.append(obj)

        # ── Stage E1: Information Probing ──
        acc_v, std_v, w_I, b_I = fit_linear_probe(v_pos.numpy(), v_neg.numpy(), fit_intercept=use_bias)
        acc_t, std_t, w_T, b_T = fit_linear_probe(t_pos.numpy(), t_neg.numpy(), fit_intercept=use_bias)

        d_I = np.mean(v_pos.numpy(), axis=0) - np.mean(v_neg.numpy(), axis=0)
        d_T = np.mean(t_pos.numpy(), axis=0) - np.mean(t_neg.numpy(), axis=0)

        print(f"  [E1 Probe] Image Acc : {acc_v:5.1f}% (±{std_v:4.1f}%) | Text Acc : {acc_t:5.1f}% (±{std_t:4.1f}%)")

        # ── Stage E2: Cross-Modal Alignment ──
        align_metrics = compute_cross_modal_alignment(w_I, w_T, d_I, d_T)
        print(f"  [E2 Align] A_normal  : {align_metrics['a_normal']:+.4f} (p={align_metrics['p_val_normal']:.3f}) | "
              f"A_centroid: {align_metrics['a_centroid']:+.4f} (p={align_metrics['p_val_centroid']:.3f})")

        # ── Stage E3: 2x2 Score Matrix & Margin (M) ──
        margin_res = compute_2x2_margin(v_pos, v_neg, t_pos, t_neg)
        print(f"  [E3 Cosine] Margin M : {margin_res['margin_mean']:+.4f} ± {margin_res['margin_std']:.4f} | "
              f"2x2 Acc: {margin_res['accuracy_pct']:5.1f}%")

        scatter_data_w.append(align_metrics["a_normal"])
        scatter_data_d.append(align_metrics["a_centroid"])
        scatter_margins.append(margin_res["margin_mean"])
        scatter_objects.append(obj)

        # ── Stage E4: Bilinear Ablation & LABCLIP (W = D + O) ──
        W_learned = train_bilinear_matcher(v_pos, v_neg, t_pos, t_neg, use_bias=use_bias)
        W_labclip = train_labclip_matcher(v_pos, v_neg, t_pos, t_neg, use_bias=use_bias)
        ablation_res = evaluate_bilinear_ablation(v_pos, v_neg, t_pos, t_neg, W_learned, W_labclip=W_labclip)

        print("  [E4 Ablation]")
        for sc_name, sc_data in ablation_res.items():
            print(f"    - {sc_name:28s}: Acc = {sc_data['accuracy_pct']:5.1f}% | Margin = {sc_data['margin_mean']:+.4f}")
            ablation_accs_summary[sc_name].append(sc_data["accuracy_pct"])

        # Record summary row
        all_obj_records.append({
            "object": obj,
            "n_pairs": len(valid_idx),
            "e1_image_probe_acc": acc_v,
            "e1_text_probe_acc": acc_t,
            "e2_a_normal": align_metrics["a_normal"],
            "e2_a_centroid": align_metrics["a_centroid"],
            "e2_p_val_centroid": align_metrics["p_val_centroid"],
            "e3_cosine_margin_mean": margin_res["margin_mean"],
            "e3_cosine_2x2_acc": margin_res["accuracy_pct"],
            "e4_cosine_acc": ablation_res["Cosine"]["accuracy_pct"],
            "e4_labclip_acc": ablation_res.get("LABCLIP (Linear Alignment)", {}).get("accuracy_pct", 0.0),
            "e4_diagonal_acc": ablation_res["Diagonal (Reweighting)"]["accuracy_pct"],
            "e4_off_diagonal_acc": ablation_res["Off-Diagonal (Cross-Inter)"]["accuracy_pct"],
            "e4_full_bilinear_acc": ablation_res["Full Bilinear (Joint)"]["accuracy_pct"],
        })

    if not all_obj_records:
        print("No valid objects analyzed.")
        return

    # Export Summary Table
    df_summary = pd.DataFrame(all_obj_records)
    summary_csv = os.path.join(output_dir, "e1_to_e4_summary_table.csv")
    df_summary.to_csv(summary_csv, index=False)

    # Correlation Analysis
    corr_w, p_w = stats.pearsonr(scatter_data_w, scatter_margins)
    spear_w, _ = stats.spearmanr(scatter_data_w, scatter_margins)
    corr_d, p_d = stats.pearsonr(scatter_data_d, scatter_margins)
    spear_d, _ = stats.spearmanr(scatter_data_d, scatter_margins)

    print("\n" + "=" * 65)
    print("  Cross-Modal Alignment vs Cosine Margin Correlation:")
    print(f"  Normal Alignment   : Pearson r = {corr_w:+.4f} (p={p_w:.4f}), Spearman rho = {spear_w:+.4f}")
    print(f"  Centroid Alignment : Pearson r = {corr_d:+.4f} (p={p_d:.4f}), Spearman rho = {spear_d:+.4f}")
    print("=" * 65)

    # Render Figures
    _render_fig1_probing(df_summary, output_dir)
    _render_fig2_alignment(df_summary, output_dir)
    _render_fig3_scatter(scatter_data_d, scatter_margins, scatter_objects, corr_d, p_d, output_dir)
    _render_fig4_bilinear_ablation(ablation_accs_summary, output_dir)

    # Full JSON Report
    report = {
        "model": model_name,
        "pretrained": pretrained,
        "use_bias": use_bias,
        "n_objects_analyzed": len(analyzed_objects),
        "analyzed_objects": analyzed_objects,
        "e1_probing_mean": {
            "image_probe_acc": float(df_summary["e1_image_probe_acc"].mean()),
            "text_probe_acc": float(df_summary["e1_text_probe_acc"].mean()),
        },
        "e2_alignment_mean": {
            "a_normal_mean": float(df_summary["e2_a_normal"].mean()),
            "a_centroid_mean": float(df_summary["e2_a_centroid"].mean()),
        },
        "e3_correlation": {
            "pearson_r_centroid_margin": float(corr_d),
            "pearson_p_centroid_margin": float(p_d),
            "spearman_rho_centroid_margin": float(spear_d),
        },
        "e4_ablation_mean_accuracy": {
            k: float(np.mean(v)) for k, v in ablation_accs_summary.items()
        },
        "table_summary": df_summary.to_dict(orient="records"),
    }

    report_path = os.path.join(output_dir, "full_mechanistic_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print("  4-Stage Mechanistic Analysis Complete!")
    print("=" * 65)
    print(f"  1. Summary Table : {summary_csv}")
    print(f"  2. Fig 1 (E1)    : {output_dir}/fig1_representation_probing.png")
    print(f"  3. Fig 2 (E2)    : {output_dir}/fig2_alignment_distribution.png")
    print(f"  4. Fig 3 (E3)    : {output_dir}/fig3_alignment_vs_margin_scatter.png")
    print(f"  5. Fig 4 (E4)    : {output_dir}/fig4_bilinear_ablation_bar.png")
    print(f"  6. Full Report   : {report_path}")
    print("=" * 65 + "\n")


# ============================================================
# Visualizations
# ============================================================
def _render_fig1_probing(df_summary: pd.DataFrame, output_dir: str):
    """Figure 1: Representation Probing Accuracies across Objects."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_summary))
    width = 0.35

    ax.bar(x - width/2, df_summary["e1_image_probe_acc"], width, label="Image Probe Acc (Vision)", color="#1f77b4")
    ax.bar(x + width/2, df_summary["e1_text_probe_acc"], width, label="Text Probe Acc (Language)", color="#e74c3c")

    ax.axhline(y=50.0, color="gray", ls="--", lw=1.5, alpha=0.7, label="Chance Level (50%)")
    ax.set_ylabel("Linear Probe Accuracy (%)", fontsize=12)
    ax.set_title("E1: Representation Presence & Polarity Signals (Linear Accessibility)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df_summary["object"], rotation=35, ha="right", fontsize=10)
    ax.set_ylim(40, 105)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "fig1_representation_probing.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _render_fig2_alignment(df_summary: pd.DataFrame, output_dir: str):
    """Figure 2: Cross-Modal Alignment vs Random Null Distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.hist(df_summary["e2_a_normal"], bins=20, color="#3498db", edgecolor="black", alpha=0.75, label="Normal Align cos(w_I, w_T)")
    ax1.axvline(x=0.0, color="red", ls="--", lw=1.5, label="Null Expectation (0.0)")
    ax1.axvline(x=df_summary["e2_a_normal"].mean(), color="green", ls="-", lw=2,
                label=f"Mean = {df_summary['e2_a_normal'].mean():+.3f}")
    ax1.set_xlabel("Alignment cos(w_I, w_T)", fontsize=11)
    ax1.set_ylabel("Number of Objects", fontsize=11)
    ax1.set_title("Probe Normal Vector Alignment", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, ls="--", alpha=0.4)

    ax2 = axes[1]
    ax2.hist(df_summary["e2_a_centroid"], bins=20, color="#9b59b6", edgecolor="black", alpha=0.75, label="Centroid Align cos(d_I, d_T)")
    ax2.axvline(x=0.0, color="red", ls="--", lw=1.5, label="Null Expectation (0.0)")
    ax2.axvline(x=df_summary["e2_a_centroid"].mean(), color="green", ls="-", lw=2,
                label=f"Mean = {df_summary['e2_a_centroid'].mean():+.3f}")
    ax2.set_xlabel("Alignment cos(d_I, d_T)", fontsize=11)
    ax2.set_ylabel("Number of Objects", fontsize=11)
    ax2.set_title("Centroid Difference Vector Alignment", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig2_alignment_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _render_fig3_scatter(
    x_vals: List[float],
    y_vals: List[float],
    obj_labels: List[str],
    r: float,
    p: float,
    output_dir: str,
):
    """Figure 3: Centroid Alignment vs Cosine Margin M Scatter."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_vals, y_vals, color="#2c3e50", alpha=0.8, s=60, edgecolors="white", label="Objects")

    # Fit linear regression trend
    if len(x_vals) > 2:
        slope, intercept, _, _, _ = stats.linregress(x_vals, y_vals)
        x_line = np.linspace(min(x_vals), max(x_vals), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="#e74c3c", ls="-", lw=2,
                label=f"Trend (Pearson r={r:+.3f}, p={p:.3f})")

    ax.axhline(y=0.0, color="gray", ls="--", lw=1.2, alpha=0.7)
    ax.axvline(x=0.0, color="gray", ls="--", lw=1.2, alpha=0.7)

    # Annotate top objects
    for i, obj in enumerate(obj_labels):
        if abs(y_vals[i]) > 0.025 or x_vals[i] > 0.2:
            ax.annotate(obj, (x_vals[i], y_vals[i]), textcoords="offset points",
                        xytext=(4, 4), fontsize=8, alpha=0.85)

    ax.set_xlabel("Centroid Shift Alignment cos(d_I, d_T)", fontsize=12)
    ax.set_ylabel("Cosine Correctness Margin M", fontsize=12)
    ax.set_title("E3: Does Semantic Alignment Predict Cosine Matching Success?", fontsize=13, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "fig3_alignment_vs_margin_scatter.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _render_fig4_bilinear_ablation(ablation_accs: Dict[str, List[float]], output_dir: str):
    """Figure 4: Cosine vs LABCLIP vs Diagonal vs Off-Diagonal vs Full Bilinear."""
    names = list(ablation_accs.keys())
    means = [float(np.mean(ablation_accs[k])) for k in names]
    stds = [float(np.std(ablation_accs[k])) for k in names]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#e74c3c", "#9b59b6", "#f39c12", "#2980b9", "#27ae60"]

    bars = ax.bar(names, means, yerr=stds, color=colors[:len(names)], edgecolor="black", linewidth=1.2, width=0.55, capsize=6)
    ax.axhline(y=50.0, color="gray", ls="--", lw=1.5, alpha=0.7, label="Chance Level (50%)")

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f"{m:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("2x2 Polarity Matching Accuracy (%)", fontsize=12)
    ax.set_title("E4: Why Cosine Fails — Cosine vs LABCLIP vs Diagonal vs Off-Diagonal vs Full Bilinear", fontsize=13, fontweight="bold")
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9.5)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.legend(fontsize=10, loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "fig4_bilinear_ablation_bar.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-Stage Unary Mechanistic Analysis")
    parser.add_argument("--csv_path", type=str, default="benchmarks/data/images/beaf_counterfactual_6col.csv")
    parser.add_argument("--output_dir", type=str, default="logs/evaluation/unary_mechanistic_analysis")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--target_objects", nargs="+", default=None)
    parser.add_argument("--min_pairs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--no_bias", "--no-bias", action="store_true", default=False,
                        help="Disable bias/intercept in linear probes and Bilinear matching head (default: bias enabled)")
    args = parser.parse_args()

    run_unary_mechanistic_analysis(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        model_name=args.model,
        pretrained=args.pretrained,
        target_objects=args.target_objects,
        min_pairs_per_obj=args.min_pairs,
        batch_size=args.batch_size,
        use_bias=not args.no_bias,
    )
