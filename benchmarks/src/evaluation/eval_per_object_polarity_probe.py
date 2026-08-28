"""
Per-Object Polarity Probe and Alignment Causal Intervention Module.

For each object o:
  1. Extracts vision probe normal vector d_I^(o) from 1:1 counterfactual image pairs (I_orig vs I_cf).
  2. Extracts text probe normal vector d_T^(o) from diverse counterfactual caption pairs (T_pos vs T_neg).
  3. Computes closed-form orthogonal rotation matrix R^(o) in SO(d) such that R^(o) d_T^(o) = d_I^(o) (cos = 1.0).
  4. Evaluates 7 intervention conditions on 2x2 counterfactual matching:
       - Baseline: Standard Cosine (A = I)
       - Intervention 1: Closed-Form 2D Rotation (A = R^(o))
       - Intervention 2: Rank-1 Polar Adapter (A = A_rank1^(o))
       - Intervention 3: LABCLIP Linear Alignment (t -> normalize(W_lab t))
       - Intervention 4: Learned Low-Rank Bilinear (A = U V^T, rank=k)
       - Intervention 5: Learned Full Bilinear (A = W^(o))
       - Control: Random Orthogonal Rotation (A = R_rand)
  5. Measures whether aligning d_T to d_I, linear alignment (LABCLIP), or learning bilinear interaction causally fixes CLIP's negation matching failure.

Outputs:
  - per_object_intervention_results.csv
  - fig_intervention_7conditions_bar.png
  - fig_alignment_vs_gain_scatter.png
  - fig_per_object_gain_waterfall.png
  - per_object_intervention_summary.json

Usage:
    python -m benchmarks.src.evaluation.eval_per_object_polarity_probe \\
        --output_dir logs/evaluation/per_object_alignment_intervention \\
        --model ViT-B-32 --pretrained openai
"""

import os
import json
import argparse
import math
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import open_clip

# Reuse existing verified infrastructure
from benchmarks.src.analysis.config import get_layer_features as _get_feats, coerce_bool_column
from benchmarks.src.analysis.beaf.beaf_loader import load_and_verify_counterfactual_pairs
from benchmarks.src.analysis.beaf.vision_mechanisms import extract_vision_features_unified
from benchmarks.src.evaluation.eval_layerwise_linear_probe import extract_layerwise_feature_dict


# ============================================================
# 1. Closed-Form Mathematical Transformation Builders
# ============================================================
def build_closed_form_rotation(d_T: np.ndarray, d_I: np.ndarray) -> np.ndarray:
    """
    Constructs an exact d-dimensional orthogonal rotation matrix R in SO(d)
    such that R @ d_T = d_I and R^T @ R = I.
    
    Uses Rodrigues 2D plane rotation formula in R^d:
        u = d_T
        v = (d_I - (u^T d_I) u) / ||d_I - (u^T d_I) u||
        R = I + sin(theta) * (v u^T - u v^T) + (cos(theta) - 1) * (u u^T + v v^T)
    """
    u = d_T / (np.linalg.norm(d_T) + 1e-12)
    target = d_I / (np.linalg.norm(d_I) + 1e-12)
    d = len(u)

    cos_theta = float(np.dot(u, target))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Parallel or anti-parallel check
    if abs(cos_theta - 1.0) < 1e-7:
        return np.eye(d)
    if abs(cos_theta + 1.0) < 1e-7:
        # Anti-parallel: reflect across any orthogonal axis
        return -np.eye(d)

    v_raw = target - cos_theta * u
    v = v_raw / (np.linalg.norm(v_raw) + 1e-12)
    sin_theta = float(np.dot(v, target))

    # Construct R = I + sin_theta * (v u^T - u v^T) + (cos_theta - 1) * (u u^T + v v^T)
    u_col = u[:, np.newaxis]
    v_col = v[:, np.newaxis]

    vuT = np.dot(v_col, u_col.T)
    uvT = np.dot(u_col, v_col.T)
    uuT = np.dot(u_col, u_col.T)
    vvT = np.dot(v_col, v_col.T)

    R = np.eye(d) + sin_theta * (vuT - uvT) + (cos_theta - 1.0) * (uuT + vvT)
    return R


def build_rank1_adapter(d_T: np.ndarray, d_I: np.ndarray) -> np.ndarray:
    """
    Constructs a rank-1 polar adapter A_rank1 = I + (d_I - d_T) @ d_T^T
    such that A_rank1 @ d_T = d_I.
    """
    u = d_T / (np.linalg.norm(d_T) + 1e-12)
    target = d_I / (np.linalg.norm(d_I) + 1e-12)
    d = len(u)

    u_col = u[:, np.newaxis]
    diff_col = (target - u)[:, np.newaxis]
    A_rank1 = np.eye(d) + np.dot(diff_col, u_col.T)
    return A_rank1


def build_random_orthogonal_rotation(d: int, seed: int = 42) -> np.ndarray:
    """Generates a Haar-random orthogonal matrix in SO(d) via QR decomposition."""
    rng = np.random.RandomState(seed)
    H = rng.randn(d, d)
    Q, R = np.linalg.qr(H)
    # Ensure det(Q) == +1 (pure rotation)
    d_diag = np.diagonal(R)
    ph = d_diag / np.abs(d_diag)
    Q = Q * ph
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# ============================================================
# 2. Learnable Matchers (Full Bilinear, Low-Rank, LABCLIP)
# ============================================================
class BilinearMatcher(nn.Module):
    """Full Bilinear scoring head: s(v, t) = v^T W t"""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.eye(embed_dim))

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        v_proj = torch.matmul(v, self.W)
        return torch.sum(v_proj * t, dim=-1)


def train_bilinear_matcher(
    v_pos: torch.Tensor,
    v_neg: torch.Tensor,
    t_pos: torch.Tensor,
    t_neg: torch.Tensor,
    epochs: int = 150,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    margin: float = 0.1,
) -> np.ndarray:
    """Trains a full bilinear matrix W on counterfactual quad tuples."""
    embed_dim = v_pos.shape[-1]
    model = BilinearMatcher(embed_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MarginRankingLoss(margin=margin)
    n = v_pos.shape[0]

    for _ in range(epochs):
        optimizer.zero_grad()
        s_pp = model(v_pos, t_pos)
        s_mm = model(v_neg, t_neg)
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


class LowRankBilinearMatcher(nn.Module):
    """
    Low-Rank Bilinear scoring head: s(v, t) = (v U) . (t V) = v (U V^T) t
    where U, V in R^(d x k).
    """
    def __init__(self, embed_dim: int, rank: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank
        self.proj_v = nn.Linear(embed_dim, rank, bias=False)
        self.proj_t = nn.Linear(embed_dim, rank, bias=False)
        nn.init.normal_(self.proj_v.weight, std=0.02)
        nn.init.normal_(self.proj_t.weight, std=0.02)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        Av = self.proj_v(v)  # (B, k)
        Bt = self.proj_t(t)  # (B, k)
        return torch.sum(Av * Bt, dim=-1)

    def get_W(self) -> np.ndarray:
        U = self.proj_v.weight.T  # (d, k)
        V = self.proj_t.weight.T  # (d, k)
        W = torch.matmul(U, V.T)  # (d, d)
        return W.detach().cpu().numpy()


def train_lowrank_bilinear_matcher(
    v_pos: torch.Tensor,
    v_neg: torch.Tensor,
    t_pos: torch.Tensor,
    t_neg: torch.Tensor,
    rank: int = 32,
    epochs: int = 150,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    margin: float = 0.1,
) -> np.ndarray:
    """Trains a rank-k bilinear factorization W = U V^T on counterfactual quad tuples."""
    embed_dim = v_pos.shape[-1]
    model = LowRankBilinearMatcher(embed_dim, rank=rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MarginRankingLoss(margin=margin)
    n = v_pos.shape[0]

    for _ in range(epochs):
        optimizer.zero_grad()
        s_pp = model(v_pos, t_pos)
        s_mm = model(v_neg, t_neg)
        s_pm = model(v_pos, t_neg)
        s_mp = model(v_neg, t_pos)

        target = torch.ones(n)
        loss = (criterion(s_pp, s_pm, target) + criterion(s_mm, s_mp, target) +
                criterion(s_pp, s_mp, target) + criterion(s_mm, s_pm, target)) / 4.0

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        W_learned = model.get_W()
    return W_learned


class LABCLIPMatcher(nn.Module):
    """
    LABCLIP Linear Alignment Head: s(v, t) = normalize(v)^T normalize(W t)
    Matches the architecture from Koishigarina et al. (2025).
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.eye(embed_dim))

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_aligned = torch.matmul(t, self.W.T)
        t_norm = F.normalize(t_aligned, dim=-1)
        v_norm = F.normalize(v, dim=-1)
        return torch.sum(v_norm * t_norm, dim=-1)


def train_labclip_matcher(
    v_pos: torch.Tensor,
    v_neg: torch.Tensor,
    t_pos: torch.Tensor,
    t_neg: torch.Tensor,
    epochs: int = 150,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    margin: float = 0.1,
) -> np.ndarray:
    """Trains a LABCLIP linear alignment matrix W using Margin Ranking Loss on 2x2 pairs."""
    embed_dim = v_pos.shape[-1]
    model = LABCLIPMatcher(embed_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MarginRankingLoss(margin=margin)
    n = v_pos.shape[0]

    for _ in range(epochs):
        optimizer.zero_grad()
        s_pp = model(v_pos, t_pos)
        s_mm = model(v_neg, t_neg)
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


# ============================================================
# 3. Evaluation Engine for Intervention Conditions
# ============================================================
def evaluate_condition_scoring(
    v_p: np.ndarray,
    v_m: np.ndarray,
    t_p: np.ndarray,
    t_m: np.ndarray,
    A: np.ndarray,
    d_I: np.ndarray,
    d_T: np.ndarray,
    is_labclip: bool = False,
) -> Dict[str, float]:
    """
    Evaluates scoring for a given transformation matrix A:
      - Standard linear/bilinear: s(v, t) = (v @ A) . t
      - LABCLIP linear alignment: s(v, t) = normalize(v) . normalize(t @ A.T)
    """
    if is_labclip:
        # LABCLIP direction alignment
        At_d = np.dot(d_T, A.T)
        norm_At = np.linalg.norm(At_d) + 1e-12
        norm_dI = np.linalg.norm(d_I) + 1e-12
        dir_alignment = float(np.dot(d_I, At_d) / (norm_dI * norm_At))

        # LABCLIP scoring
        def score_fn(v, t):
            t_proj = t @ A.T
            t_norm = t_proj / (np.linalg.norm(t_proj, axis=-1, keepdims=True) + 1e-12)
            v_norm = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
            return np.sum(v_norm * t_norm, axis=-1)

        s_pp = score_fn(v_p, t_p)
        s_pm = score_fn(v_p, t_m)
        s_mp = score_fn(v_m, t_p)
        s_mm = score_fn(v_m, t_m)
    else:
        # 1. Compute Direction Alignment
        At_d = np.dot(A, d_T)
        norm_At = np.linalg.norm(At_d) + 1e-12
        norm_dI = np.linalg.norm(d_I) + 1e-12
        dir_alignment = float(np.dot(d_I, At_d) / (norm_dI * norm_At))

        # 2. Compute 2x2 Scores
        s_pp = np.sum((v_p @ A) * t_p, axis=-1)
        s_pm = np.sum((v_p @ A) * t_m, axis=-1)
        s_mp = np.sum((v_m @ A) * t_p, axis=-1)
        s_mm = np.sum((v_m @ A) * t_m, axis=-1)

    correct_min = np.minimum(s_pp, s_mm)
    wrong_max = np.maximum(s_pm, s_mp)
    m = correct_min - wrong_max

    acc_joint = float(np.mean(m > 0) * 100.0)
    acc_pos = float(np.mean(s_pp > s_pm) * 100.0)
    acc_neg = float(np.mean(s_mm > s_mp) * 100.0)
    acc_pairwise_avg = (acc_pos + acc_neg) / 2.0

    return {
        "direction_alignment": dir_alignment,
        "acc_joint_pct": acc_joint,
        "acc_pairwise_avg_pct": acc_pairwise_avg,
        "acc_pos_pct": acc_pos,
        "acc_neg_pct": acc_neg,
        "margin_mean": float(np.mean(m)),
        "margin_std": float(np.std(m)),
        "margin_median": float(np.median(m)),
    }


# ============================================================
# 4. Main Per-Object Orchestrator
# ============================================================
def run_per_object_alignment_intervention(
    vision_csv: str = "benchmarks/data/images/beaf_counterfactual_6col.csv",
    text_csv: str = "benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv",
    image_root: str = "benchmarks/data/images",
    output_dir: str = "logs/evaluation/per_object_alignment_intervention",
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    rank: int = 32,
    min_pairs_per_obj: int = 20,
    batch_size: int = 128,
    seed: int = 42,
    use_bias: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  Per-Object Polarity Probe & Alignment Intervention       ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"  Model       : {model_name} ({pretrained}) | Device: {device}")
    print(f"  Output Dir  : {output_dir}")
    print(f"  Min Pairs   : {min_pairs_per_obj}")
    print(f"  Rank (k)    : {rank}")
    print(f"  Use Bias    : {use_bias}\n")

    # 1. Load CLIP Model
    print(f"Loading CLIP model '{model_name}' ({pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    embed_dim = 512

    # 2. Load Vision 1:1 Counterfactual Pairs
    print("\nLoading and verifying Vision 1:1 Counterfactual Pairs...")
    df_raw, df_vis_pairs, _ = load_and_verify_counterfactual_pairs(vision_csv, image_root)
    vis_orig = extract_vision_features_unified(model, preprocess, df_vis_pairs["orig_path"].tolist(), device, batch_size)
    vis_cf = extract_vision_features_unified(model, preprocess, df_vis_pairs["cf_path"].tolist(), device, batch_size)

    flags_orig = np.array(vis_orig.get("loaded_flags", [True] * len(df_vis_pairs)))
    flags_cf = np.array(vis_cf.get("loaded_flags", [True] * len(df_vis_pairs)))
    valid_vis = flags_orig & flags_cf
    df_vis_pairs = df_vis_pairs[valid_vis].reset_index(drop=True)
    feats_vis_orig = vis_orig["final_l2norm"][valid_vis]
    feats_vis_cf = vis_cf["final_l2norm"][valid_vis]

    # 3. Load Text Diverse Counterfactual Captions
    print("\nLoading Text Diverse Counterfactual Captions...")
    df_txt = pd.read_csv(text_csv)
    coerce_bool_column(df_txt, "object_in_image")

    df_txt_unique = df_txt[df_txt["object_in_image"] == True].drop_duplicates(
        subset=["positive_caption", "negative_caption"]
    ).reset_index(drop=True)

    obj_txt_aff: Dict[str, List[str]] = {}
    obj_txt_neg: Dict[str, List[str]] = {}
    for _, row in df_txt_unique.iterrows():
        a = str(row["object_a"]).strip() if "object_a" in row.index else str(row["object_name"]).split(",")[0].strip()
        b = str(row["object_b"]).strip() if "object_b" in row.index else str(row["object_name"]).split(",")[1].strip()
        pos_c = str(row["positive_caption"]).strip()
        neg_c = str(row["negative_caption"]).strip()

        obj_txt_aff.setdefault(a, []).append(pos_c)
        obj_txt_neg.setdefault(a, []).append(neg_c)
        obj_txt_aff.setdefault(b, []).append(neg_c)
        obj_txt_neg.setdefault(b, []).append(pos_c)

    # Encode all unique texts safely
    all_sents = []
    seen = set()
    for o in obj_txt_aff:
        for c in obj_txt_aff[o] + obj_txt_neg[o]:
            if c not in seen:
                all_sents.append(c)
                seen.add(c)
    sent_idx = {s: i for i, s in enumerate(all_sents)}
    print(f"  Encoding {len(all_sents)} unique sentences...")
    global_txt_feats = extract_layerwise_feature_dict(model, tokenizer, all_sents, device, "eot", batch_size)
    txt_final_feats = global_txt_feats["Final (L2 Normed)"]

    # 4. Find Common Candidate Objects
    vis_objects = set(df_vis_pairs["object_name"].unique())
    txt_objects = set([o for o in obj_txt_aff if len(obj_txt_aff[o]) >= min_pairs_per_obj and len(obj_txt_neg[o]) >= min_pairs_per_obj])
    common_objects = sorted(list(vis_objects.intersection(txt_objects)))
    common_objects = [o for o in common_objects if "," not in str(o)]

    print(f"\nTotal common valid objects for Per-Object Intervention: {len(common_objects)}")

    per_obj_records = []
    condition_names = [
        "1_Baseline_Cosine",
        "2_Closed_Form_Rotation",
        "3_Rank1_Polar_Adapter",
        "4_LABCLIP_Linear_Alignment",
        "5_Learned_LowRank_Bilinear",
        "6_Learned_Full_Bilinear",
        "7_Control_Random_Rotation",
    ]

    for obj in common_objects:
        # ── Step 1: Vision Normal d_I^(o) ──
        vis_mask = (df_vis_pairs["object_name"].values == obj)
        n_vis = int(np.sum(vis_mask))
        if n_vis < min_pairs_per_obj:
            continue

        X_v_orig = feats_vis_orig[vis_mask]
        X_v_cf = feats_vis_cf[vis_mask]

        # Vision Probe fitting
        X_v_all = np.vstack([X_v_orig, X_v_cf])
        y_v_all = np.array([1] * n_vis + [0] * n_vis)
        clf_v = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, fit_intercept=use_bias)
        clf_v.fit(X_v_all, y_v_all)
        w_v = clf_v.coef_[0]
        d_I = w_v / (np.linalg.norm(w_v) + 1e-12)

        # ── Step 2: Text Normal d_T^(o) ──
        aff_s = obj_txt_aff[obj]
        neg_s = obj_txt_neg[obj]
        n_txt = min(len(aff_s), len(neg_s))
        aff_s = aff_s[:n_txt]
        neg_s = neg_s[:n_txt]

        X_t_aff = txt_final_feats[[sent_idx[s] for s in aff_s]]
        X_t_neg = txt_final_feats[[sent_idx[s] for s in neg_s]]

        # Text Probe fitting
        X_t_all = np.vstack([X_t_aff, X_t_neg])
        y_t_all = np.array([1] * n_txt + [0] * n_txt)
        clf_t = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, fit_intercept=use_bias)
        clf_t.fit(X_t_all, y_t_all)
        w_t = clf_t.coef_[0]
        d_T = w_t / (np.linalg.norm(w_t) + 1e-12)

        # ── Step 3: Align 2x2 Evaluation Pairs ──
        n_eval = min(n_vis, n_txt)
        v_p = X_v_orig[:n_eval]
        v_m = X_v_cf[:n_eval]
        t_p = X_t_aff[:n_eval]
        t_m = X_t_neg[:n_eval]

        # ── Step 4: Build 7 Transformation Matrices A^(o) ──
        A_matrices = {
            "1_Baseline_Cosine": (np.eye(embed_dim), False),
            "2_Closed_Form_Rotation": (build_closed_form_rotation(d_T, d_I), False),
            "3_Rank1_Polar_Adapter": (build_rank1_adapter(d_T, d_I), False),
            "4_LABCLIP_Linear_Alignment": (train_labclip_matcher(
                torch.tensor(v_p, dtype=torch.float32),
                torch.tensor(v_m, dtype=torch.float32),
                torch.tensor(t_p, dtype=torch.float32),
                torch.tensor(t_m, dtype=torch.float32),
            ), True),
            "5_Learned_LowRank_Bilinear": (train_lowrank_bilinear_matcher(
                torch.tensor(v_p, dtype=torch.float32),
                torch.tensor(v_m, dtype=torch.float32),
                torch.tensor(t_p, dtype=torch.float32),
                torch.tensor(t_m, dtype=torch.float32),
                rank=rank,
            ), False),
            "6_Learned_Full_Bilinear": (train_bilinear_matcher(
                torch.tensor(v_p, dtype=torch.float32),
                torch.tensor(v_m, dtype=torch.float32),
                torch.tensor(t_p, dtype=torch.float32),
                torch.tensor(t_m, dtype=torch.float32),
            ), False),
            "7_Control_Random_Rotation": (build_random_orthogonal_rotation(embed_dim, seed=seed), False),
        }

        # ── Step 5: Evaluate All 7 Conditions ──
        obj_row = {
            "object_name": obj,
            "n_eval_pairs": n_eval,
            "raw_cos_dI_dT": float(np.dot(d_I, d_T)),
        }

        for cond_name, (A, is_labclip) in A_matrices.items():
            metrics = evaluate_condition_scoring(v_p, v_m, t_p, t_m, A, d_I, d_T, is_labclip=is_labclip)
            for k, val in metrics.items():
                obj_row[f"{cond_name}_{k}"] = val

        per_obj_records.append(obj_row)

    df_results = pd.DataFrame(per_obj_records)
    csv_out = os.path.join(output_dir, "per_object_intervention_results.csv")
    df_results.to_csv(csv_out, index=False)
    print(f"\n  Saved Results CSV: {csv_out}")

    # 5. Generate Comparative Figures and Summary JSON
    generate_intervention_visualizations(df_results, output_dir, condition_names, rank=rank)


# ============================================================
# 5. Visualization & Reporting Engine
# ============================================================
def generate_intervention_visualizations(
    df: pd.DataFrame,
    output_dir: str,
    condition_names: List[str],
    rank: int = 32,
):
    print("\nGenerating Intervention Visualizations & Summary Report...")

    summary_dict = {}
    cond_labels = [
        "Cosine (Baseline)",
        "Closed-Form Rotation R",
        "Rank-1 Adapter",
        "LABCLIP (Linear Align)",
        f"LowRank Bilinear (k={rank})",
        "Full Bilinear W",
        "Random Rotation (Control)",
    ]

    mean_aligns = []
    mean_joint_accs = []
    mean_margins = []
    mean_pos_accs = []
    mean_neg_accs = []

    for c in condition_names:
        align = float(df[f"{c}_direction_alignment"].mean())
        j_acc = float(df[f"{c}_acc_joint_pct"].mean())
        margin = float(df[f"{c}_margin_mean"].mean())
        p_acc = float(df[f"{c}_acc_pos_pct"].mean())
        n_acc = float(df[f"{c}_acc_neg_pct"].mean())

        mean_aligns.append(align)
        mean_joint_accs.append(j_acc)
        mean_margins.append(margin)
        mean_pos_accs.append(p_acc)
        mean_neg_accs.append(n_acc)

        summary_dict[c] = {
            "direction_alignment_mean": align,
            "acc_joint_mean_pct": j_acc,
            "acc_joint_std_pct": float(df[f"{c}_acc_joint_pct"].std()),
            "margin_mean": margin,
            "acc_pos_mean_pct": p_acc,
            "acc_neg_mean_pct": n_acc,
        }

    # ── Figure 1: 7-Condition Comparison Bar Chart ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    x = np.arange(len(condition_names))
    colors = ["#7f8c8d", "#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#1abc9c", "#e74c3c"]

    # Subplot 1: 2x2 Joint Matching Accuracy
    bars1 = ax1.bar(x, mean_joint_accs, color=colors[:len(condition_names)], edgecolor="black", width=0.55)
    ax1.set_ylabel("2×2 Joint Exact Matching Accuracy (%)", fontsize=12)
    ax1.set_title("2×2 Matching Accuracy across Intervention Conditions", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cond_labels, rotation=25, ha="right", fontsize=9.5)
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", ls="--", alpha=0.4)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Subplot 2: Direction Alignment cos(d_I, A d_T)
    bars2 = ax2.bar(x, mean_aligns, color=colors[:len(condition_names)], edgecolor="black", width=0.55)
    ax2.set_ylabel("Direction Alignment cos(d_I, A d_T)", fontsize=12)
    ax2.set_title("Polarity Direction Alignment cos(d_I, A d_T)", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cond_labels, rotation=25, ha="right", fontsize=9.5)
    ax2.set_ylim(-0.2, 1.15)
    ax2.axhline(0, color="gray", lw=1)
    ax2.grid(axis="y", ls="--", alpha=0.4)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f"{yval:+.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    plt.tight_layout()
    out_fig1 = os.path.join(output_dir, "fig_intervention_7conditions_bar.png")
    plt.savefig(out_fig1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_fig1}")

    # ── Figure 2: Alignment vs Gain Scatter Plot ──
    fig, ax = plt.subplots(figsize=(9, 6))
    align_gains = df["2_Closed_Form_Rotation_direction_alignment"] - df["1_Baseline_Cosine_direction_alignment"]
    margin_gains = df["2_Closed_Form_Rotation_margin_mean"] - df["1_Baseline_Cosine_margin_mean"]

    ax.scatter(align_gains, margin_gains, color="#2ecc71", edgecolors="black", s=80, alpha=0.85)
    ax.axhline(0, color="red", ls="--", lw=1.2, alpha=0.7)
    ax.axvline(0, color="gray", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel(r"Alignment Gain $\Delta \cos(d_I, R d_T) = 1.0 - \cos(d_I, d_T)$", fontsize=12)
    ax.set_ylabel(r"Margin Gain $\Delta M = M_{R} - M_{\mathrm{cos}}$", fontsize=12)
    ax.set_title("Per-Object Causal Effect: Direction Alignment vs Margin Gain", fontsize=13, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4)

    if len(align_gains) > 2:
        m_poly, b_poly = np.polyfit(align_gains, margin_gains, 1)
        x_trend = np.linspace(align_gains.min(), align_gains.max(), 50)
        ax.plot(x_trend, m_poly * x_trend + b_poly, color="#27ae60", lw=2, label=f"Fit (slope={m_poly:.3f})")
        ax.legend(fontsize=10)

    plt.tight_layout()
    out_fig2 = os.path.join(output_dir, "fig_alignment_vs_gain_scatter.png")
    plt.savefig(out_fig2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_fig2}")

    # ── Figure 3: Per-Object Accuracy Waterfall Chart ──
    df_sorted = df.sort_values(by="2_Closed_Form_Rotation_acc_joint_pct", ascending=False).reset_index(drop=True)
    top_n = min(25, len(df_sorted))
    top_objs = df_sorted.head(top_n)

    fig, ax = plt.subplots(figsize=(14, 6))
    x_top = np.arange(len(top_objs))
    w = 0.35

    ax.bar(x_top - w/2, top_objs["1_Baseline_Cosine_acc_joint_pct"], w, label="Baseline Cosine", color="#7f8c8d")
    ax.bar(x_top + w/2, top_objs["2_Closed_Form_Rotation_acc_joint_pct"], w, label="Closed-Form Rotation R", color="#2ecc71")

    ax.set_ylabel("2×2 Joint Matching Accuracy (%)", fontsize=12)
    ax.set_title(f"Top-{top_n} Objects (N >= {rank}): Impact of Closed-Form Rotation R^(o) on 2×2 Matching", fontsize=13, fontweight="bold")
    ax.set_xticks(x_top)
    ax.set_xticklabels(top_objs["object_name"], rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.legend(fontsize=11)

    plt.tight_layout()
    out_fig3 = os.path.join(output_dir, "fig_per_object_gain_waterfall.png")
    plt.savefig(out_fig3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_fig3}")

    # ── Save Full Summary JSON ──
    json_out = os.path.join(output_dir, "per_object_intervention_summary.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"  Saved Summary JSON: {json_out}\n")


# ============================================================
# Main CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Per-Object Polarity Probe & Alignment Intervention")
    parser.add_argument("--vision_csv", type=str, default="benchmarks/data/images/beaf_counterfactual_6col.csv")
    parser.add_argument("--text_csv", type=str, default="benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv")
    parser.add_argument("--image_root", type=str, default="benchmarks/data/images")
    parser.add_argument("--output_dir", type=str, default="logs/evaluation/per_object_alignment_intervention")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--rank", type=int, default=32, help="Rank k for Low-Rank Bilinear Matcher (default: 32)")
    parser.add_argument("--min_pairs", type=int, default=20, help="Minimum counterfactual pairs per object (default: 20)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_bias", "--no-bias", action="store_true", default=False,
                        help="Disable bias/intercept in linear probes (default: bias enabled)")
    args = parser.parse_args()

    run_per_object_alignment_intervention(
        vision_csv=args.vision_csv,
        text_csv=args.text_csv,
        image_root=args.image_root,
        output_dir=args.output_dir,
        model_name=args.model,
        pretrained=args.pretrained,
        rank=args.rank,
        min_pairs_per_obj=args.min_pairs,
        batch_size=args.batch_size,
        seed=args.seed,
        use_bias=not args.no_bias,
    )


if __name__ == "__main__":
    main()
