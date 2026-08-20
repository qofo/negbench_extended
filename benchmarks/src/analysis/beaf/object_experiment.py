"""
Single-Object & Multi-Object Generalization Experiment Module for BEAF.

Provides utilities for:
1. Dynamic grammar formatting for object names (articles: a/an, plurals: s/es/people).
2. Balanced 1:1 sampling of Present (object_in_image=True) vs Absent (object_in_image=False) images per object.
3. 4-Way Cross Cosine Similarity Analysis across expanded positive & negative templates.
4. Evaluation of Zero-shot transfer and probe metrics on single-object benchmarks.
"""

import os
import json
from analysis.config import l2_normalize
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# Import PyTorch probe models and unified classifier factory from probe_factory
from analysis.beaf.probe_factory import (
    create_probe_classifier,
    PyTorchProbeEstimator,
    MLPVisionPyTorch as PyTorchMLPProbe,
    LowRankBilinearPyTorch as PyTorchLowRankBilinearProbe,
)


class VisionProbeWrapper:
    def __init__(self, probe_type: str, model_obj: Any):
        self.probe_type = probe_type
        self.model_obj = model_obj

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.empty((0,), dtype=np.float32)
        if self.probe_type == "quadratic":
            X_quad = np.hstack([X, X ** 2])
            return self.model_obj.decision_function(X_quad)
        elif hasattr(self.model_obj, "decision_function"):
            return self.model_obj.decision_function(X)
        else:
            # Fallback for models with predict_proba
            probs = self.model_obj.predict_proba(X)[:, 1]
            return (probs - 0.5) * 2.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if len(X) == 0:
            return 0.5
        preds = self.predict(X)
        return float(np.mean(preds == y))


def _fit_single_vision_probe(probe_type: str, X_tr: np.ndarray, y_tr: np.ndarray, C: float = 1.0) -> VisionProbeWrapper:
    """Helper to instantiate and fit vision probe via unified probe_factory."""
    y_binary = np.where(y_tr == 1, 1, 0)
    
    if probe_type == "linear":
        clf = create_probe_classifier("logistic", C=C, seed=42)
        clf.fit(X_tr, y_binary)
        return VisionProbeWrapper("linear", clf)
    elif probe_type == "quadratic":
        X_tr_quad = np.hstack([X_tr, X_tr ** 2])
        clf = create_probe_classifier("logistic", C=C, seed=42)
        clf.fit(X_tr_quad, y_binary)
        return VisionProbeWrapper("quadratic", clf)
    elif probe_type == "poly_kernel":
        clf = SVC(kernel="poly", degree=2, C=C, random_state=42)
        clf.fit(X_tr, y_tr)
        return VisionProbeWrapper("poly_kernel", clf)
    elif probe_type == "mlp":
        clf = create_probe_classifier("mlp", hidden_dim=64, epochs=80, seed=42)
        clf.fit(X_tr, y_binary)
        return VisionProbeWrapper("mlp", clf)
    elif probe_type in ["low_rank_bilinear", "bilinear_lowrank"]:
        clf = create_probe_classifier("bilinear_lowrank", rank=4, epochs=80, seed=42)
        clf.fit(X_tr, y_binary)
        return VisionProbeWrapper("low_rank_bilinear", clf)
    else:
        clf = create_probe_classifier("logistic", C=C, seed=42)
        clf.fit(X_tr, y_binary)
        return VisionProbeWrapper("linear", clf)


def train_eval_vision_high_order_probe(
    pos_v_emb: np.ndarray,
    neg_v_emb: np.ndarray,
    probe_type: str = "linear",
    C: float = 1.0,
    cv_folds: int = 5
) -> Tuple[float, float, Optional[VisionProbeWrapper]]:
    """Train and evaluate 1st-degree or 2nd-degree high-order non-linear vision probes."""
    if len(pos_v_emb) == 0 or len(neg_v_emb) == 0:
        return 0.5, 0.0, None

    X_v = np.vstack([pos_v_emb, neg_v_emb])
    y_v = np.array([1] * len(pos_v_emb) + [-1] * len(neg_v_emb))

    n_samples = len(y_v)
    effective_folds = min(cv_folds, n_samples // 2)

    if effective_folds >= 2:
        skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X_v, y_v):
            probe = _fit_single_vision_probe(probe_type, X_v[train_idx], y_v[train_idx], C=C)
            sc = probe.score(X_v[val_idx], y_v[val_idx])
            scores.append(sc)
        mean_acc, std_acc = float(np.mean(scores)), float(np.std(scores))
    else:
        mean_acc, std_acc = 0.5, 0.0

    fitted_probe = _fit_single_vision_probe(probe_type, X_v, y_v, C=C)
    return mean_acc, std_acc, fitted_probe



def run_single_object_train_val_experiment(
    df_balanced: pd.DataFrame,
    object_name: str,
    neg_prompts: List[str],
    pos_prompts: List[str],
    neg_groups: List[str],
    pos_groups: List[str],
    model: torch.nn.Module,
    preprocess: callable,
    tokenizer: callable,
    device: str = "cuda",
    batch_size: int = 64,
    train_ratio: float = 0.7,
    random_state: int = 42,
    vision_probe_type: str = "linear",
) -> Dict[str, Any]:
    """Dedicated Single-Object Experiment with 70:30 Train/Val Split."""
    model.eval()

    # 1. Feature Extraction
    with torch.no_grad():
        pos_tokens = tokenizer(pos_prompts).to(device)
        neg_tokens = tokenizer(neg_prompts).to(device)
        pos_t_emb = model.encode_text(pos_tokens, normalize=True).cpu().numpy()
        neg_t_emb = model.encode_text(neg_tokens, normalize=True).cpu().numpy()

    pos_img_df = df_balanced[df_balanced["object_in_image"] == True]
    neg_img_df = df_balanced[df_balanced["object_in_image"] == False]

    def _extract_img_embeds(paths: List[str]) -> np.ndarray:
        embeds = []
        with torch.no_grad():
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i : i + batch_size]
                tensors = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths if os.path.exists(p)]
                if tensors:
                    b_tens = torch.stack(tensors).to(device)
                    v_emb = model.encode_image(b_tens, normalize=True).cpu().numpy()
                    embeds.append(v_emb)
        return np.vstack(embeds) if embeds else np.empty((0, pos_t_emb.shape[1]))

    pos_v_emb = _extract_img_embeds(pos_img_df["abs_image_path"].tolist())
    neg_v_emb = _extract_img_embeds(neg_img_df["abs_image_path"].tolist())

    if len(pos_v_emb) < 4 or len(neg_v_emb) < 4 or len(pos_t_emb) < 4 or len(neg_t_emb) < 4:
        return {"error": f"Insufficient samples for object {object_name}"}

    # 2. Train/Val Splits (70% Train, 30% Val)
    pos_t_tr, pos_t_val = train_test_split(pos_t_emb, train_size=train_ratio, random_state=random_state)
    neg_t_tr, neg_t_val = train_test_split(neg_t_emb, train_size=train_ratio, random_state=random_state)

    pos_v_tr, pos_v_val = train_test_split(pos_v_emb, train_size=train_ratio, random_state=random_state)
    neg_v_tr, neg_v_val = train_test_split(neg_v_emb, train_size=train_ratio, random_state=random_state)

    X_t_tr = np.vstack([pos_t_tr, neg_t_tr])
    y_t_tr = np.array([1] * len(pos_t_tr) + [-1] * len(neg_t_tr))
    X_t_val = np.vstack([pos_t_val, neg_t_val])
    y_t_val = np.array([1] * len(pos_t_val) + [-1] * len(neg_t_val))

    X_v_tr = np.vstack([pos_v_tr, neg_v_tr])
    y_v_tr = np.array([1] * len(pos_v_tr) + [-1] * len(neg_v_tr))
    X_v_val = np.vstack([pos_v_val, neg_v_val])
    y_v_val = np.array([1] * len(pos_v_val) + [-1] * len(neg_v_val))

    # 3. Train Probes on Train Split
    t_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
    t_clf.fit(X_t_tr, y_t_tr)

    v_clf = _fit_single_vision_probe(vision_probe_type, X_v_tr, y_v_tr)

    # 4. Measure Val Performance for Modality-Specific Probes
    val_t_acc = float(t_clf.score(X_t_val, y_t_val))
    val_v_acc = float(v_clf.score(X_v_val, y_v_val))


    # 5. Measure 4-Way Multimodal Joint Scoring on Val Split S(v, t) = f_V(v) * f_T(t)
    f_V_pos_val = v_clf.decision_function(pos_v_val)  # [M_val]
    f_V_neg_val = v_clf.decision_function(neg_v_val)  # [M_val]

    f_T_pos_val = t_clf.decision_function(pos_t_val)  # [N_pos_val]
    f_T_neg_val = t_clf.decision_function(neg_t_val)  # [N_neg_val]

    # Outer product matrices S(v, t) = f_V(v) * f_T(t)
    S_11 = np.outer(f_V_pos_val, f_T_pos_val)  # (Present Img, Pos Text) -> Target: S > 0
    S_22 = np.outer(f_V_neg_val, f_T_neg_val)  # (Absent Img, Neg Text) -> Target: S > 0
    S_12 = np.outer(f_V_pos_val, f_T_neg_val)  # (Present Img, Neg Text) -> Target: S < 0
    S_21 = np.outer(f_V_neg_val, f_T_pos_val)  # (Absent Img, Pos Text) -> Target: S < 0

    # Quadrant Accuracies
    acc_S11_high = float(np.mean(S_11 > 0))  # Present + Pos Text -> High (>0)
    acc_S22_high = float(np.mean(S_22 > 0))  # Absent + Neg Text -> High (>0)
    acc_S12_low  = float(np.mean(S_12 < 0))  # Present + Neg Text -> Low (<0)
    acc_S21_low  = float(np.mean(S_21 < 0))  # Absent + Pos Text -> Low (<0)

    val_joint_sign_consistency_acc = float(np.mean([acc_S11_high, acc_S22_high, acc_S12_low, acc_S21_low]))

    return {
        "object_name": object_name,
        "n_train_image_pairs": int(len(pos_v_tr)),
        "n_val_image_pairs": int(len(pos_v_val)),
        "n_train_pos_templates": int(len(pos_t_tr)),
        "n_val_pos_templates": int(len(pos_t_val)),

        # Modality-Specific Val Accuracies
        "val_text_probe_acc": val_t_acc,
        "val_vision_probe_acc": val_v_acc,

        # 4-Way Quadrant Mean Scores
        "mean_score_Q1_pos_v_pos_t": float(np.mean(S_11)),
        "mean_score_Q2_neg_v_neg_t": float(np.mean(S_22)),
        "mean_score_Q3_pos_v_neg_t": float(np.mean(S_12)),
        "mean_score_Q4_neg_v_pos_t": float(np.mean(S_21)),

        # 4-Way Sign-Consistency Rule Accuracies
        "acc_Q1_pos_v_pos_t_is_high": acc_S11_high,
        "acc_Q2_neg_v_neg_t_is_high": acc_S22_high,
        "acc_Q3_pos_v_neg_t_is_low":  acc_S12_low,
        "acc_Q4_neg_v_pos_t_is_low":  acc_S21_low,
        "val_joint_sign_consistency_acc": val_joint_sign_consistency_acc,
    }



def evaluate_text_linear_probe_single_object(
    pos_t_emb: np.ndarray,
    neg_t_emb: np.ndarray,
    C: float = 1.0,
    cv_folds: int = 5
) -> Tuple[float, float, LogisticRegression]:
    """Train and evaluate a Logistic Regression linear probe on text embeddings for a SINGLE object.
    
    Returns:
        (mean_cv_accuracy, std_cv_accuracy, fitted_classifier)
    """
    X_text = np.vstack([pos_t_emb, neg_t_emb])
    y_text = np.array([1] * len(pos_t_emb) + [-1] * len(neg_t_emb))

    clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_text, y_text, cv=skf, scoring="accuracy")

    clf.fit(X_text, y_text)
    return float(scores.mean()), float(scores.std()), clf


def evaluate_text_linear_probe_cross_object(
    train_pos_t_emb: np.ndarray,
    train_neg_t_emb: np.ndarray,
    test_pos_t_emb: np.ndarray,
    test_neg_t_emb: np.ndarray,
    C: float = 1.0
) -> float:
    """Train linear probe on source object(s) text embeddings, and evaluate on unseen target object text embeddings."""
    X_train = np.vstack([train_pos_t_emb, train_neg_t_emb])
    y_train = np.array([1] * len(train_pos_t_emb) + [-1] * len(train_neg_t_emb))

    X_test = np.vstack([test_pos_t_emb, test_neg_t_emb])
    y_test = np.array([1] * len(test_pos_t_emb) + [-1] * len(test_neg_t_emb))

    clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    unseen_acc = clf.score(X_test, y_test)
    return float(unseen_acc)


def run_leave_one_object_out_text_probe_experiment(
    object_t_embs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    C: float = 1.0
) -> Dict[str, Any]:
    """Perform Leave-One-Object-Out (LOOO) cross-validation for text linear probing across all objects.
    
    Args:
        object_t_embs: Dictionary mapping object_name -> (pos_t_emb, neg_t_emb)
    
    Returns:
        Dictionary containing per-object unseen test accuracy and macro average unseen accuracy.
    """
    object_names = list(object_t_embs.keys())
    per_object_unseen_acc = {}

    for target_obj in object_names:
        # 1. Test set: target_obj
        test_pos, test_neg = object_t_embs[target_obj]

        # 2. Train set: All other objects
        train_pos_list = [object_t_embs[o][0] for o in object_names if o != target_obj]
        train_neg_list = [object_t_embs[o][1] for o in object_names if o != target_obj]

        train_pos = np.vstack(train_pos_list)
        train_neg = np.vstack(train_neg_list)

        acc = evaluate_text_linear_probe_cross_object(
            train_pos_t_emb=train_pos,
            train_neg_t_emb=train_neg,
            test_pos_t_emb=test_pos,
            test_neg_t_emb=test_neg,
            C=C
        )
        per_object_unseen_acc[target_obj] = acc

    acc_values = list(per_object_unseen_acc.values())
    summary = {
        "looo_unseen_acc_mean": float(np.mean(acc_values)),
        "looo_unseen_acc_std": float(np.std(acc_values)),
        "per_object_unseen_acc": per_object_unseen_acc
    }
    return summary



def evaluate_unseen_template_group_text_probe(
    pos_t_emb: np.ndarray,
    neg_t_emb: np.ndarray,
    pos_groups: List[str],
    neg_groups: List[str],
    C: float = 1.0
) -> Dict[str, Any]:
    """Perform Leave-One-Template-Group-Out (LOO-Group) cross-validation.
    
    Trains LogisticRegression on 3 template groups, and tests accuracy on the 4th unseen template group.
    Prevents template keyword shortcut memorization.
    """
    unique_groups = sorted(list(set(pos_groups + neg_groups)))
    per_group_acc = {}

    for target_group in unique_groups:
        # Train set: All templates EXCEPT target_group
        train_pos_idx = [i for i, g in enumerate(pos_groups) if g != target_group]
        train_neg_idx = [i for i, g in enumerate(neg_groups) if g != target_group]

        # Test set: Templates in target_group
        test_pos_idx = [i for i, g in enumerate(pos_groups) if g == target_group]
        test_neg_idx = [i for i, g in enumerate(neg_groups) if g == target_group]

        if not train_pos_idx or not train_neg_idx or not test_pos_idx or not test_neg_idx:
            continue

        X_train = np.vstack([pos_t_emb[train_pos_idx], neg_t_emb[train_neg_idx]])
        y_train = np.array([1] * len(train_pos_idx) + [-1] * len(train_neg_idx))

        X_test = np.vstack([pos_t_emb[test_pos_idx], neg_t_emb[test_neg_idx]])
        y_test = np.array([1] * len(test_pos_idx) + [-1] * len(test_neg_idx))

        clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        acc = clf.score(X_test, y_test)
        per_group_acc[target_group] = float(acc)


    accs = list(per_group_acc.values())
    return {
        "unseen_template_group_acc_mean": float(np.mean(accs)) if accs else 0.0,
        "unseen_template_group_acc_std": float(np.std(accs)) if accs else 0.0,
        "per_group_acc": per_group_acc,
    }


def train_eval_vision_linear_probe(
    pos_v_emb: np.ndarray,
    neg_v_emb: np.ndarray,
    C: float = 1.0,
    cv_folds: int = 5
) -> Tuple[float, float, Optional[LogisticRegression]]:
    """Train and evaluate LogisticRegression probe on Present (+1) vs Absent (-1) images."""
    if len(pos_v_emb) == 0 or len(neg_v_emb) == 0:
        return 0.5, 0.0, None

    X_v = np.vstack([pos_v_emb, neg_v_emb])
    y_v = np.array([1] * len(pos_v_emb) + [-1] * len(neg_v_emb))

    n_samples = len(y_v)
    effective_folds = min(cv_folds, n_samples // 2)

    clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
    if effective_folds >= 2:
        skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_v, y_v, cv=skf, scoring="accuracy")
        mean_acc, std_acc = float(scores.mean()), float(scores.std())
    else:
        mean_acc, std_acc = 0.5, 0.0

    clf.fit(X_v, y_v)
    return mean_acc, std_acc, clf



def evaluate_dual_classifier_product_scorer(
    v_clf: Optional[Any],
    t_clf: Optional[LogisticRegression],
    pos_v_emb: np.ndarray,
    neg_v_emb: np.ndarray,
    pos_t_emb: np.ndarray,
    neg_t_emb: np.ndarray,
    use_tanh: bool = False,
) -> Dict[str, float]:
    """
    Evaluate S(v, t) = cos(v, t) * (f_V(v) * f_T(t)) Product Scorer on 1:1 Present/Absent images.

    Note (Theoretical Limitation):
        The unconditional dual probe formulation f_V(v) * f_T(t) suffers from an unconditional scoring paradox
        where f_V(v) outputs a single scalar independent of the text candidate. For expressive scoring without
        this paradox, use conditional scoring heads (e.g. BilinearScorer v^T W t).
    """
    if v_clf is None or t_clf is None or len(pos_v_emb) == 0 or len(neg_v_emb) == 0:
        return {
            "dual_probe_pos_acc": 0.5,
            "dual_probe_neg_acc": 0.5,
            "dual_probe_overall_acc": 0.5,
        }

    # Normalize vectors for cosine similarity (using centralized l2_normalize)
    pos_v_norm = l2_normalize(pos_v_emb)
    neg_v_norm = l2_normalize(neg_v_emb)
    pos_t_norm = l2_normalize(pos_t_emb)
    neg_t_norm = l2_normalize(neg_t_emb)

    # Decision functions (margins)
    if hasattr(v_clf, "decision_function"):
        f_V_pos = v_clf.decision_function(pos_v_emb)  # [M]
        f_V_neg = v_clf.decision_function(neg_v_emb)  # [M]
    else:
        f_V_pos = np.ones(len(pos_v_emb))
        f_V_neg = -np.ones(len(neg_v_emb))

    f_T_pos = np.mean(t_clf.decision_function(pos_t_emb))  # scalar > 0
    f_T_neg = np.mean(t_clf.decision_function(neg_t_emb))  # scalar < 0

    # Mean cosine similarities between images and text template ensembles
    if pos_t_norm.ndim == 2:
        cos_pos_v_pos_t = np.mean(pos_v_norm @ pos_t_norm.T, axis=1)  # [M]
        cos_pos_v_neg_t = np.mean(pos_v_norm @ neg_t_norm.T, axis=1)  # [M]
        cos_neg_v_pos_t = np.mean(neg_v_norm @ pos_t_norm.T, axis=1)  # [M]
        cos_neg_v_neg_t = np.mean(neg_v_norm @ neg_t_norm.T, axis=1)  # [M]
    else:
        cos_pos_v_pos_t = np.sum(pos_v_norm * pos_t_norm, axis=-1)
        cos_pos_v_neg_t = np.sum(pos_v_norm * neg_t_norm, axis=-1)
        cos_neg_v_pos_t = np.sum(neg_v_norm * pos_t_norm, axis=-1)
        cos_neg_v_neg_t = np.sum(neg_v_norm * neg_t_norm, axis=-1)

    # Product scores S(v, t) = cos(v, t) * (f_V(v) * f_T(t)) (without sigmoid to preserve sign alignment)
    if use_tanh:
        S_pos_v_pos_t = cos_pos_v_pos_t * np.tanh(f_V_pos * f_T_pos)
        S_pos_v_neg_t = cos_pos_v_neg_t * np.tanh(f_V_pos * f_T_neg)
        S_neg_v_pos_t = cos_neg_v_pos_t * np.tanh(f_V_neg * f_T_pos)
        S_neg_v_neg_t = cos_neg_v_neg_t * np.tanh(f_V_neg * f_T_neg)
    else:
        S_pos_v_pos_t = cos_pos_v_pos_t * (f_V_pos * f_T_pos)
        S_pos_v_neg_t = cos_pos_v_neg_t * (f_V_pos * f_T_neg)
        S_neg_v_pos_t = cos_neg_v_pos_t * (f_V_neg * f_T_pos)
        S_neg_v_neg_t = cos_neg_v_neg_t * (f_V_neg * f_T_neg)

    # Present Image: S(v_pos, t_pos) > S(v_pos, t_neg)
    pos_correct = np.sum(S_pos_v_pos_t > S_pos_v_neg_t)
    # Absent Image: S(v_neg, t_neg) > S(v_neg, t_pos)
    neg_correct = np.sum(S_neg_v_neg_t > S_neg_v_pos_t)

    total = len(pos_v_emb) + len(neg_v_emb)
    overall_acc = float(pos_correct + neg_correct) / total if total > 0 else 0.0

    return {
        "dual_probe_pos_acc": float(pos_correct) / len(pos_v_emb) if len(pos_v_emb) > 0 else 0.0,
        "dual_probe_neg_acc": float(neg_correct) / len(neg_v_emb) if len(neg_v_emb) > 0 else 0.0,
        "dual_probe_overall_acc": overall_acc,
    }


def format_object_name(object_name: str) -> Dict[str, str]:
    """Helper to compute grammatical variants for a given object name."""
    obj = object_name.strip().lower()
    
    # Pluralization rules for COCO categories
    plural_map = {
        "person": "people",
        "bus": "buses",
        "glass": "glasses",
        "wine glass": "wine glasses",
        "sandwich": "sandwiches",
        "couch": "couches",
        "mouse": "mice",
        "tooth": "teeth",
        "bench": "benches",
    }
    
    if obj in plural_map:
        plural_obj = plural_map[obj]
    elif obj.endswith("s") or obj.endswith("ch") or obj.endswith("sh") or obj.endswith("x"):
        plural_obj = obj + "es"
    elif obj.endswith("y") and len(obj) > 1 and obj[-2] not in "aeiou":
        plural_obj = obj[:-1] + "ies"
    else:
        plural_obj = obj + "s"

    # Article rules (a vs an)
    vowels = ("a", "e", "i", "o", "u")
    if obj.startswith(vowels):
        a_obj = f"an {obj}"
        A_obj = f"An {obj}"
    else:
        a_obj = f"a {obj}"
        A_obj = f"A {obj}"

    return {
        "object": obj,
        "a_object": a_obj,
        "A_object": A_obj,
        "plural_object": plural_obj,
    }


def instantiate_templates(object_name: str, template_data: Dict[str, Any]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Fill template placeholders for a specific object, returning prompts and group tags."""
    fmt = format_object_name(object_name)

    neg_prompts, neg_groups = [], []
    for item in template_data["negative_templates"]:
        if isinstance(item, dict):
            tmpl_str = item["template"]
            grp = item.get("group", "group_A")
        else:
            tmpl_str = item
            grp = "group_A"
        neg_prompts.append(tmpl_str.format(**fmt))
        neg_groups.append(grp)

    pos_prompts, pos_groups = [], []
    for item in template_data["positive_templates"]:
        if isinstance(item, dict):
            tmpl_str = item["template"]
            grp = item.get("group", "group_A")
        else:
            tmpl_str = item
            grp = "group_A"
        pos_prompts.append(tmpl_str.format(**fmt))
        pos_groups.append(grp)

    return neg_prompts, pos_prompts, neg_groups, pos_groups


def get_balanced_beaf_object_df(df: pd.DataFrame, target_object: str) -> pd.DataFrame:
    """Filter BEAF dataframe for a single object and enforce 1:1 Present/Absent image balance."""
    sub_df = df[df["object_name"].str.lower() == target_object.lower()].copy()
    if sub_df.empty:
        return sub_df

    pos_df = sub_df[sub_df["object_in_image"] == True]
    neg_df = sub_df[sub_df["object_in_image"] == False]

    n_samples = min(len(pos_df), len(neg_df))
    if n_samples == 0:
        return pd.DataFrame()

    balanced_pos = pos_df.iloc[:n_samples]
    balanced_neg = neg_df.iloc[:n_samples]

    balanced_df = pd.concat([balanced_pos, balanced_neg]).reset_index(drop=True)
    return balanced_df


def run_single_object_analysis(
    df_balanced: pd.DataFrame,
    object_name: str,
    neg_prompts: List[str],
    pos_prompts: List[str],
    neg_groups: List[str],
    pos_groups: List[str],
    model: torch.nn.Module,
    preprocess: callable,
    tokenizer: callable,
    device: str = "cuda",
    batch_size: int = 64,
    vision_probe_type: str = "linear",
) -> Dict[str, Any]:
    """Perform 4-way cross cosine similarity, zero-shot accuracy, and dual probing analysis for a single object."""
    model.eval()

    # 1. Encode Positive and Negative Text Templates
    with torch.no_grad():
        pos_tokens = tokenizer(pos_prompts).to(device)
        neg_tokens = tokenizer(neg_prompts).to(device)

        pos_t_emb = model.encode_text(pos_tokens, normalize=True).cpu().numpy()  # [N_pos, D]
        neg_t_emb = model.encode_text(neg_tokens, normalize=True).cpu().numpy()  # [N_neg, D]

    # Mean prompt embeddings
    mean_pos_t_emb = np.mean(pos_t_emb, axis=0, keepdims=True)
    mean_pos_t_emb /= np.linalg.norm(mean_pos_t_emb, axis=-1, keepdims=True)

    mean_neg_t_emb = np.mean(neg_t_emb, axis=0, keepdims=True)
    mean_neg_t_emb /= np.linalg.norm(mean_neg_t_emb, axis=-1, keepdims=True)

    # 2. Extract Visual Features for Present and Absent Images
    pos_img_df = df_balanced[df_balanced["object_in_image"] == True]
    neg_img_df = df_balanced[df_balanced["object_in_image"] == False]

    def _extract_img_embeds(paths: List[str]) -> np.ndarray:
        embeds = []
        with torch.no_grad():
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i : i + batch_size]
                tensors = []
                for p in batch_paths:
                    if os.path.exists(p):
                        img = Image.open(p).convert("RGB")
                        tensors.append(preprocess(img))
                if tensors:
                    b_tens = torch.stack(tensors).to(device)
                    v_emb = model.encode_image(b_tens, normalize=True).cpu().numpy()
                    embeds.append(v_emb)
        if embeds:
            return np.vstack(embeds)
        return np.empty((0, pos_t_emb.shape[1]))

    pos_v_emb = _extract_img_embeds(pos_img_df["abs_image_path"].tolist())  # [M, D] Present
    neg_v_emb = _extract_img_embeds(neg_img_df["abs_image_path"].tolist())  # [M, D] Absent

    if len(pos_v_emb) == 0 or len(neg_v_emb) == 0:
        return {"error": f"Failed to load images for object {object_name}"}

    # 3. Compute Similarities
    sim_pos_v_pos_t = pos_v_emb @ pos_t_emb.T
    sim_pos_v_neg_t = pos_v_emb @ neg_t_emb.T
    sim_neg_v_pos_t = neg_v_emb @ pos_t_emb.T
    sim_neg_v_neg_t = neg_v_emb @ neg_t_emb.T

    mean_sim_pos_v_pos_t = np.mean(sim_pos_v_pos_t, axis=1)  # [M]
    mean_sim_pos_v_neg_t = np.mean(sim_pos_v_neg_t, axis=1)  # [M]
    mean_sim_neg_v_pos_t = np.mean(sim_neg_v_pos_t, axis=1)  # [M]
    mean_sim_neg_v_neg_t = np.mean(sim_neg_v_neg_t, axis=1)  # [M]

    sim_text_pos_neg = pos_t_emb @ neg_t_emb.T
    sim_text_pos_pos = pos_t_emb @ pos_t_emb.T
    sim_text_neg_neg = neg_t_emb @ neg_t_emb.T

    # 4. Zero-shot Classification Accuracy (Cosine Similarity)
    pos_v_correct = np.sum(mean_sim_pos_v_pos_t > mean_sim_pos_v_neg_t)
    neg_v_correct = np.sum(mean_sim_neg_v_neg_t > mean_sim_neg_v_pos_t)

    total_images = len(pos_v_emb) + len(neg_v_emb)
    overall_acc = float(pos_v_correct + neg_v_correct) / total_images if total_images > 0 else 0.0
    pos_acc = float(pos_v_correct) / len(pos_v_emb)
    neg_acc = float(neg_v_correct) / len(neg_v_emb)

    pos_v_margin = mean_sim_pos_v_pos_t - mean_sim_pos_v_neg_t
    neg_v_margin = mean_sim_neg_v_neg_t - mean_sim_neg_v_pos_t

    # 5. Text Linear Probe (5-Fold CV & Unseen Template Group CV)
    text_cv_acc, text_cv_std, t_clf = evaluate_text_linear_probe_single_object(pos_t_emb, neg_t_emb)
    unseen_tmpl_results = evaluate_unseen_template_group_text_probe(pos_t_emb, neg_t_emb, pos_groups, neg_groups)

    # 6. Vision High-Order Non-Linear Probe & Joint Dual Scorer
    vision_cv_acc, vision_cv_std, v_clf = train_eval_vision_high_order_probe(pos_v_emb, neg_v_emb, probe_type=vision_probe_type)
    dual_scorer_res = evaluate_dual_classifier_product_scorer(v_clf, t_clf, pos_v_emb, neg_v_emb, pos_t_emb, neg_t_emb)


    results = {
        "object_name": object_name,
        "n_present_images": int(len(pos_v_emb)),
        "n_absent_images": int(len(neg_v_emb)),
        "n_positive_templates": int(len(pos_prompts)),
        "n_negative_templates": int(len(neg_prompts)),

        # Similarities
        "mean_sim_pos_v_pos_t": float(np.mean(sim_pos_v_pos_t)),
        "mean_sim_pos_v_neg_t": float(np.mean(sim_pos_v_neg_t)),
        "mean_sim_neg_v_pos_t": float(np.mean(sim_neg_v_pos_t)),
        "mean_sim_neg_v_neg_t": float(np.mean(sim_neg_v_neg_t)),

        # Text-Text Alignment
        "mean_sim_text_pos_neg": float(np.mean(sim_text_pos_neg)),
        "mean_sim_text_pos_pos": float(np.mean(sim_text_pos_pos)),
        "mean_sim_text_neg_neg": float(np.mean(sim_text_neg_neg)),

        # Zero-Shot Cosine Accuracies
        "pos_image_accuracy": pos_acc,
        "neg_image_accuracy": neg_acc,
        "overall_accuracy": overall_acc,
        "mean_pos_v_margin": float(np.mean(pos_v_margin)),
        "mean_neg_v_margin": float(np.mean(neg_v_margin)),

        # Text Linear Probe Metrics
        "text_probe_cv_acc": text_cv_acc,
        "text_probe_cv_std": text_cv_std,
        "unseen_template_group_acc_mean": unseen_tmpl_results["unseen_template_group_acc_mean"],
        "unseen_template_group_acc_std": unseen_tmpl_results["unseen_template_group_acc_std"],

        # Vision Probe & Dual Classifier Product Metrics
        "vision_probe_cv_acc": vision_cv_acc,
        "vision_probe_cv_std": vision_cv_std,
        "dual_probe_overall_acc": dual_scorer_res["dual_probe_overall_acc"],

        # Raw Embeddings & Classifiers for Cross-Object Sweeps
        "_pos_t_emb": pos_t_emb,
        "_neg_t_emb": neg_t_emb,
        "_pos_v_emb": pos_v_emb,
        "_neg_v_emb": neg_v_emb,
    }

    return results


