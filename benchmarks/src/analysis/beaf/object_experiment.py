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
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


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

        if not test_pos_idx or not test_neg_idx:
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
) -> Tuple[float, float, LogisticRegression]:
    """Train and evaluate LogisticRegression probe on Present (+1) vs Absent (-1) images."""
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
    v_clf: LogisticRegression,
    t_clf: LogisticRegression,
    pos_v_emb: np.ndarray,
    neg_v_emb: np.ndarray,
    pos_t_emb: np.ndarray,
    neg_t_emb: np.ndarray
) -> Dict[str, float]:
    """Evaluate S(v, t) = f_V(v) * f_T(t) Product Scorer on 1:1 Present/Absent images."""
    # Decision functions
    f_V_pos = v_clf.decision_function(pos_v_emb)  # [M]
    f_V_neg = v_clf.decision_function(neg_v_emb)  # [M]

    f_T_pos = np.mean(t_clf.decision_function(pos_t_emb))  # scalar
    f_T_neg = np.mean(t_clf.decision_function(neg_t_emb))  # scalar

    # Product scores S(v, t) = f_V(v) * f_T(t)
    S_pos_v_pos_t = f_V_pos * f_T_pos
    S_pos_v_neg_t = f_V_pos * f_T_neg

    S_neg_v_pos_t = f_V_neg * f_T_pos
    S_neg_v_neg_t = f_V_neg * f_T_neg

    # Present Image: S(v_pos, t_pos) > S(v_pos, t_neg)
    pos_correct = np.sum(S_pos_v_pos_t > S_pos_v_neg_t)
    # Absent Image: S(v_neg, t_neg) > S(v_neg, t_pos)
    neg_correct = np.sum(S_neg_v_neg_t > S_neg_v_pos_t)

    total = len(pos_v_emb) + len(neg_v_emb)
    overall_acc = float(pos_correct + neg_correct) / total if total > 0 else 0.0

    return {
        "dual_probe_pos_acc": float(pos_correct) / len(pos_v_emb),
        "dual_probe_neg_acc": float(neg_correct) / len(neg_v_emb),
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
    batch_size: int = 64
) -> Dict[str, Any]:
    """Perform 4-way cross cosine similarity, zero-shot accuracy, and dual linear probing analysis for a single object."""
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

    # 6. Vision Linear Probe & Joint Dual Scorer
    vision_cv_acc, vision_cv_std, v_clf = train_eval_vision_linear_probe(pos_v_emb, neg_v_emb)
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


