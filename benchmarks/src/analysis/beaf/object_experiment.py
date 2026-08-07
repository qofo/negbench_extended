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


def instantiate_templates(object_name: str, template_data: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """Fill template placeholders ({object}, {a_object}, {A_object}, {plural_object}) for a specific object."""
    fmt = format_object_name(object_name)

    neg_prompts = []
    for tmpl in template_data["negative_templates"]:
        prompt = tmpl.format(**fmt)
        neg_prompts.append(prompt)

    pos_prompts = []
    for tmpl in template_data["positive_templates"]:
        prompt = tmpl.format(**fmt)
        pos_prompts.append(prompt)

    return neg_prompts, pos_prompts


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
    model: torch.nn.Module,
    preprocess: callable,
    tokenizer: callable,
    device: str = "cuda",
    batch_size: int = 64
) -> Dict[str, Any]:
    """Perform 4-way cross cosine similarity and zero-shot accuracy analysis for a single object."""
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
    # Image-Text Full 4-Way Matrices
    # pos_v vs pos_t: [M, N_pos]
    sim_pos_v_pos_t = pos_v_emb @ pos_t_emb.T
    sim_pos_v_neg_t = pos_v_emb @ neg_t_emb.T
    sim_neg_v_pos_t = neg_v_emb @ pos_t_emb.T
    sim_neg_v_neg_t = neg_v_emb @ neg_t_emb.T

    # Mean Prompt Similarities per Image
    mean_sim_pos_v_pos_t = np.mean(sim_pos_v_pos_t, axis=1)  # [M]
    mean_sim_pos_v_neg_t = np.mean(sim_pos_v_neg_t, axis=1)  # [M]
    mean_sim_neg_v_pos_t = np.mean(sim_neg_v_pos_t, axis=1)  # [M]
    mean_sim_neg_v_neg_t = np.mean(sim_neg_v_neg_t, axis=1)  # [M]

    # Text-Text Cross Similarity Matrix
    sim_text_pos_neg = pos_t_emb @ neg_t_emb.T  # [N_pos, N_neg]
    sim_text_pos_pos = pos_t_emb @ pos_t_emb.T
    sim_text_neg_neg = neg_t_emb @ neg_t_emb.T

    # 4. Compute Zero-shot Classification Accuracy (MCQ & Binary)
    # Present Image correctly predicts Pos Text > Neg Text
    pos_v_correct = np.sum(mean_sim_pos_v_pos_t > mean_sim_pos_v_neg_t)
    # Absent Image correctly predicts Neg Text > Pos Text
    neg_v_correct = np.sum(mean_sim_neg_v_neg_t > mean_sim_neg_v_pos_t)

    total_images = len(pos_v_emb) + len(neg_v_emb)
    overall_acc = float(pos_v_correct + neg_v_correct) / total_images if total_images > 0 else 0.0
    pos_acc = float(pos_v_correct) / len(pos_v_emb)
    neg_acc = float(neg_v_correct) / len(neg_v_emb)

    # Margin calculation: S(v_pos, t_pos) - S(v_pos, t_neg)
    pos_v_margin = mean_sim_pos_v_pos_t - mean_sim_pos_v_neg_t
    neg_v_margin = mean_sim_neg_v_neg_t - mean_sim_neg_v_pos_t

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

        # Accuracy & Margins
        "pos_image_accuracy": pos_acc,
        "neg_image_accuracy": neg_acc,
        "overall_accuracy": overall_acc,
        "mean_pos_v_margin": float(np.mean(pos_v_margin)),
        "mean_neg_v_margin": float(np.mean(neg_v_margin)),
    }

    return results
