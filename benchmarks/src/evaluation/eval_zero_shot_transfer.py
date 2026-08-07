"""
Zero-Shot Cross-Dataset Transfer Evaluation Script for Low-Rank Scorers.

Research Goal: Evaluate pre-trained Low-Rank / Non-Linear Bi-Encoder scoring heads
on external Out-of-Distribution (OOD) benchmarks (e.g., SugarCrepe, Winoground,
BEAF Counterfactual, or Medical/Video CSVs) in a strict ZERO-SHOT (Eval-Only) manner.

Automatically supports BOTH MCQ/Paired benchmarks (Accuracy metrics) and
Retrieval benchmarks (Recall@1, Recall@5 metrics via Pairwise Similarity Matrix S).

Usage:
    python -m benchmarks.src.evaluation.eval_zero_shot_transfer \
        --model ViT-B-32 --pretrained openai \
        --scorer-ckpt logs/evaluation/scoring_head_experiments/checkpoints/deep_mlp_scorer.pt \
        --target-mcq COCO_val_retrieval.csv \
        --output-dir logs/evaluation/zero_shot_transfer
"""

import os
import sys
import json
import argparse
import random
import ast
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import open_clip

from src.evaluation.scoring_heads import (
    BaseScorer,
    CosineScorer,
    LowRankBilinearScorer,
    NonLinearBiEncoderScorer,
    BilinearScorer,
    DeepMLPScorer,
    build_scorer,
)
from src.evaluation.eval_scoring_heads import (
    extract_mcq_embeddings,
    compute_mcq_accuracy_breakdown,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval Task Processing
# ──────────────────────────────────────────────────────────────────────────────

def extract_retrieval_embeddings(
    model: nn.Module,
    tokenizer: Any,
    preprocess: Any,
    csv_file: str,
    device: str = "cuda",
    batch_size: int = 64,
    image_root: str = ""
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Extract frozen CLIP image embeddings (N_img, D), text embeddings (N_txt, D),
    and ground-truth image indices for Retrieval evaluation.
    """
    cache_dir = "logs/evaluation/cached_embeddings"
    os.makedirs(cache_dir, exist_ok=True)
    csv_basename = os.path.basename(csv_file).replace(".csv", "")
    cache_path = os.path.join(cache_dir, f"{csv_basename}_retrieval_embeds.pt")

    if os.path.exists(cache_path):
        print(f"\n⚡ Loading pre-cached Retrieval CLIP embeddings from disk cache: {cache_path}")
        try:
            cached = torch.load(cache_path, map_location="cpu")
            print(f"✅ Loaded cached retrieval features: Images {cached['images_emb'].shape}, Texts {cached['texts_emb'].shape}")
            return cached["images_emb"], cached["texts_emb"], cached["texts_image_index"]
        except Exception as e:
            print(f"⚠️ Failed to load cache file {cache_path}: {e}. Re-extracting...")

    model.eval()
    df = pd.read_csv(csv_file, sep=",")

    img_col = "filepath" if "filepath" in df.columns else ("image_path" if "image_path" in df.columns else df.columns[0])
    cap_col = "captions" if "captions" in df.columns else "caption"

    img_embed_list = []
    text_embed_list = []
    texts_image_index = []

    dataset_dir = os.path.dirname(os.path.abspath(csv_file))

    print(f"\nExtracting Retrieval Embeddings for {len(df)} images from {csv_file}...")

    img_counter = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Caching Retrieval Features"):
        rel_img_path = str(row[img_col])
        full_img_path = rel_img_path
        if not os.path.exists(full_img_path):
            candidates = [
                os.path.join(image_root, rel_img_path) if image_root else "",
                os.path.join(dataset_dir, rel_img_path),
                os.path.join(dataset_dir, "images", rel_img_path),
                os.path.join(os.path.dirname(dataset_dir), "images", rel_img_path),
            ]
            for cand in candidates:
                if cand and os.path.exists(cand):
                    full_img_path = cand
                    break

        if not os.path.exists(full_img_path):
            continue

        try:
            img = Image.open(full_img_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
        except Exception:
            continue

        cap_val = row[cap_col]
        if isinstance(cap_val, str) and cap_val.strip().startswith("["):
            try:
                captions = [str(c) for c in ast.literal_eval(cap_val)]
            except Exception:
                captions = [str(cap_val)]
        else:
            captions = [str(cap_val)]

        with torch.no_grad():
            img_feat = F.normalize(model.encode_image(img_tensor), dim=-1).cpu()
            tokens = tokenizer(captions).to(device)
            text_feat = F.normalize(model.encode_text(tokens), dim=-1).cpu() # (K, D)

        img_embed_list.append(img_feat)
        text_embed_list.append(text_feat)
        texts_image_index.extend([img_counter] * len(captions))
        img_counter += 1

    images_emb = torch.cat(img_embed_list, dim=0) # (N_img, D)
    texts_emb = torch.cat(text_embed_list, dim=0)  # (N_txt, D)

    torch.save({
        "images_emb": images_emb,
        "texts_emb": texts_emb,
        "texts_image_index": texts_image_index
    }, cache_path)
    print(f"💾 Saved extracted retrieval embeddings to disk cache: {cache_path}")
    print(f"✅ Extracted Retrieval Features: Images {images_emb.shape}, Texts {texts_emb.shape}")
    return images_emb, texts_emb, texts_image_index


def evaluate_zero_shot_retrieval(
    scorer: BaseScorer,
    images_emb: torch.Tensor,
    texts_emb: torch.Tensor,
    texts_image_index: List[int],
    device: str = "cuda",
    batch_size: int = 256,
    recall_k_list: List[int] = [1, 5]
) -> Dict[str, Any]:
    """
    Compute pairwise similarity matrix S (N_txt, N_img) using Scorer and calculate Recall@k.
    """
    N_txt = texts_emb.shape[0]
    N_img = images_emb.shape[0]

    if isinstance(scorer, CosineScorer):
        scores = texts_emb @ images_emb.T # (N_txt, N_img)
    else:
        scorer = scorer.to(device)
        scorer.eval()
        scores = torch.zeros(N_txt, N_img, device="cpu")

        with torch.no_grad():
            for start_t in range(0, N_txt, batch_size):
                end_t = min(start_t + batch_size, N_txt)
                t_batch = texts_emb[start_t:end_t].to(device)

                for start_i in range(0, N_img, batch_size):
                    end_i = min(start_i + batch_size, N_img)
                    i_batch = images_emb[start_i:end_i].to(device)

                    B_t, B_i = t_batch.shape[0], i_batch.shape[0]
                    i_exp = i_batch.unsqueeze(0).expand(B_t, B_i, -1)
                    t_exp = t_batch.unsqueeze(1).expand(B_t, B_i, -1)

                    sub_scores = scorer(i_exp, t_exp) # (B_t, B_i)
                    scores[start_t:end_t, start_i:end_i] = sub_scores.cpu()

    # Ground truth positive pair mask
    positive_pairs = torch.zeros_like(scores, dtype=torch.bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True

    metrics = {}
    for k in recall_k_list:
        # Text-to-Image Recall@k
        _, topk_indices = torch.topk(scores, k=k, dim=1)
        t2i_correct = positive_pairs.gather(1, topk_indices).any(dim=1).float().mean().item() * 100.0
        metrics[f"t2i_recall@{k}"] = t2i_correct

        # Image-to-Text Recall@k
        _, topk_indices_i2t = torch.topk(scores.T, k=k, dim=1)
        i2t_correct = positive_pairs.T.gather(1, topk_indices_i2t).any(dim=1).float().mean().item() * 100.0
        metrics[f"i2t_recall@{k}"] = i2t_correct

    metrics["total_accuracy"] = metrics.get("t2i_recall@1", 0.0) # Map R@1 to total_accuracy for unified logging
    metrics["positive_accuracy"] = metrics.get("t2i_recall@5", 0.0)
    metrics["negative_accuracy"] = metrics.get("i2t_recall@1", 0.0)
    metrics["hybrid_accuracy"] = metrics.get("i2t_recall@5", 0.0)
    metrics["total_samples"] = N_txt

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# MCQ Task Processing
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_zero_shot_scorer(
    scorer: BaseScorer,
    img_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    targets: torch.Tensor,
    question_types: List[str],
    device: str = "cuda",
    batch_size: int = 64
) -> Dict[str, Any]:
    """Run pure Zero-Shot inference on MCQ/Paired target dataset."""
    scorer = scorer.to(device)
    scorer.eval()

    ds = TensorDataset(img_embeds, text_embeds, targets)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds_list = []
    with torch.no_grad():
        for imgs, texts, _ in loader:
            imgs, texts = imgs.to(device), texts.to(device)
            scores = scorer(imgs, texts)
            preds = torch.argmax(scores, dim=1).cpu().numpy()
            preds_list.append(preds)

    all_preds = np.concatenate(preds_list)
    targets_np = targets.numpy()

    metrics = compute_mcq_accuracy_breakdown(all_preds, targets_np, question_types)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Transfer Evaluation of Pre-trained Scoring Heads on OOD Benchmarks")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model architecture")
    parser.add_argument("--pretrained", type=str, default="openai", help="Pretrained weights checkpoint tag")
    parser.add_argument("--scorer-ckpt", type=str, default=None, help="Path to single pre-trained scorer checkpoint (.pt)")
    parser.add_argument("--scorer-dir", type=str, default=None, help="Path to directory containing multiple pre-trained scorer checkpoints (.pt)")
    parser.add_argument("--target-mcq", type=str, required=True, help="Path to target OOD MCQ/Retrieval Benchmark CSV file")
    parser.add_argument("--image-root", type=str, default="", help="Root directory containing images")
    parser.add_argument("--output-dir", type=str, default="logs/evaluation/zero_shot_transfer", help="Output directory")
    parser.add_argument("--rank", type=int, default=32, help="Rank k for Low-Rank models")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Zero-Shot Transfer Evaluation on device: {device}")

    # Collect checkpoint paths
    ckpt_paths = []
    if args.scorer_ckpt and os.path.exists(args.scorer_ckpt):
        ckpt_paths.append(args.scorer_ckpt)

    scorer_dir = args.scorer_dir
    if not scorer_dir and not args.scorer_ckpt:
        default_dir = "logs/evaluation/scoring_head_experiments/checkpoints"
        if os.path.exists(default_dir):
            scorer_dir = default_dir

    if scorer_dir and os.path.exists(scorer_dir):
        for fname in sorted(os.listdir(scorer_dir)):
            if fname.endswith(".pt") or fname.endswith(".npz"):
                full_p = os.path.join(scorer_dir, fname)
                if full_p not in ckpt_paths:
                    ckpt_paths.append(full_p)

    # Load CLIP
    print(f"\nLoading OpenCLIP {args.model} ({args.pretrained})...")
    model, _, preprocess_val = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    # Detect dataset task type: Retrieval vs MCQ
    df_check = pd.read_csv(args.target_mcq, nrows=5)
    is_retrieval = ("retrieval" in args.target_mcq.lower()) or ("captions" in df_check.columns and "caption_0" not in df_check.columns and "positive_caption" not in df_check.columns)

    transfer_results = {}

    def _load_scorer_from_ckpt(ckpt_path: str, feature_dim: int, rank_default: int):
        from src.evaluation.scoring_heads import DualClassifierProductScorer
        if ckpt_path.endswith(".npz"):
            data = np.load(ckpt_path)
            w_t = torch.from_numpy(data["w_t"]).float()
            b_t = float(data["b_t"])
            b_v = float(data.get("b_v", 0.0))

            U_v = torch.from_numpy(data["U_v"]).float() if "U_v" in data else None
            V_v = torch.from_numpy(data["V_v"]).float() if "V_v" in data else None
            w_lin_v = torch.from_numpy(data["w_lin_v"]).float() if "w_lin_v" in data else None
            w_v = torch.from_numpy(data["w_v"]).float() if "w_v" in data else None

            v_rank = U_v.shape[1] if U_v is not None else 4
            scorer = DualClassifierProductScorer(feature_dim=w_t.shape[0], vision_rank=v_rank)
            scorer.load_weights(w_t=w_t, b_t=b_t, w_v=w_v, b_v=b_v, U_v=U_v, V_v=V_v, w_lin_v=w_lin_v)
            model_name = os.path.basename(ckpt_path).replace(".npz", "")
            return scorer, model_name

        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model_name = ckpt.get("model_name", os.path.basename(ckpt_path).replace(".pt", ""))
            rank = ckpt.get("rank", rank_default)
            scorer = build_scorer(model_name, feature_dim, rank=rank)
            scorer.load_state_dict(ckpt["state_dict"])
            return scorer, model_name

    if is_retrieval:
        print(f"\n🔍 Detected Retrieval Task Dataset: {args.target_mcq}")
        images_emb, texts_emb, texts_image_index = extract_retrieval_embeddings(
            model, tokenizer, preprocess_val, args.target_mcq, device=device, batch_size=args.batch_size, image_root=args.image_root
        )
        feature_dim = images_emb.shape[1]

        # Baseline Cosine Retrieval
        cosine_scorer = CosineScorer(feature_dim)
        cosine_metrics = evaluate_zero_shot_retrieval(cosine_scorer, images_emb, texts_emb, texts_image_index, device=device)
        print(f"\nBaseline Cosine Retrieval: T2I R@1 = {cosine_metrics['t2i_recall@1']:.2f}% | T2I R@5 = {cosine_metrics['t2i_recall@5']:.2f}% | I2T R@1 = {cosine_metrics['i2t_recall@1']:.2f}%")
        transfer_results["Baseline Cosine"] = cosine_metrics

        for ckpt_path in ckpt_paths:
            print(f"\nLoading Pre-trained Scorer Checkpoint: {ckpt_path}")
            scorer, model_name = _load_scorer_from_ckpt(ckpt_path, feature_dim, args.rank)

            ckpt_metrics = evaluate_zero_shot_retrieval(scorer, images_emb, texts_emb, texts_image_index, device=device)
            print(f"Pre-trained Scorer ({model_name}) Retrieval: T2I R@1 = {ckpt_metrics['t2i_recall@1']:.2f}% | T2I R@5 = {ckpt_metrics['t2i_recall@5']:.2f}% | I2T R@1 = {ckpt_metrics['i2t_recall@1']:.2f}%")
            transfer_results[f"Pretrained_{model_name}"] = ckpt_metrics

    else:
        print(f"\n🔍 Detected MCQ/Paired Task Dataset: {args.target_mcq}")
        img_embeds, text_embeds, targets, question_types, _ = extract_mcq_embeddings(
            model, tokenizer, preprocess_val, args.target_mcq, device=device, batch_size=args.batch_size, image_root=args.image_root
        )
        feature_dim = img_embeds.shape[1]

        cosine_scorer = CosineScorer(feature_dim)
        cosine_metrics = evaluate_zero_shot_scorer(cosine_scorer, img_embeds, text_embeds, targets, question_types, device=device)
        print(f"\nBaseline Cosine MCQ: Total Acc = {cosine_metrics['total_accuracy']:.2f}% | Pos Acc = {cosine_metrics['positive_accuracy']:.2f}% | Neg Acc = {cosine_metrics['negative_accuracy']:.2f}%")
        transfer_results["Baseline Cosine"] = cosine_metrics

        for ckpt_path in ckpt_paths:
            print(f"\nLoading Pre-trained Scorer Checkpoint: {ckpt_path}")
            scorer, model_name = _load_scorer_from_ckpt(ckpt_path, feature_dim, args.rank)

            ckpt_metrics = evaluate_zero_shot_scorer(scorer, img_embeds, text_embeds, targets, question_types, device=device)
            print(f"Pre-trained Scorer ({model_name}) MCQ: Total Acc = {ckpt_metrics['total_accuracy']:.2f}% | Pos Acc = {ckpt_metrics['positive_accuracy']:.2f}% | Neg Acc = {ckpt_metrics['negative_accuracy']:.2f}%")

            transfer_results[f"Pretrained_{model_name}"] = ckpt_metrics


    # Save JSON and Summary CSV
    out_json = os.path.join(args.output_dir, "zero_shot_transfer_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(transfer_results, f, indent=2)

    rows = []
    for name, mdata in transfer_results.items():
        rows.append({
            "Model": name,
            "Target_Dataset": os.path.basename(args.target_mcq),
            "Task_Type": "Retrieval" if is_retrieval else "MCQ",
            "Total_Accuracy_Or_R1": mdata["total_accuracy"],
            "Positive_Acc_Or_R5": mdata["positive_accuracy"],
            "Negative_Acc_Or_I2T_R1": mdata["negative_accuracy"],
            "Hybrid_Acc_Or_I2T_R5": mdata["hybrid_accuracy"],
            "Total_Samples": mdata["total_samples"]
        })

    summary_csv_path = os.path.join(args.output_dir, "zero_shot_transfer_summary.csv")
    new_df = pd.DataFrame(rows)
    if os.path.exists(summary_csv_path):
        old_df = pd.read_csv(summary_csv_path)
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["Model", "Target_Dataset"], keep="last")
        combined_df.to_csv(summary_csv_path, index=False)
    else:
        new_df.to_csv(summary_csv_path, index=False)

    print(f"\n✅ Zero-shot transfer evaluation complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
