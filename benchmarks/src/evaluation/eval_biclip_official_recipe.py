"""
Evaluate every biclip_wd*.pt checkpoint from train_biclip_official_recipe.py
(BiCLIP's own regularization, standard AdamW weight_decay on W pulling toward
the ZERO matrix) against Delta(S)>0 / 2x2 and standalone COCO T2I retrieval
R@1/R@5 -- same convention as eval_pen_coef_fine_sweep.py, which does the
identical evaluation for OUR residual+Frobenius-toward-identity checkpoints
(RESULTS.md 8-V.12 table 3), so the two regularization mechanisms land in one
directly comparable table.

Usage:
    python -m benchmarks.src.evaluation.eval_biclip_official_recipe \
        --ckpt_dir logs/evaluation/01_paper/2026-09-02_biclip_official_recipe
"""
import os
import re
import json
import glob
import argparse

import torch

from benchmarks.src.evaluation.eval_single_w_generalization import load_quads
from benchmarks.src.evaluation.score_delta_s_for_w import joint_correct_for_w
from benchmarks.src.analysis.model_loader import load_clip_for_eval
from benchmarks.src.analysis.config import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="logs/evaluation/01_paper/2026-09-02_biclip_official_recipe")
    ap.add_argument("--cache", default="logs/evaluation/cached_embeddings/COCO_val_retrieval_retrieval_embeds.pt")
    ap.add_argument("--neg_cache", default="logs/evaluation/cached_embeddings/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true_retrieval_embeds.pt")
    ap.add_argument("--csv_path", default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--image_root", default="benchmarks/data/images")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--min_pairs", type=int, default=20)
    ap.add_argument("--restrict_objects", default=None)
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--cache_dir", default="logs/evaluation/cached_embeddings/feature_cache")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="logs/evaluation/01_paper/2026-09-02_biclip_official_recipe_eval")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_paths = sorted(
        glob.glob(os.path.join(args.ckpt_dir, "biclip_wd*.pt")),
        key=lambda p: float(re.search(r"biclip_wd([\d.]+)\.pt$", p).group(1)),
    )
    print(f"[found] {len(ckpt_paths)} checkpoints")

    def load_gallery(path):
        cached = torch.load(path, map_location="cpu")
        images_emb = cached["images_emb"].float().to(device)
        texts_emb = cached["texts_emb"].float().to(device)
        texts_image_index = torch.tensor(cached["texts_image_index"], dtype=torch.long)
        return images_emb, texts_emb, texts_image_index

    def recall_at_k(scores, texts_image_index, k):
        topk = torch.topk(scores, k=k, dim=1).indices
        hit = (topk == texts_image_index.unsqueeze(1)).any(dim=1)
        return hit.float().mean().item() * 100

    std_img, std_txt, std_idx = load_gallery(args.cache)
    std_cos = (std_txt @ std_img.T).cpu()
    cosine_std_r1 = (std_cos.argmax(dim=1) == std_idx).float().mean().item() * 100
    cosine_std_r5 = recall_at_k(std_cos, std_idx, 5)
    print(f"[baseline] cosine standard R@1={cosine_std_r1:.2f}% R@5={cosine_std_r5:.2f}%")

    have_neg = os.path.exists(args.neg_cache)
    if have_neg:
        neg_img, neg_txt, neg_idx = load_gallery(args.neg_cache)
        neg_cos = (neg_txt @ neg_img.T).cpu()
        cosine_neg_r1 = (neg_cos.argmax(dim=1) == neg_idx).float().mean().item() * 100
        cosine_neg_r5 = recall_at_k(neg_cos, neg_idx, 5)
        print(f"[baseline] cosine negated R@1={cosine_neg_r1:.2f}% R@5={cosine_neg_r5:.2f}%")
    else:
        print(f"[skip] negated-query cache not found at {args.neg_cache}; standard-only")

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    v_pos, v_neg, t_pos, t_neg, groups = load_quads(args, model, preprocess, tokenizer, device,
                                                      dict(model=args.model, pretrained=str(args.pretrained),
                                                           cache_dir=args.cache_dir, enabled=args.use_cache))
    T = lambda a: torch.from_numpy(a).float().to(device)
    quads = (T(v_pos), T(v_neg), T(t_pos), T(t_neg))

    results = []
    for ckpt_path in ckpt_paths:
        wd = float(re.search(r"biclip_wd([\d.]+)\.pt$", ckpt_path).group(1))
        ckpt = torch.load(ckpt_path, map_location="cpu")
        W = ckpt["state_dict"]["W"].float().to(device)
        dW_norm = ckpt.get("dW_norm", torch.norm(W - torch.eye(W.shape[0], device=device)).item())

        std_scores = (std_txt @ W.T @ std_img.T).cpu()
        std_r1 = (std_scores.argmax(dim=1) == std_idx).float().mean().item() * 100
        std_r5 = recall_at_k(std_scores, std_idx, 5)

        row = {"weight_decay": wd, "dW_norm": dW_norm, "std_r1": std_r1, "std_r5": std_r5}

        if have_neg:
            neg_scores = (neg_txt @ W.T @ neg_img.T).cpu()
            neg_r1 = (neg_scores.argmax(dim=1) == neg_idx).float().mean().item() * 100
            neg_r5 = recall_at_k(neg_scores, neg_idx, 5)
            row["neg_r1"] = neg_r1
            row["neg_r5"] = neg_r5

        delta_s = joint_correct_for_w(W, quads)
        row["delta_s"] = delta_s

        beats_cosine = "✅" if std_r1 > cosine_std_r1 else ""
        neg_str = f"  neg_R@1={row.get('neg_r1', float('nan')):6.2f}%" if have_neg else ""
        print(f"  wd={wd:<7g} ||W-I||_F={dW_norm:6.2f}  std_R@1={std_r1:6.2f}%  std_R@5={std_r5:6.2f}%"
              f"{neg_str}  DeltaS>0={delta_s:5.2f}%  {beats_cosine}")
        results.append(row)

    with open(os.path.join(args.output_dir, "biclip_recipe_report.json"), "w") as f:
        json.dump({"cosine_std_r1": cosine_std_r1, "cosine_std_r5": cosine_std_r5,
                    "cosine_neg_r1": cosine_neg_r1 if have_neg else None,
                    "cosine_neg_r5": cosine_neg_r5 if have_neg else None,
                    "results": results}, f, indent=2)
    best = max(results, key=lambda r: r["std_r1"])
    print(f"\n[best by standard R@1] wd={best['weight_decay']}  R@1={best['std_r1']:.2f}%  (cosine={cosine_std_r1:.2f}%)")
    print(f"[saved] {args.output_dir}/biclip_recipe_report.json")


if __name__ == "__main__":
    main()
