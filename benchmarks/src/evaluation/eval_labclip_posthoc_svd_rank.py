"""
LABCLIP rank comparison, condition A: "train full-rank first, then decompose."

Takes the already-trained full-rank LABCLIP-recipe W
(train_labclip_official_recipe.py's output -- full-rank, identity-init,
broad InfoNCE on BEAF) and truncates it to the best rank-r approximation via
SVD (W_r = U_r Sigma_r V_r^T, top-r singular values), for
r in {8,16,32,64,128,256}. No retraining -- this is a purely post-hoc
decomposition of one already-converged W.

This is the "optimize at full rank, then reduce" half of the review7-inspired
question "does the optimization target need to match the final deployed rank,
or can rank be imposed after the fact?" -- contrasted against condition B
(train_lowrank_plain_infonce.py, rank imposed from the start) and condition C
(train_residual_lowrank_infonce.py, rank imposed from the start AND the
diagonal/cosine component is never touched, only a residual is learned).

Usage:
    python -m benchmarks.src.evaluation.eval_labclip_posthoc_svd_rank \
        --full_rank_ckpt logs/evaluation/01_paper/2026-09-01_labclip_official_recipe/labclip_official_recipe_bilinear.pt \
        --output_dir logs/evaluation/01_paper/2026-09-02_labclip_posthoc_svd_rank
"""
import os
import json
import argparse

import torch

from benchmarks.src.evaluation.eval_single_w_generalization import load_quads
from benchmarks.src.evaluation.score_delta_s_for_w import joint_correct_for_w
from benchmarks.src.analysis.model_loader import load_clip_for_eval
from benchmarks.src.analysis.config import set_seed


def truncate_svd(W: torch.Tensor, rank: int) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_rank_ckpt",
                     default="logs/evaluation/01_paper/2026-09-01_labclip_official_recipe/labclip_official_recipe_bilinear.pt")
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
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
    ap.add_argument("--output_dir", default="logs/evaluation/01_paper/2026-09-02_labclip_posthoc_svd_rank")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.full_rank_ckpt, map_location="cpu")
    W_full = ckpt["state_dict"]["W"].float().to(device)
    print(f"[loaded] full-rank W from {args.full_rank_ckpt}, ||W||_F={torch.norm(W_full).item():.2f}")

    def load_gallery(path):
        cached = torch.load(path, map_location="cpu")
        return (cached["images_emb"].float().to(device), cached["texts_emb"].float().to(device),
                torch.tensor(cached["texts_image_index"], dtype=torch.long))

    def recall_at_k(scores, idx, k):
        topk = torch.topk(scores, k=k, dim=1).indices
        return (topk == idx.unsqueeze(1)).any(dim=1).float().mean().item() * 100

    std_img, std_txt, std_idx = load_gallery(args.cache)
    std_cos = (std_txt @ std_img.T).cpu()
    cosine_std_r1 = (std_cos.argmax(dim=1) == std_idx).float().mean().item() * 100
    cosine_std_r5 = recall_at_k(std_cos, std_idx, 5)

    have_neg = os.path.exists(args.neg_cache)
    if have_neg:
        neg_img, neg_txt, neg_idx = load_gallery(args.neg_cache)
        neg_cos = (neg_txt @ neg_img.T).cpu()
        cosine_neg_r1 = (neg_cos.argmax(dim=1) == neg_idx).float().mean().item() * 100

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    v_pos, v_neg, t_pos, t_neg, groups = load_quads(args, model, preprocess, tokenizer, device,
                                                      dict(model=args.model, pretrained=str(args.pretrained),
                                                           cache_dir=args.cache_dir, enabled=args.use_cache))
    T = lambda a: torch.from_numpy(a).float().to(device)
    quads = (T(v_pos), T(v_neg), T(t_pos), T(t_neg))

    print(f"[baseline] cosine std R@1={cosine_std_r1:.2f}% R@5={cosine_std_r5:.2f}%"
          + (f"  neg R@1={cosine_neg_r1:.2f}%" if have_neg else ""))

    # rank = full: score the untouched full-rank W as the r=512 reference point
    ranks_to_run = list(args.ranks) + [512]
    results = []
    for r in ranks_to_run:
        W_r = W_full if r == 512 else truncate_svd(W_full, r)
        dW_norm = torch.norm(W_r - torch.eye(W_r.shape[0], device=device)).item()
        std_scores = (std_txt @ W_r.T @ std_img.T).cpu()
        std_r1 = (std_scores.argmax(dim=1) == std_idx).float().mean().item() * 100
        std_r5 = recall_at_k(std_scores, std_idx, 5)
        row = {"rank": r, "dW_norm": dW_norm, "std_r1": std_r1, "std_r5": std_r5}
        if have_neg:
            neg_scores = (neg_txt @ W_r.T @ neg_img.T).cpu()
            row["neg_r1"] = (neg_scores.argmax(dim=1) == neg_idx).float().mean().item() * 100
        row["delta_s"] = joint_correct_for_w(W_r, quads)
        neg_str = f"  neg_R@1={row.get('neg_r1', float('nan')):6.2f}%" if have_neg else ""
        print(f"  rank={r:<4d} ||W-I||_F={dW_norm:7.2f}  std_R@1={std_r1:6.2f}%  std_R@5={std_r5:6.2f}%"
              f"{neg_str}  DeltaS>0={row['delta_s']:5.2f}%")
        results.append(row)

    with open(os.path.join(args.output_dir, "posthoc_svd_report.json"), "w") as f:
        json.dump({"cosine_std_r1": cosine_std_r1, "cosine_std_r5": cosine_std_r5,
                    "cosine_neg_r1": cosine_neg_r1 if have_neg else None, "results": results}, f, indent=2)
    print(f"[saved] {args.output_dir}/posthoc_svd_report.json")


if __name__ == "__main__":
    main()
