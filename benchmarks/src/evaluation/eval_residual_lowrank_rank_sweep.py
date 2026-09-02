"""
Evaluate every residual_lowrank_r*.pt checkpoint from
train_residual_lowrank_infonce.py on both axes of review7 block2's requested
Pareto frontier: Delta(S)>0 / 2x2 diagnostic accuracy vs. standalone COCO T2I
retrieval R@1/R@5 (standard queries, cached embeddings, no retraining) --
same evaluation convention as eval_pen_coef_fine_sweep.py (all-42-concept
in-sample W, not GroupKFold OOF; see score_delta_s_for_w.py's docstring for
why that is the fair comparison here).

Usage:
    python -m benchmarks.src.evaluation.eval_residual_lowrank_rank_sweep \
        --ckpt_dir logs/evaluation/01_paper/2026-09-02_residual_lowrank_rank_sweep
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
    ap.add_argument("--ckpt_dir", default="logs/evaluation/01_paper/2026-09-02_residual_lowrank_rank_sweep")
    ap.add_argument("--cache", default="logs/evaluation/cached_embeddings/COCO_val_retrieval_retrieval_embeds.pt")
    ap.add_argument("--csv_path", default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--image_root", default="benchmarks/data/images")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--min_pairs", type=int, default=20)
    ap.add_argument("--restrict_objects", default=None)
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--cache_dir", default="logs/evaluation/cached_embeddings/feature_cache")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="logs/evaluation/01_paper/2026-09-02_residual_lowrank_rank_sweep_eval")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_paths = sorted(
        glob.glob(os.path.join(args.ckpt_dir, "residual_lowrank_r*.pt")),
        key=lambda p: int(re.search(r"_r(\d+)\.pt$", p).group(1)),
    )
    print(f"[found] {len(ckpt_paths)} checkpoints")

    cached = torch.load(args.cache, map_location="cpu")
    images_emb = cached["images_emb"].float().to(device)
    texts_emb = cached["texts_emb"].float().to(device)
    texts_image_index = torch.tensor(cached["texts_image_index"], dtype=torch.long)
    cos_scores = (texts_emb @ images_emb.T).cpu()
    cosine_r1 = (cos_scores.argmax(dim=1) == texts_image_index).float().mean().item() * 100

    def recall_at_k(scores, k):
        topk = torch.topk(scores, k=k, dim=1).indices
        hit = (topk == texts_image_index.unsqueeze(1)).any(dim=1)
        return hit.float().mean().item() * 100

    cosine_r5 = recall_at_k(cos_scores, 5)
    print(f"[baseline] cosine standalone R@1={cosine_r1:.2f}% R@5={cosine_r5:.2f}%")

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    v_pos, v_neg, t_pos, t_neg, groups = load_quads(args, model, preprocess, tokenizer, device,
                                                      dict(model=args.model, pretrained=str(args.pretrained),
                                                           cache_dir=args.cache_dir, enabled=args.use_cache))
    T = lambda a: torch.from_numpy(a).float().to(device)
    quads = (T(v_pos), T(v_neg), T(t_pos), T(t_neg))

    results = []
    for ckpt_path in ckpt_paths:
        r = int(re.search(r"_r(\d+)\.pt$", ckpt_path).group(1))
        ckpt = torch.load(ckpt_path, map_location="cpu")
        W = ckpt["state_dict"]["W"].float().to(device)
        n_params = ckpt.get("n_params", None)
        if "dW_norm" in ckpt:
            dW_norm = ckpt["dW_norm"]
        else:
            dW_norm = torch.norm(W - torch.eye(W.shape[0], device=device)).item()

        w_scores = (texts_emb @ W.T @ images_emb.T).cpu()
        r1 = (w_scores.argmax(dim=1) == texts_image_index).float().mean().item() * 100
        r5 = recall_at_k(w_scores, 5)
        delta_s = joint_correct_for_w(W, quads)

        beats_cosine = "✅" if r1 > cosine_r1 else ""
        print(f"  rank={r:<4d} n_params={str(n_params):<8s} ||dW||_F={dW_norm:6.2f}  "
              f"R@1={r1:6.2f}%  R@5={r5:6.2f}%  DeltaS>0={delta_s:5.2f}%  {beats_cosine}")
        results.append({"rank": r, "n_params": n_params, "dW_norm": dW_norm,
                         "r1": r1, "r5": r5, "delta_s": delta_s})

    with open(os.path.join(args.output_dir, "rank_sweep_report.json"), "w") as f:
        json.dump({"cosine_r1": cosine_r1, "cosine_r5": cosine_r5, "results": results}, f, indent=2)
    best_r1 = max(results, key=lambda r: r["r1"])
    best_delta = max(results, key=lambda r: r["delta_s"])
    print(f"\n[best by R@1] rank={best_r1['rank']}  R@1={best_r1['r1']:.2f}%  (cosine={cosine_r1:.2f}%)")
    print(f"[best by DeltaS>0] rank={best_delta['rank']}  DeltaS>0={best_delta['delta_s']:.2f}%")
    print(f"[saved] {args.output_dir}/rank_sweep_report.json")


if __name__ == "__main__":
    main()
