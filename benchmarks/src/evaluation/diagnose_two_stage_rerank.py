"""
Priority 5 -- two-stage retrieve-then-rerank (review7 block1 방안C), applied
post-hoc to already-trained W's with NO retraining: cosine picks the top-K
candidate images per query, then W reranks only within that shortlist. Practical
fallback regardless of which root-cause hypothesis (A/B/C) turns out to explain
the standalone collapse -- coarse recall stays governed by cosine (which never
collapsed), and W only gets a vote among candidates cosine already trusts.

Usage:
    python -m benchmarks.src.evaluation.diagnose_two_stage_rerank \
        --ckpts logs/.../margin4_rank32.pt logs/.../delta_warmstart_rank32.pt logs/.../labclip_official_recipe_bilinear.pt \
        --cache logs/evaluation/cached_embeddings/COCO_val_retrieval_retrieval_embeds.pt
"""
import os
import json
import argparse

import torch

from benchmarks.src.evaluation.diagnose_labclip_gallery_scale_and_residual import load_scores


def two_stage_rerank_r1(cos_scores: torch.Tensor, w_scores: torch.Tensor,
                         texts_image_index: torch.Tensor, k: int) -> float:
    n_txt = cos_scores.shape[0]
    _, topk_idx = torch.topk(cos_scores, k=k, dim=1)          # (n_txt, k) cosine shortlist
    w_shortlist = w_scores.gather(1, topk_idx)                  # (n_txt, k) W scores within shortlist
    rerank_choice = topk_idx.gather(1, w_shortlist.argmax(dim=1, keepdim=True)).squeeze(1)
    hit = (rerank_choice == texts_image_index).float().mean().item() * 100
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="paths to .pt checkpoints (state_dict.W)")
    ap.add_argument("--names", nargs="+", default=None, help="display names, same order as --ckpts")
    ap.add_argument("--cache", default="logs/evaluation/cached_embeddings/COCO_val_retrieval_retrieval_embeds.pt")
    ap.add_argument("--output_dir", default="logs/evaluation/01_paper/2026-09-02_two_stage_rerank")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = args.names or [os.path.basename(p).replace(".pt", "") for p in args.ckpts]

    cached = torch.load(args.cache, map_location="cpu")
    texts_image_index = torch.tensor(cached["texts_image_index"], dtype=torch.long)
    cos_scores = (cached["texts_emb"].float().to(device) @ cached["images_emb"].float().to(device).T).cpu()
    cosine_r1 = (cos_scores.argmax(dim=1) == texts_image_index).float().mean().item() * 100
    print(f"[baseline] pure cosine T2I R@1 = {cosine_r1:.2f}%")

    k_list = [10, 50, 100, 250]
    report = {"cosine_baseline_r1": cosine_r1, "per_w": {}}

    for name, ckpt_path in zip(names, args.ckpts):
        _, w_scores, _, _, _ = load_scores(ckpt_path, args.cache, device)
        row = {}
        for k in k_list:
            r1 = two_stage_rerank_r1(cos_scores, w_scores, texts_image_index, k)
            row[f"rerank_top{k}_r1"] = r1
        standalone_r1 = (w_scores.argmax(dim=1) == texts_image_index).float().mean().item() * 100
        row["standalone_w_r1"] = standalone_r1
        report["per_w"][name] = row
        print(f"\n=== {name} (standalone R@1={standalone_r1:.2f}%) ===")
        for k in k_list:
            print(f"  rerank within cosine-top{k:4d}: R@1={row[f'rerank_top{k}_r1']:.2f}%  "
                  f"(vs cosine alone {cosine_r1:.2f}%)")

    with open(os.path.join(args.output_dir, "two_stage_rerank_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[saved] {args.output_dir}/two_stage_rerank_report.json")


if __name__ == "__main__":
    main()
