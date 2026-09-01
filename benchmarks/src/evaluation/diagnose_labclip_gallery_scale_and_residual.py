"""
Two cheap, no-retraining diagnostics for the LABCLIP-official-recipe W trained on
our BEAF data (train_labclip_official_recipe.py), applied to the COCO retrieval
target it was already scored on. Both reuse the embeddings eval_zero_shot_transfer.py
already cached to disk -- no re-encoding, no retraining.

Priority 1 -- gallery-size scaling curve:
    T2I R@1 for cosine vs. W at gallery sizes [2, 4, 16, 64, 256, 1024, 5000],
    built by repeatedly sampling a random gallery of size N (always containing the
    true image) per query and averaging over draws. Distinguishes "W lost the
    signal everywhere" (flat gap at every N) from "W only fails at large-corpus
    scale" (gap widens with N).

Priority 2 -- residual scoring test:
    s = s_cos + lambda * s_W (both score matrices standardized to zero-mean/unit-std
    first, since cosine and the bilinear score are not on the same scale) on the
    FULL 5000-image gallery, sweeping lambda. Shows whether blending in W ever helps
    or only ever hurts, and how gracefully performance degrades as lambda grows.

Usage:
    python -m benchmarks.src.evaluation.diagnose_labclip_gallery_scale_and_residual \
        --ckpt logs/evaluation/01_paper/2026-09-01_labclip_official_recipe/labclip_official_recipe_bilinear.pt \
        --cache logs/evaluation/cached_embeddings/COCO_val_retrieval_retrieval_embeds.pt \
        --output_dir logs/evaluation/01_paper/2026-09-02_labclip_scale_residual_diagnostics
"""
import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F


def load_scores(ckpt_path: str, cache_path: str, device: str):
    cached = torch.load(cache_path, map_location="cpu")
    images_emb = cached["images_emb"].float()  # (N_img, D), already L2-normalized
    texts_emb = cached["texts_emb"].float()    # (N_txt, D)
    texts_image_index = torch.tensor(cached["texts_image_index"], dtype=torch.long)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    W = ckpt["state_dict"]["W"].float()

    images_emb_d = images_emb.to(device)
    texts_emb_d = texts_emb.to(device)
    W_d = W.to(device)

    cos_scores = (texts_emb_d @ images_emb_d.T).cpu()               # (N_txt, N_img)
    w_scores = (texts_emb_d @ W_d.T @ images_emb_d.T).cpu()         # (N_txt, N_img)

    return cos_scores, w_scores, texts_image_index, images_emb.shape[0], texts_emb.shape[0]


def gallery_scaling_curve(cos_scores, w_scores, texts_image_index, n_img, sizes, n_draws=50, seed=42):
    rng = np.random.default_rng(seed)
    n_txt = cos_scores.shape[0]
    results = {}
    for N in sizes:
        N = min(N, n_img)
        cos_hits, w_hits = [], []
        for draw in range(n_draws):
            # For each query, build a random gallery of size N containing the true image,
            # by sampling ONE shared random distractor pool per draw (fast, standard practice)
            # and checking whether the true image survives + wins within that pool.
            distractor_pool = rng.choice(n_img, size=N, replace=False)
            pool_t = torch.from_numpy(distractor_pool)
            true_idx = texts_image_index
            # vectorized per-query: gather scores at pool_t, plus force true image into the pool
            cos_sub = cos_scores[:, pool_t]         # (n_txt, N)
            w_sub = w_scores[:, pool_t]
            # for queries where the true image is missing from the sampled pool, append its
            # own true-image score as an (N+1)-th column so R@1 still asks "does it win vs N others"
            true_cos = cos_scores[torch.arange(n_txt), true_idx]
            true_w = w_scores[torch.arange(n_txt), true_idx]
            cos_full = torch.cat([cos_sub, true_cos.unsqueeze(1)], dim=1)
            w_full = torch.cat([w_sub, true_w.unsqueeze(1)], dim=1)
            true_col = cos_full.shape[1] - 1
            # a query "hits" R@1 iff the true image's own score is the max across the sampled
            # gallery (its true-image column is always included, so this is exact regardless
            # of whether it was also independently drawn into the random pool)
            cos_hit = (cos_full.max(dim=1).values == true_cos)
            w_hit = (w_full.max(dim=1).values == true_w)
            cos_hits.append(cos_hit.float().mean().item())
            w_hits.append(w_hit.float().mean().item())
        results[N] = {
            "cosine_r1_mean": float(np.mean(cos_hits)) * 100,
            "cosine_r1_std": float(np.std(cos_hits)) * 100,
            "w_r1_mean": float(np.mean(w_hits)) * 100,
            "w_r1_std": float(np.std(w_hits)) * 100,
        }
    return results


def residual_scoring_sweep(cos_scores, w_scores, texts_image_index, lambdas):
    # standardize both score matrices (zero-mean, unit-std) so lambda is comparable
    cos_z = (cos_scores - cos_scores.mean()) / cos_scores.std()
    w_z = (w_scores - w_scores.mean()) / w_scores.std()
    n_txt = cos_scores.shape[0]
    results = {}
    for lam in lambdas:
        combo = cos_z + lam * w_z
        top1 = combo.argmax(dim=1)
        hit = (top1 == texts_image_index).float().mean().item() * 100
        results[lam] = hit
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/evaluation/01_paper/2026-09-01_labclip_official_recipe/labclip_official_recipe_bilinear.pt")
    ap.add_argument("--cache", default="logs/evaluation/cached_embeddings/COCO_val_retrieval_retrieval_embeds.pt")
    ap.add_argument("--output_dir", default="logs/evaluation/01_paper/2026-09-02_labclip_scale_residual_diagnostics")
    ap.add_argument("--n_draws", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cos_scores, w_scores, texts_image_index, n_img, n_txt = load_scores(args.ckpt, args.cache, device)
    print(f"[data] {n_txt} texts x {n_img} images")

    print("\n=== Priority 1: gallery-size scaling curve ===")
    sizes = [2, 4, 16, 64, 256, 1024, 5000]
    scaling = gallery_scaling_curve(cos_scores, w_scores, texts_image_index, n_img, sizes, n_draws=args.n_draws)
    for N, r in scaling.items():
        print(f"  N={N:5d}   cosine R@1={r['cosine_r1_mean']:.2f}±{r['cosine_r1_std']:.2f}%   "
              f"W R@1={r['w_r1_mean']:.2f}±{r['w_r1_std']:.2f}%   gap={r['cosine_r1_mean']-r['w_r1_mean']:.2f}pp")

    print("\n=== Priority 2: residual scoring sweep (s = s_cos + lambda * s_W, standardized) ===")
    lambdas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    residual = residual_scoring_sweep(cos_scores, w_scores, texts_image_index, lambdas)
    for lam, r1 in residual.items():
        print(f"  lambda={lam:.2f}   T2I R@1={r1:.2f}%")

    with open(os.path.join(args.output_dir, "diagnostics_report.json"), "w") as f:
        json.dump({
            "gallery_scaling": {str(k): v for k, v in scaling.items()},
            "residual_sweep": {str(k): v for k, v in residual.items()},
        }, f, indent=2)
    print(f"\n[saved] {args.output_dir}/diagnostics_report.json")


if __name__ == "__main__":
    main()
