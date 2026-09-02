"""
review7 block2's "가장 중요한 추가 실험" -- a genuinely FACTORIZED low-rank
residual, W = I + U V^T with U, V in R^{d x r}, trained end-to-end under the
same broad symmetric-InfoNCE recipe as train_labclip_official_recipe.py /
train_residual_identity_infonce.py.

This is a different mechanism from train_residual_identity_infonce.py: that
script keeps a FULL-RANK dW = R^{d x d} and controls its size with a scalar
Frobenius penalty (a soft constraint). Here the rank itself is the hard
constraint -- there is no dW-norm penalty by default, since the point is to
see how much of the retrieval-recovery + diagnostic-lift found by the
full-rank+penalty residual survives when the number of free parameters is cut
by up to ~30x (r=8 -> 8,192 params in U,V vs. 262,144 in a full dW).

Init: U ~ N(0, 0.02), V = 0. This is NOT the same failure mode as the earlier
rank-1 delta-loss warm-start collapse (RESULTS.md 8-V.9) -- there the *loss
itself* (gamma - max(|alpha|,|beta|)) is first-order homogeneous in W, so a
near-zero W stalls the gradient. Symmetric InfoNCE has no such homogeneity:
at V=0, W=I exactly (a perfectly good starting point with non-trivial loss),
and d(loss)/dV = U^T @ d(loss)/d(dW) is generically non-zero. Both U and V
zero would still stall (dW/dU = dW/dV = 0 identically), so only one side is
zeroed.

Usage:
    python -m benchmarks.src.evaluation.train_residual_lowrank_infonce \
        --ranks 8 16 32 64 128 256 \
        --output_dir logs/evaluation/01_paper/2026-09-02_residual_lowrank_rank_sweep
"""
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.src.evaluation.train_labclip_official_recipe import (
    build_true_caption_pairs, encode_all, symmetric_infonce,
)
from benchmarks.src.analysis.model_loader import load_clip_for_eval
from benchmarks.src.analysis.config import set_seed


def train_one_rank(rank: int, feature_dim: int, image_embs, text_embs, device,
                    epochs: int, batch_size: int, lr: float, penalty_coef: float,
                    seed: int):
    set_seed(seed)
    U = nn.Parameter(torch.randn(feature_dim, rank, device=device) * 0.02)
    V = nn.Parameter(torch.zeros(feature_dim, rank, device=device))
    eye = torch.eye(feature_dim, device=device)
    logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07)).float().to(device))
    opt = torch.optim.Adam([U, V, logit_scale], lr=lr)

    n = image_embs.shape[0]
    bs = min(batch_size, n)
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_losses = []
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            if idx.numel() < 2:
                continue
            v = image_embs[idx]
            dW = U @ V.T
            W = eye + dW
            t = F.normalize(text_embs[idx] @ W.T, dim=-1)
            infonce = symmetric_infonce(v, t, logit_scale.exp())
            loss = infonce
            if penalty_coef > 0:
                loss = loss + penalty_coef * torch.sum(dW ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        history.append(float(np.mean(epoch_losses)))
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"    [rank={rank}] epoch {epoch:3d}  loss={history[-1]:.4f}")

    with torch.no_grad():
        dW = U @ V.T
        W_final = (eye + dW).detach().cpu()
        drift = torch.norm(dW).item()
    print(f"  [rank={rank}] done, ||dW||_F = {drift:.4f}")
    return W_final, drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--output_dir", default="logs/evaluation/01_paper/2026-09-02_residual_lowrank_rank_sweep")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--penalty_coef", type=float, default=0.0,
                     help="optional extra ||dW||_F^2 penalty on top of the rank constraint; 0 = rank only")
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = build_true_caption_pairs(args.csv_path)
    print(f"[data] {len(pairs)} unique (image, true_caption) pairs, "
          f"{pairs['image_path'].nunique()} unique images")

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    image_embs, text_embs = encode_all(pairs, model, tokenizer, preprocess, device, image_root="")
    feature_dim = image_embs.shape[1]
    image_embs = image_embs.to(device)
    text_embs = text_embs.to(device)

    for rank in args.ranks:
        print(f"\n=== rank {rank} ===")
        W_final, drift = train_one_rank(
            rank, feature_dim, image_embs, text_embs, device,
            args.epochs, args.batch_size, args.lr, args.penalty_coef, args.seed,
        )
        n_params = 2 * feature_dim * rank
        ckpt_path = os.path.join(args.output_dir, f"residual_lowrank_r{rank}.pt")
        torch.save({
            "model_name": "bilinear",
            "rank": rank,
            "n_params": n_params,
            "dW_norm": drift,
            "state_dict": {"W": W_final, "bias": torch.zeros(1)},
        }, ckpt_path)
        print(f"  [saved] {ckpt_path}  (n_params={n_params})")


if __name__ == "__main__":
    main()
