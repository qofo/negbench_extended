"""
LABCLIP rank comparison, condition B: "impose the rank from the start."

W = U V^T with U, V in R^{d x r} -- a plain low-rank bilinear form with NO
identity/cosine term -- trained end-to-end under the same broad symmetric
InfoNCE recipe as train_labclip_official_recipe.py, at each rank.

The three conditions this belongs to differ only in *where the rank
constraint enters*:
  A (eval_labclip_posthoc_svd_rank.py)   train full-rank, then SVD-truncate
  B (this script)                        W = UV^T, rank fixed before training
  C (train_residual_lowrank_infonce.py)  W = I + UV^T, rank fixed before
                                         training AND the diagonal/cosine
                                         component is kept exactly at identity

Initialization matters here in a way it does not for C, and the first run of
this script got it wrong. Copying C's convention (U ~ N(0, 0.02), V = 0)
makes W start at exactly 0, so `normalize(text @ W.T)` normalizes a zero
vector and the gradient through it vanishes: training stalled completely,
||W||_F reaching only 0.014-0.077 after 60 epochs at every rank. C never hits
this because it starts at W = I. That stall is a property of the
parametrization's init, not of the low-rank family, so the fair default here
is `--init both_random`: U, V ~ N(0, 1/sqrt(d)), which puts ||UV^T||_F in the
same range as a trained residual. `--init zero_v` reproduces the stalled run.

Usage:
    python -m benchmarks.src.evaluation.train_lowrank_plain_infonce \
        --ranks 8 16 32 64 128 256 \
        --output_dir logs/evaluation/01_paper/2026-09-02_lowrank_plain_rank_sweep
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
                    epochs: int, batch_size: int, lr: float, seed: int,
                    init: str = "both_random"):
    set_seed(seed)
    if init == "zero_v":
        U = nn.Parameter(torch.randn(feature_dim, rank, device=device) * 0.02)
        V = nn.Parameter(torch.zeros(feature_dim, rank, device=device))
    else:
        std = feature_dim ** -0.5
        U = nn.Parameter(torch.randn(feature_dim, rank, device=device) * std)
        V = nn.Parameter(torch.randn(feature_dim, rank, device=device) * std)
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
            W = U @ V.T
            t = F.normalize(text_embs[idx] @ W.T, dim=-1)
            loss = symmetric_infonce(v, t, logit_scale.exp())
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        history.append(float(np.mean(epoch_losses)))
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"    [rank={rank}] epoch {epoch:3d}  loss={history[-1]:.4f}")

    with torch.no_grad():
        W_final = (U @ V.T).detach().cpu()
        w_norm = torch.norm(W_final).item()
    print(f"  [rank={rank}] done, ||W||_F = {w_norm:.4f}")
    return W_final, w_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--output_dir", default="logs/evaluation/01_paper/2026-09-02_lowrank_plain_rank_sweep")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--init", choices=["both_random", "zero_v"], default="both_random",
                     help="both_random: U,V ~ N(0,1/sqrt(d)) (fair default). "
                          "zero_v: U~N(0,0.02), V=0 -- reproduces the stalled first run.")
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
        W_final, w_norm = train_one_rank(rank, feature_dim, image_embs, text_embs, device,
                                          args.epochs, args.batch_size, args.lr, args.seed, args.init)
        suffix = "" if args.init == "both_random" else f"_{args.init}"
        ckpt_path = os.path.join(args.output_dir, f"lowrank_plain_r{rank}{suffix}.pt")
        torch.save({
            "model_name": "bilinear",
            "rank": rank,
            "n_params": 2 * feature_dim * rank,
            "w_norm": w_norm,
            "init": args.init,
            "state_dict": {"W": W_final, "bias": torch.zeros(1)},
        }, ckpt_path)
        print(f"  [saved] {ckpt_path}")


if __name__ == "__main__":
    main()
