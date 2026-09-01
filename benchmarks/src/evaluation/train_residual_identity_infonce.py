"""
Priority 4 -- residual parametrization W = I + dW (review7 block1 방안B).

Same architecture and broad-InfoNCE training regime as
train_labclip_official_recipe.py, except the text-side matrix is written as
W = I + dW and the loss adds a Frobenius penalty on dW:

    L = L_InfoNCE(W) + penalty_coef * ||dW||_F^2

The identity-init full-rank Linear in the official LABCLIP recipe already starts
at W=I and is free to drift arbitrarily far (it reached ||W-I||_F=12.10 in
train_labclip_official_recipe.py). This experiment instead constrains that
drift directly, testing whether keeping W close to cosine while still fitting
the broad contrastive objective preserves more of cosine's retrieval strength.

Usage:
    python -m benchmarks.src.evaluation.train_residual_identity_infonce \
        --penalty_coef 0.01 \
        --output_dir logs/evaluation/01_paper/2026-09-02_residual_identity_infonce
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--output_dir", default="logs/evaluation/01_paper/2026-09-02_residual_identity_infonce")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--penalty_coef", type=float, default=0.01,
                     help="weight on ||dW||_F^2; larger keeps W closer to I (=cosine)")
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

    dW = nn.Parameter(torch.zeros(feature_dim, feature_dim, device=device))
    eye = torch.eye(feature_dim, device=device)
    logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07)).float().to(device))
    opt = torch.optim.Adam([dW, logit_scale], lr=args.lr)

    n = image_embs.shape[0]
    bs = min(args.batch_size, n)
    print(f"[train] n={n}, batch_size={bs}, penalty_coef={args.penalty_coef}")

    history = []
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        epoch_losses, epoch_infonce, epoch_penalty = [], [], []
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            if idx.numel() < 2:
                continue
            v = image_embs[idx]
            W = eye + dW
            t = F.normalize(text_embs[idx] @ W.T, dim=-1)
            infonce = symmetric_infonce(v, t, logit_scale.exp())
            penalty = args.penalty_coef * torch.sum(dW ** 2)
            loss = infonce + penalty
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
            epoch_infonce.append(infonce.item())
            epoch_penalty.append(penalty.item())
        history.append(float(np.mean(epoch_losses)))
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  loss={history[-1]:.4f}  "
                  f"infonce={np.mean(epoch_infonce):.4f}  penalty={np.mean(epoch_penalty):.4f}")

    with torch.no_grad():
        W_final = (eye + dW).detach().cpu()
        drift = torch.norm(dW).item()
    print(f"[done] ||dW||_F = {drift:.4f}")

    ckpt_path = os.path.join(args.output_dir, f"residual_identity_infonce_pen{args.penalty_coef}.pt")
    torch.save({
        "model_name": "bilinear",
        "rank": None,
        "state_dict": {"W": W_final, "bias": torch.zeros(1)},
    }, ckpt_path)
    print(f"[saved] {ckpt_path}")


if __name__ == "__main__":
    main()
