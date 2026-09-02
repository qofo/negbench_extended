"""
Priority 3 -- loss-source ablation (review7 block1 방안A's diagnostic form).

Same rank-32 LowRankMatcher architecture used by margin4/delta throughout this
project, trained under three different loss sources on the SAME underlying
BEAF AB-swap data pool, to isolate whether it is the LOSS (narrow 2x2-only vs.
broad in-batch-negative InfoNCE) or the DATA that drives the gallery-retrieval
collapse:

  (i)   delta-only         -- already trained (train_and_save_narrow_rank32_w.py),
                               not re-run here.
  (ii)  infonce-only        -- symmetric InfoNCE over broad (image, true_caption)
                               pairs (same construction as
                               train_labclip_official_recipe.py), rank-32
                               low-rank projections instead of a full-rank text
                               linear.
  (iii) hybrid              -- L_total = L_InfoNCE(broad batch) + lambda * L_Delta
                               (full narrow quads), summed every step, per
                               review7 block1's own proposed formula.

2026-09-02 update (review7's own follow-up caution): at lam=1, InfoNCE
(~2-6) and the delta hinge (~0.05-0.15) differ in scale by ~20-100x, so the
delta term's gradient is negligible and "hybrid" silently collapses to
infonce_only (RESULTS.md 8-V.12 point 3 -- rank32_hybrid == rank32_infonce_only
to the decimal place, ||W||_F=1.29 both). --balance (default on) fixes this by
rescaling the delta term at every step by the ratio of its own detached
magnitude to InfoNCE's detached magnitude, so lam multiplies two losses that
start out comparable rather than two losses 20-100x apart. Pass --no-balance
to reproduce the old (broken) raw-sum behavior for comparison.

Usage:
    python -m benchmarks.src.evaluation.train_rank32_loss_ablation \
        --output_dir logs/evaluation/01_paper/2026-09-02_rank32_loss_ablation
"""
import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.src.evaluation.eval_single_w_generalization import (
    build_family, load_quads, get_embed_dim,
)
from benchmarks.src.evaluation.train_labclip_official_recipe import (
    build_true_caption_pairs, encode_all, symmetric_infonce,
)
from benchmarks.src.analysis.model_loader import load_clip_for_eval
from benchmarks.src.analysis.config import set_seed


def delta_loss_term(net, quads, margin: float) -> torch.Tensor:
    v_pos, v_neg, t_pos, t_neg = quads
    s_pp = net(v_pos, t_pos)
    s_mm = net(v_neg, t_neg)
    s_pm = net(v_pos, t_neg)
    s_mp = net(v_neg, t_pos)
    gamma = (s_pp - s_pm - s_mp + s_mm) / 4.0
    beta = (s_pp + s_pm - s_mp - s_mm) / 4.0
    alpha = (s_pp - s_pm + s_mp - s_mm) / 4.0
    main_effect = torch.maximum(alpha.abs(), beta.abs())
    per_pair_margin = gamma - main_effect
    return F.relu(margin - per_pair_margin).mean()


def train_condition(condition: str, embed_dim: int, seed: int, device: str,
                     narrow_quads, broad_images_emb, broad_texts_emb,
                     epochs: int, batch_size: int, lr: float, margin: float,
                     lam: float, balance: bool = True):
    set_seed(seed)
    net, _ = build_family("lowrank_32", embed_dim, seed, warm_start=None)
    net = net.to(device)
    logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07)).float().to(device))
    opt = torch.optim.Adam(list(net.parameters()) + [logit_scale], lr=lr)

    n = broad_images_emb.shape[0]
    bs = min(batch_size, n)
    history = []
    net.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_losses = []
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            if idx.numel() < 2:
                continue
            opt.zero_grad()
            loss = torch.tensor(0.0, device=device)
            infonce_term = None
            if condition in ("infonce_only", "hybrid"):
                v = broad_images_emb[idx]
                t_raw = broad_texts_emb[idx]
                zv = F.normalize(net.proj_v(v), dim=-1)
                zt = F.normalize(net.proj_t(t_raw), dim=-1)
                infonce_term = symmetric_infonce(zv, zt, logit_scale.exp())
                loss = loss + infonce_term
            if condition in ("delta_only", "hybrid"):
                d_loss = delta_loss_term(net, narrow_quads, margin)
                if condition == "hybrid":
                    if balance and d_loss.item() > 1e-8:
                        # rescale delta to the same detached magnitude as
                        # infonce before applying lam, so lam trades off two
                        # comparable quantities instead of two ~20-100x
                        # apart (review7's own diagnosed failure mode).
                        scale = infonce_term.detach() / d_loss.detach()
                        weight = lam * scale
                    else:
                        weight = lam
                else:
                    weight = 1.0
                loss = loss + weight * d_loss
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        history.append(float(np.mean(epoch_losses)))
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"    [{condition}] epoch {epoch:3d}  loss={history[-1]:.4f}")
    net.eval()
    with torch.no_grad():
        W = (net.proj_v.weight.T @ net.proj_t.weight).cpu()
    return W, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--output_dir", default="logs/evaluation/01_paper/2026-09-02_rank32_loss_ablation")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=1.0, help="hybrid weight on the delta term")
    ap.add_argument("--balance", action="store_true", default=True,
                     help="rescale delta to InfoNCE's detached magnitude before applying lam (default on)")
    ap.add_argument("--no-balance", dest="balance", action="store_false")
    ap.add_argument("--conditions", nargs="+", default=["hybrid"],
                     choices=["infonce_only", "delta_only", "hybrid"])
    ap.add_argument("--tag", default=None, help="suffix for the saved checkpoint name, e.g. 'lam1_balanced'")
    ap.add_argument("--min_pairs", type=int, default=20)
    ap.add_argument("--image_root", default="benchmarks/data/images")
    ap.add_argument("--restrict_objects", default=None)
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--cache_dir", default="logs/evaluation/cached_embeddings/feature_cache")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_kw = dict(model=args.model, pretrained=str(args.pretrained),
                     cache_dir=args.cache_dir, enabled=args.use_cache)

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    embed_dim = get_embed_dim(model)

    print("[data] loading narrow 2x2 quads (delta term)...")
    v_pos, v_neg, t_pos, t_neg, groups = load_quads(args, model, preprocess, tokenizer, device, cache_kw)
    T = lambda a: torch.from_numpy(a).float().to(device)
    narrow_quads = (T(v_pos), T(v_neg), T(t_pos), T(t_neg))

    print("[data] loading broad (image, true_caption) pairs (InfoNCE term)...")
    pairs = build_true_caption_pairs(args.csv_path)
    broad_images_emb, broad_texts_emb = encode_all(pairs, model, tokenizer, preprocess, device, image_root="")
    broad_images_emb = broad_images_emb.to(device)
    broad_texts_emb = broad_texts_emb.to(device)
    print(f"[data] broad pairs: {broad_images_emb.shape[0]}, narrow quads: {narrow_quads[0].shape[0]}")

    for condition in args.conditions:
        print(f"\n=== training condition: {condition} "
              f"(lam={args.lam}, balance={args.balance if condition == 'hybrid' else 'n/a'}) ===")
        W, history = train_condition(
            condition, embed_dim, args.seed, device, narrow_quads,
            broad_images_emb, broad_texts_emb, args.epochs, args.batch_size,
            args.lr, args.margin, args.lam, args.balance,
        )
        drift = torch.norm(W).item()
        print(f"  ||W||_F = {drift:.2f}")
        suffix = f"_{args.tag}" if args.tag else ""
        out_path = os.path.join(args.output_dir, f"rank32_{condition}{suffix}.pt")
        torch.save({"model_name": "bilinear", "rank": 32,
                    "state_dict": {"W": W, "bias": torch.zeros(1)}}, out_path)
        print(f"  [saved] {out_path}")


if __name__ == "__main__":
    main()
