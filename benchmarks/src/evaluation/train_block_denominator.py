"""
E-C (IMPLEMENTATION_PLAN.md Part XVI) -- put the counterfactual twin in the denominator.

Every W trained in this repo so far saw either the block or the batch, never both
inside one softmax:

  narrow (8-V.9/10)   only the four cells, no random negatives  -> 2x2 33.02%, R@1 0.12%
  broad  (8-V.12)     only random negatives, no twin            -> 2x2 10.04%, R@1 17.13%
  hybrid (8-V.13)     L_InfoNCE + lambda*L_delta, summed         -> indistinguishable from broad

The hybrid failure was not a loss-scale artifact (it survived exact rescaling). Summing
two losses never makes the twin *compete* with the random negatives for probability
mass; it only adds a second gradient. This script instead extends the denominator of
the one symmetric InfoNCE, which is the only construction under which the absolute-
ranking pressure that preserves geometry stays on while the interaction is supervised.

Two independent bits, because 8.10.2's duality says they buy different things:

  --twin_image    the counterfactual image joins the text-anchored (T2I) candidates
                  -> pressure on gamma > |beta|  (image selection, the wide 46% target)
  --swap_caption  the polarity-flipped caption joins the image-anchored (I2T) candidates
                  -> pressure on gamma > |alpha| (caption selection, the narrow 3.95% target)

Pre-registered: A1 (twin only) moves the image side and not the caption side, A2 the
mirror, and A3 must beat both singles on the group condition -- that condition is their
intersection. If A3 does not, the "block in the denominator" account is wrong and the
Part XVI-7 fallback (the frozen embeddings cannot supply gamma) is what remains.

Structure is held fixed at the only parameterization known to preserve retrieval
(W = I + dW with a Frobenius penalty, 8-V.12/13/14): the variable here is the
denominator, not the capacity.

Usage (from the repo root):
    python -m benchmarks.src.evaluation.train_block_denominator \\
        --output_dir logs/evaluation/01_paper/2026-09-07_block_denominator
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.feature_cache import cached_encode, build_provenance, DEFAULT_CACHE_DIR
    from benchmarks.src.analysis.config import set_seed
    from benchmarks.src.analysis.paths import resolve_image_path as resolve_path
    from benchmarks.src.analysis.beaf.beaf_loader import load_and_verify_counterfactual_pairs
    from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, encode_texts_unified,
    )
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.model_loader import load_clip_for_eval
    from analysis.feature_cache import cached_encode, build_provenance, DEFAULT_CACHE_DIR
    from analysis.config import set_seed
    from analysis.paths import resolve_image_path as resolve_path
    from analysis.beaf.beaf_loader import load_and_verify_counterfactual_pairs
    from evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, encode_texts_unified,
    )

ARMS = {
    "A0_neither": (False, False),
    "A1_twin_image": (True, False),
    "A2_swap_caption": (False, True),
    "A3_both": (True, True),
}


def build_items(csv_path: str, image_root: str, model, preprocess, tokenizer, device,
                cache_kw, batch_size: int):
    """Each counterfactual pair yields two training items, and each item carries the
    two extra candidates its 2x2 block defines.

        item (v=orig, t=positive)  twin=cf    swap=negative
        item (v=cf,   t=negative)  twin=orig  swap=positive

    So the twin of an item is the other item's image and the swap is the other item's
    caption -- the block is closed, which is what makes the two bits independent.
    """
    _, pairs, _ = load_and_verify_counterfactual_pairs(csv_path, image_root)
    print(f"  {len(pairs)} verified counterfactual pairs, "
          f"{pairs['object_name'].nunique()} concepts")

    img_paths = sorted(set(pairs["orig_path"].tolist()) | set(pairs["cf_path"].tolist()))
    captions = sorted(set(pairs["positive_caption"].tolist()) | set(pairs["negative_caption"].tolist()))
    img_emb, _, flags = cached_encode(
        lambda: encode_images_unified(model, preprocess, img_paths, device, batch_size),
        kind="blockdenom_images@l2norm+raw+flags", items=img_paths, **cache_kw)
    txt_emb, _ = cached_encode(
        lambda: encode_texts_unified(model, tokenizer, captions, device, batch_size),
        kind="blockdenom_captions@l2norm+raw", items=captions, **cache_kw)
    if not np.all(flags):
        keep_img = {p for p, ok in zip(img_paths, flags) if ok}
        pairs = pairs[pairs["orig_path"].isin(keep_img) & pairs["cf_path"].isin(keep_img)]
        print(f"  dropped {(~flags).sum()} unreadable images -> {len(pairs)} pairs")

    ii = {p: i for i, p in enumerate(img_paths)}
    ti = {c: i for i, c in enumerate(captions)}
    v_idx, t_idx, vtw_idx, tsw_idx, groups = [], [], [], [], []
    for _, r in pairs.iterrows():
        o, c = ii[r["orig_path"]], ii[r["cf_path"]]
        p, n = ti[r["positive_caption"]], ti[r["negative_caption"]]
        v_idx += [o, c]
        t_idx += [p, n]
        vtw_idx += [c, o]
        tsw_idx += [n, p]
        groups += [r["object_name"], r["object_name"]]
    return (torch.from_numpy(img_emb).float(), torch.from_numpy(txt_emb).float(),
            np.array(v_idx), np.array(t_idx), np.array(vtw_idx), np.array(tsw_idx), groups)


def block_infonce(v, t, v_tw, t_sw, logit_scale, use_twin: bool, use_swap: bool):
    """One symmetric InfoNCE whose denominators optionally carry the block.

    With both bits off this is exactly `symmetric_infonce` from the official-recipe
    script; the extra candidates are appended as columns, so the positive stays on
    the diagonal and the label vector is unchanged.
    """
    b = v.shape[0]
    labels = torch.arange(b, device=v.device)
    cand_t = torch.cat([t, t_sw], 0) if use_swap else t
    cand_v = torch.cat([v, v_tw], 0) if use_twin else v
    loss_i2t = F.cross_entropy(logit_scale * v @ cand_t.t(), labels)
    loss_t2i = F.cross_entropy(logit_scale * t @ cand_v.t(), labels)
    return (loss_i2t + loss_t2i) / 2.0


def train_arm(arm: str, img_emb, txt_emb, idx, args, device):
    use_twin, use_swap = ARMS[arm]
    v_idx, t_idx, vtw_idx, tsw_idx = idx
    set_seed(args.seed)  # every arm starts from the same stream

    dim = img_emb.shape[1]
    dW = nn.Parameter(torch.zeros(dim, dim, device=device))
    eye = torch.eye(dim, device=device)
    logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)).float().to(device))
    opt = torch.optim.Adam([dW, logit_scale], lr=args.lr)

    V = img_emb.to(device)
    T = txt_emb.to(device)
    n = len(v_idx)
    bs = min(args.batch_size, n)
    vi = torch.from_numpy(v_idx).long().to(device)
    ti = torch.from_numpy(t_idx).long().to(device)
    wi = torch.from_numpy(vtw_idx).long().to(device)
    si = torch.from_numpy(tsw_idx).long().to(device)

    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        losses = []
        for start in range(0, n, bs):
            sel = perm[start:start + bs]
            if sel.numel() < 2:
                continue
            W = eye + dW
            v = V[vi[sel]]
            v_tw = V[wi[sel]]
            t = F.normalize(T[ti[sel]] @ W.T, dim=-1)
            t_sw = F.normalize(T[si[sel]] @ W.T, dim=-1)
            infonce = block_infonce(v, t, v_tw, t_sw, logit_scale.exp(), use_twin, use_swap)
            loss = infonce + args.penalty_coef * torch.sum(dW ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"    epoch {epoch:3d}  loss={np.mean(losses):.4f}  "
                  f"||dW||_F={torch.norm(dW).item():.3f}")

    with torch.no_grad():
        return (eye + dW).detach().cpu(), float(torch.norm(dW).item())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv_path",
                    default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--penalty_coef", type=float, default=0.1,
                    help="Frobenius penalty on dW; 0.1 is the 8-V.12 operating point")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--output_dir",
                    default="logs/evaluation/01_paper/2026-09-07_block_denominator")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  E-C -- the counterfactual twin inside the denominator")
    print("=" * 70)
    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    cache_kw = dict(model=args.model, pretrained=args.pretrained,
                    enabled=args.use_cache, cache_dir=args.cache_dir)

    print("\n[1/2] building items...")
    img_emb, txt_emb, v_idx, t_idx, vtw_idx, tsw_idx, groups = build_items(
        args.csv_path, args.image_root, model, preprocess, tokenizer, device,
        cache_kw, args.batch_size)
    print(f"  {len(v_idx)} training items ({img_emb.shape[0]} unique images, "
          f"{txt_emb.shape[0]} unique captions)")

    print("[2/2] training arms...")
    report = {}
    for arm in args.arms:
        use_twin, use_swap = ARMS[arm]
        print(f"\n  {arm}  (twin_image={use_twin}, swap_caption={use_swap})")
        W, drift = train_arm(arm, img_emb, txt_emb,
                             (v_idx, t_idx, vtw_idx, tsw_idx), args, device)
        path = os.path.join(args.output_dir, f"{arm}.pt")
        torch.save({"model_name": "bilinear", "rank": None,
                    "state_dict": {"W": W, "bias": torch.zeros(1)}}, path)
        report[arm] = {"drift_frobenius": drift, "checkpoint": path,
                       "twin_image": use_twin, "swap_caption": use_swap}
        print(f"    saved {path}   ||dW||_F={drift:.3f}")

    with open(os.path.join(args.output_dir, "block_denominator_training.json"), "w") as f:
        json.dump({"arms": report, "config": vars(args),
                   "provenance": build_provenance(args, items=len(v_idx))}, f,
                  indent=2, default=str)
    print(f"\nsaved: {args.output_dir}/block_denominator_training.json")


if __name__ == "__main__":
    main()
