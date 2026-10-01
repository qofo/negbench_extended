"""
InfoNCE text-alignment that KEEPS the cosine identity: residual low-rank
W = I + U V^T with a tunable Frobenius drift penalty, swept over rank AND
penalty, plus the full-rank (LABCLIP) point.

Why this shape
--------------
* A pure low-rank W = U V^T cannot sit near I for r < d: I has a flat
  spectrum, so its best rank-r approximation keeps only r/d of its energy.
  The identity must therefore be KEPT, not learned -- W = I + (low-rank
  correction).
* Init U ~ N(0, 0.02), V = 0  ->  W_0 = I exactly.  Symmetric InfoNCE has no
  first-order homogeneity in W, so W_0 = I is a valid start (non-trivial loss,
  non-zero grad through V) -- unlike the DeltaS margin loss, which stalls at
  W ~ I.
* Identity-init alone does NOT keep W near I: a plain-InfoNCE HP search (lr x
  wd x batch, selected on held-out InfoNCE loss) drifts to ||W-I||_F = 24..60
  at every setting.  A drift control is required.  Here it is the soft
  penalty  L = InfoNCE(W) + c * ||W - I||_F^2 ;  c -> inf recovers cosine,
  c = 0 is free drift.  (Hard alternatives: orthogonality W^T W = I, or
  early-stop on retrieval R@1.)

Recipe fixes over train_labclip_official_recipe.py (needed because our pool is
1,690 images x 3.28 captions, not COCO's ~1/img):
  * false-negative masking: batch rows sharing an image are not negatives of
    each other in the InfoNCE.
  * logit_scale clamped to <= ln(100), as in real CLIP.
  * AdamW + cosine LR decay; lr / batch fixed at the HP-search plateau
    (lr 1e-3, batch 256 -- val InfoNCE loss was flat, 4.55-4.68, across the
    whole grid, so c is the hyper-parameter that matters).

Usage:
  python -m benchmarks.src.evaluation.train_infonce_identity_init \
      --ranks 1 2 4 8 16 32 64 128 256 512 \
      --penalties 0 0.001 0.01 0.1 \
      --output_dir logs/evaluation/01_paper/2026-09-10_infonce_idinit
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.src.evaluation.train_labclip_official_recipe import encode_all
from benchmarks.src.evaluation.eval_single_w_generalization import load_quads
from benchmarks.src.evaluation.score_delta_s_for_w import joint_correct_for_w
from benchmarks.src.analysis.model_loader import load_clip_for_eval
from benchmarks.src.analysis.config import set_seed, coerce_bool_column

D = 512
CSV = "benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv"
STD_CACHE = "logs/evaluation/cached_embeddings/COCO_val_retrieval_retrieval_embeds.pt"
NEG_CACHE = "logs/evaluation/cached_embeddings/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true_retrieval_embeds.pt"
ENC_CACHE = "logs/evaluation/cached_embeddings/beaf_truecap_idinit_encodings.pt"


def get_encodings(device):
    if os.path.exists(ENC_CACHE):
        d = torch.load(ENC_CACHE, map_location="cpu", weights_only=False)
        return d["img"], d["txt"], d["img_id"]
    df = coerce_bool_column(pd.read_csv(CSV), "object_in_image")
    df["true_caption"] = np.where(df["object_in_image"], df["positive_caption"], df["negative_caption"])
    pairs = df[["image_path", "true_caption"]].drop_duplicates().reset_index(drop=True)
    model, preprocess, tokenizer = load_clip_for_eval("ViT-B-32", "openai", device)
    img, txt = encode_all(pairs, model, tokenizer, preprocess, device, image_root="")
    codes, _ = pd.factorize(pairs["image_path"])
    torch.save({"img": img, "txt": txt, "img_id": torch.tensor(codes)}, ENC_CACHE)
    return img, txt, torch.tensor(codes)


def masked_infonce(v, t, scale, img_id):
    logits = scale * v @ t.t()
    same = (img_id[:, None] == img_id[None, :]) & ~torch.eye(len(v), dtype=torch.bool, device=v.device)
    logits = logits.masked_fill(same, float("-inf"))
    lab = torch.arange(len(v), device=v.device)
    return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))


def train(rank, c, img, txt, img_id, device, *, lr=1e-3, epochs=200, bs=256, seed=42,
          val_idx=None, param="residual"):
    set_seed(seed)
    img, txt, img_id = img.to(device), txt.to(device), img_id.to(device)
    eye = torch.eye(D, device=device)
    full = rank >= D
    if param == "plain":
        # W = U V^T, random both sides -- no identity kept (the paper's earlier
        # plain low-rank; init at ||W-I||_F ~ 23, cosine broken from step 0).
        std = D ** -0.5
        r = min(rank, D)
        U = nn.Parameter(torch.randn(D, r, device=device) * std)
        V = nn.Parameter(torch.randn(D, r, device=device) * std)
        P = [U, V]
        dW = lambda: U @ V.t() - eye          # so W = eye + dW = U V^T
    elif full:
        P = [nn.Parameter(torch.zeros(D, D, device=device))]
        dW = lambda: P[0]
    else:
        U = nn.Parameter(torch.randn(D, rank, device=device) * 0.02)
        V = nn.Parameter(torch.zeros(D, rank, device=device))
        P = [U, V]
        dW = lambda: U @ V.t()
    ls = nn.Parameter(torch.tensor(np.log(1 / 0.07)).float().to(device))
    opt = torch.optim.AdamW(P + [ls], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(img)
    tr = np.arange(n) if val_idx is None else np.setdiff1d(np.arange(n), val_idx)
    tr = torch.tensor(tr, device=device)
    for ep in range(epochs):
        perm = tr[torch.randperm(len(tr), device=device)]
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            if len(idx) < 3:
                continue
            W = eye + dW()
            t = F.normalize(txt[idx] @ W.t(), dim=-1)
            loss = masked_infonce(img[idx], t, ls.clamp(max=np.log(100.)).exp(), img_id[idx])
            if c > 0:
                loss = loss + c * (dW() ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    with torch.no_grad():
        return (eye + dW()).detach().cpu().float()


def load_gal(path, device):
    x = torch.load(path, map_location="cpu")
    return (x["images_emb"].float().to(device), x["texts_emb"].float().to(device),
            torch.tensor(x["texts_image_index"]))


def retr(W, ie, te, idx, device):
    s = (te @ W.to(device).t() @ ie.t()).cpu()
    r1 = (s.argmax(1) == idx).float().mean().item() * 100
    r5 = (s.topk(5, 1).indices == idx[:, None]).any(1).float().mean().item() * 100
    return r1, r5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--penalties", type=float, nargs="+", default=[0.0, 1e-3, 1e-2, 1e-1])
    ap.add_argument("--param", choices=["residual", "plain"], default="residual",
                    help="residual: W=I+UV^T (identity kept). plain: W=UV^T random both sides.")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--output_dir", default="logs/evaluation/01_paper/2026-09-10_infonce_idinit")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    img, txt, img_id = get_encodings(dev)
    print(f"[data] {len(img)} pairs, {len(torch.unique(img_id))} images")

    mc, preprocess, tok = load_clip_for_eval("ViT-B-32", "openai", dev)
    qa = argparse.Namespace(csv_path=CSV, image_root="benchmarks/data/images", model="ViT-B-32",
                            pretrained="openai", min_pairs=20, restrict_objects=None, use_cache=True,
                            cache_dir="logs/evaluation/cached_embeddings/feature_cache",
                            batch_size=256, seed=args.seed)
    vp, vn, tp, tn, groups = load_quads(qa, mc, preprocess, tok, dev,
                                        dict(model="ViT-B-32", pretrained="openai",
                                             cache_dir=qa.cache_dir, enabled=True))
    T = lambda a: torch.from_numpy(a).float().to(dev)
    quads = (T(vp), T(vn), T(tp), T(tn))
    eye = torch.eye(D, device=dev)
    std, neg = load_gal(STD_CACHE, dev), load_gal(NEG_CACHE, dev)
    cos = dict(acc_2x2=joint_correct_for_w(eye, quads),
               std_r1=retr(eye, *std, dev)[0], std_r5=retr(eye, *std, dev)[1],
               neg_r1=retr(eye, *neg, dev)[0])
    print(f"[cosine] 2x2={cos['acc_2x2']:.2f}  std R@1={cos['std_r1']:.2f}  neg R@1={cos['neg_r1']:.2f}")

    rows = []
    for rank in args.ranks:
        for c in args.penalties:
            W = train(rank, c, img, txt, img_id, dev, epochs=args.epochs, seed=args.seed,
                      param=args.param)
            Wd = W.to(dev)
            sr1, sr5 = retr(Wd, *std, dev)
            nr1, _ = retr(Wd, *neg, dev)
            row = dict(rank=rank, c=c, acc_2x2=joint_correct_for_w(Wd, quads),
                       dW_norm=torch.norm(W - torch.eye(D)).item(),
                       std_r1=sr1, std_r5=sr5, neg_r1=nr1)
            rows.append(row)
            torch.save({"model_name": "bilinear", "rank": rank,
                        "state_dict": {"W": W, "bias": torch.zeros(1)}},
                       os.path.join(args.output_dir, f"idinit_r{rank}_c{c:g}.pt"))
            print(f"  rank={rank:4d} c={c:<6g}  2x2={row['acc_2x2']:6.2f}%  "
                  f"stdR@1={sr1:6.2f}%  negR@1={nr1:6.2f}%  ||W-I||={row['dW_norm']:6.2f}")

    json.dump(dict(cosine=cos, chance=100 / 6, results=rows),
              open(os.path.join(args.output_dir, "final_report.json"), "w"), indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "final_sweep.csv"), index=False)
    print(f"[saved] {args.output_dir}/final_report.json")


if __name__ == "__main__":
    main()
