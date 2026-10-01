"""
A pair-conditional rerank head, and whether it escapes the fixed-transform ceiling.

RESULTS 8.14.4 closes the fixed-transform route: the main effects that decide the 2x2
are scene-specific, so estimating them at corpus or concept level buys nothing (4.03 ->
5.60 -> 7.06 against a 74.40 oracle), and neither a bilinear projection nor an affine
centering moves the block. Both families share the same limitation -- with `t` fixed the
score is affine in `v`, so nothing in them can subtract a quantity that depends on the
particular scene.

This script drops that constraint. The score becomes

    f(v, t) = v . t + h([v * t ; v ; t]),        h an MLP with a zero-initialized head

so at initialization `f` is *exactly* cosine and every departure is something training
paid for. `v * t` is the elementwise product, whose sum is the cosine itself; a linear
map on it is a diagonal bilinear form, and the nonlinearity is what buys the escape from
the fixed-transform family.

Two objectives, because the point is to move all three axes at once:

  - a hinge on the 2x2 margin Delta = min(S++,S--) - max(S+-,S-+), the same criterion the
    rest of the repo scores;
  - a preservation term penalizing h^2 on COCO gallery pairs (matched and random alike),
    which is the anisotropic replacement for the isotropic ||W - I||_F of 8-V.12: it
    constrains the correction where retrieval actually lives instead of everywhere.

Retrieval is scored by reranking the cosine top-K, the only deployable way to use a head
that is not a dot product. That bounds R@1 by cosine's R@K, which is reported alongside
so the ceiling is visible. The 2x2 is scored out-of-fold over concepts (GroupKFold), the
external benchmarks with a final head trained on all 42 concepts -- the convention of
RESULTS 8-V.12 (6).

Usage:
    python -m benchmarks.src.evaluation.train_conditional_rerank_head \\
        --output_dir logs/evaluation/01_paper/2026-09-03_conditional_rerank_head
"""
import os
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

try:
    from benchmarks.src.evaluation.eval_single_w_generalization import load_quads
    from benchmarks.src.evaluation.scoring_heads import predict_with_tie_report
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.config import set_seed
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from evaluation.eval_single_w_generalization import load_quads
    from evaluation.scoring_heads import predict_with_tie_report
    from analysis.model_loader import load_clip_for_eval
    from analysis.config import set_seed

CACHE = "logs/evaluation/cached_embeddings"
MCQ_SETS = [
    ("COCO", f"{CACHE}/COCO_val_mcq_llama3.1_rephrased_embeds.pt"),
    ("VOC2007", f"{CACHE}/VOC2007_mcq_llama3.1_rephrased_embeds.pt"),
    ("CheXpert", f"{CACHE}/chexpert_binary_mcq_control_valid_only_embeds.pt"),
]


class ConditionalHead(nn.Module):
    """f(v,t) = v.t + h(v,t); h is zero at init so the head starts as plain cosine."""

    def __init__(self, dim: int = 512, hidden: int = 512, features: str = "prod_vt"):
        super().__init__()
        self.features = features
        in_dim = {"prod": dim, "prod_vt": 3 * dim}[features]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def correction(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = v * t
        if self.features == "prod_vt":
            z = torch.cat([z, v, t], dim=-1)
        return self.net(z).squeeze(-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (v * t).sum(-1) + self.correction(v, t)


def block_scores(head: ConditionalHead, v_p, v_a, t_p, t_n):
    """The four cells, in the (image state, text state) convention of E2."""
    return (head(v_p, t_p), head(v_a, t_p), head(v_p, t_n), head(v_a, t_n))


def margin(s11, s12, s21, s22):
    return torch.minimum(s11, s22) - torch.maximum(s12, s21)


def train_head(quads, idx, gallery, args, device) -> ConditionalHead:
    v_p, v_a, t_p, t_n = (x[idx] for x in quads)
    # g_pair maps each caption to its own image: the gallery has ~5 captions per image,
    # so a caption index is not an image index and pairing them by position would be wrong.
    g_img, g_txt, g_pair = gallery
    head = ConditionalHead(v_p.shape[1], args.hidden, args.features).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_txt, n_img = len(g_txt), len(g_img)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(len(v_p), generator=gen)
        for s in range(0, len(perm), args.batch_size):
            b = perm[s:s + args.batch_size].to(device)
            s11, s12, s21, s22 = block_scores(head, v_p[b], v_a[b], t_p[b], t_n[b])
            loss = torch.relu(args.margin - margin(s11, s12, s21, s22)).mean()

            # Preservation: the correction must vanish on gallery pairs, matched and not.
            j = torch.randint(0, n_txt, (args.preserve_batch,), generator=gen).to(device)
            k = torch.randint(0, n_img, (args.preserve_batch,), generator=gen).to(device)
            keep = head.correction(g_img[g_pair[j]], g_txt[j]).pow(2).mean() \
                + head.correction(g_img[k], g_txt[j]).pow(2).mean()
            loss = loss + args.preserve_coef * keep

            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


@torch.no_grad()
def block_accuracy(head, quads, idx) -> np.ndarray:
    v_p, v_a, t_p, t_n = (x[idx] for x in quads)
    s = block_scores(head, v_p, v_a, t_p, t_n)
    return (margin(*s) > 0).cpu().numpy()


@torch.no_grad()
def mcq_accuracy(head, pack, seed) -> Dict[str, float]:
    img, txt, tgt, qtypes = pack
    n, k, _ = txt.shape
    v = img.unsqueeze(1).expand(-1, k, -1).reshape(n * k, -1)
    scores = head(v, txt.reshape(n * k, -1)).reshape(n, k)
    pred, tie = predict_with_tie_report(scores, seed=seed)
    correct = pred == tgt.numpy()
    out = {"avg": float(correct.mean() * 100), "tie_rate": float(tie.mean() * 100)}
    q = np.asarray(qtypes)
    for name in ("positive", "negative"):
        m = q == name
        out[name] = float(correct[m].mean() * 100) if m.any() else float("nan")
    return out


@torch.no_grad()
def rerank_recall(head, img, txt, idx, topk, chunk=256) -> Dict[str, float]:
    """Rerank the cosine top-K per query. R@1 is bounded by cosine R@K, reported too."""
    hits1, hits5, ceiling = [], [], []
    for s in range(0, len(txt), chunk):
        t = txt[s:s + chunk]
        cos = t @ img.T
        cand = torch.topk(cos, k=topk, dim=1).indices                 # (B, K)
        gold = idx[s:s + chunk].to(cand.device)
        ceiling.append((cand == gold.unsqueeze(1)).any(1).float().cpu())
        b, k = cand.shape
        v = img[cand.reshape(-1)]
        tt = t.unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1)
        sc = head(v, tt).reshape(b, k)
        order = torch.argsort(sc, dim=1, descending=True)
        ranked = torch.gather(cand, 1, order)
        hits1.append((ranked[:, 0] == gold).float().cpu())
        hits5.append((ranked[:, :5] == gold.unsqueeze(1)).any(1).float().cpu())
    return {"r1": float(torch.cat(hits1).mean() * 100),
            "r5": float(torch.cat(hits5).mean() * 100),
            "topk_ceiling": float(torch.cat(ceiling).mean() * 100)}


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    def load_gallery(path):
        c = torch.load(path, map_location="cpu")
        return (c["images_emb"].float().to(device), c["texts_emb"].float().to(device),
                torch.tensor(c["texts_image_index"], dtype=torch.long))

    std_img, std_txt, std_idx = load_gallery(args.std_cache)
    neg_img, neg_txt, neg_idx = load_gallery(args.neg_cache)
    mcq = {}
    for name, path in MCQ_SETS:
        c = torch.load(path, map_location="cpu")
        mcq[name] = (c["img_embeds"].float().to(device), c["text_embeds"].float().to(device),
                     c["targets"], c["question_types"])

    # Optional base transform. RESULTS 8.14.4 found that projecting the image side against
    # the top text directions -- the beta term of the decomposition, i.e. hubness -- lifts
    # retrieval for free (30.36 -> 32.25 standard, 24.97 -> 28.24 negated) while doing
    # nothing for the 2x2. It pulls a different lever than the head, so composing them is
    # the natural test of whether the two gains are additive.
    P_T = None
    if args.base == "beta_proj":
        _, _, Vh = torch.linalg.svd(std_txt.double(), full_matrices=False)
        U = Vh[:args.base_rank].T.contiguous()
        P_T = (torch.eye(U.shape[0], dtype=torch.float64, device=device)
               - U @ U.T).float()
        std_img = std_img @ P_T.T
        neg_img = neg_img @ P_T.T
        mcq = {k: (v[0] @ P_T.T, v[1], v[2], v[3]) for k, v in mcq.items()}
        print(f"  base: beta projection, rank {args.base_rank}")

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    v_pos, v_neg, t_pos, t_neg, groups = load_quads(
        args, model, preprocess, tokenizer, device,
        dict(model=args.model, pretrained=str(args.pretrained),
             cache_dir=args.cache_dir, enabled=args.use_cache))
    T = lambda a: torch.from_numpy(np.asarray(a)).float().to(device)
    quads = (T(v_pos), T(v_neg), T(t_pos), T(t_neg))
    if P_T is not None:
        quads = (quads[0] @ P_T.T, quads[1] @ P_T.T, quads[2], quads[3])
    groups = np.asarray(groups)
    gallery = (std_img, std_txt, std_idx.to(device))
    print(f"\n  {len(groups)} pairs / {len(np.unique(groups))} concepts   "
          f"features={args.features}  preserve_coef={args.preserve_coef}\n")

    # --- out-of-fold over concepts: does it generalize to concepts never trained on? ---
    oof = np.zeros(len(groups), dtype=bool)
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=args.folds).split(
            np.arange(len(groups)), groups=groups)):
        head = train_head(quads, tr, gallery, args, device)
        oof[te] = block_accuracy(head, quads, te)
        print(f"  fold {fold}: held-out concepts {len(np.unique(groups[te])):2d}  "
              f"2x2 {oof[te].mean() * 100:6.2f}%")
    macro = float(np.mean([oof[groups == c].mean() * 100 for c in np.unique(groups)]))
    print(f"\n  2x2 OOF pooled {oof.mean() * 100:.2f}%   macro {macro:.2f}%   "
          f"(cosine 4.03% / 4.20%)")

    # --- final head on all concepts, for the external axes (8-V.12 convention) ---
    head = train_head(quads, np.arange(len(groups)), gallery, args, device)
    in_sample = block_accuracy(head, quads, np.arange(len(groups))).mean() * 100

    # How big is the correction where it is wanted vs where it must not be? The two
    # numbers say whether a weak result is the objective failing or the constraint biting.
    with torch.no_grad():
        h_block = head.correction(quads[0], quads[2]).abs().mean().item()
        gj = torch.arange(0, len(std_txt), 7, device=device)
        h_gal = head.correction(std_img[std_idx.to(device)[gj]], std_txt[gj]).abs().mean().item()
    print(f"\n  |h| block {h_block:.5f}   |h| gallery {h_gal:.5f}   "
          f"(cosine scale ~0.25, gamma scale ~0.0012)")

    result = {
        "features": args.features, "base": args.base, "base_rank": args.base_rank,
        "preserve_coef": args.preserve_coef,
        "epochs": args.epochs, "lr": args.lr, "hidden": args.hidden,
        "margin": args.margin, "topk": args.topk, "seed": args.seed,
        "delta_s_oof_pooled": float(oof.mean() * 100),
        "delta_s_oof_macro": macro,
        "delta_s_in_sample": float(in_sample),
        "h_abs_block": float(h_block), "h_abs_gallery": float(h_gal),
    }
    for name, pack in mcq.items():
        a = mcq_accuracy(head, pack, args.seed)
        result[f"{name}_avg"], result[f"{name}_pos"] = a["avg"], a["positive"]
        result[f"{name}_neg"], result[f"{name}_tie"] = a["negative"], a["tie_rate"]
    std = rerank_recall(head, std_img, std_txt, std_idx, args.topk)
    neg = rerank_recall(head, neg_img, neg_txt, neg_idx, args.topk)
    result.update(std_r1=std["r1"], std_r5=std["r5"], std_ceiling=std["topk_ceiling"],
                  neg_r1=neg["r1"], neg_r5=neg["r5"], neg_ceiling=neg["topk_ceiling"])

    print(f"\n  2x2 in-sample {in_sample:6.2f}%")
    for name in mcq:
        print(f"  {name + ' MCQ':<14} avg {result[f'{name}_avg']:6.2f}  "
              f"pos {result[f'{name}_pos']:6.2f}  neg {result[f'{name}_neg']:6.2f}")
    print(f"  retrieval std  R@1 {std['r1']:6.2f}  R@5 {std['r5']:6.2f}   "
          f"(top-{args.topk} ceiling {std['topk_ceiling']:.2f}, cosine R@1 30.36)")
    print(f"  retrieval neg  R@1 {neg['r1']:6.2f}  R@5 {neg['r5']:6.2f}   "
          f"(top-{args.topk} ceiling {neg['topk_ceiling']:.2f}, cosine R@1 24.97)")

    torch.save({"state_dict": head.state_dict(), "config": result},
               os.path.join(args.output_dir, "conditional_head.pt"))
    with open(os.path.join(args.output_dir, "conditional_head.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    pd.DataFrame([result]).to_csv(
        os.path.join(args.output_dir, "conditional_head.csv"), index=False)
    print(f"\n  Saved: {args.output_dir}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=["prod", "prod_vt"], default="prod_vt")
    ap.add_argument("--base", choices=["cosine", "beta_proj"], default="cosine",
                    help="score the head sits on top of; beta_proj is RESULTS 8.14.4's "
                         "free retrieval gain")
    ap.add_argument("--base_rank", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--preserve_coef", type=float, default=1.0)
    ap.add_argument("--preserve_batch", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--std_cache", default=f"{CACHE}/COCO_val_retrieval_retrieval_embeds.pt")
    ap.add_argument("--neg_cache",
                    default=f"{CACHE}/COCO_val_negated_retrieval_llama3.1_rephrased"
                            "_affneg_true_retrieval_embeds.pt")
    ap.add_argument("--csv_path",
                    default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--image_root", default="benchmarks/data/images")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--min_pairs", type=int, default=20)
    ap.add_argument("--restrict_objects", default=None)
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--cache_dir", default=f"{CACHE}/feature_cache")
    ap.add_argument("--batch_size_encode", type=int, default=128, dest="batch_size_encode")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir",
                    default="logs/evaluation/01_paper/2026-09-03_conditional_rerank_head")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
