"""
Part XVII stage 2 -- does more supervision, or a sharper contrast, grow the interaction?

Stage 1 removed both main effects for free and moved neither the MCQ negation subset nor
the controlled 2x2. What is left is the interaction term, and the earlier factorial named
two candidate levers for it: how much verified negation supervision the map sees (S6), and
how sharp the contrast is when it sees it (S5, where the correct/twin gap measured 0.09
standard deviations of the candidate spread at temperature 0.07).

This sweeps both on the one structure known to preserve retrieval, W = I + dW with a
Frobenius penalty, so that any movement is attributable to the two knobs rather than to
capacity:

    --data              which verified CSVs feed the map
    --contrast_size N   the softmax denominator is cut to N candidates (plus the positive)
                        instead of the whole batch -- the narrow loss is the N=1 limit and
                        the published broad recipe is N=batch
    --temperature T     0.07 is CLIP's own; lowering it concentrates the same scores

Each run reports what the project actually wants -- MCQ and retrieval together -- rather
than the diagnostic alone, and the standard split is reported beside the negated one so a
gain that is really a geometry loss cannot hide.

Usage (from the repo root):
    python -m benchmarks.src.evaluation.train_scale_and_contrast \\
        --data beaf coco_nat both --output_dir logs/evaluation/01_paper/2026-09-08_scale_contrast
"""

import os
import ast
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
    from benchmarks.src.analysis.feature_cache import (
        cached_encode, build_provenance, DEFAULT_CACHE_DIR,
    )
    from benchmarks.src.analysis.config import set_seed, coerce_bool_column
    from benchmarks.src.analysis.paths import resolve_image_path as resolve_path
    from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, encode_texts_unified,
    )
    from benchmarks.src.evaluation.scoring_heads import predict_with_tie_report
    from benchmarks.src.evaluation.retrieval import recall_at_k, batchify
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.model_loader import load_clip_for_eval
    from analysis.feature_cache import cached_encode, build_provenance, DEFAULT_CACHE_DIR
    from analysis.config import set_seed, coerce_bool_column
    from analysis.paths import resolve_image_path as resolve_path
    from evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, encode_texts_unified,
    )
    from evaluation.scoring_heads import predict_with_tie_report
    from evaluation.retrieval import recall_at_k, batchify

DATASETS = {
    "beaf": "benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv",
    "coco_nat": "benchmarks/data/images/coco_natural_negation.csv",
}
COSINE = {"standard_t2i_R@1": 30.36, "negated_t2i_R@1": 24.97, "coco_mcq_avg": 39.30}


# ============================================================
# data
# ============================================================

def build_items(csv_paths: List[str], args, model, preprocess, tokenizer, device, cache_kw):
    """(image, true caption, swapped caption) triples from the six-column schema."""
    frames = [coerce_bool_column(pd.read_csv(p), "object_in_image") for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    df["true_caption"] = np.where(df["object_in_image"],
                                  df["positive_caption"], df["negative_caption"])
    df["swap_caption"] = np.where(df["object_in_image"],
                                  df["negative_caption"], df["positive_caption"])
    df = df.drop_duplicates(subset=["image_path", "true_caption"]).reset_index(drop=True)

    # Image-diversity axis: hold the item count fixed and vary how many distinct
    # photographs those items come from. Without the item cap the two would move
    # together and a slope could not be attributed to diversity.
    if args.max_images or args.max_items:
        rng = np.random.default_rng(args.seed)
        if args.max_images:
            keep = rng.permutation(df["image_path"].unique())[:args.max_images]
            df = df[df["image_path"].isin(set(keep))].reset_index(drop=True)
        if args.max_items and len(df) > args.max_items:
            per = int(np.ceil(args.max_items / df["image_path"].nunique()))
            df = (df.groupby("image_path", group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), per), random_state=args.seed)))
            df = df.sample(min(len(df), args.max_items), random_state=args.seed).reset_index(drop=True)

    paths = sorted(df["image_path"].unique().tolist())
    real = [resolve_path(p, args.image_root) for p in paths]
    caps = sorted(set(df["true_caption"]) | set(df["swap_caption"]))
    img, _, flags = cached_encode(
        lambda: encode_images_unified(model, preprocess, real, device, args.batch_size),
        kind="scalecontrast_images@l2norm+raw+flags", items=real, **cache_kw)
    txt, _ = cached_encode(
        lambda: encode_texts_unified(model, tokenizer, caps, device, args.batch_size),
        kind="scalecontrast_captions@l2norm+raw", items=caps, **cache_kw)
    ok = {p for p, f in zip(paths, flags) if f}
    df = df[df["image_path"].isin(ok)].reset_index(drop=True)

    ii = {p: i for i, p in enumerate(paths)}
    ti = {c: i for i, c in enumerate(caps)}
    return (torch.from_numpy(img).float(), torch.from_numpy(txt).float(),
            np.array([ii[p] for p in df["image_path"]]),
            np.array([ti[c] for c in df["true_caption"]]),
            np.array([ti[c] for c in df["swap_caption"]]),
            dict(rows=len(df), images=len(ok), captions=len(caps)))


# ============================================================
# training
# ============================================================

def contrastive_loss(v, t, t_sw, scale, contrast_size: int, generator):
    """Symmetric InfoNCE whose denominator is cut to `contrast_size` competitors.

    Each row keeps its own positive, its own polarity-swapped caption, and
    ``contrast_size - 1`` other rows drawn at random. A column subset cannot express
    this -- every row's positive is a different column, so keeping all positives keeps
    the whole batch -- which is why the candidates are gathered per row instead.
    """
    b = v.shape[0]
    labels = torch.arange(b, device=v.device)
    if contrast_size <= 0 or contrast_size >= b:
        cand_t = torch.cat([t, t_sw], 0)
        return (F.cross_entropy(scale * v @ cand_t.t(), labels)
                + F.cross_entropy(scale * t @ v.t(), labels)) / 2.0

    n = max(2, contrast_size)
    # offsets in [1, b-1] keep the drawn columns distinct from the row's own positive
    off = torch.randint(1, b, (b, n - 1), generator=generator, device=v.device)
    cols = torch.cat([labels[:, None], (labels[:, None] + off) % b], dim=1)   # (b, n)

    s_t = scale * v @ t.t()                                    # (b, b)
    s_sw = scale * (v * t_sw).sum(-1, keepdim=True)            # (b, 1) own swap
    logits_i2t = torch.cat([s_t.gather(1, cols), s_sw], dim=1)
    s_v = scale * t @ v.t()
    logits_t2i = s_v.gather(1, cols)
    tgt = torch.zeros(b, dtype=torch.long, device=v.device)    # positive is column 0
    return (F.cross_entropy(logits_i2t, tgt) + F.cross_entropy(logits_t2i, tgt)) / 2.0


def train_w(data, args, device, tag: str):
    img, txt, vi, ti, si, _ = data
    set_seed(args.seed)
    dim = img.shape[1]
    dW = nn.Parameter(torch.zeros(dim, dim, device=device))
    eye = torch.eye(dim, device=device)
    opt = torch.optim.Adam([dW], lr=args.lr)
    V, T = img.to(device), txt.to(device)
    VI = torch.from_numpy(vi).long().to(device)
    TI = torch.from_numpy(ti).long().to(device)
    SI = torch.from_numpy(si).long().to(device)
    n = len(vi)
    bs = min(args.batch_size_train, n)
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)
    scale = 1.0 / args.temperature

    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        losses = []
        for start in range(0, n, bs):
            sel = perm[start:start + bs]
            if sel.numel() < 4:
                continue
            W = eye + dW
            v = V[VI[sel]]
            t = F.normalize(T[TI[sel]] @ W.T, dim=-1)
            t_sw = F.normalize(T[SI[sel]] @ W.T, dim=-1)
            loss = contrastive_loss(v, t, t_sw, scale, args.contrast_size, gen) \
                + args.penalty_coef * torch.sum(dW ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"    [{tag}] epoch {epoch:3d} loss={np.mean(losses):.4f} "
                  f"||dW||={torch.norm(dW).item():.3f}")
    return (eye + dW).detach().cpu(), float(torch.norm(dW).item())


# ============================================================
# evaluation -- the two goals, together
# ============================================================

def eval_w(W, ev, device) -> Dict[str, float]:
    Wd = W.to(device)
    out = {}
    for split in ("standard", "negated"):
        V, T, tii = ev[split]
        scores = T.to(device) @ Wd.T @ V.to(device).t()
        pos = torch.zeros_like(scores, dtype=torch.bool)
        pos[torch.arange(len(scores)), torch.tensor(tii)] = True
        out[f"{split}_t2i_R@1"] = float(
            (batchify(recall_at_k, scores, pos, 1024, device, k=1) > 0).float().mean()) * 100
        del scores, pos
        torch.cuda.empty_cache()
    for name, (V, T, tgt, qt) in ev["mcq"].items():
        s = torch.einsum("nd,nkd->nk", V.to(device) @ Wd, T.to(device))
        pred, _ = predict_with_tie_report(s.cpu(), seed=42)
        corr = pred == tgt
        out[f"{name}_mcq_avg"] = float(corr.mean()) * 100
        for q in ("positive", "negative", "hybrid"):
            sel = np.array([x == q for x in qt])
            if sel.any():
                out[f"{name}_mcq_{q}"] = float(corr[sel].mean()) * 100
    return out


def load_eval(args, model, preprocess, tokenizer, device, cache_kw):
    ev = {}
    for split, csv_path in (("standard", args.standard_csv), ("negated", args.negated_csv)):
        d = pd.read_csv(csv_path)
        paths = [resolve_path(p, args.image_root) for p in d["filepath"].tolist()]
        lists = [ast.literal_eval(r) for r in d["captions"].tolist()]
        flat = [c for row in lists for c in row]
        tii = [i for i, row in enumerate(lists) for _ in row]
        imgs, _, _ = cached_encode(
            lambda: encode_images_unified(model, preprocess, paths, device, args.batch_size),
            kind=f"gallery_images@{split}@l2norm+raw+flags", items=paths, **cache_kw)
        txts, _ = cached_encode(
            lambda: encode_texts_unified(model, tokenizer, flat, device, args.batch_size),
            kind=f"gallery_captions@{split}@l2norm+raw", items=flat, **cache_kw)
        ev[split] = (torch.from_numpy(imgs).float(), torch.from_numpy(txts).float(), tii)
    ev["mcq"] = {}
    for spec in args.mcq:
        name, path = spec.split("=", 1)
        d = torch.load(path, map_location="cpu")
        ev["mcq"][name] = (d["img_embeds"].float(), d["text_embeds"].float(),
                           d["targets"].numpy(), list(d["question_types"]))
    return ev


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=["beaf", "coco_nat", "both"])
    ap.add_argument("--contrast_sizes", type=int, nargs="+", default=[0])
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.07])
    ap.add_argument("--standard_csv", default="benchmarks/data/images/COCO_val_retrieval.csv")
    ap.add_argument("--negated_csv",
                    default="benchmarks/data/images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv")
    ap.add_argument("--mcq", nargs="+", default=[
        "coco=logs/evaluation/cached_embeddings/COCO_val_mcq_llama3.1_rephrased_embeds.pt",
        "voc=logs/evaluation/cached_embeddings/VOC2007_mcq_llama3.1_rephrased_embeds.pt"])
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--batch_size_train", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--penalty_coef", type=float, default=0.1)
    ap.add_argument("--contrast_size", type=int, default=0)
    ap.add_argument("--max_images", type=int, default=0,
                    help="cap on distinct photographs (0 = all)")
    ap.add_argument("--max_items", type=int, default=0,
                    help="cap on training items, applied after --max_images")
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--output_dir",
                    default="logs/evaluation/01_paper/2026-09-08_scale_contrast")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    print("\n" + "=" * 78)
    print("  Stage 2 -- supervision scale and contrast sharpness")
    print("=" * 78)

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    cache_kw = dict(model=args.model, pretrained=args.pretrained,
                    enabled=args.use_cache, cache_dir=args.cache_dir)
    ev = load_eval(args, model, preprocess, tokenizer, device, cache_kw)

    base = eval_w(torch.eye(512), ev, device)
    print("\n  cosine baseline: " + "  ".join(f"{k}={base[k]:.2f}" for k in
          ("standard_t2i_R@1", "negated_t2i_R@1", "coco_mcq_avg", "coco_mcq_negative")))
    off = {k: abs(base[k] - v) for k, v in COSINE.items()}
    if max(off.values()) > 0.15:
        raise SystemExit(f"GATE FAILED: cosine baseline off by {off}")
    print("  gate passed")

    cache: Dict[str, Tuple] = {}
    rows = []
    for name in args.data:
        paths = list(DATASETS.values()) if name == "both" else [DATASETS[name]]
        if name not in cache:
            cache[name] = build_items(paths, args, model, preprocess, tokenizer, device, cache_kw)
        stats = cache[name][5]
        for cs in args.contrast_sizes:
            for temp in args.temperatures:
                args.contrast_size, args.temperature = cs, temp
                tag = f"{name}|N={cs or 'batch'}|T={temp}"
                print(f"\n  {tag}  ({stats['rows']} rows, {stats['images']} images)")
                W, drift = train_w(cache[name], args, device, tag)
                m = eval_w(W, ev, device)
                m.update(data=name, contrast_size=cs, temperature=temp, drift=drift, **stats)
                rows.append(m)
                torch.save({"model_name": "bilinear", "rank": None,
                            "state_dict": {"W": W, "bias": torch.zeros(1)}},
                           os.path.join(args.output_dir,
                                        f"W_{name}_N{cs or 'batch'}_T{temp}.pt"))
                print(f"    std R@1={m['standard_t2i_R@1']:5.2f}  neg R@1={m['negated_t2i_R@1']:5.2f}  "
                      f"COCO MCQ={m['coco_mcq_avg']:5.2f} (neg {m['coco_mcq_negative']:5.2f})  "
                      f"||dW||={drift:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, "scale_contrast.csv"), index=False)
    with open(os.path.join(args.output_dir, "scale_contrast.json"), "w") as f:
        json.dump(dict(baseline=base, runs=rows, provenance=build_provenance(args)),
                  f, indent=2, default=str)
    print(f"\nsaved: {args.output_dir}/")


if __name__ == "__main__":
    main()
