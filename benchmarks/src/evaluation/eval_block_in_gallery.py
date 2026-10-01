"""
E-A/E-B (IMPLEMENTATION_PLAN.md Part XVI) -- keep the block, grow the gallery.

Every 2x2 number in this repo is a four-cell comparison; every retrieval number is
an open gallery. The two have never been put on one axis, so "the controlled 2x2
advantage does not transfer to retrieval" (8-V.11) and "the diagnosis is a
different task" have been indistinguishable. This script removes the ambiguity by
embedding the controlled block **inside** a natural gallery and sweeping the
gallery size N: the identity of the correct answer never changes with N, only the
number of natural distractors competing with it.

Two directions, because 8.10.2's duality says they are different targets:

  T2I (image gallery)   text fixed, rank images   -> constrained by gamma > |beta|
  I2T (caption gallery) image fixed, rank captions -> constrained by gamma > |alpha|

At N=2 each direction collapses to the published pairwise judgment, which is the
verification gate this script refuses to run past: T2I@N=2 must reproduce the
image-selection accuracy and I2T@N=2 the caption-selection accuracy of the 33-concept
6col coordinate (RESULTS 8.9.2 concept-macro: 6.10 / 3.81, group 0.375).

Recall is computed in **closed form**, not by Monte-Carlo gallery draws. For a query
whose correct image is beaten by b of the M pool distractors, a uniformly drawn
gallery of N-2 distractors leaves it top-1 with probability C(M-b, N-2)/C(M, N-2);
the joint "both polarities correct in the same gallery" uses the union of the two
beating sets. That is exact, seedless, and smooth in N.

The negation-violation rate is reported only on the full gallery, where it is
deterministic: among the top-k images retrieved for a negated query, the fraction
that actually contain the negated object, per COCO val2017 instance annotations.

Usage (from the repo root):
    python -m benchmarks.src.evaluation.eval_block_in_gallery \\
        --restrict_objects logs/evaluation/00_concept_sets/paper33.txt \\
        --output_dir logs/evaluation/01_paper/2026-09-07_block_in_gallery
"""

import os
import ast
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.special import gammaln

try:
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction, DEFAULT_CACHE_DIR,
    )
    from benchmarks.src.analysis.config import set_seed, coerce_bool_column
    from benchmarks.src.analysis.paths import resolve_image_path as resolve_path
    from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, encode_texts_unified, compute_hadamard_coordinates,
    )
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.model_loader import load_clip_for_eval
    from analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction, DEFAULT_CACHE_DIR,
    )
    from analysis.config import set_seed, coerce_bool_column
    from analysis.paths import resolve_image_path as resolve_path
    from evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, encode_texts_unified, compute_hadamard_coordinates,
    )

# Published 6col/33-concept pairwise judgments the N=2 column must reproduce.
# These are the CONCEPT-MACRO estimator (mean over concepts of the per-concept mean),
# which is the one RESULTS 8-R.1's warning box tells later sections to use; the pooled
# figures for the same quantities are 3.91 / 9.06 / 0.88 and must not be mixed in.
# Cross-check: erasing alpha exactly turns the group condition into the image
# condition, and RESULTS line 46 reports that ablation at 6.10% -- the same number.
GATE = {"caption": 3.81, "image": 6.10, "group": 0.375}
GATE_TOL = 0.15  # percentage points

DEFAULT_W = {
    "naive_delta_rank32":
        "logs/evaluation/01_paper/2026-09-02_narrow_rank32_w_checkpoints/delta_warmstart_rank32.pt",
    "labclip_repro":
        "logs/evaluation/01_paper/2026-09-01_labclip_official_recipe/labclip_official_recipe_bilinear.pt",
    "proposed_residual_c0.1":
        "logs/evaluation/01_paper/2026-09-02_residual_identity_infonce/residual_identity_infonce_pen0.1.pt",
}


# ============================================================
# closed-form recall for a randomly drawn gallery
# ============================================================

def _log_falling(n: int, k: int) -> float:
    """log of n!/(n-k)! ; -inf when the draw is impossible."""
    if k < 0 or n < k:
        return -np.inf
    return gammaln(n + 1) - gammaln(n - k + 1)


def survive_prob(n_beating: np.ndarray, pool: int, n_draw: int) -> np.ndarray:
    """P(a uniform sample of n_draw pool items contains none of the n_beating).

    C(pool - b, n_draw) / C(pool, n_draw), evaluated in log space.
    """
    out = np.zeros(len(n_beating), dtype=np.float64)
    denom = _log_falling(pool, n_draw)
    for i, b in enumerate(n_beating):
        out[i] = np.exp(_log_falling(pool - int(b), n_draw) - denom) if pool - b >= n_draw else 0.0
    return out


# ============================================================
# data
# ============================================================

def load_blocks(args, model, preprocess, tokenizer, device, cache_kw):
    """Per concept: (v_pres, v_abs, t_pos, t_neg), built exactly as the 2x2 audit
    builds them so the on-disk feature cache is shared rather than re-derived."""
    df = coerce_bool_column(pd.read_csv(args.csv_path), "object_in_image")
    restriction = load_object_restriction(args.restrict_objects)
    objects = sorted(df["object_name"].unique().tolist())
    if restriction:
        objects = [o for o in objects if o in set(restriction)]

    blocks = []
    for obj in objects:
        d = df[df["object_name"] == obj].reset_index(drop=True)
        d_t = d[d["object_in_image"] == True].reset_index(drop=True)
        d_f = d[d["object_in_image"] == False].reset_index(drop=True)
        n = min(len(d_t), len(d_f))
        if n < args.min_pairs:
            continue

        p_pres = [resolve_path(p, args.image_root) for p in d_t["image_path"].tolist()[:n]]
        p_abs = [resolve_path(p, args.image_root) for p in d_f["image_path"].tolist()[:n]]
        t_pos = d_t["positive_caption"].tolist()[:n]
        t_neg = d_t["negative_caption"].tolist()[:n]

        v_pres, _, m_p = cached_encode(
            lambda: encode_images_unified(model, preprocess, p_pres, device, args.batch_size),
            kind="image_pres@norm+raw+flags", items=p_pres, **cache_kw)
        v_abs, _, m_a = cached_encode(
            lambda: encode_images_unified(model, preprocess, p_abs, device, args.batch_size),
            kind="image_abs@norm+raw+flags", items=p_abs, **cache_kw)
        keep = np.where(m_p & m_a)[0]
        if len(keep) < args.min_pairs:
            continue
        v_pres, v_abs = v_pres[keep], v_abs[keep]
        t_pos = [t_pos[i] for i in keep]
        t_neg = [t_neg[i] for i in keep]

        e_pos, _ = cached_encode(
            lambda: encode_texts_unified(model, tokenizer, t_pos, device, args.batch_size),
            kind="text_pos@norm+raw", items=t_pos, **cache_kw)
        e_neg, _ = cached_encode(
            lambda: encode_texts_unified(model, tokenizer, t_neg, device, args.batch_size),
            kind="text_neg@norm+raw", items=t_neg, **cache_kw)

        blocks.append(dict(concept=obj, v_pres=v_pres, v_abs=v_abs,
                           t_pos=e_pos, t_neg=e_neg, n=len(keep)))
    return blocks


def load_gallery(args, model, preprocess, tokenizer, device, cache_kw):
    """COCO val2017 gallery: 5,000 images with exhaustive category labels, and the
    matching standard / negated caption pools for the I2T direction."""
    df = pd.read_csv(args.standard_csv)
    paths = [resolve_path(p, args.image_root) for p in df["filepath"].tolist()]
    imgs, _, flags = cached_encode(
        lambda: encode_images_unified(model, preprocess, paths, device, args.batch_size),
        kind="gallery_images@standard@l2norm+raw+flags", items=paths, **cache_kw)
    if not np.all(flags):
        raise RuntimeError(f"{(~flags).sum()} gallery images failed to load; fix paths first")

    present = [set(ast.literal_eval(s)) for s in df["positive_objects"].tolist()]

    caps, tii = {}, {}
    for split, csv_path in (("standard", args.standard_csv), ("negated", args.negated_csv)):
        d = pd.read_csv(csv_path)
        lists = [ast.literal_eval(row) for row in d["captions"].tolist()]
        flat = [c for row in lists for c in row]
        tii[split] = [i for i, row in enumerate(lists) for _ in row]
        emb, _ = cached_encode(
            lambda: encode_texts_unified(model, tokenizer, flat, device, args.batch_size),
            kind=f"gallery_captions@{split}@l2norm+raw", items=flat, **cache_kw)
        caps[split] = emb
    return imgs, present, caps, tii


def load_scorers(args, device) -> Dict[str, torch.Tensor]:
    """name -> W (D,D) with S(v, t) = v^T W t. Cosine is the identity."""
    dim = args.embed_dim
    scorers = {"cosine": torch.eye(dim)}
    catalogue = dict(DEFAULT_W)
    for spec in (args.extra_w or []):
        if "=" not in spec:
            raise ValueError(f"--extra_w expects name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        catalogue[name] = path
    for name, path in catalogue.items():
        if not os.path.exists(path):
            print(f"  [skip] {name}: {path} not found")
            continue
        W = torch.load(path, map_location="cpu")["state_dict"]["W"].float()
        if tuple(W.shape) != (dim, dim):
            raise ValueError(f"{name}: expected ({dim},{dim}), got {tuple(W.shape)}")
        scorers[name] = W
    return {k: v.to(device) for k, v in scorers.items()}


# ============================================================
# the sweep
# ============================================================

def run_direction(blocks, scorers, pool_img, pool_txt, args, direction: str):
    """One retrieval direction. Returns (per-concept rows, per-N macro summary)."""
    rows, per_n = [], {}
    pool = pool_img if direction == "t2i" else pool_txt
    M = pool.shape[0]
    n_draws = [n for n in args.gallery_sizes if n - 2 <= M]

    for name, W in scorers.items():
        acc = {n: {"pos": [], "neg": [], "both": []} for n in n_draws}
        for blk in blocks:
            v_p = torch.from_numpy(blk["v_pres"]).float().to(args.device)
            v_a = torch.from_numpy(blk["v_abs"]).float().to(args.device)
            t_p = torch.from_numpy(blk["t_pos"]).float().to(args.device)
            t_n = torch.from_numpy(blk["t_neg"]).float().to(args.device)
            vW_p, vW_a = v_p @ W, v_a @ W          # (n, D)

            if direction == "t2i":
                # query = caption, candidates = images; correct(pos)=pres, correct(neg)=abs
                s_pp = (vW_p * t_p).sum(-1)         # S11
                s_ap = (vW_a * t_p).sum(-1)         # S12
                s_pn = (vW_p * t_n).sum(-1)         # S21
                s_an = (vW_a * t_n).sum(-1)         # S22
                corr_pos, rival_pos = s_pp, s_ap
                corr_neg, rival_neg = s_an, s_pn
                pool_pos = t_p @ W.T @ pool.T       # (n, M)
                pool_neg = t_n @ W.T @ pool.T
            else:
                # query = image, candidates = captions; correct(pres)=pos, correct(abs)=neg
                s_pp = (vW_p * t_p).sum(-1)
                s_ap = (vW_a * t_p).sum(-1)
                s_pn = (vW_p * t_n).sum(-1)
                s_an = (vW_a * t_n).sum(-1)
                corr_pos, rival_pos = s_pp, s_pn    # image present: pos caption must win
                corr_neg, rival_neg = s_an, s_ap    # image absent: neg caption must win
                pool_pos = vW_p @ pool.T
                pool_neg = vW_a @ pool.T

            beat_pos = pool_pos >= corr_pos[:, None]
            beat_neg = pool_neg >= corr_neg[:, None]
            rival_ok_pos = (corr_pos > rival_pos).cpu().numpy()
            rival_ok_neg = (corr_neg > rival_neg).cpu().numpy()
            b_pos = beat_pos.sum(-1).cpu().numpy()
            b_neg = beat_neg.sum(-1).cpu().numpy()
            b_union = (beat_pos | beat_neg).sum(-1).cpu().numpy()

            for n in n_draws:
                k = n - 2
                p_pos = rival_ok_pos * survive_prob(b_pos, M, k)
                p_neg = rival_ok_neg * survive_prob(b_neg, M, k)
                p_both = rival_ok_pos * rival_ok_neg * survive_prob(b_union, M, k)
                acc[n]["pos"].append(float(np.mean(p_pos)) * 100)
                acc[n]["neg"].append(float(np.mean(p_neg)) * 100)
                acc[n]["both"].append(float(np.mean(p_both)) * 100)
                rows.append(dict(direction=direction, scorer=name, concept=blk["concept"],
                                 n_pairs=blk["n"], gallery_size=n,
                                 r1_pos=acc[n]["pos"][-1], r1_neg=acc[n]["neg"][-1],
                                 r1_both=acc[n]["both"][-1]))
        per_n[name] = {n: {k: float(np.mean(v)) for k, v in acc[n].items()} for n in n_draws}
    return rows, per_n


def violation_rate(blocks, scorers, pool_img, present, args):
    """On the full gallery: of the top-k images returned for a negated query, what
    fraction actually contain the negated object. Deterministic -- no sampling."""
    out, base = {}, []
    for blk in blocks:
        has = np.array([blk["concept"] in s for s in present], dtype=bool)
        base.append(100.0 * (has.sum() + 1) / (len(present) + 2))  # +1: the pres block image
    out["_base_rate"] = {"base_rate": float(np.mean(base))}
    for name, W in scorers.items():
        per_concept = {k: [] for k in args.violation_k}
        for blk in blocks:
            has = np.array([blk["concept"] in s for s in present], dtype=bool)
            v_p = torch.from_numpy(blk["v_pres"]).float().to(args.device)
            v_a = torch.from_numpy(blk["v_abs"]).float().to(args.device)
            t_n = torch.from_numpy(blk["t_neg"]).float().to(args.device)
            # gallery = 5,000 natural images + this pair's two block images
            s_pool = t_n @ W.T @ pool_img.T                       # (n, M)
            s_blk = torch.stack([(v_p @ W * t_n).sum(-1),
                                 (v_a @ W * t_n).sum(-1)], dim=1)  # (n, 2): pres, abs
            scores = torch.cat([s_pool, s_blk], dim=1)
            contains = np.concatenate([has, np.array([True, False])])
            contains_t = torch.from_numpy(contains).to(args.device)
            for k in args.violation_k:
                idx = scores.topk(k, dim=1).indices
                frac = contains_t[idx].float().mean(-1).cpu().numpy()
                per_concept[k].append(float(np.mean(frac)) * 100)
        out[name] = {f"violation@{k}": float(np.mean(v)) for k, v in per_concept.items()}
        out[name]["lift@1"] = out[name][f"violation@{args.violation_k[0]}"] / out["_base_rate"]["base_rate"]
    return out


def full_retrieval(scorers, pool_img, caps, tii, args):
    """The classic open-corpus numbers, both directions, so the arms sit in the same
    table as 8-V.11/12. Uses retrieval.py's own recall_at_k, not a reimplementation."""
    try:
        from benchmarks.src.evaluation.retrieval import recall_at_k, batchify
    except ImportError:
        from analysis.import_compat import reraise_unless_standalone
        reraise_unless_standalone()
        from evaluation.retrieval import recall_at_k, batchify

    out = {}
    for name, W in scorers.items():
        out[name] = {}
        for split in ("standard", "negated"):
            T = torch.from_numpy(caps[split]).float().to(args.device)
            scores = T @ W.T @ pool_img.T                      # rows = texts
            pos = torch.zeros_like(scores, dtype=torch.bool)
            pos[torch.arange(len(scores)), torch.tensor(tii[split])] = True
            for k in (1, 5):
                t2i = (batchify(recall_at_k, scores, pos, 1024, args.device, k=k) > 0).float().mean()
                i2t = (batchify(recall_at_k, scores.T, pos.T, 1024, args.device, k=k) > 0).float().mean()
                out[name][f"{split}_t2i_R@{k}"] = float(t2i) * 100
                out[name][f"{split}_i2t_R@{k}"] = float(i2t) * 100
    return out


def coefficient_table(blocks, scorers, args):
    """alpha/beta/gamma per scorer -- the coordinate the gate numbers live in."""
    out = {}
    for name, W in scorers.items():
        per = []
        for blk in blocks:
            v_p = torch.from_numpy(blk["v_pres"]).float().to(args.device)
            v_a = torch.from_numpy(blk["v_abs"]).float().to(args.device)
            t_p = torch.from_numpy(blk["t_pos"]).float().to(args.device)
            t_n = torch.from_numpy(blk["t_neg"]).float().to(args.device)
            S11 = ((v_p @ W) * t_p).sum(-1).cpu().numpy()
            S12 = ((v_a @ W) * t_p).sum(-1).cpu().numpy()
            S21 = ((v_p @ W) * t_n).sum(-1).cpu().numpy()
            S22 = ((v_a @ W) * t_n).sum(-1).cpu().numpy()
            h = compute_hadamard_coordinates(S11, S12, S21, S22)
            per.append(dict(concept=blk["concept"],
                            alpha=float(np.mean(np.abs(h["alpha"]))),
                            beta=float(np.mean(np.abs(h["beta"]))),
                            gamma=float(np.mean(h["gamma"])),
                            caption=float(np.mean(h["gamma"] > np.abs(h["alpha"]))) * 100,
                            image=float(np.mean(h["gamma"] > np.abs(h["beta"]))) * 100,
                            group=float(np.mean(h["gamma"] > np.maximum(
                                np.abs(h["alpha"]), np.abs(h["beta"])))) * 100))
        out[name] = {k: float(np.mean([p[k] for p in per]))
                     for k in ("alpha", "beta", "gamma", "caption", "image", "group")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv_path", default="benchmarks/data/images/beaf_counterfactual_6col.csv")
    ap.add_argument("--standard_csv", default="benchmarks/data/images/COCO_val_retrieval.csv")
    ap.add_argument("--negated_csv",
                    default="benchmarks/data/images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv")
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--restrict_objects", default="logs/evaluation/00_concept_sets/paper33.txt")
    ap.add_argument("--min_pairs", type=int, default=20)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--gallery_sizes", type=int, nargs="+",
                    default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 5000])
    ap.add_argument("--violation_k", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--extra_w", nargs="+", default=None,
                    help="additional scorers as name=path/to/checkpoint.pt")
    ap.add_argument("--full_retrieval", action="store_true",
                    help="also score the open 5,000-image corpus (8-V.11/12 metric)")
    ap.add_argument("--skip_gate", action="store_true",
                    help="run even if the N=2 column misses the published values")
    ap.add_argument("--output_dir",
                    default="logs/evaluation/01_paper/2026-09-07_block_in_gallery")
    args = ap.parse_args()

    set_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  E-A/E-B -- keep the block, grow the gallery")
    print("=" * 70)

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, args.device)
    cache_kw = dict(model=args.model, pretrained=args.pretrained,
                    enabled=args.use_cache, cache_dir=args.cache_dir)

    print("\n[1/5] building controlled blocks...")
    blocks = load_blocks(args, model, preprocess, tokenizer, args.device, cache_kw)
    print(f"  {len(blocks)} concepts, {sum(b['n'] for b in blocks)} pairs")

    print("[2/5] loading natural gallery...")
    imgs, present, caps, tii = load_gallery(args, model, preprocess, tokenizer, args.device, cache_kw)
    pool_img = torch.from_numpy(imgs).float().to(args.device)
    pool_txt = torch.from_numpy(np.concatenate([caps["standard"], caps["negated"]])).float().to(args.device)
    print(f"  {pool_img.shape[0]} images, {pool_txt.shape[0]} captions "
          f"(standard + negated)")

    scorers = load_scorers(args, args.device)
    print(f"  scorers: {', '.join(scorers)}")

    print("[3/5] coefficient gate (33-concept 6col coordinate)...")
    coefs = coefficient_table(blocks, scorers, args)
    c = coefs["cosine"]
    print(f"  cosine  caption={c['caption']:.2f}  image={c['image']:.2f}  group={c['group']:.2f}"
          f"   (published macro {GATE['caption']} / {GATE['image']} / {GATE['group']})")
    off = {k: abs(c[k] - GATE[k]) for k in GATE}
    if max(off.values()) > GATE_TOL and not args.skip_gate:
        raise SystemExit(
            f"GATE FAILED: cosine pairwise judgments differ from the published 6col values "
            f"by {off} (tolerance {GATE_TOL}pp). The block construction or the embeddings "
            f"are not the ones those numbers came from -- fix that before reading anything "
            f"downstream, or pass --skip_gate if you know why they differ.")
    print("  gate passed")

    print("[4/5] sweeping gallery size, both directions...")
    all_rows, summary = [], {}
    for direction in ("t2i", "i2t"):
        rows, per_n = run_direction(blocks, scorers, pool_img, pool_txt, args, direction)
        all_rows += rows
        summary[direction] = per_n
        head = per_n["cosine"]
        n2 = head[2]
        print(f"  {direction}: cosine N=2 pos={n2['pos']:.2f} neg={n2['neg']:.2f} both={n2['both']:.2f}"
              f"   N=5000 both={head[max(head)]['both']:.3f}")

    print("[5/5] negation-violation rate on the full gallery...")
    viol = violation_rate(blocks, scorers, pool_img, present, args)
    print(f"  null: {viol['_base_rate']['base_rate']:.2f}% of the gallery contains the "
          f"negated concept (macro over concepts)")
    for name, v in viol.items():
        if name == "_base_rate":
            continue
        print(f"  {name:24s} " + "  ".join(f"{k}={x:.1f}%" for k, x in v.items() if k != "lift@1")
              + f"   lift={v['lift@1']:.1f}x")

    full = {}
    if args.full_retrieval:
        print("[+] open-corpus retrieval (5,000 images x 25,014 captions)...")
        full = full_retrieval(scorers, pool_img, caps, tii, args)
        for name, v in full.items():
            print(f"  {name:24s} std T2I R@1={v['standard_t2i_R@1']:5.2f}  "
                  f"neg T2I R@1={v['negated_t2i_R@1']:5.2f}  "
                  f"std I2T R@1={v['standard_i2t_R@1']:5.2f}  "
                  f"neg I2T R@1={v['negated_i2t_R@1']:5.2f}")

    pd.DataFrame(all_rows).to_csv(
        os.path.join(args.output_dir, "block_in_gallery_per_concept.csv"), index=False)
    out = dict(coefficients=coefs, sweep=summary, violation=viol, full_retrieval=full,
               gate=dict(published=GATE, observed={k: c[k] for k in GATE}),
               config=dict(concepts=len(blocks), pairs=sum(b["n"] for b in blocks),
                           gallery_images=int(pool_img.shape[0]),
                           gallery_captions=int(pool_txt.shape[0]),
                           gallery_sizes=args.gallery_sizes),
               provenance=build_provenance(args, concepts=len(blocks)))
    with open(os.path.join(args.output_dir, "block_in_gallery_summary.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved: {args.output_dir}/")


if __name__ == "__main__":
    main()
