"""
S2 (IMPLEMENTATION_PLAN.md Part XVII) -- the training-free logic wrapper, at its ceiling.

Factored-inference methods answer a negated query by scoring atomic concepts with the
frozen encoder and executing the constraint outside the similarity: keep the positive
evidence, subtract the negated evidence. Whether that is enough is the question this
project has to answer before proposing anything of its own, because if a wrapper with
no parameters already reaches the target, a trained correction has to earn its keep
somewhere else.

This runs that family at an upper bound rather than as a reproduction. The negated
retrieval CSV records the objects each query asserts present and absent -- the very
atoms a parser would have to recover -- so reading them from the CSV removes parsing
error entirely. **A wrapper that cannot win with oracle atoms cannot win with parsed
ones**, which makes a negative result here informative and a positive one an
optimistic bound rather than a claim about any published system.

    S'(v, q) = S(v, q) + lam_pos * mean_{c in P} S(v, a_c) - lam_neg * mean_{c in N} S(v, a_c)

with a_c the embedding of "a photo of a {c}.". lam_neg = 0 recovers the baseline, so the
sweep contains its own control.

Usage (from the repo root):
    python -m benchmarks.src.evaluation.eval_atomic_score_editing \\
        --output_dir logs/evaluation/01_paper/2026-09-08_atomic_score_editing
"""

import os
import ast
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

try:
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.feature_cache import (
        cached_encode, build_provenance, DEFAULT_CACHE_DIR,
    )
    from benchmarks.src.analysis.config import set_seed
    from benchmarks.src.analysis.paths import resolve_image_path as resolve_path
    from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, encode_texts_unified,
    )
    from benchmarks.src.evaluation.retrieval import recall_at_k, batchify
    from benchmarks.src.evaluation.eval_main_effect_calibration import center_images
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.model_loader import load_clip_for_eval
    from analysis.feature_cache import cached_encode, build_provenance, DEFAULT_CACHE_DIR
    from analysis.config import set_seed
    from analysis.paths import resolve_image_path as resolve_path
    from evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, encode_texts_unified,
    )
    from evaluation.retrieval import recall_at_k, batchify
    from evaluation.eval_main_effect_calibration import center_images


def load_split(args, csv_path, split, model, preprocess, tokenizer, device, cache_kw):
    d = pd.read_csv(csv_path)
    paths = [resolve_path(p, args.image_root) for p in d["filepath"].tolist()]
    lists = [ast.literal_eval(r) for r in d["captions"].tolist()]
    flat = [c for row in lists for c in row]
    tii = [i for i, row in enumerate(lists) for _ in row]
    imgs, _, flags = cached_encode(
        lambda: encode_images_unified(model, preprocess, paths, device, args.batch_size),
        kind=f"gallery_images@{split}@l2norm+raw+flags", items=paths, **cache_kw)
    txts, _ = cached_encode(
        lambda: encode_texts_unified(model, tokenizer, flat, device, args.batch_size),
        kind=f"gallery_captions@{split}@l2norm+raw", items=flat, **cache_kw)
    pos = [set(ast.literal_eval(s)) for s in d["positive_objects"].tolist()]
    neg = [set(ast.literal_eval(s)) for s in d["negative_objects"].tolist()]
    return dict(images=torch.from_numpy(imgs).float(), texts=torch.from_numpy(txts).float(),
                tii=tii, q_pos=[pos[i] for i, r in enumerate(lists) for _ in r],
                q_neg=[neg[i] for i, r in enumerate(lists) for _ in r],
                img_present=pos)


def atom_matrix(concepts: List[str], model, tokenizer, args, device, cache_kw):
    prompts = [args.template.format(c) for c in concepts]
    emb, _ = cached_encode(
        lambda: encode_texts_unified(model, tokenizer, prompts, device, args.batch_size),
        kind="atomic_concept_prompts@l2norm+raw", items=prompts, **cache_kw)
    return torch.from_numpy(emb).float()


def membership(sets: List[set], concepts: List[str]) -> torch.Tensor:
    """(n_query, n_concept) row-normalized indicator, so the edit is a mean not a sum."""
    idx = {c: i for i, c in enumerate(concepts)}
    M = torch.zeros(len(sets), len(concepts))
    for r, s in enumerate(sets):
        hit = [idx[c] for c in s if c in idx]
        if hit:
            M[r, hit] = 1.0 / len(hit)
    return M


def recalls(scores, tii, device):
    pos = torch.zeros_like(scores, dtype=torch.bool)
    pos[torch.arange(len(scores)), torch.tensor(tii)] = True
    out = {}
    for k in (1, 5):
        out[f"t2i_R@{k}"] = float(
            (batchify(recall_at_k, scores, pos, 1024, device, k=k) > 0).float().mean()) * 100
    return out


def violation(scores, q_neg, img_present, k=1):
    idx = scores.topk(k, dim=1).indices.cpu().numpy()
    hit = np.array([[len(img_present[j] & q) > 0 for j in row] for row, q in zip(idx, q_neg)])
    return float(hit.mean()) * 100


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standard_csv", default="benchmarks/data/images/COCO_val_retrieval.csv")
    ap.add_argument("--negated_csv",
                    default="benchmarks/data/images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv")
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--template", default="a photo of a {}.")
    ap.add_argument("--lam_neg", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--lam_pos", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--output_dir",
                    default="logs/evaluation/01_paper/2026-09-08_atomic_score_editing")
    args = ap.parse_args()

    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print("  S2 -- atomic score editing with oracle atoms (upper bound)")
    print("=" * 72)
    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, dev)
    cache_kw = dict(model=args.model, pretrained=args.pretrained,
                    enabled=args.use_cache, cache_dir=args.cache_dir)

    neg = load_split(args, args.negated_csv, "negated", model, preprocess, tokenizer, dev, cache_kw)
    std = load_split(args, args.standard_csv, "standard", model, preprocess, tokenizer, dev, cache_kw)
    concepts = sorted({c for s in neg["q_pos"] + neg["q_neg"] for c in s})
    print(f"  {len(concepts)} atomic concepts, {len(neg['tii'])} negated queries")

    A = atom_matrix(concepts, model, tokenizer, args, dev, cache_kw).to(dev)   # (C, D)
    V = neg["images"].to(dev)
    T = neg["texts"].to(dev)
    base = T @ V.t()
    atom_scores = A @ V.t()                                                    # (C, n_img)
    Mpos = membership(neg["q_pos"], concepts).to(dev)
    Mneg = membership(neg["q_neg"], concepts).to(dev)
    pos_term = Mpos @ atom_scores
    neg_term = Mneg @ atom_scores
    bank = std["texts"].to(dev) @ V.t()

    rows = []
    for lp in args.lam_pos:
        for ln in args.lam_neg:
            s = base + lp * pos_term - ln * neg_term
            r = recalls(s, neg["tii"], dev)
            r.update(lam_pos=lp, lam_neg=ln, centered=False,
                     violation1=violation(s, neg["q_neg"], neg["img_present"]))
            rows.append(r)
            sc = center_images(s, bank)
            rc = recalls(sc, neg["tii"], dev)
            rc.update(lam_pos=lp, lam_neg=ln, centered=True,
                      violation1=violation(sc, neg["q_neg"], neg["img_present"]))
            rows.append(rc)

    df = pd.DataFrame(rows)
    print("\n=== negated T2I (oracle atoms) ===")
    print(f"{'lam_pos':>7} {'lam_neg':>7} {'center':>7} {'R@1':>7} {'R@5':>7} {'viol@1':>7}")
    for _, r in df.iterrows():
        print(f"{r.lam_pos:7.2f} {r.lam_neg:7.2f} {str(r.centered):>7} "
              f"{r['t2i_R@1']:7.2f} {r['t2i_R@5']:7.2f} {r.violation1:7.2f}")

    df.to_csv(os.path.join(args.output_dir, "atomic_score_editing.csv"), index=False)
    with open(os.path.join(args.output_dir, "atomic_score_editing.json"), "w") as f:
        json.dump({"rows": rows, "concepts": concepts, "config": vars(args),
                   "provenance": build_provenance(args)}, f, indent=2, default=str)
    print(f"\nsaved: {args.output_dir}/")


if __name__ == "__main__":
    main()
