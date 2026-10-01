"""
S1 (IMPLEMENTATION_PLAN.md Part XVII) -- suppress the main effects at query time.

The identity s(u,v) = m + alpha*u + beta*v + gamma*uv names two query-independent
nuisance terms. beta = d_I . m_T is an image that attracts every query -- which is
exactly what the retrieval literature calls a hub -- and alpha = m_I . d_T is its
mirror on the caption side. Removing them needs no training at all: subtract, from
every score, the mean that term contributes over a bank.

The point of running this is not that centering is new; it is that the coordinate
system makes a **structural** prediction about where each correction can possibly act.

    ranking images for a fixed caption (T2I, retrieval)   -> caption term is a row
                                                             constant, so only image
                                                             centering can change it
    ranking captions for a fixed image (I2T, MCQ)         -> image term is a row
                                                             constant, so only caption
                                                             centering can change it

Two of the four cells are therefore *exactly* zero, not just small, and the script
asserts that rather than reporting it. What is left to measure is how much each
correction buys on its own target, and in particular whether it buys more on negated
queries than on standard ones -- which is what the coefficient reading predicts,
since negation is precisely where the interaction term is weakest relative to the
main effects.

Banks. Image centering needs a bank of captions and caption centering a bank of
images. Both are label-free but transductive, so the default reports the honest
version: the bank for a negated split is the *standard* caption set, never the
queries being scored.

Usage (from the repo root):
    python -m benchmarks.src.evaluation.eval_main_effect_calibration \\
        --output_dir logs/evaluation/01_paper/2026-09-08_main_effect_calibration
"""

import os
import ast
import json
import argparse
from typing import Dict, List, Tuple

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
    from benchmarks.src.evaluation.scoring_heads import predict_with_tie_report
    from benchmarks.src.evaluation.retrieval import recall_at_k, batchify
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
    from evaluation.scoring_heads import predict_with_tie_report
    from evaluation.retrieval import recall_at_k, batchify

# Published cosine baselines the run must reproduce before anything is read.
GATE = {
    "standard_t2i_R@1": 30.36, "negated_t2i_R@1": 24.97,
    "coco_mcq_avg": 39.30, "coco_mcq_pos": 69.14, "coco_mcq_neg": 6.84,
}
GATE_TOL = 0.15


# ============================================================
# corrections
# ============================================================

def center_images(scores: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """Remove each image's query-independent attractiveness (the beta term).

    ``scores`` is (n_query, n_image); ``bank`` is (n_bank, n_image) built from a
    caption bank disjoint from the queries. Subtracting a per-column mean cannot
    change any per-image ranking of captions, only the per-caption ranking of images.
    """
    return scores - bank.mean(dim=0, keepdim=True)


def center_captions(scores: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """Mirror of :func:`center_images`: remove each caption's generic attractiveness.

    ``scores`` is (n_query_image, n_caption); ``bank`` is (n_bank_image, n_caption).
    """
    return scores - bank.mean(dim=0, keepdim=True)


def inverted_softmax(scores: torch.Tensor, bank: torch.Tensor, temp: float) -> torch.Tensor:
    """Normalize each image column by how strongly the bank competes for it."""
    return scores - temp * torch.logsumexp(bank / temp, dim=0, keepdim=True)


# ============================================================
# metrics
# ============================================================

def recalls(scores: torch.Tensor, texts_image_index: List[int], device: str) -> Dict[str, float]:
    pos = torch.zeros_like(scores, dtype=torch.bool)
    pos[torch.arange(len(scores)), torch.tensor(texts_image_index)] = True
    out = {}
    for k in (1, 5):
        t2i = (batchify(recall_at_k, scores, pos, 1024, device, k=k) > 0).float().mean()
        i2t = (batchify(recall_at_k, scores.t(), pos.t(), 1024, device, k=k) > 0).float().mean()
        out[f"t2i_R@{k}"] = float(t2i) * 100
        out[f"i2t_R@{k}"] = float(i2t) * 100
    return out


def mcq_accuracy(scores: torch.Tensor, targets: np.ndarray,
                 qtypes: List[str], seed: int) -> Dict[str, float]:
    """Per-question-type accuracy, breaking exact ties at random.

    Every NegBench MCQ row carries ``correct_answer = 0``, so a plain argmax scores an
    uninformative scorer at 100%; this uses the repo's tie-aware predictor instead.
    """
    pred, tied = predict_with_tie_report(scores.cpu(), seed=seed)
    correct = (pred == targets)
    out = {"avg": float(correct.mean()) * 100, "tie_rate": float(np.mean(tied)) * 100}
    for qt in sorted(set(qtypes)):
        sel = np.array([q == qt for q in qtypes])
        out[qt] = float(correct[sel].mean()) * 100
    return out


# ============================================================
# data
# ============================================================

def load_retrieval(args, model, preprocess, tokenizer, device, cache_kw):
    out = {}
    for split, csv_path in (("standard", args.standard_csv), ("negated", args.negated_csv)):
        d = pd.read_csv(csv_path)
        paths = [resolve_path(p, args.image_root) for p in d["filepath"].tolist()]
        lists = [ast.literal_eval(r) for r in d["captions"].tolist()]
        flat = [c for row in lists for c in row]
        tii = [i for i, row in enumerate(lists) for _ in row]
        imgs, _, flags = cached_encode(
            lambda: encode_images_unified(model, preprocess, paths, device, args.batch_size),
            kind=f"gallery_images@{split}@l2norm+raw+flags", items=paths, **cache_kw)
        if not np.all(flags):
            raise RuntimeError(f"{(~flags).sum()} gallery images failed to load for {split}")
        txts, _ = cached_encode(
            lambda: encode_texts_unified(model, tokenizer, flat, device, args.batch_size),
            kind=f"gallery_captions@{split}@l2norm+raw", items=flat, **cache_kw)
        out[split] = dict(images=torch.from_numpy(imgs).float(),
                          texts=torch.from_numpy(txts).float(), tii=tii)
    return out


def load_mcq(name: str, path: str) -> Dict:
    d = torch.load(path, map_location="cpu")
    return dict(images=d["img_embeds"].float(), texts=d["text_embeds"].float(),
                targets=d["targets"].numpy(), qtypes=list(d["question_types"]), name=name)


# ============================================================
# run
# ============================================================

def run_retrieval(data, args) -> Dict:
    dev = args.device
    res = {}
    bank_texts = data["standard"]["texts"].to(dev)      # bank is always the standard split
    for split in ("standard", "negated"):
        V = data[split]["images"].to(dev)
        T = data[split]["texts"].to(dev)
        tii = data[split]["tii"]
        base = T @ V.t()
        bank = bank_texts @ V.t()
        variants = {
            "cosine": base,
            "image_centered": center_images(base, bank),
            "inverted_softmax": inverted_softmax(base, bank, args.temp),
            "caption_centered": base - base.mean(dim=1, keepdim=True),
        }
        res[split] = {k: recalls(v, tii, dev) for k, v in variants.items()}
        del base, bank, variants
        torch.cuda.empty_cache()
    return res


def violation_rates(data, args) -> Dict:
    """Is the gain negation-specific, or just a generic hubness gain?

    Every negated query names, in the CSV it was generated from, the objects it
    asserts are absent. A retrieved image violates the query when it actually
    contains one of them. The null is the share of the gallery that does.
    """
    dev = args.device
    d = pd.read_csv(args.negated_csv)
    present = [set(ast.literal_eval(s)) for s in d["positive_objects"].tolist()]
    negated = [set(ast.literal_eval(s)) for s in d["negative_objects"].tolist()]
    lists = [ast.literal_eval(r) for r in d["captions"].tolist()]
    q_neg = [negated[i] for i, row in enumerate(lists) for _ in row]

    V = data["negated"]["images"].to(dev)
    T = data["negated"]["texts"].to(dev)
    bank = data["standard"]["texts"].to(dev) @ V.t()
    base = T @ V.t()
    variants = {
        "cosine": base,
        "image_centered": center_images(base, bank),
        "inverted_softmax": inverted_softmax(base, bank, args.temp),
    }
    contains = np.zeros((len(q_neg), len(present)), dtype=bool)
    out = {}
    null = float(np.mean([len(s & q) > 0 for q in q_neg for s in present[:1]])) if False else None
    for name, sc in variants.items():
        rates = {}
        for k in (1, 5):
            idx = sc.topk(k, dim=1).indices.cpu().numpy()
            hit = np.array([[len(present[j] & q) > 0 for j in row] for row, q in zip(idx, q_neg)])
            rates[f"violation@{k}"] = float(hit.mean()) * 100
        out[name] = rates
    # null: probability a uniformly drawn gallery image violates the query
    per_q = [float(np.mean([len(p & q) > 0 for p in present])) for q in q_neg[:args.null_sample]]
    out["_null"] = {"violation@random": float(np.mean(per_q)) * 100}
    return out


def run_mcq(mcq: Dict, args) -> Dict:
    dev = args.device
    V = mcq["images"].to(dev)                       # (N, D)
    T = mcq["texts"].to(dev)                        # (N, 4, D)
    base = torch.einsum("nd,nkd->nk", V, T)         # each item scores its own 4 options
    # Caption centering needs each option's mean score over a bank of images. The bank
    # is this benchmark's own image set, which is label-free but transductive.
    bank = V[torch.randperm(V.shape[0], generator=torch.Generator().manual_seed(args.seed))
             [:args.bank_size]]                     # (B, D)
    ref = torch.einsum("bd,nkd->nkb", bank, T).mean(dim=-1)   # (N, 4)
    variants = {
        "cosine": base,
        "caption_centered": base - ref,
        "image_centered": base - base.mean(dim=1, keepdim=True),
    }
    return {k: mcq_accuracy(v, mcq["targets"], mcq["qtypes"], args.seed)
            for k, v in variants.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standard_csv", default="benchmarks/data/images/COCO_val_retrieval.csv")
    ap.add_argument("--negated_csv",
                    default="benchmarks/data/images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv")
    ap.add_argument("--mcq", nargs="+", default=[
        "coco=logs/evaluation/cached_embeddings/COCO_val_mcq_llama3.1_rephrased_embeds.pt",
        "voc2007=logs/evaluation/cached_embeddings/VOC2007_mcq_llama3.1_rephrased_embeds.pt",
    ])
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--bank_size", type=int, default=2000)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--null_sample", type=int, default=500,
                    help="queries sampled when estimating the violation null")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--skip_gate", action="store_true")
    ap.add_argument("--output_dir",
                    default="logs/evaluation/01_paper/2026-09-08_main_effect_calibration")
    args = ap.parse_args()

    set_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print("  S1 -- query-time main-effect suppression")
    print("=" * 72)

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, args.device)
    cache_kw = dict(model=args.model, pretrained=args.pretrained,
                    enabled=args.use_cache, cache_dir=args.cache_dir)

    print("\n[1/3] retrieval...")
    data = load_retrieval(args, model, preprocess, tokenizer, args.device, cache_kw)
    ret = run_retrieval(data, args)

    print("[2/3] negation-violation rate under each correction...")
    viol = violation_rates(data, args)
    for name, v in viol.items():
        print(f"  {name:18s} " + "  ".join(f"{k}={x:.2f}%" for k, x in v.items()))

    print("[2/3] MCQ...")
    mcqs = {}
    for spec in args.mcq:
        name, path = spec.split("=", 1)
        mcqs[name] = run_mcq(load_mcq(name, path), args)

    print("[3/3] gates and identities...")
    obs = {
        "standard_t2i_R@1": ret["standard"]["cosine"]["t2i_R@1"],
        "negated_t2i_R@1": ret["negated"]["cosine"]["t2i_R@1"],
        "coco_mcq_avg": mcqs["coco"]["cosine"]["avg"],
        "coco_mcq_pos": mcqs["coco"]["cosine"].get("positive", float("nan")),
        "coco_mcq_neg": mcqs["coco"]["cosine"].get("negative", float("nan")),
    }
    off = {k: abs(obs[k] - v) for k, v in GATE.items()}
    print("  baseline vs published: " + "  ".join(f"{k}={obs[k]:.2f}({v})" for k, v in GATE.items()))
    if max(off.values()) > GATE_TOL and not args.skip_gate:
        raise SystemExit(f"GATE FAILED: {off} exceed {GATE_TOL}pp -- fix the pipeline first")
    print("  gate passed")

    # The two null cells are identities, not findings: assert them.
    ident = {
        "caption_centering_leaves_t2i": max(
            abs(ret[s]["caption_centered"][f"t2i_R@{k}"] - ret[s]["cosine"][f"t2i_R@{k}"])
            for s in ret for k in (1, 5)),
        "image_centering_leaves_mcq": max(
            abs(m["image_centered"]["avg"] - m["cosine"]["avg"]) for m in mcqs.values()),
    }
    print(f"  identity checks (must be ~0): {ident}")

    print("\n=== retrieval (R@1 / R@5) ===")
    for split in ("standard", "negated"):
        for name, v in ret[split].items():
            print(f"  {split:9s} {name:18s} T2I {v['t2i_R@1']:5.2f}/{v['t2i_R@5']:5.2f}   "
                  f"I2T {v['i2t_R@1']:5.2f}/{v['i2t_R@5']:5.2f}")
    print("\n=== MCQ ===")
    for bench, r in mcqs.items():
        for name, v in r.items():
            extra = "  ".join(f"{k}={v[k]:.2f}" for k in sorted(v) if k not in ("avg", "tie_rate"))
            print(f"  {bench:8s} {name:18s} avg={v['avg']:5.2f}  {extra}  (tie {v['tie_rate']:.1f}%)")

    out = dict(retrieval=ret, mcq=mcqs, violation=viol, gate=dict(published=GATE, observed=obs),
               identities=ident, config=vars(args),
               provenance=build_provenance(args))
    with open(os.path.join(args.output_dir, "main_effect_calibration.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved: {args.output_dir}/")


if __name__ == "__main__":
    main()
