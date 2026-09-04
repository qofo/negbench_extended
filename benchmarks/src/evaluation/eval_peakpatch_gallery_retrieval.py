"""
Does the beta -> retrieval-loss law survive PeakPatch? A family-boundary test.

RESULTS 8-R fits, over nine models, a law relating the signed image main effect to how
much T2I recall is lost when the query carries a negation: beta(signed) -> R@1 drop,
r = +0.835. RESULTS 8-N.8 fits the mirror law on the text side: alpha(signed) -> the MCQ
positive-minus-negative gap, r = +0.970. Both were fit on nine models that share one
property -- their score is the cosine of two embeddings -- and neither states that as a
condition.

PeakPatch breaks the text-side law badly. Its ECN+SCN alpha is -66.4e-3, more negative
than CoN-CLIP's -27.3e-3, so the nine-model line predicts a gap of -126.5%p; the observed
gap is +40.8%p, a residual of +167%p against a nine-model residual SD of 6.9%p. The
proposed explanation is that the law is about embedding geometry and the SCN is not:
it adds a bounded correction (max 0.2) straight to the score, on the scale of the whole
cosine similarity (~0.22), so the "alpha" measured for ECN+SCN is largely the correction
term itself rather than a property of any embedding.

If that explanation is right it makes a split prediction about the image side, which this
script tests:

  ecn      replaces the text embedding, so the score stays an inner product and the
           condition stays inside the family -> the beta law should HOLD
  ecn+scn  scores through the SCN, which is not an inner product -> OUTSIDE the family
           -> the beta law should BREAK, in the same direction and for the same reason

Confirming both halves turns "PeakPatch is an outlier" into "the laws hold on the family
of cosine-scored dual encoders, and here is the boundary", which is what the manuscript
should claim. Falsifying it means the family story is wrong and the outlier needs a
different explanation -- report either outcome as it comes.

Retrieval reuses retrieval.py's own recall_at_k/batchify and the same two COCO CSVs as
the nine-model sweep (RESULTS 8-R) and the delta-loss gallery run (RESULTS 8-V.11), so the
cosine row must reproduce 30.36 / 24.97 exactly; it is the harness check.

Usage (from the repo root):
    python -m benchmarks.src.evaluation.eval_peakpatch_gallery_retrieval \\
        --peakpatch_root PeakPatch \\
        --output_dir logs/evaluation/01_paper/2026-09-04_peakpatch_gallery_retrieval
"""
import os
import ast
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from benchmarks.src.evaluation.eval_peakpatch_2x2_audit import (
    load_peakpatch, encode_texts_peakpatch, assert_encode_text_consistency,
)
from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import encode_images_unified
from benchmarks.src.evaluation.retrieval import recall_at_k, batchify
from benchmarks.src.analysis.model_loader import load_clip_for_eval
from benchmarks.src.analysis.feature_cache import cached_encode, build_provenance
from benchmarks.src.analysis.paths import resolve_image_path as resolve_path
from benchmarks.src.analysis.config import set_seed

CONDITIONS = ("cosine", "ecn", "ecn+scn")


def load_gallery(csv_path, image_root):
    """Same loader as eval_delta_loss_gallery_retrieval, so the populations match."""
    df = pd.read_csv(csv_path)
    paths = [resolve_path(p, image_root) for p in df["filepath"].tolist()]
    captions, texts_image_index = [], []
    for i, c in enumerate(df["captions"]):
        caps = ast.literal_eval(c)
        captions.extend(caps)
        texts_image_index.extend([i] * len(caps))
    return paths, captions, texts_image_index


def compute_recalls(scores: torch.Tensor, texts_image_index, batch_size: int = 250):
    positive_pairs = torch.zeros_like(scores, dtype=torch.bool)
    positive_pairs[torch.arange(len(scores)), torch.tensor(texts_image_index)] = True
    out = {}
    for k in (1, 5):
        out[k] = 100.0 * (
            batchify(recall_at_k, scores, positive_pairs, batch_size, "cpu", k=k) > 0
        ).float().mean().item()
    return out


def scn_score_matrix(pp, sc_feats: Dict[int, torch.Tensor], ecn_emb: np.ndarray,
                     V: torch.Tensor, device: str, text_chunk: int = 2048,
                     img_batch: int = 128) -> torch.Tensor:
    """
    (n_texts, n_images) SCN-corrected scores, chunked over texts.

    score_pairwise precomputes an (n_img, n_text) similarity per selected layer, which at
    5,000 x 25,014 would be ~1GB per layer on the GPU; chunking the text axis keeps every
    intermediate small. The anchor-layer slot is overwritten with the ECN output exactly
    as scripts/eval_mcq.py and score_block_with_scn do, so this is the same scorer the
    2x2 audit measured -- with one difference worth naming: score_pairwise engages the
    confidence gate when the checkpoint sets it, while score_mcq does not. The released
    SCN has confidence_gate off (asserted below), so the two paths coincide here.
    """
    sc = pp["sc"]
    anchor = max(pp["sc_layers"])
    if getattr(sc, "confidence_gate", False):
        raise SystemExit(
            "released SCN has confidence_gate=True: score_pairwise would take a different "
            "path than the score_mcq used for the 2x2 audit, so the two are not comparable.")

    n_text = ecn_emb.shape[0]
    out = torch.empty(n_text, V.shape[0], dtype=torch.float32)
    with torch.no_grad():
        for s in range(0, n_text, text_chunk):
            e = min(s + text_chunk, n_text)
            layers = {l: sc_feats[l][s:e].float().to(device) for l in pp["sc_layers"]}
            layers[anchor] = torch.from_numpy(ecn_emb[s:e]).float().to(device)
            block = sc.score_pairwise(layers, V, batch_size=img_batch)  # (n_img, chunk)
            out[s:e] = block.T.cpu()
            del layers, block
            torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--peakpatch_root", default="PeakPatch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--image_root", default="benchmarks/data/images")
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--cache_dir", default="logs/evaluation/cached_embeddings/feature_cache")
    ap.add_argument("--standard_csv", default="benchmarks/data/images/COCO_val_retrieval.csv")
    ap.add_argument("--negated_csv",
                    default="benchmarks/data/images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_kw = dict(model=args.model, pretrained=str(args.pretrained),
                    cache_dir=args.cache_dir, enabled=args.use_cache)

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    pp = load_peakpatch(args.peakpatch_root, device)
    assert_encode_text_consistency(model, tokenizer, device)

    results: Dict[str, Dict[str, Dict[int, float]]] = {}
    for split, csv_path in [("standard", args.standard_csv), ("negated", args.negated_csv)]:
        paths, captions, texts_image_index = load_gallery(csv_path, args.image_root)
        print(f"\n[{split}] {len(paths)} gallery images, {len(captions)} queries")

        images_emb, _, img_flags = cached_encode(
            lambda: encode_images_unified(model, preprocess, paths, device, args.batch_size),
            kind=f"gallery_images@{split}@l2norm+raw+flags", items=paths, **cache_kw)
        if not np.all(img_flags):
            raise RuntimeError(f"{(~img_flags).sum()} gallery images failed to load for {split}; "
                               "dropping them silently would misalign image indices.")
        V = torch.from_numpy(images_emb).float().to(device)

        # Not routed through cached_encode: that cache stores a tuple of ndarrays with
        # allow_pickle=False, and this returns a {layer: tensor} dict alongside the two
        # embeddings. One text-tower walk over the queries is cheap enough to just do.
        print("  encoding queries through the text tower (plain / ECN / SCN layers)...")
        plain, ecn, sc_feats = encode_texts_peakpatch(
            model, tokenizer, captions, pp, device, args.batch_size)

        scores = {
            "cosine": (torch.from_numpy(plain).float().to(device) @ V.T).cpu(),
            "ecn": (torch.from_numpy(ecn).float().to(device) @ V.T).cpu(),
        }
        print("  scoring ecn+scn through score_pairwise...")
        scores["ecn+scn"] = scn_score_matrix(pp, sc_feats, ecn, V, device)

        results[split] = {}
        for name in CONDITIONS:
            r = compute_recalls(scores[name], texts_image_index)
            results[split][name] = r
            print(f"    {name:10s} R@1={r[1]:6.2f}%  R@5={r[5]:6.2f}%")
        del scores, V
        torch.cuda.empty_cache()

    print("\n=== R@1 / R@5 drop (standard - negated) ===")
    drops = {}
    for name in CONDITIONS:
        drops[name] = {k: results["standard"][name][k] - results["negated"][name][k]
                       for k in (1, 5)}
        print(f"  {name:10s} R@1 {drops[name][1]:+6.2f}pp    R@5 {drops[name][5]:+6.2f}pp")

    cos_r1 = results["standard"]["cosine"][1]
    print(f"\n[harness check] cosine standard R@1 = {cos_r1:.2f}% "
          f"(RESULTS 8-R / 8-V.11 report 30.36%)")
    if abs(cos_r1 - 30.36) > 0.05:
        print("  WARNING: cosine baseline does not reproduce; the scoring path differs "
              "from the nine-model sweep and these numbers are not comparable to it.")

    summary = dict(
        results={s: {n: {f"R@{k}": v for k, v in r.items()} for n, r in d.items()}
                 for s, d in results.items()},
        drops={n: {f"R@{k}": v for k, v in d.items()} for n, d in drops.items()},
        provenance=build_provenance(args),
    )
    path = os.path.join(args.output_dir, "peakpatch_gallery_retrieval.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
