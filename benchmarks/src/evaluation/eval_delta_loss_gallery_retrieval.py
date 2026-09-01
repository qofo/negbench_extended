"""Does the delta-loss-trained W actually help real gallery retrieval?

Nearly every review6.md block flagged this as the single most important missing
piece: all of the rank-32 results so far (29.19% margin4, 32.58%/32.94% delta) are
2x2 block accuracy on 42 AB-swap concepts, not retrieval against an open gallery.
This trains a FINAL rank-32 W on all 42 concepts (no held-out fold -- the question
here is transfer to a different task and a different, much larger image/caption
population, not held-out-concept generalization within the same task) for both
losses, then scores the existing COCO 5,000-image T2I retrieval gallery
(standard + LLM-rephrased negated captions) three ways: cosine (v.t, the existing
baseline), margin4 W, delta+warmstart W. Recall@1/@5 reuse retrieval.py's own
recall_at_k/batchify -- the exact function this repo's nine-model retrieval sweep
already uses -- so the numbers are directly comparable to PAPER.md's existing
T2I table.

The score convention: S(v,t) = v^T W t, v=image embedding, t=text embedding
(matches train_matcher's model(v_pos, t_pos) call order). For a batched score
matrix with rows=texts, columns=images (retrieval.py's convention,
scores[i,j] = text_i . image_j for cosine), the W-scored equivalent is
scores = texts_emb @ W.T @ images_emb.T.

Usage (from the repo root):
    python -m benchmarks.src.evaluation.eval_delta_loss_gallery_retrieval \\
        --output_dir logs/evaluation/01_paper/2026-09-01_delta_loss_gallery_retrieval
"""
import os
import ast
import json
import argparse

import numpy as np
import pandas as pd
import torch

from benchmarks.src.evaluation.eval_single_w_generalization import (
    build_family, train_matcher, load_quads, get_embed_dim, fit_global_direction,
)
from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import (
    encode_images_unified, encode_texts_unified,
)
from benchmarks.src.evaluation.retrieval import recall_at_k, batchify
from benchmarks.src.analysis.model_loader import load_clip_for_eval
from benchmarks.src.analysis.feature_cache import cached_encode
from benchmarks.src.analysis.paths import resolve_image_path as resolve_path
from benchmarks.src.analysis.config import set_seed


def train_final_w(args, model, preprocess, tokenizer, device, cache_kw, loss_kind, warmstart):
    """Train one rank-32 W on ALL 42 concepts, no held-out fold -- for deployment
    against a genuinely external gallery/query set, not concept generalization."""
    embed_dim = get_embed_dim(model)
    v_pos, v_neg, t_pos, t_neg, groups = load_quads(args, model, preprocess, tokenizer, device, cache_kw)
    T = lambda a: torch.from_numpy(a).float().to(device)
    quads_all = (T(v_pos), T(v_neg), T(t_pos), T(t_neg))

    warm_start = None
    if warmstart:
        warm_start = (fit_global_direction(v_pos, v_neg, args.seed),
                      fit_global_direction(t_pos, t_neg, args.seed))
    set_seed(args.seed)
    net, _ = build_family("lowrank_32", embed_dim, args.seed, warm_start=warm_start)
    net = net.to(device)
    net = train_matcher(net, quads_all, epochs=args.epochs, lr=args.lr,
                        weight_decay=args.weight_decay, margin=args.margin, loss_kind=loss_kind)
    with torch.no_grad():
        W = (net.proj_v.weight.T @ net.proj_t.weight).cpu()  # (D, D), S(v,t) = v^T W t
    return W


def load_gallery(csv_path, image_root):
    df = pd.read_csv(csv_path)
    paths = [resolve_path(p, image_root) for p in df["filepath"].tolist()]
    all_captions, texts_image_index = [], []
    for i, c in enumerate(df["captions"]):
        caps = ast.literal_eval(c)
        all_captions.extend(caps)
        texts_image_index.extend([i] * len(caps))
    return paths, all_captions, texts_image_index


def compute_recalls(scores: torch.Tensor, texts_image_index, n_images: int, device: str,
                    batch_size: int = 250):
    positive_pairs = torch.zeros_like(scores, dtype=torch.bool)
    positive_pairs[torch.arange(len(scores)), torch.tensor(texts_image_index)] = True
    out = {}
    for k in (1, 5):
        out[f"image_retrieval_recall@{k}"] = (
            batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=k) > 0
        ).float().mean().item()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--min_pairs", type=int, default=20)
    ap.add_argument("--csv_path", default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--image_root", default="benchmarks/data/images")
    ap.add_argument("--restrict_objects", default=None)
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--cache_dir", default="logs/evaluation/cached_embeddings/feature_cache")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=0.1)
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

    print("[1/3] training final rank-32 W (all 42 concepts, no held-out fold)...")
    W_margin4 = train_final_w(args, model, preprocess, tokenizer, device, cache_kw,
                              loss_kind="margin4", warmstart=False).to(device)
    W_delta = train_final_w(args, model, preprocess, tokenizer, device, cache_kw,
                            loss_kind="delta", warmstart=True).to(device)
    print(f"  ||W_margin4||_F = {torch.linalg.norm(W_margin4).item():.2f}   "
          f"||W_delta||_F = {torch.linalg.norm(W_delta).item():.2f}")

    print("[2/3] encoding COCO gallery + standard/negated captions...")
    results = {}
    for split_name, csv_path in [("standard", args.standard_csv), ("negated", args.negated_csv)]:
        paths, captions, texts_image_index = load_gallery(csv_path, args.image_root)
        images_emb, _, img_flags = cached_encode(
            lambda: encode_images_unified(model, preprocess, paths, device, args.batch_size),
            kind=f"gallery_images@{split_name}@l2norm+raw+flags", items=paths, **cache_kw)
        texts_emb, _ = cached_encode(
            lambda: encode_texts_unified(model, tokenizer, captions, device, args.batch_size),
            kind=f"gallery_captions@{split_name}@l2norm+raw", items=captions, **cache_kw)
        if not np.all(img_flags):
            raise RuntimeError(f"{(~img_flags).sum()} gallery images failed to load for {split_name}; "
                               "fix paths before scoring (silently dropping would misalign image indices).")
        V = torch.from_numpy(images_emb).float().to(device)
        Ttext = torch.from_numpy(texts_emb).float().to(device)

        print(f"  [3/3] scoring {split_name}: {len(paths)} images, {len(captions)} captions...")
        scorers = {
            "cosine": Ttext @ V.T,
            "margin4_rank32": Ttext @ W_margin4.T @ V.T,
            "delta_warmstart_rank32": Ttext @ W_delta.T @ V.T,
        }
        results[split_name] = {}
        for name, scores in scorers.items():
            r = compute_recalls(scores.cpu(), texts_image_index, len(paths), "cpu")
            results[split_name][name] = {k: 100.0 * v for k, v in r.items()}
            print(f"    {name:24s} R@1={results[split_name][name]['image_retrieval_recall@1']:.2f}%  "
                  f"R@5={results[split_name][name]['image_retrieval_recall@5']:.2f}%")

    summary = {"results": results, "drop_R@1": {}, "drop_R@5": {}}
    for name in ["cosine", "margin4_rank32", "delta_warmstart_rank32"]:
        for k in (1, 5):
            key = f"image_retrieval_recall@{k}"
            drop = results["standard"][name][key] - results["negated"][name][key]
            summary[f"drop_R@{k}"][name] = drop
    print("\n=== R@1 drop (standard - negated) ===")
    for name, drop in summary["drop_R@1"].items():
        print(f"  {name:24s} {drop:+.2f}pp")

    with open(os.path.join(args.output_dir, "gallery_retrieval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved: {args.output_dir}/gallery_retrieval_summary.json")


if __name__ == "__main__":
    main()
