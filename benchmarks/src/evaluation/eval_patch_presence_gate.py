"""
S3 gate (IMPLEMENTATION_PLAN.md Part XVII) -- is there presence information left in the
patches that pooling threw away?

The duality measured in this repo says a text-side correction cannot widen the caption
target: only {m_I, d_I} can, and every post-hoc method in the literature is text-side.
An image-side correction is therefore the open quadrant -- but only if the frozen vision
tower still holds presence information that the pooled embedding does not expose. The
pooled probe reaches 0.65. Object presence is a local property and CLS pooling is a
global summary, so the hypothesis is that the patches hold more.

**This script is a gate, not a method.** If the patch readouts do not beat the pooled
0.65, the image-side branch loses its premise and the honest conclusion is that the
binding bottleneck is not recoverable from frozen vision features at all -- which is
itself a stronger diagnosis than the one the paper currently makes.

Four readouts on the same counterfactual pairs, all evaluated with pair-wise GroupKFold
so the two halves of a minimal pair never straddle a fold (the split that turned an
earlier image probe from 29% into 63%):

    pooled_cosine     CLS -> ln_post -> proj, scored against "a photo of a {c}."
    maxpatch_cosine   every patch projected the same way, best patch kept
    pooled_probe      linear probe on the pooled embedding      (the published 0.65)
    maxpatch_probe    linear probe on the element-wise max over projected patches

The two cosine rows need no training at all, so a gap between them isolates pooling
from probing.

Usage (from the repo root):
    python -m benchmarks.src.evaluation.eval_patch_presence_gate \\
        --restrict_objects logs/evaluation/00_concept_sets/paper33.txt \\
        --output_dir logs/evaluation/01_paper/2026-09-08_patch_presence_gate
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

try:
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction, DEFAULT_CACHE_DIR,
    )
    from benchmarks.src.analysis.config import set_seed, coerce_bool_column
    from benchmarks.src.analysis.paths import resolve_image_path as resolve_path
    from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import encode_texts_unified
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.model_loader import load_clip_for_eval
    from analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction, DEFAULT_CACHE_DIR,
    )
    from analysis.config import set_seed, coerce_bool_column
    from analysis.paths import resolve_image_path as resolve_path
    from evaluation.eval_e2_hadamard_decomposition import encode_texts_unified

PUBLISHED_POOLED_PROBE = 0.65


@torch.no_grad()
def _last_block_value_only(block, x):
    """Run the final block with the attention mixing bypassed (the MaskCLIP readout).

    CLIP's last attention layer is tuned to build a good CLS token, and in doing so it
    mixes every patch toward the global summary -- which is exactly why patch tokens
    read out of the standard forward pass localize poorly. Substituting the value
    projection for the attended output keeps each token's own content and is the
    published fix for dense CLIP features. Everything else (residual, LayerScale, MLP)
    is left as the block defines it, so the only change is the mixing.
    """
    d = block.attn.embed_dim
    w = block.attn.in_proj_weight[2 * d:3 * d]
    b = block.attn.in_proj_bias[2 * d:3 * d] if block.attn.in_proj_bias is not None else None
    v = torch.nn.functional.linear(block.ln_1(x), w, b)
    v = block.attn.out_proj(v)
    x = x + block.ls_1(v)
    return x + block.ls_2(block.mlp(block.ln_2(x)))


@torch.no_grad()
def encode_patches(model, preprocess, paths: List[str], device: str, batch_size: int,
                   vproj: bool = False):
    """Project every patch token into the joint space, not just the CLS token.

    Walks the vendored fork's visual tower by hand for the same reason
    ``extract_vision_features_unified`` does -- the tower runs its blocks in LND -- but
    keeps the patch axis instead of collapsing it. Returns (pooled, patches, flags) with
    pooled (N, D) and patches (N, P, D), both L2-normalized in the joint space.
    """
    from PIL import Image

    visual = model.visual
    conv1, cls_emb = visual.conv1, visual.class_embedding
    pos_emb, ln_pre = visual.positional_embedding, visual.ln_pre
    blocks = visual.transformer.resblocks
    ln_post, proj = visual.ln_post, visual.proj

    pooled, patches, flags = [], [], []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        tensors, ok = [], []
        for p in chunk:
            try:
                tensors.append(preprocess(Image.open(p).convert("RGB")))
                ok.append(True)
            except Exception:
                ok.append(False)
        flags += ok
        if not tensors:
            continue
        x = torch.stack(tensors).to(device)
        x = conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        ce = cls_emb.to(x.dtype).unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([ce, x], dim=1) + pos_emb.to(x.dtype)
        x = ln_pre(x).permute(1, 0, 2)
        for i, b in enumerate(blocks):
            last = (i == len(blocks) - 1)
            x = _last_block_value_only(b, x) if (last and vproj) else b(x)
        x = x.permute(1, 0, 2)
        x = ln_post(x)
        if proj is not None:
            x = x @ proj
        x = x / x.norm(dim=-1, keepdim=True)
        pooled.append(x[:, 0].float().cpu())
        patches.append(x[:, 1:].float().cpu())
    return (torch.cat(pooled).numpy(), torch.cat(patches).numpy(), np.array(flags))


def pairwise_accuracy(score_pres: np.ndarray, score_abs: np.ndarray) -> float:
    """Share of minimal pairs where the present image outscores its counterfactual."""
    return float(np.mean(score_pres > score_abs)) * 100


def probe_accuracy(feats: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                   seed: int, n_splits: int = 5) -> float:
    """Pair-wise GroupKFold linear probe -- the split the published 0.65 uses."""
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        return float("nan")
    correct = np.zeros(len(labels), dtype=bool)
    for tr, te in GroupKFold(n_splits=n_splits).split(feats, labels, groups):
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(feats[tr], labels[tr])
        correct[te] = clf.predict(feats[te]) == labels[te]
    return float(correct.mean()) * 100


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv_path", default="benchmarks/data/images/beaf_counterfactual_6col.csv")
    ap.add_argument("--restrict_objects", default="logs/evaluation/00_concept_sets/paper33.txt")
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--min_pairs", type=int, default=20)
    ap.add_argument("--template", default="a photo of a {}.")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--topk", type=int, default=5, help="patches kept by the top-k readouts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--output_dir",
                    default="logs/evaluation/01_paper/2026-09-08_patch_presence_gate")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print("  S3 gate -- does the patch axis hold presence that pooling drops?")
    print("=" * 72)
    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    cache_kw = dict(model=args.model, pretrained=args.pretrained,
                    enabled=args.use_cache, cache_dir=args.cache_dir)

    df = coerce_bool_column(pd.read_csv(args.csv_path), "object_in_image")
    restriction = load_object_restriction(args.restrict_objects)
    objects = sorted(df["object_name"].unique().tolist())
    if restriction:
        objects = [o for o in objects if o in set(restriction)]

    per_concept, agg = [], {k: [] for k in
                            ("pooled_cosine", "maxpatch_cosine", "topkpatch_cosine",
                             "meanpatch_cosine", "pooled_probe", "maxpatch_probe",
                             "topkpatch_probe", "pooled_plus_patch_probe",
                             "vproj_max_cosine", "vproj_topk_cosine",
                             "vproj_max_probe", "pooled_plus_vproj_probe")}
    for obj in objects:
        d = df[df["object_name"] == obj].reset_index(drop=True)
        d_t = d[d["object_in_image"] == True].reset_index(drop=True)
        d_f = d[d["object_in_image"] == False].reset_index(drop=True)
        n = min(len(d_t), len(d_f))
        if n < args.min_pairs:
            continue
        p_pres = [resolve_path(p, args.image_root) for p in d_t["image_path"].tolist()[:n]]
        p_abs = [resolve_path(p, args.image_root) for p in d_f["image_path"].tolist()[:n]]

        pooled_p, patch_p, ok_p = cached_encode(
            lambda: encode_patches(model, preprocess, p_pres, device, args.batch_size),
            kind="patchgate_pres@pooled+patches+flags", items=p_pres, **cache_kw)
        pooled_a, patch_a, ok_a = cached_encode(
            lambda: encode_patches(model, preprocess, p_abs, device, args.batch_size),
            kind="patchgate_abs@pooled+patches+flags", items=p_abs, **cache_kw)
        vpatch_p = cached_encode(
            lambda: encode_patches(model, preprocess, p_pres, device, args.batch_size, vproj=True),
            kind="patchgate_pres@vproj+pooled+patches+flags", items=p_pres, **cache_kw)[1]
        vpatch_a = cached_encode(
            lambda: encode_patches(model, preprocess, p_abs, device, args.batch_size, vproj=True),
            kind="patchgate_abs@vproj+pooled+patches+flags", items=p_abs, **cache_kw)[1]
        keep = np.where(ok_p & ok_a)[0]
        if len(keep) < args.min_pairs:
            continue
        pooled_p, patch_p = pooled_p[keep], patch_p[keep]
        pooled_a, patch_a = pooled_a[keep], patch_a[keep]
        vpatch_p, vpatch_a = vpatch_p[keep], vpatch_a[keep]

        prompt = [args.template.format(obj)]
        t, _ = cached_encode(
            lambda: encode_texts_unified(model, tokenizer, prompt, device, args.batch_size),
            kind="patchgate_concept@l2norm+raw", items=prompt, **cache_kw)
        t = t[0]

        s_pool_p, s_pool_a = pooled_p @ t, pooled_a @ t
        sp_p, sp_a = patch_p @ t, patch_a @ t                 # (n, P) per-patch scores
        s_max_p, s_max_a = sp_p.max(axis=1), sp_a.max(axis=1)
        topk = min(args.topk, sp_p.shape[1])
        s_top_p = np.sort(sp_p, axis=1)[:, -topk:].mean(axis=1)
        s_top_a = np.sort(sp_a, axis=1)[:, -topk:].mean(axis=1)
        s_mean_p, s_mean_a = sp_p.mean(axis=1), sp_a.mean(axis=1)

        def topk_feat(P):
            idx = np.argsort(P @ t, axis=1)[:, -topk:]
            return np.take_along_axis(P, idx[:, :, None], axis=1).mean(axis=1)

        vp_p, vp_a = vpatch_p @ t, vpatch_a @ t
        s_vmax_p, s_vmax_a = vp_p.max(axis=1), vp_a.max(axis=1)
        vtopk = min(args.topk, vp_p.shape[1])
        s_vtop_p = np.sort(vp_p, axis=1)[:, -vtopk:].mean(axis=1)
        s_vtop_a = np.sort(vp_a, axis=1)[:, -vtopk:].mean(axis=1)
        feats_vmax = np.concatenate([vpatch_p.max(axis=1), vpatch_a.max(axis=1)])
        feats_vcat = np.concatenate([np.hstack([pooled_p, vpatch_p.max(axis=1)]),
                                     np.hstack([pooled_a, vpatch_a.max(axis=1)])])

        feats_pool = np.concatenate([pooled_p, pooled_a])
        feats_max = np.concatenate([patch_p.max(axis=1), patch_a.max(axis=1)])
        feats_top = np.concatenate([topk_feat(patch_p), topk_feat(patch_a)])
        feats_cat = np.concatenate([np.hstack([pooled_p, patch_p.max(axis=1)]),
                                    np.hstack([pooled_a, patch_a.max(axis=1)])])
        labels = np.concatenate([np.ones(len(pooled_p)), np.zeros(len(pooled_a))]).astype(int)
        groups = np.concatenate([np.arange(len(pooled_p))] * 2)

        row = dict(
            concept=obj, n_pairs=len(keep),
            pooled_cosine=pairwise_accuracy(s_pool_p, s_pool_a),
            maxpatch_cosine=pairwise_accuracy(s_max_p, s_max_a),
            pooled_probe=probe_accuracy(feats_pool, labels, groups, args.seed),
            maxpatch_probe=probe_accuracy(feats_max, labels, groups, args.seed),
            topkpatch_cosine=pairwise_accuracy(s_top_p, s_top_a),
            meanpatch_cosine=pairwise_accuracy(s_mean_p, s_mean_a),
            topkpatch_probe=probe_accuracy(feats_top, labels, groups, args.seed),
            pooled_plus_patch_probe=probe_accuracy(feats_cat, labels, groups, args.seed),
            vproj_max_cosine=pairwise_accuracy(s_vmax_p, s_vmax_a),
            vproj_topk_cosine=pairwise_accuracy(s_vtop_p, s_vtop_a),
            vproj_max_probe=probe_accuracy(feats_vmax, labels, groups, args.seed),
            pooled_plus_vproj_probe=probe_accuracy(feats_vcat, labels, groups, args.seed),
        )
        per_concept.append(row)
        for k in agg:
            agg[k].append(row[k])
        print(f"  {obj:14s} n={row['n_pairs']:3d}  pooled={row['pooled_cosine']:5.1f}/"
              f"{row['pooled_probe']:5.1f}  patch={row['maxpatch_cosine']:5.1f}/"
              f"{row['maxpatch_probe']:5.1f}  vproj={row['vproj_max_cosine']:5.1f}/"
              f"{row['vproj_max_probe']:5.1f}")

    summary = {k: float(np.mean(v)) for k, v in agg.items()}
    n_better = int(np.sum(np.array(agg["maxpatch_probe"]) > np.array(agg["pooled_probe"])))
    print("\n=== concept-macro ===")
    for k, v in summary.items():
        print(f"  {k:18s} {v:6.2f}")
    print(f"  maxpatch_probe beats pooled_probe in {n_better}/{len(per_concept)} concepts")
    best = max(summary[k] for k in
               ("maxpatch_probe", "topkpatch_probe", "pooled_plus_patch_probe",
                "vproj_max_probe", "pooled_plus_vproj_probe"))
    gate = best > PUBLISHED_POOLED_PROBE * 100 + 3.0
    print(f"\n  GATE ({'PASS' if gate else 'FAIL'}): best patch-based probe "
          f"{best:.2f} vs published pooled {PUBLISHED_POOLED_PROBE*100:.0f} "
          f"(+3pp margin required)")

    pd.DataFrame(per_concept).to_csv(
        os.path.join(args.output_dir, "patch_presence_per_concept.csv"), index=False)
    with open(os.path.join(args.output_dir, "patch_presence_gate.json"), "w") as f:
        json.dump(dict(summary=summary, gate_passed=bool(gate), n_better=n_better,
                       n_concepts=len(per_concept), per_concept=per_concept,
                       provenance=build_provenance(args)), f, indent=2, default=str)
    print(f"\nsaved: {args.output_dir}/")


if __name__ == "__main__":
    main()
