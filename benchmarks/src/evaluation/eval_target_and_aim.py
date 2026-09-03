"""
Does a negation intervention widen the target, or only re-aim at it?

The counterfactual caption criterion is gamma > |alpha|, and with

    alpha = m_I . d_T      beta = d_I . m_T      gamma = d_I . d_T

both sides are inner products against vectors in the plane span{m_I, d_I}. So the
criterion sees the text polarity vector d_T only through its projection there, and
splits every intervention into two things it could be doing:

  target  the angular measure of directions in that plane which satisfy the
          criterion. Set by m_I and d_I -- image-side quantities.
  aim     where d_T's projection actually points. Text-side.

An intervention that leaves the image tower frozen cannot change the target, however
much it rebuilds the text representation; it can only re-aim. One that fine-tunes the
image tower can do both. This script measures both quantities for every model and
intervention on the same pairs, and reports

    aim quality = observed accuracy / target size

which is 1 when the model is no better than a random direction in the plane.

Scale matters differently for the two criteria, and getting it wrong inflates one of
them. gamma and alpha are both linear in d_T, so gamma > |alpha| is invariant to
||d_T|| and the target depends on the angle alone. gamma > |beta| is not: beta does
not involve d_T, so that target has to be swept at the pair's actual in-plane
magnitude ||P(d_T)||, not at unit length.

Usage:
    python -m benchmarks.src.evaluation.eval_target_and_aim \\
        --restrict_objects logs/evaluation/01_paper/2026-08-28_e2_hadamard_decomposition/e2_per_concept_decomposition.csv \\
        --use_cache --peakpatch_root PeakPatch \\
        --output_dir logs/evaluation/01_paper/2026-09-03_target_and_aim
"""
import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.cli import (
        add_run_args, add_data_args, add_cache_args, add_restriction_args,
        add_concept_args,
    )
    from benchmarks.src.analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction,
    )
    from benchmarks.src.analysis.config import set_seed, coerce_bool_column
    from benchmarks.src.analysis.paths import resolve_image_path as resolve_path
    from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, encode_texts_unified,
    )
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.model_loader import load_clip_for_eval
    from analysis.cli import (
        add_run_args, add_data_args, add_cache_args, add_restriction_args,
        add_concept_args,
    )
    from analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction,
    )
    from analysis.config import set_seed, coerce_bool_column
    from analysis.paths import resolve_image_path as resolve_path
    from evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, encode_texts_unified,
    )

# (display name, --model, --pretrained, is a negation-specific intervention)
MODELS = [
    ("ViT-B/32 (OpenAI)", "ViT-B-32", "openai", False),
    ("ViT-B/16", "ViT-B-16", "openai", False),
    ("ViT-L/14", "ViT-L-14", "openai", False),
    ("LAION-2B", "ViT-B-32", "laion2b_s34b_b79k", False),
    ("SigLIP B/16", "ViT-B-16-SigLIP", "webli", False),
    ("CoN-CLIP", "ViT-B-32",
     "benchmarks/models/ConCLIP/conclip_b32_openclip_version.pt", True),
    ("NegCLIP", "ViT-B-32", "benchmarks/models/NegCLIP/negclip.pth", True),
    ("NegCLIP-NegFull", "ViT-B-32",
     "benchmarks/models/NegCLIP_CC12M_NegFull_ViT-B-32_lr1e-8_clw0.99_mlw0.01/checkpoint.pt",
     True),
    ("CLIP-NegFull", "ViT-B-32",
     "benchmarks/models/CLIP_CC12M_NegFull_ViT-B-32_lr1e-8_clw0.99_mlw0.01/checkpoint.pt",
     True),
]
N_ANGLES = 720


def plane_quantities(m_I, d_I, m_T, d_T):
    """
    Everything the two criteria depend on, in the plane span{m_I, d_I}.

    Returns the per-pair coefficients plus the angular target sizes.
    """
    e1 = m_I / np.maximum(np.linalg.norm(m_I, axis=-1, keepdims=True), 1e-12)
    perp = d_I - np.sum(d_I * e1, axis=-1, keepdims=True) * e1
    e2 = perp / np.maximum(np.linalg.norm(perp, axis=-1, keepdims=True), 1e-12)

    norm_mI = np.linalg.norm(m_I, axis=-1)
    g1 = np.sum(d_I * e1, axis=-1)
    g2 = np.sum(d_I * e2, axis=-1)
    beta = np.sum(d_I * m_T, axis=-1)

    # The pair's actual in-plane magnitude; the image target is only defined at it.
    p1 = np.sum(d_T * e1, axis=-1)
    p2 = np.sum(d_T * e2, axis=-1)
    r = np.hypot(p1, p2)

    alpha = norm_mI * p1
    gamma = g1 * p1 + g2 * p2

    theta = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
    u1, u2 = np.cos(theta)[None, :], np.sin(theta)[None, :]
    # Caption: both sides carry a factor of r, so it cancels -- angle only.
    cap_hits = (g1[:, None] * u1 + g2[:, None] * u2) > np.abs(norm_mI[:, None] * u1)
    # Image: beta carries no factor of r, so sweep at the pair's own magnitude.
    gamma_at_r = r[:, None] * (g1[:, None] * u1 + g2[:, None] * u2)
    img_hits = gamma_at_r > np.abs(beta)[:, None]

    return dict(alpha=alpha, beta=beta, gamma=gamma,
                target_caption=cap_hits.mean(axis=1) * 100,
                target_image=img_hits.mean(axis=1) * 100,
                in_plane_share=r / np.maximum(np.linalg.norm(d_T, axis=-1), 1e-12))


def summarize(name: str, q: Dict[str, np.ndarray], groups: np.ndarray,
              n_boot: int, rng) -> Dict:
    cap = q["gamma"] > np.abs(q["alpha"])
    img = q["gamma"] > np.abs(q["beta"])
    tgt_cap, tgt_img = q["target_caption"].mean(), q["target_image"].mean()

    def boot(flags):
        keys = np.unique(groups)
        by = {k: flags[groups == k] for k in keys}
        d = np.array([np.concatenate([by[k] for k in rng.choice(keys, len(keys), True)]).mean()
                      for _ in range(n_boot)]) * 100
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    cap_lo, cap_hi = boot(cap)
    return dict(
        name=name, n_pairs=len(cap),
        target_caption=float(tgt_cap), target_image=float(tgt_img),
        acc_caption=float(cap.mean() * 100), acc_image=float(img.mean() * 100),
        acc_group=float((cap & img).mean() * 100),
        aim_caption=float(cap.mean() * 100 / max(tgt_cap, 1e-9)),
        aim_image=float(img.mean() * 100 / max(tgt_img, 1e-9)),
        acc_caption_lo=cap_lo, acc_caption_hi=cap_hi,
        in_plane_share=float(q["in_plane_share"].mean()),
        abs_alpha=float(np.abs(q["alpha"]).mean()),
        abs_beta=float(np.abs(q["beta"]).mean()),
        gamma=float(q["gamma"].mean()),
    )


def load_pairs(df, objects, model, preprocess, tokenizer, device, args, cache_kw,
               ecn=None, pp=None):
    """Per-pair factor vectors for one model, optionally with an ECN applied to text."""
    MI, DI, MT, DT, names = [], [], [], [], []
    for obj in objects:
        d = df[df["object_name"] == obj].reset_index(drop=True)
        d_t = d[d["object_in_image"] == True].reset_index(drop=True)
        d_f = d[d["object_in_image"] == False].reset_index(drop=True)
        n = min(len(d_t), len(d_f))
        if n < args.min_pairs:
            continue
        p_pres = [resolve_path(p, args.image_root) for p in d_t["image_path"][:n]]
        p_abs = [resolve_path(p, args.image_root) for p in d_f["image_path"][:n]]
        v_p, _, mp = cached_encode(
            lambda: encode_images_unified(model, preprocess, p_pres, device, args.batch_size),
            kind="image_pres@norm+raw+flags", items=p_pres, **cache_kw)
        v_m, _, ma = cached_encode(
            lambda: encode_images_unified(model, preprocess, p_abs, device, args.batch_size),
            kind="image_abs@norm+raw+flags", items=p_abs, **cache_kw)
        keep = np.where(mp & ma)[0]
        if len(keep) < args.min_pairs:
            continue
        t_pos = [d_t["positive_caption"].tolist()[:n][i] for i in keep]
        t_neg = [d_t["negative_caption"].tolist()[:n][i] for i in keep]

        if ecn is None:
            tp, _ = cached_encode(
                lambda: encode_texts_unified(model, tokenizer, t_pos, device, args.batch_size),
                kind="text_pos@norm+raw", items=t_pos, **cache_kw)
            tn, _ = cached_encode(
                lambda: encode_texts_unified(model, tokenizer, t_neg, device, args.batch_size),
                kind="text_neg@norm+raw", items=t_neg, **cache_kw)
        else:
            from evaluation.eval_peakpatch_2x2_audit import encode_texts_peakpatch
            _, tp, _ = encode_texts_peakpatch(model, tokenizer, t_pos, pp, device,
                                              args.batch_size)
            _, tn, _ = encode_texts_peakpatch(model, tokenizer, t_neg, pp, device,
                                              args.batch_size)

        MI.append(0.5 * (v_p[keep] + v_m[keep])); DI.append(0.5 * (v_p[keep] - v_m[keep]))
        MT.append(0.5 * (tp + tn)); DT.append(0.5 * (tp - tn))
        names += [obj] * len(keep)
    return (np.concatenate(MI), np.concatenate(DI),
            np.concatenate(MT), np.concatenate(DT), np.array(names))


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    df = coerce_bool_column(pd.read_csv(args.csv_path), "object_in_image")
    restriction = load_object_restriction(args.restrict_objects)
    objects = sorted(df["object_name"].unique())
    if restriction:
        objects = [o for o in objects if o in set(restriction)]
    print(f"\n  Concepts: {len(objects)}   angles swept: {N_ANGLES}\n")

    rows = []
    for name, arch, pretrained, is_intervention in MODELS:
        print(f"  [{name}]")
        model, preprocess, tokenizer = load_clip_for_eval(arch, pretrained, device)
        cache_kw = dict(model=arch, pretrained=pretrained,
                        enabled=args.use_cache, cache_dir=args.cache_dir)
        m_I, d_I, m_T, d_T, groups = load_pairs(
            df, objects, model, preprocess, tokenizer, device, args, cache_kw)
        r = summarize(name, plane_quantities(m_I, d_I, m_T, d_T), groups, args.n_boot, rng)
        r["family"] = "intervention" if is_intervention else "backbone"
        r["image_tower"] = "fine-tuned" if is_intervention else "pretrained"
        rows.append(r)
        print(f"    target cap {r['target_caption']:6.2f}%  acc {r['acc_caption']:6.2f}%  "
              f"aim {r['aim_caption']:5.2f}x   |   target img {r['target_image']:6.2f}%  "
              f"acc {r['acc_image']:6.2f}%  aim {r['aim_image']:5.2f}x")
        del model
        torch.cuda.empty_cache()

    if args.peakpatch_root:
        from evaluation.eval_peakpatch_2x2_audit import load_peakpatch
        print("\n  [PeakPatch ECN (frozen ViT-B/32)]")
        pp = load_peakpatch(args.peakpatch_root, device)
        model, preprocess, tokenizer = load_clip_for_eval("ViT-B-32", "openai", device)
        cache_kw = dict(model="ViT-B-32", pretrained="openai",
                        enabled=args.use_cache, cache_dir=args.cache_dir)
        m_I, d_I, m_T, d_T, groups = load_pairs(
            df, objects, model, preprocess, tokenizer, device, args, cache_kw,
            ecn=pp["ec"], pp=pp)
        r = summarize("PeakPatch ECN", plane_quantities(m_I, d_I, m_T, d_T),
                      groups, args.n_boot, rng)
        r["family"], r["image_tower"] = "intervention", "frozen"
        rows.append(r)
        print(f"    target cap {r['target_caption']:6.2f}%  acc {r['acc_caption']:6.2f}%  "
              f"aim {r['aim_caption']:5.2f}x   |   target img {r['target_image']:6.2f}%  "
              f"acc {r['acc_image']:6.2f}%  aim {r['aim_image']:5.2f}x")

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(args.output_dir, "target_and_aim.csv"), index=False)

    print("\n" + "=" * 92)
    print(f"  {'model':<20}{'tower':<12}{'tgt cap':>9}{'acc cap':>9}{'aim':>7}"
          f"{'tgt img':>9}{'acc img':>9}{'aim':>7}{'group':>8}")
    for _, r in d.iterrows():
        print(f"  {r['name']:<20}{r['image_tower']:<12}{r['target_caption']:9.2f}"
              f"{r['acc_caption']:9.2f}{r['aim_caption']:7.2f}{r['target_image']:9.2f}"
              f"{r['acc_image']:9.2f}{r['aim_image']:7.2f}{r['acc_group']:8.2f}")

    base = d[d.name == "ViT-B/32 (OpenAI)"].iloc[0]
    print(f"\n  --- read against the frozen baseline (target cap "
          f"{base['target_caption']:.2f}%) ---")
    for _, r in d.iterrows():
        dt = r["target_caption"] - base["target_caption"]
        print(f"    {r['name']:<20}{r['image_tower']:<12}"
              f"target {dt:+6.2f}pp   aim {r['aim_caption']:5.2f}x")

    with open(os.path.join(args.output_dir, "target_and_aim.json"), "w") as f:
        json.dump(dict(n_angles=N_ANGLES, rows=d.to_dict(orient="records"),
                       provenance=build_provenance(args)), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {args.output_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(ap, "logs/evaluation/target_and_aim", seed=42, batch_size=128)
    add_data_args(ap, csv_path="benchmarks/data/images/beaf_counterfactual_6col.csv",
                  image_root="benchmarks/data/images")
    add_cache_args(ap)
    add_restriction_args(ap, "Concept set to share with the e2 runs")
    add_concept_args(ap, help_text="Minimum counterfactual pairs per object")
    ap.add_argument("--peakpatch_root", type=str, default="PeakPatch")
    ap.add_argument("--n_boot", type=int, default=2000)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
