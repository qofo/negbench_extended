"""
Is the ECN's effect on the 2x2 actually a rescale of the text polarity direction?

Section 8.9 measured gamma x1.89 and |alpha| x1.90 under PeakPatch's ECN, leaving
gamma/|alpha| at 0.084, and explained it by noting that both coefficients are linear
in the text polarity direction d_T:

    alpha = m_I . d_T      beta = d_I . m_T      gamma = d_I . d_T

That explanation is an inference from two ratios agreeing. This script tests it
directly, and tests its dual, so the mechanism is measured rather than asserted.

  A. Is the ECN a rescale?  Compute d_T and m_T before and after the correction. A
     pure rescale means cos(d_T', d_T) = 1 and ||d_T'||/||d_T|| = lambda, and then
     gamma and alpha must both scale by exactly that lambda. Reported against what
     actually happened, so the story fails loudly if the ECN mostly rotates d_T and
     the 1.89/1.90 agreement was a coincidence.

  B. The dual, on the same pairs.  Rescaling d_I by lambda gives beta' = lambda*beta
     and gamma' = lambda*gamma with alpha untouched, so gamma/|beta| is preserved and
     gamma/|alpha| moves -- the mirror of what a text-side rescale does. Sweeping
     lambda on both sides turns "an image-side intervention would behave dually" from
     an algebraic remark into a measurement on real embeddings.

This is deliberately not an external method. DCSM (Kang et al., ICCV 2025) is not the
image-side counterpart it is sometimes taken for: it leaves both encoders frozen, its
Functional Rows replace *text* token rows, and it scores a dense token-by-patch matrix
with a trained CNN rather than an inner product -- so the d_I/d_T factorization this
script probes does not apply to it at all.

Usage:
    python -m benchmarks.src.evaluation.eval_polarity_rescale_mechanism \\
        --peakpatch_root PeakPatch \\
        --restrict_objects logs/evaluation/01_paper/2026-08-28_e2_hadamard_decomposition/e2_per_concept_decomposition.csv \\
        --use_cache --output_dir logs/evaluation/01_paper/2026-09-02_polarity_rescale_mechanism
"""
import os
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

try:
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.cli import (
        add_model_args, add_run_args, add_data_args, add_cache_args,
        add_restriction_args, add_concept_args,
    )
    from benchmarks.src.analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction,
    )
    from benchmarks.src.analysis.config import set_seed, coerce_bool_column
    from benchmarks.src.analysis.paths import resolve_image_path as resolve_path
    from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, compute_hadamard_coordinates,
    )
    from benchmarks.src.evaluation.eval_peakpatch_2x2_audit import (
        load_peakpatch, encode_texts_peakpatch, assert_encode_text_consistency,
    )
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.model_loader import load_clip_for_eval
    from analysis.cli import (
        add_model_args, add_run_args, add_data_args, add_cache_args,
        add_restriction_args, add_concept_args,
    )
    from analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction,
    )
    from analysis.config import set_seed, coerce_bool_column
    from analysis.paths import resolve_image_path as resolve_path
    from evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, compute_hadamard_coordinates,
    )
    from evaluation.eval_peakpatch_2x2_audit import (
        load_peakpatch, encode_texts_peakpatch, assert_encode_text_consistency,
    )

LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
CHANCE_GROUP = 100 / 6


def md(plus: np.ndarray, minus: np.ndarray):
    """The mean/difference pair the factor coordinates are built from."""
    return 0.5 * (plus + minus), 0.5 * (plus - minus)


def rowwise_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.maximum(den, 1e-12)


def metrics(h: Dict[str, np.ndarray]) -> Dict[str, float]:
    cap = h["gamma"] > h["abs_alpha"]
    img = h["gamma"] > h["abs_beta"]
    return dict(
        abs_alpha=float(h["abs_alpha"].mean()), abs_beta=float(h["abs_beta"].mean()),
        gamma=float(h["gamma"].mean()),
        g_over_a=float(h["gamma"].mean() / h["abs_alpha"].mean()),
        g_over_b=float(h["gamma"].mean() / h["abs_beta"].mean()),
        caption=float(cap.mean() * 100), image=float(img.mean() * 100),
        group=float((cap & img).mean() * 100),
    )


def quad(m_I, d_I, m_T, d_T):
    """Rebuild the four scores from the factor vectors."""
    v_p, v_m = m_I + d_I, m_I - d_I
    t_p, t_m = m_T + d_T, m_T - d_T
    return (np.sum(v_p * t_p, -1), np.sum(v_m * t_p, -1),
            np.sum(v_p * t_m, -1), np.sum(v_m * t_m, -1))


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    df = coerce_bool_column(pd.read_csv(args.csv_path), "object_in_image")
    restriction = load_object_restriction(args.restrict_objects)
    objects = sorted(df["object_name"].unique())
    if restriction:
        objects = [o for o in objects if o in set(restriction)]

    pp = load_peakpatch(args.peakpatch_root, device)
    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    assert_encode_text_consistency(model, tokenizer, device)
    cache_kw = dict(model=args.model, pretrained=args.pretrained,
                    enabled=args.use_cache, cache_dir=args.cache_dir)

    MI, DI, MT, DT, MT2, DT2, names = [], [], [], [], [], [], []
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
        pos_plain, pos_ecn, _ = encode_texts_peakpatch(
            model, tokenizer, t_pos, pp, device, args.batch_size)
        neg_plain, neg_ecn, _ = encode_texts_peakpatch(
            model, tokenizer, t_neg, pp, device, args.batch_size)

        mi, di = md(v_p[keep], v_m[keep])
        mt, dt = md(pos_plain, neg_plain)
        mt2, dt2 = md(pos_ecn, neg_ecn)
        MI.append(mi); DI.append(di); MT.append(mt); DT.append(dt)
        MT2.append(mt2); DT2.append(dt2); names += [obj] * len(keep)

    m_I, d_I = np.concatenate(MI), np.concatenate(DI)
    m_T, d_T = np.concatenate(MT), np.concatenate(DT)
    m_T2, d_T2 = np.concatenate(MT2), np.concatenate(DT2)
    n_pairs = len(m_I)
    print(f"\n  {len(set(names))} concepts, {n_pairs} pairs")

    # ── A. Is the ECN a rescale of d_T? ─────────────────────────────────
    lam_d = np.linalg.norm(d_T2, axis=-1) / np.maximum(np.linalg.norm(d_T, axis=-1), 1e-12)
    cos_d = rowwise_cos(d_T2, d_T)
    lam_m = np.linalg.norm(m_T2, axis=-1) / np.maximum(np.linalg.norm(m_T, axis=-1), 1e-12)
    cos_m = rowwise_cos(m_T2, m_T)

    print("\n  [A] What the ECN does to the text factor vectors")
    print(f"    ||d_T'||/||d_T||   mean {lam_d.mean():6.3f}   median {np.median(lam_d):6.3f}")
    print(f"    cos(d_T', d_T)     mean {cos_d.mean():6.3f}   median {np.median(cos_d):6.3f}")
    print(f"    ||m_T'||/||m_T||   mean {lam_m.mean():6.3f}   median {np.median(lam_m):6.3f}")
    print(f"    cos(m_T', m_T)     mean {cos_m.mean():6.3f}   median {np.median(cos_m):6.3f}")

    obs_cos = compute_hadamard_coordinates(*quad(m_I, d_I, m_T, d_T))
    obs_ecn = compute_hadamard_coordinates(*quad(m_I, d_I, m_T2, d_T2))
    # The pure-rescale counterfactual: keep the ECN's growth of d_T, drop its rotation
    # and its change to m_T. If the mechanism story holds this reproduces the ECN.
    pred = compute_hadamard_coordinates(*quad(m_I, d_I, m_T, lam_d[:, None] * d_T))

    print("\n  [A] Observed vs the pure-rescale counterfactual")
    print(f"    {'':<22}{'|alpha|':>11}{'|beta|':>11}{'gamma':>11}"
          f"{'g/|a|':>8}{'g/|b|':>8}{'caption':>9}{'image':>8}")
    rows = {"cosine": obs_cos, "ECN (observed)": obs_ecn,
            "pure rescale of d_T": pred}
    table = {}
    for label, h in rows.items():
        r = metrics(h)
        table[label] = r
        print(f"    {label:<22}{r['abs_alpha']:11.3e}{r['abs_beta']:11.3e}{r['gamma']:11.3e}"
              f"{r['g_over_a']:8.3f}{r['g_over_b']:8.3f}{r['caption']:9.2f}{r['image']:8.2f}")

    # ── B. The dual, measured ───────────────────────────────────────────
    print("\n  [B] Sweeping a controlled rescale on each side")
    print(f"    {'side':<8}{'lambda':>8}{'gamma':>11}{'g/|a|':>9}{'g/|b|':>9}"
          f"{'caption':>9}{'image':>8}{'group':>8}")
    sweep = []
    for side in ("text", "image"):
        for lam in LAMBDAS:
            if side == "text":
                h = compute_hadamard_coordinates(*quad(m_I, d_I, m_T, lam * d_T))
            else:
                h = compute_hadamard_coordinates(*quad(m_I, lam * d_I, m_T, d_T))
            r = metrics(h)
            sweep.append(dict(side=side, lam=lam, **r))
            print(f"    {side:<8}{lam:8.2f}{r['gamma']:11.3e}{r['g_over_a']:9.3f}"
                  f"{r['g_over_b']:9.3f}{r['caption']:9.2f}{r['image']:8.2f}{r['group']:8.2f}")

    ds = pd.DataFrame(sweep)
    ds.to_csv(os.path.join(args.output_dir, "rescale_sweep.csv"), index=False)

    # The duality is exact, so state it as a check rather than a trend.
    t = ds[ds.side == "text"]
    i = ds[ds.side == "image"]
    print("\n  [B] Duality checks (exact, so these are pass/fail)")
    print(f"    text rescale leaves g/|a| fixed : "
          f"spread {t.g_over_a.max() - t.g_over_a.min():.2e}")
    print(f"    image rescale leaves g/|b| fixed: "
          f"spread {i.g_over_b.max() - i.g_over_b.min():.2e}")
    print(f"    text rescale moves g/|b|        : "
          f"{t.g_over_b.min():.3f} -> {t.g_over_b.max():.3f}")
    print(f"    image rescale moves g/|a|       : "
          f"{i.g_over_a.min():.3f} -> {i.g_over_a.max():.3f}")

    out = dict(
        n_concepts=len(set(names)), n_pairs=n_pairs,
        ecn_vector_change=dict(
            d_T_norm_ratio_mean=float(lam_d.mean()), d_T_cos_mean=float(cos_d.mean()),
            m_T_norm_ratio_mean=float(lam_m.mean()), m_T_cos_mean=float(cos_m.mean())),
        coefficients=table,
        sweep=ds.to_dict(orient="records"),
        provenance=build_provenance(args, extra=dict(lambdas=list(LAMBDAS))),
    )
    with open(os.path.join(args.output_dir, "mechanism_summary.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser, "ViT-B-32", "openai")
    add_run_args(parser, "logs/evaluation/polarity_rescale_mechanism", seed=42, batch_size=128)
    add_data_args(parser, csv_path="benchmarks/data/images/beaf_counterfactual_6col.csv",
                  image_root="benchmarks/data/images")
    add_cache_args(parser)
    add_restriction_args(parser, "Concept set to share with the e2 runs")
    add_concept_args(parser, help_text="Minimum counterfactual pairs per object")
    parser.add_argument("--peakpatch_root", type=str, default="PeakPatch")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
