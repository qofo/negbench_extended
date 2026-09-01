"""Separate the probe normal w from the state-transition direction d, per concept.

PAPER.md section 5 explains the interaction term with an additive model,

    gamma  ~=  1/4 * a_I * a_T * cos(d_I, d_T),

which follows from writing v_a = mu_I + (a/2) d_I and t_b = mu_T + (b/2) d_T and reading off
the ab coefficient. That derivation needs d to be the *mean difference between the two states*.
But `eval_per_object_alignment_intervention.py` builds the vectors it calls `d_I`/`d_T` from a
logistic probe (``d_I_tr = w_v / ||w_v||``), and a probe normal is not the mean difference: under
a Gaussian model w ~ Sigma^-1 d, so the two coincide only for isotropic covariance. Section 5
already reports the symptom -- rotating the probe normals to cos = 1.0 delivered 4.85x gamma
against a predicted 9.37x, and the movement direction recovered post hoc was ~0.52, not 1.0.

This script measures both families on the same pairs and asks which one the additive model
actually wants:

    d_I = mean(v_present) - mean(v_absent)      w_I = probe normal separating the image states
    d_T = mean(t_positive) - mean(t_negative)   w_T = probe normal separating the caption states

and reports cos(w_I, d_I), cos(w_T, d_T), cos(d_I, d_T), cos(w_I, w_T) against the measured gamma.

Pre-registered criteria (IMPLEMENTATION_PLAN.md Part III-1):
    G1  gamma correlates with cos(d_I, d_T) across concepts (Spearman, one-sided)
    G2  cos(w, d) < 0.95 -- the two vectors are in fact different
    G3  the d-based additive fit is no worse than the w-based one

These are descriptive geometry estimates, not held-out performance claims: the directions are
estimated on all pairs of a concept, and the reported fold spread is a stability check only.

Usage (from the repo root):
    python -m benchmarks.src.evaluation.eval_direction_vs_probe_normal \
        --csv_path benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv \
        --output_dir logs/evaluation/01_paper/2026-08-31_direction_vs_probe/vitb32_openai
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

try:
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.cli import (
        add_model_args, add_run_args, add_data_args, add_cache_args,
        add_restriction_args, add_concept_args, add_bias_args,
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
        add_model_args, add_run_args, add_data_args, add_cache_args,
        add_restriction_args, add_concept_args, add_bias_args,
    )
    from analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction,
    )
    from analysis.config import set_seed, coerce_bool_column
    from analysis.paths import resolve_image_path as resolve_path
    from evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, encode_texts_unified,
    )


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))


def _probe_normal(pos: np.ndarray, neg: np.ndarray, seed: int, use_bias: bool) -> np.ndarray:
    """Logistic normal separating the two states, oriented so that `pos` scores higher."""
    X = np.vstack([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, fit_intercept=use_bias)
    clf.fit(X, y)
    return clf.coef_[0]


def analyse_concept(v_pres, v_abs, t_pos, t_neg, seed: int, use_bias: bool) -> Dict[str, float]:
    """Both direction families, the measured gamma, and the two additive predictions."""
    dv = v_pres - v_abs                       # per-pair image state transition
    dt = t_pos - t_neg                        # per-pair caption state transition
    gamma_per_pair = 0.25 * np.sum(dv * dt, axis=1)

    d_I, d_T = dv.mean(axis=0), dt.mean(axis=0)
    w_I = _probe_normal(v_pres, v_abs, seed, use_bias)
    w_T = _probe_normal(t_pos, t_neg, seed, use_bias)

    # Additive prediction from the mean-difference directions: gamma ~= 1/4 d_I . d_T.
    gamma_pred_d = 0.25 * float(np.dot(d_I, d_T))

    # The same quantity restricted to the probe-normal directions: the rank-1 projection of
    # d_I . d_T onto (w_I, w_T). a^w is the state separation measured along the unit normal.
    u_I, u_T = _unit(w_I), _unit(w_T)
    a_I_w = float(np.dot(d_I, u_I))
    a_T_w = float(np.dot(d_T, u_T))
    gamma_pred_w = 0.25 * a_I_w * a_T_w * float(np.dot(u_I, u_T))

    # Fold-to-fold spread of cos(w, d): a stability check on the probe fit, not a CV score.
    spread = []
    n = len(v_pres)
    if n >= 10:
        for tr, _ in KFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(n)):
            w_i_tr = _probe_normal(v_pres[tr], v_abs[tr], seed, use_bias)
            spread.append(_cos(w_i_tr, dv[tr].mean(axis=0)))

    return dict(
        n_pairs=n,
        gamma_measured=float(gamma_per_pair.mean()),
        gamma_pred_d=gamma_pred_d,
        gamma_pred_w=gamma_pred_w,
        cos_wI_dI=_cos(w_I, d_I),
        cos_wT_dT=_cos(w_T, d_T),
        cos_dI_dT=_cos(d_I, d_T),
        cos_wI_wT=_cos(w_I, w_T),
        norm_dI=float(np.linalg.norm(d_I)),
        norm_dT=float(np.linalg.norm(d_T)),
        a_I_along_w=a_I_w,
        a_T_along_w=a_T_w,
        cos_wI_dI_fold_std=float(np.std(spread)) if spread else float("nan"),
    )


def render(df: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.hist(df["cos_wI_dI"], bins=18, alpha=0.75, label=r"$\cos(w_I, d_I)$ image")
    ax.hist(df["cos_wT_dT"], bins=18, alpha=0.75, label=r"$\cos(w_T, d_T)$ text")
    ax.axvline(0.95, color="crimson", ls="--", lw=1.4, label="G2 threshold 0.95")
    ax.set_xlabel("cosine"); ax.set_ylabel("concepts")
    ax.set_title("Probe normal vs state-transition direction")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(df["cos_dI_dT"], df["gamma_measured"], s=34, alpha=0.8, label=r"$\cos(d_I,d_T)$")
    ax.scatter(df["cos_wI_wT"], df["gamma_measured"], s=34, alpha=0.8, marker="^",
               label=r"$\cos(w_I,w_T)$")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("cross-modal alignment"); ax.set_ylabel(r"measured $\gamma$")
    ax.set_title(r"Which alignment tracks $\gamma$?")
    ax.legend(fontsize=8)

    ax = axes[2]
    lo = float(min(df["gamma_measured"].min(), df["gamma_pred_d"].min(), df["gamma_pred_w"].min()))
    hi = float(max(df["gamma_measured"].max(), df["gamma_pred_d"].max(), df["gamma_pred_w"].max()))
    ax.plot([lo, hi], [lo, hi], color="grey", ls=":", lw=1.2)
    ax.scatter(df["gamma_measured"], df["gamma_pred_d"], s=34, alpha=0.8, label="mean-difference $d$")
    ax.scatter(df["gamma_measured"], df["gamma_pred_w"], s=34, alpha=0.8, marker="^",
               label="probe normal $w$")
    ax.set_xlabel(r"measured $\gamma$"); ax.set_ylabel(r"additive-model $\gamma$")
    ax.set_title("Additive model: which vectors fit?")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_direction_vs_probe.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Probe normal w vs state-transition direction d")
    add_model_args(parser, "ViT-B-32", "openai")
    add_run_args(parser, "logs/evaluation/direction_vs_probe", seed=42, batch_size=128)
    add_data_args(parser,
                  csv_path="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv",
                  image_root="benchmarks/data/images")
    add_cache_args(parser)
    add_restriction_args(parser, "Comma list, or path to txt/csv/json, limiting evaluation to an exact concept set")
    add_concept_args(parser)
    add_bias_args(parser)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bias = not args.no_bias
    cache_kw = dict(model=args.model, pretrained=args.pretrained,
                    cache_dir=args.cache_dir, enabled=args.use_cache)

    print("=" * 72)
    print("  Probe normal (w) vs state-transition direction (d)")
    print(f"  Model {args.model} ({args.pretrained}) | device {device} | min_pairs {args.min_pairs}")
    print(f"  CSV   {args.csv_path}")
    print("=" * 72)

    df = pd.read_csv(args.csv_path)
    coerce_bool_column(df, "object_in_image")
    concepts = [o for o in sorted(df["object_name"].unique()) if "," not in str(o)]
    restrict = load_object_restriction(args.restrict_objects)
    if restrict is not None:
        concepts = [o for o in concepts if o in set(restrict)]

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)

    records = []
    for obj in concepts:
        d_obj = df[df["object_name"] == obj].reset_index(drop=True)
        d_true = d_obj[d_obj["object_in_image"] == True].reset_index(drop=True)
        d_false = d_obj[d_obj["object_in_image"] == False].reset_index(drop=True)
        n = min(len(d_true), len(d_false))
        if n < args.min_pairs:
            continue

        p_pres = [resolve_path(p, args.image_root) for p in d_true["image_path"].tolist()[:n]]
        p_abs = [resolve_path(p, args.image_root) for p in d_false["image_path"].tolist()[:n]]
        txt_pos = d_true["positive_caption"].tolist()[:n]
        txt_neg = d_true["negative_caption"].tolist()[:n]

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
        txt_pos = [txt_pos[i] for i in keep]
        txt_neg = [txt_neg[i] for i in keep]

        t_pos, _ = cached_encode(
            lambda: encode_texts_unified(model, tokenizer, txt_pos, device, args.batch_size),
            kind="text_pos@norm+raw", items=txt_pos, **cache_kw)
        t_neg, _ = cached_encode(
            lambda: encode_texts_unified(model, tokenizer, txt_neg, device, args.batch_size),
            kind="text_neg@norm+raw", items=txt_neg, **cache_kw)

        rec = analyse_concept(v_pres, v_abs, t_pos, t_neg, args.seed, use_bias)
        rec["object_name"] = obj
        records.append(rec)
        print(f"  [{obj:22s}] N={rec['n_pairs']:4d} | cos(wI,dI)={rec['cos_wI_dI']:+.3f} "
              f"cos(wT,dT)={rec['cos_wT_dT']:+.3f} | cos(dI,dT)={rec['cos_dI_dT']:+.3f} "
              f"cos(wI,wT)={rec['cos_wI_wT']:+.3f} | gamma={rec['gamma_measured']:+.5f}")

    if not records:
        raise SystemExit("No concept met --min_pairs; nothing to report.")

    out = pd.DataFrame(records)[
        ["object_name", "n_pairs", "cos_wI_dI", "cos_wT_dT", "cos_dI_dT", "cos_wI_wT",
         "norm_dI", "norm_dT", "a_I_along_w", "a_T_along_w",
         "gamma_measured", "gamma_pred_d", "gamma_pred_w", "cos_wI_dI_fold_std"]]
    csv_path = os.path.join(args.output_dir, "per_concept_directions.csv")
    out.to_csv(csv_path, index=False)

    g = out["gamma_measured"].values
    rho_d, p_d = spearmanr(g, out["cos_dI_dT"].values)
    rho_w, p_w = spearmanr(g, out["cos_wI_wT"].values)
    fit_d = pearsonr(g, out["gamma_pred_d"].values)
    fit_w = pearsonr(g, out["gamma_pred_w"].values)
    err_d = float(np.mean(np.abs(out["gamma_pred_d"] - g)) / (np.mean(np.abs(g)) + 1e-12))
    err_w = float(np.mean(np.abs(out["gamma_pred_w"] - g)) / (np.mean(np.abs(g)) + 1e-12))

    summary = {
        "n_concepts": int(len(out)),
        "total_pairs": int(out["n_pairs"].sum()),
        "macro_cos_wI_dI": float(out["cos_wI_dI"].mean()),
        "macro_cos_wT_dT": float(out["cos_wT_dT"].mean()),
        "macro_cos_dI_dT": float(out["cos_dI_dT"].mean()),
        "macro_cos_wI_wT": float(out["cos_wI_wT"].mean()),
        "spearman_gamma_vs_cos_dI_dT": {"rho": float(rho_d), "p": float(p_d)},
        "spearman_gamma_vs_cos_wI_wT": {"rho": float(rho_w), "p": float(p_w)},
        "additive_fit_mean_difference": {"pearson_r": float(fit_d[0]), "p": float(fit_d[1]),
                                         "rel_abs_error": err_d},
        "additive_fit_probe_normal": {"pearson_r": float(fit_w[0]), "p": float(fit_w[1]),
                                      "rel_abs_error": err_w},
        "criteria": {
            "G1_gamma_tracks_cos_dI_dT": bool(p_d < 0.05 and rho_d > 0),
            "G2_w_differs_from_d": bool(out["cos_wI_dI"].mean() < 0.95
                                        or out["cos_wT_dT"].mean() < 0.95),
            "G3_d_fit_no_worse_than_w": bool(err_d <= err_w),
        },
        "provenance": build_provenance(args, n_concepts=int(len(out)),
                                       n_pairs=int(out["n_pairs"].sum())),
    }
    with open(os.path.join(args.output_dir, "direction_vs_probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    render(out, args.output_dir)

    print("\n" + "=" * 72)
    print(f"  concepts {summary['n_concepts']} | pairs {summary['total_pairs']}")
    print(f"  macro cos(w_I,d_I) = {summary['macro_cos_wI_dI']:+.4f}")
    print(f"  macro cos(w_T,d_T) = {summary['macro_cos_wT_dT']:+.4f}")
    print(f"  macro cos(d_I,d_T) = {summary['macro_cos_dI_dT']:+.4f}"
          f"   vs cos(w_I,w_T) = {summary['macro_cos_wI_wT']:+.4f}")
    print(f"  gamma ~ cos(d_I,d_T): rho={rho_d:+.3f} p={p_d:.3g}"
          f"   |  gamma ~ cos(w_I,w_T): rho={rho_w:+.3f} p={p_w:.3g}")
    print(f"  additive fit  d: r={fit_d[0]:+.3f} rel.err={err_d:.3f}"
          f"   |  w: r={fit_w[0]:+.3f} rel.err={err_w:.3f}")
    for k, v in summary["criteria"].items():
        print(f"  {k:32s} : {'PASS' if v else 'FAIL'}")
    print("=" * 72)
    print(f"  Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
