"""
The 2x2 coefficients as what they are: four inner products between two vectors per modality.

Everything in RESULTS 6, 8 and 8-N is reported in the names alpha, beta, gamma. Those
names hide that the four coefficients are one object -- the 2x2 table of inner products
between the image side's two vectors and the text side's two vectors. Writing

    m_I = (v_pres + v_abs)/2     d_I = (v_pres - v_abs)/2
    m_T = (t_pos  + t_neg )/2    d_T = (t_pos  - t_neg )/2

gives, for u,v in {-1,+1} indexing image state and text polarity,

    S_uv = (m_I + u d_I) . (m_T + v d_T)
         = m_I.m_T  +  v (m_I.d_T)  +  u (d_I.m_T)  +  uv (d_I.d_T)

so the coefficient vector is exactly

    m_I.m_T = C (grand mean)    m_I.d_T = alpha    d_I.m_T = beta    d_I.d_T = gamma

and the success condition gamma > max(|alpha|,|beta|) says: **the d-by-d cell must beat
both m-by-d cells.** The m-by-m cell never enters, which is why an overall semantic match
cannot decide a negation.

Two facts make this coordinate worth reporting on its own. First, the split is orthogonal
and normalized: the encoders emit unit vectors, so m.d = 0 and ||m||^2 + ||d||^2 = 1 in
each modality. Each modality's pair of states is therefore a point on a circle, with ||d||
alone saying how far apart the two states are. Second, every cell factors as

    (cell) = ||x|| ||y|| cos(x, y)

so a small cell has two possible causes that the alpha/beta/gamma names cannot separate:
short vectors, or a bad angle. This script reports the magnitudes, the cosines and the
products side by side for the same ten conditions as E5 (nine models plus PeakPatch's
frozen-tower ECN), so a reader can see which cause is operating in each.

Aggregation is pooled over pairs, with the concept macro mean carried alongside; RESULTS
8.9's warning box records a case where the two disagree, so both are written rather than
one being chosen.

Usage (from the repo root):
    python -m benchmarks.src.evaluation.eval_dm_factor_table \\
        --restrict_objects logs/evaluation/01_paper/2026-08-28_e2_hadamard_decomposition/e2_per_concept_decomposition.csv \\
        --use_cache --peakpatch_root PeakPatch \\
        --output_dir logs/evaluation/01_paper/2026-09-16_dm_factor_table
"""
import os
import json
import argparse
from typing import Dict

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.cli import (
        add_run_args, add_data_args, add_cache_args, add_restriction_args,
        add_concept_args,
    )
    from benchmarks.src.analysis.feature_cache import build_provenance, load_object_restriction
    from benchmarks.src.analysis.config import set_seed, coerce_bool_column
    from benchmarks.src.evaluation.eval_target_and_aim import MODELS, load_pairs
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.model_loader import load_clip_for_eval
    from analysis.cli import (
        add_run_args, add_data_args, add_cache_args, add_restriction_args,
        add_concept_args,
    )
    from analysis.feature_cache import build_provenance, load_object_restriction
    from analysis.config import set_seed, coerce_bool_column
    from evaluation.eval_target_and_aim import MODELS, load_pairs

EPS = 1e-12
CELLS = ["mm", "md", "dm", "dd"]
CELL_LABEL = {"mm": "m_I.m_T (=C)", "md": "m_I.d_T (=alpha)",
              "dm": "d_I.m_T (=beta)", "dd": "d_I.d_T (=gamma)"}


def factor_quantities(m_I, d_I, m_T, d_T) -> Dict[str, np.ndarray]:
    """Per-pair magnitudes, cosines and the four inner products."""
    nrm = lambda x: np.linalg.norm(x, axis=-1)
    dot = lambda a, b: np.sum(a * b, axis=-1)

    n_mI, n_dI, n_mT, n_dT = nrm(m_I), nrm(d_I), nrm(m_T), nrm(d_T)
    q = dict(norm_m_I=n_mI, norm_d_I=n_dI, norm_m_T=n_mT, norm_d_T=n_dT,
             mm=dot(m_I, m_T), md=dot(m_I, d_T), dm=dot(d_I, m_T), dd=dot(d_I, d_T))

    for cell, (a, b) in dict(mm=(n_mI, n_mT), md=(n_mI, n_dT),
                             dm=(n_dI, n_mT), dd=(n_dI, n_dT)).items():
        q[f"cos_{cell}"] = q[cell] / np.maximum(a * b, EPS)
        q[f"mag_{cell}"] = a * b

    q["ratio_d_over_m_I"] = n_dI / np.maximum(n_mI, EPS)
    q["ratio_d_over_m_T"] = n_dT / np.maximum(n_mT, EPS)

    # The split is orthogonal and lives on the unit circle only if the encoder emits
    # unit vectors. Reported rather than assumed: a nonzero residual invalidates the
    # "||d|| alone measures state separation" reading for that condition.
    q["orth_resid_I"] = np.abs(dot(m_I, d_I))
    q["orth_resid_T"] = np.abs(dot(m_T, d_T))
    q["unit_resid_I"] = np.abs(n_mI ** 2 + n_dI ** 2 - 1.0)
    q["unit_resid_T"] = np.abs(n_mT ** 2 + n_dT ** 2 - 1.0)

    q["delta"] = 2 * q["dd"] - 2 * np.maximum(np.abs(q["md"]), np.abs(q["dm"]))
    q["joint"] = (q["delta"] > 0).astype(float)
    return q


def per_concept_frame(name: str, q: Dict[str, np.ndarray], groups: np.ndarray) -> pd.DataFrame:
    """
    One row per concept, so the pooled means can be checked for sign cancellation.

    A pooled mean gamma is positive if half the concepts sit at +x and the other half
    at -x/2; only the distribution says whether the coefficient is a property of the
    model or of the averaging. The ratio is built the way RESULTS 8-N builds it --
    signed mean gamma over the larger of the two mean absolute main effects -- so a
    concept row here is comparable to a model row there.
    """
    out = []
    for g in np.unique(groups):
        s = groups == g
        gam = float(q["dd"][s].mean())
        a = float(np.abs(q["md"][s]).mean())
        b = float(np.abs(q["dm"][s]).mean())
        out.append(dict(name=name, object_name=g, n_pairs=int(s.sum()),
                        gamma=gam, abs_alpha=a, abs_beta=b,
                        ratio=gam / max(a, b, EPS),
                        cos_dd=float(q["cos_dd"][s].mean()),
                        norm_d_I=float(q["norm_d_I"][s].mean()),
                        norm_d_T=float(q["norm_d_T"][s].mean()),
                        joint_pct=float(q["joint"][s].mean() * 100)))
    return pd.DataFrame(out)


def cluster_bootstrap(q: Dict[str, np.ndarray], groups: np.ndarray,
                      n_boot: int, rng) -> Dict[str, float]:
    """
    Concept-clustered percentile CIs for the quantities the conclusions rest on.

    Pairs of one concept share a caption pair and a base scene, so they are not
    independent draws; resampling pairs would understate the spread. Concepts are the
    cluster, as in the rest of this project's interval estimates.

    The interval on the *signed* mean of m_I.d_T is the one that carries an argument:
    a model whose text polarity direction is genuinely orthogonal to the image mean
    has an interval covering zero, whereas one that merely averages two large opposite
    populations to near-zero does not have to.
    """
    keys = np.unique(groups)
    idx = {k: np.where(groups == k)[0] for k in keys}
    draws = {k: [] for k in ("cos_dd", "md", "dd", "ratio", "norm_d_I")}
    for _ in range(n_boot):
        take = np.concatenate([idx[k] for k in rng.choice(keys, len(keys), True)])
        draws["cos_dd"].append(q["cos_dd"][take].mean())
        draws["md"].append(q["md"][take].mean())
        draws["dd"].append(q["dd"][take].mean())
        draws["norm_d_I"].append(q["norm_d_I"][take].mean())
        draws["ratio"].append(q["dd"][take].mean() / max(
            np.abs(q["md"][take]).mean(), np.abs(q["dm"][take]).mean(), EPS))
    out = {}
    for k, v in draws.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[f"{k}_lo"], out[f"{k}_hi"] = float(lo), float(hi)
    out["md_ci_covers_zero"] = bool(out["md_lo"] <= 0.0 <= out["md_hi"])
    return out


def summarize(name: str, q: Dict[str, np.ndarray], groups: np.ndarray,
              per_concept: pd.DataFrame) -> Dict:
    """
    Pooled means, plus every reading of the success ratio that could be meant by it.

    The ratio gamma / max(|alpha|,|beta|) has three inequivalent definitions and they
    do not agree, so all three are written rather than one being silently chosen:

      ratio_pooled       mean(gamma) / max(mean|alpha|, mean|beta|) over pairs. This is
                         the headline column and the one comparable to RESULTS 8.9.
      ratio_macro        the same built from concept means, then averaged over concepts.
      ratio_pairwise     mean over pairs of the per-pair ratio. Dominated by pairs whose
                         main effects are near zero, so it is reported but not used.

    ``joint_pct`` is the only definition that needs no choice: it is the fraction of
    pairs satisfying gamma_p > max(|alpha_p|,|beta_p|), which is exactly the 2x2
    criterion evaluated pair by pair.
    """
    row = dict(name=name, n_pairs=int(len(q["dd"])), n_concepts=int(len(np.unique(groups))))
    for k, v in q.items():
        row[k] = float(np.mean(v))
    for cell in CELLS:
        row[f"abs_{cell}"] = float(np.mean(np.abs(q[cell])))
        row[f"{cell}_macro"] = float(np.mean(
            [q[cell][groups == g].mean() for g in np.unique(groups)]))

    row["joint_pct"] = float(np.mean(q["joint"]) * 100)
    row["ratio_pooled"] = row["dd"] / max(row["abs_md"], row["abs_dm"], EPS)
    row["ratio_macro"] = float(per_concept["ratio"].mean())
    pairwise = q["dd"] / np.maximum(np.maximum(np.abs(q["md"]), np.abs(q["dm"])), EPS)
    row["ratio_pairwise"] = float(pairwise.mean())
    row["ratio_pairwise_median"] = float(np.median(pairwise))

    row["n_concepts_gamma_pos"] = int((per_concept["gamma"] > 0).sum())
    row["n_concepts_ratio_gt_1"] = int((per_concept["ratio"] > 1).sum())
    row["concept_ratio_min"] = float(per_concept["ratio"].min())
    row["concept_ratio_max"] = float(per_concept["ratio"].max())
    row["pct_pairs_gamma_pos"] = float((q["dd"] > 0).mean() * 100)
    return row


def make_figure(d: pd.DataFrame, path: str) -> None:
    x = np.arange(len(d))
    w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    for i, cell in enumerate(CELLS):
        axes[0].bar(x + (i - 1.5) * w, d[f"abs_{cell}"], w, label=CELL_LABEL[cell])
        axes[1].bar(x + (i - 1.5) * w, d[f"cos_{cell}"].abs(), w, label=CELL_LABEL[cell])

    axes[0].set_yscale("log")
    axes[0].set_ylabel("|inner product|")
    axes[0].set_title("The four cells: d-by-d is the smallest everywhere")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("|cosine|")
    axes[1].set_title("The same cells as angles: magnitude is not the whole story")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(d["name"], rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    df = coerce_bool_column(pd.read_csv(args.csv_path), "object_in_image")
    restriction = load_object_restriction(args.restrict_objects)
    objects = sorted(df["object_name"].unique())
    if restriction:
        objects = [o for o in objects if o in set(restriction)]
    print(f"\n  Concepts: {len(objects)}\n")

    rng = np.random.RandomState(args.seed)
    rows, concept_frames = [], []

    def record(name, tower, q, groups):
        pc = per_concept_frame(name, q, groups)
        concept_frames.append(pc)
        r = summarize(name, q, groups, pc)
        r.update(cluster_bootstrap(q, groups, args.n_boot, rng))
        r["image_tower"] = tower
        rows.append(r)
        print(f"    ||d_I||={r['norm_d_I']:.4f} ||d_T||={r['norm_d_T']:.4f}  "
              f"cos(d_I,d_T)={r['cos_dd']:+.4f} [{r['cos_dd_lo']:+.4f},{r['cos_dd_hi']:+.4f}]  "
              f"ratio={r['ratio_pooled']:.3f} [{r['ratio_lo']:.3f},{r['ratio_hi']:.3f}]  "
              f"alpha CI covers 0: {r['md_ci_covers_zero']}  ratio>1: "
              f"{r['n_concepts_ratio_gt_1']}/{r['n_concepts']}")

    for name, arch, pretrained, is_intervention in MODELS:
        print(f"  [{name}]")
        model, preprocess, tokenizer = load_clip_for_eval(arch, pretrained, device)
        cache_kw = dict(model=arch, pretrained=pretrained,
                        enabled=args.use_cache, cache_dir=args.cache_dir)
        m_I, d_I, m_T, d_T, groups = load_pairs(
            df, objects, model, preprocess, tokenizer, device, args, cache_kw)
        record(name, "fine-tuned" if is_intervention else "pretrained",
               factor_quantities(m_I, d_I, m_T, d_T), groups)
        del model
        torch.cuda.empty_cache()

    if args.peakpatch_root:
        try:
            from benchmarks.src.evaluation.eval_peakpatch_2x2_audit import load_peakpatch
        except ImportError:
            from analysis.import_compat import reraise_unless_standalone
            reraise_unless_standalone()
            from evaluation.eval_peakpatch_2x2_audit import load_peakpatch
        print("\n  [PeakPatch ECN (frozen ViT-B/32)]")
        pp = load_peakpatch(args.peakpatch_root, device)
        model, preprocess, tokenizer = load_clip_for_eval("ViT-B-32", "openai", device)
        cache_kw = dict(model="ViT-B-32", pretrained="openai",
                        enabled=args.use_cache, cache_dir=args.cache_dir)
        m_I, d_I, m_T, d_T, groups = load_pairs(
            df, objects, model, preprocess, tokenizer, device, args, cache_kw,
            ecn=pp["ec"], pp=pp)
        record("PeakPatch ECN", "frozen", factor_quantities(m_I, d_I, m_T, d_T), groups)

    if args.include_random_init:
        # Separates geometry produced by contrastive training from geometry the
        # architecture already has. The cache is bypassed deliberately: its key carries
        # only (arch, "None", items), so a random-init run from another script -- whose
        # RNG consumption before model creation differs -- would collide with this one
        # and hand over arrays from different weights.
        print("\n  [Random init (control)]")
        set_seed(args.seed)
        model, preprocess, tokenizer = load_clip_for_eval("ViT-B-32", None, device)
        m_I, d_I, m_T, d_T, groups = load_pairs(
            df, objects, model, preprocess, tokenizer, device, args,
            dict(model="ViT-B-32", pretrained="None", enabled=False,
                 cache_dir=args.cache_dir))
        record("Random init", "random", factor_quantities(m_I, d_I, m_T, d_T), groups)
        del model
        torch.cuda.empty_cache()

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(args.output_dir, "dm_factor_table.csv"), index=False)
    pc = pd.concat(concept_frames, ignore_index=True)
    pc.to_csv(os.path.join(args.output_dir, "dm_factor_per_concept.csv"), index=False)
    make_figure(d, os.path.join(args.output_dir, "fig_dm_factor_table.png"))

    print("\n" + "=" * 108)
    print(f"  {'model':<20}{'|m_I|':>8}{'|d_I|':>8}{'|m_T|':>8}{'|d_T|':>8}"
          f"{'m.m':>10}{'m_I.d_T':>10}{'d_I.m_T':>10}{'d_I.d_T':>10}{'dd/max':>9}{'2x2%':>7}")
    for _, r in d.iterrows():
        print(f"  {r['name']:<20}{r['norm_m_I']:8.4f}{r['norm_d_I']:8.4f}"
              f"{r['norm_m_T']:8.4f}{r['norm_d_T']:8.4f}{r['mm']:10.5f}"
              f"{r['md']:+10.5f}{r['dm']:+10.5f}{r['dd']:+10.5f}"
              f"{r['ratio_pooled']:9.3f}{r['joint_pct']:7.2f}")

    print("\n  --- the same four cells as cosines ---")
    print(f"  {'model':<20}{'cos(m,m)':>10}{'cos(m_I,d_T)':>14}{'cos(d_I,m_T)':>14}"
          f"{'cos(d_I,d_T)':>14}{'|d_I|/|m_I|':>13}{'|d_T|/|m_T|':>13}")
    for _, r in d.iterrows():
        print(f"  {r['name']:<20}{r['cos_mm']:10.4f}{r['cos_md']:+14.4f}"
              f"{r['cos_dm']:+14.4f}{r['cos_dd']:+14.4f}"
              f"{r['ratio_d_over_m_I']:13.4f}{r['ratio_d_over_m_T']:13.4f}")

    print("\n  --- the success ratio under its three definitions, and the concept spread ---")
    print(f"  {'model':<20}{'pooled':>9}{'macro':>9}{'pairwise':>10}{'gamma>0':>9}"
          f"{'ratio>1':>9}{'min R_O':>9}{'max R_O':>9}{'pairs g>0 %':>13}")
    for _, r in d.iterrows():
        print(f"  {r['name']:<20}{r['ratio_pooled']:9.3f}{r['ratio_macro']:9.3f}"
              f"{r['ratio_pairwise']:10.3f}"
              f"{str(int(r['n_concepts_gamma_pos'])) + '/' + str(int(r['n_concepts'])):>9}"
              f"{str(int(r['n_concepts_ratio_gt_1'])) + '/' + str(int(r['n_concepts'])):>9}"
              f"{r['concept_ratio_min']:9.3f}{r['concept_ratio_max']:9.3f}"
              f"{r['pct_pairs_gamma_pos']:13.2f}")

    print("\n  --- concept-clustered 95% CIs (n_boot = "
          f"{args.n_boot}) ---")
    print(f"  {'model':<20}{'cos(d_I,d_T)':>26}{'ratio':>22}{'alpha (m_I.d_T)':>30}{'0 in CI':>9}")
    for _, r in d.iterrows():
        print(f"  {r['name']:<20}"
              f"{r['cos_dd']:+.4f} [{r['cos_dd_lo']:+.4f},{r['cos_dd_hi']:+.4f}]".rjust(26)
              + f"{r['ratio_pooled']:.3f} [{r['ratio_lo']:.3f},{r['ratio_hi']:.3f}]".rjust(22)
              + f"{r['md']:+.5f} [{r['md_lo']:+.5f},{r['md_hi']:+.5f}]".rjust(30)
              + f"{str(bool(r['md_ci_covers_zero'])):>9}")

    worst = dict(orth_I=float(d["orth_resid_I"].max()), orth_T=float(d["orth_resid_T"].max()),
                 unit_I=float(d["unit_resid_I"].max()), unit_T=float(d["unit_resid_T"].max()))
    print(f"\n  orthogonal/unit-split residuals (max over conditions): "
          f"m.d = {max(worst['orth_I'], worst['orth_T']):.2e}, "
          f"||m||^2+||d||^2-1 = {max(worst['unit_I'], worst['unit_T']):.2e}")

    with open(os.path.join(args.output_dir, "dm_factor_table.json"), "w") as f:
        json.dump(dict(rows=d.to_dict(orient="records"), split_residuals=worst,
                       provenance=build_provenance(args)), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {args.output_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(ap, "logs/evaluation/dm_factor_table", seed=42, batch_size=128)
    add_data_args(ap, csv_path="benchmarks/data/images/beaf_counterfactual_6col.csv",
                  image_root="benchmarks/data/images")
    add_cache_args(ap)
    add_restriction_args(ap, "Concept set to share with the e2 runs")
    add_concept_args(ap, help_text="Minimum counterfactual pairs per object")
    ap.add_argument("--peakpatch_root", type=str, default="PeakPatch")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--include_random_init", action="store_true", default=False,
                    help="Add a randomly initialized ViT-B/32 control, which separates "
                         "geometry produced by training from geometry the architecture has")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
