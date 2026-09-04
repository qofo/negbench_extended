"""Place the three PeakPatch conditions on the two nine-model laws.

RESULTS 8-R fits  R@1 drop ~ signed beta  over nine models (r=+0.835); RESULTS 8-N.8 fits
the mirror  MCQ (positive - negative) gap ~ signed alpha  (r=+0.970). Both were fit across
independently trained models. PeakPatch supplies what that sweep cannot: an intervention
that moves the coefficients on a single frozen backbone, so the laws can be tested as
within-model claims rather than between-model correlations.

Estimator note -- this is the part that is easy to get wrong. mk_retrieval_data.py and
mk_external_data.py aggregate a model as the *mean over concepts* of the per-concept means
(e2_per_concept_decomposition.csv), so a concept with 80 pairs and one with 20 count the
same. The 2x2 audit's summary JSON instead reports the pooled mean over all 1,357 pairs.
For beta the two differ by a factor of two (3.664 vs 1.728 x1e-3), which is more than the
whole PeakPatch effect, so the per-pair CSV is re-aggregated here the nine-model way. The
check that this is right: the cosine condition then reproduces the ViT-B/32 (OpenAI) row of
both nine-model tables exactly (alpha 6.395, beta 3.664, gamma 0.554).

Run from the repo root:
    python paper_figures/mk_peakpatch_law_residuals.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

AUDIT = "logs/evaluation/01_paper/2026-09-02_peakpatch_2x2_audit/peakpatch_per_pair.csv"
GALLERY = "logs/evaluation/01_paper/2026-09-04_peakpatch_gallery_retrieval/peakpatch_gallery_retrieval.json"
COND = ("cosine", "ecn", "ecn+scn")
LABEL = {"cosine": "cosine (=ViT-B/32)", "ecn": "ECN", "ecn+scn": "ECN+SCN"}


def peakpatch_coefficients():
    """Concept-mean-of-means, x1e3 -- the aggregation the nine-model sweep uses."""
    per = pd.read_csv(AUDIT)
    cm = per.groupby(["condition", "object_name"]).mean(numeric_only=True)
    return cm.groupby("condition").mean() * 1e3


def fit(x, y):
    r, p = pearsonr(x, y)
    m, b = np.polyfit(x, y, 1)
    return m, b, (y - (m * x + b)).std(ddof=2), r, p


def report(name, x, y, names, pp_x, pp_y, unit):
    m, b, sd, r, p = fit(x, y)
    print(f"\n=== {name}:  y = {m:+.4f}x {b:+.3f}   r={r:+.3f}  p={p:.2e}  "
          f"residual SD={sd:.3f}{unit}  n={len(x)} ===")
    print("  nine models:")
    for n, xi, yi in zip(names, x, y):
        pr = m * xi + b
        print(f"    {n:20s} x={xi:+8.3f}  pred={pr:8.2f}  obs={yi:7.2f}  "
              f"resid={yi - pr:+7.2f} ({(yi - pr) / sd:+5.2f} SD)")
    print("  PeakPatch:")
    rows = []
    for c in COND:
        pr = m * pp_x[c] + b
        rows.append(dict(condition=c, x=pp_x[c], pred=pr, obs=pp_y[c],
                         resid=pp_y[c] - pr, sd=(pp_y[c] - pr) / sd))
        print(f"    {c:20s} x={pp_x[c]:+8.3f}  pred={pr:8.2f}  obs={pp_y[c]:7.2f}  "
              f"resid={pp_y[c] - pr:+7.2f} ({(pp_y[c] - pr) / sd:+5.2f} SD)")
    return m, b, sd, r, pd.DataFrame(rows)


def band(ax, x, m, b, sd, color="#888"):
    xs = np.linspace(x.min(), x.max(), 2)
    ax.plot(xs, m * xs + b, "-", color=color, lw=1.2, zorder=1)
    ax.fill_between(xs, m * xs + b - sd, m * xs + b + sd, color=color, alpha=0.15, zorder=0)


def main():
    cm = peakpatch_coefficients()
    J = json.load(open(GALLERY))
    R = pd.read_csv("paper_figures/fig_retrieval_data.csv")
    E = pd.read_csv("paper_figures/fig_external_data.csv")
    E["gap"] = E.coco_pos - E.coco_neg

    # harness check: the cosine condition must be the ViT-B/32 (OpenAI) row of both tables
    ref = R[R.model == "ViT-B/32 (OpenAI)"].iloc[0]
    for field, got in (("beta_signed", cm.loc["cosine", "beta"]),
                       ("alpha_signed", cm.loc["cosine", "alpha"]),
                       ("gamma", cm.loc["cosine", "gamma"])):
        if abs(ref[field] - got) > 1e-6:
            raise SystemExit(f"[harness] cosine {field} = {got:.6f} but the nine-model table "
                             f"has {ref[field]:.6f}; the estimators no longer agree.")
    print("[harness] cosine condition reproduces the ViT-B/32 (OpenAI) coefficients exactly")

    mb, bb, sdb, rb, dfb = report(
        "beta law: R@1 drop (pp) ~ signed beta", R.beta_signed.values, R.drop1.values,
        R.model.tolist(), {c: cm.loc[c, "beta"] for c in COND},
        {c: J["drops"][c]["R@1"] for c in COND}, "pp")
    ma, ba, sda, ra, dfa = report(
        "alpha law: MCQ pos-neg gap (pp) ~ signed alpha", E.alpha_signed.values, E.gap.values,
        E.model.tolist(), {c: cm.loc[c, "alpha"] for c in COND},
        {"cosine": 62.30, "ecn": 56.95, "ecn+scn": 40.77}, "pp")

    out = pd.concat([dfb.assign(law="beta->R@1 drop"), dfa.assign(law="alpha->MCQ gap")])
    out.to_csv("paper_figures/fig_peakpatch_law_residuals.csv", index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, x, y, m, b, sd, xl, yl, ttl in (
            (a1, R.beta_signed.values, R.drop1.values, mb, bb, sdb,
             r"signed $\beta\ \times10^{3}$", "R@1 drop (pp)",
             rf"$\beta$ law   $r=+{rb:.3f}$"),
            (a2, E.alpha_signed.values, E.gap.values, ma, ba, sda,
             r"signed $\alpha\ \times10^{3}$", "MCQ positive $-$ negative gap (pp)",
             rf"$\alpha$ law   $r=+{ra:.3f}$")):
        band(ax, x, m, b, sd)
        ax.scatter(x, y, s=28, c="#3b6ea5", zorder=3, label="nine models")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl, fontsize=11)
        ax.axhline(0, color="#ccc", lw=0.6, zorder=0)

    pb = [(cm.loc[c, "beta"], J["drops"][c]["R@1"]) for c in COND]
    pa = [(cm.loc[c, "alpha"], g) for c, g in zip(COND, (62.30, 56.95, 40.77))]
    for ax, pts, m, b in ((a1, pb, mb, bb), (a2, pa, ma, ba)):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", color="#c0392b", ms=6, lw=1.2, zorder=4, label="PeakPatch")
        for (xi, yi), c in zip(pts, COND):
            ax.annotate(LABEL[c], (xi, yi), textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color="#c0392b")
            ax.plot([xi, xi], [yi, m * xi + b], ":", color="#c0392b", lw=1, zorder=2)
        ax.legend(fontsize=8, loc="best")

    # ECN+SCN sits at alpha = -92.9 -> predicted gap -195; keep the nine models readable
    a2.set_xlim(-32, 20); a2.set_ylim(-35, 80)
    a2.annotate(f"ECN+SCN: $\\alpha={pa[2][0]:.1f}$, pred ${ma * pa[2][0] + ba:.0f}$, "
                f"obs $+{pa[2][1]:.1f}$\n(off scale, ${(pa[2][1] - (ma * pa[2][0] + ba)) / sda:+.0f}$ SD)",
                xy=(-31, -30), fontsize=8, color="#c0392b")
    fig.suptitle("Both laws are between-model correlations, not laws about interventions",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("paper_figures/fig_peakpatch_law_residuals.png", dpi=180)
    print("\nsaved: paper_figures/fig_peakpatch_law_residuals.png")
    print("saved: paper_figures/fig_peakpatch_law_residuals.csv")


if __name__ == "__main__":
    main()
