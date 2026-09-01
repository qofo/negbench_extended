"""Is the delta-loss rank ladder actually better than the margin4 ladder?

review6.md (block B) correctly objects to reading rank 32's independent 95% CI
[26.84, 34.26] against margin4's 29.19% as a "win": 29.19% sits inside that CI, and
under the paper's own CI-overlap-means-undetermined standard that comparison is not
significant. But margin4 and delta were fit on the *same* GroupKFold concept split
and the *same* seed -- they are paired observations, not independent samples, so an
independent-CI comparison throws away most of the power the pairing buys. This
re-reads the same two already-completed runs (no new training) as a paired
per-concept bootstrap: d_c = acc_delta(c) - acc_margin4(c) for each of the 42
concepts, resampled with replacement over concepts.

Usage (from the repo root, after both ladders have been run):
    python -m benchmarks.src.evaluation.paired_bootstrap_loss_comparison \\
        --margin4_csv logs/evaluation/01_paper/2026-09-01_single_w_abswap/single_w_per_concept.csv \\
        --delta_csv logs/evaluation/01_paper/2026-09-01_single_w_abswap_deltaloss/single_w_per_concept.csv
"""
import argparse

import numpy as np
import pandas as pd

FAMILIES = ["lowrank_1", "lowrank_2", "lowrank_4", "lowrank_8", "lowrank_16", "lowrank_32", "full"]


def paired_bootstrap(margin4: pd.DataFrame, delta: pd.DataFrame, family: str,
                     seed: int = 42, n_boot: int = 20000):
    m = margin4[margin4.family == family].set_index("object_name")["oof_acc_pct"]
    d = delta[delta.family == family].set_index("object_name")["oof_acc_pct"]
    common = sorted(set(m.index) & set(d.index))
    diffs = np.array([d[c] - m[c] for c in common])  # delta - margin4, per concept
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.array([diffs[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return dict(family=family, n_concepts=n, mean_diff_pp=float(diffs.mean()),
               ci_lower_pp=float(lo), ci_upper_pp=float(hi),
               p_diff_le_zero=float((boot_means <= 0).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--margin4_csv", required=True)
    ap.add_argument("--delta_csv", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=20000)
    args = ap.parse_args()

    margin4 = pd.read_csv(args.margin4_csv)
    delta = pd.read_csv(args.delta_csv)

    print(f"{'family':12s} {'mean diff':>10s}  {'95% CI (paired)':>20s}  {'P(diff<=0)':>10s}   verdict")
    for family in FAMILIES:
        r = paired_bootstrap(margin4, delta, family, seed=args.seed, n_boot=args.n_boot)
        if r["ci_lower_pp"] > 0:
            verdict = "delta significantly better"
        elif r["ci_upper_pp"] < 0:
            verdict = "margin4 significantly better"
        else:
            verdict = "no significant difference"
        print(f"{family:12s} {r['mean_diff_pp']:+9.2f}pp  "
              f"[{r['ci_lower_pp']:+6.2f}, {r['ci_upper_pp']:+6.2f}]        "
              f"{100*r['p_diff_le_zero']:9.1f}%   {verdict}")


if __name__ == "__main__":
    main()
