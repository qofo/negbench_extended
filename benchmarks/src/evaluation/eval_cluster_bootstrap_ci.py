"""
Concept-clustered bootstrap intervals for the 2x2 decision accuracies.

The pairs are not independent: every concept contributes many pairs that share a
caption template, and pairs within a concept share scenes. Reporting a pooled
percentage against a chance line therefore overstates the evidence, which is what a
reviewer objected to. This script resamples *concepts* with replacement -- the unit
the dependence lives at -- and reports percentile intervals for every claim the
manuscript makes against chance.

Two claims are covered:

  1. The nine-model audit: "no model clears chance on any of the three decisions."
     Read from each model's e2 per-pair decomposition.
  2. The PeakPatch conditions: cosine / ECN / ECN+SCN on the same 33 concepts.

Usage:
    python -m benchmarks.src.evaluation.eval_cluster_bootstrap_ci \\
        --output_dir logs/evaluation/01_paper/2026-09-03_cluster_bootstrap_ci
"""
import os
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd

P = "logs/evaluation/01_paper/"
NINE = [("ViT-B/32 (OpenAI)", P + "2026-08-28_r6_main_effect_ablation_33concepts/"),
        ("ViT-B/16", P + "2026-08-30_r8_vitb16/e2_hadamard_decomposition/"),
        ("ViT-L/14", P + "2026-08-30_r8_vitl14/e2_hadamard_decomposition/"),
        ("LAION-2B", P + "2026-08-30_negfam_laion2b/e2/"),
        ("SigLIP B/16", P + "2026-08-30_negfam_siglip_b16/e2/"),
        ("CoN-CLIP", P + "2026-08-30_negfam_conclip/e2/"),
        ("NegCLIP", P + "2026-08-30_negfam_negclip/e2/"),
        ("NegCLIP-NegFull", P + "2026-08-30_negfam_negclip_negfull/e2/"),
        ("CLIP-NegFull", P + "2026-08-30_negfam_clip_negfull/e2/")]
PEAKPATCH = P + "2026-09-02_peakpatch_2x2_audit/peakpatch_per_pair.csv"

CHANCE = {"caption": 25.0, "image": 25.0, "group": 100 / 6}


def cluster_bootstrap(flags: np.ndarray, groups: np.ndarray, n_boot: int,
                      rng: np.random.RandomState) -> Dict[str, float]:
    """
    Percentile CI for a proportion, resampling whole concepts with replacement.

    Returns the pooled point estimate alongside the interval; the point estimate is
    the pooled rate, not the bootstrap mean, so it matches the manuscript's tables.
    """
    keys = np.unique(groups)
    by_key = {k: flags[groups == k] for k in keys}
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        draws[b] = np.concatenate([by_key[k] for k in pick]).mean() * 100
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(point=float(flags.mean() * 100), lo=float(lo), hi=float(hi),
                n_concepts=len(keys), n_pairs=len(flags))


def report(name: str, df: pd.DataFrame, n_boot: int, rng, rows: List[Dict]):
    for crit in ("caption", "image", "group"):
        if crit == "caption":
            f = (df.gamma > df.abs_alpha).to_numpy()
        elif crit == "image":
            f = (df.gamma > df.abs_beta).to_numpy()
        else:
            f = (df.gamma > np.maximum(df.abs_alpha, df.abs_beta)).to_numpy()
        ci = cluster_bootstrap(f, df.object_name.to_numpy(), n_boot, rng)
        ch = CHANCE[crit]
        verdict = "below" if ci["hi"] < ch else ("above" if ci["lo"] > ch else "spans")
        rows.append(dict(subject=name, criterion=crit, chance=ch, verdict=verdict, **ci))
        print(f"    {name:<20}{crit:<9}{ci['point']:6.2f}%  "
              f"95% CI [{ci['lo']:5.2f}, {ci['hi']:5.2f}]  vs {ch:5.2f}  -> {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", type=str,
                    default="logs/evaluation/01_paper/2026-09-03_cluster_bootstrap_ci")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    rows: List[Dict] = []

    print(f"\n  Concept-clustered bootstrap, {args.n_boot} resamples, seed {args.seed}")
    print("\n  [1] Nine-model audit")
    for name, d in NINE:
        report(name, pd.read_csv(d + "e2_per_pair_decomposition.csv"), args.n_boot, rng, rows)

    print("\n  [2] PeakPatch conditions")
    pp = pd.read_csv(PEAKPATCH)
    for cond in ("cosine", "ecn", "ecn+scn"):
        report(cond, pp[pp.condition == cond].reset_index(drop=True), args.n_boot, rng, rows)

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(args.output_dir, "cluster_bootstrap_ci.csv"), index=False)

    nine = d[d.subject.isin([n for n, _ in NINE])]
    print("\n  --- claims the intervals support ---")
    for crit in ("caption", "image", "group"):
        s = nine[nine.criterion == crit]
        print(f"    nine models, {crit:<8}: CI strictly below chance in "
              f"{(s.verdict == 'below').sum()}/{len(s)}")
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(dict(n_boot=args.n_boot, seed=args.seed,
                       rows=d.to_dict(orient="records")), f, indent=2)
    print(f"\n  Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
