"""Figure 2 of the camera-ready: 2x2 accuracy of the shared low-rank bilinear score vs rank r.

Layout follows the accepted version's figure (every power of two from r=1 to r=512 on a log2
axis, one curve for the low-rank transform, a star for LABCLIP, the 16.67% chance line) and the
plotting conventions of the reference conference papers in conference_paper/ (English axis and
legend text, framed legend inside the axes, no grid).

What changed from the accepted version is the protocol behind the numbers:
  solid line  : pooled accuracy on objects held out of training (GroupKFold over object_name,
                5 folds, Delta(S) hinge loss, warm start); the 95% CI from a concept-level cluster
                bootstrap is written to the data CSV but not drawn (removed for readability)
  dashed line : accuracy on the training objects (mean over the 5 folds)
  star        : full matrix trained with LABCLIP's contrastive recipe on the same 42 objects,
                scored on those training objects

Sources (defaults; the re-run passes its own with --run / --labclip):
  logs/evaluation/01_paper/2026-09-18_single_w_abswap_delta_ws_rank1to512/single_w_summary.json
  logs/evaluation/01_paper/2026-09-18_single_w_abswap_delta_ws_rank1to512/single_w_per_concept.csv
  logs/evaluation/01_paper/2026-09-10_labclip_rank_strategy_ext/rank_strategy_report.json (A, rank 512)
  --labclip also takes score_delta_s_for_w's delta_s_report.json; rank-512 post-hoc SVD of the full
  W is the full W itself, so both files carry the same quantity.

Run from the repo root:  python paper_figures/mk_fig2_rank_heldout.py [--run DIR] [--labclip JSON]
                                                                     [--out PNG] [--data_out CSV]
"""
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["axes.unicode_minus"] = False

RUN = "logs/evaluation/01_paper/2026-09-18_single_w_abswap_delta_ws_rank1to512"
LABCLIP = "logs/evaluation/01_paper/2026-09-10_labclip_rank_strategy_ext/rank_strategy_report.json"
OUT = "paper_figures/fig2_rank_heldout.png"
DATA_OUT = "paper_figures/fig2_rank_heldout_data.csv"

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
CHANCE = 100 / 6
B = 10000
SEED = 42
WIDTH_IN, HEIGHT_IN = 3.30, 1.26

RED, DARK, GREY = "#c44e52", "#8c2d2d", "#7f7f7f"
FS, FS_S = 6.5, 5.8


def cluster_bootstrap_pooled(sub, rng):
    """95% CI of the pooled accuracy, resampling objects with replacement."""
    n = sub["n_pairs"].to_numpy(float)
    k = sub["oof_acc_pct"].to_numpy(float) / 100 * n
    idx = rng.integers(0, len(sub), size=(B, len(sub)))
    pooled = 100 * k[idx].sum(1) / n[idx].sum(1)
    return np.percentile(pooled, [2.5, 97.5])


def load_labclip(path):
    rr = json.load(open(path))
    if "conditions" in rr:  # eval_labclip_rank_strategy_comparison
        return [x for x in rr["conditions"]["A_posthoc_svd"] if x["rank"] == 512][0]["delta_s"]
    return rr["labclip"]    # score_delta_s_for_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--labclip", default=LABCLIP)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--data_out", default=DATA_OUT)
    args = ap.parse_args()

    summ = json.load(open(f"{args.run}/single_w_summary.json"))
    fam = {f["family"]: f for f in summ["families"]}
    per = pd.read_csv(f"{args.run}/single_w_per_concept.csv")
    labclip = load_labclip(args.labclip)
    rng = np.random.default_rng(SEED)

    rows = []
    for name in ["identity"] + [f"lowrank_{r}" for r in RANKS]:
        sub = per[per.family == name]
        lo, hi = cluster_bootstrap_pooled(sub, rng)
        pooled = 100 * (sub.oof_acc_pct / 100 * sub.n_pairs).sum() / sub.n_pairs.sum()
        assert abs(pooled - fam[name]["oof_pooled_acc_pct"]) < 1e-6, name
        rows.append(dict(family=name, n_objects=len(sub), n_pairs=int(sub.n_pairs.sum()),
                         unseen_pooled=pooled, ci_lo=lo, ci_hi=hi,
                         train_foldmean=fam[name]["in_sample_acc_pct"], beats_chance=lo > CHANCE))
    rows.append(dict(family="labclip_recipe_full_train_objects", train_foldmean=labclip))
    df = pd.DataFrame(rows)
    df.to_csv(args.data_out, index=False)
    print(df.round(2).to_string(index=False))

    t = df.set_index("family")
    un = np.array([t.loc[f"lowrank_{r}", "unseen_pooled"] for r in RANKS])
    lo = np.array([t.loc[f"lowrank_{r}", "ci_lo"] for r in RANKS])
    hi = np.array([t.loc[f"lowrank_{r}", "ci_hi"] for r in RANKS])
    tr = np.array([t.loc[f"lowrank_{r}", "train_foldmean"] for r in RANKS])
    cos = t.loc["identity", "unseen_pooled"]

    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), dpi=400)
    ax = fig.add_axes([0.098, 0.215, 0.89, 0.765])

    ax.axhline(CHANCE, color=GREY, ls=(0, (4, 2.5)), lw=0.7, zorder=1)
    ax.text(40, CHANCE + 1.5, "Random 16.67%", fontsize=FS_S, color=GREY, va="bottom")

    h_tr, = ax.plot(RANKS, tr, ls=(0, (3.5, 1.8)), marker="o", ms=2.6, lw=0.9, color=RED,
                    mfc="white", mec=RED, mew=0.7, zorder=3, label="Training objects")
    h_un, = ax.plot(RANKS, un, "o-", ms=2.8, lw=1.2, color=RED, zorder=4, label="Unseen objects")
    h_lb, = ax.plot([512], [labclip], "*", ms=7, color=DARK, mec="white", mew=0.3, zorder=5,
                    label="LABCLIP (training objects)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.set_xticklabels([str(r) for r in RANKS])
    ax.minorticks_off()
    ax.set_xlim(0.8, 640)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", labelsize=FS, length=2, width=0.5, pad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.set_xlabel("Rank $r$", fontsize=FS, labelpad=1.0)
    ax.set_ylabel("2×2 accuracy (%)", fontsize=FS, labelpad=1.5)
    leg = ax.legend(handles=[h_un, h_tr, h_lb], fontsize=FS_S - 0.5, loc="center right",
                    bbox_to_anchor=(0.995, 0.57), frameon=True, fancybox=False,
                    edgecolor="#bbbbbb", framealpha=1.0, handlelength=1.8,
                    borderpad=0.35, labelspacing=0.22, borderaxespad=0.4)
    leg.get_frame().set_linewidth(0.5)

    fig.savefig(args.out, dpi=400, facecolor="white")
    print("saved:", args.out)


if __name__ == "__main__":
    main()
