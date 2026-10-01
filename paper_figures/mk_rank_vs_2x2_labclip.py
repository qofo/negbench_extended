"""rank vs. 2x2 matching accuracy -- the capacity sweep, with LABCLIP's own recipe overlaid.

Two ways to spend a rank-r bilinear budget W in  s(v,t) = v^T W t :

  (1) train W directly on the 2x2 block task (four-term margin-ranking loss),
      scored on concepts it never saw (concept GroupKFold OOF).
      Source: logs/.../2026-09-01_single_w_abswap_deltaloss  -- the numbers the
      manuscript quotes (identity 4.03 -> lowrank_32 32.58 -> full 28.99).

  (2) train W with LABCLIP's *official* recipe: a plain low-rank W = U V^T, no
      identity term, symmetric in-batch InfoNCE over image/true-caption pairs,
      same loss/opt as train_labclip_official_recipe.py, rank fixed before
      training.  Source: 2026-09-10_labclip_rank_strategy_ext condition B,
      swept r = 1..512.  Scored in-sample on all 42 concepts (a ceiling; OOF
      could only be lower).

The full-rank LABCLIP W from the official recipe (identity-init nn.Linear(512,512),
same InfoNCE) is drawn as a single separate marker at r = 512 -- at r >= 512 the
low-rank family W = UV^T spans every 512x512 matrix, so it is the *same
hypothesis class* as LABCLIP, but a different trained member (random vs identity
init, and here a different loss for curve 1).

Run from the repo root:
    python paper_figures/mk_rank_vs_2x2_labclip.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

DELTA = "logs/evaluation/01_paper/2026-09-01_single_w_abswap_deltaloss"
RANKSTRAT = "logs/evaluation/01_paper/2026-09-10_labclip_rank_strategy_ext/rank_strategy_report.json"
OUT = "paper_figures/fig_rank_vs_2x2_labclip.png"
CHANCE = 100 / 6
FULL_RANK = 512
SEED = 42


def pooled(fam, family):
    g = fam[fam.family == family]
    return float((g.oof_acc_pct * g.n_test_pairs).sum() / g.n_test_pairs.sum())


def pooled_ci(con, family, n_boot=20000):
    g = con[con.family == family]
    a, w = g.oof_acc_pct.to_numpy(), g.n_pairs.to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    draws = (a[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


fam = pd.read_csv(f"{DELTA}/single_w_per_family.csv")
con = pd.read_csv(f"{DELTA}/single_w_per_concept.csv")

ranks_d = [1, 2, 4, 8, 16, 32]
acc_d = np.array([pooled(fam, f"lowrank_{r}") for r in ranks_d])
ci_d = np.array([pooled_ci(con, f"lowrank_{r}") for r in ranks_d])
acc_cos = pooled(fam, "identity")
acc_full_d = pooled(fam, "full")
ci_full_d = pooled_ci(con, "full")

rep = json.load(open(RANKSTRAT))
B = sorted(rep["conditions"]["B_plain_lowrank"], key=lambda r: r["rank"])
ranks_b = [r["rank"] for r in B]
acc_b = [r["delta_s"] for r in B]
labclip_full = [r for r in rep["conditions"]["A_posthoc_svd"] if r["rank"] == 512][0]["delta_s"]

fig, ax = plt.subplots(figsize=(7.8, 5.0))
fig.subplots_adjust(left=0.095, right=0.995, top=0.965, bottom=0.115)

ax.axvspan(360, 780, color="#f2f2f2", zorder=0)
ax.text(FULL_RANK, 40.2, "완전 계수", fontsize=9, color="#888", ha="center")

ax.axhline(CHANCE, color="#888", ls="--", lw=1.2, zorder=1)
ax.text(0.86, CHANCE + 0.7, f"우연 기준 {CHANCE:.2f}%", fontsize=8.8, color="#666")
ax.axhline(acc_cos, color="#333", ls="-", lw=1.3, zorder=1)
ax.text(0.86, acc_cos + 0.8, f"코사인 ($W=I$) {acc_cos:.2f}%", fontsize=8.8, color="#333")

# curve 1 -- block-task margin loss, concept OOF
ax.errorbar(ranks_d, acc_d, yerr=[acc_d - ci_d[:, 0], ci_d[:, 1] - acc_d],
            fmt="o-", color="#4c72b0", lw=1.9, ms=6.5, capsize=3, zorder=5,
            label="① 2×2 블록 과제 직접 학습 (마진 손실, 미학습 개념 OOF)")
ax.errorbar([FULL_RANK], [acc_full_d],
            yerr=[[acc_full_d - ci_full_d[0]], [ci_full_d[1] - acc_full_d]],
            fmt="o", color="#4c72b0", ms=8, capsize=3, zorder=5, mfc="white", mew=1.8)
ax.text(FULL_RANK, acc_full_d + 4.7, "완전 $W$\n(파라미터 8배,\n이득 없음)",
        fontsize=8.0, color="#4c72b0", ha="center", va="bottom")

# curve 2 -- LABCLIP official recipe (InfoNCE), plain low-rank, in-sample 42
ax.plot(ranks_b, acc_b, "s--", color="#c44e52", lw=1.7, ms=5.5, zorder=4,
        label="② LABCLIP 공식 recipe (대칭 InfoNCE), 저계수 $W=UV^\\top$, 42개념 in-sample")

# LABCLIP official full-rank, identity-init -- its own marker
ax.scatter([FULL_RANK], [labclip_full], marker="*", s=240, color="#8c2d2d",
           edgecolor="white", lw=0.8, zorder=6,
           label=f"LABCLIP 원본 (identity-init 완전 $W$, InfoNCE) {labclip_full:.2f}%")

ax.annotate("LABCLIP recipe는 어떤 rank에서도\n우연 기준에 못 미침", xy=(300, 14.6),
            xytext=(60, 22.0), fontsize=8.3, color="#c44e52", ha="left",
            arrowprops=dict(arrowstyle="->", color="#c44e52", lw=1.1,
                            connectionstyle="arc3,rad=-0.25"), zorder=6)

ax.set_xscale("log", base=2)
ax.set_xticks(ranks_d + [64, 128, 256, FULL_RANK])
ax.set_xticklabels([str(r) for r in ranks_d] + ["64", "128", "256", "512"])
ax.minorticks_off()
ax.set_xlabel("저계수 $W$ 의 rank $r$", fontsize=11)
ax.set_ylabel("$2\\times2$ 매칭 정확도 (%)", fontsize=11)
ax.set_xlim(0.8, 780)
ax.set_ylim(-2, 42)
ax.grid(alpha=0.22, lw=0.7)
h,l=ax.get_legend_handles_labels()
order=[l.index(x) for x in sorted(l, key=lambda z:{"\u2460":0}.get(z[0],1) if z else 2)]
ax.legend([h[i] for i in order],[l[i] for i in order],fontsize=8.0, loc="lower right", framealpha=0.96, borderaxespad=0.6)

fig.savefig(OUT, dpi=175)
print(f"cosine {acc_cos:.2f}  full(margin,OOF) {acc_full_d:.2f} {ci_full_d}")
print("curve1 margin OOF  " + "  ".join(f"r{r}={a:.2f}[{c[0]:.1f},{c[1]:.1f}]"
      for r, a, c in zip(ranks_d, acc_d, ci_d)))
print("curve2 LABCLIP recipe InfoNCE  " + "  ".join(f"r{r}={a:.2f}" for r, a in zip(ranks_b, acc_b)))
print(f"LABCLIP full-rank identity-init = {labclip_full:.2f}")
print(f"saved {OUT}")
