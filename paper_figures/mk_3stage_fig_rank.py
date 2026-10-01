"""Figure 2 for PAPER_3STAGE.md -- the scoring-rule ladder on a rank axis, replacing Table 3.

Every point is the same 42-concept AB-swap block read by a different scoring rule, scored on
concepts that rule never saw (concept-level GroupKFold). Rank is the x axis because rank is
the quantity the section varies; the three reference rules (identity, random W, diagonal W)
and the full bilinear form are all full rank by construction, so they sit at r = 512 rather
than being given a fictitious position:

  - the learned low-rank curve clears the exchangeable baseline from rank 2 on but flattens,
    and the intervals overlap across rank 2 to 32, so the optimum is not resolved;
  - the full bilinear form is the same family at full rank with 8x the parameters of rank 32
    and does *worse*, so the gain is not degrees of freedom;
  - a random low-rank map at the same rank AND the same parameter count sits below cosine,
    so the gain is not the low-rank structure either. Structure and training are both needed.

Cosine (the identity) and the random full W are drawn as horizontal lines instead of markers:
the reader needs the cosine gap readable at every rank, and three markers would collide at
x = 512.

Point estimates are the pair-weighted (pooled) accuracy quoted in the text; intervals are a
concept-level bootstrap of that same pooled estimator, so the bar is centred on the number
it belongs to.

Run from the repo root:
    python paper_figures/mk_3stage_fig_rank.py
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

LADDER = "logs/evaluation/01_paper/2026-09-01_single_w_abswap"
CONTROL = "logs/evaluation/01_paper/2026-09-01_random_lowrank32_control"
OUT = "paper_figures/fig_3stage_2.png"
CHANCE = 100 / 6  # exchangeable baseline, 1 / C(4,2)
FULL_RANK = 512
SEED = 42


def pooled(per_family: pd.DataFrame, family: str) -> float:
    g = per_family[per_family.family == family]
    return float((g.oof_acc_pct * g.n_test_pairs).sum() / g.n_test_pairs.sum())


def pooled_ci(per_concept: pd.DataFrame, family: str, n_boot: int = 20000):
    """Clustered CI: resample concepts, recompute the same pair-weighted mean."""
    g = per_concept[per_concept.family == family]
    a, w = g.oof_acc_pct.to_numpy(), g.n_pairs.to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    draws = (a[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


fam = pd.read_csv(f"{LADDER}/single_w_per_family.csv")
con = pd.read_csv(f"{LADDER}/single_w_per_concept.csv")
ctl_f = pd.read_csv(f"{CONTROL}/single_w_per_family.csv")

ranks = [1, 2, 4, 8, 16, 32]
acc = np.array([pooled(fam, f"lowrank_{r}") for r in ranks])
ci = np.array([pooled_ci(con, f"lowrank_{r}") for r in ranks])

acc_cos, acc_rand = pooled(fam, "identity"), pooled(fam, "random")
acc_diag, acc_full = pooled(fam, "diagonal"), pooled(fam, "full")
ci_full = pooled_ci(con, "full")
ci_diag = pooled_ci(con, "diagonal")
acc_rlr = pooled(ctl_f, "random_lowrank_32")

fig, ax = plt.subplots(figsize=(7.0, 4.6))
fig.subplots_adjust(left=0.10, right=0.985, top=0.965, bottom=0.155)

# Full-rank rules live to the right of the sweep; shade that region so the axis break in
# meaning (a swept hyper-parameter vs. four fixed families) is visible rather than implied.
ax.axvspan(110, 1000, color="#f2f2f2", zorder=0)
ax.text(FULL_RANK, 34.6, "완전 계수", fontsize=9.0, color="#777", ha="center")

ax.axhline(CHANCE, color="#888", ls="--", lw=1.2, zorder=1)
ax.text(0.86, CHANCE + 0.8, f"교환 가능 기준 {CHANCE:.2f}%", fontsize=8.8, color="#666")
ax.axhline(acc_cos, color="#333", ls="-", lw=1.4, zorder=1)
ax.text(0.86, acc_cos + 0.9, f"항등 $W$ = 코사인 {acc_cos:.2f}%", fontsize=8.8, color="#333")
ax.axhline(acc_rand, color="#999", ls=":", lw=1.3, zorder=1)
ax.text(0.86, acc_rand - 1.7, f"무작위 $W$ {acc_rand:.2f}%", fontsize=8.8, color="#777")

ax.errorbar(ranks, acc, yerr=[acc - ci[:, 0], ci[:, 1] - acc],
            fmt="o-", color="#4c72b0", lw=1.8, ms=6.5, capsize=3.2, zorder=4)
ax.text(1.06, 32.0, "학습된 저계수 $UV^{\\top}$", fontsize=9.8, color="#4c72b0")

ax.errorbar([FULL_RANK], [acc_full], yerr=[[acc_full - ci_full[0]], [ci_full[1] - acc_full]],
            fmt="s", color="#c44e52", ms=8, capsize=3.2, zorder=4)
ax.text(FULL_RANK, 18.4, "학습된 완전 $W$", fontsize=9.2, color="#c44e52",
        ha="center", va="top")

ax.errorbar([FULL_RANK], [acc_diag], yerr=[[acc_diag - ci_diag[0]], [ci_diag[1] - acc_diag]],
            fmt="^", color="#dd8452", ms=8, capsize=3.2, zorder=4)
ax.text(FULL_RANK * 0.80, acc_diag + 0.1, "학습된 대각 $W$", fontsize=9.2,
        color="#dd8452", ha="right", va="center")

ax.scatter([32], [acc_rlr], marker="x", s=75, color="#555", lw=2.0, zorder=4)
ax.text(42, 2.1, "무작위 저계수 (비학습)", fontsize=9.2, color="#555", ha="left")

# The two controls that carry the argument, stated on the figure so the caption need not.
ax.annotate("", xy=(FULL_RANK * 0.72, acc_full + 1.4), xytext=(38, acc[-1] + 0.6),
            arrowprops=dict(arrowstyle="->", color="#c44e52", lw=1.2,
                            connectionstyle="arc3,rad=-0.30"), zorder=3)
ax.text(150, 32.4, "파라미터 8배,\n이득은 없음", fontsize=8.6, color="#c44e52", ha="center")
ax.annotate("", xy=(32, acc_rlr + 0.8), xytext=(32, acc[-1] - 1.2),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.1, ls=(0, (3, 2))), zorder=3)
ax.text(37, 13.2, "같은 rank,\n같은 파라미터 수", fontsize=8.6, color="#555", ha="left")

ax.set_xscale("log", base=2)
ax.set_xticks(ranks + [FULL_RANK])
ax.set_xticklabels([str(r) for r in ranks] + ["512"])
ax.minorticks_off()
ax.set_xlabel("rank $r$   (학습 파라미터 수 $= 1{,}024\\,r$)", fontsize=11)
ax.set_ylabel("$2\\times2$ 판정 정확도 (%)  ·  미학습 개념 홀드아웃", fontsize=10.5)
ax.set_xlim(0.8, 1000)
ax.set_ylim(-2.0, 36)
ax.grid(alpha=0.22, lw=0.7)

fig.savefig(OUT, dpi=170)
print(f"cosine {acc_cos:.2f}  random {acc_rand:.2f}  diagonal {acc_diag:.2f}  "
      f"random-lowrank32 {acc_rlr:.2f}  full {acc_full:.2f} {ci_full}")
print("lowrank " + "  ".join(f"r{r}={a:.2f}[{c[0]:.1f},{c[1]:.1f}]"
                             for r, a, c in zip(ranks, acc, ci)))
print(f"saved {OUT}")
