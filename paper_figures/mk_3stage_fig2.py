"""Figure 2 for PAPER_3STAGE.md -- the scoring-rule ladder, as a curve rather than a table.

Every point is the same 42-concept AB-swap block read by a different scoring rule, scored
on concepts that rule never saw (concept-level GroupKFold). Putting parameter count on the
x axis rather than rank lets the three reference conditions sit on the same axis as the
learned low-rank sweep, which is what makes the argument legible in one picture:

  - the learned low-rank curve rises far above cosine and clears the exchangeable
    baseline from rank 2 on, but flattens -- the confidence intervals overlap across
    rank 2 to 32, so the optimum is not resolved;
  - the full bilinear form has 8x the parameters of rank 32 and does *worse*, so the gain
    is not degrees of freedom;
  - a random low-rank map with exactly rank 32's parameter count sits below cosine, so
    the gain is not the low-rank structure either. Structure and training are both needed.

Values are read from the ladder run itself; the pooled accuracy is the pair-weighted mean
over folds and the error bars are a concept-level bootstrap, so the two aggregations the
text quotes are both visible.

Run from the repo root:
    python paper_figures/mk_3stage_fig2.py
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
SEED = 42


def pooled(per_family: pd.DataFrame, family: str) -> float:
    g = per_family[per_family.family == family]
    return float((g.oof_acc_pct * g.n_test_pairs).sum() / g.n_test_pairs.sum())


def macro_ci(per_concept: pd.DataFrame, family: str, n_boot: int = 20000):
    a = per_concept[per_concept.family == family].oof_acc_pct.to_numpy()
    rng = np.random.default_rng(SEED)
    draws = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    return float(a.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


fam = pd.read_csv(f"{LADDER}/single_w_per_family.csv")
con = pd.read_csv(f"{LADDER}/single_w_per_concept.csv")
ctl = pd.read_csv(f"{CONTROL}/single_w_per_family.csv")

ranks = [1, 2, 4, 8, 16, 32]
params = [1024 * r for r in ranks]
acc = [pooled(fam, f"lowrank_{r}") for r in ranks]
lo = [macro_ci(con, f"lowrank_{r}")[1] for r in ranks]
hi = [macro_ci(con, f"lowrank_{r}")[2] for r in ranks]

acc_full, acc_diag = pooled(fam, "full"), pooled(fam, "diagonal")
lo_f, hi_f = macro_ci(con, "full")[1:]
acc_cos, acc_rand = pooled(fam, "identity"), pooled(fam, "random")
acc_rlr = pooled(ctl, "random_lowrank_32")

fig, ax = plt.subplots(figsize=(7.0, 4.6))
fig.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.135)

ax.axhline(CHANCE, color="#888", ls="--", lw=1.2, zorder=1)
ax.text(5.2e5, CHANCE + 0.9, f"교환 가능 기준 {CHANCE:.2f}%", fontsize=8.8,
        color="#666", ha="right")
ax.axhline(acc_cos, color="#333", ls="-", lw=1.4, zorder=1)
ax.text(5.2e5, acc_cos + 0.9, f"항등 $W$ = 코사인 {acc_cos:.2f}%", fontsize=8.8,
        color="#333", ha="right")
ax.axhline(acc_rand, color="#999", ls=":", lw=1.2, zorder=1)
ax.text(5.2e5, acc_rand - 1.5, f"무작위 $W$ {acc_rand:.2f}%", fontsize=8.8,
        color="#777", ha="right")

ax.errorbar(params, acc, yerr=[np.array(acc) - lo, np.array(hi) - np.array(acc)],
            fmt="o-", color="#4c72b0", lw=1.8, ms=6.5, capsize=3.2, zorder=4)
for x, y, r in zip(params, acc, ranks):
    ax.annotate(f"$r$={r}", (x, y), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=8.6, color="#4c72b0")

ax.errorbar([262144], [acc_full], yerr=[[acc_full - lo_f], [hi_f - acc_full]],
            fmt="s", color="#c44e52", ms=8, capsize=3.2, zorder=4)
ax.scatter([512], [acc_diag], marker="^", s=70, color="#dd8452", zorder=4)
ax.scatter([32768], [acc_rlr], marker="x", s=70, color="#666", lw=2.0, zorder=4)

# Labelled in place rather than through a legend box: with only four series the box
# always lands on one of the isolated markers.
ax.text(1.05e3, 23.6, "학습된 저계수 $UV^{\\top}$", fontsize=9.6, color="#4c72b0")
ax.text(2.62e5, 21.4, "학습된\n완전 $W$", fontsize=9.2, color="#c44e52", ha="center", va="top")
ax.text(5.6e2, 6.3, "학습된 대각 $W$", fontsize=9.2, color="#dd8452")
ax.text(3.6e4, 1.9, "무작위 저계수 $r$=32 (비학습)", fontsize=9.2, color="#555")

ax.annotate("", xy=(2.30e5, acc_full + 1.2), xytext=(3.7e4, acc[-1] - 1.0),
            arrowprops=dict(arrowstyle="->", color="#c44e52", lw=1.2,
                            connectionstyle="arc3,rad=-0.28"), zorder=3)
ax.text(8.6e4, 24.4, "파라미터 8배,\n정확도는 하락", fontsize=8.4, color="#c44e52", ha="center")

ax.set_xscale("log")
ax.set_xlabel("학습 파라미터 수", fontsize=11)
ax.set_ylabel("$2\\times2$ 판정 정확도 (%)  ·  미학습 개념 홀드아웃", fontsize=10.5)
ax.set_xlim(2.6e2, 6.0e5)
ax.set_ylim(-2.0, 36)
ax.grid(alpha=0.22, lw=0.7)

fig.savefig(OUT, dpi=170)
print(f"cosine {acc_cos:.2f}  random {acc_rand:.2f}  diagonal {acc_diag:.2f}  "
      f"random-lowrank32 {acc_rlr:.2f}  full {acc_full:.2f}")
print("lowrank " + "  ".join(f"r{r}={a:.2f}" for r, a in zip(ranks, acc)))
print(f"saved {OUT}")
