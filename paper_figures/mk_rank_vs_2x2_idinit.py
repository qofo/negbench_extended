"""rank vs. 2x2, InfoNCE with an IDENTITY-PRESERVING residual W = I + UV^T and a
tunable drift penalty c*||W-I||_F^2.  Left: 2x2 matching accuracy.  Right: COCO
T2I R@1 (how much of the cosine identity survives).  The two panels share the
c-lines, so the reader sees the trade-off directly: the penalty that keeps
retrieval (right, c=0.1 at/above cosine) is the penalty that keeps 2x2 flat
(left, c=0.1 never clears ~8%).

Data: logs/.../2026-09-10_infonce_idinit/final_sweep.csv  (this run)
Reference: logs/.../2026-09-01_single_w_abswap_deltaloss  -- the manuscript's
           block-task margin-loss curve, concept GroupKFold OOF.

Run from repo root:  python paper_figures/mk_rank_vs_2x2_idinit.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

SWEEP = "logs/evaluation/01_paper/2026-09-10_infonce_idinit/final_sweep.csv"
REP = "logs/evaluation/01_paper/2026-09-10_infonce_idinit/final_report.json"
DELTA = "logs/evaluation/01_paper/2026-09-01_single_w_abswap_deltaloss/single_w_per_family.csv"
DELTA_CON = "logs/evaluation/01_paper/2026-09-01_single_w_abswap_deltaloss/single_w_per_concept.csv"
OUT = "paper_figures/fig_rank_vs_2x2_idinit.png"
CHANCE = 100 / 6
SEED = 42

df = pd.read_csv(SWEEP)
rep = json.load(open(REP))
cos = rep["cosine"]

fam = pd.read_csv(DELTA)
con = pd.read_csv(DELTA_CON)
def pooled(f):
    g = fam[fam.family == f]
    return float((g.oof_acc_pct * g.n_test_pairs).sum() / g.n_test_pairs.sum())
def pooled_ci(f, nb=20000):
    g = con[con.family == f]; a, w = g.oof_acc_pct.to_numpy(), g.n_pairs.to_numpy(float)
    rng = np.random.default_rng(SEED); i = rng.integers(0, len(a), size=(nb, len(a)))
    d = (a[i] * w[i]).sum(1) / w[i].sum(1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)
d_ranks = [1, 2, 4, 8, 16, 32]
d_acc = np.array([pooled(f"lowrank_{r}") for r in d_ranks] + [pooled("full")])
d_x = d_ranks + [512]

cs = sorted(df.c.unique())
col = {0.0: "#c44e52", 0.001: "#dd8452", 0.01: "#55a868", 0.1: "#4c72b0"}
lab = {0.0: "$c=0$  (표류 자유)", 0.001: "$c=10^{-3}$", 0.01: "$c=10^{-2}$",
       0.1: "$c=10^{-1}$  (항등 유지)"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.9))
fig.subplots_adjust(left=0.065, right=0.995, top=0.86, bottom=0.12, wspace=0.19)

# ---- left: 2x2
axL.axhline(CHANCE, color="#888", ls="--", lw=1.1)
axL.text(0.85, CHANCE + 0.9, f"우연 {CHANCE:.2f}%", fontsize=8.5, color="#666")
axL.axhline(cos["acc_2x2"], color="#333", ls="-", lw=1.2)
axL.text(0.85, cos["acc_2x2"] + 1.0, f"코사인 {cos['acc_2x2']:.2f}%", fontsize=8.5, color="#333")
axL.plot(d_x, d_acc, "k:o", lw=1.5, ms=5, zorder=6,
         label="참고: 블록 과제 마진 손실 (개념 OOF, 논문 §2.4)")
for c in cs:
    g = df[df.c == c].sort_values("rank")
    axL.plot(g["rank"], g.acc_2x2, "o-", color=col[c], lw=1.8, ms=5, label=lab[c])
axL.set_xscale("log", base=2); axL.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
axL.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]); axL.minorticks_off()
axL.set_xlabel("저계수 보정의 rank $r$", fontsize=10.5)
axL.set_ylabel("$2\\times2$ 매칭 정확도 (%)", fontsize=10.5)
axL.set_xlim(0.8, 640); axL.set_ylim(-2, 50); axL.grid(alpha=0.22, lw=0.6)
axL.set_title("(a) 부정 결합 정확도", fontsize=10.5)
axL.legend(fontsize=7.8, loc="upper left", framealpha=0.95)

# ---- right: retrieval R@1
axR.axhline(cos["std_r1"], color="#333", ls="-", lw=1.2)
axR.text(0.85, cos['std_r1'] - 2.4, f"코사인 R@1 {cos['std_r1']:.2f}%", fontsize=8.5, color="#333")
for c in cs:
    g = df[df.c == c].sort_values("rank")
    axR.plot(g["rank"], g.std_r1, "o-", color=col[c], lw=1.8, ms=5, label=lab[c])
axR.set_xscale("log", base=2); axR.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
axR.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]); axR.minorticks_off()
axR.set_xlabel("저계수 보정의 rank $r$", fontsize=10.5)
axR.set_ylabel("COCO T2I R@1 (%)  ·  항등행렬 보존 정도", fontsize=10.5)
axR.set_xlim(0.8, 640); axR.set_ylim(-2, 40); axR.grid(alpha=0.22, lw=0.6)
axR.set_title("(b) 검색 성능 (코사인 = 항등)", fontsize=10.5)
axR.legend(fontsize=7.8, loc="lower left", framealpha=0.95)

fig.suptitle("항등행렬을 유지한 InfoNCE 정렬:  $W = I + UV^\\top$,  손실 $= $ InfoNCE $ + c\\,\\|W-I\\|_F^2$",
             fontsize=10.8, y=0.975)
fig.savefig(OUT, dpi=170)
print("cosine", cos)
for c in cs:
    g = df[df.c == c].sort_values("rank")
    print(f"c={c}: " + "  ".join(f"r{int(r)}:{a:.1f}/{s:.1f}"
          for r, a, s in zip(g["rank"], g.acc_2x2, g.std_r1)))
print("saved", OUT)
