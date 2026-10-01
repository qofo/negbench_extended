"""Figure 2 for the restructured paper: where the rank constraint should enter,
and the spectral reason why.

(a) rank vs standard T2I R@1 for the three places the rank constraint can
    enter -- (A) post-hoc SVD truncation of a full-rank W, (B) a plain
    factorization W = U V^T, (C) the residual form W = I + U V^T -- against
    plain cosine as a fixed reference. A's rank-512 point is the untruncated
    full-rank linear alignment of prior work.
(b) Frobenius energy retained by the best rank-r approximation of I, W and
    dW = W - I -- the identity is the worst possible low-rank target, which is
    why it must be carried outside the rank budget rather than re-encoded.
(c) distance from the identity vs retrieval and the 2x2 diagnostic, over the
    Frobenius-penalty sweep, with the unregularized rank-8 residual as an
    off-family reference: retrieval peaks just off the identity while the
    diagnostic rises monotonically as W drifts away from it.

Usage:
    python paper_figures/mk_design_comparison.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

REPORT = "logs/evaluation/01_paper/2026-09-02_labclip_rank_strategy/rank_strategy_report.json"
OUT = "paper_figures/fig_design_comparison.png"

with open(REPORT) as f:
    d = json.load(f)

cosine_r1 = d["cosine"]["std_r1"]
conds = d["conditions"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.2, 4.2))

# rank 제약이 들어가는 위치 세 가지를 같은 축에 올린다. A만 완전랭크(512)까지
# 이어지며, 그 끝점이 절단하지 않은 LABCLIP[4]의 완전랭크 변환이다.
strategies = [
    ("A_posthoc_svd",      "(A) 학습 후 SVD 절단",         "#c44e52", "--", "s"),
    ("B_plain_lowrank",    "(B) 평문 $W = UV^\\top$",      "#8172b2", ":",  "^"),
    ("C_residual_lowrank", "(C) 잔차 $W = I + UV^\\top$",  "#4c72b0", "-",  "o"),
]
for key, label, color, ls, marker in strategies:
    rows = sorted(conds[key], key=lambda r: r["rank"])
    ax1.plot([r["rank"] for r in rows], [r["std_r1"] for r in rows],
             marker=marker, color=color, ls=ls, lw=2.2, ms=7, label=label)

ax1.axhline(cosine_r1, color="#555555", ls="-.", lw=1.8)
ax1.text(512, cosine_r1 + 1.0, f"기존 코사인 {cosine_r1:.1f}%",
         fontsize=9.5, color="#555555", ha="right")

ax1.set_xscale("log", base=2)
ticks = [8, 16, 32, 64, 128, 256, 512]
ax1.set_xticks(ticks)
ax1.set_xticklabels([str(x) for x in ticks])
ax1.set_xlabel("rank $r$")
ax1.set_ylabel("COCO 갤러리 검색 R@1 (%)")
ax1.set_title("(a) rank 제약이 들어가는 위치", fontsize=11)
ax1.set_ylim(-2, 36)
ax1.legend(fontsize=9, loc="center left", framealpha=0.95)
ax1.grid(alpha=0.25)

# ---- panel (b): spectral energy retention ----
spec = d["spectrum"]
ranks = [8, 16, 32, 64, 128, 256]
series = [
    ("identity", "$I$ (코사인)", "#555555", "-."),
    ("W_full", "학습된 $W$", "#c44e52", "--"),
    ("dW_full_minus_I", "$\\Delta W = W - I$", "#4c72b0", "-"),
]
for key, label, color, ls in series:
    ys = [spec[key][f"top{r}_energy_pct"] for r in ranks]
    ax2.plot(ranks, ys, marker="o", color=color, ls=ls, label=label, lw=2, ms=6)

ax2.set_xscale("log", base=2)
ax2.set_xticks(ranks)
ax2.set_xticklabels([str(r) for r in ranks])
ax2.set_xlabel("상위 특이값 개수 $r$")
ax2.set_ylabel("보존된 프로베니우스 에너지 (%)")
ax2.set_title("(b) 항등은 압축 불가능하다", fontsize=11)
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(alpha=0.25)
ax2.set_ylim(0, 104)

# ---- panel (c): distance from identity vs retrieval / diagnostic ----
# 규제 계수 스윕과 무규제 저계수 대조점. ||dW||_F 와 2x2 판정 값은 RESULTS.md
# 8-V.12(규제 스윕) / 8-V.13(rank 스윕)의 기록값이며, 표준 R@1 은 아래 진단
# 로그의 gallery_scaling(N=5000)에서 그대로 읽는다.
SWEEP = [  # (label, ||dW||_F, std R@1, 2x2 delta_s, run dir or None for cosine)
    ("코사인", 0.00, cosine_r1, d["cosine"]["delta_s"], None),
    ("$c$=0.1", 1.79, None, 5.36, "2026-09-02_scale_residual_diag_pen0.1"),
    ("$c$=0.01", 6.27, None, 8.23,
     "2026-09-02_scale_residual_diag_residual_identity_pen0.01"),
    ("$c$=0.001", 10.66, None, 10.08, "2026-09-02_scale_residual_diag_pen0.001"),
]
LOGROOT = "logs/evaluation/01_paper"
xs, r1s, dss, labels = [], [], [], []
for label, dist, r1, ds, run in SWEEP:
    if run is not None:
        with open(os.path.join(LOGROOT, run, "diagnostics_report.json")) as f:
            r1 = json.load(f)["gallery_scaling"]["5000"]["w_r1_mean"]
    xs.append(dist); r1s.append(r1); dss.append(ds); labels.append(label)

ax3.plot(xs, r1s, marker="o", color="#4c72b0", ls="-", lw=2.2, ms=8,
         label="검색 R@1 (좌축)")
for x, y, label in zip(xs, r1s, labels):
    ax3.annotate(label, (x, y), textcoords="offset points", xytext=(0, 9),
                 ha="center", fontsize=9, color="#4c72b0")

# 규제 없이 저계수만으로 제약한 대조점 — 파라미터는 가장 적은데 항등에서는 더 멀다.
lr8 = [r for r in conds["C_residual_lowrank"] if r["rank"] == 8][0]
ax3.plot([5.97], [lr8["std_r1"]], marker="D", color="#8172b2", ms=9, ls="none",
         label="저계수 rank 8, 규제 없음")

ax3.axhline(cosine_r1, color="#555555", ls="-.", lw=1.4, zorder=0)
ax3.set_xlabel("항등에서의 거리 $\\|\\Delta W\\|_F$")
ax3.set_ylabel("COCO 갤러리 검색 R@1 (%)", color="#4c72b0")
ax3.tick_params(axis="y", labelcolor="#4c72b0")
ax3.set_ylim(14, 36)
ax3.set_title("(c) 항등에서 멀어질 때의 교환", fontsize=11)
ax3.grid(alpha=0.25)

ax3b = ax3.twinx()
ax3b.plot(xs, dss, marker="s", color="#c44e52", ls="--", lw=2.0, ms=7,
          label="2×2 판정 (우축)")
ax3b.plot([5.97], [lr8["delta_s"]], marker="D", color="#c44e52", ms=7,
          alpha=0.45, ls="none")
ax3b.set_ylabel("2×2 판정 정확도 (%)", color="#c44e52")
ax3b.tick_params(axis="y", labelcolor="#c44e52")
ax3b.set_ylim(0, 12)

h1, l1 = ax3.get_legend_handles_labels()
h2, l2 = ax3b.get_legend_handles_labels()
ax3.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="lower left", framealpha=0.95)

plt.tight_layout()
os.makedirs("paper_figures", exist_ok=True)
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"[saved] {OUT}")
