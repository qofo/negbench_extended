"""Figure 2: does the learned similarity function actually satisfy the paper's
own success condition?

Per-concept scatter of the interaction term gamma against the dominant main
effect max(|alpha|, |beta|), on log-log axes. The diagonal is the success
boundary -- a concept is solvable iff its point sits above it. Cosine and the
prior full-rank linear alignment sit entirely below; the proposed low-rank
bilinear head trained with the identity-derived loss sits entirely above.

Input comes from decompose_w_coefficients.py.

Usage:
    python paper_figures/mk_success_condition.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

REPORT = "logs/evaluation/01_paper/2026-09-02_w_coefficients/w_coefficients_report.json"
OUT = "paper_figures/fig_success_condition.png"

with open(REPORT) as f:
    d = json.load(f)

SERIES = [
    ("cosine (W=I)", "코사인", "#555555", "o", 55),
    ("labclip", "완전랭크 선형 정렬 [4]", "#c44e52", "s", 45),
    ("proposed", "제안 (저계수 + 유도 손실)", "#4c72b0", "D", 55),
]

fig, ax = plt.subplots(figsize=(5.6, 5.3))

for key, label, color, marker, size in SERIES:
    pc = d[key]["per_concept"]
    x = np.maximum(np.array(pc["abs_alpha"]), np.array(pc["abs_beta"]))
    y = np.array(pc["gamma"])
    wins = d[key]["n_concepts_gamma_wins"]
    n = d[key]["n_concepts"]
    ax.scatter(x, y, s=size, marker=marker, facecolor=color, edgecolor="white",
               linewidth=0.6, alpha=0.85, zorder=3,
               label=f"{label} — {wins}/{n} 충족")

lo, hi = 1.2e-3, 6e-2
ax.plot([lo, hi], [lo, hi], color="#222222", ls="--", lw=1.6, zorder=2)
ax.fill_between([lo, hi], [lo, hi], hi * 8, color="#4c72b0", alpha=0.07, zorder=0)
ax.text(1.7e-3, 2.3e-2, "성공 영역\n$\\gamma > \\max(|\\alpha|,|\\beta|)$",
        fontsize=9.5, color="#33475e", va="center")
ax.text(2.0e-2, 6.5e-3, "성공 조건 경계", fontsize=9, color="#222222",
        rotation=32, ha="center", va="center")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lo, hi)
ax.set_ylim(3.5e-4, 5e-1)
ax.set_xlabel("지배 주효과 $\\max(|\\alpha|,\\ |\\beta|)$")
ax.set_ylabel("교차항 $\\gamma$")
ax.set_title("개념별 성공 조건 충족 여부", fontsize=11.5)
ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16),
          ncol=1, frameon=False, handletextpad=0.4)
ax.grid(alpha=0.22, which="both")

plt.tight_layout()
os.makedirs("paper_figures", exist_ok=True)
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"[saved] {OUT}")
