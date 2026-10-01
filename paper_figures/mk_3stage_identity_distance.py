"""Distance from the identity predicts retrieval.

Not referenced by the current PAPER_3STAGE.md draft, which scopes itself to controlled
judgement and leaves open-gallery retrieval out; kept because it is the figure for the
retrieval claim if that scope is ever widened.

Every learned bilinear W in the paper is one point: the three rank-constrained
parameterizations (post-hoc SVD truncation, plain U V^T, residual I + U V^T) across
their rank sweeps, plain cosine at the origin, and the regularized identity residual
that the paper proposes. The x axis is ||W - I||_F and the y axis is standard COCO
T2I R@1, so the retrieval claim -- that retrieval is governed by distance
from the identity rather than by rank -- is legible as one line.

Run from the repo root:
    python paper_figures/mk_3stage_identity_distance.py
"""
import json

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

REPORT = "logs/evaluation/01_paper/2026-09-02_labclip_rank_strategy/rank_strategy_report.json"
OUT = "paper_figures/fig_3stage_identity_distance.png"

# The regularized residual (c = 0.1) is trained by train_residual_identity_infonce.py and
# evaluated in RESULTS.md 8-V.12; it is not part of the rank-strategy report above.
PROPOSED = {"dW_norm": 1.79, "std_r1": 31.78}

with open(REPORT) as f:
    d = json.load(f)

cos_r1 = d["cosine"]["std_r1"]
conds = d["conditions"]

series = [
    ("A. 사후 SVD 절단", conds["A_posthoc_svd"], "#c44e52", "^"),
    ("B. 평문 $UV^{\\top}$", conds["B_plain_lowrank"], "#dd8452", "s"),
    ("C. 잔차 $I + UV^{\\top}$", conds["C_residual_lowrank"], "#4c72b0", "o"),
]

xs = [0.0, PROPOSED["dW_norm"]]
ys = [cos_r1, PROPOSED["std_r1"]]
for _, pts, _, _ in series:
    xs += [p["dW_norm"] for p in pts]
    ys += [p["std_r1"] for p in pts]
xs, ys = np.asarray(xs), np.asarray(ys)
r = float(np.corrcoef(xs, ys)[0, 1])

fig, ax = plt.subplots(figsize=(6.6, 4.5))
fig.subplots_adjust(left=0.115, right=0.98, top=0.94, bottom=0.135)

slope, intercept = np.polyfit(xs, ys, 1)
grid = np.linspace(-0.6, xs.max() + 1.2, 50)
ax.plot(grid, slope * grid + intercept, color="#999", lw=1.2, ls="--", zorder=1)

for label, pts, color, marker in series:
    ax.scatter([p["dW_norm"] for p in pts], [p["std_r1"] for p in pts],
               s=46, c=color, marker=marker, edgecolor="white", lw=0.7,
               label=label, zorder=3)

ax.scatter([0.0], [cos_r1], s=140, c="#333", marker="*", zorder=4, label="코사인 ($W = I$)")
ax.scatter([PROPOSED["dW_norm"]], [PROPOSED["std_r1"]], s=110, c="#2a9d8f",
           marker="D", edgecolor="white", lw=0.9, zorder=4, label="제안 (항등 잔차 + 규제)")

ax.annotate("rank 8", xy=(conds["C_residual_lowrank"][0]["dW_norm"],
                          conds["C_residual_lowrank"][0]["std_r1"]),
            xytext=(8, 6), textcoords="offset points", fontsize=8.4, color="#4c72b0")
ax.annotate("rank 256", xy=(conds["C_residual_lowrank"][-1]["dW_norm"],
                            conds["C_residual_lowrank"][-1]["std_r1"]),
            xytext=(6, -13), textcoords="offset points", fontsize=8.4, color="#4c72b0")

ax.set_xlabel(r"항등 행렬로부터의 거리  $\Vert W - I \Vert_{F}$", fontsize=11)
ax.set_ylabel("COCO 표준 T2I R@1 (%)", fontsize=11)
ax.set_xlim(-0.8, xs.max() + 1.4)
ax.set_ylim(-2, max(ys) + 4)
ax.grid(alpha=0.25, lw=0.7)
ax.text(0.035, 0.06, f"Pearson $r$ = {r:.3f}   ($n$ = {len(xs)})",
        transform=ax.transAxes, ha="left", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.42", facecolor="#f4f4f6", edgecolor="#bbb"))
ax.legend(fontsize=8.8, loc="upper right", framealpha=0.95)

fig.savefig(OUT, dpi=170)
print(f"n = {len(xs)}   Pearson r = {r:.4f}")
print(f"saved {OUT}")
