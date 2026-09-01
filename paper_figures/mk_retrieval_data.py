"""Aggregate the nine-model T2I retrieval sweep against the Hadamard coefficients.

Run from the repo root after benchmarks/scripts/run_negfam_retrieval.sh:
    python paper_figures/mk_retrieval_data.py
"""
import os, re, sys
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

OUT = "logs/evaluation/01_paper/2026-08-31_negfam_retrieval"
# Coefficient source: the single-object (6col) runs, deliberately, not the AB-swap sweep.
# NegBench MCQ and the negated-retrieval captions are themselves single-object negations,
# and a coefficient predicts an external task only when both are measured on the same
# construction: signed alpha from 6col gives r=+0.970 against the positive-negative gap,
# the same alpha from AB-swap gives r=+0.302, because AB-swap compresses alpha's
# between-model range from ~34 units to ~1.5 and removes the variance that carried it.
# The AB-swap sweep is canonical everywhere else (see mk_coefficients.py).
P = "logs/evaluation/01_paper/"
M = [("ViT-B/32 (OpenAI)","vitb32_openai",   P+"2026-08-28_r6_main_effect_ablation_33concepts/"),
     ("ViT-B/16",         "vitb16_openai",   P+"2026-08-30_r8_vitb16/e2_hadamard_decomposition/"),
     ("ViT-L/14",         "vitl14_openai",   P+"2026-08-30_r8_vitl14/e2_hadamard_decomposition/"),
     ("LAION-2B",         "vitb32_laion2b",  P+"2026-08-30_negfam_laion2b/e2/"),
     ("SigLIP B/16",      "vitb16_siglip",   P+"2026-08-30_negfam_siglip_b16/e2/"),
     ("CoN-CLIP",         "conclip",         P+"2026-08-30_negfam_conclip/e2/"),
     ("NegCLIP",          "negclip",         P+"2026-08-30_negfam_negclip/e2/"),
     ("NegCLIP-NegFull",  "negclip_negfull", P+"2026-08-30_negfam_negclip_negfull/e2/"),
     ("CLIP-NegFull",     "clip_negfull",    P+"2026-08-30_negfam_clip_negfull/e2/")]

# The negated-retrieval captions are "<affirmed content>, but there is no X" -- the same
# affirm-and-negate shape as the AB-swap pairs, unlike the MCQ items, which are single-object
# negations. Which coordinate matches the evaluation construction is therefore an empirical
# question here, so both are runnable:  python paper_figures/mk_retrieval_data.py abswap
if len(sys.argv) > 1 and sys.argv[1] == "abswap":
    AB = {"ViT-B/32 (OpenAI)": "vitb32_openai", "ViT-B/16": "vitb16_openai",
          "ViT-L/14": "vitl14_openai", "LAION-2B": "vitb32_laion2b",
          "SigLIP B/16": "vitb16_siglip", "CoN-CLIP": "conclip", "NegCLIP": "negclip",
          "NegCLIP-NegFull": "negclip_negfull", "CLIP-NegFull": "clip_negfull"}
    M = [(n, r, P + "2026-08-31_negfam_e2_abswap/" + AB[n] + "/") for n, r, _ in M]
    CSV_OUT = "paper_figures/fig_retrieval_data_abswap.csv"
    print("=== coefficients: AB-swap ===")
else:
    CSV_OUT = "paper_figures/fig_retrieval_data.csv"
    print("=== coefficients: single-object (6col) ===")

def parse(run):
    txt = open(os.path.join(OUT, run, "out.log"), encoding="utf-8", errors="replace").read()
    g = {}
    for m in re.finditer(r"(coco(?:-negated)?)-image_retrieval_recall@(\d+):\s*([0-9.]+)", txt):
        g[f"{m.group(1)}@{m.group(2)}"] = float(m.group(3)) * 100
    return g

rows = []
for name, run, e2 in M:
    g = parse(run)
    c = pd.read_csv(e2 + "e2_per_concept_decomposition.csv")
    a, b, gm = c.alpha_mean.mean(), c.beta_mean.mean(), c.gamma_mean.mean()
    A, B = c.abs_alpha_mean.mean(), c.abs_beta_mean.mean()
    rows.append(dict(
        model=name,
        beta_signed=b*1e3, abs_beta=B*1e3, alpha_signed=a*1e3, abs_alpha=A*1e3, gamma=gm*1e3,
        ratio_max=float(np.median(c.gamma_mean/np.maximum(c.abs_alpha_mean, c.abs_beta_mean))),
        ratio_beta=float(np.median(c.gamma_mean/c.abs_beta_mean)),   # the T2I-matched ratio
        r1=g.get("coco@1"), r5=g.get("coco@5"),
        nr1=g.get("coco-negated@1"), nr5=g.get("coco-negated@5")))
d = pd.DataFrame(rows)
d["drop1"] = d.r1 - d.nr1                    # absolute R@1 degradation under negation
d["drop5"] = d.r5 - d.nr5
d["rel1"]  = d.drop1 / d.r1 * 100            # relative degradation (robustness check)
pd.set_option("display.width", 250)
print(d.round(3).to_string(index=False)); print()

def s(x, y, l):
    m = np.isfinite(x) & np.isfinite(y)
    r, p = pearsonr(x[m], y[m]); rs, ps = spearmanr(x[m], y[m])
    print(f"{l:44s} r={r:+.3f} p={p:.2e} | rho={rs:+.3f} p={ps:.2e} | n={m.sum()}")

print("--- 이론이 지정하는 방향 예측: beta = 존재 선호 -> 부정 질의에서 더 크게 하락 ---")
s(d.beta_signed.values, d.drop1.values, "beta(부호) -> R@1 하락폭 (절대)")
s(d.beta_signed.values, d.drop5.values, "beta(부호) -> R@5 하락폭 (절대)")
s(d.beta_signed.values, d.rel1.values,  "beta(부호) -> R@1 하락폭 (상대 %)")
print()
print("--- 지표가 부정 검색 성적 자체를 예측하는가 ---")
s(d.ratio_beta.values, d.nr1.values, "gamma/|beta| (과제 정합) -> 부정 R@1")
s(d.ratio_max.values,  d.nr1.values, "gamma/max               -> 부정 R@1")
s(d.ratio_max.values,  d.r1.values,  "gamma/max               -> 표준 R@1 (대조)")
d.to_csv(CSV_OUT, index=False)
print(f"\nsaved {CSV_OUT}")
