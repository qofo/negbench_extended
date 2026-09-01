import json, glob, os, re
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

OUT = "logs/evaluation/01_paper/2026-08-31_negfam_mcq"
# Coefficient source: the single-object (6col) runs, deliberately, not the AB-swap sweep.
# NegBench MCQ and the negated-retrieval captions are themselves single-object negations,
# and a coefficient predicts an external task only when both are measured on the same
# construction: signed alpha from 6col gives r=+0.970 against the positive-negative gap,
# the same alpha from AB-swap gives r=+0.302, because AB-swap compresses alpha's
# between-model range from ~34 units to ~1.5 and removes the variance that carried it.
# The AB-swap sweep is canonical everywhere else (see mk_coefficients.py).
P = "logs/evaluation/01_paper/"
# (display name, mcq run dir, e2 dir)
M = [("ViT-B/32 (OpenAI)","vitb32_openai",   P+"2026-08-28_r6_main_effect_ablation_33concepts/"),
     ("ViT-B/16",         "vitb16_openai",   P+"2026-08-30_r8_vitb16/e2_hadamard_decomposition/"),
     ("ViT-L/14",         "vitl14_openai",   P+"2026-08-30_r8_vitl14/e2_hadamard_decomposition/"),
     ("LAION-2B",         "vitb32_laion2b",  P+"2026-08-30_negfam_laion2b/e2/"),
     ("SigLIP B/16",      "vitb16_siglip",   P+"2026-08-30_negfam_siglip_b16/e2/"),
     ("CoN-CLIP",         "conclip",         P+"2026-08-30_negfam_conclip/e2/"),
     ("NegCLIP",          "negclip",         P+"2026-08-30_negfam_negclip/e2/"),
     ("NegCLIP-NegFull",  "negclip_negfull", P+"2026-08-30_negfam_negclip_negfull/e2/"),
     ("CLIP-NegFull",     "clip_negfull",    P+"2026-08-30_negfam_clip_negfull/e2/")]

def parse_log(run):
    """Pull the 'Eval Epoch' metric lines out of out.log."""
    txt = open(os.path.join(OUT, run, "out.log"), encoding="utf-8", errors="replace").read()
    got = {}
    for m in re.finditer(r"(coco-mcq|voc2007-mcq)-(\w+_accuracy):\s*([0-9.]+)", txt):
        got[f"{m.group(1)}-{m.group(2)}"] = float(m.group(3))
    return got

rows = []
for name, run, e2 in M:
    g = parse_log(run)
    c = pd.read_csv(e2 + "e2_per_concept_decomposition.csv")
    p = pd.read_csv(e2 + "e2_per_pair_decomposition.csv")
    ratio_med = float(np.median(c.gamma_mean / np.maximum(c.abs_alpha_mean, c.abs_beta_mean)))
    rows.append(dict(
        model=name,
        ratio=ratio_med,
        alpha_signed=c.alpha_mean.mean()*1e3,
        abs_alpha=c.abs_alpha_mean.mean()*1e3, abs_beta=c.abs_beta_mean.mean()*1e3,
        gamma=c.gamma_mean.mean()*1e3,
        group=(p.gamma > np.maximum(p.abs_alpha, p.abs_beta)).mean()*100,
        coco=g.get("coco-mcq-total_accuracy", np.nan)*100,
        coco_pos=g.get("coco-mcq-positive_accuracy", np.nan)*100,
        coco_neg=g.get("coco-mcq-negative_accuracy", np.nan)*100,
        coco_hyb=g.get("coco-mcq-hybrid_accuracy", np.nan)*100,
        voc=g.get("voc2007-mcq-total_accuracy", np.nan)*100,
    ))
d = pd.DataFrame(rows)
# existence-detection AUC from the earlier figure data
aux = pd.read_csv("paper_figures/fig_metric_data.csv").set_index("name")
d["auc"] = [float(aux.loc[n.replace("ViT-B/32 (OpenAI)","OpenAI CLIP")
                          .replace("SigLIP B/16","SigLIP B/16"), "auc"])
            if n.replace("ViT-B/32 (OpenAI)","OpenAI CLIP") in aux.index else np.nan
            for n in d.model]
pd.set_option("display.width", 200)
print(d.round(3).to_string(index=False))
print()
for target in ["coco", "voc"]:
    ok = d[target].notna()
    for pred, lab in [("ratio","γ/max"), ("auc","존재탐지 AUC"), ("group","2×2 group")]:
        m = ok & d[pred].notna()
        if m.sum() < 3: continue
        r, pv = pearsonr(d[pred][m], d[target][m]); rs, ps = spearmanr(d[pred][m], d[target][m])
        print(f"{target.upper():5s} ~ {lab:12s}  Pearson r={r:+.3f} (p={pv:.3f})   Spearman rho={rs:+.3f} (p={ps:.3f})   n={m.sum()}")
    print()
d.to_csv("paper_figures/fig_external_data.csv", index=False)
print("saved paper_figures/fig_external_data.csv")
