"""Rebuild fig_metric_data.csv from the AB-swap sweep.

`ratio` (median gamma / max(|alpha|,|beta|)) and `acc` (2x2 group accuracy, pooled over pairs)
come from the canonical AB-swap coefficient runs. `auc` is minimal-pair existence detection,
computed on the image side alone, so the caption design does not touch it and it is carried
forward. `win` is retained for schema stability; no figure reads it.

Run from the repo root, before mk_metric_predicts.py:
    python paper_figures/mk_metric_data.py
"""
import numpy as np, pandas as pd

P = "logs/evaluation/01_paper/2026-08-31_negfam_e2_abswap/"
RUN = {"OpenAI CLIP": "vitb32_openai", "ViT-B/16": "vitb16_openai", "ViT-L/14": "vitl14_openai",
       "LAION-2B": "vitb32_laion2b", "SigLIP B/16": "vitb16_siglip", "CoN-CLIP": "conclip",
       "NegCLIP": "negclip", "NegCLIP-NegFull": "negclip_negfull", "CLIP-NegFull": "clip_negfull"}

prev = pd.read_csv("paper_figures/fig_metric_data.csv").set_index("name")

rows = []
for name, run in RUN.items():
    c = pd.read_csv(P + run + "/e2_per_concept_decomposition.csv")
    p = pd.read_csv(P + run + "/e2_per_pair_decomposition.csv")
    rows.append(dict(
        name=name,
        ratio=float(np.median(c.gamma_mean / np.maximum(c.abs_alpha_mean, c.abs_beta_mean))),
        acc=float((p.gamma > np.maximum(p.alpha.abs(), p.beta.abs())).mean() * 100),
        auc=float(prev.loc[name, "auc"]),
        win=float(prev.loc[name, "win"]),
    ))

d = pd.DataFrame(rows).sort_values("ratio").reset_index(drop=True)
d.to_csv("paper_figures/fig_metric_data.csv", index=False)
print(d.round(4).to_string(index=False))
print("\nsaved paper_figures/fig_metric_data.csv")
