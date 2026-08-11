"""
BEAF Counterfactual Statistical Analysis Module.

Provides pure numerical computation functions for:
- 2x2 Factorial ANOVA (Text Main Effect, Visual Main Effect, Interaction Effect)
- Quadrant Bootstrap Confidence Intervals
"""

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd


def compute_2x2_factorial_anova(
    sim_orig_pos: np.ndarray,
    sim_orig_neg: np.ndarray,
    sim_cf_pos: np.ndarray,
    sim_cf_neg: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute 2x2 Factorial ANOVA Main Effects & Interaction Effect per sample pair with 95% Bootstrap CI.

    Raw 2x2 Similarity Matrix per pair:
                caption=pos   caption=neg
    image=orig      A (orig_pos)  B (orig_neg)
    image=cf        C (cf_pos)    D (cf_neg)

    3 Orthogonal Derived Metrics:
    - Text Main Effect    = ((A - B) + (C - D)) / 2
    - Visual Main Effect  = ((A - C) + (B - D)) / 2
    - Interaction Effect = (A - B) - (C - D) == (A - C) - (B - D)
    """
    A = sim_orig_pos
    B = sim_orig_neg
    C = sim_cf_pos
    D = sim_cf_neg

    text_main_effect   = ((A - B) + (C - D)) / 2.0
    visual_main_effect = ((A - C) + (B - D)) / 2.0
    interaction_effect = (A - B) - (C - D)

    anova_df = pd.DataFrame({
        "sim_A_orig_pos":       A,
        "sim_B_orig_neg":       B,
        "sim_C_cf_pos":         C,
        "sim_D_cf_neg":         D,
        "text_main_effect":     text_main_effect,
        "visual_main_effect":   visual_main_effect,
        "interaction_effect":   interaction_effect,
    })

    n = len(A)
    rng = np.random.default_rng(seed=seed)

    boot_t, boot_v, boot_i = [], [], []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        boot_t.append(np.mean(text_main_effect[idx]))
        boot_v.append(np.mean(visual_main_effect[idx]))
        boot_i.append(np.mean(interaction_effect[idx]))

    summary_anova = {
        "text_main_effect": {
            "mean": round(float(np.mean(text_main_effect)), 6),
            "std":  round(float(np.std(text_main_effect)), 6),
            "ci_95_low": round(float(np.percentile(boot_t, 2.5)), 6),
            "ci_95_high": round(float(np.percentile(boot_t, 97.5)), 6),
        },
        "visual_main_effect": {
            "mean": round(float(np.mean(visual_main_effect)), 6),
            "std":  round(float(np.std(visual_main_effect)), 6),
            "ci_95_low": round(float(np.percentile(boot_v, 2.5)), 6),
            "ci_95_high": round(float(np.percentile(boot_v, 97.5)), 6),
        },
        "interaction_effect": {
            "mean": round(float(np.mean(interaction_effect)), 6),
            "std":  round(float(np.std(interaction_effect)), 6),
            "ci_95_low": round(float(np.percentile(boot_i, 2.5)), 6),
            "ci_95_high": round(float(np.percentile(boot_i, 97.5)), 6),
            "negative_interaction_pct": round(float(np.mean(interaction_effect < 0) * 100), 2),
        },
    }

    return anova_df, summary_anova


def compute_quadrant_bootstrap_ci(
    sim_orig_pos: np.ndarray,
    sim_orig_neg: np.ndarray,
    sim_cf_pos: np.ndarray,
    margin: float = 0.01,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Calculate quadrant proportions with noise margin and 95% Bootstrap Confidence Intervals.

    delta_text = sim(orig, pos) - sim(orig, neg): text discriminability on original image
    delta_vis  = sim(orig, pos) - sim(cf, pos):   visual sensitivity to object removal

    Quadrant definitions (with margin):
      Q1 (Both Sensitive)  : delta_text > margin  AND  delta_vis > margin
      Q2 (Visual-Only)     : delta_text <= margin  AND  delta_vis > margin
      Q3 (Neither)         : delta_text <= margin  AND  delta_vis <= margin
      Q4 (Text-Only)       : delta_text > margin   AND  delta_vis <= margin
    """
    delta_text = sim_orig_pos - sim_orig_neg
    delta_vis  = sim_orig_pos - sim_cf_pos
    n = len(delta_text)

    def _get_quadrants(dt, dv):
        q1 = (dt > margin) & (dv > margin)
        q2 = (dt <= margin) & (dv > margin)
        q3 = (dt <= margin) & (dv <= margin)
        q4 = (dt > margin) & (dv <= margin)
        q_near_zero = (np.abs(dt) <= margin) | (np.abs(dv) <= margin)
        return {
            "q1_both_sensitive_pct": float(np.mean(q1) * 100),
            "q2_visual_only_pct":    float(np.mean(q2) * 100),
            "q3_neither_pct":        float(np.mean(q3) * 100),
            "q4_text_only_pct":      float(np.mean(q4) * 100),
            "near_zero_margin_pct":  float(np.mean(q_near_zero) * 100),
        }

    point_estimates = _get_quadrants(delta_text, delta_vis)

    rng = np.random.default_rng(seed=seed)
    boot_dist = {k: [] for k in point_estimates}
    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        q_boot = _get_quadrants(delta_text[boot_idx], delta_vis[boot_idx])
        for k, v in q_boot.items():
            boot_dist[k].append(v)

    summary_ci = {}
    for k, v in point_estimates.items():
        low  = float(np.percentile(boot_dist[k], 2.5))
        high = float(np.percentile(boot_dist[k], 97.5))
        summary_ci[k] = {
            "mean_pct":   round(v, 2),
            "ci_95_low":  round(low, 2),
            "ci_95_high": round(high, 2),
        }
    return summary_ci
