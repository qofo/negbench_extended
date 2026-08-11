"""
BEAF Counterfactual Statistical Analysis Module.

Provides pure numerical computation functions for:
- 2x2 Factorial ANOVA (Text Main Effect, Visual Main Effect, Interaction Effect)
- Quadrant Bootstrap Confidence Intervals
- Per-Object Layerwise Cosine Similarity & Linear Probing Statistics
"""

from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


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
        "independence_diagnostics": {
            # r(text_main, visual_main): ideally near 0 — large |r| means effects are not orthogonal.
            # CLIP's joint training objective may cause systematic correlation here.
            "r_text_vs_visual_main_effect": round(float(scipy_stats.pearsonr(text_main_effect, visual_main_effect)[0]), 6),
            # r(A-B, C-D): measures whether text discriminability (A-B) is consistent
            # regardless of which image is shown. Near 0 = assumption holds.
            "r_AB_vs_CD": round(float(scipy_stats.pearsonr(A - B, C - D)[0]), 6),
            "interpretation": (
                "⚠️ High |r| (>0.3) suggests ANOVA orthogonality assumption may be violated. "
                "Text and visual effects may share a common latent cause in CLIP's representation."
            ),
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




def compute_per_object_layerwise_stats(
    vis_orig,
    vis_cf,
    df_pairs,
    seed=42,
):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score

    def _get_feats(vis, key):
        if key in vis["layers"]:
            return vis["layers"][key]
        elif key == "Pre-Projection":
            return vis["pre_proj"]
        else:
            return vis["final_l2norm"]

    layer_keys = list(vis_orig["layers"].keys())
    all_keys   = layer_keys + ["Pre-Projection", "+Final L2Norm"]

    object_names   = df_pairs["object_name"].values
    pair_ids       = df_pairs["pair_id"].values if "pair_id" in df_pairs.columns else np.arange(len(df_pairs))
    unique_objects = sorted(df_pairs["object_name"].unique().tolist())

    per_cos = {k: [] for k in all_keys}
    per_prb = {k: [] for k in all_keys}
    cos_obj = {k: {} for k in all_keys}
    prb_obj = {k: {} for k in all_keys}

    print(f"  [Per-Object Layerwise (GroupKFold pair_id)] {len(unique_objects)} unique objects found.")

    raw_records = []
    for obj in unique_objects:
        mask      = (object_names == obj)
        n_obj     = int(np.sum(mask))
        obj_pairs = pair_ids[mask]

        if n_obj < 2:
            print(f"    Skip {repr(obj)} (n={n_obj} < 2 pairs).")
            continue

        groups_all = np.concatenate([obj_pairs, obj_pairs])
        n_unique_groups = len(np.unique(obj_pairs))
        n_folds = min(5, n_unique_groups)

        for lk in all_keys:
            X_o = _get_feats(vis_orig, lk)[mask]
            X_c = _get_feats(vis_cf,   lk)[mask]

            X_o_n = X_o / (np.linalg.norm(X_o, axis=1, keepdims=True) + 1e-8)
            X_c_n = X_c / (np.linalg.norm(X_c, axis=1, keepdims=True) + 1e-8)

            cos_mean = float(np.mean(np.sum(X_o_n * X_c_n, axis=1)))

            X_all = np.vstack([X_o_n, X_c_n])
            y_all = np.array([1] * n_obj + [0] * n_obj)

            if n_folds < 2:
                probe_acc = 50.0
            else:
                gkf = GroupKFold(n_splits=n_folds)
                cv_splits = list(gkf.split(X_all, y_all, groups=groups_all))
                clf = LogisticRegression(C=0.1, max_iter=1000, random_state=seed)
                scores = cross_val_score(clf, X_all, y_all, cv=cv_splits, scoring="accuracy")
                probe_acc = float(np.mean(scores) * 100)

            per_cos[lk].append(cos_mean)
            per_prb[lk].append(probe_acc)
            cos_obj[lk][obj] = cos_mean
            prb_obj[lk][obj] = probe_acc
            raw_records.append({
                "object_name":     obj,
                "layer_name":      lk,
                "n_pairs":         n_obj,
                "cosine_sim_mean": cos_mean,
                "probe_acc_pct":   probe_acc,
            })
        print(f"    OK {repr(obj)} (n={n_obj} pairs, {n_folds}-fold GroupKFold)")

    raw_df = pd.DataFrame(raw_records)

    summary = {}
    for lk in all_keys:
        c_vals = per_cos[lk]
        p_vals = per_prb[lk]
        summary[lk] = {
            "cosine_sim": {
                "mean":       round(float(np.mean(c_vals)),  6) if c_vals else float("nan"),
                "std":        round(float(np.std(c_vals)),   6) if c_vals else float("nan"),
                "per_object": cos_obj[lk],
            },
            "probe_acc": {
                "mean":       round(float(np.mean(p_vals)), 4) if p_vals else float("nan"),
                "std":        round(float(np.std(p_vals)),  4) if p_vals else float("nan"),
                "per_object": prb_obj[lk],
            },
        }

    n_obj_valid = len(raw_df["object_name"].unique()) if len(raw_df) > 0 else 0
    print(f"  [Per-Object Layerwise] Complete: {len(raw_df)} records, "
          f"{n_obj_valid} objects x {len(all_keys)} layers.")
    return raw_df, summary
