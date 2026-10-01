"""
What happens when both main effects are removed at once.

The success condition is an identity, not a model:

    Delta(S) = 2*gamma - 2*max(|alpha|, |beta|)      so   success <=> gamma > max(|alpha|,|beta|)

E3 (RESULTS 8.8) removed alpha and beta *separately* and concluded that suppressing a
main effect is not enough -- correctly, because whichever one is left still binds. The
pair of them together was never scored, and it is a different statement: with both at
zero the condition collapses to `gamma > 0`, so the accuracy stops depending on the
size of the interaction and depends only on its sign. RESULTS 8-AB already records that
gamma is the one coefficient whose sign is right (42/42 concepts, CI strictly positive),
which makes the empty cell an untested prediction rather than an unlikely one.

`compute_main_effect_ablation(mode="both")` has existed since E3 and
`eval_e2_hadamard_decomposition.py` already computes it; nothing converted it into the
2x2 accuracy of that coordinate. This script does exactly that, and asserts the identity
it implies -- the both-ablation accuracy must equal the fraction of pairs with gamma > 0.

Population is `load_quads` from `eval_single_w_generalization`, i.e. literally the arrays
the shared-W ladder (RESULTS 8-V.1) was scored on, so 4.03% and 29.19% from that table
and the numbers here refer to the same pairs.

Usage:
    python -m benchmarks.src.evaluation.eval_main_effect_ablation_ladder \\
        --use_cache --min_pairs 20 \\
        --output_dir logs/evaluation/01_paper/2026-09-03_main_effect_ablation_ladder
"""
import os
import json
import argparse
from typing import Dict

import numpy as np
import pandas as pd
import torch

try:
    from benchmarks.src.evaluation.eval_single_w_generalization import load_quads
    from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import (
        compute_main_effect_ablation,
    )
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.cli import (
        add_run_args, add_data_args, add_cache_args, add_restriction_args,
        add_concept_args,
    )
    from benchmarks.src.analysis.feature_cache import build_provenance
    from benchmarks.src.analysis.config import set_seed
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from evaluation.eval_single_w_generalization import load_quads
    from evaluation.eval_e2_hadamard_decomposition import compute_main_effect_ablation
    from analysis.model_loader import load_clip_for_eval
    from analysis.cli import (
        add_run_args, add_data_args, add_cache_args, add_restriction_args,
        add_concept_args,
    )
    from analysis.feature_cache import build_provenance
    from analysis.config import set_seed


MODES = ["none", "perobj_alpha", "perobj_beta", "both"]
LABELS = {
    "none": "없음 (코사인)",
    "global_alpha": "global_alpha (전역 이미지 평균)",
    "perobj_alpha": "perobj_alpha (쌍별 m_I)",
    "perobj_beta": "perobj_beta (쌍별 m_T)",
    "both": "both (α·β 동시)",
}


def cosine_coordinates(v_pres, v_abs, t_pos, t_neg) -> Dict[str, np.ndarray]:
    """Hadamard coordinates of the unmodified block, in the sign convention of E2."""
    s11 = np.sum(v_pres * t_pos, axis=1)
    s12 = np.sum(v_abs * t_pos, axis=1)
    s21 = np.sum(v_pres * t_neg, axis=1)
    s22 = np.sum(v_abs * t_neg, axis=1)
    return dict(alpha=(s11 + s12 - s21 - s22) / 4,
                beta=(s11 - s12 + s21 - s22) / 4,
                gamma=(s11 - s12 - s21 + s22) / 4)


def ablate_with_estimate(v_pres, v_abs, t_pos, t_neg, hat_mu_I, hat_mu_T):
    """The `both` ablation with the pair's own means replaced by supplied estimates.

    `compute_main_effect_ablation(mode="both")` projects against the pair's own m_I and
    m_T, which a deployable scorer cannot form. Substituting a coarser estimate keeps the
    intervention identical in shape and asks only how much of it survives when the
    direction is guessed rather than known. `hat_mu_I`/`hat_mu_T` are (N, D) unit vectors,
    broadcast per pair, so a global estimate is just the same row repeated.
    """
    mu_I = 0.5 * (v_pres + v_abs)
    u_img = 0.5 * (v_pres - v_abs)
    mu_T = 0.5 * (t_pos + t_neg)
    v_txt = 0.5 * (t_pos - t_neg)

    v_perp = v_txt - np.sum(v_txt * hat_mu_I, axis=-1, keepdims=True) * hat_mu_I
    u_perp = u_img - np.sum(u_img * hat_mu_T, axis=-1, keepdims=True) * hat_mu_T

    v_pres_a, v_abs_a = mu_I + u_perp, mu_I - u_perp
    t_pos_a, t_neg_a = mu_T + v_perp, mu_T - v_perp
    return cosine_coordinates(v_pres_a, v_abs_a, t_pos_a, t_neg_a)


def _unit_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def estimate_directions(v_pres, v_abs, t_pos, t_neg, groups, level):
    """Per-pair unit estimates of the image-mean and text-mean directions.

    `global`  one direction for every pair -- what a fixed W can encode.
    `concept` the concept's own mean -- needs to know which concept is being scored.
    `pair`    the pair's own mean, i.e. the oracle `both` ablation.
    """
    n = len(v_pres)
    img = 0.5 * (v_pres + v_abs)
    txt = 0.5 * (t_pos + t_neg)
    if level == "pair":
        return _unit_rows(img), _unit_rows(txt)
    if level == "global":
        return (np.repeat(_unit_rows(img.mean(0, keepdims=True)), n, axis=0),
                np.repeat(_unit_rows(txt.mean(0, keepdims=True)), n, axis=0))
    if level == "concept":
        hi, ht = np.empty_like(img), np.empty_like(txt)
        for c in np.unique(groups):
            m = groups == c
            hi[m] = _unit_rows(img[m].mean(0, keepdims=True))
            ht[m] = _unit_rows(txt[m].mean(0, keepdims=True))
        return hi, ht
    raise ValueError(f"unknown level: {level}")


def score(coords: Dict[str, np.ndarray], groups: np.ndarray) -> Dict[str, float]:
    """2x2 joint accuracy from the coordinates, pooled over pairs and macro over concepts."""
    alpha, beta, gamma = coords["alpha"], coords["beta"], coords["gamma"]
    delta = 2 * gamma - 2 * np.maximum(np.abs(alpha), np.abs(beta))
    hit = delta > 0
    per_concept = {str(c): float(hit[groups == c].mean() * 100)
                   for c in np.unique(groups)}
    return dict(
        acc_pooled_pct=float(hit.mean() * 100),
        acc_macro_pct=float(np.mean(list(per_concept.values()))),
        abs_alpha_pooled=float(np.abs(alpha).mean()),
        abs_beta_pooled=float(np.abs(beta).mean()),
        gamma_mean=float(gamma.mean()),
        gamma_positive_pct=float((gamma > 0).mean() * 100),
        per_concept=per_concept,
    )


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    cache_kw = dict(model=args.model, pretrained=str(args.pretrained),
                    cache_dir=args.cache_dir, enabled=args.use_cache)

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    v_pres, v_abs, t_pos, t_neg, groups = load_quads(
        args, model, preprocess, tokenizer, device, cache_kw)
    v_pres, v_abs, t_pos, t_neg = (np.asarray(x, dtype=np.float64)
                                   for x in (v_pres, v_abs, t_pos, t_neg))
    groups = np.asarray(groups)
    print(f"\n  {len(v_pres)} pairs across {len(np.unique(groups))} concepts\n")

    mu_I_global = v_pres.mean(axis=0) + v_abs.mean(axis=0)

    results = {}
    for mode in MODES:
        if mode == "none":
            coords = cosine_coordinates(v_pres, v_abs, t_pos, t_neg)
        else:
            coords = compute_main_effect_ablation(
                v_pres, v_abs, t_pos, t_neg, mu_I_global,
                mode=mode, renormalize=False)
        results[mode] = score(coords, groups)

    # The point of the experiment: with both main effects gone the condition is gamma > 0,
    # so the accuracy is the sign statistic of the interaction and nothing else.
    both = results["both"]
    assert abs(both["acc_pooled_pct"] - both["gamma_positive_pct"]) < 1e-9, (
        "both-ablation accuracy must equal the fraction of pairs with gamma > 0; got "
        f"{both['acc_pooled_pct']:.6f} vs {both['gamma_positive_pct']:.6f}")

    hdr = (f"  {'소거':<28}{'2x2 pooled':>13}{'2x2 macro':>12}"
           f"{'|alpha|':>12}{'|beta|':>12}{'gamma':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for mode in MODES:
        r = results[mode]
        print(f"  {LABELS[mode]:<28}{r['acc_pooled_pct']:13.2f}{r['acc_macro_pct']:12.2f}"
              f"{r['abs_alpha_pooled']:12.5f}{r['abs_beta_pooled']:12.5f}"
              f"{r['gamma_mean']:12.5f}")

    print(f"\n  gamma > 0 인 쌍: {both['gamma_positive_pct']:.2f}%  "
          f"= both 소거 정확도 (항등 확인됨)")

    # How much of the oracle survives when the two mean directions are estimated at a
    # coarser granularity? `pair` reproduces the `both` row above and is the upper bound.
    print(f"\n  {'평균 방향 추정 수준':<28}{'2x2 pooled':>13}{'2x2 macro':>12}")
    print("  " + "-" * 51)
    for level in ("global", "concept", "pair"):
        hi, ht = estimate_directions(v_pres, v_abs, t_pos, t_neg, groups, level)
        r = score(ablate_with_estimate(v_pres, v_abs, t_pos, t_neg, hi, ht), groups)
        results[f"estimate_{level}"] = r
        print(f"  {level:<28}{r['acc_pooled_pct']:13.2f}{r['acc_macro_pct']:12.2f}")

    rows = []
    for c in sorted(np.unique(groups)):
        row = {"concept": c, "n_pairs": int((groups == c).sum())}
        row.update({m: results[m]["per_concept"][str(c)] for m in MODES})
        rows.append(row)
    csv_path = os.path.join(args.output_dir, "main_effect_ablation_per_concept.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    summary = {m: {k: v for k, v in results[m].items() if k != "per_concept"}
               for m in results}
    summary["provenance"] = build_provenance(args)
    summary["provenance"]["n_pairs"] = int(len(v_pres))
    summary["provenance"]["n_concepts"] = int(len(np.unique(groups)))
    with open(os.path.join(args.output_dir, "main_effect_ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {args.output_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(ap, "logs/evaluation/main_effect_ablation_ladder", seed=42, batch_size=128)
    add_data_args(
        ap,
        csv_path="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv",
        image_root="benchmarks/data/images")
    add_cache_args(ap)
    add_restriction_args(ap, "Concept set to share with the shared-W ladder")
    add_concept_args(ap, help_text="Minimum counterfactual pairs per object")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="openai")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
