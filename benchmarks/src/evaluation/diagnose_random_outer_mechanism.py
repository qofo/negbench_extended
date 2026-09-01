"""Why is the random-outer control (0.44%) so far below the 16.67% combinatorial reference?

`eval_binary_composition_ceiling.py --controls` reports that a rank-1 score built from two
random unit directions succeeds on only 0.44% of blocks -- lower even than the 0.67% empirical
no-binding null (`--crossconcept_null`). That is not a second coincidence needing its own
explanation; it follows from the algebra of the rank-1 construction plus the geometry of CLIP's
embedding space, and this script measures both pieces directly.

1. Algebraic necessary condition. For S_ab = a_state * b_state (a rank-1 score), the success
   rule min(S_++, S_--) > max(S_+-, S_-+) can only hold if BOTH modalities flip sign between
   their two states: (v_pres . w_I)(v_abs . w_I) < 0 AND (t_pos . w_T)(t_neg . w_T) < 0. If
   either side fails to flip, all four products from that side share a sign and the max/min
   split collapses -- success is not merely unlikely, it is impossible. This script confirms
   that empirically: it should measure P(success | not both flip) = 0.0000%.

2. Geometric reason a random direction rarely produces that flip. CLIP embeddings are not
   centred at the origin -- each concept's four embedding sets (v_pres, v_abs, t_pos, t_neg)
   share a large common component (their mean, ||mu||) that dominates an arbitrary projection,
   while the actual state-difference signal (d_I = mean(v_pres - v_abs), d_T = mean(t_pos -
   t_neg)) is comparatively small. A random w mostly measures the shared bulk, so the two
   states of a pair project to nearly the same value with the same sign -- no flip. The
   ||d||/||mu|| ratio, measured per concept, quantifies how small that signal is.

Usage (from the repo root, reusing the same cached embeddings as eval_binary_composition_ceiling.py):
    python -m benchmarks.src.evaluation.diagnose_random_outer_mechanism \
        --csv_path benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv \
        --output_dir logs/evaluation/01_paper/2026-09-01_random_outer_mechanism --use_cache
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import torch

try:
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.cli import (
        add_model_args, add_run_args, add_data_args, add_cache_args,
        add_restriction_args, add_concept_args,
    )
    from benchmarks.src.analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction,
    )
    from benchmarks.src.analysis.config import set_seed, coerce_bool_column
    from benchmarks.src.analysis.paths import resolve_image_path as resolve_path
    from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, encode_texts_unified,
    )
    from benchmarks.src.evaluation.eval_binary_composition_ceiling import (
        block_success, _random_unit,
    )
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.model_loader import load_clip_for_eval
    from analysis.cli import (
        add_model_args, add_run_args, add_data_args, add_cache_args,
        add_restriction_args, add_concept_args,
    )
    from analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction,
    )
    from analysis.config import set_seed, coerce_bool_column
    from analysis.paths import resolve_image_path as resolve_path
    from evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, encode_texts_unified,
    )
    from evaluation.eval_binary_composition_ceiling import block_success, _random_unit


def diagnose_concept(v_pres, v_abs, t_pos, t_neg, n_draws: int, rng: np.random.Generator) -> dict:
    """n_draws random (w_I, w_T) pairs scored on every pair of this concept."""
    n = len(v_pres)
    success, both_flip = [], []
    for _ in range(n_draws):
        w_i, w_t = _random_unit(v_pres.shape[1], rng), _random_unit(t_pos.shape[1], rng)
        a_p, a_m = v_pres @ w_i, v_abs @ w_i
        b_p, b_m = t_pos @ w_t, t_neg @ w_t
        flip_a = (a_p * a_m) < 0
        flip_b = (b_p * b_m) < 0
        both_flip.append(flip_a & flip_b)
        success.append(block_success(a_p * b_p, a_m * b_p, a_p * b_m, a_m * b_m))
    success = np.concatenate(success)
    both_flip = np.concatenate(both_flip)

    d_I = np.mean(v_pres - v_abs, axis=0)
    d_T = np.mean(t_pos - t_neg, axis=0)
    mu_I = np.mean(np.vstack([v_pres, v_abs]), axis=0)
    mu_T = np.mean(np.vstack([t_pos, t_neg]), axis=0)

    return {
        "n_pairs": n, "n_blocks": len(success),
        "success_rate": float(success.mean()),
        "both_flip_rate": float(both_flip.mean()),
        "success_given_both_flip": float(success[both_flip].mean()) if both_flip.any() else float("nan"),
        "success_given_not_both_flip": float(success[~both_flip].mean()) if (~both_flip).any() else float("nan"),
        "d_over_mu_image": float(np.linalg.norm(d_I) / (np.linalg.norm(mu_I) + 1e-12)),
        "d_over_mu_text": float(np.linalg.norm(d_T) / (np.linalg.norm(mu_T) + 1e-12)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Decompose why the random-outer control lands so far below chance")
    add_model_args(parser, "ViT-B-32", "openai")
    add_run_args(parser, "logs/evaluation/random_outer_mechanism", seed=42, batch_size=128)
    add_data_args(parser,
                  csv_path="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv",
                  image_root="benchmarks/data/images")
    add_cache_args(parser)
    add_restriction_args(parser, "Comma list, or path to txt/csv/json, limiting evaluation to an exact concept set")
    add_concept_args(parser)
    parser.add_argument("--n_draws", type=int, default=60,
                        help="Random (w_I, w_T) draws per concept")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_kw = dict(model=args.model, pretrained=args.pretrained,
                    cache_dir=args.cache_dir, enabled=args.use_cache)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.csv_path)
    coerce_bool_column(df, "object_in_image")
    concepts = [o for o in sorted(df["object_name"].unique()) if "," not in str(o)]
    restrict = load_object_restriction(args.restrict_objects)
    if restrict is not None:
        concepts = [o for o in concepts if o in set(restrict)]

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)

    rows = []
    for obj in concepts:
        d_obj = df[df["object_name"] == obj].reset_index(drop=True)
        d_true = d_obj[d_obj["object_in_image"] == True].reset_index(drop=True)
        d_false = d_obj[d_obj["object_in_image"] == False].reset_index(drop=True)
        n = min(len(d_true), len(d_false))
        if n < args.min_pairs:
            continue

        p_pres = [resolve_path(p, args.image_root) for p in d_true["image_path"].tolist()[:n]]
        p_abs = [resolve_path(p, args.image_root) for p in d_false["image_path"].tolist()[:n]]
        txt_pos = d_true["positive_caption"].tolist()[:n]
        txt_neg = d_true["negative_caption"].tolist()[:n]

        v_pres, _, m_p = cached_encode(
            lambda: encode_images_unified(model, preprocess, p_pres, device, args.batch_size),
            kind="image_pres@norm+raw+flags", items=p_pres, **cache_kw)
        v_abs, _, m_a = cached_encode(
            lambda: encode_images_unified(model, preprocess, p_abs, device, args.batch_size),
            kind="image_abs@norm+raw+flags", items=p_abs, **cache_kw)
        keep = np.where(m_p & m_a)[0]
        if len(keep) < args.min_pairs:
            continue
        v_pres, v_abs = v_pres[keep], v_abs[keep]
        txt_pos = [txt_pos[i] for i in keep]
        txt_neg = [txt_neg[i] for i in keep]

        t_pos, _ = cached_encode(
            lambda: encode_texts_unified(model, tokenizer, txt_pos, device, args.batch_size),
            kind="text_pos@norm+raw", items=txt_pos, **cache_kw)
        t_neg, _ = cached_encode(
            lambda: encode_texts_unified(model, tokenizer, txt_neg, device, args.batch_size),
            kind="text_neg@norm+raw", items=txt_neg, **cache_kw)

        res = diagnose_concept(v_pres, v_abs, t_pos, t_neg, args.n_draws, rng)
        res["object_name"] = obj
        rows.append(res)
        print(f"  [{obj:22s}] N={res['n_pairs']:4d} success={100*res['success_rate']:5.2f}% "
              f"both_flip={100*res['both_flip_rate']:5.2f}% "
              f"d/mu(img)={res['d_over_mu_image']:.3f}")

    if not rows:
        raise SystemExit("No concept met --min_pairs; nothing to report.")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.output_dir, "per_concept_mechanism.csv"), index=False)

    total_blocks = int(out["n_blocks"].sum())
    pooled_success = float((out["success_rate"] * out["n_blocks"]).sum() / total_blocks)
    pooled_both_flip = float((out["both_flip_rate"] * out["n_blocks"]).sum() / total_blocks)
    # Recombine the two conditional rates using each concept's block counts as weights would
    # need the raw flags; report the unweighted cross-concept mean instead, which is what the
    # paper cites (macro, not pooled, for the conditional rates).
    macro_given_flip = float(out["success_given_both_flip"].dropna().mean())
    macro_given_not_flip = float(out["success_given_not_both_flip"].dropna().mean())
    mean_d_over_mu = float(out["d_over_mu_image"].mean())

    summary = {
        "n_concepts": int(len(out)), "total_blocks": total_blocks,
        "pooled_success_rate": pooled_success,
        "pooled_both_flip_rate": pooled_both_flip,
        "macro_success_given_both_flip": macro_given_flip,
        "macro_success_given_not_both_flip": macro_given_not_flip,
        "mean_d_over_mu_image": mean_d_over_mu,
        "mean_d_over_mu_text": float(out["d_over_mu_text"].mean()),
        "provenance": build_provenance(args, n_concepts=int(len(out)), n_pairs=total_blocks),
    }
    with open(os.path.join(args.output_dir, "random_outer_mechanism.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print(f"  concepts={summary['n_concepts']}  total (pair x draw) blocks={total_blocks}")
    print(f"  pooled random_outer success rate       : {100*pooled_success:.2f}%")
    print(f"  P(both sides flip sign)                : {100*pooled_both_flip:.2f}%")
    print(f"  P(success | both flip)                 : {100*macro_given_flip:.2f}%  (expect ~50%)")
    print(f"  P(success | NOT both flip)              : {100*macro_given_not_flip:.4f}%  (expect ~0)")
    print(f"  mean ||d_I|| / ||mu_I|| across concepts : {mean_d_over_mu:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
