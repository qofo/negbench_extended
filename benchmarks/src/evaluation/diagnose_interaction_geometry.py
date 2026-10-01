"""
Why the interaction term is small: the geometry behind gamma << max(|alpha|,|beta|).

RESULTS 8.14 established *that* the two main effects bury the interaction, and
§(random-outer mechanism) recorded one number for *why* -- the shared component is
about 7.5x the state-difference signal. That number was a concept-level ratio, so it
folds together two very different causes and the manuscript could not say which:

  (a) **within-pair smallness.** The two images of a minimal pair differ by one
      inpainted object, so their embeddings are nearly identical and the per-pair
      difference vector is short to begin with.
  (b) **across-pair inconsistency.** Even if each pair has a healthy difference
      vector, the concept-level mean shrinks it further when those vectors point in
      scene-dependent directions.

The distinction matters for the paper's claim. (a) alone would say the interaction is
small because the control is tight -- a property of the benchmark, not of CLIP. (b)
says there is no shared "absence of dog" direction to find, which is the same fact
that makes the main effects scene-specific and therefore un-removable by any fixed or
concept-conditional W. This script separates them, and adds two references the ratio
needs to be interpreted at all:

  * the **cone baseline** -- cosine between images of *different* scenes, which is
    what any two CLIP embeddings share regardless of the pair construction;
  * a **randomly initialized encoder**, which tells whether the geometry is produced
    by contrastive training or is already there in the architecture.

The four Hadamard coefficients are inner products between each modality's mean and
difference vector (`eval_e2_hadamard_decomposition.compute_main_effect_ablation`):

    alpha = mu_I . d_T    beta = d_I . mu_T    gamma = d_I . d_T    m = mu_I . mu_T

so gamma is the only one bilinear in the two short vectors. Everything reported here
is a factor of that product.

Usage (from the repo root):
    python -m benchmarks.src.evaluation.diagnose_interaction_geometry \\
        --use_cache --min_pairs 20 \\
        --output_dir logs/evaluation/01_paper/2026-09-10_interaction_geometry
"""

import os
import json
import argparse
from typing import Dict

import numpy as np
import torch

try:
    from benchmarks.src.evaluation.eval_single_w_generalization import load_quads
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
    from analysis.model_loader import load_clip_for_eval
    from analysis.cli import (
        add_run_args, add_data_args, add_cache_args, add_restriction_args,
        add_concept_args,
    )
    from analysis.feature_cache import build_provenance
    from analysis.config import set_seed


def _norm(x):
    return np.linalg.norm(x, axis=-1)


def side_geometry(a: np.ndarray, b: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    """Mean/difference geometry of one modality's (state+, state-) embeddings.

    Returns the per-pair ratio, the concept-level ratio, and the directional
    consistency that separates them. With mu = (a+b)/2 and d = a-b,

        ||d|| / ||mu||  =  2 * sqrt((1-c) / (1+c))        for unit a, b with cos = c

    so the per-pair ratio is a restatement of how similar the two states are, while
    ``consistency`` = ||mean_pairs(d)|| / mean_pairs(||d||) is a genuinely different
    quantity: 1.0 means every pair of the concept moves the same way, 0 means the
    directions cancel.
    """
    mu = 0.5 * (a + b)
    d = a - b
    cos_ab = np.sum(a * b, axis=-1) / np.maximum(_norm(a) * _norm(b), 1e-12)
    per_pair = _norm(d) / np.maximum(_norm(mu), 1e-12)

    concept_ratio, consistency = [], []
    for g in np.unique(groups):
        m = groups == g
        d_bar = d[m].mean(axis=0)
        mu_bar = mu[m].mean(axis=0)
        concept_ratio.append(np.linalg.norm(d_bar) / max(np.linalg.norm(mu_bar), 1e-12))
        consistency.append(np.linalg.norm(d_bar) / max(_norm(d[m]).mean(), 1e-12))
    return dict(
        cos_between_states=float(cos_ab.mean()),
        per_pair_ratio=float(per_pair.mean()),
        concept_ratio=float(np.mean(concept_ratio)),
        consistency=float(np.mean(consistency)),
        mu_norm=float(_norm(mu).mean()),
        d_norm=float(_norm(d).mean()),
    )


def cone_baseline(v: np.ndarray, rng, n: int = 200000) -> Dict[str, float]:
    """Cosine between embeddings of two *different* scenes, and the ratio it implies.

    This is the anisotropy every CLIP embedding pair carries. Comparing it with the
    minimal-pair cosine says how much of the shared component comes from the images
    being the same scene rather than from the embedding space being a narrow cone.
    """
    i = rng.integers(0, len(v), size=n)
    j = rng.integers(0, len(v), size=n)
    keep = i != j
    i, j = i[keep], j[keep]
    c = np.sum(v[i] * v[j], axis=-1) / np.maximum(_norm(v[i]) * _norm(v[j]), 1e-12)
    c_bar = float(c.mean())
    return dict(cos_between_scenes=c_bar,
                implied_ratio=float(2.0 * np.sqrt(max(1 - c_bar, 0) / max(1 + c_bar, 1e-12))))


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    out = {}
    for tag, pretrained in [("pretrained", args.pretrained), ("random_init", None)]:
        if tag == "random_init" and not args.include_random_init:
            continue
        cache_kw = dict(model=args.model, pretrained=str(pretrained),
                        cache_dir=args.cache_dir, enabled=args.use_cache)
        model, preprocess, tokenizer = load_clip_for_eval(args.model, pretrained, device)
        v_pres, v_abs, t_pos, t_neg, groups = load_quads(
            args, model, preprocess, tokenizer, device, cache_kw)
        v_pres, v_abs, t_pos, t_neg = (np.asarray(x, dtype=np.float64)
                                       for x in (v_pres, v_abs, t_pos, t_neg))
        groups = np.asarray(groups)
        print(f"\n[{tag}] {len(v_pres)} pairs, {len(np.unique(groups))} concepts")

        img = side_geometry(v_pres, v_abs, groups)
        txt = side_geometry(t_pos, t_neg, groups)
        cone = cone_baseline(np.concatenate([v_pres, v_abs], axis=0), rng)
        out[tag] = dict(image=img, text=txt, cone=cone,
                        n_pairs=int(len(v_pres)), n_concepts=int(len(np.unique(groups))))

        print(f"  image  cos(pres,abs)={img['cos_between_states']:.4f}  "
              f"per-pair ||d||/||mu||={img['per_pair_ratio']:.4f}  "
              f"concept={img['concept_ratio']:.4f}  consistency={img['consistency']:.4f}")
        print(f"  text   cos(pos,neg)={txt['cos_between_states']:.4f}  "
              f"per-pair ||d||/||mu||={txt['per_pair_ratio']:.4f}  "
              f"concept={txt['concept_ratio']:.4f}  consistency={txt['consistency']:.4f}")
        print(f"  cone   cos(scene_i,scene_j)={cone['cos_between_scenes']:.4f} "
              f"-> implied ||d||/||mu||={cone['implied_ratio']:.4f}")

    out["provenance"] = build_provenance(args)
    path = os.path.join(args.output_dir, "interaction_geometry.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(ap, "logs/evaluation/interaction_geometry", seed=42, batch_size=128)
    add_data_args(
        ap,
        csv_path="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv",
        image_root="benchmarks/data/images")
    add_cache_args(ap)
    add_restriction_args(ap, "Concept set to share with the shared-W ladder")
    add_concept_args(ap, help_text="Minimum counterfactual pairs per object")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="openai")
    ap.add_argument("--include_random_init", action="store_true",
                    help="Also measure a randomly initialized encoder, which separates "
                         "geometry produced by contrastive training from geometry the "
                         "architecture has before any training.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
