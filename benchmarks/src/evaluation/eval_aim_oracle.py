"""
How far does perfect aiming get, and what is the ceiling for a text-side method?

E5 split what an intervention can change into target (the angular measure of in-plane
directions satisfying a criterion, fixed by m_I and d_I) and aim (where d_T points).
Interventions varied 5x in aim while the caption target stayed in a 1.23x band. This
script asks what aiming alone could achieve.

The criterion turns out to have an exact and much more useful form. Writing it out,

    gamma > |alpha|
      <=>  d_I . d_T > |m_I . d_T|
      <=>  (d_I - m_I) . d_T > 0  and  (d_I + m_I) . d_T > 0
      <=>  v_+ . d_T > 0  and  v_- . d_T < 0                                    (*)

since v_+ = m_I + d_I and v_- = m_I - d_I. So counterfactual caption discrimination is
*identical* to asking whether the text polarity direction separates the
object-present image from the object-absent image by a hyperplane through the origin.
Aiming is linear separation, and the ceiling is a bias-free linear classifier.

That fixes the unit of analysis. In the single-object coordinate every pair of a
concept carries the identical caption pair, so a function of the text alone emits one
d_T per concept -- and by (*) the best any such function can do is the accuracy of the
best bias-free linear separator of that concept's present/absent images. Three levels:

  global       one d_T shared by all concepts.
  per-concept  one d_T per concept. **The exact ceiling for a deterministic text-side
               method here.** Reported both in-sample (the oracle) and under
               GroupKFold over base scenes (what a method could actually learn).
  per-pair     a different d_T per pair; needs to see the images, so it is an absolute
               bound rather than a reachable one. By (*) it is satisfiable whenever
               v_+ and v_- are not positive multiples, i.e. essentially always.

The gap between per-concept and per-pair is exactly the information a text-side
objective lacks: which of a concept's image pairs it is being scored against.

For the image and group criteria the same d_T is used with the observed m_T, so those
columns are a pure re-aiming of the polarity direction and nothing else.

Usage:
    python -m benchmarks.src.evaluation.eval_aim_oracle \\
        --restrict_objects logs/evaluation/01_paper/2026-08-28_e2_hadamard_decomposition/e2_per_concept_decomposition.csv \\
        --use_cache --output_dir logs/evaluation/01_paper/2026-09-03_aim_oracle
"""
import os
import json
import argparse
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

try:
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.cli import (
        add_run_args, add_data_args, add_cache_args, add_restriction_args,
        add_concept_args,
    )
    from benchmarks.src.analysis.feature_cache import build_provenance, load_object_restriction
    from benchmarks.src.analysis.config import set_seed, coerce_bool_column
    from benchmarks.src.evaluation.eval_target_and_aim import MODELS, load_pairs
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.model_loader import load_clip_for_eval
    from analysis.cli import (
        add_run_args, add_data_args, add_cache_args, add_restriction_args,
        add_concept_args,
    )
    from analysis.feature_cache import build_provenance, load_object_restriction
    from analysis.config import set_seed, coerce_bool_column
    from evaluation.eval_target_and_aim import MODELS, load_pairs


def fit_direction(v_p: np.ndarray, v_m: np.ndarray, seed: int) -> np.ndarray:
    """
    Best bias-free separator of present from absent, i.e. the best aim by (*).

    Logistic regression with fit_intercept=False is exactly the right estimator: the
    criterion is a sign condition through the origin, so an intercept would answer a
    different question.
    """
    X = np.concatenate([v_p, v_m])
    y = np.concatenate([np.ones(len(v_p)), np.zeros(len(v_m))])
    clf = LogisticRegression(fit_intercept=False, max_iter=2000, C=1.0,
                             random_state=seed)
    clf.fit(X, y)
    return clf.coef_[0]


def caption_hits(v_p: np.ndarray, v_m: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Condition (*): positive on the present image, negative on the absent one."""
    return (v_p @ w > 0) & (v_m @ w < 0)


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    df = coerce_bool_column(pd.read_csv(args.csv_path), "object_in_image")
    restriction = load_object_restriction(args.restrict_objects)
    objects = sorted(df["object_name"].unique())
    if restriction:
        objects = [o for o in objects if o in set(restriction)]
    print(f"\n  Concepts: {len(objects)}\n")

    rows = []
    for name, arch, pretrained, is_intervention in MODELS:
        model, preprocess, tokenizer = load_clip_for_eval(arch, pretrained, device)
        cache_kw = dict(model=arch, pretrained=pretrained,
                        enabled=args.use_cache, cache_dir=args.cache_dir)
        m_I, d_I, m_T, d_T, groups = load_pairs(
            df, objects, model, preprocess, tokenizer, device, args, cache_kw)
        v_p, v_m = m_I + d_I, m_I - d_I

        a = np.sum(m_I * d_T, axis=-1)
        b = np.sum(d_I * m_T, axis=-1)
        g = np.sum(d_I * d_T, axis=-1)
        act_cap = (g > np.abs(a))
        act_img = (g > np.abs(b))

        # per-pair: (*) is satisfiable unless v_+ and v_- are positive multiples
        gram = np.sum(v_p * v_m, axis=-1)
        nrm = np.linalg.norm(v_p, axis=-1) * np.linalg.norm(v_m, axis=-1)
        per_pair = float((gram / np.maximum(nrm, 1e-12) < 1 - 1e-9).mean() * 100)

        # global: one direction for every concept
        w_g = fit_direction(v_p, v_m, args.seed)
        cap_global = caption_hits(v_p, v_m, w_g)

        # per-concept, in-sample (the oracle) and held out over base scenes
        cap_pc = np.zeros(len(v_p), dtype=bool)
        cap_cv = np.zeros(len(v_p), dtype=bool)
        img_pc = np.zeros(len(v_p), dtype=bool)
        for c in np.unique(groups):
            s = np.where(groups == c)[0]
            w = fit_direction(v_p[s], v_m[s], args.seed)
            cap_pc[s] = caption_hits(v_p[s], v_m[s], w)
            # the re-aimed d_T keeps its observed magnitude, so only the aim changes
            w_hat = w / max(np.linalg.norm(w), 1e-12)
            dt_new = w_hat[None, :] * np.linalg.norm(d_T[s], axis=-1, keepdims=True)
            img_pc[s] = (np.sum(d_I[s] * dt_new, -1) > np.abs(b[s]))

            if len(s) >= 10:
                folds = min(5, len(s))
                for tr, te in GroupKFold(n_splits=folds).split(
                        v_p[s], groups=np.arange(len(s)) % folds):
                    w_tr = fit_direction(v_p[s][tr], v_m[s][tr], args.seed)
                    cap_cv[s[te]] = caption_hits(v_p[s][te], v_m[s][te], w_tr)
            else:
                cap_cv[s] = cap_pc[s]

        row = dict(
            name=name, family="intervention" if is_intervention else "backbone",
            caption_actual=float(act_cap.mean() * 100),
            caption_global=float(cap_global.mean() * 100),
            caption_per_concept=float(cap_pc.mean() * 100),
            caption_per_concept_cv=float(cap_cv.mean() * 100),
            caption_per_pair=per_pair,
            image_actual=float(act_img.mean() * 100),
            image_per_concept=float(img_pc.mean() * 100),
            group_actual=float((act_cap & act_img).mean() * 100),
            group_per_concept=float((cap_pc & img_pc).mean() * 100),
        )
        rows.append(row)
        print(f"  {name:<20} caption  actual {row['caption_actual']:6.2f}  "
              f"global {row['caption_global']:6.2f}  per-concept {row['caption_per_concept']:6.2f} "
              f"(CV {row['caption_per_concept_cv']:6.2f})  per-pair {per_pair:6.2f}")
        del model
        torch.cuda.empty_cache()

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(args.output_dir, "aim_oracle.csv"), index=False)

    print("\n" + "=" * 100)
    print(f"  {'model':<20}{'cap act':>9}{'global':>9}{'per-cpt':>9}{'CV':>8}"
          f"{'img act':>9}{'img pc':>9}{'grp act':>9}{'grp pc':>9}")
    for _, r in d.iterrows():
        print(f"  {r['name']:<20}{r['caption_actual']:9.2f}{r['caption_global']:9.2f}"
              f"{r['caption_per_concept']:9.2f}{r['caption_per_concept_cv']:8.2f}"
              f"{r['image_actual']:9.2f}{r['image_per_concept']:9.2f}"
              f"{r['group_actual']:9.2f}{r['group_per_concept']:9.2f}")

    b = d[d.name == "ViT-B/32 (OpenAI)"].iloc[0]
    best = d.loc[d.caption_actual.idxmax()]
    print(f"\n  --- frozen ViT-B/32: aiming alone ---")
    print(f"    caption  {b['caption_actual']:.2f}%  ->  text-side ceiling "
          f"{b['caption_per_concept']:.2f}%  (held out {b['caption_per_concept_cv']:.2f}%)")
    print(f"    group    {b['group_actual']:.2f}%  ->  {b['group_per_concept']:.2f}%")
    print(f"    best observed aimer: {best['name']} at {best['caption_actual']:.2f}%")

    with open(os.path.join(args.output_dir, "aim_oracle.json"), "w") as f:
        json.dump(dict(rows=d.to_dict(orient="records"),
                       provenance=build_provenance(args)), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {args.output_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(ap, "logs/evaluation/aim_oracle", seed=42, batch_size=128)
    add_data_args(ap, csv_path="benchmarks/data/images/beaf_counterfactual_6col.csv",
                  image_root="benchmarks/data/images")
    add_cache_args(ap)
    add_restriction_args(ap, "Concept set to share with the e2 runs")
    add_concept_args(ap, help_text="Minimum counterfactual pairs per object")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
