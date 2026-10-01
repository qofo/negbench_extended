"""
Do the alignment rotation and the main-effect ablation transfer to NegBench MCQ?

Table 2 of the 2-page draft scores two interventions on the controlled 2x2 only:

  * the closed-form per-object rotation R^(o) with cos(d_I, R d_T) = 1, and
  * the exact removal of one or both main effects.

Neither was ever scored on a benchmark, so "the bottleneck is main-effect magnitude"
rests on a metric no downstream task uses. This script puts both on NegBench COCO MCQ
and on the controlled 2x2 in the same run, so the two axes are directly comparable.

The obstacle is that both interventions are defined on a *quad* (v+, v-, t+, t-): the
oracle needs the pair's own m_I = (v+ + v-)/2 and d_I = (v+ - v-)/2, and an MCQ item has
one image and four captions, so neither exists. Every condition here is therefore the
per-object generalisation, with the four moments estimated once per concept from BEAF
and then applied to individual embeddings:

    rotation      t' = R^(o) t                                    (text side, exact)
    zero-alpha    t' = m_T^(o) + P_perp(mu_I^(o)) (t - m_T^(o))    (text side)
    zero-beta     v' = m_I^(o) + P_perp(mu_T^(o)) (v - m_I^(o))    (image side)

Restricted to a BEAF quad these reduce to `compute_main_effect_ablation`, so the 2x2
column reproduces the oracle numbers and the MCQ column is the same transform carried
to a task where the quad is unavailable. `--scope global` replaces the per-concept
moments with one set shared by every item, which is what a deployed scorer could use.

MCQ prediction goes through `scoring_heads.predict_with_tie_report`, never argmax:
`correct_answer` is 0 for every row of the canonical CSV, so a scorer that ties would
otherwise score 100%. Tie rates are reported alongside every accuracy.

Usage:
    python -m benchmarks.src.evaluation.eval_alignment_ablation_on_mcq \
        --use_cache --min_pairs 20 \
        --output_dir logs/evaluation/01_paper/2026-09-05_alignment_ablation_on_mcq
"""
import os
import re
import json
import argparse
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from benchmarks.src.evaluation.eval_single_w_generalization import load_quads
    from benchmarks.src.evaluation.eval_per_object_alignment_intervention import (
        build_closed_form_rotation,
    )
    from benchmarks.src.evaluation.scoring_heads import predict_with_tie_report
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
    from evaluation.eval_per_object_alignment_intervention import build_closed_form_rotation
    from evaluation.scoring_heads import predict_with_tie_report
    from analysis.model_loader import load_clip_for_eval
    from analysis.cli import (
        add_run_args, add_data_args, add_cache_args, add_restriction_args,
        add_concept_args,
    )
    from analysis.feature_cache import build_provenance
    from analysis.config import set_seed

MCQ_CACHE = "logs/evaluation/cached_embeddings/COCO_val_mcq_llama3.1_rephrased_embeds.pt"
CONDITIONS = ["cosine", "rotation", "zero_alpha", "zero_beta", "zero_both",
              "rotation_zero_both"]


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def concept_moments(v_pres, v_abs, t_pos, t_neg, groups) -> Dict[str, Dict[str, np.ndarray]]:
    """The four per-concept moments the interventions are built from."""
    out = {}
    for obj in np.unique(groups):
        m = groups == obj
        out[str(obj)] = dict(
            mu_I=0.5 * (v_pres[m] + v_abs[m]).mean(0),
            d_I=0.5 * (v_pres[m] - v_abs[m]).mean(0),
            mu_T=0.5 * (t_pos[m] + t_neg[m]).mean(0),
            d_T=0.5 * (t_pos[m] - t_neg[m]).mean(0),
        )
    out["__global__"] = dict(
        mu_I=0.5 * (v_pres + v_abs).mean(0), d_I=0.5 * (v_pres - v_abs).mean(0),
        mu_T=0.5 * (t_pos + t_neg).mean(0), d_T=0.5 * (t_pos - t_neg).mean(0),
    )
    return out


def make_transform(mom: Dict[str, np.ndarray], condition: str):
    """Return (text_fn, image_fn); both act on individual embeddings."""
    d = len(mom["mu_I"])
    R = build_closed_form_rotation(mom["d_T"], mom["d_I"]) if "rotation" in condition \
        else np.eye(d)
    hat_mu_I, hat_mu_T = _unit(mom["mu_I"]), _unit(mom["mu_T"])
    kill_a = condition in ("zero_alpha", "zero_both", "rotation_zero_both")
    kill_b = condition in ("zero_beta", "zero_both", "rotation_zero_both")

    def text_fn(t):
        t = t @ R.T
        if kill_a:
            # the rotation moved the text, so project off mu_I in the rotated frame
            mT = mom["mu_T"] @ R.T
            r = t - mT
            t = mT + r - np.outer(r @ hat_mu_I, hat_mu_I)
        return t

    def image_fn(v):
        if kill_b:
            r = v - mom["mu_I"]
            v = mom["mu_I"] + r - np.outer(r @ hat_mu_T, hat_mu_T)
        return v

    return text_fn, image_fn


def score_2x2(v_pres, v_abs, t_pos, t_neg, groups, moments, condition, scope):
    """Pooled Delta(S) > 0 accuracy under the condition."""
    ok = np.zeros(len(v_pres), dtype=bool)
    for obj in np.unique(groups):
        m = groups == obj
        key = "__global__" if scope == "global" else str(obj)
        tf, vf = make_transform(moments[key], condition)
        vp, va = vf(v_pres[m]), vf(v_abs[m])
        tp, tn = tf(t_pos[m]), tf(t_neg[m])
        s11 = np.sum(vp * tp, -1); s12 = np.sum(va * tp, -1)
        s21 = np.sum(vp * tn, -1); s22 = np.sum(va * tn, -1)
        ok[m] = np.minimum(s11, s22) > np.maximum(s12, s21)
    return float(ok.mean() * 100)


def assign_objects(captions_per_item, concepts) -> np.ndarray:
    """Name the BEAF concept an MCQ item is about; '' when it is not exactly one."""
    pats = {c: re.compile(r"\b" + re.escape(c) + r"s?\b", re.I) for c in concepts}
    out = []
    for caps in captions_per_item:
        joined = " ".join(caps)
        hits = [c for c, p in pats.items() if p.search(joined)]
        out.append(hits[0] if len(hits) == 1 else "")
    return np.array(out)


def score_mcq(img, txt, qtypes, objs, moments, condition, scope, seed):
    """MCQ accuracy under the condition; index 0 is always the ground truth."""
    n = len(img)
    scores = np.zeros((n, txt.shape[1]))
    keys = ["__global__"] * n if scope == "global" else list(objs)
    for key in sorted(set(keys)):
        m = np.array([k == key for k in keys])
        tf, vf = make_transform(moments[key], condition)
        v = vf(img[m])                                  # (M, D)
        t = tf(txt[m].reshape(-1, txt.shape[-1])).reshape(m.sum(), txt.shape[1], -1)
        scores[m] = np.einsum("md,mkd->mk", v, t)
    pred, tie = predict_with_tie_report(torch.tensor(scores, dtype=torch.float64), seed=seed)
    correct = (pred == 0)
    res = {"avg": float(correct.mean() * 100), "tie": float(np.mean(tie) * 100)}
    for q in ("positive", "negative", "hybrid"):
        m = np.array([x == q for x in qtypes])
        res[q] = float(correct[m].mean() * 100) if m.any() else float("nan")
    return res


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
    concepts = sorted(np.unique(groups).tolist())
    print(f"\n  BEAF 2x2: {len(v_pres)} pairs, {len(concepts)} concepts")

    moments = concept_moments(v_pres, v_abs, t_pos, t_neg, groups)

    d = torch.load(MCQ_CACHE, map_location="cpu")
    img = d["img_embeds"].double().numpy()
    txt = d["text_embeds"].double().numpy()
    qtypes = d["question_types"]
    mcq_df = pd.read_csv(args.mcq_csv)
    caps = mcq_df[[f"caption_{i}" for i in range(4)]].astype(str).values.tolist()
    assert len(caps) == len(img), "MCQ cache and CSV are out of sync"
    objs = assign_objects(caps, concepts)
    covered = objs != ""
    print(f"  COCO MCQ: {len(img)} items, {covered.sum()} ({covered.mean()*100:.1f}%) "
          f"name exactly one of the {len(concepts)} concepts\n")

    rows = []
    for scope in ("global", "per_object"):
        for cond in CONDITIONS:
            acc2 = score_2x2(v_pres, v_abs, t_pos, t_neg, groups, moments, cond, scope)
            if scope == "global":
                mcq = score_mcq(img, txt, qtypes, objs, moments, cond, scope, args.seed)
                sub = "all"
            else:
                mcq = score_mcq(img[covered], txt[covered],
                                [q for q, c in zip(qtypes, covered) if c],
                                objs[covered], moments, cond, scope, args.seed)
                sub = "concept-matched"
            rows.append(dict(scope=scope, condition=cond, mcq_subset=sub,
                             acc_2x2=acc2, **{f"mcq_{k}": v for k, v in mcq.items()}))

    df = pd.DataFrame(rows)
    hdr = (f"  {'scope':<11}{'condition':<20}{'2x2':>8}{'MCQ avg':>9}{'pos':>8}"
           f"{'neg':>8}{'hyb':>8}{'tie%':>7}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for _, r in df.iterrows():
        print(f"  {r['scope']:<11}{r['condition']:<20}{r['acc_2x2']:8.2f}"
              f"{r['mcq_avg']:9.2f}{r['mcq_positive']:8.2f}{r['mcq_negative']:8.2f}"
              f"{r['mcq_hybrid']:8.2f}{r['mcq_tie']:7.2f}")

    df.to_csv(os.path.join(args.output_dir, "alignment_ablation_on_mcq.csv"), index=False)
    summary = {"rows": rows,
               "mcq_coverage_pct": float(covered.mean() * 100),
               "n_concepts": len(concepts), "n_pairs": int(len(v_pres)),
               "provenance": build_provenance(args)}
    with open(os.path.join(args.output_dir, "alignment_ablation_on_mcq.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {args.output_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(ap, "logs/evaluation/alignment_ablation_on_mcq", seed=42, batch_size=128)
    add_data_args(ap,
                  csv_path="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv",
                  image_root="benchmarks/data/images")
    add_cache_args(ap)
    add_restriction_args(ap, "Concept set to share with the shared-W ladder")
    add_concept_args(ap, help_text="Minimum counterfactual pairs per object")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="openai")
    ap.add_argument("--mcq_csv", type=str,
                    default="benchmarks/data/images/COCO_val_mcq_llama3.1_rephrased.csv")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
