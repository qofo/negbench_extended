"""
Is polarity still linearly readable in PeakPatch's corrected text embedding?

Section 4.1 of the paper reports that a per-object linear probe recovers caption
polarity from the frozen CLIP text embedding at macro 0.86. PeakPatch (Lu et al.,
ECCV 2026) replaces that embedding with an ECN-corrected one and lifts NegBench
COCO negation accuracy from 6.8 to 54.7 (our reproduction). The 2x2 audit
(eval_peakpatch_2x2_audit.py) showed the correction rebuilds the polarity
direction almost orthogonally to the original (cos 0.012) while leaving the
caption-selection condition gamma > |alpha| in place.

That leaves one question the audit does not answer: does the corrected embedding
carry *more* linearly readable polarity, the same, or less? A drop would mean the
MCQ gain comes from something other than a cleaner polarity representation; a rise
would mean readability improved without the decision boundary following.

This script answers it directly. For each concept it takes the 2x2's text control
-- the positive and negative caption of the same row, same word set, different
binding -- encodes both with plain CLIP and with the ECN, and fits the same paired
per-object probe used for Table 1 (GroupKFold on the pair, so the two halves of a
minimal pair never straddle the split).

Text only: no images are encoded, so the concept set is the CSV's, restricted the
same way as the audit, without the audit's image-availability filter.

Usage:
    python -m benchmarks.src.evaluation.eval_peakpatch_text_probe \
        --peakpatch_root PeakPatch \
        --restrict_objects logs/evaluation/01_paper/2026-08-28_e2_hadamard_decomposition/e2_per_concept_decomposition.csv \
        --output_dir logs/evaluation/01_paper/2026-09-06_peakpatch_text_probe
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
import torch

try:
    from benchmarks.src.analysis.cli import (
        add_model_args, add_run_args, add_data_args, add_restriction_args, add_concept_args)
    from benchmarks.src.analysis.feature_cache import load_object_restriction, build_provenance
    from benchmarks.src.analysis.config import set_seed, coerce_bool_column
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.evaluation.eval_peakpatch_2x2_audit import (
        load_peakpatch, encode_texts_peakpatch, assert_encode_text_consistency)
    from benchmarks.src.evaluation.eval_unary_mechanistic_analysis import fit_linear_probe
except ModuleNotFoundError:
    from analysis.cli import (
        add_model_args, add_run_args, add_data_args, add_restriction_args, add_concept_args)
    from analysis.feature_cache import load_object_restriction, build_provenance
    from analysis.config import set_seed, coerce_bool_column
    from analysis.model_loader import load_clip_for_eval
    from evaluation.eval_peakpatch_2x2_audit import (
        load_peakpatch, encode_texts_peakpatch, assert_encode_text_consistency)
    from evaluation.eval_unary_mechanistic_analysis import fit_linear_probe

CONDITIONS = ("cosine", "ecn")


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 68)
    print("  Polarity probe on PeakPatch's corrected text embedding")
    print("=" * 68)

    df = coerce_bool_column(pd.read_csv(args.csv_path), "object_in_image")
    restriction = load_object_restriction(args.restrict_objects)
    objects = sorted(df["object_name"].unique().tolist())
    if restriction:
        objects = [o for o in objects if o in set(restriction)]
    print(f"\n  Concepts after restriction: {len(objects)}")

    pp = load_peakpatch(args.peakpatch_root, device)
    if (args.model, args.pretrained) != (pp["arch"], pp["pretrained"]):
        raise SystemExit(f"backbone mismatch: {args.model}/{args.pretrained} vs "
                         f"{pp['arch']}/{pp['pretrained']}")
    model, _, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    assert_encode_text_consistency(model, tokenizer, device)

    rows = []
    for obj in objects:
        d = df[(df["object_name"] == obj) & (df["object_in_image"] == True)].reset_index(drop=True)
        n = len(d)
        if n < args.min_pairs:
            continue
        t_pos = d["positive_caption"].tolist()
        t_neg = d["negative_caption"].tolist()

        pos_plain, pos_ecn, _ = encode_texts_peakpatch(model, tokenizer, t_pos, pp, device, args.batch_size)
        neg_plain, neg_ecn, _ = encode_texts_peakpatch(model, tokenizer, t_neg, pp, device, args.batch_size)

        rec = {"object_name": obj, "n_pairs": n}
        for cond, P, N in (("cosine", pos_plain, neg_plain), ("ecn", pos_ecn, neg_ecn)):
            acc, std, w, b, strat = fit_linear_probe(P, N, seed=args.seed, paired=True)
            rec[f"{cond}_acc"] = acc
            rec[f"{cond}_std"] = std
            rec[f"{cond}_strat_acc"] = strat
        # how far the correction moved this concept's polarity direction
        d_plain = (pos_plain - neg_plain).mean(0)
        d_ecn = (pos_ecn - neg_ecn).mean(0)
        rec["cos_dT_shift"] = float(
            d_plain @ d_ecn / (np.linalg.norm(d_plain) * np.linalg.norm(d_ecn) + 1e-12))
        rows.append(rec)
        print(f"  {obj:22s} n={n:4d}  cosine {rec['cosine_acc']:6.2f}  "
              f"ecn {rec['ecn_acc']:6.2f}  d(cos)={rec['cos_dT_shift']:+.3f}")

    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(args.output_dir, "peakpatch_text_probe_per_concept.csv"), index=False)

    summary = {"n_concepts": len(per), "n_pairs": int(per["n_pairs"].sum()),
               "min_pairs": args.min_pairs, "seed": args.seed,
               "provenance": build_provenance(args)}
    for cond in CONDITIONS:
        a = per[f"{cond}_acc"].to_numpy()
        summary[cond] = {"macro_acc": float(a.mean()), "sd_across_concepts": float(a.std(ddof=1)),
                         "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
                         "n_below_60": int((a < 60).sum()),
                         "macro_strat_acc": float(per[f"{cond}_strat_acc"].mean())}
    delta = per["ecn_acc"] - per["cosine_acc"]
    summary["ecn_minus_cosine"] = {
        "macro_delta_pp": float(delta.mean()), "median_delta_pp": float(np.median(delta)),
        "n_concepts_up": int((delta > 0).sum()), "n_concepts_down": int((delta < 0).sum()),
        "mean_cos_dT_shift": float(per["cos_dT_shift"].mean())}

    with open(os.path.join(args.output_dir, "peakpatch_text_probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "-" * 68)
    for cond in CONDITIONS:
        s = summary[cond]
        print(f"  {cond:7s} macro {s['macro_acc']:6.2f}  (sd {s['sd_across_concepts']:5.2f}, "
              f"median {s['median']:6.2f}, range {s['min']:.1f}-{s['max']:.1f})")
    e = summary["ecn_minus_cosine"]
    print(f"  ecn - cosine: {e['macro_delta_pp']:+.2f}pp macro, "
          f"up {e['n_concepts_up']}/{len(per)}, mean cos(d_T, d_T^ecn) = {e['mean_cos_dT_shift']:+.3f}")
    print(f"\n  wrote {args.output_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p, "ViT-B-32", "openai")
    add_run_args(p, "logs/evaluation/peakpatch_text_probe", seed=42, batch_size=128)
    add_data_args(p, csv_path="benchmarks/data/images/beaf_counterfactual_6col.csv",
                  image_root="benchmarks/data/images")
    add_restriction_args(p, "Concept set to share with the 2x2 audit")
    add_concept_args(p, help_text="Minimum caption pairs per object")
    p.add_argument("--peakpatch_root", type=str, default="PeakPatch")
    run(p.parse_args())


if __name__ == "__main__":
    main()
