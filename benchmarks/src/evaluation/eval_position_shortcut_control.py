"""Does the text probe read which object is negated, or just where it sits in the sentence?

Section 2 argues the AB-swap pair blocks the negation-marker shortcut, because the two captions
share 94.8% of their token multiset. That argument is about *tokens*. It leaves the *order*
untouched: in

    positive: There is a truck in this image, but no train.
    negative: There is a train in this image, but no truck.

the asserted object is simply the first one mentioned, so a probe that learned nothing but
"the first noun is the asserted one" would score as well as one that understood the binding.

`beaf_counterfactual_ab_swap_diverse.csv` already contains the control, because half of its
captions put the negated clause first:

    pos_first  There is a truck in this image, but no train.
    neg_first  There is no cup in this image, but there is a fork.

A positional probe must invert across that boundary. Training on one position and scoring on
the other therefore separates the two hypotheses, and the four template families
(standard / lacking / absent / free_of) give the same test for syntactic frame.

Captions are de-duplicated per concept first: the same sentence pair recurs across many image
pairs, and leaving the duplicates in would put identical strings on both sides of the split.

Pre-registered criteria (IMPLEMENTATION_PLAN.md Part III-3):
    S1  cross-position transfer >= 90% of within-position accuracy
    S2  cross-family transfer   >= 90% of within-family accuracy
Failing either means the reported text-probe numbers owe part of their height to surface
structure, and should be reported split by that factor.

Usage (from the repo root):
    python -m benchmarks.src.evaluation.eval_position_shortcut_control \
        --output_dir logs/evaluation/01_paper/2026-08-31_position_shortcut/vitb32_openai
"""

import os
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

try:
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.cli import (
        add_model_args, add_run_args, add_cache_args, add_restriction_args, add_bias_args,
    )
    from benchmarks.src.analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction,
    )
    from benchmarks.src.analysis.config import set_seed
    from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import encode_texts_unified
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.model_loader import load_clip_for_eval
    from analysis.cli import (
        add_model_args, add_run_args, add_cache_args, add_restriction_args, add_bias_args,
    )
    from analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction,
    )
    from analysis.config import set_seed
    from evaluation.eval_e2_hadamard_decomposition import encode_texts_unified

# The text-side unit here is the unique caption pair, not the image counterfactual pair that
# --min_pairs governs, so it carries its own floor. 8 keeps 42 concepts, matching the concept
# count of the AB-swap coefficient runs; the stricter 20 leaves 21 and is reported alongside.
DEFAULT_MIN_CAPTION_PAIRS = 8
POSITIONS = ("pos_first", "neg_first")


def _fit(tp: np.ndarray, tn: np.ndarray, seed: int, use_bias: bool) -> LogisticRegression:
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, fit_intercept=use_bias)
    clf.fit(np.vstack([tp, tn]), np.array([1] * len(tp) + [0] * len(tn)))
    return clf


def _score(clf: LogisticRegression, tp: np.ndarray, tn: np.ndarray) -> Dict[str, float]:
    """Pairwise accuracy (both sides right) and AUC over the pooled positives and negatives."""
    s_p, s_n = clf.decision_function(tp), clf.decision_function(tn)
    pairwise = float(np.mean((s_p >= 0) & (s_n < 0)))
    y = np.array([1] * len(s_p) + [0] * len(s_n))
    auc = float(roc_auc_score(y, np.concatenate([s_p, s_n]))) if len(set(y)) > 1 else float("nan")
    return {"pairwise": pairwise, "auc": auc}


def _within(tp: np.ndarray, tn: np.ndarray, seed: int, use_bias: bool) -> Dict[str, float]:
    """Held-out score inside one stratum, so it is comparable to the transfer score."""
    n = len(tp)
    if n < 4:
        return {"pairwise": float("nan"), "auc": float("nan")}
    acc, aucs = [], []
    for tr, te in KFold(n_splits=min(5, n), shuffle=True, random_state=seed).split(np.arange(n)):
        clf = _fit(tp[tr], tn[tr], seed, use_bias)
        s = _score(clf, tp[te], tn[te])
        acc.append(s["pairwise"])
        if not np.isnan(s["auc"]):
            aucs.append(s["auc"])
    return {"pairwise": float(np.mean(acc)),
            "auc": float(np.mean(aucs)) if aucs else float("nan")}


def analyse_concept(sub: pd.DataFrame, emb_pos: np.ndarray, emb_neg: np.ndarray,
                    seed: int, use_bias: bool) -> Dict[str, float]:
    rec: Dict[str, float] = {}
    pos_mask = (sub["pos_position"] == "pos_first").values

    tp_a, tn_a = emb_pos[pos_mask], emb_neg[pos_mask]
    tp_b, tn_b = emb_pos[~pos_mask], emb_neg[~pos_mask]

    w_a, w_b = _within(tp_a, tn_a, seed, use_bias), _within(tp_b, tn_b, seed, use_bias)
    rec["within_pos_first"] = w_a["pairwise"]
    rec["within_neg_first"] = w_b["pairwise"]
    rec["within_position"] = float(np.nanmean([w_a["pairwise"], w_b["pairwise"]]))
    rec["within_position_auc"] = float(np.nanmean([w_a["auc"], w_b["auc"]]))

    x_ab = _score(_fit(tp_a, tn_a, seed, use_bias), tp_b, tn_b)   # train pos_first -> test neg_first
    x_ba = _score(_fit(tp_b, tn_b, seed, use_bias), tp_a, tn_a)   # and the reverse
    rec["cross_pos_to_neg"] = x_ab["pairwise"]
    rec["cross_neg_to_pos"] = x_ba["pairwise"]
    rec["cross_position"] = float(np.nanmean([x_ab["pairwise"], x_ba["pairwise"]]))
    rec["cross_position_auc"] = float(np.nanmean([x_ab["auc"], x_ba["auc"]]))

    # Leave-one-template-family-out, when the concept carries more than one family.
    fams = sorted(sub["template_family"].unique())
    if len(fams) > 1:
        w_in, w_out, a_out = [], [], []
        for fam in fams:
            m = (sub["template_family"] == fam).values
            if m.sum() < 2 or (~m).sum() < 2:
                continue
            clf = _fit(emb_pos[~m], emb_neg[~m], seed, use_bias)
            s = _score(clf, emb_pos[m], emb_neg[m])
            w_out.append(s["pairwise"])
            if not np.isnan(s["auc"]):
                a_out.append(s["auc"])
            w_in.append(_within(emb_pos[~m], emb_neg[~m], seed, use_bias)["pairwise"])
        if w_out:
            rec["within_family"] = float(np.mean(w_in))
            rec["cross_family"] = float(np.mean(w_out))
            rec["cross_family_auc"] = float(np.mean(a_out)) if a_out else float("nan")
    return rec


def render(df: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    pairs = [("within_position", "cross_position"), ("within_family", "cross_family")]
    labels, w_vals, c_vals = [], [], []
    for w, c in pairs:
        if w in df and c in df:
            labels.append(w.split("_")[1])
            w_vals.append(100 * df[w].mean())
            c_vals.append(100 * df[c].mean())
    x = np.arange(len(labels))
    ax.bar(x - 0.18, w_vals, 0.36, label="within stratum", color="#2a9d8f")
    ax.bar(x + 0.18, c_vals, 0.36, label="transferred across", color="#e76f51")
    for i, (w, c) in enumerate(zip(w_vals, c_vals)):
        ax.text(i - 0.18, w + 1, f"{w:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.18, c + 1, f"{c:.1f}", ha="center", fontsize=8)
        ax.plot([i - 0.18, i + 0.18], [0.9 * w, 0.9 * w], color="crimson", ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("pairwise accuracy (%)"); ax.set_ylim(0, 108)
    ax.set_title("Text probe transferred across surface structure\n(dashed = 90% of within, the S1/S2 floor)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(100 * df["within_position"], 100 * df["cross_position"], s=36, alpha=0.8)
    ax.plot([0, 100], [0, 100], color="grey", ls=":", lw=1.2)
    ax.plot([0, 100], [0, 90], color="crimson", ls="--", lw=1.2, label="90% of within")
    ax.set_xlabel("within-position (%)"); ax.set_ylabel("cross-position (%)")
    ax.set_title("Per concept"); ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_position_shortcut.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Positional / syntactic shortcut control for the text probe")
    add_model_args(parser, "ViT-B-32", "openai")
    add_run_args(parser, "logs/evaluation/position_shortcut", seed=42, batch_size=128)
    parser.add_argument("--csv_path", type=str,
                        default="benchmarks/data/images/beaf_counterfactual_ab_swap_diverse.csv")
    parser.add_argument("--min_caption_pairs", type=int, default=DEFAULT_MIN_CAPTION_PAIRS,
                        help=f"Unique caption pairs a concept needs in EACH position "
                             f"(default: {DEFAULT_MIN_CAPTION_PAIRS}; this is the text-side unit, "
                             f"distinct from the image-side --min_pairs)")
    add_cache_args(parser)
    add_restriction_args(parser, "Comma list, or path to txt/csv/json, limiting evaluation to an exact concept set")
    add_bias_args(parser)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bias = not args.no_bias
    cache_kw = dict(model=args.model, pretrained=args.pretrained,
                    cache_dir=args.cache_dir, enabled=args.use_cache)

    print("=" * 72)
    print("  Positional / syntactic shortcut control for the AB-swap text probe")
    print(f"  Model {args.model} ({args.pretrained}) | device {device}")
    print(f"  min unique caption pairs per position: {args.min_caption_pairs}")
    print("=" * 72)

    df = pd.read_csv(args.csv_path)
    df = df.drop_duplicates(["object_a", "positive_caption", "negative_caption"]).reset_index(drop=True)
    restrict = load_object_restriction(args.restrict_objects)
    if restrict is not None:
        df = df[df["object_a"].isin(set(restrict))].reset_index(drop=True)

    model, _, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)

    rows = []
    for obj, sub in df.groupby("object_a"):
        sub = sub.reset_index(drop=True)
        n_pf = int((sub["pos_position"] == "pos_first").sum())
        n_nf = int((sub["pos_position"] == "neg_first").sum())
        if min(n_pf, n_nf) < args.min_caption_pairs:
            continue

        txt_pos = sub["positive_caption"].tolist()
        txt_neg = sub["negative_caption"].tolist()
        e_pos, _ = cached_encode(
            lambda: encode_texts_unified(model, tokenizer, txt_pos, device, args.batch_size),
            kind="text_pos@norm+raw", items=txt_pos, **cache_kw)
        e_neg, _ = cached_encode(
            lambda: encode_texts_unified(model, tokenizer, txt_neg, device, args.batch_size),
            kind="text_neg@norm+raw", items=txt_neg, **cache_kw)

        rec = analyse_concept(sub, e_pos, e_neg, args.seed, use_bias)
        rec.update({"object_name": obj, "n_caption_pairs": len(sub),
                    "n_pos_first": n_pf, "n_neg_first": n_nf})
        rows.append(rec)
        print(f"  [{obj:22s}] pairs={len(sub):4d} | within-pos={100*rec['within_position']:5.1f}% "
              f"cross-pos={100*rec['cross_position']:5.1f}% "
              f"(p->n {100*rec['cross_pos_to_neg']:5.1f}%, n->p {100*rec['cross_neg_to_pos']:5.1f}%) "
              f"| cross-fam={100*rec.get('cross_family', float('nan')):5.1f}%")

    if not rows:
        raise SystemExit("No concept met --min_caption_pairs; nothing to report.")

    out = pd.DataFrame(rows)
    front = ["object_name", "n_caption_pairs", "n_pos_first", "n_neg_first"]
    out = out[front + [c for c in out.columns if c not in front]]
    out.to_csv(os.path.join(args.output_dir, "per_concept_shortcut.csv"), index=False)

    def m(col):
        return float(out[col].mean()) if col in out else float("nan")

    ratio_pos = m("cross_position") / max(m("within_position"), 1e-9)
    ratio_fam = m("cross_family") / max(m("within_family"), 1e-9)
    strict = out[(out["n_pos_first"] >= 20) & (out["n_neg_first"] >= 20)]

    summary = {
        "n_concepts": int(len(out)),
        "n_caption_pairs": int(out["n_caption_pairs"].sum()),
        "macro": {c: m(c) for c in
                  ("within_position", "cross_position", "within_position_auc", "cross_position_auc",
                   "cross_pos_to_neg", "cross_neg_to_pos", "within_family", "cross_family",
                   "cross_family_auc")},
        "transfer_ratio_position": ratio_pos,
        "transfer_ratio_family": ratio_fam,
        "strict_subset_min20": {
            "n_concepts": int(len(strict)),
            "within_position": float(strict["within_position"].mean()) if len(strict) else float("nan"),
            "cross_position": float(strict["cross_position"].mean()) if len(strict) else float("nan"),
        },
        "criteria": {"S1_position_transfer_ge_90pct": bool(ratio_pos >= 0.90),
                     "S2_family_transfer_ge_90pct": bool(ratio_fam >= 0.90)},
        "provenance": build_provenance(args, n_concepts=int(len(out)),
                                       n_caption_pairs=int(out["n_caption_pairs"].sum())),
    }
    with open(os.path.join(args.output_dir, "position_shortcut_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    render(out, args.output_dir)

    print("\n" + "=" * 72)
    print(f"  concepts {summary['n_concepts']} | unique caption pairs {summary['n_caption_pairs']}")
    print(f"  within position : {100*m('within_position'):6.2f}%  (AUC {m('within_position_auc'):.3f})")
    print(f"  cross  position : {100*m('cross_position'):6.2f}%  (AUC {m('cross_position_auc'):.3f})"
          f"   ratio {ratio_pos:.3f}")
    print(f"      pos->neg {100*m('cross_pos_to_neg'):6.2f}%   neg->pos {100*m('cross_neg_to_pos'):6.2f}%")
    print(f"  within family   : {100*m('within_family'):6.2f}%")
    print(f"  cross  family   : {100*m('cross_family'):6.2f}%   ratio {ratio_fam:.3f}")
    s = summary["strict_subset_min20"]
    print(f"  strict (>=20 both positions, n={s['n_concepts']}): "
          f"within {100*s['within_position']:.2f}%  cross {100*s['cross_position']:.2f}%")
    for k, v in summary["criteria"].items():
        print(f"  {k:32s} : {'PASS' if v else 'FAIL'}")
    print("=" * 72)
    print(f"  Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
