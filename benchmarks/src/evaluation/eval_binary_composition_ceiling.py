"""How far do two decoded bits get, where the cosine gets 0.88%?

The paper's central claim -- the information is represented, the similarity cannot use it --
rests on two numbers computed in different spaces: a probe AUC around 0.76-0.91 and a 2x2
matching accuracy below chance. Nothing forces them onto the same axis, so the gap between
them is rhetorical rather than measured.

This script closes that gap by scoring the same 2x2 blocks with three scorers of increasing
access, all evaluated with the paper's own success rule
min(S_++, S_--) > max(S_+-, S_-+):

    A  cosine            S = v . t                                   (what CLIP does)
    B  binary probes     S = f(sign(w_I . v), sign(w_T . t))         (two decoded bits only)
    C  oracle bits       S = f(a, b)                                 (ground-truth states)
    D  margin product    S = (w_I . v)(w_T . t)                      (rank-1 bilinear W = w_I w_T^T)

B is the quantity of interest. Its scorer sees four possible inputs, so any f is one of the
weak orderings of four cells; all 4^4 assignments are enumerated and the best is selected on
the training folds, never on the evaluation folds. Note that B does not require either probe
to be correct -- two probes that are wrong together still recover the pairing, which is why B
is not min(image accuracy, text accuracy).

D is included because (w_I . v)(w_T . t) = v^T (w_I w_T^T) t exactly: it is the rank-1 bilinear
head of section 5 with its factors fixed to the unimodal probe normals rather than learned.

The probes are per concept and cross-fitted five ways over that concept's pairs, so every
reported number is out-of-fold. This is a ceiling on what the decoded information supports,
not a deployable scorer: like the 67.43% projection in section 5, it is granted a per-concept
readout that a single gallery retriever does not have.

Pre-registered criteria (IMPLEMENTATION_PLAN.md Part III-2):
    U1  B beats the 16.67% chance rate significantly  -> information is usable once recomposed
    U1' B does not beat chance                        -> what the probes read is not what
                                                         matching needs; report it that way
    U2  D beats the cosine baseline substantially

Two controls answer questions a reader should ask of D (--controls):
    random_outer     w_I, w_T drawn at random -- is it the rank-1 *structure* or the directions?
    shuffled_probe   probes fit on label-shuffled data -- same procedure, no signal.
Both must land near the cosine baseline for the rank-1 reading to stand.

--crossconcept_null pairs each concept's image pairs with another concept's caption pairs. The
1/C(4,2) = 16.67% reference assumes the four scores are in uniformly random order, which the
actual similarity distribution need not satisfy; scoring blocks that contain no true binding
measures the reference empirically instead of assuming it.

Usage (from the repo root):
    python -m benchmarks.src.evaluation.eval_binary_composition_ceiling \
        --csv_path benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv \
        --output_dir logs/evaluation/01_paper/2026-08-31_binary_ceiling/vitb32_openai
"""

import os
import json
import itertools
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

try:
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.cli import (
        add_model_args, add_run_args, add_data_args, add_cache_args,
        add_restriction_args, add_concept_args, add_bias_args,
    )
    from benchmarks.src.analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction,
    )
    from benchmarks.src.analysis.config import set_seed, coerce_bool_column
    from benchmarks.src.analysis.paths import resolve_image_path as resolve_path
    from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, encode_texts_unified,
    )
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.model_loader import load_clip_for_eval
    from analysis.cli import (
        add_model_args, add_run_args, add_data_args, add_cache_args,
        add_restriction_args, add_concept_args, add_bias_args,
    )
    from analysis.feature_cache import (
        cached_encode, build_provenance, load_object_restriction,
    )
    from analysis.config import set_seed, coerce_bool_column
    from analysis.paths import resolve_image_path as resolve_path
    from evaluation.eval_e2_hadamard_decomposition import (
        encode_images_unified, encode_texts_unified,
    )

# The four cells of a block, as (image state, text state) with +1 = present / positive.
CELLS = [(+1, +1), (-1, +1), (+1, -1), (-1, -1)]
# Every assignment of the four (a_hat, b_hat) patterns to a score level: all weak orderings.
ALL_F = list(itertools.product(range(4), repeat=4))
PATTERN_INDEX = {p: i for i, p in enumerate(CELLS)}
CHANCE_2X2 = 1.0 / 6.0


def block_success(s_pp: np.ndarray, s_mp: np.ndarray,
                  s_pm: np.ndarray, s_mm: np.ndarray) -> np.ndarray:
    """The paper's rule: both state-matched cells strictly beat both mismatched cells."""
    return np.minimum(s_pp, s_mm) > np.maximum(s_pm, s_mp)


def score_with_f(f: Tuple[int, ...], a_hat: Dict[str, np.ndarray],
                 b_hat: Dict[str, np.ndarray]) -> np.ndarray:
    """Apply a table f to the four cells of every block and return per-block success."""
    cols = {}
    for img_key, img_state in (("p", +1), ("m", -1)):
        for txt_key, txt_state in (("p", +1), ("m", -1)):
            pat_a = a_hat[img_key]
            pat_b = b_hat[txt_key]
            idx = np.array([PATTERN_INDEX[(int(x), int(y))] for x, y in zip(pat_a, pat_b)])
            cols[img_key + txt_key] = np.array([f[i] for i in idx], dtype=float)
    return block_success(cols["pp"], cols["mp"], cols["pm"], cols["mm"])


def best_f_on(a_hat: Dict[str, np.ndarray], b_hat: Dict[str, np.ndarray]) -> Tuple[int, ...]:
    best, best_acc = ALL_F[0], -1.0
    for f in ALL_F:
        acc = score_with_f(f, a_hat, b_hat).mean()
        if acc > best_acc:
            best, best_acc = f, acc
    return best


def _random_unit(dim: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.normal(size=dim)
    return w / (np.linalg.norm(w) + 1e-12)


def control_rungs(v_pres, v_abs, t_pos, t_neg, seed: int, use_bias: bool) -> Dict[str, np.ndarray]:
    """Random-direction and shuffled-label counterparts of the D rung, same 5-fold protocol."""
    n = len(v_pres)
    out = {k: np.zeros(n, dtype=bool) for k in ("random_outer", "shuffled_probe")}
    rng = np.random.default_rng(seed)
    for tr, te in KFold(n_splits=min(5, n), shuffle=True, random_state=seed).split(np.arange(n)):
        w_i, w_t = _random_unit(v_pres.shape[1], rng), _random_unit(t_pos.shape[1], rng)
        a_p, a_m = v_pres[te] @ w_i, v_abs[te] @ w_i
        b_p, b_m = t_pos[te] @ w_t, t_neg[te] @ w_t
        out["random_outer"][te] = block_success(a_p * b_p, a_m * b_p, a_p * b_m, a_m * b_m)

        # Same fitting procedure, labels permuted so no polarity signal can be learned.
        y = rng.permutation(np.array([1] * len(tr) + [0] * len(tr)))
        cv = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, fit_intercept=use_bias)
        cv.fit(np.vstack([v_pres[tr], v_abs[tr]]), y)
        ct = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, fit_intercept=use_bias)
        ct.fit(np.vstack([t_pos[tr], t_neg[tr]]), rng.permutation(y))
        wi, wt = cv.coef_[0], ct.coef_[0]
        a_p, a_m = v_pres[te] @ wi, v_abs[te] @ wi
        b_p, b_m = t_pos[te] @ wt, t_neg[te] @ wt
        out["shuffled_probe"][te] = block_success(a_p * b_p, a_m * b_p, a_p * b_m, a_m * b_m)
    return out


def evaluate_concept(v_pres, v_abs, t_pos, t_neg, seed: int, use_bias: bool) -> Dict[str, np.ndarray]:
    """Out-of-fold per-pair success flags for each rung of the ladder."""
    n = len(v_pres)
    out = {k: np.zeros(n, dtype=bool)
           for k in ("cosine", "binary", "oracle", "margin", "margin_centred")}
    acc_img = np.zeros(n, dtype=bool)
    acc_txt = np.zeros(n, dtype=bool)

    # A: cosine needs no fitting.
    out["cosine"] = block_success(np.sum(v_pres * t_pos, axis=1), np.sum(v_abs * t_pos, axis=1),
                                  np.sum(v_pres * t_neg, axis=1), np.sum(v_abs * t_neg, axis=1))

    # C: oracle bits are the true states, so f = a*b separates by construction.
    out["oracle"] = block_success(np.ones(n), -np.ones(n), -np.ones(n), np.ones(n))

    n_splits = min(5, n)
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.arange(n)):
        clf_v = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, fit_intercept=use_bias)
        clf_v.fit(np.vstack([v_pres[tr], v_abs[tr]]), np.array([1] * len(tr) + [0] * len(tr)))
        clf_t = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, fit_intercept=use_bias)
        clf_t.fit(np.vstack([t_pos[tr], t_neg[tr]]), np.array([1] * len(tr) + [0] * len(tr)))
        w_I, w_T = clf_v.coef_[0], clf_t.coef_[0]

        # The decision boundary is w.x + b = 0, not w.x = 0. L2-normalized CLIP embeddings sit
        # far from the origin, so dropping the intercept moves the boundary off the data entirely.
        def bits(clf, X):
            return np.where(clf.decision_function(X) >= 0, 1, -1)

        a_tr = {"p": bits(clf_v, v_pres[tr]), "m": bits(clf_v, v_abs[tr])}
        b_tr = {"p": bits(clf_t, t_pos[tr]), "m": bits(clf_t, t_neg[tr])}
        f = best_f_on(a_tr, b_tr)          # table chosen on training folds only

        a_te = {"p": bits(clf_v, v_pres[te]), "m": bits(clf_v, v_abs[te])}
        b_te = {"p": bits(clf_t, t_pos[te]), "m": bits(clf_t, t_neg[te])}
        out["binary"][te] = score_with_f(f, a_te, b_te)

        # D1 is exactly v^T (w_I w_T^T) t -- the rank-1 bilinear head with fixed factors.
        # D2 uses the centred detector outputs, which is what the bits threshold.
        m_vp, m_va = v_pres[te] @ w_I, v_abs[te] @ w_I
        m_tp, m_tn = t_pos[te] @ w_T, t_neg[te] @ w_T
        out["margin"][te] = block_success(m_vp * m_tp, m_va * m_tp, m_vp * m_tn, m_va * m_tn)

        c_vp, c_va = clf_v.decision_function(v_pres[te]), clf_v.decision_function(v_abs[te])
        c_tp, c_tn = clf_t.decision_function(t_pos[te]), clf_t.decision_function(t_neg[te])
        out["margin_centred"][te] = block_success(c_vp * c_tp, c_va * c_tp, c_vp * c_tn, c_va * c_tn)

        acc_img[te] = (a_te["p"] == 1) & (a_te["m"] == -1)
        acc_txt[te] = (b_te["p"] == 1) & (b_te["m"] == -1)

    out["probe_img_pairwise"] = acc_img
    out["probe_txt_pairwise"] = acc_txt
    return out


def render(df: pd.DataFrame, pooled: Dict[str, float], out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    rungs = ["cosine", "binary", "margin", "margin_centred", "oracle"]
    labels = ["A cosine\n$v\\cdot t$", "B binary probes\n$f(\\hat a,\\hat b)$",
              "D rank-1 bilinear\n$(w_I\\!\\cdot\\!v)(w_T\\!\\cdot\\!t)$",
              "D' centred\nmargins", "C oracle bits\n$f(a,b)$"]
    vals = [100 * pooled[r] for r in rungs]
    ax = axes[0]
    bars = ax.bar(range(len(rungs)), vals, color=["#8d99ae", "#2a9d8f", "#e9c46a", "#f4a261", "#264653"])
    ax.axhline(100 * CHANCE_2X2, color="crimson", ls="--", lw=1.4, label="chance 16.67%")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.2f}%", ha="center", fontsize=9)
    ax.set_xticks(range(len(rungs))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("2x2 accuracy (%)"); ax.set_ylim(0, 108)
    ax.set_title("What two decoded bits reach")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(100 * df["cosine"], 100 * df["binary"], s=36, alpha=0.8)
    lim = max(5.0, float(100 * df["binary"].max()) * 1.1)
    ax.plot([0, lim], [0, lim], color="grey", ls=":", lw=1.2)
    ax.axhline(100 * CHANCE_2X2, color="crimson", ls="--", lw=1.2)
    ax.set_xlabel("cosine 2x2 accuracy (%)"); ax.set_ylabel("binary-probe 2x2 accuracy (%)")
    ax.set_title("Per concept")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_binary_ceiling.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Binary-composition ceiling for 2x2 matching")
    add_model_args(parser, "ViT-B-32", "openai")
    add_run_args(parser, "logs/evaluation/binary_ceiling", seed=42, batch_size=128)
    add_data_args(parser,
                  csv_path="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv",
                  image_root="benchmarks/data/images")
    add_cache_args(parser)
    add_restriction_args(parser, "Comma list, or path to txt/csv/json, limiting evaluation to an exact concept set")
    add_concept_args(parser)
    add_bias_args(parser)
    parser.add_argument("--controls", action="store_true", default=False,
                        help="Add the random-direction and shuffled-label counterparts of rung D")
    parser.add_argument("--crossconcept_null", action="store_true", default=False,
                        help="Also score blocks whose captions come from a different concept, "
                             "measuring the chance reference empirically")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bias = not args.no_bias
    cache_kw = dict(model=args.model, pretrained=args.pretrained,
                    cache_dir=args.cache_dir, enabled=args.use_cache)

    print("=" * 72)
    print("  Binary-composition ceiling: cosine -> two bits -> oracle")
    print(f"  Model {args.model} ({args.pretrained}) | device {device} | min_pairs {args.min_pairs}")
    print(f"  CSV   {args.csv_path}")
    print("=" * 72)

    df = pd.read_csv(args.csv_path)
    coerce_bool_column(df, "object_in_image")
    concepts = [o for o in sorted(df["object_name"].unique()) if "," not in str(o)]
    restrict = load_object_restriction(args.restrict_objects)
    if restrict is not None:
        concepts = [o for o in concepts if o in set(restrict)]

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)

    keys = ["cosine", "binary", "oracle", "margin", "margin_centred",
            "probe_img_pairwise", "probe_txt_pairwise"]
    if args.controls:
        keys += ["random_outer", "shuffled_probe"]
    rows, pooled_flags, store = [], {k: [] for k in keys}, {}
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

        res = evaluate_concept(v_pres, v_abs, t_pos, t_neg, args.seed, use_bias)
        if args.controls:
            res.update(control_rungs(v_pres, v_abs, t_pos, t_neg, args.seed, use_bias))
        if args.crossconcept_null:
            store[obj] = (v_pres, v_abs, t_pos, t_neg)
        row = {"object_name": obj, "n_pairs": len(keep)}
        for k, v in res.items():
            row[k] = float(v.mean())
            pooled_flags[k].append(v)
        rows.append(row)
        print(f"  [{obj:22s}] N={row['n_pairs']:4d} | cosine={100*row['cosine']:6.2f}% "
              f"binary={100*row['binary']:6.2f}% margin={100*row['margin']:6.2f}% "
              f"| probe img={100*row['probe_img_pairwise']:5.1f}% txt={100*row['probe_txt_pairwise']:5.1f}%")

    if not rows:
        raise SystemExit("No concept met --min_pairs; nothing to report.")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.output_dir, "per_concept_ceiling.csv"), index=False)

    pooled = {k: float(np.concatenate(v).mean()) for k, v in pooled_flags.items() if v}

    # Empirical chance reference: same images, captions borrowed from the next concept.
    if args.crossconcept_null:
        names = list(store)
        flags = []
        for i, nm in enumerate(names):
            v_p, v_a, _, _ = store[nm]
            _, _, t_p, t_n = store[names[(i + 1) % len(names)]]
            m = min(len(v_p), len(t_p))
            flags.append(block_success(np.sum(v_p[:m] * t_p[:m], axis=1),
                                       np.sum(v_a[:m] * t_p[:m], axis=1),
                                       np.sum(v_p[:m] * t_n[:m], axis=1),
                                       np.sum(v_a[:m] * t_n[:m], axis=1)))
        pooled["crossconcept_null"] = float(np.concatenate(flags).mean())
    n_pooled = int(len(np.concatenate(pooled_flags["binary"])))
    k_binary = int(np.concatenate(pooled_flags["binary"]).sum())
    bt = binomtest(k_binary, n_pooled, CHANCE_2X2, alternative="greater")

    summary = {
        "n_concepts": int(len(out)), "total_pairs": n_pooled,
        "chance_2x2": CHANCE_2X2,
        "pooled": dict(pooled),
        "macro": {k: float(out[k].mean()) for k in keys if k in out},
        "binomial_test_binary_vs_chance": {"successes": k_binary, "n": n_pooled,
                                           "p_one_sided": float(bt.pvalue)},
        "criteria": {
            "U1_binary_beats_chance": bool(bt.pvalue < 0.05 and pooled["binary"] > CHANCE_2X2),
            "U2_margin_beats_cosine": bool(max(pooled["margin"], pooled["margin_centred"])
                                           > 2 * max(pooled["cosine"], 1e-9)),
        },
        "provenance": build_provenance(args, n_concepts=int(len(out)), n_pairs=n_pooled),
    }
    with open(os.path.join(args.output_dir, "binary_ceiling_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    render(out, pooled, args.output_dir)

    print("\n" + "=" * 72)
    print(f"  concepts {summary['n_concepts']} | pairs {n_pooled} | chance {100*CHANCE_2X2:.2f}%")
    print(f"  A cosine          : pooled {100*pooled['cosine']:6.2f}%  macro {100*out['cosine'].mean():6.2f}%")
    print(f"  B binary probes   : pooled {100*pooled['binary']:6.2f}%  macro {100*out['binary'].mean():6.2f}%")
    print(f"  D margin product  : pooled {100*pooled['margin']:6.2f}%  macro {100*out['margin'].mean():6.2f}%")
    print(f"  D' centred margins: pooled {100*pooled['margin_centred']:6.2f}%  macro {100*out['margin_centred'].mean():6.2f}%")
    print(f"  C oracle bits     : pooled {100*pooled['oracle']:6.2f}%")
    print(f"  probe pairwise    : image {100*pooled['probe_img_pairwise']:5.1f}%  "
          f"text {100*pooled['probe_txt_pairwise']:5.1f}%")
    if args.controls:
        print(f"  [ctrl] random 외적 : pooled {100*pooled['random_outer']:6.2f}%  "
              f"macro {100*out['random_outer'].mean():6.2f}%")
        print(f"  [ctrl] 라벨 셔플   : pooled {100*pooled['shuffled_probe']:6.2f}%  "
              f"macro {100*out['shuffled_probe'].mean():6.2f}%")
    if args.crossconcept_null:
        print(f"  [null] 교차 개념   : pooled {100*pooled['crossconcept_null']:6.2f}%  "
              f"(조합론적 기준 {100*CHANCE_2X2:.2f}%)")
    print(f"  B vs chance       : one-sided binomial p = {bt.pvalue:.3g}")
    for k, v in summary["criteria"].items():
        print(f"  {k:26s} : {'PASS' if v else 'FAIL'}")
    print("=" * 72)
    print(f"  Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
