"""Does warm-starting rank 1 from the probe-normal outer product fix its collapse?

diagnose_delta_loss_norm_growth.py ruled out review6's "scale exploitation"
hypothesis for the rank-1 delta-loss collapse (1.94%, worse than the untrained
w_I w_T^T outer product's 24.96%): ||W|| stays near zero, it doesn't blow up.
That leaves "the optimizer never finds its way out of the near-zero
initialization" as the live explanation. If so, starting from an already-good
point rather than near-zero should route around the problem entirely --
review6's proposed fix.

This fits one GLOBAL (concept-agnostic) rank-1 direction per modality via
logistic regression on the training fold's pooled data (matching how the rest
of the ladder is concept-agnostic), uses it to initialise LowRankMatcher(rank=1)
instead of the small random draw, and trains with both losses for comparison.
Only fold 0 -- a diagnostic, not the full GroupKFold ladder.

Usage (from the repo root, reuses cached embeddings):
    python -m benchmarks.src.evaluation.diagnose_delta_loss_warmstart
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from benchmarks.src.evaluation.eval_single_w_generalization import (
    LowRankMatcher, train_matcher, load_quads, get_embed_dim, joint_correct,
)
from benchmarks.src.analysis.model_loader import load_clip_for_eval
from benchmarks.src.analysis.config import set_seed


def fit_global_direction(pos: np.ndarray, neg: np.ndarray, seed: int) -> np.ndarray:
    """One global (concept-agnostic) logistic-regression normal, no intercept --
    matches the D-rung's `v^T(w_I w_T^T)t` convention used elsewhere this session."""
    X = np.vstack([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg))
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed, fit_intercept=False)
    clf.fit(X, y)
    w = clf.coef_[0]
    return w / (np.linalg.norm(w) + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--min_pairs", type=int, default=20)
    ap.add_argument("--csv_path", default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--image_root", default="benchmarks/data/images")
    ap.add_argument("--restrict_objects", default=None)
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--cache_dir", default="logs/evaluation/cached_embeddings/feature_cache")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=0.1)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_kw = dict(model=args.model, pretrained=str(args.pretrained),
                    cache_dir=args.cache_dir, enabled=args.use_cache)

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    embed_dim = get_embed_dim(model)

    v_pos, v_neg, t_pos, t_neg, groups = load_quads(args, model, preprocess, tokenizer, device, cache_kw)
    T = lambda a: torch.from_numpy(a).float().to(device)
    quads_all = (T(v_pos), T(v_neg), T(t_pos), T(t_neg))

    splits = list(GroupKFold(n_splits=5).split(np.zeros(len(groups)), groups=groups))

    print(f"{'fold':6s}{'init':10s} {'loss':8s} {'in-sample%':>11s} {'OOF%':>8s}")
    all_oof = {(ik, lk): np.zeros(len(groups), dtype=bool)
               for ik in ["random", "warmstart"] for lk in ["margin4", "delta"]}
    for fold, (tr, te) in enumerate(splits):
        q_tr = tuple(x[tr] for x in quads_all)
        q_te = tuple(x[te] for x in quads_all)
        w_I = fit_global_direction(v_pos[tr], v_neg[tr], args.seed)
        w_T = fit_global_direction(t_pos[tr], t_neg[tr], args.seed)

        for init_kind in ["random", "warmstart"]:
            for loss_kind in ["margin4", "delta"]:
                set_seed(args.seed + fold)
                net = LowRankMatcher(embed_dim, rank=1).to(device)
                if init_kind == "warmstart":
                    with torch.no_grad():
                        net.proj_v.weight.copy_(torch.from_numpy(w_I).float().to(device).unsqueeze(0))
                        net.proj_t.weight.copy_(torch.from_numpy(w_T).float().to(device).unsqueeze(0))
                net = train_matcher(net, q_tr, epochs=args.epochs, lr=args.lr,
                                    weight_decay=args.weight_decay, margin=args.margin,
                                    loss_kind=loss_kind)
                in_sample = 100.0 * joint_correct(net, q_tr).mean()
                ok_te = joint_correct(net, q_te)
                all_oof[(init_kind, loss_kind)][te] = ok_te
                oof = 100.0 * ok_te.mean()
                print(f"{fold:<6d}{init_kind:10s} {loss_kind:8s} {in_sample:11.2f} {oof:8.2f}")

    print("\npooled OOF across all 5 folds (rank 1):")
    for (init_kind, loss_kind), flags in all_oof.items():
        print(f"  {init_kind:10s} {loss_kind:8s} {100.0 * flags.mean():6.2f}%")


if __name__ == "__main__":
    main()
