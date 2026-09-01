"""Does the delta loss cheat by growing ||W|| instead of improving structure?

review6.md (block B) raises a real mathematical fact: gamma and max(|alpha|,|beta|)
are both linear in W, so L_delta = ReLU(m - (gamma - max(|alpha|,|beta|))) is
1st-order homogeneous in W -- a block whose margin is already positive can drive
the hinge to zero simply by scaling W up, with no structural improvement. The
hypothesis: this shortcut dominates most at low rank (rank 1 collapses to 1.94%,
worse than the untrained w_I w_T^T outer product's 24.96%), because there is little
else the optimizer can do there.

This trains one fold (fold 0 only -- a diagnostic, not a full ladder) for both
losses and inspects the trained parameters' effective Frobenius norm afterward,
rather than discarding them as the real ladder run does.

Usage (from the repo root, reuses cached embeddings so this is fast):
    python -m benchmarks.src.evaluation.diagnose_delta_loss_norm_growth
"""
import argparse

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from benchmarks.src.evaluation.eval_single_w_generalization import (
    build_family, train_matcher, load_quads, get_embed_dim,
)
from benchmarks.src.analysis.model_loader import load_clip_for_eval
from benchmarks.src.analysis.config import set_seed

FAMILIES = ["diagonal", "lowrank_1", "lowrank_2", "lowrank_4", "lowrank_8",
            "lowrank_16", "lowrank_32", "full"]


def effective_w_fro_norm(net, family: str) -> float:
    if family.startswith("lowrank_"):
        W = net.proj_v.weight.T @ net.proj_t.weight  # (D, D) effective transform
        return float(torch.linalg.norm(W).item())
    if family == "diagonal":
        return float(torch.linalg.norm(net.w).item())
    if family == "full":
        return float(torch.linalg.norm(net.W).item())
    raise ValueError(family)


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
    tr, _ = splits[0]
    q_tr = tuple(x[tr] for x in quads_all)

    print(f"{'family':12s} {'loss':8s} {'||W||_F':>10s}  {'in-sample acc%':>15s}")
    for family in FAMILIES:
        for loss_kind in ["margin4", "delta"]:
            set_seed(args.seed)
            net, _ = build_family(family, embed_dim, args.seed)
            net = net.to(device)
            net = train_matcher(net, q_tr, epochs=args.epochs, lr=args.lr,
                                weight_decay=args.weight_decay, margin=args.margin,
                                loss_kind=loss_kind)
            norm = effective_w_fro_norm(net, family)
            with torch.no_grad():
                v_p, v_m, t_p, t_m = q_tr
                s_pp, s_mm = net(v_p, t_p), net(v_m, t_m)
                s_pm, s_mp = net(v_p, t_m), net(v_m, t_p)
                ok = (torch.minimum(s_pp, s_mm) > torch.maximum(s_pm, s_mp)).float().mean().item() * 100
            print(f"{family:12s} {loss_kind:8s} {norm:10.4f}  {ok:15.2f}")


if __name__ == "__main__":
    main()
