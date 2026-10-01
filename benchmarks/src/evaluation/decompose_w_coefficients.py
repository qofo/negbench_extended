"""
Hadamard-decompose the 2x2 similarity block under an arbitrary bilinear W.

`score_delta_s_for_w.py` answers "what 2x2 joint accuracy does this W reach";
this script answers *why* -- it reports the three factorial coefficients behind
that accuracy, so a W that moves the diagnostic can be attributed to raising the
interaction term gamma rather than to merely shrinking the main effects.

    S_ab = C + a*beta + b*alpha + a*b*gamma      (a = image state, b = text state)
    Delta(S) = min(S_11, S_22) - max(S_12, S_21) = 2*gamma - 2*max(|alpha|, |beta|)

Same population and scope note as `score_delta_s_for_w.py`: the 42-concept
AB-swap set every W in the transfer table was fit on, scored in-sample (no
held-out fold), which is what makes the rows mutually comparable. Aggregation
follows `eval_e2_hadamard_decomposition.py`: per-pair coefficients (|alpha| and
|beta| as per-pair absolute values), averaged within a concept, then macro-
averaged over concepts.

Usage:
    python -m benchmarks.src.evaluation.decompose_w_coefficients \
        --ckpts logs/evaluation/01_paper/2026-09-02_residual_identity_infonce/residual_identity_infonce_pen0.1.pt \
        --names proposed_c0.1
"""
import os
import json
import argparse

import numpy as np
import torch

from benchmarks.src.evaluation.eval_single_w_generalization import load_quads
from benchmarks.src.analysis.model_loader import load_clip_for_eval
from benchmarks.src.analysis.config import set_seed


def decompose(W: torch.Tensor, quads, groups) -> dict:
    """Per-pair decomposition under W, aggregated per concept then macro-averaged."""
    v_pos, v_neg, t_pos, t_neg = quads
    # S_ij with i = text state, j = image state (eval_e2_hadamard_decomposition.py).
    S11 = torch.einsum("nd,de,ne->n", v_pos, W, t_pos)   # text +, image +
    S12 = torch.einsum("nd,de,ne->n", v_neg, W, t_pos)   # text +, image -
    S21 = torch.einsum("nd,de,ne->n", v_pos, W, t_neg)   # text -, image +
    S22 = torch.einsum("nd,de,ne->n", v_neg, W, t_neg)   # text -, image -
    S11, S12, S21, S22 = (x.detach().cpu().numpy().astype(np.float64)
                          for x in (S11, S12, S21, S22))

    beta = (S11 - S12 + S21 - S22) / 4.0     # image main effect
    alpha = (S11 + S12 - S21 - S22) / 4.0    # text main effect
    gamma = (S11 - S12 - S21 + S22) / 4.0    # interaction

    delta_emp = np.minimum(S11, S22) - np.maximum(S12, S21)
    delta_ana = 2.0 * gamma - 2.0 * np.maximum(np.abs(alpha), np.abs(beta))
    max_err = float(np.max(np.abs(delta_emp - delta_ana)))
    assert max_err < 1e-6, f"identity violated (max error {max_err:.3e})"

    groups = np.asarray(groups)
    per_concept = {"abs_alpha": [], "abs_beta": [], "gamma": [], "acc": [], "delta": []}
    for g in np.unique(groups):
        m = groups == g
        per_concept["abs_alpha"].append(np.abs(alpha[m]).mean())
        per_concept["abs_beta"].append(np.abs(beta[m]).mean())
        per_concept["gamma"].append(gamma[m].mean())
        per_concept["acc"].append((delta_emp[m] > 0).mean() * 100.0)
        per_concept["delta"].append(delta_emp[m].mean())

    macro = {k: float(np.mean(v)) for k, v in per_concept.items()}
    concept_names = [str(g) for g in np.unique(groups)]
    thr = max(macro["abs_alpha"], macro["abs_beta"])
    return {
        "abs_alpha": macro["abs_alpha"],
        "abs_beta": macro["abs_beta"],
        "gamma": macro["gamma"],
        "gamma_over_dominant_main_effect": macro["gamma"] / thr if thr else float("nan"),
        "mean_delta_s": macro["delta"],
        "acc_pooled_pct": float((delta_emp > 0).mean() * 100.0),
        "acc_macro_pct": macro["acc"],
        "n_concepts_gamma_wins": int(sum(
            g > max(abs(a), abs(b)) for g, a, b in
            zip(per_concept["gamma"], per_concept["abs_alpha"], per_concept["abs_beta"]))),
        "n_concepts": len(per_concept["gamma"]),
        "identity_max_error": max_err,
        "per_concept": {
            "concept": concept_names,
            "abs_alpha": [float(x) for x in per_concept["abs_alpha"]],
            "abs_beta": [float(x) for x in per_concept["abs_beta"]],
            "gamma": [float(x) for x in per_concept["gamma"]],
            "acc_pct": [float(x) for x in per_concept["acc"]],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--names", nargs="*", default=[])
    ap.add_argument("--csv_path", default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--image_root", default="benchmarks/data/images")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--min_pairs", type=int, default=20)
    ap.add_argument("--restrict_objects", default=None)
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--cache_dir", default="logs/evaluation/cached_embeddings/feature_cache")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="logs/evaluation/01_paper/2026-09-02_w_coefficients")
    args = ap.parse_args()
    assert len(args.ckpts) == len(args.names), "--ckpts and --names must pair up"

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_kw = dict(model=args.model, pretrained=str(args.pretrained),
                    cache_dir=args.cache_dir, enabled=args.use_cache)

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    v_pos, v_neg, t_pos, t_neg, groups = load_quads(
        args, model, preprocess, tokenizer, device, cache_kw)
    T = lambda a: torch.from_numpy(a).float().to(device)
    quads = (T(v_pos), T(v_neg), T(t_pos), T(t_neg))
    dim = quads[0].shape[1]
    print(f"[data] {quads[0].shape[0]} quads across {len(set(groups))} concepts")

    # Cosine (W = I) first -- it must reproduce the published coefficients, which
    # is the check that this aggregation matches the paper's.
    conditions = [("cosine (W=I)", torch.eye(dim, device=device))]
    for name, path in zip(args.names, args.ckpts):
        ckpt = torch.load(path, map_location="cpu")
        conditions.append((name, ckpt["state_dict"]["W"].float().to(device)))

    report = {}
    hdr = f"{'condition':<24s} {'|alpha|':>9s} {'|beta|':>9s} {'gamma':>9s} {'g/max':>7s} {'2x2 %':>7s} {'wins':>7s}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for name, W in conditions:
        r = decompose(W, quads, groups)
        report[name] = r
        print(f"{name:<24s} {r['abs_alpha']:9.5f} {r['abs_beta']:9.5f} {r['gamma']:9.5f} "
              f"{r['gamma_over_dominant_main_effect']:7.3f} {r['acc_pooled_pct']:7.2f} "
              f"{r['n_concepts_gamma_wins']:3d}/{r['n_concepts']:<3d}")

    out = os.path.join(args.output_dir, "w_coefficients_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
