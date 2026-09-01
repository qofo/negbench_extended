"""
Score an arbitrary saved W checkpoint (BilinearScorer state_dict format) on our own
Delta(S) > 0 / 2x2 joint-accuracy diagnostic, on the SAME 42-concept AB-swap
population every other W in this comparison was trained on.

Important scope note: this scores each W on the population it was fit on (no
held-out fold) -- consistent with how the narrow/LABCLIP/residual W's used for the
retrieval/MCQ transfer table were all trained (train_and_save_narrow_rank32_w.py,
train_labclip_official_recipe.py, train_residual_identity_infonce.py all fit on ALL
42 concepts, no GroupKFold held-out split, since the point of those runs is transfer
to a genuinely external task/population, not concept generalization within BEAF).
This number is therefore NOT directly comparable to Table 1's properly cross-
validated GroupKFold OOF accuracy (e.g. delta+warmstart rank32's 33.02+-0.61%) --
it answers "what does this exact deployed W achieve on the diagnostic it could see
during training", which is the fair, consistent quantity across all four rows of
THIS table.

Usage:
    python -m benchmarks.src.evaluation.score_delta_s_for_w \
        --ckpts logs/.../margin4_rank32.pt logs/.../labclip_official_recipe_bilinear.pt \
        --names naive labclip
"""
import os
import json
import argparse

import torch

from benchmarks.src.evaluation.eval_single_w_generalization import load_quads
from benchmarks.src.analysis.model_loader import load_clip_for_eval
from benchmarks.src.analysis.config import set_seed


def joint_correct_for_w(W: torch.Tensor, quads) -> float:
    v_pos, v_neg, t_pos, t_neg = quads
    s_pp = torch.einsum("nd,de,ne->n", v_pos, W, t_pos)
    s_mm = torch.einsum("nd,de,ne->n", v_neg, W, t_neg)
    s_pm = torch.einsum("nd,de,ne->n", v_pos, W, t_neg)
    s_mp = torch.einsum("nd,de,ne->n", v_neg, W, t_pos)
    ok = torch.minimum(s_pp, s_mm) > torch.maximum(s_pm, s_mp)
    return ok.float().mean().item() * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--csv_path", default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--image_root", default="benchmarks/data/images")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--min_pairs", type=int, default=20)
    ap.add_argument("--restrict_objects", default=None)
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--cache_dir", default="logs/evaluation/cached_embeddings/feature_cache")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="logs/evaluation/01_paper/2026-09-02_delta_s_for_w")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_kw = dict(model=args.model, pretrained=str(args.pretrained),
                     cache_dir=args.cache_dir, enabled=args.use_cache)

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    v_pos, v_neg, t_pos, t_neg, groups = load_quads(args, model, preprocess, tokenizer, device, cache_kw)
    T = lambda a: torch.from_numpy(a).float().to(device)
    quads = (T(v_pos), T(v_neg), T(t_pos), T(t_neg))
    print(f"[data] {quads[0].shape[0]} quads across {len(set(groups))} concepts")

    report = {}
    for name, ckpt_path in zip(args.names, args.ckpts):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        W = ckpt["state_dict"]["W"].float().to(device)
        acc = joint_correct_for_w(W, quads)
        report[name] = acc
        print(f"  {name:30s}  Delta(S)>0 / 2x2 joint accuracy = {acc:.2f}%")

    with open(os.path.join(args.output_dir, "delta_s_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[saved] {args.output_dir}/delta_s_report.json")


if __name__ == "__main__":
    main()
