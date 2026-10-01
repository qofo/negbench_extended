"""
How much of the main-effect ablation survives a deployable estimator?

`eval_main_effect_ablation_ladder.py` shows that zeroing both main effects takes the
controlled 2x2 from 4.03% to 74.40% while barely moving the interaction (RESULTS 8.14.1).
That ablation is an oracle: it uses the pair's own m_I = (v+ + v-)/2 and m_T, which needs
both images and both captions of the block. A scorer that runs on one image and one
caption cannot form either.

It can, however, use a *fixed* estimate of the directions those means live along, and the
algebra says which side each one belongs to:

    alpha = m_I . d_T   ->  make the TEXT vectors orthogonal to the image mean direction
    beta  = d_I . m_T   ->  make the IMAGE vectors orthogonal to the text mean direction

Both are projections, so each estimator is a single bilinear form and the whole family
sits on the same ruler as every other W in this repo:

    S(v, t) = v^T P_T P_I t,     P_I = I - U_I U_I^T,   P_T = I - U_T U_T^T

with U_I the top-k directions of the image embedding distribution and U_T of the text
distribution. Two properties make this different from every learned W measured so far:
the departure from the identity is rank-2k rather than full, and it targets the main
effects rather than the interaction -- the one lever RESULTS 8.14.2 shows nothing has
pulled yet.

Reference statistics come from the COCO retrieval gallery by default (5,000 images,
25,014 captions), which is disjoint from the BEAF block being scored, so the estimate
carries no information about the evaluation pairs. `--reference block` uses the block's
own embeddings instead, as an upper bound on what a better estimator could buy.

Everything is scored from cached embeddings: the 2x2 block, COCO/VOC2007/CheXpert MCQ,
and COCO T2I retrieval with standard and negated queries.

Usage:
    python -m benchmarks.src.evaluation.eval_main_effect_estimators \\
        --output_dir logs/evaluation/01_paper/2026-09-03_main_effect_estimators
"""
import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from benchmarks.src.evaluation.eval_single_w_generalization import load_quads
    from benchmarks.src.evaluation.score_delta_s_for_w import joint_correct_for_w
    from benchmarks.src.evaluation.scoring_heads import predict_with_tie_report
    from benchmarks.src.analysis.model_loader import load_clip_for_eval
    from benchmarks.src.analysis.config import set_seed
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from evaluation.eval_single_w_generalization import load_quads
    from evaluation.score_delta_s_for_w import joint_correct_for_w
    from evaluation.scoring_heads import predict_with_tie_report
    from analysis.model_loader import load_clip_for_eval
    from analysis.config import set_seed

CACHE = "logs/evaluation/cached_embeddings"
MCQ_SETS = [
    ("COCO", f"{CACHE}/COCO_val_mcq_llama3.1_rephrased_embeds.pt"),
    ("VOC2007", f"{CACHE}/VOC2007_mcq_llama3.1_rephrased_embeds.pt"),
    ("CheXpert", f"{CACHE}/chexpert_binary_mcq_control_valid_only_embeds.pt"),
]


def top_directions(X: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k right singular directions of the (uncentered) embedding matrix.

    Uncentered on purpose: m_I and m_T are embeddings, not deviations, and the first
    singular direction of an uncentered L2-normalized set is the mean direction itself.
    """
    if k <= 0:
        return torch.zeros(X.shape[1], 0, dtype=X.dtype, device=X.device)
    _, _, Vh = torch.linalg.svd(X - X.mean(0) * 0, full_matrices=False)
    return Vh[:k].T.contiguous()


def projector(U: torch.Tensor, dim: int, device) -> torch.Tensor:
    eye = torch.eye(dim, dtype=torch.float64, device=device)
    if U.shape[1] == 0:
        return eye
    return eye - U.to(torch.float64) @ U.to(torch.float64).T


def recall_at_k(scores: torch.Tensor, idx: torch.Tensor, k: int) -> float:
    topk = torch.topk(scores, k=k, dim=1).indices
    return (topk == idx.unsqueeze(1)).any(dim=1).float().mean().item() * 100


def mcq_accuracy(W: torch.Tensor, pack, seed: int) -> Dict[str, float]:
    """argmax over options of v^T W t, split by NegBench question type."""
    img, txt, tgt, qtypes = pack
    scores = torch.einsum("nd,de,nke->nk", img, W.to(img.dtype), txt)
    pred, tie = predict_with_tie_report(scores, seed=seed)
    correct = (pred == tgt.numpy())
    out = {"avg": float(correct.mean() * 100), "tie_rate": float(tie.mean() * 100)}
    q = np.asarray(qtypes)
    for name in ("positive", "negative"):
        m = q == name
        out[name] = float(correct[m].mean() * 100) if m.any() else float("nan")
    return out


def build_conditions(U_I_all, U_T_all, ranks, dim, device) -> List[Tuple[str, torch.Tensor]]:
    """Each condition is a single W; the identity is the cosine baseline."""
    eye = torch.eye(dim, dtype=torch.float64, device=device)
    conds: List[Tuple[str, torch.Tensor]] = [("cosine (W = I)", eye)]
    for k in ranks:
        P_I = projector(U_I_all[:, :k], dim, device)   # applied to the text side
        P_T = projector(U_T_all[:, :k], dim, device)   # applied to the image side
        conds.append((f"alpha only  (P_I, k={k})", P_I))
        conds.append((f"beta only   (P_T, k={k})", P_T))
        conds.append((f"both        (P_T P_I, k={k})", P_T @ P_I))
    return conds


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    def load_gallery(path):
        c = torch.load(path, map_location="cpu")
        return (c["images_emb"].float().to(device), c["texts_emb"].float().to(device),
                torch.tensor(c["texts_image_index"], dtype=torch.long))

    std_img, std_txt, std_idx = load_gallery(args.std_cache)
    neg_img, neg_txt, neg_idx = load_gallery(args.neg_cache)

    mcq = {}
    for name, path in MCQ_SETS:
        c = torch.load(path, map_location="cpu")
        mcq[name] = (c["img_embeds"].float().to(device), c["text_embeds"].float().to(device),
                     c["targets"], c["question_types"])

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    v_pos, v_neg, t_pos, t_neg, groups = load_quads(
        args, model, preprocess, tokenizer, device,
        dict(model=args.model, pretrained=str(args.pretrained),
             cache_dir=args.cache_dir, enabled=args.use_cache))
    T = lambda a: torch.from_numpy(np.asarray(a)).float().to(device)
    quads = (T(v_pos), T(v_neg), T(t_pos), T(t_neg))
    dim = quads[0].shape[1]
    print(f"\n  block: {len(v_pos)} pairs / {len(np.unique(groups))} concepts   "
          f"reference: {args.reference}\n")

    if args.reference == "gallery":
        img_ref, txt_ref = std_img, std_txt
    else:
        img_ref = torch.cat([quads[0], quads[1]], 0)
        txt_ref = torch.cat([quads[2], quads[3]], 0)
    U_I = top_directions(img_ref.double(), max(args.ranks))
    U_T = top_directions(txt_ref.double(), max(args.ranks))

    eye64 = torch.eye(dim, dtype=torch.float64, device=device)
    rows = []
    for label, W in build_conditions(U_I, U_T, args.ranks, dim, device):
        Wf = W.float()
        s_std = (std_txt @ Wf.T @ std_img.T)
        s_neg = (neg_txt @ Wf.T @ neg_img.T)
        row = {
            "condition": label,
            "dW_norm": float(torch.norm(W - eye64).item()),
            "delta_s": joint_correct_for_w(Wf, quads),
            "std_r1": (s_std.argmax(dim=1).cpu() == std_idx).float().mean().item() * 100,
            "std_r5": recall_at_k(s_std.cpu(), std_idx, 5),
            "neg_r1": (s_neg.argmax(dim=1).cpu() == neg_idx).float().mean().item() * 100,
            "neg_r5": recall_at_k(s_neg.cpu(), neg_idx, 5),
        }
        for name, pack in mcq.items():
            a = mcq_accuracy(Wf, pack, args.seed)
            row[f"{name}_avg"] = a["avg"]
            row[f"{name}_pos"] = a["positive"]
            row[f"{name}_neg"] = a["negative"]
            row[f"{name}_tie"] = a["tie_rate"]
        rows.append(row)
        print(f"  {label:<28} |W-I|={row['dW_norm']:6.3f}  2x2={row['delta_s']:6.2f}  "
              f"R@1 std={row['std_r1']:6.2f} neg={row['neg_r1']:6.2f}  "
              f"COCO MCQ {row['COCO_avg']:5.2f}/{row['COCO_pos']:5.2f}/{row['COCO_neg']:5.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, "main_effect_estimators.csv"), index=False)
    with open(os.path.join(args.output_dir, "main_effect_estimators.json"), "w") as f:
        json.dump({"rows": rows, "reference": args.reference, "ranks": args.ranks,
                   "seed": args.seed, "model": args.model,
                   "pretrained": str(args.pretrained)}, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {args.output_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--reference", choices=["gallery", "block"], default="gallery",
                    help="where the mean/principal directions are estimated from")
    ap.add_argument("--std_cache", default=f"{CACHE}/COCO_val_retrieval_retrieval_embeds.pt")
    ap.add_argument("--neg_cache",
                    default=f"{CACHE}/COCO_val_negated_retrieval_llama3.1_rephrased"
                            "_affneg_true_retrieval_embeds.pt")
    ap.add_argument("--csv_path",
                    default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--image_root", default="benchmarks/data/images")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--min_pairs", type=int, default=20)
    ap.add_argument("--restrict_objects", default=None)
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--cache_dir", default=f"{CACHE}/feature_cache")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir",
                    default="logs/evaluation/01_paper/2026-09-03_main_effect_estimators")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
