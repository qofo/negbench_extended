"""
LABCLIP rank comparison -- the three ways a rank constraint can enter, in one table.

  A "post-hoc SVD"     train full-rank W (LABCLIP recipe), then truncate to
                       rank r by SVD.                    [no retraining]
  B "plain low-rank"   W = U V^T, rank fixed before training, trained
                       end-to-end with the same broad InfoNCE.
  C "residual low-rank" W = I + U V^T -- rank fixed before training AND the
                       diagonal/cosine component held exactly at identity,
                       so the rank budget is spent only on the correction.

All three are scored identically: Delta(S)>0 on the 42-concept AB-swap 2x2
diagnostic, and COCO T2I retrieval R@1/R@5 on the standard and negated query
sets (cached embeddings, no re-encoding).

Why the three differ is spectral, and the spectrum is worth reporting next to
the accuracies: the identity has a completely flat spectrum (all singular
values 1), so it is the worst possible target for low-rank approximation --
the best rank-r approximation of I retains only r/D of its Frobenius energy.
A and B must therefore spend their rank budget re-encoding an incompressible
identity; C carries it exactly, for free, and spends the whole budget on a
correction that is empirically far more compressible.

Usage:
    python -m benchmarks.src.evaluation.eval_labclip_rank_strategy_comparison \
        --output_dir logs/evaluation/01_paper/2026-09-02_labclip_rank_strategy
"""
import os
import re
import json
import glob
import argparse

import torch

from benchmarks.src.evaluation.eval_single_w_generalization import load_quads
from benchmarks.src.evaluation.score_delta_s_for_w import joint_correct_for_w
from benchmarks.src.analysis.model_loader import load_clip_for_eval
from benchmarks.src.analysis.config import set_seed


def spectrum_report(M: torch.Tensor, ranks) -> dict:
    S = torch.linalg.svdvals(M.float())
    energy = (S ** 2).sum()
    return {f"top{r}_energy_pct": ((S[:r] ** 2).sum() / energy).item() * 100 for r in ranks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_rank_ckpt",
                     default="logs/evaluation/01_paper/2026-09-01_labclip_official_recipe/labclip_official_recipe_bilinear.pt")
    ap.add_argument("--plain_dir", default="logs/evaluation/01_paper/2026-09-02_lowrank_plain_rank_sweep")
    ap.add_argument("--residual_dir", default="logs/evaluation/01_paper/2026-09-02_residual_lowrank_rank_sweep")
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--cache", default="logs/evaluation/cached_embeddings/COCO_val_retrieval_retrieval_embeds.pt")
    ap.add_argument("--neg_cache", default="logs/evaluation/cached_embeddings/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true_retrieval_embeds.pt")
    ap.add_argument("--csv_path", default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--image_root", default="benchmarks/data/images")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--min_pairs", type=int, default=20)
    ap.add_argument("--restrict_objects", default=None)
    ap.add_argument("--use_cache", action="store_true", default=True)
    ap.add_argument("--cache_dir", default="logs/evaluation/cached_embeddings/feature_cache")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="logs/evaluation/01_paper/2026-09-02_labclip_rank_strategy")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_gallery(path):
        c = torch.load(path, map_location="cpu")
        return (c["images_emb"].float().to(device), c["texts_emb"].float().to(device),
                torch.tensor(c["texts_image_index"], dtype=torch.long))

    def recall_at_k(scores, idx, k):
        topk = torch.topk(scores, k=k, dim=1).indices
        return (topk == idx.unsqueeze(1)).any(dim=1).float().mean().item() * 100

    std_img, std_txt, std_idx = load_gallery(args.cache)
    neg_img, neg_txt, neg_idx = load_gallery(args.neg_cache)
    std_cos = (std_txt @ std_img.T).cpu()
    neg_cos = (neg_txt @ neg_img.T).cpu()
    base = {
        "std_r1": (std_cos.argmax(dim=1) == std_idx).float().mean().item() * 100,
        "std_r5": recall_at_k(std_cos, std_idx, 5),
        "neg_r1": (neg_cos.argmax(dim=1) == neg_idx).float().mean().item() * 100,
        "neg_r5": recall_at_k(neg_cos, neg_idx, 5),
    }

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    v_pos, v_neg, t_pos, t_neg, groups = load_quads(args, model, preprocess, tokenizer, device,
                                                      dict(model=args.model, pretrained=str(args.pretrained),
                                                           cache_dir=args.cache_dir, enabled=args.use_cache))
    T = lambda a: torch.from_numpy(a).float().to(device)
    quads = (T(v_pos), T(v_neg), T(t_pos), T(t_neg))
    eye = torch.eye(512, device=device)
    base["delta_s"] = joint_correct_for_w(eye, quads)
    print(f"[cosine] std R@1={base['std_r1']:.2f}% R@5={base['std_r5']:.2f}%  "
          f"neg R@1={base['neg_r1']:.2f}% R@5={base['neg_r5']:.2f}%  DeltaS>0={base['delta_s']:.2f}%")

    def score_W(W):
        W = W.float().to(device)
        s_std = (std_txt @ W.T @ std_img.T).cpu()
        s_neg = (neg_txt @ W.T @ neg_img.T).cpu()
        return {
            "dW_norm": torch.norm(W - eye).item(),
            "std_r1": (s_std.argmax(dim=1) == std_idx).float().mean().item() * 100,
            "std_r5": recall_at_k(s_std, std_idx, 5),
            "neg_r1": (s_neg.argmax(dim=1) == neg_idx).float().mean().item() * 100,
            "neg_r5": recall_at_k(s_neg, neg_idx, 5),
            "delta_s": joint_correct_for_w(W, quads),
        }

    full_ckpt = torch.load(args.full_rank_ckpt, map_location="cpu")
    W_full = full_ckpt["state_dict"]["W"].float()

    conditions = {}

    # A -- post-hoc SVD truncation of the trained full-rank W
    U, S, Vh = torch.linalg.svd(W_full, full_matrices=False)
    rows = []
    for r in args.ranks:
        W_r = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
        row = score_W(W_r); row["rank"] = r
        rows.append(row)
    rows.append({**score_W(W_full), "rank": 512})
    conditions["A_posthoc_svd"] = rows

    # B -- plain low-rank UV^T trained from scratch (fair both_random init;
    #      the stalled zero-V run is kept on disk as *_zero_v.pt and excluded here)
    plain_paths = [p for p in glob.glob(os.path.join(args.plain_dir, "lowrank_plain_r*.pt"))
                    if re.search(r"_r(\d+)\.pt$", p)]
    rows = []
    for p in sorted(plain_paths, key=lambda x: int(re.search(r"_r(\d+)\.pt$", x).group(1))):
        r = int(re.search(r"_r(\d+)\.pt$", p).group(1))
        row = score_W(torch.load(p, map_location="cpu")["state_dict"]["W"]); row["rank"] = r
        rows.append(row)
    conditions["B_plain_lowrank"] = rows

    # C -- residual low-rank I + UV^T (diagonal/cosine kept at identity)
    rows = []
    for p in sorted(glob.glob(os.path.join(args.residual_dir, "residual_lowrank_r*.pt")),
                     key=lambda x: int(re.search(r"_r(\d+)\.pt$", x).group(1))):
        r = int(re.search(r"_r(\d+)\.pt$", p).group(1))
        row = score_W(torch.load(p, map_location="cpu")["state_dict"]["W"]); row["rank"] = r
        rows.append(row)
    conditions["C_residual_lowrank"] = rows

    label = {"A_posthoc_svd": "A  사후 SVD 분해 (full-rank 학습 후 절단)",
              "B_plain_lowrank": "B  평문 저계수 W=UV^T (처음부터 rank 고정)",
              "C_residual_lowrank": "C  잔차 저계수 W=I+UV^T (대각=코사인 고정)"}
    for key, rows in conditions.items():
        print(f"\n=== {label[key]} ===")
        for row in rows:
            flag = "✅" if row["std_r1"] > base["std_r1"] else ""
            print(f"  rank={row['rank']:<4d} ||W-I||_F={row['dW_norm']:7.2f}  "
                  f"std R@1={row['std_r1']:6.2f}% R@5={row['std_r5']:6.2f}%  "
                  f"neg R@1={row['neg_r1']:6.2f}% R@5={row['neg_r5']:6.2f}%  "
                  f"DeltaS>0={row['delta_s']:5.2f}%  {flag}")

    spec = {
        "W_full": spectrum_report(W_full, args.ranks),
        "dW_full_minus_I": spectrum_report(W_full - torch.eye(512), args.ranks),
        "identity": spectrum_report(torch.eye(512), args.ranks),
    }
    print("\n=== 스펙트럼 (프로베니우스 에너지 보존율, %) ===")
    for name, d in spec.items():
        print(f"  {name:18s} " + "  ".join(f"top{r}={d[f'top{r}_energy_pct']:5.2f}%" for r in args.ranks))

    with open(os.path.join(args.output_dir, "rank_strategy_report.json"), "w") as f:
        json.dump({"cosine": base, "conditions": conditions, "spectrum": spec}, f, indent=2)
    print(f"\n[saved] {args.output_dir}/rank_strategy_report.json")


if __name__ == "__main__":
    main()
