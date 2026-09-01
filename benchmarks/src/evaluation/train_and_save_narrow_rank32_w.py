"""
Re-run the narrow-task (2x2 block, 42-concept AB-swap) rank-32 W training from
eval_delta_loss_gallery_retrieval.py, this time SAVING both W matrices to disk
(that script only ever held them in memory) so the gallery-scale-curve and
residual-scoring diagnostics (diagnose_labclip_gallery_scale_and_residual.py) can
be pointed at the original catastrophic-collapse W's (0.12-0.13% R@1), not just
the LABCLIP-official-recipe W, for a controlled comparison of the two curve shapes.

Same hyperparameters, same seed, same training call as eval_delta_loss_gallery_retrieval.py
-- this is expected to reproduce ||W_margin4||_F=617.79, ||W_delta||_F=298.09 and the
same 0.12-0.13% collapse if re-scored, confirming this is the identical W.

Usage:
    python -m benchmarks.src.evaluation.train_and_save_narrow_rank32_w \
        --output_dir logs/evaluation/01_paper/2026-09-02_narrow_rank32_w_checkpoints
"""
import os
import argparse

import torch

from benchmarks.src.evaluation.eval_delta_loss_gallery_retrieval import train_final_w
from benchmarks.src.analysis.model_loader import load_clip_for_eval
from benchmarks.src.analysis.config import set_seed


def main():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--output_dir", default="logs/evaluation/01_paper/2026-09-02_narrow_rank32_w_checkpoints")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_kw = dict(model=args.model, pretrained=str(args.pretrained),
                    cache_dir=args.cache_dir, enabled=args.use_cache)

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)

    print("training W_margin4 (rank32, all 42 concepts, no held-out fold)...")
    W_margin4 = train_final_w(args, model, preprocess, tokenizer, device, cache_kw,
                              loss_kind="margin4", warmstart=False)
    print("training W_delta (rank32, warmstart, all 42 concepts, no held-out fold)...")
    W_delta = train_final_w(args, model, preprocess, tokenizer, device, cache_kw,
                            loss_kind="delta", warmstart=True)

    print(f"||W_margin4||_F = {torch.linalg.norm(W_margin4).item():.2f}   "
          f"||W_delta||_F = {torch.linalg.norm(W_delta).item():.2f}")

    for name, W in [("margin4_rank32", W_margin4), ("delta_warmstart_rank32", W_delta)]:
        out_path = os.path.join(args.output_dir, f"{name}.pt")
        torch.save({"model_name": "bilinear", "rank": 32,
                    "state_dict": {"W": W, "bias": torch.zeros(1)}}, out_path)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
