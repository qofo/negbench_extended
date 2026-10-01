"""Score several bilinear W checkpoints on NegBench-style MCQ CSVs, keyed by checkpoint file name.

Same code path as eval_zero_shot_transfer.py (extract_mcq_embeddings, evaluate_zero_shot_scorer with
predict_with_tie_report), but that script keys results by the checkpoint's model_name, so several
'bilinear' checkpoints overwrite one another in its JSON. This wrapper extracts the embeddings once per
benchmark and stores one result per checkpoint file.

Usage (repo root):
    PYTHONPATH="$PWD:$PWD/benchmarks:$PWD/benchmarks/src" python -m benchmarks.src.evaluation.eval_w_ckpts_mcq \
        --ckpts logs/.../a.pt logs/.../b.pt \
        --targets benchmarks/data/images/COCO_val_mcq_llama3.1_rephrased.csv \
        --output logs/evaluation/01_paper/<run>/mcq_by_ckpt.json
"""
import argparse
import json
import os

import open_clip
import torch

from src.evaluation.scoring_heads import CosineScorer, build_scorer
from src.evaluation.eval_scoring_heads import extract_mcq_embeddings
from src.evaluation.eval_zero_shot_transfer import evaluate_zero_shot_scorer
from benchmarks.src.analysis.config import set_seed


def load_scorer(path, feature_dim):
    ckpt = torch.load(path, map_location="cpu")
    scorer = build_scorer(ckpt.get("model_name", "bilinear"), feature_dim, rank=ckpt.get("rank", 32))
    scorer.load_state_dict(ckpt["state_dict"])
    return scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default="logs/evaluation/cached_embeddings",
                    help="MCQ embedding cache, keyed by CSV name only; point at an empty dir to re-encode")
    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    out = {}
    for csv in args.targets:
        img, txt, targets, qtypes, _ = extract_mcq_embeddings(
            model, tokenizer, preprocess, csv, device=device, batch_size=64, image_root="",
            cache_dir=args.cache_dir)
        dim = img.shape[1]
        res = {"cosine": evaluate_zero_shot_scorer(CosineScorer(dim), img, txt, targets, qtypes, device=device)}
        for path in args.ckpts:
            set_seed(args.seed)
            res[os.path.basename(path)] = evaluate_zero_shot_scorer(
                load_scorer(path, dim), img, txt, targets, qtypes, device=device)
        out[os.path.basename(csv)] = res
        for name, m in res.items():
            print(f"{os.path.basename(csv)} | {name:28s} total {m['total_accuracy']:6.2f} "
                  f"pos {m.get('positive_accuracy', float('nan')):6.2f} "
                  f"neg {m.get('negative_accuracy', float('nan')):6.2f} tie {m['tie_rate_pct']:.2f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", args.output)


if __name__ == "__main__":
    main()
