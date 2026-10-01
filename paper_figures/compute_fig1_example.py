"""Compute the four 2x2 scores of the Figure 1 example pair and their Hadamard coordinates.

The example is one AB-swap counterfactual pair (object "pizza", template p1273): the same COCO
scene with the cups removed (I+, pizza present) and with the pizza removed (I-), crossed with the
word-swapped captions T+ ("no cup, but ... a pizza") and T- ("no pizza, but ... a cup").

Scores are cosine similarities of the L2-normalised final embeddings, encoded with the same
functions `eval_single_w_generalization.load_quads` uses, so the four numbers are the cells that
experiment scores for this pair. Key convention: the first sign is the caption, the second the
image, e.g. Smp = S(T-, I+).

    C     = mean of the four scores
    alpha = (Spp + Spm - Smp - Smm) / 4   text main effect
    beta  = (Spp + Smp - Spm - Smm) / 4   image main effect
    gamma = (Spp - Spm - Smp + Smm) / 4   interaction
    delta = min(Spp, Smm) - max(Spm, Smp)

Run from the repo root:
    python paper_figures/compute_fig1_example.py --output paper_figures/fig1_example_pair.json
"""
import argparse
import json

import pandas as pd
import torch

from benchmarks.src.analysis.config import set_seed, coerce_bool_column
from benchmarks.src.analysis.model_loader import load_clip_for_eval
from benchmarks.src.analysis.paths import resolve_image_path
from benchmarks.src.evaluation.eval_e2_hadamard_decomposition import (
    encode_images_unified, encode_texts_unified,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--image_root", default="benchmarks/data/images")
    ap.add_argument("--object_name", default="pizza")
    ap.add_argument("--source_template", default="diverse_lack_neg_first_02_p1273")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="paper_figures/fig1_example_pair.json")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.csv_path)
    coerce_bool_column(df, "object_in_image")
    pair = df[(df.object_name == args.object_name) & (df.source_template == args.source_template)]
    assert len(pair) == 2 and pair.object_in_image.sum() == 1, pair
    assert pair.positive_caption.nunique() == 1 and pair.negative_caption.nunique() == 1, pair
    row_p = pair[pair.object_in_image].iloc[0]
    row_m = pair[~pair.object_in_image].iloc[0]

    model, preprocess, tokenizer = load_clip_for_eval(args.model, args.pretrained, device)
    paths = [resolve_image_path(p, args.image_root) for p in (row_p.image_path, row_m.image_path)]
    v, _, loaded = encode_images_unified(model, preprocess, paths, device)
    assert loaded.all(), paths
    t, _ = encode_texts_unified(model, tokenizer, [row_p.positive_caption, row_p.negative_caption], device)
    v, t = torch.from_numpy(v).double(), torch.from_numpy(t).double()

    s = t @ v.T  # rows: T+, T- ; columns: I+, I-
    spp, spm, smp, smm = s[0, 0].item(), s[0, 1].item(), s[1, 0].item(), s[1, 1].item()
    alpha = (spp + spm - smp - smm) / 4
    beta = (spp + smp - spm - smm) / 4
    gamma = (spp - spm - smp + smm) / 4
    out = {
        "IP": row_p.image_path, "IN": row_m.image_path,
        "TP": row_p.positive_caption, "TN": row_p.negative_caption,
        "Spp": spp, "Spm": spm, "Smp": smp, "Smm": smm,
        "C": (spp + spm + smp + smm) / 4,
        "alpha": alpha, "beta": beta, "gamma": gamma,
        "delta": min(spp, smm) - max(spm, smp),
        "ratio": gamma / max(abs(alpha), abs(beta)),
        "provenance": {"csv_path": args.csv_path, "object_name": args.object_name,
                       "source_template": args.source_template, "model": args.model,
                       "pretrained": args.pretrained, "seed": args.seed},
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "provenance"}, indent=1, ensure_ascii=False))
    print("saved:", args.output)


if __name__ == "__main__":
    main()
