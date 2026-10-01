"""Held-out (unseen / seen objects) version of the recipe behind the accepted Figure 2.

train_infonce_identity_init.py trains W = I + U V^T (text side) with symmetric masked InfoNCE on every
BEAF (image, true caption) pair and scores the 2x2 judgment on the same 42 objects it trained on. This
script keeps that recipe unchanged (same train(), lr 1e-3, batch 256, 200 epochs, drift penalty c,
seed 42) and only changes the evaluation split: GroupKFold(5) over object_name on the 42 scored objects,
the same folds eval_single_w_generalization.py uses. In each fold W is trained on the pairs of every object
that is not held out (a pair that also belongs to a held-out object is dropped), then scored on the
held-out objects (unseen) and on the fold's training objects (seen).

Usage (repo root):
    python -m benchmarks.src.evaluation.eval_infonce_idinit_heldout \
        --ranks 1 2 4 8 16 32 64 128 256 512 \
        --output_dir logs/evaluation/01_paper/2026-09-18_infonce_idinit_heldout
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from benchmarks.src.evaluation.train_infonce_identity_init import CSV, get_encodings, train
from benchmarks.src.evaluation.eval_single_w_generalization import load_quads
from benchmarks.src.analysis.model_loader import load_clip_for_eval
from benchmarks.src.analysis.config import set_seed, coerce_bool_column


def correct(W, quads):
    vp, vn, tp, tn = quads
    s = lambda v, t: torch.einsum("nd,de,ne->n", v, W, t)
    ok = torch.minimum(s(vp, tp), s(vn, tn)) > torch.maximum(s(vp, tn), s(vn, tp))
    return ok.cpu().numpy()


def pair_objects():
    """Objects each training pair belongs to, in get_encodings()'s pair order."""
    df = coerce_bool_column(pd.read_csv(CSV), "object_in_image")
    df["true_caption"] = np.where(df["object_in_image"], df["positive_caption"], df["negative_caption"])
    pairs = df[["image_path", "true_caption"]].drop_duplicates().reset_index(drop=True)
    objs = df.groupby(["image_path", "true_caption"])["object_name"].apply(set)
    return [objs[(p, c)] for p, c in zip(pairs.image_path, pairs.true_caption)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--c", type=float, default=0.0, help="drift penalty; 0 = the accepted Figure 2")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="logs/evaluation/01_paper/2026-09-18_infonce_idinit_heldout")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    img, txt, img_id = get_encodings(dev)
    pobj = pair_objects()
    assert len(pobj) == len(img), (len(pobj), len(img))

    mc, preprocess, tok = load_clip_for_eval("ViT-B-32", "openai", dev)
    qa = argparse.Namespace(csv_path=CSV, image_root="benchmarks/data/images", model="ViT-B-32",
                            pretrained="openai", min_pairs=20, restrict_objects=None, use_cache=True,
                            cache_dir="logs/evaluation/cached_embeddings/feature_cache",
                            batch_size=256, seed=args.seed)
    vp, vn, tp, tn, groups = load_quads(qa, mc, preprocess, tok, dev,
                                        dict(model="ViT-B-32", pretrained="openai",
                                             cache_dir=qa.cache_dir, enabled=True))
    T = lambda a: torch.from_numpy(a).float().to(dev)
    quads = (T(vp), T(vn), T(tp), T(tn))
    groups = np.asarray(groups)
    folds = list(GroupKFold(n_splits=args.n_splits).split(np.zeros(len(groups)), groups=groups))
    print(f"[data] {len(img)} training pairs, {len(groups)} scored quads, "
          f"{len(set(groups))} scored objects, {len(folds)} folds")

    rows, per_obj = [], []
    for rank in args.ranks:
        unseen = np.zeros(len(groups), bool)
        seen = []
        for tr_q, te_q in folds:
            held = set(groups[te_q])
            drop = np.array([bool(o & held) for o in pobj])
            W = train(rank, args.c, img, txt, img_id, dev, epochs=args.epochs, seed=args.seed,
                      val_idx=np.where(drop)[0], param="residual").to(dev)
            ok = correct(W, quads)
            unseen[te_q] = ok[te_q]
            seen.append(100.0 * ok[tr_q].mean())
        row = dict(rank=rank, c=args.c, unseen_pooled=100.0 * unseen.mean(),
                   seen_foldmean=float(np.mean(seen)))
        rows.append(row)
        for obj in sorted(set(groups)):
            m = groups == obj
            per_obj.append(dict(family=f"infonce_residual_{rank}", object_name=obj,
                                n_pairs=int(m.sum()), oof_acc_pct=100.0 * unseen[m].mean()))
        print(f"  rank={rank:4d}  unseen={row['unseen_pooled']:6.2f}%  seen={row['seen_foldmean']:6.2f}%")

    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "heldout_sweep.csv"), index=False)
    pd.DataFrame(per_obj).to_csv(os.path.join(args.output_dir, "heldout_per_object.csv"), index=False)
    json.dump(dict(recipe="W = I + U V^T (text side), masked symmetric InfoNCE, lr 1e-3, batch 256",
                   epochs=args.epochs, c=args.c, seed=args.seed, n_splits=args.n_splits,
                   cv="GroupKFold(object_name) on the 42 scored objects", results=rows),
              open(os.path.join(args.output_dir, "heldout_summary.json"), "w"), indent=2)
    print("[saved]", args.output_dir)


if __name__ == "__main__":
    main()
