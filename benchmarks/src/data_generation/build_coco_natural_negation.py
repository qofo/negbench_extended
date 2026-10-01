"""
Part XVII stage 2 -- verified natural negation pairs from local COCO images.

The training pool used so far is 2,771 inpainted counterfactual pairs over 77 concepts,
which is two to four orders of magnitude smaller than what the negation literature
fine-tunes on. This builds the largest honest scale-up available *without downloading
anything*: unedited COCO val2014 photographs whose category labels come from the COCO
instance annotations, composed into the same "there is A but no B" form the benchmarks
use.

What it does and does not scale. Image diversity is capped by what is on disk -- 500
unedited val2014 jpgs, of which the annotated ones outside the val2017 retrieval gallery
are usable. What grows is the number of *verified concept combinations* per image, from
one to hundreds. So this tests whether more supervision over the same photographs moves
the interaction term; it cannot test image diversity, and any conclusion has to say so.

Contamination. val2017 is the retrieval gallery and the MCQ image source, so every image
id appearing there is dropped, and the count of dropped ids is written into the summary
rather than left implicit.

Output is the six-column paired schema, so the existing loaders read it unchanged:
``positive_caption`` is true of the image and ``negative_caption`` is its A/B swap, which
is false of the same image -- the hard negative, not another image's caption.

Usage (from the repo root):
    python -m benchmarks.src.data_generation.build_coco_natural_negation \\
        --per_image 100 --out benchmarks/data/images/coco_natural_negation.csv
"""

import os
import re
import json
import argparse
from typing import Dict, List, Set

import numpy as np
import pandas as pd

try:
    from benchmarks.src.analysis.config import COCO80
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.config import COCO80

TEMPLATES = [
    "There is a {a} in this image, but no {b}.",
    "This image features a {a}, but no {b} is present.",
    "A {a} is present in this image, but there is no {b}.",
]


def load_labels(cache_dir: str) -> Dict[int, Set[str]]:
    """image_id -> set of COCO category names, from the cached HF-mirror parquet."""
    import pyarrow.parquet as pq

    frames = []
    for name in ("coco_train_labels.parquet", "coco_val_labels.parquet"):
        path = os.path.join(cache_dir, name)
        if os.path.exists(path):
            frames.append(pq.read_table(path).to_pandas())
    if not frames:
        raise SystemExit(f"no cached label parquet under {cache_dir}")
    df = pd.concat(frames, ignore_index=True)
    out = {}
    for row in df.itertuples(index=False):
        objs = row.objects
        cats = objs["category"] if isinstance(objs, dict) else objs
        # `category` is the mirror's integer ClassLabel index, not a name. Reading it
        # raw is what put "There is a 34, but no 56." in every caption of the first
        # build of this file; map through the shared table instead.
        out[int(row.image_id)] = {COCO80[int(c)] for c in np.asarray(cats).ravel()
                                  if 0 <= int(c) < len(COCO80)}
    return out


def val2017_ids(cache_dir: str) -> Set[int]:
    import pyarrow.parquet as pq
    path = os.path.join(cache_dir, "coco_val_labels.parquet")
    return set(int(i) for i in pq.read_table(path).to_pandas()["image_id"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image_dir", default="benchmarks/data/coco/images/val2014")
    ap.add_argument("--label_cache_dir",
                    default="logs/evaluation/00_concept_sets/coco_category_labels")
    ap.add_argument("--per_image", type=int, default=100,
                    help="verified (present, absent) combinations sampled per image")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="benchmarks/data/images/coco_natural_negation.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    labels = load_labels(args.label_cache_dir)
    gallery = val2017_ids(args.label_cache_dir)
    vocabulary = sorted({c for s in labels.values() for c in s})

    files = []
    for f in sorted(os.listdir(args.image_dir)):
        m = re.match(r"^COCO_val2014_0*(\d+)\.jpg$", f)
        if m:
            files.append((int(m.group(1)), f))

    rows, kept, dropped_gallery, dropped_unlabeled = [], 0, 0, 0
    for image_id, fname in files:
        if image_id in gallery:
            dropped_gallery += 1
            continue
        if image_id not in labels:
            dropped_unlabeled += 1
            continue
        present = sorted(labels[image_id])
        absent = [c for c in vocabulary if c not in labels[image_id]]
        if not present or not absent:
            continue
        kept += 1
        rel = os.path.join("data/coco/images/val2014", fname)
        combos = [(a, b) for a in present for b in absent]
        idx = rng.permutation(len(combos))[:args.per_image]
        for j, k in enumerate(idx):
            a, b = combos[k]
            tmpl = TEMPLATES[j % len(TEMPLATES)]
            rows.append(dict(
                image_path=rel,
                object_name=a,
                positive_caption=tmpl.format(a=a, b=b),
                negative_caption=tmpl.format(a=b, b=a),   # the A/B swap: false of this image
                object_in_image=True,
                source_template=f"coco_nat_{j % len(TEMPLATES)}",
            ))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    summary = dict(
        rows=len(df), images_kept=kept, images_dropped_in_val2017_gallery=dropped_gallery,
        images_dropped_unlabeled=dropped_unlabeled, concepts=int(df["object_name"].nunique()),
        vocabulary=len(vocabulary), per_image=args.per_image, seed=args.seed, out=args.out,
    )
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
