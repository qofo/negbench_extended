"""
The 2x2 block again, with the absent side never edited.

RESULTS 8.14 measures the main-effect ablation on BEAF counterfactual pairs, where the
"absent" image is an inpainted edit of the "present" one. Every review of this project
raises the same objection: the interaction term may be reading the edit, not the object.
RESULTS 8-V.6 answered that for the E1 *detection AUC* by swapping the negative class for
verified-unedited photographs (macro AUC 0.759 -> 0.811), but the decomposition itself --
alpha, beta, gamma, and the 74.40% identity -- was never recomputed off inpainted pixels.

This builds the CSV that lets it be. It is the minimal edit to
`beaf_counterfactual_ab_swap_by_concept.csv`: the present rows, their captions, their
concepts and their pairing all stay byte-identical, and only the absent row's
``image_path`` is replaced by an unedited COCO val2014 photograph that COCO's own instance
annotations confirm does not contain that concept. So a difference in the result is
attributable to the absent image source and to nothing else.

What it does not control. A verified-natural absent image differs from the present one in
background, composition and every other object, so these are *not* minimal pairs -- the
scene is no longer held fixed. The two designs are complementary and neither replaces the
other: the inpainted block controls the scene and risks the edit signature, this block
removes the edit signature and gives up scene control. Any claim built on it has to say so.

Usage (from the repo root):
    python -m benchmarks.src.data_generation.build_natural_absence_pairs \\
        --out benchmarks/data/images/beaf_natural_absence_by_concept.csv
"""

import os
import re
import json
import glob
import argparse
from typing import Dict, List, Set

import numpy as np
import pandas as pd

try:
    from benchmarks.src.analysis.config import COCO80, set_seed
except ImportError:
    from analysis.import_compat import reraise_unless_standalone
    reraise_unless_standalone()
    from analysis.config import COCO80, set_seed

_ID_RE = re.compile(r"COCO_val2014_0*(\d+)\.jpg$")


def load_labels(cache_dir: str) -> Dict[int, Set[str]]:
    """image_id -> set of COCO category *names* (see config.COCO80 for why the map)."""
    import pyarrow.parquet as pq

    frames = []
    for name in ("coco_train_labels.parquet", "coco_val_labels.parquet"):
        path = os.path.join(cache_dir, name)
        if os.path.exists(path):
            frames.append(pq.read_table(path).to_pandas())
    if not frames:
        raise SystemExit(f"no cached label parquet under {cache_dir}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="image_id")
    out = {}
    for row in df.itertuples(index=False):
        objs = row.objects
        cats = objs["category"] if isinstance(objs, dict) else objs
        out[int(row.image_id)] = {COCO80[int(c)] for c in np.asarray(cats).ravel()
                                  if 0 <= int(c) < len(COCO80)}
    return out


def pristine_images(image_dir: str) -> List[str]:
    """Unedited val2014 files only -- BEAF's edits carry a `_NN` suffix before the extension."""
    files = sorted(glob.glob(os.path.join(image_dir, "COCO_val2014_*.jpg")))
    return [f for f in files if _ID_RE.search(os.path.basename(f))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv_path",
                    default="benchmarks/data/images/beaf_counterfactual_ab_swap_by_concept.csv")
    ap.add_argument("--image_dir", default="benchmarks/data/coco/images/val2014")
    ap.add_argument("--label_cache_dir",
                    default="logs/evaluation/00_concept_sets/coco_category_labels")
    ap.add_argument("--out", default="benchmarks/data/images/beaf_natural_absence_by_concept.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.csv_path)
    labels = load_labels(args.label_cache_dir)
    files = pristine_images(args.image_dir)
    labelled = [(int(_ID_RE.search(os.path.basename(f)).group(1)), f) for f in files]
    labelled = [(i, f) for i, f in labelled if i in labels]
    print(f"  {len(files)} unedited val2014 files, {len(labelled)} of them COCO-labelled")

    # An image is a usable "absent" example for a concept when the annotations list every
    # object in it and this concept is not among them.
    by_concept: Dict[str, List[str]] = {}
    for concept in sorted(df["object_name"].unique()):
        cands = [f for i, f in labelled if concept not in labels[i]]
        by_concept[concept] = cands

    rows, stats, dropped = [], [], []
    for concept, grp in df.groupby("object_name", sort=True):
        cands = by_concept.get(concept, [])
        n_false = int((grp["object_in_image"].astype(str).str.lower() == "false").sum())
        if len(cands) < 1:
            dropped.append(concept)
            continue
        # Sample without replacement while the pool allows it, then cycle a reshuffled
        # pool -- so a small pool repeats images evenly instead of over-weighting a few.
        picks = []
        while len(picks) < n_false:
            picks.extend(rng.permutation(cands).tolist())
        picks = picks[:n_false]

        k = 0
        for _, r in grp.iterrows():
            r = r.copy()
            if str(r["object_in_image"]).lower() == "false":
                rel = os.path.join("data/coco/images/val2014", os.path.basename(picks[k]))
                r["image_path"] = rel
                k += 1
            rows.append(r)
        stats.append(dict(concept=concept, n_pairs=n_false, absent_pool=len(cands),
                          reused=max(0, n_false - len(cands))))

    out_df = pd.DataFrame(rows)[df.columns.tolist()]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)

    summary = dict(
        rows=int(len(out_df)), concepts=int(out_df["object_name"].nunique()),
        unedited_files=len(files), labelled_files=len(labelled),
        concepts_without_absent_pool=dropped, seed=args.seed,
        source_csv=args.csv_path, out=args.out,
        note=("present rows untouched; only the absent image_path is replaced by a "
              "COCO-annotation-verified unedited photograph. Scene is NOT controlled."),
        per_concept=stats,
    )
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_concept"}, indent=2))


if __name__ == "__main__":
    main()
